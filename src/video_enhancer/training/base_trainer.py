"""Base Trainer for all video enhancement fine-tuning tasks.

Encapsulates the PyTorch training loop, AMP (mixed precision), gradient 
accumulation, and checkpointing. Avoids duplication across Real-ESRGAN, 
GFPGAN, NAFNet, and RIFE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from video_enhancer.config.schema import TrainingConfig
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class BaseFineTuner:
    """A VRAM-optimized training loop engine."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        scheduler: Any = None,
    ):
        self.config = config
        
        # Performance optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        # Move model to device
        self.device = torch.device(f"cuda:{config.device.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        # AMP setup
        self.use_amp = config.device.use_amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if config.device.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
        self.amp_dtype = amp_dtype
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.global_step = 0
        self.start_epoch = 0
        
        # Logging
        self.log_dir = Path(config.log_dir) / config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Checkpointing
        self.save_dir = Path(config.checkpoint.save_dir) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if config.checkpoint.resume_path:
            self.load_checkpoint(Path(config.checkpoint.resume_path))

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Must be implemented by subclasses. Should return a dict of losses."""
        raise NotImplementedError("Subclasses must implement train_step")

    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Must be implemented by subclasses. Should return a dict of metrics."""
        raise NotImplementedError("Subclasses must implement val_step")

    def fit(self):
        """Main training loop."""
        logger.info(f"Starting training on device: {self.device} (AMP: {self.use_amp})")
        self.model.train()
        
        # We track epochs just for iteration, but we rely on total_steps
        epoch = self.start_epoch
        while self.global_step < self.config.scheduler.total_steps:
            for i, batch in enumerate(self.train_loader):
                if self.global_step >= self.config.scheduler.total_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}

                # Forward + Backward with AMP
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    loss_dict = self.train_step(batch)
                    total_loss = sum(loss_dict.values())
                    # Scale loss for gradient accumulation
                    total_loss = total_loss / self.config.optimizer.gradient_accumulation_steps

                self.scaler.scale(total_loss).backward()

                # Detach metrics for logging to avoid keeping the computational graph alive
                detached_loss_dict = {k: v.detach().clone() for k, v in loss_dict.items()}
                
                # Explicitly free memory for inputs and graph intermediates
                del batch
                del loss_dict
                del total_loss

                # Optimizer step (with gradient accumulation)
                if (i + 1) % self.config.optimizer.gradient_accumulation_steps == 0:
                    if self.config.optimizer.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # set_to_none=True saves memory compared to zero_grad()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        self._log_metrics(detached_loss_dict, "train")

                    # Validation & Checkpointing
                    if self.global_step % self.config.checkpoint.save_every_n_steps == 0:
                        if self.val_loader is not None:
                            self.validate()
                        self.save_checkpoint()

            epoch += 1
            
        logger.info("Training complete.")
        self.writer.close()

    @torch.no_grad()
    def validate(self):
        """Run validation loop."""
        self.model.eval()
        val_losses = {}
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                metrics = self.val_step(batch)
                
            for k, v in metrics.items():
                val_losses[k] = val_losses.get(k, 0.0) + v.item()
                
            del batch
            del metrics
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Average
        for k in val_losses:
            val_losses[k] /= len(self.val_loader)
            
        self._log_metrics(val_losses, "val")
        self.model.train()

    def _log_metrics(self, metrics: dict, prefix: str):
        log_str = f"Step {self.global_step:06d} | "
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self.writer.add_scalar(f"{prefix}/{k}", val, self.global_step)
            log_str += f"{k}: {val:.4f} "
            
        if prefix == "train":
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("train/lr", lr, self.global_step)
            log_str += f"| lr: {lr:.2e}"
            
        logger.info(log_str)

    def save_checkpoint(self):
        """Save model and optimizer state."""
        ckpt_path = self.save_dir / f"step_{self.global_step:06d}.pth"
        state = {
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
            
        torch.save(state, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        
        # Cleanup old checkpoints
        checkpoints = sorted(self.save_dir.glob("step_*.pth"))
        if len(checkpoints) > self.config.checkpoint.keep_last_n:
            for old_ckpt in checkpoints[:-self.config.checkpoint.keep_last_n]:
                old_ckpt.unlink()

    def load_checkpoint(self, path: Path):
        """Resume from checkpoint."""
        logger.info(f"Resuming from checkpoint: {path}")
        state = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        if self.scheduler and "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
            
        self.global_step = state["global_step"]
        # Approximate start epoch (assuming we don't save exactly at epoch boundaries,
        # it's just for logging).
        self.start_epoch = self.global_step // len(self.train_loader) if len(self.train_loader) > 0 else 0
