"""Trainer for Real-ESRGAN."""

from typing import Dict
from pathlib import Path
import torch

from video_enhancer.training.base_trainer import BaseFineTuner
from video_enhancer.losses.losses import EnhancementLoss
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class RealESRGANFineTuner(BaseFineTuner):
    """Fine-tunes the Real-ESRGAN generator with a Discriminator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. Monkey patch torchvision before importing basicsr
        import sys
        import torchvision.transforms.functional as TF
        if "torchvision.transforms.functional_tensor" not in sys.modules:
            sys.modules["torchvision.transforms.functional_tensor"] = TF
        from basicsr.archs.discriminator_arch import UNetDiscriminatorSN

        # 2. Initialize Discriminator and its optimizer
        self.net_d = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True).to(self.device)
        self.opt_d = torch.optim.Adam(
            self.net_d.parameters(), 
            lr=self.config.optimizer.lr, 
            betas=(0.9, 0.99)
        )
        self.scaler_d = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.gan_loss = torch.nn.BCEWithLogitsLoss()
        
        # 3. Initialize Generator Loss (L1 + Perceptual)
        self.criterion = EnhancementLoss(
            pixel_weight=self.config.loss.pixel_weight,
            perceptual_weight=self.config.loss.perceptual_weight,
            device=self.device
        )
        # GAN weight for Generator (default 0.1 if not in config)
        self.gan_weight = getattr(self.config.loss, "gan_weight", 0.1)
        self.d_update_interval = getattr(self.config.optimizer, "d_update_interval", 1)

        # 4. Optional torch.compile (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                logger.info("Attempting torch.compile() on Generator and Discriminator...")
                compiled_model = torch.compile(self.model)
                compiled_net_d = torch.compile(self.net_d)
                
                # Trigger compilation with dummy pass to catch errors (e.g. missing Triton on Windows)
                with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    seq_len = getattr(self.config.data, "sequence_length", 1)
                    dummy_in = torch.zeros(1, 3 * seq_len, 64, 64, device=self.device)
                    _ = compiled_model(dummy_in)
                    
                self.model = compiled_model
                self.net_d = compiled_net_d
                logger.info("Successfully applied torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {type(e).__name__}. Falling back to uncompiled execution.")

    def fit(self):
        """Override fit() to implement alternating GAN training steps."""
        logger.info(f"Starting GAN training on device: {self.device} (AMP: {self.use_amp})")
        self.model.train()
        self.net_d.train()
        
        epoch = self.start_epoch
        while self.global_step < self.config.scheduler.total_steps:
            for i, batch in enumerate(self.train_loader):
                if self.global_step >= self.config.scheduler.total_steps:
                    break
                
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                lr_img, hr_img = batch["lr"], batch["hr"]

                # ==============================
                # 1. Train Discriminator
                # ==============================
                update_d = (self.global_step % self.d_update_interval == 0)
                
                if update_d:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        # Generate fake
                        with torch.no_grad():
                            fake_hr = self.model(lr_img)
                        
                        # Real and Fake forward passes
                        pred_real = self.net_d(hr_img)
                        pred_fake = self.net_d(fake_hr.detach())
                        
                        # BCE loss
                        l_d_real = self.gan_loss(pred_real, torch.ones_like(pred_real))
                        l_d_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
                        loss_d = (l_d_real + l_d_fake) / 2
                        
                        # Scale for grad accumulation
                        loss_d_scaled = loss_d / self.config.optimizer.gradient_accumulation_steps
    
                    self.scaler_d.scale(loss_d_scaled).backward()
                else:
                    # Provide dummy tensors for G's loss_dict so logging doesn't fail
                    l_d_real = torch.tensor(0.0, device=self.device)
                    l_d_fake = torch.tensor(0.0, device=self.device)

                # ==============================
                # 2. Train Generator
                # ==============================
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    fake_hr = self.model(lr_img)
                    
                    # Pixel + Perceptual loss
                    loss_dict = self.criterion(fake_hr, hr_img)
                    
                    # Adversarial loss (G wants D to predict fake as real)
                    pred_fake_g = self.net_d(fake_hr)
                    l_g_gan = self.gan_loss(pred_fake_g, torch.ones_like(pred_fake_g))
                    
                    loss_dict["l_g_gan"] = l_g_gan * self.gan_weight
                    loss_dict["l_d_real"] = l_d_real.detach()
                    loss_dict["l_d_fake"] = l_d_fake.detach()
                    
                    # Total Generator loss
                    total_loss_g = sum(v for k, v in loss_dict.items() if not k.startswith("l_d_"))
                    total_loss_g_scaled = total_loss_g / self.config.optimizer.gradient_accumulation_steps

                self.scaler.scale(total_loss_g_scaled).backward()

                # Detach for logging
                detached_loss_dict = {k: v.detach().clone() for k, v in loss_dict.items()}
                
                # Cleanup
                del batch, lr_img, hr_img, fake_hr, pred_fake_g
                if update_d:
                    del pred_real, pred_fake, loss_d
                del loss_dict, total_loss_g

                # ==============================
                # 3. Optimizer Step
                # ==============================
                if (i + 1) % self.config.optimizer.gradient_accumulation_steps == 0:
                    if self.config.optimizer.max_grad_norm > 0:
                        if update_d:
                            self.scaler_d.unscale_(self.opt_d)
                            torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), self.config.optimizer.max_grad_norm)
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.max_grad_norm)
                    
                    # D step
                    if update_d:
                        self.scaler_d.step(self.opt_d)
                        self.scaler_d.update()
                        self.opt_d.zero_grad(set_to_none=True)
                    
                    # G step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    if self.scheduler:
                        self.scheduler.step()

                    self.global_step += 1

                    if self.global_step % self.config.log_every_n_steps == 0:
                        self._log_metrics(detached_loss_dict, "train")

                    if self.global_step % self.config.checkpoint.save_every_n_steps == 0:
                        if self.val_loader is not None:
                            self.validate()
                        self.save_checkpoint()

            epoch += 1
            
        logger.info("GAN Training complete.")
        self.writer.close()

    def save_checkpoint(self):
        """Override to save discriminator state."""
        ckpt_path = self.save_dir / f"step_{self.global_step:06d}.pth"
        state = {
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "net_d_state": self.net_d.state_dict(),
            "opt_d_state": self.opt_d.state_dict(),
            "scaler_d_state": self.scaler_d.state_dict(),
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
            
        torch.save(state, ckpt_path)
        logger.info(f"Saved GAN checkpoint to {ckpt_path}")
        
        checkpoints = sorted(self.save_dir.glob("step_*.pth"))
        if len(checkpoints) > self.config.checkpoint.keep_last_n:
            for old_ckpt in checkpoints[:-self.config.checkpoint.keep_last_n]:
                old_ckpt.unlink()

    def load_checkpoint(self, path: Path):
        """Override to load discriminator state."""
        logger.info(f"Resuming GAN from checkpoint: {path}")
        state = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        
        if "net_d_state" in state:
            self.net_d.load_state_dict(state["net_d_state"])
            self.opt_d.load_state_dict(state["opt_d_state"])
            self.scaler_d.load_state_dict(state["scaler_d_state"])
        
        if self.scheduler and "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
            
        self.global_step = state["global_step"]
        self.start_epoch = self.global_step // len(self.train_loader) if len(self.train_loader) > 0 else 0

    @torch.no_grad()
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_img = batch["lr"]
        hr_img = batch["hr"]
        
        pred_hr = self.model(lr_img)
        val_l1 = torch.nn.functional.l1_loss(pred_hr, hr_img)
        
        mse = torch.nn.functional.mse_loss(pred_hr, hr_img)
        if mse == 0:
            psnr = torch.tensor(100.0, device=self.device)
        else:
            psnr = 10 * torch.log10(1.0 / mse)
            
        return {
            "l1": val_l1,
            "psnr": psnr
        }
