"""Trainer for NAFNet (Deblurring).

NAFNet relies purely on L1 loss (or MSE loss). It does not need Perceptual 
or GAN losses, which saves us massive amounts of VRAM!
"""

from typing import Dict
import torch
import torch.nn.functional as F

from video_enhancer.training.base_trainer import BaseFineTuner
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class NAFNetFineTuner(BaseFineTuner):
    """Fine-tunes NAFNet for deblurring and artifact removal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # NAFNet uses pure L1 loss or PSNR loss.
        # We don't initialize the heavy EnhancementLoss (LPIPS) to save VRAM.

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_img = batch["lr"]
        hr_img = batch["hr"]
        
        # Forward pass (generator)
        pred_hr = self.model(lr_img)
        
        # Compute L1 Loss
        l1_loss = F.l1_loss(pred_hr, hr_img)
        
        return {"l1": l1_loss}

    @torch.no_grad()
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_img = batch["lr"]
        hr_img = batch["hr"]
        
        # Forward pass
        pred_hr = self.model(lr_img)
        
        # 1. Validation Loss (L1)
        val_l1 = F.l1_loss(pred_hr, hr_img)
        
        # 2. PSNR (Peak Signal to Noise Ratio)
        mse = F.mse_loss(pred_hr, hr_img)
        if mse == 0:
            psnr = torch.tensor(100.0, device=self.device)
        else:
            psnr = 10 * torch.log10(1.0 / mse)
            
        return {
            "l1": val_l1,
            "psnr": psnr
        }
