"""Trainer for GFPGAN (Face Enhancement).

GFPGAN uses L1 and LPIPS losses just like Real-ESRGAN, but requires the input 
images to be tightly cropped/aligned faces. The Data Pipeline (which uses facexlib) 
will handle the cropping before it reaches this trainer.
"""

from typing import Dict
import torch

from video_enhancer.training.base_trainer import BaseFineTuner
from video_enhancer.losses.losses import EnhancementLoss
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class GFPGANFineTuner(BaseFineTuner):
    """Fine-tunes the GFPGAN generator for face enhancement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Similar to Phase 4, we use L1 + Perceptual (no Discriminator to save 6GB VRAM)
        self.criterion = EnhancementLoss(
            pixel_weight=self.config.loss.pixel_weight,
            perceptual_weight=self.config.loss.perceptual_weight,
            device=self.device
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_img = batch["lr"]
        hr_img = batch["hr"]
        
        # GFPGAN forward pass
        # The clean arch returns (generated_image, None) because it doesn't return latents
        out = self.model(lr_img)
        pred_hr = out[0] if isinstance(out, tuple) else out
        
        # Compute losses
        loss_dict = self.criterion(pred_hr, hr_img)
        
        return loss_dict

    @torch.no_grad()
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        lr_img = batch["lr"]
        hr_img = batch["hr"]
        
        out = self.model(lr_img)
        pred_hr = out[0] if isinstance(out, tuple) else out
        
        val_l1 = torch.nn.functional.l1_loss(pred_hr, hr_img)
        
        mse = torch.nn.functional.mse_loss(pred_hr, hr_img)
        psnr = torch.tensor(100.0, device=self.device) if mse == 0 else 10 * torch.log10(1.0 / mse)
            
        return {
            "l1": val_l1,
            "psnr": psnr
        }
