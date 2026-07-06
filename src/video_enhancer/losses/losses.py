"""Loss functions for Video Enhancement.

Includes Pixel Loss (L1/L2) and Perceptual Loss (LPIPS).
GAN discriminator is omitted for Phase 4 to ensure it fits in 6GB VRAM,
as training a heavy Discriminator alongside the Generator easily exceeds 6GB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancementLoss(nn.Module):
    """Combines Pixel (L1) and Perceptual (VGG/LPIPS) loss."""
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        device: torch.device | str = "cuda"
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.device = device
        
        if self.perceptual_weight > 0:
            if not LPIPS_AVAILABLE:
                raise ImportError("lpips package is required for perceptual loss. Run `pip install lpips`")
            
            logger.info("Initializing LPIPS (VGG) for perceptual loss...")
            # We use vgg as it's standard for Real-ESRGAN
            self.lpips = lpips.LPIPS(net="vgg").to(self.device)
            
            # Monkey-patch LPIPS to drop the 5th layer (slice5) to save VRAM
            if hasattr(self.lpips.net, 'slice5'):
                del self.lpips.net.slice5
            
            def truncated_vgg_forward(X):
                h = self.lpips.net.slice1(X)
                h_relu1_2 = h
                h = self.lpips.net.slice2(h)
                h_relu2_2 = h
                h = self.lpips.net.slice3(h)
                h_relu3_3 = h
                h = self.lpips.net.slice4(h)
                h_relu4_3 = h
                from collections import namedtuple
                VggOutputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
                return VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, None)
                
            self.lpips.net.forward = truncated_vgg_forward
            self.lpips.L = 4  # Tell LPIPS to only compute the first 4 layers
            
            # Freeze LPIPS
            for p in self.lpips.parameters():
                p.requires_grad = False
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates combined losses.
        
        Args:
            pred: Generated high-resolution image [B, C, H, W] in [0, 1]
            target: Ground truth high-resolution image [B, C, H, W] in [0, 1]
            
        Returns:
            Dictionary of individual loss components.
        """
        losses = {}
        
        # 1. Pixel Loss (L1)
        if self.pixel_weight > 0:
            losses["l1"] = F.l1_loss(pred, target) * self.pixel_weight
            
        # 2. Perceptual Loss (LPIPS)
        if self.perceptual_weight > 0:
            # LPIPS expects inputs in [-1, 1], so we scale [0, 1] -> [-1, 1]
            pred_scaled = pred * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            
            # lpips returns [B, 1, 1, 1], we take the mean over the batch
            losses["perceptual"] = self.lpips(pred_scaled, target_scaled).mean() * self.perceptual_weight
            
        return losses
