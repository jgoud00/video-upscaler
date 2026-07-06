"""RIFE Model Wrapper.

For Frame Interpolation, the network takes (frame0, frame2) and predicts frame1.
We wrap the official Practical-RIFE IFNet to match our pipeline's I/O interface.
"""

import torch
import torch.nn as nn
from video_enhancer.config.schema import FreezeConfig, LoRAConfig
from video_enhancer.models.adapters.lora import inject_lora, print_trainable_parameters
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class RIFEWrapper(nn.Module):
    """Wraps the official IFNet_HDv3 to match our pipeline interface."""
    def __init__(self):
        super().__init__()
        from video_enhancer.models.pretrained.rife_core.IFNet_HDv3 import IFNet
        self.net = IFNet()

    def forward(self, img0, img1):
        import torch.nn.functional as F
        
        n, c, h, w = img0.shape
        tmp = 64
        pad_h = ((h - 1) // tmp + 1) * tmp - h
        pad_w = ((w - 1) // tmp + 1) * tmp - w
        
        if pad_h > 0 or pad_w > 0:
            img0 = F.pad(img0, (0, pad_w, 0, pad_h))
            img1 = F.pad(img1, (0, pad_w, 0, pad_h))
            
        # RIFE IFNet expects a 6-channel tensor concatenated along dim=1
        x = torch.cat((img0, img1), dim=1)
        
        # Forward pass through official IFNet
        scale_list = [16, 8, 4, 2, 1]
        flow_list, mask, merged = self.net(x, scale_list=scale_list)
        
        # merged is a list of results from the 5 blocks. 
        # merged[4] is the final high-quality output.
        out = merged[4]
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            
        return torch.clamp(out, 0, 1)


def build_rife(
    checkpoint_path: str | None = None,
    freeze_config: FreezeConfig | None = None,
    lora_config: LoRAConfig | None = None,
) -> nn.Module:
    """Builds RIFE (IFNet) and applies freezing/LoRA strategies."""
    
    model = RIFEWrapper()
    
    if checkpoint_path:
        logger.info(f"Loading official RIFE weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle DistributedDataParallel "module." prefix if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace('module.', '') if k.startswith('module.') else k
            clean_state_dict[clean_key] = v
            
        model.net.load_state_dict(clean_state_dict, strict=False)
        
    # 1. Apply Freezing Strategy
    if freeze_config and freeze_config.freeze_strategy != "none":
        logger.info("Freezing RIFE encoder.")
        # We can freeze the first blocks of IFNet
        for p in model.net.block0.parameters():
            p.requires_grad = False
        for p in model.net.block1.parameters():
            p.requires_grad = False
            
    # 2. Apply LoRA
    if lora_config and lora_config.enabled:
        logger.info(f"Applying LoRA (rank={lora_config.rank}) to RIFE")
        model = inject_lora(
            model, 
            target_module_names=lora_config.target_modules,
            rank=lora_config.rank,
            alpha=lora_config.alpha
        )

    print_trainable_parameters(model)
    return model
