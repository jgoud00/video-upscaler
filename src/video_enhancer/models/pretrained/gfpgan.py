"""GFPGAN Model Wrapper."""

import sys
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Monkey-patch basicsr compatibility
if "torchvision.transforms.functional_tensor" not in sys.modules:
    sys.modules["torchvision.transforms.functional_tensor"] = sys.modules["torchvision.transforms.functional"]

from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from video_enhancer.config.schema import FreezeConfig, LoRAConfig
from video_enhancer.models.adapters.lora import inject_lora, print_trainable_parameters
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


def build_gfpgan(
    checkpoint_path: str | None = None,
    freeze_config: FreezeConfig | None = None,
    lora_config: LoRAConfig | None = None,
) -> nn.Module:
    """Builds GFPGAN and applies freezing/LoRA strategies."""
    
    # Initialize the architecture matching GFPGANv1.4
    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True
    )
    
    if checkpoint_path:
        logger.info(f"Loading GFPGAN weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
            
        model.load_state_dict(state_dict, strict=True)
        
    # 1. Apply Freezing Strategy
    if freeze_config and freeze_config.freeze_strategy != "none":
        logger.info("Freezing GFPGAN StyleGAN2 generator base.")
        # GFPGAN has a StyleGAN2 generator. To fit in 6GB VRAM, we freeze most of it
        # and only train the condition network or the final to_rgb layers.
        for p in model.parameters():
            p.requires_grad = False
            
        # Unfreeze just the spatial feature transform (SFT) or final layers
        for name, param in model.named_parameters():
            if "condition_scale" in name or "condition_shift" in name or "to_rgb" in name:
                param.requires_grad = True

    # 2. Apply LoRA
    if lora_config and lora_config.enabled:
        logger.info(f"Applying LoRA (rank={lora_config.rank}) to GFPGAN")
        # GFPGAN uses Conv2d in the condition network and upsample layers
        model = inject_lora(
            model, 
            target_module_names=lora_config.target_modules,
            rank=lora_config.rank,
            alpha=lora_config.alpha
        )

    print_trainable_parameters(model)
    return model
