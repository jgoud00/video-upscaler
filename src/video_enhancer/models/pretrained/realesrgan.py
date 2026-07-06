"""Real-ESRGAN Model Wrapper.

📘 Concept — Freezing and Adapting
The official RealESRGAN_x4plus checkpoint uses RRDBNet with 23 blocks.
Training all 23 blocks (16.7M params) on 6GB VRAM is very risky and prone 
to catastrophic forgetting of general textures. We freeze the early blocks 
and apply LoRA to the later blocks.
"""

from typing import Dict, Any
import sys
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Monkey-patch basicsr compatibility for newer torchvision versions
if "torchvision.transforms.functional_tensor" not in sys.modules:
    sys.modules["torchvision.transforms.functional_tensor"] = sys.modules["torchvision.transforms.functional"]

from basicsr.archs.rrdbnet_arch import RRDBNet

from video_enhancer.config.schema import FreezeConfig, LoRAConfig
from video_enhancer.models.adapters.lora import inject_lora, print_trainable_parameters
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


def build_realesrgan(
    checkpoint_path: str | None = None,
    freeze_config: FreezeConfig | None = None,
    lora_config: LoRAConfig | None = None,
    sequence_length: int = 1,
    use_gradient_checkpointing: bool = False,
) -> nn.Module:
    """Builds RRDBNet and applies freezing/LoRA strategies."""
    
    num_in_ch = 3 * sequence_length
    
    # Initialize the architecture matching RealESRGAN_x4plus
    model = RRDBNet(
        num_in_ch=num_in_ch,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    
    if checkpoint_path:
        logger.info(f"Loading Real-ESRGAN weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Handle cases where the checkpoint is wrapped in 'params_ema' or 'params'
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
            
        # Dynamically adapt the first convolution layer if sequence_length > 1
        if sequence_length > 1 and "conv_first.weight" in state_dict:
            w = state_dict["conv_first.weight"]
            if w.shape[1] == 3:
                logger.info(f"Adapting conv_first weights from 3 channels to {num_in_ch} channels")
                # Repeat the channels and scale down to preserve the activation magnitude
                w = w.repeat(1, sequence_length, 1, 1) / sequence_length
                state_dict["conv_first.weight"] = w
            
        model.load_state_dict(state_dict, strict=True)
        
    num_frozen = 0
    # 1. Apply Freezing Strategy
    if freeze_config and freeze_config.freeze_strategy != "none":
        num_frozen = freeze_config.num_frozen_blocks
        logger.info(f"Freezing strategy: {freeze_config.freeze_strategy} (freezing first {num_frozen} blocks)")
        
        # Freeze stem conv
        for p in model.conv_first.parameters():
            p.requires_grad = False
            
        # Freeze N RRDB blocks
        for i in range(min(num_frozen, len(model.body))):
            for p in model.body[i].parameters():
                p.requires_grad = False
                
    # 2. Apply Gradient Checkpointing to trainable blocks
    if use_gradient_checkpointing:
        logger.info(f"Applying gradient checkpointing to trainable RRDB blocks ({num_frozen} to {len(model.body)}).")
        import torch.utils.checkpoint as cp
        
        class CheckpointWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            
            def forward(self, x):
                # Ensure x has requires_grad so checkpointing triggers backward for parameters
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                return cp.checkpoint(self.module, x, use_reentrant=False)
                
        for i in range(min(num_frozen, len(model.body)), len(model.body)):
            model.body[i] = CheckpointWrapper(model.body[i])

    # 3. Apply LoRA (if enabled)
    if lora_config and lora_config.enabled:
        logger.info(f"Applying LoRA (rank={lora_config.rank}, alpha={lora_config.alpha})")
        # Inject LoRA into the unfrozen blocks
        model = inject_lora(
            model, 
            target_module_names=lora_config.target_modules,
            rank=lora_config.rank,
            alpha=lora_config.alpha
        )
        
    # Print summary
    print_trainable_parameters(model)
    
    return model
