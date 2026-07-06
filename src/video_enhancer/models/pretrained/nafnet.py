"""NAFNet Model Wrapper.

NAFNet is a Nonlinear Activation Free Network.
For a 6GB GPU, we use Gradient Checkpointing if memory becomes an issue.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from video_enhancer.models.pretrained.nafnet_arch import NAFNetLocal
from video_enhancer.config.schema import FreezeConfig, LoRAConfig
from video_enhancer.models.adapters.lora import inject_lora, print_trainable_parameters
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


def build_nafnet(
    checkpoint_path: str | None = None,
    freeze_config: FreezeConfig | None = None,
    lora_config: LoRAConfig | None = None,
    use_gradient_checkpointing: bool = True
) -> nn.Module:
    """Builds NAFNet (width64) and applies freezing/LoRA strategies."""
    
    # Initialize the architecture matching NAFNet-width64
    model = NAFNetLocal(
        width=64, 
        enc_blk_nums=[2, 2, 4, 8], 
        middle_blk_num=12, 
        dec_blk_nums=[2, 2, 2, 2]
    )
    
    if checkpoint_path:
        logger.info(f"Loading NAFNet weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "params" in state_dict:
            state_dict = state_dict["params"]
            
        model.load_state_dict(state_dict, strict=True)
        
    # 1. Apply Freezing Strategy (Optional, less common for Deblur, but we support it)
    if freeze_config and freeze_config.freeze_strategy != "none":
        num_frozen = freeze_config.num_frozen_blocks
        logger.info(f"Freezing strategy: {freeze_config.freeze_strategy} (freezing {num_frozen} blocks)")
        
        # We freeze the intro and the first few encoder blocks
        for p in model.intro.parameters():
            p.requires_grad = False
            
        for i in range(min(num_frozen, len(model.encoders))):
            for p in model.encoders[i].parameters():
                p.requires_grad = False
                
    # 2. Apply LoRA (if enabled)
    if lora_config and lora_config.enabled:
        logger.info(f"Applying LoRA (rank={lora_config.rank}, alpha={lora_config.alpha})")
        model = inject_lora(
            model, 
            target_module_names=lora_config.target_modules,
            rank=lora_config.rank,
            alpha=lora_config.alpha
        )

    # 3. Gradient Checkpointing Setup
    # NAFNet uses ~42M parameters. To fit in 6GB, we must use gradient checkpointing 
    # for the heavy blocks.
    if use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for NAFNet middle/decoder blocks to save VRAM.")
        
        # We hook into the forward pass of the blocks that we want to checkpoint.
        # This requires monkey-patching the forward function of the NAFBlock or the model itself.
        # A simpler way is to replace the Sequential blocks with a checkpointed wrapper.
        
        class CheckpointedSequential(nn.Module):
            def __init__(self, seq):
                super().__init__()
                self.seq = seq
                
            def forward(self, x):
                # Using checkpoint on the entire sequence chunk
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                    
                if x.requires_grad:
                    return checkpoint(create_custom_forward(self.seq), x, use_reentrant=False)
                return self.seq(x)

        # Wrap the middle blocks
        model.middle_blks = CheckpointedSequential(model.middle_blks)
        
        # Wrap the decoder blocks
        for i in range(len(model.decoders)):
            model.decoders[i] = CheckpointedSequential(model.decoders[i])
            
    # Print summary
    print_trainable_parameters(model)
    
    return model
