"""Low-Rank Adaptation (LoRA) specifically designed for Conv2d layers.

📘 Concept — LoRA for Convolutions
Standard LoRA replaces a dense weight matrix `W` with `W + A*B` where A and B 
are low-rank matrices. For a 2D convolution `W` of shape `(out_c, in_c, k, k)`, 
we can approximate the update by adding a 1x1 convolution `A` (in_c -> rank) 
followed by a 1x1 convolution `B` (rank -> out_c). This massively reduces 
trainable parameters when `rank << min(in_c, out_c)`, allowing us to fine-tune 
heavy models like GFPGAN and Real-ESRGAN in just 6GB of VRAM.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dLoRA(nn.Module):
    """A LoRA wrapper for an existing nn.Conv2d layer."""

    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.base_conv = base_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the base layer
        for param in self.base_conv.parameters():
            param.requires_grad = False

        # If the base layer has a small number of channels, LoRA might not make sense
        in_channels = base_conv.in_channels
        out_channels = base_conv.out_channels
        
        # Rank cannot be larger than the channels
        actual_rank = min(rank, in_channels, out_channels)
        
        # A: down-projection (1x1 conv)
        self.lora_A = nn.Conv2d(
            in_channels=in_channels,
            out_channels=actual_rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # B: up-projection (1x1 conv)
        self.lora_B = nn.Conv2d(
            in_channels=actual_rank,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize A with Kaiming uniform and B with zeros so initial forward pass is identity."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        base_out = self.base_conv(x)
        
        # LoRA forward pass (trainable)
        # We need to account for stride/padding of the original convolution.
        # Since A and B are 1x1, if the base conv has stride > 1, we must apply 
        # that same stride to A (or pool before A) to match spatial dimensions.
        # However, Real-ESRGAN mostly uses stride=1 convs in its body.
        
        # Simplification: we only apply LoRA to stride=1, padding=same convolutions
        # to avoid complex spatial dimension mismatches.
        if self.base_conv.stride != (1, 1):
            return base_out

        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        return base_out + lora_out


def inject_lora(
    module: nn.Module,
    target_module_names: list[str],
    rank: int = 8,
    alpha: float = 16.0
) -> nn.Module:
    """Recursively replaces target Conv2d layers with Conv2dLoRA.
    
    Args:
        module: The PyTorch model.
        target_module_names: Substrings to match (e.g., ["conv1", "body"]).
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Check if this layer's name matches our targets
            if any(target in name for target in target_module_names):
                lora_layer = Conv2dLoRA(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
        else:
            # Recurse
            inject_lora(child, target_module_names, rank, alpha)
            
    return module


def print_trainable_parameters(model: nn.Module) -> None:
    """Utility to print the reduction in trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} | Total params: {total:,} | Percentage: {100 * trainable / total:.2f}%")
