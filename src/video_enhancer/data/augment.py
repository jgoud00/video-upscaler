"""Geometric augmentations for image patches."""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torchvision.transforms.functional as F
from PIL import Image


def apply_augmentations(
    patches: Sequence[torch.Tensor | Image.Image],
    flip: bool = True,
    rotate: bool = True,
) -> tuple[torch.Tensor | Image.Image, ...]:
    """Applies consistent geometric augmentations across a sequence of patches.
    
    Useful for ensuring that LR and HR patch pairs receive the exact same 
    rotation and flip operations.
    
    Args:
        patches: Sequence of image patches (e.g., [lr_patch, hr_patch])
        flip: Whether to apply random horizontal/vertical flips.
        rotate: Whether to apply random 90-degree rotations.
        
    Returns:
        Tuple of augmented patches.
    """
    if not patches:
        return tuple()

    # Determine random transforms
    h_flip = flip and random.random() < 0.5
    v_flip = flip and random.random() < 0.5
    rot_angle = random.choice([0, 90, 180, 270]) if rotate else 0

    augmented = []
    for patch in patches:
        if h_flip:
            patch = F.hflip(patch)
        if v_flip:
            patch = F.vflip(patch)
        if rot_angle > 0:
            # For 90 degree increments, F.rotate is exact and fast.
            patch = F.rotate(patch, rot_angle)
        augmented.append(patch)

    return tuple(augmented)
