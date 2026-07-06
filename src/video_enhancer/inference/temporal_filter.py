"""Temporal Filtering to prevent flickering.

This implements a lightweight Exponential Moving Average (EMA) blend 
over enhanced frames during inference. It runs instantly and consumes 
virtually 0 extra VRAM, perfect for the 6GB limit.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional


class TemporalEMAFilter:
    """Blends current frame with previous frames to reduce temporal jitter."""
    
    def __init__(self, alpha: float = 0.8, scene_cut_threshold: float = 0.3):
        """
        Args:
            alpha: Blend factor. 1.0 = no blending (current frame only), 
                   0.0 = completely freeze at first frame.
                   Default 0.8 means 80% current frame, 20% history.
            scene_cut_threshold: SSIM threshold below which we assume a scene cut 
                                 has occurred and reset the filter.
        """
        self.alpha = alpha
        self.scene_cut_threshold = scene_cut_threshold
        self.prev_frame: Optional[np.ndarray] = None

    def _detect_scene_cut(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> bool:
        """Detects if the scene has changed abruptly using SSIM."""
        # Calculate SSIM on grayscale downsampled images for speed
        # Using a fixed win_size of 3 to avoid errors if the frame is very small
        try:
            score = ssim(current_frame, prev_frame, channel_axis=-1, win_size=3)
            return score < self.scene_cut_threshold
        except Exception:
            # Fallback to simple MSE if SSIM fails
            mse = np.mean((current_frame.astype(np.float32) - prev_frame.astype(np.float32)) ** 2)
            return mse > 5000  # Arbitrary high threshold

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Applies EMA blending to the frame.
        
        Args:
            frame: Numpy array [H, W, C] in range [0, 255], uint8.
            
        Returns:
            Filtered frame [H, W, C] in range [0, 255], uint8.
        """
        if self.prev_frame is None:
            self.prev_frame = frame.astype(np.float32)
            return frame

        # Check for scene cut to avoid ghosting across cuts
        if self._detect_scene_cut(frame, self.prev_frame.astype(np.uint8)):
            self.prev_frame = frame.astype(np.float32)
            return frame

        # Apply Exponential Moving Average
        current_float = frame.astype(np.float32)
        blended = self.alpha * current_float + (1.0 - self.alpha) * self.prev_frame
        
        self.prev_frame = blended
        
        return np.clip(blended, 0, 255).astype(np.uint8)


class PyTorchTemporalEMA:
    """PyTorch native equivalent for applying EMA directly on GPU tensors."""
    def __init__(self, alpha: float = 0.8, scene_cut_threshold: float = 0.1):
        self.alpha = alpha
        self.scene_cut_threshold = scene_cut_threshold
        self.prev_tensor: Optional[torch.Tensor] = None

    def process(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: [1, C, H, W] tensor in [0, 1] on GPU.
        """
        if self.prev_tensor is None:
            self.prev_tensor = tensor.detach().clone()
            return tensor

        # Fast MSE scene cut detection
        mse = torch.nn.functional.mse_loss(tensor, self.prev_tensor)
        if mse > self.scene_cut_threshold:
            self.prev_tensor = tensor.detach().clone()
            return tensor
            
        blended = self.alpha * tensor + (1.0 - self.alpha) * self.prev_tensor
        self.prev_tensor = blended.detach().clone()
        return blended
