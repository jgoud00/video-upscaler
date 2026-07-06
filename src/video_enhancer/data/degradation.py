"""Synthetic degradation pipeline for training super-resolution and deblurring models.

📘 Concept — The Second-Order Degradation Model
Real-ESRGAN introduced the concept of applying degradations twice to better 
simulate real-world complexities. For example, an image might be compressed by 
a camera (first-order), then uploaded to a social media platform which resizes 
and compresses it again (second-order). We emulate this by chaining blur, 
downsampling, noise, and JPEG artifacts.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def filter2D(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of cv2.filter2D.
    
    Args:
        img: Tensor of shape (C, H, W)
        kernel: Tensor of shape (kH, kW)
        
    Returns:
        Filtered Tensor of shape (C, H, W)
    """
    k = kernel.size(-1)
    # Reshape kernel for conv2d
    weight = kernel.view(1, 1, k, k).repeat(img.size(0), 1, 1, 1)
    
    # Pad input
    padding = k // 2
    img_padded = F.pad(img.unsqueeze(0), (padding, padding, padding, padding), mode="reflect")
    
    return F.conv2d(img_padded, weight, groups=img.size(0)).squeeze(0)


def generate_gaussian_kernel(kernel_size: int = 21, sigma: float = 2.0) -> torch.Tensor:
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    y = torch.arange(kernel_size).float() - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def generate_sinc_kernel(kernel_size: int = 21, cutoff: float = 0.5) -> torch.Tensor:
    """Generate a 2D Sinc filter (circular low-pass) to simulate ringing artifacts.
    
    Args:
        kernel_size: Size of the kernel
        cutoff: Cutoff frequency [0.1, 1.0] (1.0 = Nyquist limit)
    """
    x = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(x, x)
    radius = np.sqrt(xx**2 + yy**2)
    
    # Avoid division by zero at center
    radius[radius == 0] = 1e-8
    
    # Sinc function
    kernel = np.sin(np.pi * cutoff * radius) / (np.pi * cutoff * radius)
    
    # Multiply by a window (e.g. Hamming) to reduce ripple
    window = np.outer(np.hamming(kernel_size), np.hamming(kernel_size))
    kernel = kernel * window
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    return torch.from_numpy(kernel).float()


class DegradationPipeline:
    """Applies complex, randomized degradations to a high-resolution image."""
    
    def __init__(
        self,
        downscale_range: tuple[float, float] = (2.0, 4.0),
        noise_sigma_range: tuple[float, float] = (0.0, 25.0),
        jpeg_quality_range: tuple[int, int] = (30, 95),
        blur_kernel_size: int = 21,
        blur_sigma_range: tuple[float, float] = (0.1, 3.0),
        enable_second_order: bool = True,
    ):
        self.downscale_range = downscale_range
        self.noise_sigma_range = noise_sigma_range
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma_range = blur_sigma_range
        self.enable_second_order = enable_second_order

    def add_jpeg_noise(self, img: torch.Tensor, quality: int) -> torch.Tensor:
        """Add JPEG compression artifacts using OpenCV (CPU).
        
        Args:
            img: Tensor of shape (C, H, W) in [0, 1]
            quality: JPEG quality [1, 100]
        """
        # Convert to numpy uint8
        img_np = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Encode and decode JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        
        # Convert back to Tensor
        out = torch.from_numpy(decimg).float() / 255.0
        return out.permute(2, 0, 1)

    def apply_first_order(self, img: torch.Tensor) -> torch.Tensor:
        """Apply one pass of degradation."""
        # 1. Blur
        sigma = random.uniform(*self.blur_sigma_range)
        kernel = generate_gaussian_kernel(self.blur_kernel_size, sigma).to(img.device)
        img = filter2D(img, kernel)

        # 2. Downsample
        scale = random.uniform(*self.downscale_range)
        h, w = img.shape[-2:]
        new_h, new_w = int(h / scale), int(w / scale)
        # Randomize interpolation mode
        mode = random.choice(["bilinear", "bicubic"])
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode=mode, align_corners=False).squeeze(0)

        # 3. Noise
        noise_sigma = random.uniform(*self.noise_sigma_range) / 255.0
        if noise_sigma > 0:
            noise = torch.randn_like(img) * noise_sigma
            img = img + noise

        # 4. JPEG artifacts
        quality = random.randint(*self.jpeg_quality_range)
        img = self.add_jpeg_noise(img, quality).to(img.device)

        return img.clamp(0, 1)

    def __call__(self, hr_img: torch.Tensor) -> torch.Tensor:
        """Applies degradation to produce the LR counterpart.
        
        Args:
            hr_img: High-resolution clean image (C, H, W), values [0, 1]
            
        Returns:
            Low-resolution degraded image (C, H', W'), values [0, 1]
        """
        out = self.apply_first_order(hr_img)
        if self.enable_second_order:
            out = self.apply_first_order(out)
            
        # Sinc filter for ringing artifacts (80% probability)
        # This is crucial for Real-ESRGAN's ability to de-ring over-sharpened images
        if random.random() < 0.8:
            cutoff = random.uniform(0.1, 1.0)
            sinc_kernel = generate_sinc_kernel(self.blur_kernel_size, cutoff).to(out.device)
            out = filter2D(out, sinc_kernel)
            
        return out.clamp(0, 1)
