"""Dataset for loading high-resolution images and generating paired LR/HR patches."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from video_enhancer.config.schema import DataConfig, DegradationConfig
from video_enhancer.data.augment import apply_augmentations
from video_enhancer.data.degradation import DegradationPipeline


class VideoPatchDataset(Dataset):
    """Loads clean HR images, crops patches, and generates degraded LR counterparts."""

    def __init__(
        self,
        data_dir: Path,
        data_config: DataConfig,
        deg_config: DegradationConfig,
        scale: int = 4,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing high-resolution training frames.
            data_config: Configuration for patches and augmentation.
            deg_config: Configuration for the degradation pipeline.
            scale: Target upsampling scale (e.g., 4 for x4 super-resolution).
        """
        self.data_dir = Path(data_dir)
        self.data_config = data_config
        self.scale = scale
        
        # Discover all images
        self.image_paths = sorted(
            p for p in self.data_dir.rglob("*") 
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        # Ensure target HR patch size is a multiple of scale
        hr_size = self.data_config.patch_size * self.scale
        self.hr_patch_size = hr_size

        self.degradation = DegradationPipeline(
            downscale_range=deg_config.downscale_range,
            noise_sigma_range=deg_config.noise_sigma_range,
            jpeg_quality_range=deg_config.jpeg_quality_range,
            blur_kernel_size=deg_config.blur_kernel_size,
            blur_sigma_range=deg_config.blur_sigma_range,
            enable_second_order=deg_config.enable_second_order,
        )

        # Cache images in RAM if the dataset is small enough (prevents Disk I/O bottleneck)
        self.image_cache = {}
        self.use_cache = len(self.image_paths) < 500
        if self.use_cache:
            print(f"Dataset has {len(self.image_paths)} images (<500). In-memory caching enabled.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_len = getattr(self.data_config, "sequence_length", 1)
        
        # Calculate indices for the sequence
        # We want `idx` to be the center frame (or roughly center if even).
        half = seq_len // 2
        indices = []
        base_path = self.image_paths[idx].parent
        
        # If the image comes from a static dataset like DIV2K, duplicate it to form a static sequence.
        # This teaches the video model how to cleanly handle still frames without motion artifacts.
        is_static = "DIV2K" in base_path.name or "Flickr2K" in base_path.name
        
        for i in range(seq_len):
            if is_static:
                indices.append(idx)
            else:
                target_idx = max(0, min(len(self) - 1, idx + i - half))
                # Ensure we don't bleed across different video folders or into image datasets
                if self.image_paths[target_idx].parent == base_path:
                    indices.append(target_idx)
                else:
                    indices.append(idx)  # Replicate padding at video boundaries
        
        hr_patches = []
        
        # We must use the same crop and augmentation for all frames in the sequence
        top, left = None, None
        
        for curr_idx in indices:
            img_path = self.image_paths[curr_idx]
            
            # Load HR image
            if self.use_cache and curr_idx in self.image_cache:
                hr_img = self.image_cache[curr_idx]
            else:
                try:
                    with Image.open(img_path) as img:
                        hr_img = img.convert("RGB")
                    if self.use_cache:
                        self.image_cache[curr_idx] = hr_img
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
                    return self.__getitem__(random.randint(0, len(self) - 1))

            w, h = hr_img.size
            
            # If image is smaller than patch size, pad it
            pad_w = max(0, self.hr_patch_size - w)
            pad_h = max(0, self.hr_patch_size - h)
            if pad_w > 0 or pad_h > 0:
                hr_img = F.pad(hr_img, (0, 0, pad_w, pad_h), padding_mode="reflect")
                w, h = hr_img.size

            # Determine crop coordinates only on the first frame
            if top is None:
                top = random.randint(0, h - self.hr_patch_size)
                left = random.randint(0, w - self.hr_patch_size)
                
            # Random Crop
            hr_patch = F.crop(hr_img, top, left, self.hr_patch_size, self.hr_patch_size)
            hr_patches.append(hr_patch)
            
        # 3. Augmentation (flip, rotate) - apply to all frames consistently
        hr_patches = apply_augmentations(
            hr_patches, 
            flip=self.data_config.augment_flip, 
            rotate=self.data_config.augment_rotate
        )

        # Convert to tensors
        hr_tensors = [F.to_tensor(p) for p in hr_patches]
        lr_tensors = []
        
        # 4. Degradation pipeline
        for hr_t in hr_tensors:
            lr_t = self.degradation(hr_t)
            lr_t = torch.nn.functional.interpolate(
                lr_t.unsqueeze(0), 
                size=(self.data_config.patch_size, self.data_config.patch_size), 
                mode="bicubic", 
                align_corners=False
            ).squeeze(0)
            lr_t = lr_t.clamp(0, 1)
            lr_tensors.append(lr_t)

        # 5. Output
        if seq_len == 1:
            lr_out = lr_tensors[0]
            hr_out = hr_tensors[0]
        else:
            # Concatenate along channel dimension [C*seq_len, H, W]
            lr_out = torch.cat(lr_tensors, dim=0) 
            # The center frame is the target
            hr_out = hr_tensors[half]
            
        return {
            "lr": lr_out,
            "hr": hr_out,
            "path": str(self.image_paths[idx])
        }
