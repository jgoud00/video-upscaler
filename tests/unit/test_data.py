"""Unit tests for the data pipeline."""

import pytest
import torch
from PIL import Image

from video_enhancer.config.schema import DataConfig, DegradationConfig
from video_enhancer.data.augment import apply_augmentations
from video_enhancer.data.dataset import VideoPatchDataset
from video_enhancer.data.degradation import DegradationPipeline, filter2D, generate_gaussian_kernel


def test_apply_augmentations():
    # Create two identical fake image tensors
    img1 = torch.rand(3, 64, 64)
    img2 = img1.clone()
    
    # We want to ensure that apply_augmentations modifies both identically
    out1, out2 = apply_augmentations([img1, img2], flip=True, rotate=True)
    
    # Check they remain identical after randomized transforms
    assert torch.allclose(out1, out2)
    
    # Check shape is preserved
    assert out1.shape == (3, 64, 64)


def test_gaussian_kernel():
    kernel = generate_gaussian_kernel(kernel_size=5, sigma=1.0)
    assert kernel.shape == (5, 5)
    # Sum should be very close to 1
    assert torch.isclose(kernel.sum(), torch.tensor(1.0))


def test_filter2d():
    img = torch.ones(3, 10, 10)
    kernel = generate_gaussian_kernel(3, 1.0)
    out = filter2D(img, kernel)
    
    assert out.shape == (3, 10, 10)
    # Blurring an image of all ones with a normalized kernel should yield all ones
    # Use atol=1e-4 because of edge reflection and float precision
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_degradation_pipeline():
    pipeline = DegradationPipeline(
        downscale_range=(2.0, 4.0),
        noise_sigma_range=(5.0, 10.0),
        jpeg_quality_range=(30, 50),
        blur_kernel_size=7,
        blur_sigma_range=(0.5, 1.5),
        enable_second_order=False
    )
    
    hr_img = torch.rand(3, 256, 256)
    lr_img = pipeline(hr_img)
    
    # Should be valid tensor
    assert isinstance(lr_img, torch.Tensor)
    # Values should be clamped [0, 1]
    assert lr_img.min() >= 0.0
    assert lr_img.max() <= 1.0
    # Should be downscaled
    assert lr_img.shape[1] < 256
    assert lr_img.shape[2] < 256


def test_video_patch_dataset(tmp_path):
    # Setup fake data directory
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    
    # Create a dummy image
    img = Image.new("RGB", (800, 600), color="red")
    img.save(data_dir / "frame_001.png")
    
    data_config = DataConfig(patch_size=64)  # LR 64x64
    deg_config = DegradationConfig()
    
    dataset = VideoPatchDataset(
        data_dir=data_dir,
        data_config=data_config,
        deg_config=deg_config,
        scale=4  # HR will be 256x256
    )
    
    assert len(dataset) == 1
    
    item = dataset[0]
    assert "lr" in item
    assert "hr" in item
    assert "path" in item
    
    lr = item["lr"]
    hr = item["hr"]
    
    # Check expected shapes
    assert hr.shape == (3, 256, 256)
    assert lr.shape == (3, 64, 64)
    
    # Check value ranges
    assert hr.min() >= 0.0 and hr.max() <= 1.0
    assert lr.min() >= 0.0 and lr.max() <= 1.0
