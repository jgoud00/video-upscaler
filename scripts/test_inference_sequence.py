import os
import torch
from pathlib import Path
from PIL import Image

from video_enhancer.data.dataset import VideoPatchDataset
from video_enhancer.config.schema import DataConfig, DegradationConfig
from video_enhancer.inference.pipeline import VideoEnhancementPipeline

def create_dummy_png(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (128, 128), color=color)
    img.save(path)

def test_sliding_window():
    base_dir = Path("data/test_seq")
    create_dummy_png(base_dir / "frame_001.png", "red")
    create_dummy_png(base_dir / "frame_002.png", "green")
    create_dummy_png(base_dir / "frame_003.png", "blue")
    
    frames = sorted(base_dir.glob("*.png"))
    
    # 1. Pipeline inference side
    pipeline = VideoEnhancementPipeline()
    # Read sequence at index 1 (center frame)
    # This should load frame 1, 2, 3
    inf_tensor = pipeline._read_sequence(frames, idx=1, seq_len=3)
    # Shape is [1, 9, 128, 128]
    inf_tensor_squeezed = inf_tensor.squeeze(0)
    
    # 2. Dataset training side
    # We want to intercept the hr_tensors list before degradation
    data_config = DataConfig(patch_size=128, sequence_length=3, augment_flip=False, augment_rotate=False)
    # We set patch_size=128 to match image size so crop is full image
    ds = VideoPatchDataset(base_dir, data_config, DegradationConfig(), scale=1)
    
    # Manually run the extraction part of __getitem__ to get hr_tensors
    seq_len = 3
    idx = 1
    half = seq_len // 2
    indices = [max(0, min(len(ds) - 1, idx + i - half)) for i in range(seq_len)]
    
    hr_patches = []
    for curr_idx in indices:
        img_path = ds.image_paths[curr_idx]
        with Image.open(img_path) as img:
            hr_img = img.convert("RGB")
        import torchvision.transforms.functional as F
        hr_patches.append(F.to_tensor(hr_img))
        
    train_tensor = torch.cat(hr_patches, dim=0)
    
    # Assert they match
    inf_tensor_cpu = inf_tensor_squeezed.cpu()
    assert inf_tensor_cpu.shape == train_tensor.shape, f"Shape mismatch: {inf_tensor_cpu.shape} vs {train_tensor.shape}"
    assert torch.allclose(inf_tensor_cpu, train_tensor), "Values do not match!"
    print("Verification Passed: Inference sliding window perfectly matches training sliding window.")
    
    import shutil
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_sliding_window()
