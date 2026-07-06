import os
from pathlib import Path
from PIL import Image
from video_enhancer.data.dataset import VideoPatchDataset
from video_enhancer.config.schema import DataConfig, DegradationConfig

def create_dummy_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (100, 100), color='red')
    img.save(path)

def main():
    base_dir = Path("data/train_test")
    # Create subdirectories
    div2k_dir = base_dir / "DIV2K_train_HR"
    own_dir = base_dir / "own_footage"
    
    # Create 3 images in DIV2K and 2 in own_footage
    create_dummy_png(div2k_dir / "0001.png")
    create_dummy_png(div2k_dir / "0002.png")
    create_dummy_png(div2k_dir / "0003.png")
    
    create_dummy_png(own_dir / "frame_001.png")
    create_dummy_png(own_dir / "frame_002.png")
    
    # Init dataset
    ds = VideoPatchDataset(
        data_dir=base_dir,
        data_config=DataConfig(),
        deg_config=DegradationConfig(),
        scale=4
    )
    
    print(f"Dataset found {len(ds)} images in {base_dir}")
    assert len(ds) == 5, "Dataset did not find all 5 images!"
    print("Verification Passed: dataset.py successfully loads from multiple subdirectories using rglob.")
    
    # Cleanup
    import shutil
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    main()
