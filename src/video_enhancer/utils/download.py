"""Helper utility to download pretrained models."""

import urllib.request
from pathlib import Path
from typing import Optional

from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)

URLS = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
}

def download_pretrained_model(model_name: str, save_dir: Path | str = "checkpoints/pretrained") -> Path:
    """Download a pretrained model if it doesn't exist.
    
    Args:
        model_name: Key in the URLS dictionary.
        save_dir: Directory to save the model.
        
    Returns:
        Path to the downloaded model.
    """
    if model_name not in URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(URLS.keys())}")
        
    url = URLS[model_name]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / f"{model_name}.pth"
    
    if file_path.exists():
        logger.info(f"Model {model_name} already exists at {file_path}")
        return file_path
        
    logger.info(f"Downloading {model_name} from {url}...")
    try:
        # Simple urllib download. In a production system we'd add a progress bar (e.g. rich.progress)
        urllib.request.urlretrieve(url, file_path)
        logger.info(f"Successfully downloaded to {file_path}")
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        if file_path.exists():
            file_path.unlink()
        raise e
        
    return file_path
