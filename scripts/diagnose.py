"""System and CUDA diagnostic script.

Run this after installation to verify the environment is correctly set up.
Reports: Python version, PyTorch version, CUDA availability, GPU details,
VRAM capacity, cuDNN status, and ffmpeg availability.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_enhancer.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("diagnose")


def check_python() -> None:
    logger.info(f"Python: {sys.version}")
    logger.info(f"Executable: {sys.executable}")


def check_torch() -> None:
    try:
        import torch

        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_mem / (1024 ** 2)
                logger.info(
                    f"GPU {i}: {props.name} | "
                    f"Compute: {props.major}.{props.minor} | "
                    f"VRAM: {total_mb:.0f} MB"
                )

            # Quick allocation test
            try:
                test_tensor = torch.zeros(1, 3, 256, 256, device="cuda", dtype=torch.float16)
                allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                logger.info(f"Test allocation (fp16 256×256): {allocated_mb:.1f} MB ✓")
                del test_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                logger.error(f"GPU allocation test failed: {e}")
        else:
            logger.warning("No CUDA GPU detected — training will not be possible")
            logger.info("Inference will fall back to CPU (slower)")

        # AMP support
        if torch.cuda.is_available():
            amp_supported = hasattr(torch.cuda, "amp") or hasattr(torch, "autocast")
            logger.info(f"AMP (mixed precision) support: {amp_supported}")
            bf16 = torch.cuda.is_bf16_supported()
            logger.info(f"BFloat16 support: {bf16}")

    except ImportError:
        logger.error("PyTorch is NOT installed")


def check_dependencies() -> None:
    deps = {
        "basicsr": "basicsr",
        "realesrgan": "realesrgan",
        "gfpgan": "gfpgan",
        "facexlib": "facexlib",
        "cv2": "opencv-python",
        "lpips": "lpips",
        "pydantic": "pydantic",
        "yaml": "pyyaml",
        "rich": "rich",
        "tensorboard": "tensorboard",
    }
    for import_name, pip_name in deps.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            logger.info(f"  {pip_name}: {version} ✓")
        except ImportError:
            logger.warning(f"  {pip_name}: NOT INSTALLED")


def check_ffmpeg() -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10,
            )
            first_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            logger.info(f"ffmpeg: {first_line}")
            logger.info(f"  Path: {ffmpeg_path}")
        except Exception as e:
            logger.warning(f"ffmpeg found but version check failed: {e}")
    else:
        logger.warning("ffmpeg: NOT FOUND on PATH")
        logger.warning("  Video processing requires ffmpeg. Install via: winget install Gyan.FFmpeg")


def check_disk_space() -> None:
    project_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = project_root / "checkpoints"
    data_dir = project_root / "data"
    for d in [checkpoints_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    import os
    total, used, free = shutil.disk_usage(project_root)
    logger.info(
        f"Disk ({project_root.drive}): "
        f"{free / (1024**3):.1f} GB free / {total / (1024**3):.1f} GB total"
    )


def main() -> None:
    logger.info("=" * 60)
    logger.info("Video Enhancer — System Diagnostics")
    logger.info("=" * 60)

    logger.info("\n[bold]1. Python[/bold]")
    check_python()

    logger.info("\n[bold]2. PyTorch & CUDA[/bold]")
    check_torch()

    logger.info("\n[bold]3. Dependencies[/bold]")
    check_dependencies()

    logger.info("\n[bold]4. ffmpeg[/bold]")
    check_ffmpeg()

    logger.info("\n[bold]5. Disk Space[/bold]")
    check_disk_space()

    logger.info("\n" + "=" * 60)
    logger.info("Diagnostics complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
