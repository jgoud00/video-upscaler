# Video Enhancer

Fine-tune pretrained deep learning models for video enhancement on consumer hardware (RTX 3050 6GB VRAM).

## Target Models

| Model | Task | Parameters | Status |
|---|---|---|---|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Super-Resolution | 16.7M | Phase 4 |
| [NAFNet](https://github.com/megvii-research/NAFNet) | Deblurring / Denoising | ~42M | Phase 5 |
| [GFPGAN](https://github.com/TencentARC/GFPGAN) | Face Enhancement | ~86M | Phase 6 |
| [RIFE](https://github.com/hzwer/Practical-RIFE) | Frame Interpolation | ~9.8M | Phase 7 |

## VRAM-Constrained Strategy

All fine-tuning fits in 6GB VRAM via:
- **Partial freezing** — freeze early feature-extraction layers
- **Mixed precision (AMP fp16)** — halves memory for params/activations
- **Small patches** — 128×128 LR input (512×512 HR output)
- **Gradient accumulation** — effective batch size 4+ with bs=1
- **Gradient checkpointing** — trades compute for memory on heavy models
- **One model at a time** — never load multiple models for training

## Project Structure

```
video-enhancer/
├── configs/           # YAML training/inference configs
├── src/video_enhancer/
│   ├── config/        # Config schema (Pydantic) + loader
│   ├── data/          # Datasets, degradation pipeline, augmentation
│   ├── models/        # Pretrained wrappers, LoRA adapters, registry
│   ├── losses/        # Pixel, perceptual, GAN, temporal losses
│   ├── training/      # BaseFineTuner, checkpointing, AMP, resume
│   ├── inference/     # Video inference, model chaining
│   ├── evaluation/    # PSNR/SSIM/LPIPS, benchmarking
│   ├── pipelines/     # Video → frames → model(s) → video
│   ├── utils/         # Logging, video I/O, VRAM profiling
│   └── gui/           # Desktop GUI (Phase 11)
├── scripts/           # CLI entrypoints
├── tests/             # pytest unit + integration tests
├── checkpoints/       # pretrained/ and finetuned/ (gitignored)
├── data/              # Training data (gitignored)
└── logs/              # TensorBoard logs (gitignored)
```

## Quick Start

```bash
# Create venv and install
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

# Verify GPU
python scripts/diagnose.py

# Fine-tune (example)
python scripts/finetune.py --config configs/finetune_realesrgan.yaml

# Inference
python scripts/infer.py --config configs/inference.yaml --input video.mp4
```

## License

MIT. Individual pretrained models retain their original licenses:
- Real-ESRGAN: BSD-3-Clause
- GFPGAN: Apache 2.0
- RIFE: MIT
- NAFNet: MIT
