"""Download standard SR benchmark datasets and generate baseline test pairs.

Creates synthetic test images if download fails, then runs pretrained Real-ESRGAN
to produce enhanced outputs for baseline evaluation.
"""

import sys
import urllib.request
import ssl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "test"
GT_DIR = TEST_DIR / "gt"
LR_DIR = TEST_DIR / "lr"
ENHANCED_DIR = TEST_DIR / "enhanced"


def generate_synthetic_test_images(gt_dir: Path, count: int = 8) -> None:
    """Generate diverse synthetic HR test images for benchmarking.

    Creates images with varied texture patterns that stress-test SR models:
    gradients, checkerboards, noise textures, circular patterns, etc.
    """
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)

    generators = [
        ("gradient_h", lambda: _gen_gradient(256, 256, "horizontal")),
        ("gradient_d", lambda: _gen_gradient(256, 256, "diagonal")),
        ("checker", lambda: _gen_checkerboard(256, 256, 16)),
        ("circles", lambda: _gen_circles(256, 256)),
        ("texture_fine", lambda: _gen_texture(256, 256, rng, scale=2)),
        ("texture_coarse", lambda: _gen_texture(256, 256, rng, scale=8)),
        ("face_proxy", lambda: _gen_face_proxy(256, 256)),
        ("mixed", lambda: _gen_mixed(256, 256, rng)),
    ]

    for i, (name, gen_fn) in enumerate(generators[:count]):
        img = gen_fn()
        path = gt_dir / f"{name}.png"
        cv2.imwrite(str(path), img)
        print(f"  Generated: {name}.png ({img.shape[1]}x{img.shape[0]})")


def _gen_gradient(h: int, w: int, direction: str) -> np.ndarray:
    if direction == "horizontal":
        grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    elif direction == "diagonal":
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        grad = ((xx + yy) / 2 * 255).astype(np.uint8)
    else:
        grad = np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    return cv2.merge([grad, grad, grad])


def _gen_checkerboard(h: int, w: int, cell: int) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((y // cell) + (x // cell)) % 2 == 0:
                img[y:y+cell, x:x+cell] = 255
    return cv2.merge([img, img, img])


def _gen_circles(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    colors = [(200, 100, 50), (50, 200, 100), (100, 50, 200), (200, 200, 50)]
    for i, r in enumerate(range(min(h, w) // 2, 10, -20)):
        cv2.circle(img, (cx, cy), r, colors[i % len(colors)], -1)
    return img


def _gen_texture(h: int, w: int, rng: np.random.RandomState, scale: int = 4) -> np.ndarray:
    small = rng.randint(0, 256, (h // scale, w // scale, 3), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(img, 0, 255).astype(np.uint8)


def _gen_face_proxy(h: int, w: int) -> np.ndarray:
    """Oval + features — a proxy to test face-like regions without real faces."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(img, (cx, cy), (60, 80), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (cx - 20, cy - 15), 6, (50, 50, 50), -1)  # left eye
    cv2.circle(img, (cx + 20, cy - 15), 6, (50, 50, 50), -1)  # right eye
    cv2.ellipse(img, (cx, cy + 25), (15, 8), 0, 0, 180, (150, 80, 80), 2)  # mouth
    return img


def _gen_mixed(h: int, w: int, rng: np.random.RandomState) -> np.ndarray:
    """Mix of flat regions + sharp edges + fine texture."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    # Flat colored quadrants
    img[:h//2, :w//2] = [40, 60, 120]
    img[:h//2, w//2:] = [120, 40, 60]
    # Noisy bottom half
    noise = rng.randint(0, 256, (h//2, w, 3), dtype=np.uint8)
    img[h//2:, :] = noise
    # Sharp diagonal line
    cv2.line(img, (0, 0), (w, h), (255, 255, 0), 2)
    return img


def create_degraded_lr(hr_img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Synthetically degrade: downscale + JPEG compression + noise."""
    h, w = hr_img.shape[:2]
    lr_h, lr_w = h // scale, w // scale

    lr = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

    noise = np.random.normal(0, 5, lr.shape).astype(np.float32)
    lr = np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, buf = cv2.imencode(".jpg", lr, encode_param)
    lr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return lr


def try_download_set5(gt_dir: Path) -> bool:
    """Attempt to download Set5 from a known working URL."""
    urls = [
        ("https://raw.githubusercontent.com/Le0v1n/Set5-Set14-BSD100-Urban100/main/Set5/baby.png", "baby.png"),
        ("https://raw.githubusercontent.com/Le0v1n/Set5-Set14-BSD100-Urban100/main/Set5/bird.png", "bird.png"),
        ("https://raw.githubusercontent.com/Le0v1n/Set5-Set14-BSD100-Urban100/main/Set5/butterfly.png", "butterfly.png"),
        ("https://raw.githubusercontent.com/Le0v1n/Set5-Set14-BSD100-Urban100/main/Set5/head.png", "head.png"),
        ("https://raw.githubusercontent.com/Le0v1n/Set5-Set14-BSD100-Urban100/main/Set5/woman.png", "woman.png"),
    ]

    gt_dir.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    downloaded = 0
    for url, filename in urls:
        dest = gt_dir / filename
        if dest.exists():
            downloaded += 1
            continue
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")

    return downloaded >= 3


def run_realesrgan_enhancement(lr_dir: Path, enhanced_dir: Path) -> None:
    """Run pretrained Real-ESRGAN on all LR images."""
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "pretrained" / "RealESRGAN_x4plus.pth"
    if not ckpt_path.exists():
        print(f"ERROR: Pretrained checkpoint not found at {ckpt_path}")
        return

    import torchvision.transforms.functional as TF
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        sys.modules["torchvision.transforms.functional_tensor"] = TF

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(ckpt_path),
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )

    lr_images = sorted(f for f in lr_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    print(f"Enhancing {len(lr_images)} images with Real-ESRGAN...")

    for img_path in lr_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            output, _ = upsampler.enhance(img, outscale=4)
            out_path = enhanced_dir / img_path.with_suffix(".png").name
            cv2.imwrite(str(out_path), output)
            print(f"  OK {img_path.name} -> {out_path.name}")
        except Exception as e:
            print(f"  FAIL {img_path.name}: {e}")

    del upsampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print("Step 0: Download Test Data & Generate Baseline")
    print("=" * 60)

    # 1. Try downloading Set5
    print("\n[1/3] Attempting to download Set5 benchmark...")
    downloaded = try_download_set5(GT_DIR)

    if not downloaded:
        print("  Download failed. Generating synthetic test images instead.")
        generate_synthetic_test_images(GT_DIR)
    else:
        # Also add synthetic images for texture diversity
        print("  Adding synthetic test images for texture coverage...")
        generate_synthetic_test_images(GT_DIR, count=4)

    # 2. Create LR degraded versions
    print(f"\n[2/3] Creating degraded LR versions...")
    LR_DIR.mkdir(parents=True, exist_ok=True)

    gt_images = sorted(f for f in GT_DIR.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    for img_path in gt_images:
        hr_img = cv2.imread(str(img_path))
        if hr_img is None:
            continue
        h, w = hr_img.shape[:2]
        if h < 32 or w < 32:
            continue
        lr_img = create_degraded_lr(hr_img, scale=4)
        cv2.imwrite(str(LR_DIR / img_path.name), lr_img)
        print(f"  {img_path.name}: HR {w}x{h} -> LR {lr_img.shape[1]}x{lr_img.shape[0]}")

    # 3. Run Real-ESRGAN
    print(f"\n[3/3] Running Real-ESRGAN on degraded images...")
    run_realesrgan_enhancement(LR_DIR, ENHANCED_DIR)

    print("\n" + "=" * 60)
    print("Test data ready!")
    print(f"  GT:       {GT_DIR} ({len(list(GT_DIR.glob('*.png')))} images)")
    print(f"  LR:       {LR_DIR} ({len(list(LR_DIR.glob('*.png')))} images)")
    print(f"  Enhanced: {ENHANCED_DIR} ({len(list(ENHANCED_DIR.glob('*.png')))} images)")
    print(f"\nNext: python scripts/evaluate.py --input-dir {ENHANCED_DIR} --gt-dir {GT_DIR} --measure-perf")
    print("=" * 60)


if __name__ == "__main__":
    main()
