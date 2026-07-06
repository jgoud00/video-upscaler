"""Comprehensive evaluation metrics for video enhancement.

Metric hierarchy (for GAN-based outputs):
  PRIMARY:    LPIPS — perceptual similarity via deep features
  NO-REF:     NIQE, BRISQUE — no ground truth needed (real-world footage)
  SECONDARY:  PSNR, SSIM — pixel-aligned, useful for literature comparison
  TEMPORAL:   T-loss — frame-to-frame consistency in static regions
  PERFORMANCE: FPS, peak VRAM

Why this ordering matters:
  PSNR/SSIM penalize plausible hallucinated detail that doesn't pixel-align
  with ground truth, even when it looks better to a human. GAN restorers
  (Real-ESRGAN, CodeFormer) produce perceptually superior but pixel-shifted
  textures, so LPIPS (which operates on deep features, not pixels) is the
  correct primary signal. PSNR/SSIM are still logged for regression detection
  and literature comparability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    PYIQA_AVAILABLE = False


@dataclass
class MetricsResult:
    """Container for all evaluation metrics."""
    # Primary (perceptual)
    lpips: Optional[float] = None
    # No-reference
    niqe: Optional[float] = None
    brisque: Optional[float] = None
    # Secondary (pixel-aligned)
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    # Temporal
    t_loss: Optional[float] = None
    # Performance
    fps: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    # Metadata
    num_images: int = 0
    has_ground_truth: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsCalculator:
    """Computes all evaluation metrics for enhanced images/frames."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # LPIPS (primary metric)
        self._lpips_net = None
        if LPIPS_AVAILABLE:
            try:
                self._lpips_net = lpips.LPIPS(net="vgg").to(self.device)
                self._lpips_net.eval()
                for p in self._lpips_net.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"Warning: Failed to initialize LPIPS (network issue?): {e}")
                self._lpips_net = None

        # NIQE / BRISQUE (no-reference)
        self._niqe = None
        self._brisque = None
        if PYIQA_AVAILABLE:
            self._niqe = pyiqa.create_metric("niqe", device=self.device)
            self._brisque = pyiqa.create_metric("brisque", device=self.device)

    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert HWC uint8 BGR image to 1CHW float32 [0,1] RGB tensor."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def compute_lpips(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """LPIPS between pred and gt (both HWC uint8 BGR). Lower = better."""
        if self._lpips_net is None:
            raise RuntimeError("lpips not installed")
        with torch.no_grad():
            p = self._img_to_tensor(pred) * 2.0 - 1.0  # [0,1] -> [-1,1]
            g = self._img_to_tensor(gt) * 2.0 - 1.0
            return self._lpips_net(p, g).item()

    def compute_niqe(self, img: np.ndarray) -> float:
        """NIQE score (no reference). Lower = better."""
        if self._niqe is None:
            raise RuntimeError("pyiqa not installed")
        with torch.no_grad():
            t = self._img_to_tensor(img)
            return self._niqe(t).item()

    def compute_brisque(self, img: np.ndarray) -> float:
        """BRISQUE score (no reference). Lower = better."""
        if self._brisque is None:
            raise RuntimeError("pyiqa not installed")
        with torch.no_grad():
            t = self._img_to_tensor(img)
            return self._brisque(t).item()

    @staticmethod
    def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
        """PSNR in dB. Higher = better."""
        return float(compute_psnr(gt, pred, data_range=255))

    @staticmethod
    def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
        """SSIM. Higher = better."""
        min_dim = min(pred.shape[0], pred.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(compute_ssim(gt, pred, channel_axis=-1, win_size=win_size, data_range=255))

    @staticmethod
    def compute_t_loss(frames: list[np.ndarray], roi_fraction: float = 0.25) -> float:
        """Temporal consistency: mean absolute pixel diff between consecutive
        frames in a static ROI (top-left quadrant by default).

        Lower = more temporally stable. Zero would mean the model ignores
        real motion, so this is "lower is generally better, but not literally zero".

        Args:
            frames: List of HWC uint8 images (enhanced output frames in order).
            roi_fraction: Fraction of frame height/width to use as static ROI.

        Returns:
            Mean absolute pixel difference across consecutive frame pairs in the ROI.
        """
        if len(frames) < 2:
            return 0.0

        diffs = []
        for i in range(len(frames) - 1):
            h, w = frames[i].shape[:2]
            rh, rw = int(h * roi_fraction), int(w * roi_fraction)
            roi_a = frames[i][:rh, :rw].astype(np.float32)
            roi_b = frames[i + 1][:rh, :rw].astype(np.float32)
            diffs.append(np.mean(np.abs(roi_a - roi_b)))

        return float(np.mean(diffs))

    def evaluate_directory(
        self,
        input_dir: Path,
        gt_dir: Optional[Path] = None,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> MetricsResult:
        """Evaluate all images in a directory.

        Args:
            input_dir: Directory of enhanced/predicted images.
            gt_dir: Directory of ground-truth images (same filenames). Optional.
            extensions: Image file extensions to include.

        Returns:
            MetricsResult with averaged metrics.
        """
        input_dir = Path(input_dir)
        pred_files = sorted(
            f for f in input_dir.iterdir()
            if f.suffix.lower() in extensions
        )
        if not pred_files:
            raise FileNotFoundError(f"No images found in {input_dir}")

        has_gt = gt_dir is not None and Path(gt_dir).exists()
        gt_dir = Path(gt_dir) if has_gt else None

        # Accumulators
        lpips_vals, niqe_vals, brisque_vals = [], [], []
        psnr_vals, ssim_vals = [], []
        all_frames = []

        for pred_path in pred_files:
            pred_img = cv2.imread(str(pred_path))
            if pred_img is None:
                continue

            # No-reference metrics (always computed)
            if self._niqe is not None:
                niqe_vals.append(self.compute_niqe(pred_img))
            if self._brisque is not None:
                brisque_vals.append(self.compute_brisque(pred_img))

            # Full-reference metrics (only with GT)
            if has_gt:
                gt_path = gt_dir / pred_path.name
                if gt_path.exists():
                    gt_img = cv2.imread(str(gt_path))
                    if gt_img is not None:
                        # Resize GT to match pred if needed (SR changes resolution)
                        if gt_img.shape[:2] != pred_img.shape[:2]:
                            gt_img = cv2.resize(
                                gt_img,
                                (pred_img.shape[1], pred_img.shape[0]),
                                interpolation=cv2.INTER_CUBIC,
                            )
                        if self._lpips_net is not None:
                            lpips_vals.append(self.compute_lpips(pred_img, gt_img))
                        psnr_vals.append(self.compute_psnr(pred_img, gt_img))
                        ssim_vals.append(self.compute_ssim(pred_img, gt_img))

            all_frames.append(pred_img)

        result = MetricsResult(
            num_images=len(pred_files),
            has_ground_truth=has_gt,
        )

        if lpips_vals:
            result.lpips = float(np.mean(lpips_vals))
        if niqe_vals:
            result.niqe = float(np.mean(niqe_vals))
        if brisque_vals:
            result.brisque = float(np.mean(brisque_vals))
        if psnr_vals:
            result.psnr = float(np.mean(psnr_vals))
        if ssim_vals:
            result.ssim = float(np.mean(ssim_vals))

        # T-loss (temporal consistency across sequential frames)
        if len(all_frames) >= 2:
            result.t_loss = self.compute_t_loss(all_frames)

        return result

    def measure_inference_performance(
        self,
        model: torch.nn.Module,
        input_size: tuple = (1, 3, 256, 256),
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> tuple[float, float]:
        """Measure inference FPS and peak VRAM.

        Returns:
            (fps, peak_vram_mb)
        """
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.randn(*input_size, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        fps = num_runs / elapsed
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return fps, peak_vram_mb
