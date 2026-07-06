"""Tile-based inference for large images on limited VRAM.

Splits input tensors into overlapping tiles, processes each individually,
and reassembles with linear blending to eliminate seam artifacts.
The output buffer lives on CPU; only the active tile occupies GPU VRAM.
"""

import torch
import numpy as np
from typing import Callable

from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class TileProcessor:
    """Processes large images tile-by-tile to stay within VRAM limits.

    Strategy: keep the full-res output buffer on CPU. For each tile:
      1. Slice the input tile (small, on GPU)
      2. Run model forward pass (GPU, fp16)
      3. Move result to CPU immediately
      4. Accumulate into CPU output buffer with blending
    This way GPU only ever holds: model weights + 1 tile input + 1 tile output.
    """

    def __init__(self, tile_size: int = 256, tile_overlap: int = 24, scale: int = 4):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.scale = scale
        self._logged = False

    def process_tensor(
        self, img: torch.Tensor, model_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Process a [1,C,H,W] tensor through model_fn using tiled inference."""
        device = img.device
        _, c, h, w = img.shape
        tile = self.tile_size
        overlap = self.tile_overlap
        stride = tile - overlap
        scale = self.scale

        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride
        img_padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, ph, pw = img_padded.shape

        out_h, out_w = ph * scale, pw * scale

        # CPU buffers — keeps GPU free for model + single tile only
        output = torch.zeros((1, c, out_h, out_w), dtype=torch.float32)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

        # Blend window on CPU (small: tile_size*scale squared)
        blend_1d = torch.ones(tile * scale, dtype=torch.float32)
        ramp = overlap * scale
        if ramp > 0:
            blend_1d[:ramp] = torch.linspace(0, 1, ramp)
            blend_1d[-ramp:] = torch.linspace(1, 0, ramp)
        blend_window = (blend_1d.unsqueeze(1) * blend_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        y_starts = list(range(0, ph - tile + 1, stride))
        x_starts = list(range(0, pw - tile + 1, stride))
        total_tiles = len(y_starts) * len(x_starts)

        if not self._logged:
            logger.info(f"Tiling: {len(y_starts)}x{len(x_starts)} = {total_tiles} tiles "
                        f"(tile={tile}, overlap={overlap}, input={h}x{w} -> {h*scale}x{w*scale})")
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(device)
                logger.info(f"GPU VRAM: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
            self._logged = True

        for y in y_starts:
            for x in x_starts:
                # Slice tile on GPU, run model, immediately move to CPU
                crop = img_padded[:, :, y:y+tile, x:x+tile]
                tile_out = model_fn(crop).float().cpu()

                # Free GPU memory from this tile
                del crop
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                oy, ox = y * scale, x * scale
                th, tw = tile_out.shape[2], tile_out.shape[3]
                output[:, :, oy:oy+th, ox:ox+tw] += tile_out * blend_window
                weights[:, :, oy:oy+th, ox:ox+tw] += blend_window
                del tile_out

        output = output / weights.clamp(min=1e-8)
        final_h, final_w = h * scale, w * scale
        return output[:, :, :final_h, :final_w]

    def process_numpy(
        self, img: np.ndarray, model_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Process a [1,C,H,W] numpy array through model_fn using tiled inference."""
        _, c, h, w = img.shape
        tile = self.tile_size
        overlap = self.tile_overlap
        stride = tile - overlap
        scale = self.scale

        pad_h = (stride - (h % stride)) % stride
        pad_w = (stride - (w % stride)) % stride
        img_padded = np.pad(img, ((0,0), (0,0), (0,pad_h), (0,pad_w)), mode='reflect')
        _, _, ph, pw = img_padded.shape

        out_h, out_w = ph * scale, pw * scale
        output = np.zeros((1, c, out_h, out_w), dtype=np.float32)
        weights = np.zeros((1, 1, out_h, out_w), dtype=np.float32)

        blend_1d = np.ones(tile * scale, dtype=np.float32)
        ramp = overlap * scale
        if ramp > 0:
            blend_1d[:ramp] = np.linspace(0, 1, ramp)
            blend_1d[-ramp:] = np.linspace(1, 0, ramp)
        blend_window = (blend_1d[None, :] * blend_1d[:, None])[None, None, :, :]

        y_starts = list(range(0, ph - tile + 1, stride))
        x_starts = list(range(0, pw - tile + 1, stride))
        total_tiles = len(y_starts) * len(x_starts)

        if not self._logged:
            logger.info(f"Tiling (ONNX): {len(y_starts)}x{len(x_starts)} = {total_tiles} tiles")
            self._logged = True

        for y in y_starts:
            for x in x_starts:
                crop = img_padded[:, :, y:y+tile, x:x+tile]
                tile_out = model_fn(crop)

                oy, ox = y * scale, x * scale
                th, tw = tile_out.shape[2], tile_out.shape[3]
                output[:, :, oy:oy+th, ox:ox+tw] += tile_out * blend_window
                weights[:, :, oy:oy+th, ox:ox+tw] += blend_window

        output = output / np.clip(weights, 1e-8, None)
        return output[:, :, :h*scale, :w*scale]
