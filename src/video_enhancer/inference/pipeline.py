"""Unified Inference Pipeline."""

import os
from pathlib import Path
from typing import List, Dict

import cv2
import torch
import numpy as np
from tqdm import tqdm

from video_enhancer.config.schema import TrainingConfig, TaskType
from video_enhancer.models.registry import create_model
from video_enhancer.inference.temporal_filter import TemporalEMAFilter
from video_enhancer.inference.tile_processor import TileProcessor
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class VideoEnhancementPipeline:
    """Orchestrates the sequential application of enhancement models to video frames.
    
    To strictly adhere to the 6GB VRAM limit, models are loaded, executed across 
    all frames, and then completely unloaded before the next model is loaded.
    """
    
    def __init__(self, device: str = "cuda", tile_size: int = 256, tile_overlap: int = 24):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.temporal_filter = TemporalEMAFilter(alpha=0.8, scene_cut_threshold=0.3)
        self.tile_processor = TileProcessor(tile_size=tile_size, tile_overlap=tile_overlap, scale=4)
        
    def _load_model_for_inference(self, config: TrainingConfig) -> torch.nn.Module:
        """Loads a model and sets it to eval mode."""
        logger.info(f"Loading {config.task.value} model into VRAM...")
        model = create_model(config)
        model = model.to(self.device)
        model.eval()
        return model

    def _unload_model(self, model: torch.nn.Module) -> None:
        """Removes the model from VRAM."""
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _read_frame(self, path: Path) -> torch.Tensor:
        """Reads image and converts to RGB tensor [1, C, H, W] in [0, 1]."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)
        
    def _save_frame(self, tensor: torch.Tensor, path: Path) -> None:
        """Saves tensor back to an image file."""
        tensor = tensor.squeeze(0).cpu().clamp(0, 1)
        img = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)

    def _read_sequence(self, frames: List[Path], idx: int, seq_len: int) -> torch.Tensor:
        """Reads a sequence of frames and concatenates them along the channel dimension.
        Matches training dataset logic: duplicate boundary frames if out of bounds.
        """
        half = seq_len // 2
        indices = [max(0, min(len(frames) - 1, idx + i - half)) for i in range(seq_len)]
        
        tensors = []
        for curr_idx in indices:
            tensors.append(self._read_frame(frames[curr_idx]))
            
        # Each is [1, 3, H, W], concat along dim=1 to get [1, 3*seq_len, H, W]
        return torch.cat(tensors, dim=1)

    @torch.no_grad()
    def process_frames(
        self, 
        input_dir: Path, 
        output_dir: Path, 
        configs: List[TrainingConfig],
        apply_temporal_filter: bool = True
    ) -> None:
        """Processes all frames in the directory through a chain of models.
        
        Args:
            input_dir: Directory containing extracted frames (frame_00000001.png, ...)
            output_dir: Directory to save the final enhanced frames.
            configs: List of TrainingConfigs defining the models to apply in order.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = sorted(input_dir.glob("frame_*.png"))
        if not frame_files:
            logger.error(f"No frames found in {input_dir}")
            return
            
        # We need a working directory if there are multiple models
        working_dir = output_dir / "temp"
        working_dir.mkdir(parents=True, exist_ok=True)
        
        current_input_dir = input_dir
        
        for idx, config in enumerate(configs):
            logger.info(f"--- Starting Pass {idx+1}/{len(configs)}: {config.task.value} ---")
            
            # 1. Load the model
            model = self._load_model_for_inference(config)
            
            # Determine where to save these intermediate frames
            is_last_pass = (idx == len(configs) - 1)
            current_output_dir = output_dir if is_last_pass else working_dir / f"pass_{idx}"
            current_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 2. Process frames
            current_frames = sorted(current_input_dir.glob("frame_*.png"))
            
            # If RIFE (Interpolation), logic is different (requires pairs)
            if config.task == TaskType.INTERPOLATION:
                self._process_interpolation(model, current_frames, current_output_dir)
            else:
                def _model_fn(tile: torch.Tensor) -> torch.Tensor:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=config.device.use_amp):
                        result = model(tile)
                        if isinstance(result, tuple):
                            result = result[0]
                    return result

                seq_len = getattr(config.data, "sequence_length", 1)
                for idx, frame_path in enumerate(tqdm(current_frames, desc=f"Processing {config.task.value}")):
                    if seq_len > 1:
                        tensor = self._read_sequence(current_frames, idx, seq_len)
                    else:
                        tensor = self._read_frame(frame_path)
                    
                    out_tensor = self.tile_processor.process_tensor(tensor, _model_fn)
                    
                    if is_last_pass and apply_temporal_filter:
                        out_numpy = (out_tensor.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                        filtered_numpy = self.temporal_filter.process_frame(out_numpy)
                        out_tensor = torch.from_numpy(filtered_numpy).float().permute(2, 0, 1).unsqueeze(0) / 255.0

                    out_path = current_output_dir / frame_path.name
                    self._save_frame(out_tensor, out_path)
                    
            # 3. Unload model to free VRAM for the next pass
            self._unload_model(model)
            current_input_dir = current_output_dir
            
        logger.info("Pipeline processing complete!")

    def _process_interpolation(self, model: torch.nn.Module, frames: List[Path], out_dir: Path):
        """Special handling for RIFE frame interpolation (2x framerate)."""
        # For N frames, we generate N-1 intermediate frames, totaling 2N-1 frames.
        frame_idx = 1
        for i in tqdm(range(len(frames) - 1), desc="Interpolating frames"):
            img0 = self._read_frame(frames[i])
            img1 = self._read_frame(frames[i+1])
            
            # Original frame 1
            self._save_frame(img0, out_dir / f"frame_{frame_idx:08d}.png")
            frame_idx += 1
            
            # Intermediate predicted frame (img0.5)
            mid_img = model(img0, img1)
            self._save_frame(mid_img, out_dir / f"frame_{frame_idx:08d}.png")
            frame_idx += 1
            
        # Save the very last original frame
        last_img = self._read_frame(frames[-1])
        self._save_frame(last_img, out_dir / f"frame_{frame_idx:08d}.png")
