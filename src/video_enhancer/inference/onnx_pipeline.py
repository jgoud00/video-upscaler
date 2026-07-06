"""ONNX Runtime Inference Pipeline."""

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from video_enhancer.inference.temporal_filter import TemporalEMAFilter
from video_enhancer.inference.tile_processor import TileProcessor
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


class ONNXEnhancementPipeline:
    """Executes a chain of ONNX models via TensorRT/CUDA Execution Providers.
    
    This replaces PyTorch for inference, providing massively reduced VRAM 
    usage and much faster execution speeds.
    """
    
    def __init__(self, use_tensorrt: bool = False, tile_size: int = 192, tile_overlap: int = 16, scale: int = 4):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Please install onnxruntime-gpu.")
            
        self.providers = []
        if use_tensorrt and 'TensorrtExecutionProvider' in ort.get_available_providers():
            self.providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            self.providers.append('CUDAExecutionProvider')
        self.providers.append('CPUExecutionProvider')
        
        logger.info(f"Initialized ONNX Pipeline with providers: {self.providers}")
        self.temporal_filter = TemporalEMAFilter(alpha=0.8, scene_cut_threshold=0.3)
        self.tile_processor = TileProcessor(tile_size=tile_size, tile_overlap=tile_overlap, scale=scale)
        
    def _read_frame(self, path: Path) -> np.ndarray:
        """Reads image into [1, C, H, W] float32 numpy array in [0, 1]."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)
        
    def _save_frame(self, tensor: np.ndarray, path: Path) -> None:
        """Saves [1, C, H, W] numpy array back to an image file."""
        tensor = np.clip(tensor[0], 0, 1)
        img = (np.transpose(tensor, (1, 2, 0)) * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)

    def process_frames(
        self, 
        input_dir: Path, 
        output_dir: Path, 
        onnx_model_paths: List[Path],
        apply_temporal_filter: bool = True
    ) -> None:
        """Processes all frames in the directory through a chain of ONNX models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        working_dir = output_dir / "temp_onnx"
        working_dir.mkdir(parents=True, exist_ok=True)
        
        current_input_dir = input_dir
        
        for idx, model_path in enumerate(onnx_model_paths):
            logger.info(f"--- Starting Pass {idx+1}/{len(onnx_model_paths)}: {model_path.name} ---")
            
            # Load ONNX Session (this allocates VRAM)
            session = ort.InferenceSession(str(model_path), providers=self.providers)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            is_last_pass = (idx == len(onnx_model_paths) - 1)
            current_output_dir = output_dir if is_last_pass else working_dir / f"pass_{idx}"
            current_output_dir.mkdir(parents=True, exist_ok=True)
            
            current_frames = sorted(current_input_dir.glob("frame_*.png"))
            
            # Check if this is RIFE (which needs 2 inputs)
            is_rife = len(session.get_inputs()) == 2
            
            if is_rife:
                input_name_2 = session.get_inputs()[1].name
                frame_idx = 1
                for i in tqdm(range(len(current_frames) - 1), desc=f"ONNX Interpolation ({model_path.name})"):
                    img0 = self._read_frame(current_frames[i])
                    img1 = self._read_frame(current_frames[i+1])
                    
                    self._save_frame(img0, current_output_dir / f"frame_{frame_idx:08d}.png")
                    frame_idx += 1
                    
                    mid_img = session.run([output_name], {input_name: img0, input_name_2: img1})[0]
                    self._save_frame(mid_img, current_output_dir / f"frame_{frame_idx:08d}.png")
                    frame_idx += 1
                    
                last_img = self._read_frame(current_frames[-1])
                self._save_frame(last_img, current_output_dir / f"frame_{frame_idx:08d}.png")
            else:
                def _onnx_fn(tile: np.ndarray) -> np.ndarray:
                    return session.run([output_name], {input_name: tile})[0]

                for frame_path in tqdm(current_frames, desc=f"ONNX Processing ({model_path.name})"):
                    tensor = self._read_frame(frame_path)
                    out_tensor = self.tile_processor.process_numpy(tensor, _onnx_fn)
                    
                    if is_last_pass and apply_temporal_filter:
                        out_numpy = (np.clip(out_tensor[0], 0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
                        filtered_numpy = self.temporal_filter.process_frame(out_numpy)
                        out_tensor = np.expand_dims(filtered_numpy.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)

                    out_path = current_output_dir / frame_path.name
                    self._save_frame(out_tensor, out_path)
                    
            # Explicitly delete session to free VRAM for next ONNX model
            del session
            current_input_dir = current_output_dir
            
        logger.info("ONNX Pipeline processing complete!")
