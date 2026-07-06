import argparse
import os
from pathlib import Path

from video_enhancer.utils.ffmpeg_utils import extract_frames
from video_enhancer.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("prepare_data")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video for training.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1.0 for diverse training data)")
    parser.add_argument("--out_dir", type=str, default="data/train/own_footage", help="Output directory for frames")
    
    args = parser.parse_args()
    
    input_video = Path(args.input)
    output_dir = Path(args.out_dir)
    
    if not input_video.exists():
        logger.error(f"Input video not found: {input_video}")
        return
        
    logger.info(f"Extracting frames from {input_video} at {args.fps} FPS to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        extract_frames(input_video, output_dir, fps=args.fps)
        logger.info(f"Extraction complete! Frames saved to {output_dir}")
        num_frames = len(list(output_dir.glob("*.png")))
        logger.info(f"Total frames extracted: {num_frames}")
    except Exception as e:
        logger.exception("Extraction failed:")

if __name__ == "__main__":
    main()
