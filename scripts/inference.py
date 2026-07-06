"""CLI Entrypoint for the Unified Inference Pipeline."""

import argparse
import shutil
from pathlib import Path

from video_enhancer.config.loader import load_config
from video_enhancer.inference.pipeline import VideoEnhancementPipeline
from video_enhancer.utils.ffmpeg_utils import extract_audio, extract_frames, get_video_fps, merge_frames_to_video
from video_enhancer.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("inference")


def main():
    parser = argparse.ArgumentParser(description="Enhance a video using chained models.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output video path")
    parser.add_argument("--configs", type=str, nargs="+", required=True, help="List of YAML configs to run in order")
    parser.add_argument("--no-temporal", action="store_true", help="Disable the temporal EMA filter")
    parser.add_argument("--fps", type=float, default=0.0, help="Override output FPS")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size for inference (smaller = less VRAM)")
    parser.add_argument("--tile-overlap", type=int, default=24, help="Overlap between tiles for seamless blending")
    
    args = parser.parse_args()
    
    input_video = Path(args.input)
    output_video = Path(args.output)
    
    if not input_video.exists():
        logger.error(f"Input video {input_video} does not exist.")
        return
        
    # Load all configs
    configs = [load_config(Path(c)) for c in args.configs]
    
    # Create temporary directories
    temp_dir = output_video.parent / f"{output_video.stem}_temp"
    frames_in_dir = temp_dir / "frames_in"
    frames_out_dir = temp_dir / "frames_out"
    audio_path = temp_dir / "audio.aac"
    
    try:
        # 1. FFmpeg Extraction
        logger.info("=== Phase 1: Extraction ===")
        has_audio = extract_audio(input_video, audio_path)
        fps = args.fps if args.fps > 0 else get_video_fps(input_video)
        
        extract_frames(input_video, frames_in_dir)
        
        # 2. Model Pipeline
        logger.info("=== Phase 2: AI Enhancement ===")
        pipeline = VideoEnhancementPipeline(
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap
        )
        pipeline.process_frames(
            input_dir=frames_in_dir,
            output_dir=frames_out_dir,
            configs=configs,
            apply_temporal_filter=not args.no_temporal
        )
        
        # 3. FFmpeg Merging
        logger.info("=== Phase 3: Merging ===")
        # If RIFE was used, the FPS effectively doubles
        has_rife = any(c.task.value == "interpolation" for c in configs)
        final_fps = fps * 2 if has_rife else fps
        
        merge_frames_to_video(
            frames_dir=frames_out_dir,
            audio_path=audio_path if has_audio else None,
            output_video=output_video,
            fps=final_fps
        )
        
        logger.info(f"Video enhancement complete! Saved to {output_video}")
        
    except Exception as e:
        logger.exception("Inference failed:")
    finally:
        # Cleanup
        if temp_dir.exists():
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
