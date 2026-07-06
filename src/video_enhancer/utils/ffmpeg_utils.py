import subprocess
import shutil
import os
import cv2
from pathlib import Path
from video_enhancer.utils.logging import get_logger

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

logger = get_logger(__name__)


def get_ffmpeg_path() -> str:
    """Gets the absolute path to the FFmpeg executable."""
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
        
    raise RuntimeError("FFmpeg is not installed or not on PATH.")


def extract_audio(video_path: Path, output_audio_path: Path) -> bool:
    """Extracts audio from a video file without re-encoding."""
    ffmpeg_exe = get_ffmpeg_path()
        
    logger.info(f"Extracting audio from {video_path} to {output_audio_path}")
    
    cmd = [
        ffmpeg_exe, "-y", "-i", str(video_path), 
        "-q:a", "0", "-map", "a", str(output_audio_path)
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        logger.warning(f"Failed to extract audio (maybe video has no audio?). FFmpeg said: {result.stderr.decode()}")
        return False
        
    return True


def extract_frames(video_path: Path, output_dir: Path, fps: int = 0) -> None:
    """Extracts all frames from a video into a directory.
    
    If fps is > 0, it extracts at the specified fps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting frames to {output_dir}")
    
    # %08d.png gives us frame_00000001.png
    output_pattern = str(output_dir / "frame_%08d.png")
    
    ffmpeg_exe = get_ffmpeg_path()
    cmd = [ffmpeg_exe, "-y", "-i", str(video_path)]
    if fps > 0:
        cmd.extend(["-r", str(fps)])
    
    cmd.extend(["-qscale:v", "1", "-qmin", "1", "-qmax", "1", output_pattern])
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to extract frames: {result.stderr.decode()}")


def get_video_fps(video_path: Path) -> float:
    """Uses OpenCV to get the video FPS."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            return 30.0
        return float(fps)
    except Exception as e:
        logger.warning(f"Could not determine FPS via cv2 ({e}), defaulting to 30.0")
        return 30.0


def merge_frames_to_video(
    frames_dir: Path, 
    audio_path: Path | None, 
    output_video: Path, 
    fps: float
) -> None:
    """Merges frames back into an mp4 video, multiplexing audio if available."""
    logger.info(f"Merging frames from {frames_dir} into {output_video} at {fps} fps.")
    
    frames_pattern = str(frames_dir / "frame_%08d.png")
    
    ffmpeg_exe = get_ffmpeg_path()
    # High-quality h264 encoding
    cmd = [
        ffmpeg_exe, "-y", 
        "-framerate", str(fps), 
        "-i", frames_pattern
    ]
    
    if audio_path and audio_path.exists():
        cmd.extend(["-i", str(audio_path)])
        
    cmd.extend([
        "-c:v", "libx264", 
        "-crf", "18",           # Very high quality
        "-pix_fmt", "yuv420p"
    ])
    
    if audio_path and audio_path.exists():
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
        
    cmd.append(str(output_video))
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to merge video: {result.stderr.decode()}")
