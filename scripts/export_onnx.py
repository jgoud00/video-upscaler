"""Script to export trained PyTorch models to ONNX with Dynamic Axes."""

import argparse
from pathlib import Path

import torch
from video_enhancer.config.loader import load_config
from video_enhancer.models.registry import create_model
from video_enhancer.config.schema import TaskType
from video_enhancer.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("onnx_export")


def export_to_onnx(config_path: Path, output_path: Path):
    """Exports a model to ONNX format with dynamic spatial axes."""
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    logger.info(f"Initializing {config.task.value} model...")
    device = torch.device("cpu") # Exporting is usually safest on CPU
    model = create_model(config).to(device)
    model.eval()

    # Determine input shape and dynamic axes based on task
    if config.task == TaskType.INTERPOLATION:
        # RIFE takes two concatenated frames (6 channels)
        # We need a temporal dimension or concatenated channel dimension
        dummy_input = (torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
        input_names = ["frame0", "frame1"]
        dynamic_axes = {
            "frame0": {0: "batch_size", 2: "height", 3: "width"},
            "frame1": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    else:
        seq_len = getattr(config.data, "sequence_length", 1)
        num_in_ch = 3 * seq_len
        dummy_input = torch.randn(1, num_in_ch, 256, 256)
        input_names = ["input"]
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting model to {output_path} with dynamic axes...")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        logger.info(f"Successfully exported ONNX model to {output_path}")
    except Exception as e:
        logger.exception("Failed to export ONNX model.")


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output .onnx path")
    args = parser.parse_args()

    export_to_onnx(Path(args.config), Path(args.output))

if __name__ == "__main__":
    main()
