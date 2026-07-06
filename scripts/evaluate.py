"""Evaluate enhanced images/frames with the full metrics suite.

Usage:
  # Full evaluation with ground truth:
  python scripts/evaluate.py --input-dir data/test/enhanced --gt-dir data/test/gt

  # No-reference only (real-world footage, no GT):
  python scripts/evaluate.py --input-dir data/test/enhanced

  # With performance measurement:
  python scripts/evaluate.py --input-dir data/test/enhanced --gt-dir data/test/gt --measure-perf
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from video_enhancer.evaluation.metrics import MetricsCalculator, MetricsResult


def format_results_table(result: MetricsResult, console: Console) -> None:
    """Print a formatted metrics report."""

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Video Enhancer — Evaluation Report[/bold cyan]",
        border_style="cyan"
    ))

    # PRIMARY
    table = Table(show_header=True, header_style="bold green", title="PRIMARY (Perceptual)", title_style="bold green")
    table.add_column("Metric", style="white", width=30)
    table.add_column("Value", style="bold yellow", justify="right", width=20)
    table.add_column("Direction", style="dim", width=15)
    if result.lpips is not None:
        table.add_row("LPIPS", f"{result.lpips:.4f}", "lower = better")
    else:
        table.add_row("LPIPS", "[dim]N/A (no GT)[/dim]", "—")
    console.print(table)
    console.print()

    # NO-REFERENCE
    table = Table(show_header=True, header_style="bold blue", title="NO-REFERENCE (No GT needed)", title_style="bold blue")
    table.add_column("Metric", style="white", width=30)
    table.add_column("Value", style="bold yellow", justify="right", width=20)
    table.add_column("Direction", style="dim", width=15)
    if result.niqe is not None:
        table.add_row("NIQE", f"{result.niqe:.4f}", "lower = better")
    else:
        table.add_row("NIQE", "[dim]pyiqa not installed[/dim]", "—")
    if result.brisque is not None:
        table.add_row("BRISQUE", f"{result.brisque:.4f}", "lower = better")
    else:
        table.add_row("BRISQUE", "[dim]pyiqa not installed[/dim]", "—")
    console.print(table)
    console.print()

    # SECONDARY
    table = Table(show_header=True, header_style="bold magenta", title="SECONDARY (Pixel-aligned — for reference only)", title_style="bold magenta")
    table.add_column("Metric", style="white", width=30)
    table.add_column("Value", style="bold yellow", justify="right", width=20)
    table.add_column("Direction", style="dim", width=15)
    if result.psnr is not None:
        table.add_row("PSNR", f"{result.psnr:.2f} dB", "higher = better")
    else:
        table.add_row("PSNR", "[dim]N/A (no GT)[/dim]", "—")
    if result.ssim is not None:
        table.add_row("SSIM", f"{result.ssim:.4f}", "higher = better")
    else:
        table.add_row("SSIM", "[dim]N/A (no GT)[/dim]", "—")
    console.print(table)
    console.print()

    # TEMPORAL
    table = Table(show_header=True, header_style="bold red", title="TEMPORAL (Frame consistency)", title_style="bold red")
    table.add_column("Metric", style="white", width=30)
    table.add_column("Value", style="bold yellow", justify="right", width=20)
    table.add_column("Direction", style="dim", width=15)
    if result.t_loss is not None:
        table.add_row("T-loss", f"{result.t_loss:.4f}", "generally better")
    else:
        table.add_row("T-loss", "[dim]< 2 frames[/dim]", "—")
    console.print(table)
    console.print()

    # PERFORMANCE
    table = Table(show_header=True, header_style="bold white", title="PERFORMANCE", title_style="bold white")
    table.add_column("Metric", style="white", width=30)
    table.add_column("Value", style="bold yellow", justify="right", width=20)
    table.add_column("Notes", style="dim", width=15)
    if result.fps is not None:
        table.add_row("Inference FPS", f"{result.fps:.1f}", "256×256 input")
    else:
        table.add_row("Inference FPS", "[dim]not measured[/dim]", "—")
    if result.peak_vram_mb is not None:
        table.add_row("Peak VRAM", f"{result.peak_vram_mb:.0f} MB", "")
    else:
        table.add_row("Peak VRAM", "[dim]not measured[/dim]", "—")
    console.print(table)
    console.print()

    # Summary
    console.print(f"[dim]Images evaluated: {result.num_images} | Ground truth: {'Yes' if result.has_ground_truth else 'No'}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced images with comprehensive metrics.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of enhanced/predicted images")
    parser.add_argument("--gt-dir", type=str, default=None, help="Directory of ground-truth images (optional)")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--measure-perf", action="store_true", help="Measure inference FPS and VRAM (loads Real-ESRGAN)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for metric computation")
    args = parser.parse_args()

    console = Console()

    console.print("[bold]Initializing metrics calculator...[/bold]")
    calc = MetricsCalculator(device=args.device)

    console.print(f"[bold]Evaluating images in:[/bold] {args.input_dir}")
    if args.gt_dir:
        console.print(f"[bold]Ground truth from:[/bold] {args.gt_dir}")

    result = calc.evaluate_directory(
        input_dir=Path(args.input_dir),
        gt_dir=Path(args.gt_dir) if args.gt_dir else None,
    )

    # Optional performance measurement
    if args.measure_perf and torch.cuda.is_available():
        console.print("[bold]Measuring inference performance (Real-ESRGAN)...[/bold]")
        try:
            from video_enhancer.models.pretrained.realesrgan import build_realesrgan
            from video_enhancer.config.schema import FreezeConfig

            ckpt = Path("checkpoints/pretrained/RealESRGAN_x4plus.pth")
            if ckpt.exists():
                model = build_realesrgan(checkpoint_path=str(ckpt), freeze_config=FreezeConfig(freeze_strategy="none"))
                model = model.to(args.device).eval()
                with torch.no_grad():
                    fps, vram = calc.measure_inference_performance(model, input_size=(1, 3, 256, 256))
                result.fps = fps
                result.peak_vram_mb = vram
                del model
                torch.cuda.empty_cache()
            else:
                console.print(f"[yellow]Checkpoint not found at {ckpt}, skipping perf measurement[/yellow]")
        except Exception as e:
            console.print(f"[red]Performance measurement failed: {e}[/red]")

    format_results_table(result, console)

    # Save JSON
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"[green]Results saved to {out_path}[/green]")


if __name__ == "__main__":
    main()
