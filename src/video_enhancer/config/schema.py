"""Typed configuration schema using Pydantic v2.

📘 Concept — Why Pydantic over dataclasses?
Both can define typed config schemas. Pydantic adds:
  - Automatic validation (wrong types, missing fields → clear error messages)
  - YAML/JSON/dict deserialization with coercion (e.g., string "128" → int 128)
  - Nested model composition with defaults
  - `model_dump()` for clean serialization back to dict/YAML
For a project with many config knobs (4 models × training/inference), catching
config typos early is worth the dependency.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported fine-tuning tasks."""
    SUPER_RESOLUTION = "super_resolution"
    DEBLUR = "deblur"
    FACE_ENHANCE = "face_enhance"
    FRAME_INTERPOLATION = "frame_interpolation"
    FACE_ENHANCEMENT = "face_enhancement"
    INTERPOLATION = "interpolation"


class DeviceConfig(BaseModel):
    """Hardware settings."""
    gpu_id: int = 0
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"


class DataConfig(BaseModel):
    """Dataset and augmentation settings."""
    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")
    patch_size: int = Field(128, ge=32, le=512, description="LR patch crop size")
    batch_size: int = Field(1, ge=1, le=32)
    num_workers: int = Field(2, ge=0)
    sequence_length: int = Field(1, ge=1, description="Number of frames to load as sequence")
    augment_flip: bool = True
    augment_rotate: bool = True


class DegradationConfig(BaseModel):
    """Synthetic degradation pipeline settings."""
    downscale_range: tuple[float, float] = (2.0, 4.0)
    noise_sigma_range: tuple[float, float] = (0.0, 25.0)
    jpeg_quality_range: tuple[int, int] = (30, 95)
    blur_kernel_size: int = 21
    blur_sigma_range: tuple[float, float] = (0.1, 3.0)
    enable_second_order: bool = True


class FreezeConfig(BaseModel):
    """Layer freezing strategy for VRAM reduction.

    📘 Concept — Partial Freezing / Catastrophic Forgetting
    When fine-tuning a pretrained model, updating all layers risks "catastrophic
    forgetting" — the model loses its learned general features while adapting to
    your specific (small) dataset. Freezing early layers preserves low-level
    feature extractors (edges, textures) that generalize well, while allowing
    later layers to specialize. This also drastically reduces trainable params,
    optimizer state memory, and gradient memory.
    """
    freeze_strategy: str = "partial"  # "none", "partial", "all_but_head"
    num_frozen_blocks: int = Field(11, ge=0, description="Number of early blocks to freeze")
    freeze_discriminator_early: bool = True


class LoRAConfig(BaseModel):
    """Low-rank adapter settings.

    📘 Concept — LoRA (Low-Rank Adaptation)
    Instead of updating a full weight matrix W (shape d_in × d_out), LoRA adds
    two small matrices A (d_in × r) and B (r × d_out) where r << min(d_in, d_out).
    The effective update is W + A·B, but only A and B are trained — reducing
    trainable params by orders of magnitude. Originally designed for transformers,
    it can be applied to conv layers by reshaping the kernel.
    """
    enabled: bool = False
    rank: int = Field(8, ge=1, le=64)
    alpha: float = Field(16.0, gt=0)
    target_modules: list[str] = Field(default_factory=lambda: ["conv_body", "conv_up"])


class OptimizerConfig(BaseModel):
    """Optimizer and scheduler settings."""
    name: str = "adamw"
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    gradient_accumulation_steps: int = Field(4, ge=1)
    max_grad_norm: float = Field(1.0, gt=0)


class SchedulerConfig(BaseModel):
    """Learning rate scheduler."""
    name: str = "cosine"  # "cosine", "step", "constant"
    warmup_steps: int = 500
    total_steps: int = 50_000
    min_lr: float = 1e-7


class CheckpointConfig(BaseModel):
    """Checkpoint save/load settings."""
    save_dir: Path = Path("checkpoints/finetuned")
    pretrained_path: Optional[Path] = None
    resume_path: Optional[Path] = None
    save_every_n_steps: int = 1000
    keep_last_n: int = 3


class LossConfig(BaseModel):
    """Loss function weights."""
    pixel_weight: float = 1.0
    perceptual_weight: float = 1.0
    gan_weight: float = 0.1
    temporal_weight: float = 0.0  # enabled in Phase 8


class TrainingConfig(BaseModel):
    """Top-level training configuration."""
    task: TaskType
    experiment_name: str = "default"
    seed: int = 42

    device: DeviceConfig = DeviceConfig()
    data: DataConfig = DataConfig()
    degradation: DegradationConfig = DegradationConfig()
    freeze: FreezeConfig = FreezeConfig()
    lora: LoRAConfig = LoRAConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    loss: LossConfig = LossConfig()

    gradient_checkpointing: bool = False
    log_dir: Path = Path("logs")
    log_every_n_steps: int = 50
    val_every_n_steps: int = 500
