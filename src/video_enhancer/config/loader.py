"""Load and merge YAML config files into typed Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from video_enhancer.config.schema import TrainingConfig
from video_enhancer.utils.logging import get_logger

logger = get_logger(__name__)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a single YAML file into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(config_path: Path, base_path: Path | None = None) -> TrainingConfig:
    """Load a training config, optionally merging with a base config.

    If the config YAML contains a `_base_` key, that file is loaded first
    and the current config is merged on top (current wins).

    Args:
        config_path: Path to the task-specific YAML config.
        base_path: Optional explicit base config. Overrides `_base_` key.

    Returns:
        Validated TrainingConfig instance.
    """
    data = load_yaml(config_path)

    # Support `_base_: path/to/base.yaml` in the config file
    implicit_base = data.pop("_base_", None)
    if base_path is None and implicit_base is not None:
        base_path = config_path.parent / implicit_base

    if base_path is not None:
        base_data = load_yaml(base_path)
        data = _deep_merge(base_data, data)

    config = TrainingConfig(**data)
    logger.info(f"Loaded config: task={config.task.value}, experiment={config.experiment_name}")
    return config
