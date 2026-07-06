"""Structured logging with Rich console output and file rotation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_LOG_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_console = Console(stderr=True)
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_filename: str = "train.log",
) -> None:
    """Configure root logger with Rich console handler and optional file handler.

    Call once at application startup. Subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(level)

    # Rich console handler — human-friendly, colored output
    console_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # File handler — machine-parseable, rotated
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "urllib3", "basicsr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Use module __name__ as convention."""
    return logging.getLogger(name)
