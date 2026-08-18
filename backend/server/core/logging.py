"""Logging configuration for VisionSuite backend."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from server.core.config import settings


def configure_logging() -> None:
    """Configure root logging for the backend."""
    log_dir = Path(settings.OUTPUT_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "visionsuite_api.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
