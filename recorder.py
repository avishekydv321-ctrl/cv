"""Snapshot and video recording helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils import OUTPUT_DIR, timestamp

log = logging.getLogger("recorder")


def save_snapshot(frame: np.ndarray, directory: Path = OUTPUT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"capture-{timestamp()}.png"
    cv2.imwrite(str(path), frame)
    log.info("Saved snapshot %s", path)
    return path


class VideoRecorder:
    """MP4 writer that lazily initialises on the first written frame."""

    def __init__(self, fps: float = 25.0, directory: Path = OUTPUT_DIR) -> None:
        self.fps = fps
        self.directory = directory
        self._writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, frame: np.ndarray) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self.path = self.directory / f"recording-{timestamp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        log.info("Recording to %s", self.path)
        return self.path

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self.start(frame)
        assert self._writer is not None
        self._writer.write(frame)

    def stop(self) -> Optional[Path]:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("Recording saved: %s", self.path)
        return self.path
