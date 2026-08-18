"""Shared infrastructure for the modular AI detectors.

Every module follows the same contract:
  * constructed once and cached by the worker
  * exposes ``available`` (bool) — whether its model/dependency is usable
  * exposes ``process(frame, config) -> ModuleResult`` for live camera frames
  * optional file-based entry points (``classify_file``, ``scan_file`` ...)

Modules must never raise during ``process`` — the worker catches everything,
logs it and keeps the UI alive.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from utils import Detection, MODELS_DIR, ensure_model as _ensure_model

log = logging.getLogger("modules.base")


# --------------------------------------------------------------------------
# Device selection: prefer CUDA when available, else MPS/CPU — never crash.
# --------------------------------------------------------------------------
def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is usable, otherwise 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch not installed / broken
        pass
    return "cpu"


DEVICE = get_device()
log.info("AI device: %s", DEVICE)


def load_onnx(name: str) -> Any:
    """Load an ONNX Runtime session; None if onnxruntime is unavailable."""
    try:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        session = ort.InferenceSession(str(MODELS_DIR / name), providers=providers)
        return session
    except Exception as exc:  # pragma: no cover
        log.warning("ONNX session %s unavailable: %s", name, exc)
        return None


def ensure_model(url: str, filename: str) -> Optional[Path]:
    """Download ``filename`` into models/ if missing. Returns path or None.

    Thin wrapper over the consolidated ``utils.ensure_model`` so modules keep
    their familiar ``ensure_model(url, filename)`` signature.
    """
    return _ensure_model(filename, url)


# --------------------------------------------------------------------------
# Result container shared by every module.
# --------------------------------------------------------------------------
@dataclass
class ModuleResult:
    """Standard return value for module.process()."""

    detections: list[Detection] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    inference_ms: float = 0.0


class BaseModule:
    """Optional-dependency-safe base class for all AI modules."""

    #: identifier used in config.AI_MODULES / the GUI toggle
    key: str = "base"
    #: per-class colour used for drawing (BGR)
    color: tuple[int, int, int] = (0, 240, 180)

    def __init__(self) -> None:
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """Whether the module can run right now (checks deps lazily)."""
        if self._available is None:
            try:
                self._available = self._check_available()
            except Exception:  # pragma: no cover
                self._available = False
        return self._available

    def _check_available(self) -> bool:
        """Override: return True when dependencies/models are present."""
        return True

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        """Override: run the module on a live frame and return a ModuleResult."""
        return ModuleResult()

    def _run(self, frame: np.ndarray, config: Any) -> ModuleResult:
        """Handled by the worker: skip unavailable modules, run `process`."""
        if not self.available:
            result = ModuleResult()
            result.info["available"] = False
            return result
        return self.process(frame, config)

    def close(self) -> None:
        """Override: release heavyweight resources (models, sessions...)."""
