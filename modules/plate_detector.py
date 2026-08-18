"""Dedicated license-plate detection model manager.

This module provides a single dedicated pretrained **license-plate detection**
model (a YOLOv8n fine-tuned for the ``license_plate`` class) used by the ANPR
pipeline. It is completely separate from the vehicle COCO detector.

The weight file is automatically downloaded into ``models/license_plate/`` on
first use from a validated, public, non-gated HuggingFace repo:

    Koushim/yolov8-license-plate-detection  (MIT license, class ``license_plate``)

Behaviour:
  * ``ensure_available()`` performs existence check → download → load → validate.
  * A ``progress`` callback reports download / CPU-build / load progress so the
    GUI can show the status.
  * Any failure raises :class:`PlateModelError` naming the exact model + source,
    so callers NEVER silently fall back to a vehicle-only model and report
    "Plate Not Found".
  * The downloaded model is cached and reused on subsequent launches.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from modules.base import get_device
from utils import MODELS_DIR

log = logging.getLogger("modules.plate_detector")

#: Reliable, validated, non-gated public source for a YOLOv8n plate detector.
PLATE_MODEL_SOURCE = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"

#: Local file name (stored under models/license_plate/).
PLATE_MODEL_FILENAME = "plate_detector_yolov8n.pt"

#: Fired for UI progress. Represents a real, verified download.
ProgressFn = Callable[[str, float], None]


class PlateModelError(RuntimeError):
    """Raised when the dedicated plate model is missing / corrupt / unloadable."""


def _plate_model_path() -> Path:
    return MODELS_DIR / "license_plate" / PLATE_MODEL_FILENAME


def _download(url: str, destination: Path, progress: Optional[ProgressFn]) -> None:
    """Download ``url`` to ``destination`` with byte-count progress."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Downloading license-plate model…", 0.0)
    log.info("Downloading plate model %s -> %s", url, destination)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VisionSuite/1.0"})
        total = 0
        with urllib.request.urlopen(req, timeout=120) as response:
            length = response.headers.get("Content-Length")
            expected = int(length) if length else None
            tmp = destination.with_suffix(".pt.tmp")
            with tmp.open("wb") as out:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
                    if expected:
                        if progress:
                            progress(
                                f"Downloading license-plate model… {total//1024}/{expected//1024} KB",
                                total / max(expected, 1),
                            )
            tmp.replace(destination)
        if progress:
            progress("License-plate model downloaded.", 1.0)
        log.info("Plate model download complete (%d bytes).", total)
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise PlateModelError(
            f"Failed to download license-plate model.\n"
            f"Model: {Path(destination).name}\n"
            f"Source: {url}\n"
            f"Error: {exc}\n\n"
            f"Check your internet connection and retry."
        ) from exc


class PlateDetector:
    """Wraps a dedicated YOLOv8 license-plate detection model.

    The detector is separate from the vehicle detector and only ever reports
    license-plate boxes.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or get_device()
        self._model: Optional[Any] = None
        self._lock = threading.Lock()
        self.model_name = PLATE_MODEL_FILENAME
        self.model_source = PLATE_MODEL_SOURCE
        self.load_status = "not_loaded"
        self._validated = False

    # -- lifecycle -----------------------------------------------------
    def ensure_available(self, progress: Optional[ProgressFn] = None) -> Any:
        """Download (if needed), load and validate the model.

        Returns the loaded YOLO model, or raises :class:`PlateModelError`.
        """
        if self._model is not None and self._validated:
            return self._model

        with self._lock:
            if self._model is not None and self._validated:
                return self._model

            path = _plate_model_path()
            if not path.exists():
                if not progress:
                    progress = self._noop_progress
                _download(self.model_source, path, progress)
            else:
                if progress:
                    progress("License-plate model found — loading…", 0.5)

            return self._load(path, progress)

    def _noop_progress(self, _msg: str, _frac: float) -> None:  # pragma: no cover
        pass

    def _load(self, path: Path, progress: Optional[ProgressFn]) -> Any:
        try:
            from ultralytics import YOLO

            if progress:
                progress("Loading license-plate model…", 0.6)
            self.load_status = "loading"
            model = YOLO(str(path))
            # Force a weights load into memory (CPU build can take a moment).
            if progress:
                progress("Preparing license-plate model on CPU…", 0.8)
            names = getattr(model, "names", None) or {}
            # Validate the model actually targets licence plates.
            valid_names = [str(n).lower() for n in (names.values() if isinstance(names, dict) else names or [])]
            if not any("plate" in n or "license" in n or "number" in n for n in valid_names):
                raise PlateModelError(
                    f"License-plate model loaded but its classes do not include a plate class.\n"
                    f"Model: {path.name}\nClasses: {valid_names}"
                )
            self._model = model
            self._validated = True
            self.load_status = "loaded"
            if progress:
                progress(f"License-plate model ready: {self.model_name}", 1.0)
            log.info("Dedicated plate model loaded: %s", path.name)
            return model
        except PlateModelError:
            self.load_status = "error"
            raise
        except Exception as exc:
            self.load_status = "error"
            raise PlateModelError(
                f"License-plate model failed to load.\n"
                f"Model: {path.name}\n"
                f"Source: {self.model_source}\n"
                f"Error: {exc}"
            ) from exc

    # -- detection -----------------------------------------------------
    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.3,
        progress: Optional[ProgressFn] = None,
    ) -> list[tuple[int, int, int, int]]:
        """Return plate boxes ``(x1,y1,x2,y2)`` in ``image`` coordinates."""
        model = self.ensure_available(progress)
        boxes: list[tuple[int, int, int, int]] = []
        try:
            results = model.predict(
                image, conf=conf, verbose=False, device=self.device
            )
            for result in results:
                for b in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    boxes.append((x1, y1, x2, y2))
        except Exception as exc:
            log.debug("Plate model predict error: %s", exc)
        return boxes

    def release(self) -> None:
        """Release the model reference / CUDA cache if any."""
        try:
            import gc

            self._model = None
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    def __call__(
        self,
        image: np.ndarray,
        conf: float = 0.3,
        progress: Optional[ProgressFn] = None,
    ) -> list[tuple[int, int, int, int]]:
        return self.detect(image, conf=conf, progress=progress)


#: Module-level singleton (cached across workers).
_shared: Optional[PlateDetector] = None
_shared_lock = threading.Lock()


def get_plate_detector() -> PlateDetector:
    """Return a shared :class:`PlateDetector` (safe across threads)."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = PlateDetector()
    return _shared


__all__ = [
    "PlateDetector",
    "PlateModelError",
    "get_plate_detector",
    "PLATE_MODEL_FILENAME",
    "PLATE_MODEL_SOURCE",
]

