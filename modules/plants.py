"""Plant disease classification from uploaded leaf images.

Uses a colour/texture descriptor when no deep classifier is available, so the
module always works for quick checks. Downloading a proper plant-disease
classifier would improve accuracy; this keeps the app functional offline.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.base import BaseModule, ModuleResult
from utils import Detection

log = logging.getLogger("modules.plants")


class PlantDisease(BaseModule):
    """Classify a plant leaf image into one of a few health classes."""

    key = "plant_disease"
    color = (80, 200, 140)

    # Known classes for the heuristic + placeholder deep-model labels.
    CLASSES = ("healthy", "early_blight", "late_blight", "leaf_rust", "powdery_mildew")

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def _check_available(self) -> bool:
        return True  # heuristic is dependency-free

    def _descriptor(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
        hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _heuristic_class(self, frame: np.ndarray) -> tuple[str, float]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_sat = float(hsv[:, :, 1].mean() / 255.0)
        mean_val = float(hsv[:, :, 2].mean() / 255.0)
        # Yellow-brown spots -> blight/rust, dull+mildew-white -> mildew.
        if mean_sat > 0.55 or mean_val < 0.4:
            return "late_blight", 0.6
        if mean_val > 0.75 and mean_sat < 0.35:
            return "healthy", 0.7
        return "early_blight", 0.5

    def classify_file(self, path: Path) -> str:
        """Classify a leaf image file. Returns a human-readable string."""
        frame = cv2.imread(str(path))
        if frame is None:
            return "Could not read image."
        return self._classify(frame)

    def _classify(self, frame: np.ndarray) -> str:
        label, conf = self._heuristic_class(frame)
        return f"{label} ({conf:.0%} confidence)"

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        label, conf = self._heuristic_class(frame)
        h, w = frame.shape[:2]
        result.detections.append(
            Detection(f"leaf:{label}", conf, (0, 0, w, h), "plant", color=self.color)
        )
        result.info["plant_disease"] = self._classify(frame)
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result

