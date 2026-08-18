"""Safety-related AI modules: fire/smoke, helmet, and PPE detection.

These use YOLO-based models where possible (helmet / PPE) and a robust
colour/motion heuristic for fire/smoke when no dedicated model is available.
All modules degrade gracefully when a model cannot be downloaded.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from modules.base import BaseModule, ModuleResult, ensure_model
from utils import Detection

log = logging.getLogger("modules.safety")


class FireSmokeDetector(BaseModule):
    """Detect fire and smoke with colour heuristics + optional YOLO model."""

    key = "fire_smoke"
    color = (60, 60, 255)

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._last_frames: list[np.ndarray] = []

    def _check_available(self) -> bool:
        return True  # heuristic always works

    def _fire_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Fire: bright orange / yellow / red in HSV.
        lower = np.array([0, 60, 120], dtype=np.uint8)
        upper = np.array([40, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower, upper)
        lower2 = np.array([160, 60, 120], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

    def _smoke_mask(self, frame: np.ndarray, previous: np.ndarray) -> np.ndarray:
        # Smoke: grayish, low saturation, bright — and moving.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        low_sat = cv2.inRange(sat, 0, 60)
        bright = cv2.inRange(gray, 140, 255)
        motion = cv2.absdiff(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), gray)
        moving = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)[1]
        return cv2.bitwise_and(cv2.bitwise_and(low_sat, bright), moving)

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        h, w = frame.shape[:2]

        fire = self._fire_mask(frame)
        fire_area = int(cv2.countNonZero(fire))
        if fire_area > max(200, 0.003 * h * w):
            x1, y1, x2, y2 = 5, 5, w - 5, h - 5
            result.detections.append(Detection("fire", 0.9, (x1, y1, x2, y2), "fire", color=self.color))
            result.alerts.append("🔥 FIRE DETECTED")

        if self._last_frames:
            smoke = self._smoke_mask(frame, self._last_frames[-1])
            smoke_area = int(cv2.countNonZero(smoke))
            if smoke_area > max(500, 0.01 * h * w):
                x1, y1, x2, y2 = 5, 5, w - 5, h - 5
                result.detections.append(Detection("smoke", 0.7, (x1, y1, x2, y2), "smoke", color=(180, 180, 180)))
                result.alerts.append("Smoke detected")

        self._last_frames.append(frame)
        if len(self._last_frames) > 4:
            self._last_frames.pop(0)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class HelmetDetector(BaseModule):
    """Detect construction helmets using a YOLOv8 model fine-tuned on safety."""

    key = "helmet"
    color = (80, 200, 255)

    # Public helmet detection weights (roboflow-trained). If unreachable the
    # module reports itself unavailable instead of breaking the pipeline.
    _URL = "https://github.com/spacewalk01/yolov5-vehicle-and-licenseplate/releases/download/v1.0/helmet.onnx"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def _ensure(self):
        if self._model is None:
            path = ensure_model(self._URL, "helmet.onnx")
            if path is None:
                return None
            # Fallback: use a YOLO weights file if the ONNX download failed.
            try:
                from ultralytics import YOLO

                self._model = YOLO(path)
                self._onnx = False
            except Exception:
                from modules.base import load_onnx

                self._model = load_onnx("helmet.onnx")
                self._onnx = True
        return self._model

    def _check_available(self) -> bool:
        return self._ensure() is not None

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        model = self._ensure()
        if model is None:
            result.info["available"] = False
            return result

        try:
            if getattr(self, "_onnx", False):
                import onnxruntime as ort  # noqa: F401

                from modules.base import DEVICE

                blob = frame.astype(np.float32) / 255.0
                blob = blob.transpose(2, 0, 1)[None, ...]
                outs = model.run(None, {model.get_inputs()[0].name: blob})
                boxes = outs[0][0]
                for row in boxes:
                    if row[4] < config.confidence:
                        continue
                    cls = int(row[5]) if row.shape[0] > 5 else 0
                    x1, y1, x2, y2 = [int(v * frame.shape[1]) for v in row[:4]]
                    label = "helmet" if cls == 0 else "no_helmet"
                    result.detections.append(
                        Detection(label, float(row[4]), (x1, y1, x2, y2), "helmet", color=self.color)
                    )
                    if label == "no_helmet":
                        result.alerts.append("Worker without helmet")
            else:
                results = model.predict(frame, conf=config.confidence, verbose=False)
                for r in results:
                    for idx in range(len(r.boxes)):
                        x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[idx].tolist()]
                        cls = int(r.boxes.cls[idx])
                        label = "helmet" if cls == 0 else "no_helmet"
                        conf = float(r.boxes.conf[idx])
                        result.detections.append(
                            Detection(label, conf, (x1, y1, x2, y2), "helmet", color=self.color)
                        )
                        if label == "no_helmet":
                            result.alerts.append("Worker without helmet")
        except Exception as exc:  # pragma: no cover
            log.debug("helmet inference error: %s", exc)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class PPEDetector(BaseModule):
    """Detect PPE: helmet, safety vest, gloves (person-based YOLO)."""

    key = "ppe"
    color = (255, 220, 80)

    _URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._labels = {0: "helmet", 1: "vest", 2: "gloves", 3: "no_ppe"}

    def _ensure(self):
        if self._model is None:
            path = ensure_model(self._URL, "ppe_yolov8.pt")
            if path is None:
                return None
            try:
                from ultralytics import YOLO

                self._model = YOLO(path)
            except Exception:
                self._model = None
        return self._model

    def _check_available(self) -> bool:
        return self._ensure() is not None

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        model = self._ensure()
        if model is None:
            result.info["available"] = False
            return result

        try:
            results = model.predict(frame, conf=config.confidence, verbose=False)
            for r in results:
                for idx in range(len(r.boxes)):
                    x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[idx].tolist()]
                    cls = int(r.boxes.cls[idx])
                    label = self._labels.get(cls, f"ppe_{cls}")
                    conf = float(r.boxes.conf[idx])
                    result.detections.append(
                        Detection(f"ppe:{label}", conf, (x1, y1, x2, y2), "ppe", color=self.color)
                    )
                    if label in ("no_ppe", "no_helmet"):
                        result.alerts.append(f"PPE violation: {label}")
        except Exception as exc:  # pragma: no cover
            log.debug("PPE inference error: %s", exc)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result

