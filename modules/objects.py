"""Object-oriented AI modules: vehicles, animals, currency notes.

Vehicles are detected with YOLOv8/11 COCO classes and counted with a simple
line-crossing tracker. Animal detection reuses the COCO "animal" classes.
Currency uses a small YOLO model if available; otherwise reports unavailable.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from modules.base import BaseModule, DEVICE, ModuleResult, ensure_model
from tracker import CentroidTracker
from utils import Detection

log = logging.getLogger("modules.objects")

# COCO class ids relevant to vehicles and animals.
_VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_ANIMAL_CLASSES = {14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
                   19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}

_CURRENCY_URL = "https://github.com/spacewalk01/yolov5-vehicle-and-licenseplate/releases/download/v1.0/currency.onnx"


class VehicleModule(BaseModule):
    """Detect, track and count vehicles crossing a vertical line."""

    key = "vehicle"
    color = (120, 220, 80)

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._tracker = CentroidTracker()
        self.count = 0
        self._line_x = 0.5

    def _ensure(self):
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO("yolov8n.pt")
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

        h, w = frame.shape[:2]
        line_x = int(w * self._line_x)
        try:
            preds = model.predict(frame, conf=config.confidence, verbose=False)[0]
            dets = []
            for idx in range(len(preds.boxes)):
                cls = int(preds.boxes.cls[idx])
                if cls not in _VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = [int(v) for v in preds.boxes.xyxy[idx].tolist()]
                conf = float(preds.boxes.conf[idx])
                det = Detection(_VEHICLE_CLASSES[cls], conf, (x1, y1, x2, y2), "vehicle", color=self.color)
                dets.append(det)
                result.detections.append(det)

            tracks = self._tracker.update(dets)
            # count vehicles crossing the line west->east
            for track in tracks.values():
                cx = track.centroid[0]
                if track.centroid[0] < line_x:
                    continue
                if abs(cx - line_x) < 8:
                    self.count += 1
            result.info["vehicles"] = len(dets)
            result.info["vehicle_count"] = self.count
            result.info["tracks"] = len(tracks)
        except Exception as exc:  # pragma: no cover
            log.debug("vehicle inference error: %s", exc)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class AnimalModule(BaseModule):
    """Detect common animals (COCO classes) and label the species."""

    key = "animal"
    color = (255, 160, 160)

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO("yolov8n.pt")
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
            preds = model.predict(frame, conf=config.confidence, verbose=False)[0]
            for idx in range(len(preds.boxes)):
                cls = int(preds.boxes.cls[idx])
                if cls not in _ANIMAL_CLASSES:
                    continue
                x1, y1, x2, y2 = [int(v) for v in preds.boxes.xyxy[idx].tolist()]
                conf = float(preds.boxes.conf[idx])
                result.detections.append(
                    Detection(_ANIMAL_CLASSES[cls], conf, (x1, y1, x2, y2), "animal", color=self.color)
                )
            result.info["animals"] = len(result.detections)
        except Exception as exc:  # pragma: no cover
            log.debug("animal inference error: %s", exc)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class CurrencyModule(BaseModule):
    """Recognise currency notes using a small YOLO model.

    If the currency model is not available, the module reports unavailable
    rather than degrading to a meaningless result.
    """

    key = "currency"
    color = (120, 255, 200)

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def _ensure(self):
        if self._model is None:
            path = ensure_model(_CURRENCY_URL, "currency.onnx")
            if path is None:
                return None
            from modules.base import load_onnx

            self._model = load_onnx("currency.onnx")
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
            blob = frame.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[None, ...]
            outs = model.run(None, {model.get_inputs()[0].name: blob})[0][0]
            h, w = frame.shape[:2]
            for row in outs:
                if row[4] < config.confidence:
                    continue
                x1, y1, x2, y2 = [int(v * w) for v in row[:4]]
                cls = int(row[5]) if row.shape[0] > 5 else 0
                label = f"note:{cls}"
                result.detections.append(
                    Detection(label, float(row[4]), (x1, y1, x2, y2), "currency", color=self.color)
                )
        except Exception as exc:  # pragma: no cover
            log.debug("currency inference error: %s", exc)

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result
