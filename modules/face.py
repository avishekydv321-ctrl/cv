"""Face-related AI modules.

Individual face identity uses OpenCV Zoo YuNet detection + SFace embeddings.
Mask, emotion and age/gender are served by ONNX Runtime classifiers when
their optional model files are available.

The shared MediaPipe Tasks detector remains available for the face mesh and
other existing face-analysis modules.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

from detector import _get_face_detector
from face_db import FaceDB
from modules.base import BaseModule, ModuleResult, ensure_model, load_onnx
from utils import Detection


def _faces_in_frame(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return [(x1, y1, x2, y2), ...] face boxes via the shared Tasks detector."""
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in _get_face_detector().detect(frame)]


log = logging.getLogger("modules.face")

# Small public ONNX classifiers (ONNX Model Zoo / community). If a resource is
# unreachable they simply report "unavailable" — the app stays functional.
_MASK_URL = "https://github.com/elyase/face-mask-detection/releases/download/v1.0/mask_model.onnx"
_EMOTION_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
_AGEGEN_URL = "https://github.com/onnx/models/raw/main/vision/body_analysis/age_gender/model/age_gender.onnx"
_FACE_DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
_FACE_EMBEDDING_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
_FACE_EMBEDDING_MODEL = "opencv_sface_2021dec"

EMOTIONS = ("neutral", "happy", "sad", "surprise", "anger")


class FaceRecognition(BaseModule):
    """Register and recognise faces against a persistent identity DB."""

    key = "face_recognition"
    color = (120, 220, 255)

    def __init__(self, db: Optional[FaceDB] = None) -> None:
        super().__init__()
        self.db = db or FaceDB()
        self._detector = None
        self._recognizer = None
        self._model_error = ""
        self._last_history: dict[str, float] = {}

    # -- backends ------------------------------------------------------
    def _check_available(self) -> bool:
        return self._ensure()

    def _ensure(self) -> bool:
        if self._detector is not None and self._recognizer is not None:
            return True
        if self._model_error:
            return False
        try:
            detector_path = ensure_model(_FACE_DETECTOR_URL, "face_detection_yunet_2023mar.onnx")
            embedding_path = ensure_model(
                _FACE_EMBEDDING_URL, "face_recognition_sface_2021dec.onnx"
            )
            if detector_path is None or embedding_path is None:
                raise RuntimeError("Could not download the YuNet/SFace face models.")
            self._detector = cv2.FaceDetectorYN_create(
                str(detector_path), "", (320, 320), 0.6, 0.3, 5000
            )
            self._recognizer = cv2.FaceRecognizerSF_create(str(embedding_path), "")
            return True
        except Exception as exc:
            self._model_error = str(exc)
            log.warning("Individual face recognition unavailable: %s", exc)
            return False

    def _extract_faces(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], list[float]]]:
        """Return each face box and its normalized SFace embedding."""
        if not self._ensure():
            return []
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None:
            return []
        extracted = []
        for face in faces:
            x, y, box_width, box_height = (int(value) for value in face[:4])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + box_width), min(height, y + box_height)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            aligned = self._recognizer.alignCrop(frame, face)
            feature = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-8:
                continue
            feature /= norm
            extracted.append(((x1, y1, x2, y2), feature.astype(float).tolist()))
        return extracted

    # -- API used by the GUI ------------------------------------------
    def embed_face(self, frame: np.ndarray) -> Optional[list[float]]:
        """Return the largest face's SFace embedding, or None when no face exists."""
        faces = self._extract_faces(frame)
        if not faces:
            return None
        _, embedding = max(
            faces,
            key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
        )
        return embedding

    def _match(self, embedding: list[float]) -> tuple[Optional[str], Optional[str], float]:
        query = np.asarray(embedding, dtype=np.float32)
        best_name, best_id, best_score = None, None, -1.0
        for name, unique_id, known_values in self.db.samples_with_ids(_FACE_EMBEDDING_MODEL):
            known = np.asarray(known_values, dtype=np.float32)
            if known.shape != query.shape:
                continue
            score = float(np.dot(query, known) / (np.linalg.norm(known) * np.linalg.norm(query) + 1e-8))
            if score > best_score:
                best_name, best_id, best_score = name, unique_id, score
        return best_name, best_id, max(0.0, best_score)

    # -- pipeline ------------------------------------------------------
    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        if not self._ensure():
            result.info["available"] = False
            result.info["error"] = self._model_error
            return result

        names = []
        threshold = float(getattr(config, "face_similarity_threshold", 0.45))
        for box, embedding in self._extract_faces(frame):
            name, unique_id, score = self._match(embedding)
            if name is None or score < threshold:
                name, unique_id = "Unknown Person", None
                result.alerts.append("Unknown face detected")
            else:
                if getattr(config, "face_history_enabled", False):
                    now = time.monotonic()
                    if now - self._last_history.get(unique_id, 0.0) >= 2.0:
                        self.db.log_recognition(name, score)
                        self._last_history[unique_id] = now
                result.info.setdefault("recognized", []).append(name)
            result.detections.append(
                Detection(name, score, box, "individual_face", color=self.color)
            )
            names.append(f"{name} ({score:.0%})")
        result.info["recognition"] = ", ".join(names) or "No faces"
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class FaceMask(BaseModule):
    """Detect whether detected faces are wearing a mask."""

    key = "face_mask"
    color = (80, 220, 120)

    def __init__(self) -> None:
        super().__init__()
        self._session = None
        self._fd = None

    def _ensure(self):
        if self._session is None:
            self._session = load_onnx("mask_model.onnx") or ensure_model(_MASK_URL, "mask_model.onnx")
        # Use the shared Tasks face detector instead of legacy mp.solutions.
        face_det = _get_face_detector()
        self._fd = face_det if face_det.available else None
        return self._session

    def _check_available(self) -> bool:
        return self._ensure() is not None

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        if self._ensure() is None or self._fd is None:
            result.info["available"] = False
            return result

        for x1, y1, x2, y2 in _faces_in_frame(frame):
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
            blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[None, ...]
            try:
                out = self._session.run(None, {self._session.get_inputs()[0].name: blob})[0][0]
                idx = int(np.argmax(out))
                label = "mask" if idx == 0 else "no_mask"
                conf = float(np.max(out))
            except Exception as exc:  # pragma: no cover
                log.debug("mask inference error: %s", exc)
                continue
            color = (80, 220, 120) if label == "mask" else (60, 60, 220)
            result.detections.append(Detection(label, conf, (x1, y1, x2, y2), "face_mask", color=color))
            if label == "no_mask":
                result.alerts.append("Person without mask detected")
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class EmotionDetector(BaseModule):
    """Basic emotion classification (FER+ 8-class model)."""

    key = "emotion"
    color = (240, 200, 120)

    def __init__(self) -> None:
        super().__init__()
        self._session = None
        self._fd = None

    def _ensure(self):
        if self._session is None:
            self._session = load_onnx("emotion-ferplus-8.onnx") or ensure_model(
                _EMOTION_URL, "emotion-ferplus-8.onnx"
            )
        # Use the shared Tasks face detector instead of legacy mp.solutions.
        face_det = _get_face_detector()
        self._fd = face_det if face_det.available else None
        return self._session

    def _check_available(self) -> bool:
        return self._ensure() is not None

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        if self._ensure() is None or self._fd is None:
            result.info["available"] = False
            return result

        labels = []
        for x1, y1, x2, y2 in _faces_in_frame(frame):
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
            blob = gray[None, None, ...]
            try:
                out = self._session.run(None, {self._session.get_inputs()[0].name: blob})[0][0]
                idx = int(np.argmax(out))
                label = EMOTIONS[idx] if idx < len(EMOTIONS) else "neutral"
                conf = float(np.max(out))
            except Exception as exc:  # pragma: no cover
                log.debug("emotion inference error: %s", exc)
                continue
            result.detections.append(Detection(f"emotion:{label}", conf, (x1, y1, x2, y2), "emotion", color=self.color))
            labels.append(f"{label} ({conf:.0%})")
        result.info["emotions"] = ", ".join(labels) or "—"
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class AgeGenderEstimator(BaseModule):
    """Estimate age bracket and gender for each detected face."""

    key = "age_gender"
    color = (200, 160, 240)

    def __init__(self) -> None:
        super().__init__()
        self._session = None
        self._fd = None

    def _ensure(self):
        if self._session is None:
            self._session = load_onnx("age_gender.onnx") or ensure_model(
                _AGEGEN_URL, "age_gender.onnx"
            )
        # Use the shared Tasks face detector instead of legacy mp.solutions.
        face_det = _get_face_detector()
        self._fd = face_det if face_det.available else None
        return self._session

    def _check_available(self) -> bool:
        return self._ensure() is not None

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        if self._ensure() is None or self._fd is None:
            result.info["available"] = False
            return result

        labels = []
        for x1, y1, x2, y2 in _faces_in_frame(frame):
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            blob = cv2.resize(frame[y1:y2, x1:x2], (64, 64)).astype(np.float32)
            blob = (blob / 127.5 - 1.0).astype(np.float32)
            blob = blob.transpose(2, 0, 1)[None, ...]
            try:
                outs = self._session.run(None, {self._session.get_inputs()[0].name: blob})
                age = float(outs[0].reshape(-1)[0]) if outs else 0.0
                gender = "M" if outs and outs[1].reshape(-1).argmax() == 1 else "F"
                label = f"{gender}, {int(age)}y"
            except Exception as exc:  # pragma: no cover
                log.debug("age/gender inference error: %s", exc)
                continue
            result.detections.append(Detection("age_gender:" + label, 0.9, (x1, y1, x2, y2), "age_gender", color=self.color))
            labels.append(label)
        result.info["age_gender"] = ", ".join(labels) or "—"
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result
