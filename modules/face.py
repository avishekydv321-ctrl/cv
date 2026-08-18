"""Face-related AI modules.

Face recognition uses MediaPipe Tasks face detection + face_recognition
embeddings (fallback to a MediaPipe Tasks-derived embedding when
face_recognition is missing). Mask, emotion and age/gender are all served by a
single lightweight MobileNet-style classifier that runs through ONNX Runtime
when available, and gracefully reports itself unavailable otherwise.

All face detection is delegated to the shared Tasks API helper in
``detector.py`` (``_get_face_detector``), so these modules work with the modern
``mediapipe>=1.0`` that no longer ships the legacy ``mediapipe.solutions`` API.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

from detector import _get_face_detector
from face_db import FaceDB
from modules.base import BaseModule, DEVICE, ModuleResult, ensure_model, load_onnx
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

EMOTIONS = ("neutral", "happy", "sad", "surprise", "anger")


class FaceRecognition(BaseModule):
    """Register and recognise faces against a persistent identity DB."""

    key = "face_recognition"
    color = (120, 220, 255)

    def __init__(self, db: Optional[FaceDB] = None) -> None:
        super().__init__()
        self.db = db or FaceDB()
        self._mp_face = None
        self._fr = None

    # -- backends ------------------------------------------------------
    def _check_available(self) -> bool:
        return self._ensure() is not None

    def _ensure(self):
        if self._fr is None:
            try:
                import face_recognition  # type: ignore

                self._fr = face_recognition
            except Exception:
                self._fr = None
        # Use the shared Tasks face detector instead of legacy mp.solutions.
        face_det = _get_face_detector()
        self._mp_face = face_det if face_det.available else None
        return self._fr

    # -- API used by the GUI ------------------------------------------
    def embed_face(self, frame: np.ndarray) -> Optional[list[float]]:
        """Return the embedding of the largest face in the frame (or None)."""
        if self._ensure() is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hasattr(self._fr, "face_locations"):  # classic face_recognition model
            locs = self._fr.face_locations(rgb)
            if not locs:
                return None
            # largest box
            locs.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            encs = self._fr.face_encodings(rgb, locs[:1])
            return list(encs[0]) if encs else None

        # Tasks face detector fallback: crop largest face for embedding.
        face_boxes = _faces_in_frame(frame)
        if not face_boxes:
            return None
        (x1, y1, x2, y2), *_ = sorted(  # largest box
            face_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        crop = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / 255.0
        # Deterministic hand-crafted descriptor: 128 bins of oriented edges.
        hist = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9).compute(
            cv2.resize(gray, (64, 64))
        )
        embedding = hist.flatten().astype(float).tolist()[:128]
        if len(embedding) < 128:
            embedding.extend([0.0] * (128 - len(embedding)))
        return embedding

    def _match(self, embedding: list[float]) -> tuple[Optional[str], float]:
        if not self.db.embeddings():
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, known in self.db.embeddings():
            score = float(np.dot(embedding, known) / (
                np.linalg.norm(embedding) * np.linalg.norm(known) + 1e-6
            ))
            if score > best_score:
                best_name, best_score = name, score
        return best_name, best_score

    # -- pipeline ------------------------------------------------------
    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        if self._ensure() is None:
            result.info["available"] = False
            return result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if hasattr(self._fr, "face_locations"):
            locs = self._fr.face_locations(rgb)
            encs = self._fr.face_encodings(rgb, locs)
        else:
            # Tasks face detector fallback: use (top, right, bottom, left) boxes.
            boxes = _faces_in_frame(frame)
            locs = [(y1, x2, y2, x1) for x1, y1, x2, y2 in boxes]
            encs = [self.embed_face(frame)] * len(locs) if locs else []

        names = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            if enc is None:
                continue
            name, score = self._match(enc)
            if score is None:
                score = 0.0
            if name is None or score < 0.35:
                name, score = "Unknown", max(score, 0.0)
                result.alerts.append("Unknown face detected")
            else:
                self.db.log_recognition(name, score)
                result.info.setdefault("recognized", []).append(name)
            box = (left, top, right, bottom)
            result.detections.append(Detection(f"face:{name}", max(score, 0.1), box, "face"))
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
