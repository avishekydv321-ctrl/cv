"""Detection back-ends: YOLO objects, mediapipe tasks face/hands/pose, OCR & codes.

Every back-end is lazily initialised so the app starts instantly and keeps
working when an optional dependency (Tesseract, pyzbar, ultralytics) is
missing - the corresponding mode simply reports itself as unavailable.

mediapipe is used through the modern **Tasks API**
(``mediapipe.tasks.python.vision``), which is the only API shipped by
``mediapipe>=1.0``. The legacy ``mediapipe.solutions`` API no longer exists in
that release, so we bind to the Tasks API and auto-download the small
``.task`` model bundles into ``models/`` on first use. If a model cannot be
fetched, the mode degrades gracefully (returns empty results) instead of
crashing the pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from utils import Detection, MODELS_DIR, ensure_model as _ensure_model

log = logging.getLogger("detector")

# mediapipe tasks model catalogue: basename -> download URL.
_MP_MODELS: dict[str, str] = {
    "face_detector.task": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

# Cache of resolved task model paths so we never re-attempt a failed download
# every frame.
_model_paths: dict[str, Optional[Path]] = {}


def _resolve_model(name: str) -> Optional[Path]:
    """Return the path to a mediapipe tasks model, downloading it if needed."""
    if name in _model_paths:
        return _model_paths[name]

    # Existing non-empty file short-circuits the download.
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        _model_paths[name] = dest
        return dest

    url = _MP_MODELS.get(name)
    if url is None:
        _model_paths[name] = None
        return None

    # Reuse the consolidated downloader (caches failures, never re-downloads).
    resolved = _ensure_model(name, url)
    _model_paths[name] = resolved or None
    return _model_paths[name]


def _mp_image(frame: np.ndarray) -> Any:
    """Wrap a BGR OpenCV frame as a mediapipe tasks ``Image``."""
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Guarantee a contiguous 3-channel BGR array for drawing in place."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(frame)


class ObjectDetector:
    """Thin wrapper over Ultralytics YOLO with hot-swappable weights.

    Prefers YOLOv11 weights when the installed ``ultralytics`` version supports
    them, and transparently falls back to YOLOv8 if YOLOv11 is unavailable.
    """

    def __init__(self, weights: str = "yolov8n.pt") -> None:
        self.weights = weights
        self._model: Any = None
        self._fallback: Any = None
        self._active_weights: str = ""

    def _load(self, weights: str) -> Any:
        from ultralytics import YOLO  # imported lazily (slow import)

        log.info("Loading YOLO weights: %s", weights)
        return YOLO(weights)

    def _ensure(self) -> Any:
        """Return a usable model, trying the configured weight then YOLOv8.

        The resolved model is cached so we never re-attempt a failed download
        (or re-load the fallback) on every frame - that network churn caused
        severe per-frame lag when the requested weight was missing locally.
        """
        if self.weights != self._active_weights:
            self._model = None
            self._fallback = None
            self._active_weights = self.weights

        if self._model is not None:
            return self._model

        # A fallback was already resolved on a previous frame - reuse it.
        if self._fallback is not None:
            self._model = self._fallback
            return self._model

        # Directly load the requested weights (YOLOv11 or YOLOv8).
        try:
            self._model = self._load(self.weights)
            return self._model
        except Exception as exc:  # pragma: no cover - depends on ultralytics build
            log.warning("Could not load %s (%s); will try YOLOv8 fallback", self.weights, exc)

        # YOLOv11 requested but unavailable -> fall back to a YOLOv8 weight.
        if "11" in self.weights:
            fallback_path = self.weights.replace("11", "8")
            try:
                self._fallback = self._load(fallback_path)
                # Cache the fallback as the active model so subsequent frames
                # skip the failed YOLOv11 attempt entirely.
                self._model = self._fallback
                return self._model
            except Exception as exc:  # pragma: no cover
                log.error("YOLOv8 fallback also failed: %s", exc)
        return self._model

    def set_weights(self, weights: str) -> None:
        if weights != self.weights:
            self.weights = weights
            self._model = None
            self._fallback = None
            self._active_weights = ""

    def detect(self, frame: np.ndarray, confidence: float = 0.45, track: bool = False) -> list[Detection]:
        try:
            model = self._ensure()
        except Exception as exc:  # pragma: no cover - depends on environment
            log.error("YOLO unavailable: %s", exc)
            return []

        if track:
            try:
                results = model.track(frame, conf=confidence, persist=True, verbose=False)
            except Exception as e:
                log.warning("YOLO track failed, falling back to predict: %s", e)
                results = model.predict(frame, conf=confidence, verbose=False)
        else:
            results = model.predict(frame, conf=confidence, verbose=False)

        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                track_id = int(box.id[0].item()) if box.id is not None else None
                label = str(names[int(box.cls[0])])
                label_str = f"{label} #{track_id}" if track_id is not None else label
                detections.append(
                    Detection(
                        label=label_str,
                        confidence=float(box.conf[0]),
                        box=(x1, y1, x2, y2),
                        source="object",
                    )
                )
        return detections


# --------------------------------------------------------------------------
# Shared face-detection helper used by the main FaceDetector mode AND the
# module-level face-based detectors (FaceMask, EmotionDetector, etc.) that
# only need bounding-box crops for downstream ONNX models.
# --------------------------------------------------------------------------
_FACE_DETECTOR_SINGLETON: Optional["_TasksFaceDetector"] = None


def _get_face_detector():
    """Return a shared singleton instance of the Tasks face detector."""
    global _FACE_DETECTOR_SINGLETON
    if _FACE_DETECTOR_SINGLETON is None:
        _FACE_DETECTOR_SINGLETON = _TasksFaceDetector()
    return _FACE_DETECTOR_SINGLETON


class _TasksFaceDetector:
    """Lazy-initialised Tasks face-detector that yields bounding-box lists."""

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector as MPD,
            FaceDetectorOptions,
            RunningMode,
        )

        face_path = _resolve_model("face_detector.task")
        self._detector = None
        if face_path is not None:
            try:
                options = FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(face_path)),
                    running_mode=RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self._detector = MPD.create_from_options(options)
            except Exception as exc:  # pragma: no cover
                log.warning("_TasksFaceDetector init failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._detector is not None

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Return list of (x1, y1, x2, y2, score) for every detected face."""
        if self._detector is None:
            return []
        h, w = frame.shape[:2]
        image = _mp_image(frame)
        try:
            result = self._detector.detect(image)
            boxes = []
            for det in result.detections:
                bb = det.bounding_box
                x1 = max(0, int(bb.origin_x))
                y1 = max(0, int(bb.origin_y))
                x2 = min(w, int(bb.origin_x + bb.width))
                y2 = min(h, int(bb.origin_y + bb.height))
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                score = float(det.categories[0].score) if det.categories else 0.0
                boxes.append((x1, y1, x2, y2, score))
            return boxes
        except Exception as exc:  # pragma: no cover
            log.debug("_TasksFaceDetector detect error: %s", exc)
            return []


class FaceDetector:
    """mediapipe tasks face detection + landmark mesh overlay."""

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector as MPD,
            FaceDetectorOptions,
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        self._mp = None
        try:
            import mediapipe as mp

            self._mp = mp
        except Exception:
            pass

        face_path = _resolve_model("face_detector.task")
        landmark_path = _resolve_model("face_landmarker.task")
        self._detector = None
        self._landmarker = None

        if face_path is not None:
            try:
                options = FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(face_path)),
                    running_mode=RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                )
                self._detector = MPD.create_from_options(options)
            except Exception as exc:  # pragma: no cover
                log.warning("FaceDetector init failed: %s", exc)
                self._detector = None

        if landmark_path is not None:
            try:
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(landmark_path)),
                    running_mode=RunningMode.IMAGE,
                    num_faces=4,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
            except Exception as exc:  # pragma: no cover
                log.warning("FaceLandmarker init failed: %s", exc)
                self._landmarker = None

    @property
    def available(self) -> bool:
        return self._detector is not None

    def detect(self, frame: np.ndarray, draw_mesh: bool = True) -> list[Detection]:
        if self._detector is None:
            return []

        h, w = frame.shape[:2]
        image = _mp_image(frame)
        detections: list[Detection] = []

        try:
            result = self._detector.detect(image)
            for index, det in enumerate(result.detections):
                bb = det.bounding_box
                x1 = max(0, int(bb.origin_x))
                y1 = max(0, int(bb.origin_y))
                x2 = min(w, int(bb.origin_x + bb.width))
                y2 = min(h, int(bb.origin_y + bb.height))
                score = float(det.categories[0].score) if det.categories else 0.0
                detections.append(
                    Detection(f"face #{index + 1}", score, (x1, y1, x2, y2), "face")
                )
        except Exception as exc:  # pragma: no cover
            log.debug("Face detect error: %s", exc)

        if draw_mesh and self._landmarker is not None:
            try:
                mesh_result = self._landmarker.detect(image)
                drawing = self._drawing_utils()
                if drawing is not None:
                    for landmarks in getattr(mesh_result, "face_landmarks", []) or []:
                        drawing(
                            frame, landmarks, self._face_connections(),
                            None, self._tesselation_style(),
                        )
            except Exception as exc:  # pragma: no cover
                log.debug("Face mesh error: %s", exc)

        return detections

    def _drawing_utils(self):
        try:
            from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

            return draw_landmarks
        except Exception:
            return None

    def _face_connections(self):
        try:
            from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

            return FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        except Exception:
            return None

    def _tesselation_style(self):
        try:
            from mediapipe.tasks.python.vision.drawing_styles import (
                get_default_face_mesh_tesselation_style,
            )

            return get_default_face_mesh_tesselation_style()
        except Exception:
            return None


# --------------------------------------------------------------------------
# Shared hand-landmark helper used by the HandTracker mode AND the
# SignLanguage module (which needs raw landmarks + handedness, not just boxes).
# --------------------------------------------------------------------------
_HAND_LANDMARKER_SINGLETON: Optional["_TasksHandLandmarker"] = None


def _get_hand_landmarker():
    """Return a shared singleton instance of the Tasks hand landmarker."""
    global _HAND_LANDMARKER_SINGLETON
    if _HAND_LANDMARKER_SINGLETON is None:
        _HAND_LANDMARKER_SINGLETON = _TasksHandLandmarker()
    return _HAND_LANDMARKER_SINGLETON


class _TasksHandLandmarker:
    """Lazy-initialised Tasks hand landmarker yielding landmarks + handedness."""

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker as MPL,
            HandLandmarkerOptions,
            RunningMode,
        )

        hand_path = _resolve_model("hand_landmarker.task")
        self._hands = None
        if hand_path is not None:
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=RunningMode.IMAGE,
                    num_hands=2,
                )
                self._hands = MPL.create_from_options(options)
            except Exception as exc:  # pragma: no cover
                log.warning("_TasksHandLandmarker init failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._hands is not None

    def detect(self, frame: np.ndarray) -> tuple[list[Any], list[str]]:
        """Return (landmark_lists, side_labels). Each landmark is a list of
        ``NormalizedLandmark`` objects with ``x`` and ``y`` attributes."""
        if self._hands is None:
            return [], []
        image = _mp_image(frame)
        try:
            result = self._hands.detect(image)
            hands = getattr(result, "hand_landmarks", []) or []
            sides = []
            for hd in getattr(result, "handedness", []) or []:
                sides.append(hd[0].category_name if hd else "Right")
            return hands, sides
        except Exception as exc:  # pragma: no cover
            log.debug("_TasksHandLandmarker detect error: %s", exc)
            return [], []


class HandTracker:
    """Hand landmarks with finger counting, pinch and simple gestures."""

    TIPS = (4, 8, 12, 16, 20)

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker as MPL,
            HandLandmarkerOptions,
            RunningMode,
        )

        hand_path = _resolve_model("hand_landmarker.task")
        self._hands = None
        if hand_path is not None:
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=RunningMode.IMAGE,
                    num_hands=2,
                )
                self._hands = MPL.create_from_options(options)
            except Exception as exc:  # pragma: no cover
                log.warning("HandLandmarker init failed: %s", exc)
                self._hands = None

    @property
    def available(self) -> bool:
        return self._hands is not None

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        if self._hands is None:
            return {"hands": [], "count": 0}
        frame = _to_bgr(frame)
        image = _mp_image(frame)
        hands: list[dict[str, Any]] = []

        try:
            result = self._hands.detect(image)
            drawing = self._drawing_utils()
            for landmarks, handedness in zip(
                getattr(result, "hand_landmarks", []) or [],
                getattr(result, "handedness", []) or [],
            ):
                if drawing is not None:
                    drawing(
                        frame, landmarks, self._hand_connections(),
                        self._hand_landmark_style(), self._hand_connection_style(),
                    )
                points = [(lm.x, lm.y) for lm in landmarks]
                side = handedness[0].category_name if handedness else "Right"
                fingers = self._count_fingers(points, side)
                pinch = float(np.hypot(points[4][0] - points[8][0], points[4][1] - points[8][1]))
                hands.append({
                    "side": side,
                    "fingers": fingers,
                    "pinch": pinch < 0.05,
                    "gesture": self._gesture(fingers, pinch),
                })
        except Exception as exc:  # pragma: no cover
            log.debug("Hand process error: %s", exc)

        return {"hands": hands, "count": len(hands)}

    def _drawing_utils(self):
        try:
            from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

            return draw_landmarks
        except Exception:
            return None

    def _hand_connections(self):
        try:
            from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

            return HandLandmarksConnections.HAND_CONNECTIONS
        except Exception:
            return None

    def _hand_landmark_style(self):
        try:
            from mediapipe.tasks.python.vision.drawing_styles import (
                get_default_hand_landmarks_style,
            )

            return get_default_hand_landmarks_style()
        except Exception:
            return None

    def _hand_connection_style(self):
        try:
            from mediapipe.tasks.python.vision.drawing_styles import (
                get_default_hand_connections_style,
            )

            return get_default_hand_connections_style()
        except Exception:
            return None

    def _count_fingers(self, points: list[tuple[float, float]], side: str) -> int:
        up = 0
        thumb_open = points[4][0] < points[3][0] if side == "Right" else points[4][0] > points[3][0]
        up += int(thumb_open)
        for tip in self.TIPS[1:]:
            up += int(points[tip][1] < points[tip - 2][1])
        return up

    @staticmethod
    def _gesture(fingers: int, pinch: float) -> str:
        if pinch < 0.05:
            return "Pinch"
        return {0: "Fist", 1: "Point", 2: "Peace", 5: "Open palm"}.get(fingers, f"{fingers} fingers")


class PoseEstimator:
    """Full-body skeleton plus squat and push-up repetition counters."""

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker as MPP,
            PoseLandmarkerOptions,
            RunningMode,
        )

        pose_path = _resolve_model("pose_landmarker.task")
        self._pose = None
        if pose_path is not None:
            try:
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(pose_path)),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                )
                self._pose = MPP.create_from_options(options)
            except Exception as exc:  # pragma: no cover
                log.warning("PoseLandmarker init failed: %s", exc)
                self._pose = None
        self.squats = 0
        self.pushups = 0
        self._squat_down = False
        self._pushup_down = False

    @property
    def available(self) -> bool:
        return self._pose is not None

    def reset(self) -> None:
        self.squats = self.pushups = 0
        self._squat_down = self._pushup_down = False

    @staticmethod
    def angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """Interior angle at point b, in degrees."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine = float(np.dot(ba, bc) / ((np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def process(self, frame: np.ndarray) -> dict[str, Any]:
        if self._pose is None:
            return {"detected": False, "squats": self.squats, "pushups": self.pushups}
        frame = _to_bgr(frame)
        image = _mp_image(frame)

        try:
            result = self._pose.detect(image)
            if not getattr(result, "pose_landmarks", None):
                return {"detected": False, "squats": self.squats, "pushups": self.pushups}

            drawing = self._drawing_utils()
            if drawing is not None:
                drawing(
                    frame, result.pose_landmarks[0], self._pose_connections(),
                    self._pose_landmark_style(), self._pose_connection_style(),
                )
            lm = result.pose_landmarks[0]
            p = lambda i: (lm[i].x, lm[i].y)  # noqa: E731 - concise local accessor

            knee = self.angle(p(23), p(25), p(27))
            elbow = self.angle(p(11), p(13), p(15))

            if knee < 95 and not self._squat_down:
                self._squat_down = True
            elif knee > 160 and self._squat_down:
                self._squat_down = False
                self.squats += 1

            if elbow < 90 and not self._pushup_down:
                self._pushup_down = True
            elif elbow > 160 and self._pushup_down:
                self._pushup_down = False
                self.pushups += 1

            return {
                "detected": True,
                "knee_angle": round(knee, 1),
                "elbow_angle": round(elbow, 1),
                "squats": self.squats,
                "pushups": self.pushups,
            }
        except Exception as exc:  # pragma: no cover
            log.debug("Pose process error: %s", exc)
            return {"detected": False, "squats": self.squats, "pushups": self.pushups}

    def _drawing_utils(self):
        try:
            from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks

            return draw_landmarks
        except Exception:
            return None

    def _pose_connections(self):
        try:
            from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

            return PoseLandmarksConnections.POSE_LANDMARKS
        except Exception:
            return None

    def _pose_landmark_style(self):
        try:
            from mediapipe.tasks.python.vision.drawing_styles import get_default_pose_landmarks_style

            return get_default_pose_landmarks_style()
        except Exception:
            return None

    def _pose_connection_style(self):
        # The Tasks ``drawing_styles`` module has no dedicated pose-connection
        # style; a default is applied by ``draw_landmarks`` when None is passed.
        return None


class CodeAndTextReader:
    """QR/barcode decoding (pyzbar) and OCR (Tesseract), both optional."""

    def __init__(self) -> None:
        self._qr = cv2.QRCodeDetector()

    def read_codes(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        try:
            from pyzbar import pyzbar

            for code in pyzbar.decode(frame):
                x, y, w, h = code.rect
                detections.append(
                    Detection(code.data.decode("utf-8", "ignore"), 1.0, (x, y, x + w, y + h), "code")
                )
            return detections
        except Exception:  # pyzbar or zbar runtime missing -> OpenCV fallback
            data, points, _ = self._qr.detectAndDecode(frame)
            if data and points is not None:
                pts = points.reshape(-1, 2).astype(int)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                detections.append(Detection(data, 1.0, (int(x1), int(y1), int(x2), int(y2)), "code"))
            return detections

    def read_text(self, frame: np.ndarray) -> str:
        try:
            import pytesseract

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return pytesseract.image_to_string(gray).strip()
        except Exception as exc:  # pragma: no cover
            log.debug("OCR unavailable: %s", exc)
            return ""


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw labelled boxes with a readable filled caption strip.

    Boxes are validated defensively: any detection whose box is malformed
    (wrong length, non-numeric, or zero-area) is skipped so a single bad
    entry can never crash the whole frame pipeline.
    """
    from utils import color_for

    height, width = frame.shape[:2]
    for det in detections:
        box = det.box
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # Clamp to the frame bounds.
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        color = det.color or color_for(det.label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{det.label} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, caption, (x1 + 4, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 20), 1, cv2.LINE_AA)
    return frame
