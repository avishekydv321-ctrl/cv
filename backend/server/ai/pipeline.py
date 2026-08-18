"""AI inference pipeline reusing existing VisionSuite modules."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np

from server.core.config import settings

logger = logging.getLogger("ai.pipeline")


class VisionPipeline:
    """Wrapper around the existing VisionSuite AI modules for web API usage."""

    def __init__(self):
        self._models_loaded = False
        self._yolo: Any = None
        self._face_detector: Any = None
        self._hand_tracker: Any = None
        self._pose_estimator: Any = None
        self._code_reader: Any = None
        self._modules: dict[str, Any] = {}
        self._device = settings.DEVICE
        logger.info("VisionPipeline initialized on device: %s", self._device)

    def _ensure_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                weights = getattr(settings, "YOLO_WEIGHTS", "yolov8n.pt")
                self._yolo = YOLO(weights)
                logger.info("YOLO model loaded: %s", weights)
            except Exception as exc:
                logger.error("Failed to load YOLO: %s", exc)
                raise
        return self._yolo

    def _ensure_face_detector(self):
        if self._face_detector is None:
            try:
                from detector import FaceDetector
                self._face_detector = FaceDetector()
                logger.info("Face detector loaded")
            except Exception as exc:
                logger.warning("Face detector unavailable: %s", exc)
        return self._face_detector

    def _ensure_hand_tracker(self):
        if self._hand_tracker is None:
            try:
                from detector import HandTracker
                self._hand_tracker = HandTracker()
                logger.info("Hand tracker loaded")
            except Exception as exc:
                logger.warning("Hand tracker unavailable: %s", exc)
        return self._hand_tracker

    def _ensure_pose_estimator(self):
        if self._pose_estimator is None:
            try:
                from detector import PoseEstimator
                self._pose_estimator = PoseEstimator()
                logger.info("Pose estimator loaded")
            except Exception as exc:
                logger.warning("Pose estimator unavailable: %s", exc)
        return self._pose_estimator

    def _ensure_code_reader(self):
        if self._code_reader is None:
            try:
                from detector import CodeAndTextReader
                self._code_reader = CodeAndTextReader()
                logger.info("Code reader loaded")
            except Exception as exc:
                logger.warning("Code reader unavailable: %s", exc)
        return self._code_reader

    def _ensure_module(self, key: str) -> Any:
        if key not in self._modules:
            try:
                from modules import get_module
                self._modules[key] = get_module(key)
                logger.info("Module loaded: %s", key)
            except Exception as exc:
                logger.warning("Module %s unavailable: %s", key, exc)
                self._modules[key] = None
        return self._modules[key]

    def detect_objects(self, frame: np.ndarray, confidence: float = 0.45) -> list[dict[str, Any]]:
        """Run YOLO object detection on a frame."""
        t0 = time.perf_counter()
        detections = []
        try:
            model = self._ensure_yolo()
            results = model.predict(frame, conf=confidence, verbose=False, device=self._device)
            for result in results:
                names = result.names
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    label = str(names[int(box.cls[0])])
                    conf = float(box.conf[0])
                    detections.append({
                        "label": label,
                        "confidence": conf,
                        "box": [x1, y1, x2, y2],
                        "source": "object",
                    })
        except Exception as exc:
            logger.error("Object detection failed: %s", exc)
        latency = (time.perf_counter() - t0) * 1000.0
        logger.debug("Object detection: %d results in %.1fms", len(detections), latency)
        return detections

    def detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run face detection on a frame."""
        t0 = time.perf_counter()
        detections = []
        face_det = self._ensure_face_detector()
        if face_det is None:
            return detections
        try:
            raw = face_det.detect(frame)
            for idx, (x1, y1, x2, y2, score) in enumerate(raw):
                detections.append({
                    "label": f"face #{idx + 1}",
                    "confidence": score,
                    "box": [x1, y1, x2, y2],
                    "source": "face",
                })
        except Exception as exc:
            logger.error("Face detection failed: %s", exc)
        logger.debug("Face detection: %d results in %.1fms", len(detections), (time.perf_counter() - t0) * 1000.0)
        return detections

    def detect_hands(self, frame: np.ndarray) -> dict[str, Any]:
        """Run hand tracking on a frame."""
        t0 = time.perf_counter()
        hands_data: dict[str, Any] = {"hands": [], "count": 0}
        hand_tracker = self._ensure_hand_tracker()
        if hand_tracker is None:
            return hands_data
        try:
            hands_data = hand_tracker.process(frame)
        except Exception as exc:
            logger.error("Hand tracking failed: %s", exc)
        logger.debug("Hand tracking in %.1fms", (time.perf_counter() - t0) * 1000.0)
        return hands_data

    def detect_pose(self, frame: np.ndarray) -> dict[str, Any]:
        """Run pose estimation on a frame."""
        t0 = time.perf_counter()
        pose_data: dict[str, Any] = {"detected": False, "squats": 0, "pushups": 0}
        pose_est = self._ensure_pose_estimator()
        if pose_est is None:
            return pose_data
        try:
            pose_data = pose_est.process(frame)
        except Exception as exc:
            logger.error("Pose estimation failed: %s", exc)
        logger.debug("Pose estimation in %.1fms", (time.perf_counter() - t0) * 1000.0)
        return pose_data

    def read_codes(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run QR/barcode detection on a frame."""
        t0 = time.perf_counter()
        codes = []
        reader = self._ensure_code_reader()
        if reader is None:
            return codes
        try:
            dets = reader.read_codes(frame)
            for det in dets:
                codes.append({
                    "label": det.label,
                    "confidence": det.confidence,
                    "box": list(det.box),
                    "source": det.source,
                })
        except Exception as exc:
            logger.error("Code reading failed: %s", exc)
        logger.debug("Code reading: %d results in %.1fms", len(codes), (time.perf_counter() - t0) * 1000.0)
        return codes

    def run_module(self, key: str, frame: np.ndarray, confidence: float = 0.45) -> dict[str, Any]:
        """Run a toggleable AI module on a frame."""
        t0 = time.perf_counter()
        module = self._ensure_module(key)
        if module is None:
            return {"available": False, "detections": [], "info": {}, "alerts": [], "inference_ms": 0.0}

        class FakeConfig:
            confidence = confidence

        try:
            result = module._run(frame, FakeConfig())
            return {
                "available": True,
                "detections": [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "box": list(d.box),
                        "source": d.source,
                        "color": list(d.color) if d.color else None,
                    }
                    for d in result.detections
                ],
                "info": result.info,
                "alerts": result.alerts,
                "inference_ms": result.inference_ms,
            }
        except Exception as exc:
            logger.error("Module %s failed: %s", key, exc)
            return {"available": False, "detections": [], "info": {}, "alerts": [], "inference_ms": 0.0}

    def process_frame(
        self,
        frame: np.ndarray,
        modules: list[str],
        confidence: float = 0.45,
    ) -> dict[str, Any]:
        """Run the full detection pipeline on a frame."""
        t0 = time.perf_counter()
        all_detections: list[dict[str, Any]] = []
        all_info: dict[str, Any] = {}
        all_alerts: list[str] = []
        module_times: dict[str, float] = {}

        # Base detection (YOLO objects)
        try:
            obj_dets = self.detect_objects(frame, confidence)
            all_detections.extend(obj_dets)
            counts: dict[str, int] = {}
            for det in obj_dets:
                cls = det["label"].split(" #")[0]
                counts[cls] = counts.get(cls, 0) + 1
            all_info["counts"] = counts
        except Exception as exc:
            logger.error("Base object detection error: %s", exc)

        # Run enabled modules
        for key in modules:
            try:
                mod_result = self.run_module(key, frame, confidence)
                all_detections.extend(mod_result.get("detections", []))
                module_times[key] = mod_result.get("inference_ms", 0.0)
                all_alerts.extend(mod_result.get("alerts", []))
                if mod_result.get("info"):
                    all_info[key] = mod_result["info"]
            except Exception as exc:
                logger.error("Module %s integration error: %s", key, exc)

        processing_time = (time.perf_counter() - t0) * 1000.0

        # Draw detections on frame
        try:
            annotated = self._draw_detections(frame, all_detections)
        except Exception:
            annotated = frame

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        import base64
        annotated_b64 = base64.b64encode(buf).decode("utf-8") if buf is not None else None

        return {
            "success": True,
            "fps": 0.0,
            "detections": all_detections,
            "info": all_info,
            "alerts": all_alerts,
            "modules": module_times,
            "processing_time_ms": round(processing_time, 1),
            "annotated_image_base64": annotated_b64,
        }

    def process_image(
        self,
        image_bytes: bytes,
        modules: list[str],
        confidence: float = 0.45,
    ) -> dict[str, Any]:
        """Process a single uploaded image."""
        t0 = time.perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Could not decode image", "processing_time_ms": 0.0, "detections": []}

        result = self.process_frame(frame, modules, confidence)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return result

    def process_anpr_image(self, image_bytes: bytes, confidence: float = 0.35) -> dict[str, Any]:
        """Process an image for ANPR (vehicle + plate detection + OCR)."""
        t0 = time.perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Could not decode image", "processing_time_ms": 0.0, "detections": []}

        try:
            from modules.anpr_video import VideoANPREngine
            engine = VideoANPREngine(confidence=confidence)
            engine.prepare()
            result = engine.process(frame)
            vehicles = result.vehicles
            plate_text = ""
            plate_conf = 0.0
            plate_box = None
            vehicle_box = None

            if vehicles:
                v = vehicles[0]
                vehicle_box = list(v.box)
                if v.plate and v.plate.text:
                    plate_text = v.plate.text
                    plate_conf = v.plate.confidence
                    plate_box = list(v.plate.box)

            annotated = result.frame
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            import base64
            annotated_b64 = base64.b64encode(buf).decode("utf-8") if buf is not None else None

            return {
                "success": True,
                "plate_text": plate_text,
                "confidence": plate_conf,
                "vehicle_box": vehicle_box,
                "plate_box": plate_box,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "annotated_image_base64": annotated_b64,
                "vehicle_count": result.vehicle_count,
                "plate_count": result.plate_count,
            }
        except Exception as exc:
            logger.error("ANPR processing failed: %s", exc)
            return {"success": False, "error": str(exc), "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def process_ocr_image(self, image_bytes: bytes) -> dict[str, Any]:
        """Process an image for OCR text extraction."""
        t0 = time.perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Could not decode image", "processing_time_ms": 0.0, "detections": []}

        try:
            import pytesseract
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh).strip()
            conf = 0.8 if text else 0.0

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            import base64
            annotated_b64 = base64.b64encode(buf).decode("utf-8") if buf is not None else None

            return {
                "success": True,
                "text": text,
                "confidence": conf,
                "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "annotated_image_base64": annotated_b64,
            }
        except Exception as exc:
            logger.error("OCR processing failed: %s", exc)
            return {"success": False, "error": str(exc), "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def get_model_status(self) -> dict[str, Any]:
        """Return status of all available models."""
        status: dict[str, Any] = {
            "yolo": {"available": False, "status": "not_loaded"},
            "face_detector": {"available": False, "status": "not_loaded"},
            "hand_tracker": {"available": False, "status": "not_loaded"},
            "pose_estimator": {"available": False, "status": "not_loaded"},
            "code_reader": {"available": False, "status": "not_loaded"},
            "modules": {},
        }

        try:
            self._ensure_yolo()
            status["yolo"] = {"available": True, "status": "loaded"}
        except Exception:
            pass

        try:
            fd = self._ensure_face_detector()
            status["face_detector"] = {"available": fd is not None and fd.available, "status": "loaded" if fd and fd.available else "unavailable"}
        except Exception:
            pass

        try:
            ht = self._ensure_hand_tracker()
            status["hand_tracker"] = {"available": ht is not None and ht.available, "status": "loaded" if ht and ht.available else "unavailable"}
        except Exception:
            pass

        try:
            pe = self._ensure_pose_estimator()
            status["pose_estimator"] = {"available": pe is not None and pe.available, "status": "loaded" if pe and pe.available else "unavailable"}
        except Exception:
            pass

        try:
            cr = self._ensure_code_reader()
            status["code_reader"] = {"available": cr is not None, "status": "loaded" if cr else "unavailable"}
        except Exception:
            pass

        # Check toggleable modules
        module_keys = [
            "face_recognition", "face_mask", "emotion", "age_gender",
            "sign_language", "fire_smoke", "helmet", "ppe",
            "vehicle", "anpr", "currency", "animal", "plant_disease",
            "document_scan", "qr_codes",
        ]
        for key in module_keys:
            try:
                mod = self._ensure_module(key)
                if mod is not None:
                    status["modules"][key] = {"available": mod.available, "status": "loaded" if mod.available else "unavailable"}
                else:
                    status["modules"][key] = {"available": False, "status": "failed_to_load"}
            except Exception:
                status["modules"][key] = {"available": False, "status": "error"}

        return status

    def _draw_detections(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        from utils import color_for
        height, width = frame.shape[:2]
        for det in detections:
            try:
                box = det.get("box", [])
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = (int(round(float(v))) for v in box)
                if x2 <= x1 or y2 <= y1:
                    continue
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                label = det.get("label", "")
                color = det.get("color")
                if color is None:
                    color = color_for(label)
                else:
                    color = tuple(color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                caption = f"{label} {det.get('confidence', 0):.0%}"
                (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
                cv2.putText(frame, caption, (x1 + 4, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 20), 1, cv2.LINE_AA)
            except Exception:
                continue
        return frame


# Singleton instance
_pipeline: Optional[VisionPipeline] = None


def get_pipeline() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
    return _pipeline
