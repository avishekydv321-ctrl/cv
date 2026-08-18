"""Frame pipeline: pulls from the camera, runs detectors, emits results."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from analytics import SessionAnalytics
from camera import CameraStream
from config import AppConfig
from detector import (
    CodeAndTextReader,
    FaceDetector,
    HandTracker,
    ObjectDetector,
    PoseEstimator,
    draw_detections,
)
from processing import apply_filter, ImageEnhancer
from tracker import CentroidTracker
from utils import Detection, FPSMeter
from modules import get_module

log = logging.getLogger("app")


@dataclass
class FrameResult:
    """Everything the GUI needs to render one processed frame."""

    frame: np.ndarray
    fps: float
    detections: list[Detection] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    modules: dict[str, float] = field(default_factory=dict)  # module -> inference ms


class VisionWorker(QObject):
    """Runs the whole vision pipeline on its own QThread."""

    frame_ready = Signal(object)   # FrameResult
    status = Signal(str)
    failed = Signal(str)
    alert = Signal(str)            # real-time alert toasts

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._camera: Optional[CameraStream] = None
        self._running = False
        self._fps = FPSMeter()
        self._tracker = CentroidTracker()
        self._objects = ObjectDetector(config.yolo_model)
        self._faces: Optional[FaceDetector] = None
        self._hands: Optional[HandTracker] = None
        self._pose: Optional[PoseEstimator] = None
        self._sign: Optional[Any] = None
        self._reader = CodeAndTextReader()
        self._module_cache: dict[str, Any] = {}
        self.enhancer = ImageEnhancer()
        self.frame_index = 0
        self.csv_rows: list[dict[str, Any]] = []
        # Analytics dashboard + system monitor (CPU/RAM/GPU).
        self.analytics = SessionAnalytics()
        self._fps_limit_prev = 0.0

    # -- AI modules -----------------------------------------------------
    def _module_for(self, key: str):
        """Return a cached singleton instance of the given AI module."""
        if key not in self._module_cache:
            self._module_cache[key] = get_module(key)
        return self._module_cache[key]

    def reset_modules(self) -> None:
        """Release module singletons (memory of loaded graphs)."""
        for module in self._module_cache.values():
            close = getattr(module, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover
                    pass
        self._module_cache.clear()

    # -- lazy back-ends ----------------------------------------------
    # These back-ends may fail to initialise (missing mediapipe model, broken
    # dependency, etc.). We cache the failure so we never retry every frame,
    # and we return None so callers can degrade gracefully instead of crashing
    # the whole pipeline loop.
    @property
    def faces(self) -> Optional[FaceDetector]:
        if self._faces is None:
            try:
                self._faces = FaceDetector()
            except Exception as exc:  # pragma: no cover
                log.warning("FaceDetector unavailable: %s", exc)
                self._faces = None
        return self._faces

    @property
    def hands(self) -> Optional[HandTracker]:
        if self._hands is None:
            try:
                self._hands = HandTracker()
            except Exception as exc:  # pragma: no cover
                log.warning("HandTracker unavailable: %s", exc)
                self._hands = None
        return self._hands

    @property
    def pose(self) -> Optional[PoseEstimator]:
        if self._pose is None:
            try:
                self._pose = PoseEstimator()
            except Exception as exc:  # pragma: no cover
                log.warning("PoseEstimator unavailable: %s", exc)
                self._pose = None
        return self._pose

    @property
    def sign(self) -> Optional[Any]:
        """Lazy Sign Language backend (used when Signs is a base mode)."""
        if self._sign is None:
            try:
                from modules.handsign import SignLanguage

                self._sign = SignLanguage()
            except Exception as exc:  # pragma: no cover
                log.warning("SignLanguage unavailable: %s", exc)
                self._sign = None
        return self._sign

    # -- control -----------------------------------------------------
    def start(self) -> None:
        """Entry point invoked once the worker thread starts."""
        cfg = self.config
        try:
            self._camera = CameraStream(
                index=cfg.camera_index,
                resolution=cfg.resolution,
                source=cfg.source,
                ip_url=cfg.ip_url,
                media_path=cfg.media_path,
            )
            self._camera.start()
        except Exception as exc:
            log.exception("Source start failed")
            self.failed.emit(str(exc))
            return

        self._running = True
        self.status.emit(f"Source live: {cfg.source}")
        self._loop()

    def stop(self) -> None:
        self._running = False
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self._tracker.reset()
        self.reset_modules()
        self.analytics.reset()
        self.status.emit("Camera stopped")

    def set_paused(self, paused: bool) -> None:
        if self._camera is None:
            return
        self._camera.pause() if paused else self._camera.resume()
        self.status.emit("Paused" if paused else "Resumed")

    # -- pipeline ----------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            # Grab the camera reference once per iteration: stop() may set
            # self._camera to None from the GUI thread while we are reading it.
            camera = self._camera
            if camera is None:
                break
            frame = camera.read()
            if frame is None:
                continue
            # Optional semi-fixed frame-rate cap to reduce CPU on fast feeds.
            limit = self.config.fps_limit
            if limit > 0:
                now = __import__("time").perf_counter()
                if self._fps_limit_prev and (now - self._fps_limit_prev) < (1.0 / limit):
                    continue
                self._fps_limit_prev = now
            try:
                self.frame_ready.emit(self._process(frame))
            except Exception as exc:  # never let one bad frame kill the app
                log.exception("Frame processing error")
                self.status.emit(f"Frame error: {exc}")

    def _process(self, frame: np.ndarray) -> FrameResult:
        self.frame_index += 1
        config = self.config

        # Apply low-quality enhancement preprocessors
        frame = self.enhancer.enhance(frame, config)

        if config.mirror:
            frame = cv2.flip(frame, 1)
        if config.image_filter != "None":
            frame = apply_filter(frame, config.image_filter)

        detections: list[Detection] = []
        info: dict[str, Any] = {}
        alerts: list[str] = []
        module_times: dict[str, float] = {}

        if config.detection_enabled:
            # Snapshot the enabled sets upfront: the GUI thread mutates them
            # while the worker loop runs, and iterating a live set that is
            # being modified raises "RuntimeError: Set changed size during
            # iteration". A snapshot keeps the frame cohesive and thread-safe.
            active_modes = list(config.enabled_modes)
            # Run every active base mode. Multiple modes can be toggled at
            # once; their detections and info/metadata are merged below.
            for mode in active_modes:
                if mode == "Objects":
                    self._objects.set_weights(config.yolo_model)
                    track_enabled = getattr(config, "tracking_enabled", True)
                    obj_dets = self._objects.detect(frame, config.confidence, track=track_enabled)
                    detections.extend(obj_dets)
                    counts: dict[str, int] = {}
                    for det in obj_dets:
                        cls_name = det.label.split(" #")[0]
                        counts[cls_name] = counts.get(cls_name, 0) + 1
                    info["counts"] = counts
                    if track_enabled:
                        info["tracks"] = len([d for d in obj_dets if " #" in d.label])
                    else:
                        info["tracks"] = len(self._tracker.update(obj_dets))
                elif mode == "Faces":
                    faces = self.faces
                    if faces is not None:
                        detections.extend(faces.detect(frame))
                    info["faces"] = len(detections) if "faces" not in info else info["faces"]
                elif mode == "Hands":
                    hands = self.hands
                    if hands is not None:
                        info.update(hands.process(frame))
                elif mode == "Pose":
                    pose = self.pose
                    if pose is not None:
                        info.update(pose.process(frame))
                elif mode == "Sign Language":
                    sign = self.sign
                    if sign is not None:
                        result = sign.process(frame, config)
                        detections.extend(result.detections)
                        module_times["sign"] = result.inference_ms
                        if result.info:
                            info["signs"] = result.info.get("signs", "—")
                elif mode == "OCR / Codes":
                    detections.extend(self._reader.read_codes(frame))
                    if self.frame_index % 30 == 0:
                        info["text"] = self._reader.read_text(frame)

        # Run every toggled AI module in parallel with the base mode. Each
        # module merges its detections/info/alerts into the aggregated result.
        # Snapshot the set of enabled modules for thread-safety (see above).
        active_modules = list(config.enabled_modules)
        for key in active_modules:
            try:
                module = self._module_for(key)
                result = module._run(frame, config)
            except Exception as exc:  # a broken module must never stop the loop
                log.exception("Module %s integration error", key)
                continue

            detections.extend(result.detections)
            module_times[key] = result.inference_ms
            alerts.extend(result.alerts)
            if result.info:
                info[key] = result.info

            # Intercept face recognition for attendance mode logging
            if key == "face_recognition" and result.info:
                recognized_names = result.info.get("recognized", [])
                if recognized_names and getattr(config, "attendance_mode", False):
                    import csv
                    import time as tm
                    from utils import OUTPUT_DIR
                    attendance_file = OUTPUT_DIR / "attendance.csv"
                    file_exists = attendance_file.exists()
                    try:
                        with open(attendance_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["Name", "Timestamp", "Camera ID"])
                            for name in recognized_names:
                                writer.writerow([
                                    name,
                                    tm.strftime("%Y-%m-%d %H:%M:%S"),
                                    f"Camera {config.camera_index}"
                                ])
                    except Exception as e:
                        log.warning("Failed to write attendance log: %s", e)

            # Intercept ANPR for database logging and blacklisted vehicle alert check
            if key == "anpr" and "plate" in result.info:
                plate_text = result.info["plate"].strip()
                if plate_text:
                    import os
                    import time as tm
                    from utils import OUTPUT_DIR, timestamp
                    snapshot_dir = OUTPUT_DIR / "snapshots"
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    snapshot_filename = f"plate_{timestamp()}_{plate_text}.jpg"
                    snapshot_filename = "".join(c for c in snapshot_filename if c.isalnum() or c in "._-")
                    snapshot_path = snapshot_dir / snapshot_filename
                    try:
                        cv2.imwrite(str(snapshot_path), frame)
                        # Log to database via face_recognition module's DB instance
                        self._module_for("face_recognition").db.log_plate(
                            plate_text, 0.9, f"Camera {config.camera_index}", str(snapshot_path)
                        )
                    except Exception as e:
                        log.warning("Failed to save plate snapshot: %s", e)

                    # Check blacklist
                    blacklist_str = getattr(config, "blacklist_plates", "")
                    blacklist = [p.strip().upper() for p in blacklist_str.split(",") if p.strip()]
                    if plate_text.upper() in blacklist:
                        alert_msg = f"🚨 BLACKLISTED VEHICLE DETECTED: {plate_text}"
                        result.alerts.append(alert_msg)
                        alerts.append(alert_msg)

            # Surface important alerts as real-time toasts.
            for alert in result.alerts:
                self.alert.emit(alert)

        frame = draw_detections(frame, detections)
        fps = self._fps.tick()

        if config.show_fps:
            cv2.putText(frame, f"FPS {fps:5.1f}", (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 240, 180), 2, cv2.LINE_AA)

        if config.record_csv and detections:
            self.csv_rows.extend(det.as_row(self.frame_index) for det in detections)

        # Feed the analytics dashboard (rolling history + session stats).
        if config.analytics_enabled:
            self.analytics.record(self.frame_index, detections)

        return FrameResult(frame=frame, fps=fps, detections=detections,
                           info=info, alerts=alerts, modules=module_times)
