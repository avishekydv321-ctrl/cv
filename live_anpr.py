"""Background worker for Live ANPR processing.

Runs in a QThread so the UI stays responsive while a live camera feed (USB
webcam, IP / RTSP / CCTV stream) is being analysed for vehicles and license
plates.

It reuses the self-contained :class:`VideoANPREngine` for vehicle detection,
tracking and plate OCR, and the project's :class:`CameraStream` for capture.
Unique plate reads are deduplicated per track ID so a plate is only recorded
once while the same vehicle stays in view, and every unique detection is saved
to ``output/anpr/`` and persisted to the :class:`ANPRStore`.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from anpr_store import ANPRStore
from camera import CameraStream
from modules.anpr_video import VideoANPREngine
from modules.plate_detector import PlateModelError
from utils import OUTPUT_DIR, timestamp

log = logging.getLogger("live_anpr")


@dataclass
class LiveANPRRecord:
    """One unique plate read from a live feed."""

    plate_number: str
    ocr_confidence: float
    vehicle_type: str
    track_id: int
    timestamp_s: float
    source: str
    vehicle_image: str = ""
    plate_image: str = ""
    frame_number: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "Plate Number": self.plate_number,
            "OCR Confidence": f"{self.ocr_confidence:.0%}",
            "Vehicle Type": self.vehicle_type,
            "Track ID": self.track_id,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp_s)),
            "Source": self.source,
            "Frame": self.frame_number,
            "Vehicle Image": self.vehicle_image,
            "Plate Image": self.plate_image,
        }


class LiveANPRWorker(QThread):
    """Process a live camera stream for vehicle detection + plate reading."""

    frame_ready = Signal(object)   # ANPRFrameResult (annotated frame + metadata)
    status = Signal(str)
    result_ready = Signal(object)  # LiveANPRRecord
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        source: str = "webcam",
        camera_index: int = 0,
        ip_url: str = "",
        resolution: tuple[int, int] = (1280, 720),
        confidence: float = 0.35,
        debug: bool = False,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.camera_index = camera_index
        self.ip_url = ip_url
        self.resolution = resolution
        self.confidence = confidence
        self.debug = debug
        self._engine = VideoANPREngine(confidence=confidence, debug=debug)
        self._store = ANPRStore()
        self._camera: Optional[CameraStream] = None
        self._records: list[LiveANPRRecord] = []
        self._seen: dict[int, str] = {}  # track_id -> last recorded plate text
        self._stop = False
        self._lock = threading.Lock()
        self._started_at = time.time()

    # -- control -------------------------------------------------------
    def request_stop(self) -> None:
        with self._lock:
            self._stop = True

    def records(self) -> list[LiveANPRRecord]:
        return list(self._records)

    # -- main loop -----------------------------------------------------
    def run(self) -> None:
        try:
            self._camera = CameraStream(
                index=self.camera_index,
                resolution=self.resolution,
                source=self.source,
                ip_url=self.ip_url,
            )
            self._camera.start()
        except Exception as exc:  # pragma: no cover
            log.exception("Live ANPR source start failed")
            self.error.emit(str(exc))
            return

        source_label = self.ip_url or f"Camera #{self.camera_index}"
        self.status.emit(f"Live ANPR running on {source_label}")

        # Prepare the dedicated license-plate model before frame processing.
        try:
            self._engine.prepare(progress=lambda msg, frac: self.status.emit(msg))
            self.status.emit("License-plate model ready. Processing live feed…")
        except PlateModelError as exc:
            self.error.emit(str(exc))
            if self._camera is not None:
                self._camera.stop()
                self._camera = None
            self._store.close()
            self.finished.emit()
            return

        self._engine.reset()

        veh_dir = OUTPUT_DIR / "anpr" / "vehicles"
        plat_dir = OUTPUT_DIR / "anpr" / "plates"
        veh_dir.mkdir(parents=True, exist_ok=True)
        plat_dir.mkdir(parents=True, exist_ok=True)

        frame_index = 0
        try:
            while not self._stop:
                camera = self._camera
                if camera is None:
                    break
                frame = camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_index += 1
                result = self._engine.process(frame)
                annotated = self._annotate(frame, result)
                result.frame = annotated
                self.frame_ready.emit(result)

                # Record unique plate reads + persist.
                self._collect_records(result, frame_index, frame, veh_dir, plat_dir, source_label)
                time.sleep(0.001)
        except Exception as exc:  # pragma: no cover
            log.exception("Live ANPR worker error")
            self.error.emit(str(exc))
        finally:
            if self._camera is not None:
                self._camera.stop()
                self._camera = None
            self._store.close()
            self.finished.emit()

    # -- annotation ----------------------------------------------------
    def _annotate(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw vehicle boxes, track IDs and plate text on the frame."""
        for v in result.vehicles:
            x1, y1, x2, y2 = v.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 80), 2)
            caption = f"{v.vehicle_type} #{v.track_id}"
            cv2.putText(frame, caption, (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 80), 2, cv2.LINE_AA)
            if v.plate and v.plate.text:
                px1, py1, px2, py2 = v.plate.box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (120, 200, 255), 2)
                cv2.putText(frame, v.plate.text, (px1, max(14, py1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 2, cv2.LINE_AA)
        return frame

    # -- recording -----------------------------------------------------
    def _seen_text(self, track_id: int) -> str:
        with self._lock:
            return self._seen.get(track_id, "")

    def _mark_seen(self, track_id: int, text: str) -> None:
        with self._lock:
            self._seen[track_id] = text

    def _collect_records(self, result, frame_index: int, frame: np.ndarray,
                         veh_dir: Path, plat_dir: Path, source_label: str) -> None:
        """Save unique vehicle/plate crops and persist to the ANPR store."""
        for v in result.vehicles:
            if not (v.plate and v.plate.text):
                continue
            # Deduplicate: skip a plate we've already recorded for this track.
            if self._seen_text(v.track_id) == v.plate.text:
                continue
            self._mark_seen(v.track_id, v.plate.text)

            stamp = timestamp()
            veh_name = f"veh_{stamp}_t{v.track_id}_f{frame_index}.jpg"
            plat_name = f"plate_{stamp}_t{v.track_id}_{v.plate.text}.jpg"
            plat_name = "".join(c for c in plat_name if c.isalnum() or c in "._-")

            veh_path = veh_dir / veh_name
            plat_path = plat_dir / plat_name
            try:
                if v.vehicle_crop is not None:
                    cv2.imwrite(str(veh_path), v.vehicle_crop)
                if v.plate.image is not None:
                    cv2.imwrite(str(plat_path), v.plate.image)
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to save crops: %s", exc)

            record = LiveANPRRecord(
                plate_number=v.plate.text,
                ocr_confidence=v.plate.confidence,
                vehicle_type=v.vehicle_type,
                track_id=v.track_id,
                timestamp_s=time.time(),
                source=source_label,
                vehicle_image=str(veh_path),
                plate_image=str(plat_path),
                frame_number=frame_index,
            )
            self._records.append(record)
            self.result_ready.emit(record)

# Persist to the shared ANPR history store.
            try:
                self._store.add(
                    plate=v.plate.text,
                    confidence=v.plate.confidence,
                    vehicle_type=v.vehicle_type,
                    track_id=v.track_id,
                    vehicle_image=str(veh_path),
                    plate_image=str(plat_path),
                    source=source_label,
                    frame_number=frame_index,
                )
            except Exception as exc:  # pragma: no cover
                log.warning("Failed to persist ANPR record: %s", exc)

            # Persist to LOCAL PostgreSQL (background, duplicate-suppressed).
            try:
                from db_worker import db_queue

                db_queue.record_plate(
                    plate_number=v.plate.text,
                    ocr_confidence=v.plate.confidence,
                    vehicle_type=v.vehicle_type,
                    tracking_id=v.track_id,
                    camera_source=source_label,
                    video_filename=source_label,
                    frame_number=frame_index,
                    vehicle_image_path=str(veh_path),
                    plate_image_path=str(plat_path),
                )
            except Exception as exc:  # pragma: no cover
                log.debug("PostgreSQL plate write skipped: %s", exc)


__all__ = ["LiveANPRWorker", "LiveANPRRecord"]
