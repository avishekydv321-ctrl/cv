"""Background worker for Video ANPR processing.

Runs in a QThread so the UI stays responsive while a video is being analysed.
Exposes thread-safe controls (pause / resume / stop / seek / frame-step) and
emits signals for frame previews, progress, status, results and errors.

Every detected vehicle yields an ``ANPRRecord`` capturing:
  * vehicle image (saved to output/anpr/vehicles/)
  * cropped license plate image (output/anpr/plates/)
  * plate number, OCR confidence
  * vehicle type, tracking ID
  * timestamp in the video and frame number
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
from modules.anpr_video import VideoANPREngine
from modules.plate_detector import PlateModelError
from utils import OUTPUT_DIR, timestamp

log = logging.getLogger("video_anpr")

SUPPORTED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")


@dataclass
class ANPRRecord:
    """One saved result row for a detected vehicle."""

    plate_number: str
    ocr_confidence: float
    vehicle_type: str
    track_id: int
    timestamp_s: float
    frame_number: int
    vehicle_image: str = ""   # path
    plate_image: str = ""     # path

    def as_dict(self) -> dict[str, Any]:
        def fmt(ts: float) -> str:
            m, s = divmod(int(ts), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        return {
            "Plate Number": self.plate_number,
            "OCR Confidence": f"{self.ocr_confidence:.0%}",
            "Vehicle Type": self.vehicle_type,
            "Track ID": self.track_id,
            "Timestamp": fmt(self.timestamp_s),
            "Frame": self.frame_number,
            "Vehicle Image": self.vehicle_image,
            "Plate Image": self.plate_image,
        }


class ANPRVideoWorker(QThread):
    """Process a video file for vehicle detection + license plate reading."""

    frame_ready = Signal(object)      # ANPRFrameResult (annotated frame + metadata)
    progress = Signal(int, int)       # current_frame, total_frames
    status = Signal(str)
    result_ready = Signal(object)     # ANPRRecord
    finished = Signal()
    error = Signal(str)

    def __init__(self, video_path: str, confidence: float = 0.35, debug: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.confidence = confidence
        self.debug = debug
        self._engine = VideoANPREngine(confidence=confidence, debug=debug)
        self._store = ANPRStore()
        self._records: list[ANPRRecord] = []
        self._paused = False
        self._stop = False
        self._step = False
        self._seek_frame: Optional[int] = None
        self._lock = threading.Lock()
        self._last_saved: dict[int, str] = {}  # track_id -> last plate text
        self._seen: dict[int, str] = {}        # track_id -> last recorded plate text

    # -- control -------------------------------------------------------
    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def request_stop(self) -> None:
        with self._lock:
            self._stop = True
            self._paused = False

    def step_frame(self) -> None:
        """Advance one frame while paused."""
        with self._lock:
            self._step = True
            self._paused = False

    def seek(self, frame_number: int) -> None:
        with self._lock:
            self._seek_frame = frame_number

    def set_confidence(self, conf: float) -> None:
        self.confidence = conf
        self._engine.confidence = conf

    def records(self) -> list[ANPRRecord]:
        return list(self._records)

    # -- main loop -----------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Could not open video: {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.status.emit(f"Processing {Path(self.video_path).name} — {total_frames} frames")

        # Ensure output directories.
        veh_dir = OUTPUT_DIR / "anpr" / "vehicles"
        plat_dir = OUTPUT_DIR / "anpr" / "plates"
        veh_dir.mkdir(parents=True, exist_ok=True)
        plat_dir.mkdir(parents=True, exist_ok=True)

        self._engine.reset()
        frame_index = 0
        t0 = time.perf_counter()

        # Prepare the dedicated license-plate model before frame processing.
        try:
            self._engine.prepare(progress=lambda msg, frac: self.status.emit(msg))
            self.status.emit("License-plate model ready. Processing video…")
        except PlateModelError as exc:
            self.error.emit(str(exc))
            cap.release()
            self._store.close()
            self.finished.emit()
            return

        try:
            while not self._stop:
                # --- handle controls ---------------------------------
                with self._lock:
                    pause = self._paused
                    step = self._step
                    self._step = False
                    seek = self._seek_frame
                    self._seek_frame = None
                if seek is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                    self.progress.emit(seek, total_frames)
                if pause and not step:
                    time.sleep(0.02)
                    continue

                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                result = self._engine.process(frame)
                # Annotate the frame with vehicle boxes + plate text.
                annotated = self._annotate(frame, result)
                result.frame = annotated

                self.frame_ready.emit(result)
                self.progress.emit(frame_index, total_frames)

                # Save records for vehicles with a readable plate.
                self._collect_records(result, frame_index, frame, veh_dir, plat_dir, fps)

                # Throttle to roughly real-time when not stepping.
                if step:
                    with self._lock:
                        self._paused = True  # pause again after a single step
                else:
                    time.sleep(0.001)

            elapsed = time.perf_counter() - t0
            self.status.emit(f"Finished in {elapsed:.1f}s — {len(self._records)} plate reads")
            self.frame_ready.emit(result_placeholder(fps))
        except Exception as exc:  # pragma: no cover
            log.exception("Video ANPR worker error")
            self.error.emit(str(exc))
        finally:
            cap.release()
            self._store.close()
            self.finished.emit()

    def _annotate(self, frame: np.ndarray, result) -> np.ndarray:
        """Draw vehicle boxes + plate text on the frame."""
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

    def _collect_records(self, result, frame_index: int, frame: np.ndarray,
                         veh_dir: Path, plat_dir: Path, fps: float,
                         source: str = "") -> None:
        """Save unique vehicle/plate crops, emit an ANPRRecord, persist to store."""
        ts = frame_index / (fps or 25.0)
        for v in result.vehicles:
            if not (v.plate and v.plate.text):
                continue
            # Deduplicate: skip a plate we've already recorded for this track.
            if self._seen.get(v.track_id) == v.plate.text:
                continue
            self._seen[v.track_id] = v.plate.text
            self._last_saved[v.track_id] = v.plate.text

            stamp = timestamp()
            veh_name = f"veh_{stamp}_t{v.track_id}_f{frame_index}.jpg"
            plat_name = f"plate_{stamp}_t{v.track_id}_{v.plate.text}.jpg"
            # Sanitise filename.
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

            record = ANPRRecord(
                plate_number=v.plate.text,
                ocr_confidence=v.plate.confidence,
                vehicle_type=v.vehicle_type,
                track_id=v.track_id,
                timestamp_s=ts,
                frame_number=frame_index,
                vehicle_image=str(veh_path),
                plate_image=str(plat_path),
            )
            self._records.append(record)
            self.result_ready.emit(record)

            # Persist to the shared ANPR history store (live data).
            try:
                self._store.add(
                    plate=v.plate.text,
                    confidence=v.plate.confidence,
                    vehicle_type=v.vehicle_type,
                    track_id=v.track_id,
                    vehicle_image=str(veh_path),
                    plate_image=str(plat_path),
                    source=source or Path(self.video_path).name,
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
                    camera_source=source or Path(self.video_path).name,
                    video_filename=Path(self.video_path).name,
                    frame_number=frame_index,
                    vehicle_image_path=str(veh_path),
                    plate_image_path=str(plat_path),
                )
            except Exception as exc:  # pragma: no cover
                log.debug("PostgreSQL plate write skipped: %s", exc)


def result_placeholder(fps: float):
    """Build an empty ANPRFrameResult for the final 'finished' signal."""
    from modules.anpr_video import ANPRFrameResult

    return ANPRFrameResult(frame=np.zeros((4, 4, 3), dtype=np.uint8), fps=fps,
                           vehicles=[], vehicle_count=0, plate_count=0)

