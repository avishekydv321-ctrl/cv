"""Camera and media source enumeration + threaded frame grabbing.

Supports multiple source types without restarting the pipeline:
  * webcam (device index)
  * IP / RTSP stream (URL)
  * video file (local path)
  * image file (returned repeatedly as a static frame)

The ``CameraStream`` grabber always exposes the freshest frame, decoupling
capture from processing so the UI stays smooth even when the detector is
slower than the source's native rate.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("camera")


def list_cameras(max_index: int = 6) -> list[int]:
    """Probe device indices and return those that yield a frame."""
    found: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(index)
        finally:
            cap.release()
    log.info("Available cameras: %s", found or "none")
    return found


def _open_source(source: str, index: int, url: str, media_path: str) -> Optional[cv2.VideoCapture]:
    """Open a :class:`cv2.VideoCapture` for the requested source type.

    Returns ``None`` when the source cannot be opened.
    """
    try:
        if source == "ip":
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        elif source == "video_file":
            cap = cv2.VideoCapture(media_path, cv2.CAP_FFMPEG)
        else:  # webcam (index)
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if not cap.isOpened():
            cap.release()
            return None
        return cap
    except Exception as exc:  # pragma: no cover - depends on backend
        log.warning("Could not open source %s: %s", source, exc)
        return None


class CameraStream:
    """Background grabber over any supported media source."""

    def __init__(
        self,
        index: int = 0,
        resolution: tuple[int, int] = (1280, 720),
        source: str = "webcam",
        ip_url: str = "",
        media_path: str = "",
    ) -> None:
        self.index = index
        self.resolution = resolution
        self.source = source
        self.ip_url = ip_url
        self.media_path = media_path
        self.speed = 1.0
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._image: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._paused = threading.Event()

    # -- lifecycle ---------------------------------------------------
    def start(self) -> None:
        if self._running.is_set():
            return

        # Image source: load once and reuse (no capture loop needed).
        if self.source == "image":
            image = cv2.imread(self.media_path)
            if image is None:
                raise RuntimeError(f"Could not read image: {self.media_path}")
            self._image = image
            with self._lock:
                self._frame = image.copy()
            self._running.set()
            log.info("Image source active: %s", self.media_path)
            return

        self._cap = _open_source(self.source, self.index, self.ip_url, self.media_path)
        if self._cap is None:
            label = self.ip_url or self.media_path or f"#{self.index}"
            raise RuntimeError(f"Could not open source: {self.source} ({label})")

        if self.source == "webcam":
            width, height = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()
        log.info("Source started: %s (%s)", self.source, self.ip_url or self.media_path or f"#{self.index}")

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None
        self._image = None
        log.info("Source stopped: %s", self.source)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    @property
    def running(self) -> bool:
        return self._running.is_set()

    # -- frames ------------------------------------------------------
    def _loop(self) -> None:
        while self._running.is_set() and self._cap is not None:
            if self._paused.is_set():
                time.sleep(0.02)
                continue

            t_start = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                # End of video file: loop back to the start automatically.
                if self.source == "video_file":
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                log.warning("Dropped frame from source %s", self.source)
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame

            # Throttle loop to match video framerate adjusted by playback speed
            if self.source == "video_file":
                fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
                delay = 1.0 / (fps * max(0.1, self.speed))
                elapsed = time.perf_counter() - t_start
                if elapsed < delay:
                    time.sleep(delay - elapsed)

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame, or None if not available."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame number in video source."""
        if self._cap is not None and self.source == "video_file":
            with self._lock:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = self._cap.read()
                if ok:
                    self._frame = frame

    def step_frame(self, forward: bool = True) -> None:
        """Step one frame forward or backward in video source."""
        if self._cap is not None and self.source == "video_file":
            with self._lock:
                curr = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                target = curr + 1 if forward else max(0, curr - 2)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = self._cap.read()
                if ok:
                    self._frame = frame

    def get_video_stats(self) -> dict[str, Any]:
        """Return playback progress stats for video sources."""
        if self._cap is not None and self.source == "video_file":
            curr = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
            return {
                "current_frame": curr,
                "total_frames": total,
                "fps": fps,
                "duration_s": total / fps if fps > 0 else 0.0,
                "position_s": curr / fps if fps > 0 else 0.0,
            }
        return {}

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier."""
        self.speed = speed

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
