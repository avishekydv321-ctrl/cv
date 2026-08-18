"""VisionSuite — background database worker.

High-frequency detection writes are queued here and flushed by a daemon thread
so the PySide6 GUI thread and the vision worker never block on slow PostgreSQL
queries. If PostgreSQL is unavailable, queued writes are dropped gracefully
(no crash) and simply logged at DEBUG level.

Usage:
    from db_worker import db_queue
    db_queue.record_plate(plate_number="HR26AB1234", ...)   # non-blocking
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

import repositories

log = logging.getLogger("db_worker")


class _DBQueue:
    """A singleton FIFO of (callable, args, kwargs) executed on a daemon thread."""

    def __init__(self, maxsize: int = 2000) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = True
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="db-worker", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                task = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                continue
            fn, args, kwargs = task
            try:
                fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - depends on DB
                log.debug("db-worker task failed: %s", exc)
            finally:
                self._q.task_done()

    def _submit(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        try:
            self._q.put_nowait((fn, args, kwargs))
        except queue.Full:  # pragma: no cover - avoid unbounded memory
            log.debug("DB queue full; dropping write.")

    # -- typed convenience wrappers ---------------------------------------
    def record_plate(self, **kwargs: Any) -> None:
        self._submit(repositories.record_plate, **kwargs)

    def record_detection(self, **kwargs: Any) -> None:
        self._submit(repositories.record_detection, **kwargs)

    def record_ocr(self, text: str, confidence: float = 0.0,
                   source: str = "", image_path: str = "") -> None:
        self._submit(repositories.record_ocr, text, confidence, source, image_path)

    def add_face(self, name: str, embedding: Any) -> None:
        self._submit(repositories.add_face_embedding, name, embedding)

    def touch_user(self, name: str) -> None:
        self._submit(repositories.touch_user_seen, name)

    def upsert_user(self, name: str, status: str = "active", image_path: str = "") -> None:
        self._submit(repositories.upsert_user, name, status, image_path)

    def set_setting(self, key: str, value: str) -> None:
        self._submit(repositories.set_setting, key, value)

    def log_event(self, level: str, message: str, logger: str = "") -> None:
        self._submit(repositories.log_system, level, message, logger)

    def flush(self, timeout: float = 3.0) -> None:
        """Best-effort wait for queued tasks (used on shutdown)."""
        try:
            self._q.join()
        except Exception:  # pragma: no cover
            pass

    def shutdown(self) -> None:
        self._running = False
        self.flush(timeout=1.0)


# Module-level singleton.
db_queue = _DBQueue()


__all__ = ["db_queue"]
