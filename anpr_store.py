"""Persistent store for ANPR (Number Plate Recognition) history.

A dedicated SQLite database (``data/anpr.db``) that stores every unique plate
read across both **Live ANPR** and **Video ANPR**, so the user has a single
searchable, exportable history of all recognitions.

The schema holds every field the results table / exports need:
  * plate number, OCR confidence
  * vehicle type, tracking ID
  * vehicle snapshot path, cropped plate image path
  * date & time, source (camera id / IP url / video filename)
  * video frame number

Thread-safety: the worker threads write (``add``) and the GUI thread reads
(``all`` / ``search``). SQLite serialises writes; we use a small lock so a
single connection is shared safely across threads.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from utils import ROOT

log = logging.getLogger("anpr_store")

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "anpr.db"


class ANPRStore:
    """SQLite-backed history of license plate recognitions."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anpr_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    vehicle_type TEXT NOT NULL DEFAULT '',
                    track_id INTEGER NOT NULL DEFAULT 0,
                    vehicle_image TEXT NOT NULL DEFAULT '',
                    plate_image TEXT NOT NULL DEFAULT '',
                    recognized_at REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    frame_number INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anpr_plate ON anpr_history(plate)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_anpr_time ON anpr_history(recognized_at)"
            )
            self._conn.commit()

    # -- write ---------------------------------------------------------
    def add(
        self,
        plate: str,
        confidence: float,
        vehicle_type: str = "",
        track_id: int = 0,
        vehicle_image: str = "",
        plate_image: str = "",
        source: str = "",
        frame_number: int = 0,
    ) -> int:
        """Insert one recognition row and return its id."""
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO anpr_history
                    (plate, confidence, vehicle_type, track_id, vehicle_image,
                     plate_image, recognized_at, source, frame_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plate,
                    float(confidence),
                    vehicle_type,
                    int(track_id),
                    vehicle_image,
                    plate_image,
                    time.time(),
                    source,
                    int(frame_number),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # -- read ----------------------------------------------------------
    def all(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return the most recent recognitions."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM anpr_history
                ORDER BY recognized_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 500) -> list[dict[str, Any]]:
        """Return recognitions matching a free-text query."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM anpr_history
                WHERE plate LIKE ? OR vehicle_type LIKE ? OR source LIKE ?

                ORDER BY recognized_at DESC LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM anpr_history"
            ).fetchone()
        return int(row["n"]) if row else 0

    def clear(self) -> None:
        """Delete all history rows."""
        with self._lock:
            self._conn.execute("DELETE FROM anpr_history")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover
                pass


__all__ = ["ANPRStore", "DB_PATH"]
