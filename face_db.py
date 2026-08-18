"""Persistent face identity database.

Stores face embeddings (128-d vectors) and recognition history in a small
SQLite database under ``./data``. Used by the Face Recognition module for
registration and real-time identification.

Supports **multiple samples per user** (``face_samples`` table) so recognition
stays accurate across different angles, lighting and expressions.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from utils import ROOT

log = logging.getLogger("face_db")

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "faces.db"


class FaceDB:
    """SQLite-backed face registry + recognition history."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding TEXT NOT NULL,   -- JSON list[float] of length 128
                created_at REAL NOT NULL
            )
            """
        )
        # Multi-sample storage: one row per registered face sample, so each
        # user can have many embeddings (different angles / lighting).
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,   -- JSON list[float]
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                confidence REAL NOT NULL,
                recognized_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plate_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                confidence REAL NOT NULL,
                camera_id TEXT NOT NULL,
                recognized_at REAL NOT NULL,
                snapshot_path TEXT
            )
            """
        )
        self._conn.commit()

    # -- registration ---------------------------------------------------
    def register(self, name: str, embedding: list[float]) -> None:
        """Insert or replace an identity embedding (backward-compatible)."""
        payload = json.dumps([float(v) for v in embedding])
        self._conn.execute(
            """
            INSERT INTO identities (name, embedding, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding
            """,
            (name, payload, time.time()),
        )
        # Also store it as a sample so multi-sample matching works.
        self.add_sample(name, embedding)
        self._conn.commit()
        log.info("Registered face identity: %s", name)

    def add_sample(self, name: str, embedding: list[float]) -> None:
        """Append another face sample (embedding) to a user's profile."""
        payload = json.dumps([float(v) for v in embedding])
        self._conn.execute(
            "INSERT INTO face_samples (name, embedding, created_at) VALUES (?, ?, ?)",
            (name, payload, time.time()),
        )
        self._conn.commit()

    def samples(self, name: str) -> list[list[float]]:
        """Return all stored embeddings for one user."""
        rows = self._conn.execute(
            "SELECT embedding FROM face_samples WHERE name=?", (name,)
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["embedding"]))
            except Exception:  # pragma: no cover - corrupt row
                continue
        return result

    def samples_for_all(self) -> dict[str, list[list[float]]]:
        """Return {name: [embedding, ...], ...} across every registered user."""
        rows = self._conn.execute("SELECT name, embedding FROM face_samples").fetchall()
        result: dict[str, list[list[float]]] = {}
        for row in rows:
            try:
                result.setdefault(row["name"], []).append(json.loads(row["embedding"]))
            except Exception:  # pragma: no cover - corrupt row
                continue
        return result

    def remove(self, name: str) -> None:
        self._conn.execute("DELETE FROM identities WHERE name=?", (name,))
        self._conn.execute("DELETE FROM face_samples WHERE name=?", (name,))
        self._conn.execute("DELETE FROM recognition_history WHERE name=?", (name,))
        self._conn.commit()
        log.info("Removed face identity: %s", name)

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an identity across all related tables."""
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        self._conn.execute(
            "UPDATE identities SET name=? WHERE name=?", (new_name, old_name)
        )
        self._conn.execute(
            "UPDATE face_samples SET name=? WHERE name=?", (new_name, old_name)
        )
        self._conn.execute(
            "UPDATE recognition_history SET name=? WHERE name=?", (new_name, old_name)
        )
        self._conn.commit()
        log.info("Renamed identity %s -> %s", old_name, new_name)

    def embeddings(self) -> list[tuple[str, list[float]]]:
        """Legacy single-embedding lookup (first sample per user)."""
        rows = self._conn.execute("SELECT name, embedding FROM identities").fetchall()
        result = []
        for row in rows:
            try:
                result.append((row["name"], json.loads(row["embedding"])))
            except Exception:  # pragma: no cover - corrupt row
                continue
        return result

    def names(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM identities ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    # -- history ---------------------------------------------------------
    def log_recognition(self, name: str, confidence: float) -> None:
        self._conn.execute(
            "INSERT INTO recognition_history (name, confidence, recognized_at) VALUES (?, ?, ?)",
            (name, float(confidence), time.time()),
        )
        self._conn.commit()

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT name, confidence, recognized_at FROM recognition_history "
            "ORDER BY recognized_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- plate history ---------------------------------------------------
    def log_plate(self, plate: str, confidence: float, camera_id: str, snapshot_path: str = "") -> None:
        self._conn.execute(
            """
            INSERT INTO plate_log (plate, confidence, camera_id, recognized_at, snapshot_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (plate, float(confidence), camera_id, time.time(), snapshot_path),
        )
        self._conn.commit()

    def get_plate_history(self, search_query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        if search_query:
            rows = self._conn.execute(
                """
                SELECT id, plate, confidence, camera_id, recognized_at, snapshot_path 
                FROM plate_log 
                WHERE plate LIKE ? 
                ORDER BY recognized_at DESC LIMIT ?
                """,
                (f"%{search_query}%", limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, plate, confidence, camera_id, recognized_at, snapshot_path 
                FROM plate_log 
                ORDER BY recognized_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_plate_record(self, record_id: int) -> None:
        self._conn.execute("DELETE FROM plate_log WHERE id=?", (record_id,))
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover
            pass
