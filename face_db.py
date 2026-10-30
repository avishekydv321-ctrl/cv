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
import math
import sqlite3
import threading
import time
import uuid
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
        self._lock = threading.RLock()
        try:
            path.chmod(0o600)
        except OSError:
            log.warning("Could not restrict permissions on face database %s", path)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding TEXT NOT NULL,   -- JSON list[float] of length 128
                created_at REAL NOT NULL,
                unique_id TEXT,
                embedding_model TEXT NOT NULL DEFAULT 'legacy'
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
                created_at REAL NOT NULL,
                embedding_model TEXT NOT NULL DEFAULT 'legacy'
            )
            """
        )
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(identities)")}
        if "unique_id" not in columns:
            self._conn.execute("ALTER TABLE identities ADD COLUMN unique_id TEXT")
        if "embedding_model" not in columns:
            self._conn.execute(
                "ALTER TABLE identities ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'legacy'"
            )
        sample_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(face_samples)")
        }
        if "embedding_model" not in sample_columns:
            self._conn.execute(
                "ALTER TABLE face_samples ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'legacy'"
            )
        for row in self._conn.execute(
            "SELECT id FROM identities WHERE unique_id IS NULL OR unique_id=''"
        ).fetchall():
            self._conn.execute(
                "UPDATE identities SET unique_id=? WHERE id=?",
                (f"legacy-{row['id']}", row["id"]),
            )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS identities_unique_id_idx ON identities(unique_id)"
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
        """Insert a legacy identity, retaining an existing ID when available."""
        row = self._conn.execute(
            "SELECT unique_id, embedding_model FROM identities WHERE name=?", (name.strip(),)
        ).fetchone()
        if row and row["embedding_model"] != "legacy":
            raise ValueError("Use model-aware registration for this identity to avoid mixing embeddings.")
        unique_id = row["unique_id"] if row else f"legacy-{uuid.uuid4().hex}"
        embedding_model = row["embedding_model"] if row else "legacy"
        self.register_person(name, unique_id, [embedding], embedding_model)

    def register_person(
        self,
        name: str,
        unique_id: str,
        embeddings: list[list[float]],
        embedding_model: str = "legacy",
    ) -> None:
        """Create or append samples for one person, enforcing unique ID and name."""
        name, unique_id = name.strip(), unique_id.strip()
        if not name or not unique_id or not embeddings:
            raise ValueError("Name, unique ID, and at least one face sample are required.")
        vectors = []
        dimension = None
        for embedding in embeddings:
            vector = [float(value) for value in embedding]
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError("Face embeddings must contain finite values.")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("All face samples must use the same embedding model.")
            vectors.append(vector)

        with self._lock:
            by_name = self._conn.execute(
                "SELECT unique_id FROM identities WHERE name=?", (name,)
            ).fetchone()
            by_id = self._conn.execute(
                "SELECT name FROM identities WHERE unique_id=?", (unique_id,)
            ).fetchone()
            if by_name and by_name["unique_id"] != unique_id:
                raise ValueError("That name is already registered with a different unique ID.")
            if by_id and by_id["name"] != name:
                raise ValueError("That unique ID is already assigned to another person.")

            now = time.time()
            self._conn.execute(
                     """INSERT INTO identities
                         (name, embedding, created_at, unique_id, embedding_model)
                         VALUES (?, ?, ?, ?, ?)
                         ON CONFLICT(name) DO UPDATE SET
                         embedding=excluded.embedding, embedding_model=excluded.embedding_model""",
                     (name, json.dumps(vectors[0]), now, unique_id, embedding_model),
            )
            self._conn.executemany(
                     """INSERT INTO face_samples (name, embedding, created_at, embedding_model)
                         VALUES (?, ?, ?, ?)""",
                     [(name, json.dumps(vector), now, embedding_model) for vector in vectors],
            )
            self._conn.commit()
        log.info("Registered face identity %s (%s) with %d samples", name, unique_id, len(vectors))

    def add_sample(self, name: str, embedding: list[float]) -> None:
        """Append another face sample (embedding) to a user's profile."""
        with self._lock:
            row = self._conn.execute(
                "SELECT unique_id, embedding_model FROM identities WHERE name=?", (name,)
            ).fetchone()
            if row and row["embedding_model"] != "legacy":
                raise ValueError("Use model-aware registration for this identity to avoid mixing embeddings.")
            unique_id = row["unique_id"] if row else f"legacy-{uuid.uuid4().hex}"
            embedding_model = row["embedding_model"] if row else "legacy"
            self.register_person(name, unique_id, [embedding], embedding_model)

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
        with self._lock:
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

    def identity_records(self, search: str = "") -> list[dict[str, Any]]:
        """Return registered people and sample counts, optionally filtered."""
        query = f"%{search.strip()}%"
        rows = self._conn.execute(
            """SELECT i.name, i.unique_id, COUNT(s.id) AS sample_count
               FROM identities AS i LEFT JOIN face_samples AS s ON s.name=i.name
               WHERE i.name LIKE ? OR i.unique_id LIKE ?
               GROUP BY i.id ORDER BY i.name""",
            (query, query),
        ).fetchall()
        return [dict(row) for row in rows]

    def samples_with_ids(
        self, embedding_model: str
    ) -> list[tuple[str, str, list[float]]]:
        """Return all dimension-agnostic samples with their identity metadata."""
        rows = self._conn.execute(
            """SELECT i.name, i.unique_id, s.embedding
                    FROM face_samples AS s JOIN identities AS i ON i.name=s.name
                    WHERE s.embedding_model=? AND i.embedding_model=?""",
                (embedding_model, embedding_model),
        ).fetchall()
        result = []
        for row in rows:
            try:
                result.append((row["name"], row["unique_id"], json.loads(row["embedding"])))
            except (TypeError, ValueError):
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
