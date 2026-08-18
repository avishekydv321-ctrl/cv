"""VisionSuite — migrate existing local data (SQLite/JSON/CSV) into PostgreSQL.

This intentionally NON-DESTRUCTIVE migration reads from the app's existing
SQLite stores (``data/anpr.db``, ``data/faces.db``) and any CSVs under
``output/`` and inserts the records into the local PostgreSQL database.

The original SQLite/CSV files are left untouched. Only after a successful
migration are you free to archive/delete them manually.

Usage:
    python3 migrate_data.py            # migrate everything
    python3 migrate_data.py --dry-run  # show what would be migrated
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import database
import repositories
from utils import ROOT, configure_logging

log = logging.getLogger("migrate_data")

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
SQLITE_ANPR = DATA_DIR / "anpr.db"
SQLITE_FACES = DATA_DIR / "faces.db"


def _utc_from_ts(ts) -> datetime:
    """Convert a Unix epoch (or ISO string) into a UTC datetime."""
    if ts is None:
        return datetime.utcnow()
    if isinstance(ts, (int, float)):
        return datetime.utcfromtimestamp(float(ts))
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return datetime.utcnow()


def migrate_anpr(dry: bool = False) -> int:
    """Migrate rows from data/anpr.db -> license_plates. Returns count."""
    if not SQLITE_ANPR.exists():
        log.info("No %s found; skipping ANPR migration.", SQLITE_ANPR.name)
        return 0
    conn = sqlite3.connect(str(SQLITE_ANPR))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT plate, confidence, vehicle_type, track_id, vehicle_image, "
            "plate_image, recognized_at, source, frame_number FROM anpr_history "
            "ORDER BY recognized_at"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("Could not read anpr_history: %s", exc)
        return 0
    finally:
        conn.close()

    if dry:
        log.info("[DRY] Would migrate %d ANPR rows.", len(rows))
        return len(rows)

    migrated = 0
    for r in rows:
        plate = (r["plate"] or "").strip()
        if not plate:
            continue
        recipients = (r["source"] or "").split("/")
        video = recipients[-1] if recipients else ""
        ok = repositories.record_plate(
            plate_number=plate,
            ocr_confidence=float(r["confidence"] or 0.0),
            vehicle_type=r["vehicle_type"] or "",
            tracking_id=int(r["track_id"] or 0),
            camera_source=r["source"] or "",
            video_filename=video,
            frame_number=int(r["frame_number"] or 0),
            vehicle_image_path=r["vehicle_image"] or "",
            plate_image_path=r["plate_image"] or "",
        )
        if ok is not None:
            migrated += 1
    log.info("Migrated %d ANPR rows -> license_plates.", migrated)
    return migrated


def migrate_faces(dry: bool = False) -> int:
    """Migrate rows from data/faces.db -> users + face_embeddings. Returns count."""
    if not SQLITE_FACES.exists():
        log.info("No %s found; skipping face migration.", SQLITE_FACES.name)
        return 0
    conn = sqlite3.connect(str(SQLITE_FACES))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT name, embedding, created_at FROM identities ORDER BY created_at"
        ).fetchall()
        samples = conn.execute(
            "SELECT name, embedding, created_at FROM face_samples ORDER BY created_at"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("Could not read faces.db: %s", exc)
        return 0
    finally:
        conn.close()

    if dry:
        log.info("[DRY] Would migrate %d identities / %d face samples.",
                 len(rows), len(samples))
        return len(rows)

    migrated = 0
    for r in rows:
        name = (r["name"] or "").strip()
        if not name:
            continue
        if repositories.upsert_user(name) is not None:
            migrated += 1
    for r in samples:
        name = (r["name"] or "").strip()
        try:
            emb = json.loads(r["embedding"])
        except Exception:
            continue
        if repositories.add_face_embedding(name, emb) is not None:
            migrated += 1
    log.info("Migrated %d face identities/samples -> users/face_embeddings.", migrated)
    return migrated


def migrate_csvs(dry: bool = False) -> int:
    """Migrate detection CSVs under output/ -> detections table. Returns count."""
    migrated = 0
    files = list(OUTPUT_DIR.glob("detections-*.csv")) if OUTPUT_DIR.exists() else []
    files += list(OUTPUT_DIR.glob("*.csv"))
    files = list({p for p in files})
    if dry:
        log.info("[DRY] Would scan %d CSV file(s) in output/.", len(files))
        return migrated
    for path in files:
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = (row.get("label") or "").strip()
                    if not label:
                        continue
                    obj_name = label.split(" #")[0]
                    track_id = 0
                    if " #" in label:
                        try:
                            track_id = int(label.split(" #")[1])
                        except ValueError:
                            track_id = 0
                    ok = repositories.record_detection(
                        detection_type="object",
                        object_class=obj_name,
                        confidence=float(row.get("confidence") or 0.0),
                        source=row.get("source") or path.name,
                        tracking_id=track_id,
                    )
                    if ok is not None:
                        migrated += 1
        except Exception as exc:  # pragma: no cover
            log.warning("CSV migration failed for %s: %s", path.name, exc)
    log.info("Migrated %d detection rows from CSVs -> detections.", migrated)
    return migrated


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate VisionSuite local data to PostgreSQL.")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be migrated.")
    args = parser.parse_args()
    configure_logging(logging.INFO)

    st = database.get_pg_status()
    if not st.get("connected"):
        print("\n⚠️  Local PostgreSQL is unavailable.")
        print("    Please start PostgreSQL and set up your .env first.")
        print("    See SETUP_POSTGRES.md for instructions.\n")
        print("    If PostgreSQL IS running on localhost:5432, check POSTGRES_* env vars.\n")
        return 2

    database.init_db()
    dry = args.dry_run

    anpr_count = migrate_anpr(dry)
    face_count = migrate_faces(dry)
    csv_count = migrate_csvs(dry)

    print("\n=== Migration Summary ===")
    print(f"  ANPR rows migrated : {anpr_count}")
    print(f"  Face records       : {face_count}")
    print(f"  Detection rows     : {csv_count}")

    if dry:
        print("\n(Dry-run — no data was written. Original files untouched.)")
    else:
        print("\nMigration complete. Original SQLite/CSV files were NOT deleted.")
        print("You may back them up/archive them manually when you are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
