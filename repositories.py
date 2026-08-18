"""VisionSuite — PostgreSQL repositories (data access layer).

Every function here is safe to call ONLY from background/worker threads (not
the Qt GUI thread). Each function opens its own short-lived session, so it is
safe to call concurrently from multiple workers.

High-frequency writes (e.g. ANPR detections) are de-duplicated before they hit
the database so we never write the same plate hundreds of times per second.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, or_, select

import database
from database import get_session, is_available
from models import (
    AppSetting, Detection, FaceEmbedding, LicensePlate, OCRResult,
    SessionRecord, SystemLog, User,
)

log = logging.getLogger("repositories")

# ANPR duplicate-suppression window (seconds). A plate for the same tracking ID
# seen again within this window only bumps its recognition_count.
PLATE_DUP_WINDOW_SECONDS = 10.0


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------#
# Generic availability guard.
# ---------------------------------------------------------------------------#
def _guard() -> Optional[bool]:
    """Return True if DB is available; otherwise log and return None (callers stop)."""
    if not is_available():
        log.debug("PostgreSQL unavailable; skipping DB write.")
        return None
    return True


# ---------------------------------------------------------------------------#
# Users / Face embeddings
# ---------------------------------------------------------------------------#
def upsert_user(name: str, status: str = "active", image_path: str = "") -> Optional[int]:
    """Insert or update a user by name. Returns user id or None."""
    if _guard() is None:
        return None
    try:
        with get_session() as s:
            user = s.execute(select(User).where(User.name == name)).scalar_one_or_none()
            if user is None:
                user = User(name=name, status=status, image_path=image_path)
                s.add(user)
                s.commit()
                s.refresh(user)
            else:
                user.status = status
                if image_path:
                    user.image_path = image_path
                s.commit()
            return user.id
    except Exception as exc:  # pragma: no cover
        log.warning("upsert_user failed for %s: %s", name, exc)
        return None


def add_face_embedding(name: str, embedding: list[float]) -> Optional[int]:
    """Add a face sample for a user. Returns embedding id or None."""
    if _guard() is None:
        return None
    try:
        uid = upsert_user(name)
        if uid is None:
            return None
        with get_session() as s:
            emb = FaceEmbedding(user_id=uid, embedding=json.dumps([float(v) for v in embedding]))
            s.add(emb)
            s.commit()
            s.refresh(emb)
            return emb.id
    except Exception as exc:  # pragma: no cover
        log.warning("add_face_embedding failed for %s: %s", name, exc)
        return None


def get_user_embeddings() -> list[dict[str, Any]]:
    """Return all users with their embeddings (for matching)."""
    if _guard() is None:
        return []
    try:
        with get_session() as s:
            rows = s.execute(
                select(User, FaceEmbedding).join(FaceEmbedding, FaceEmbedding.user_id == User.id)
            ).all()
            out: dict[str, list[list[float]]] = {}
            for user, emb in rows:
                try:
                    out.setdefault(user.name, []).append(json.loads(emb.embedding))
                except Exception:
                    continue
            return [{"name": n, "embeddings": e} for n, e in out.items()]
    except Exception as exc:  # pragma: no cover
        log.warning("get_user_embeddings failed: %s", exc)
        return []


def touch_user_seen(name: str) -> None:
    """Increment recognition count and set last_seen for a user."""
    if _guard() is None:
        return
    try:
        with get_session() as s:
            user = s.execute(select(User).where(User.name == name)).scalar_one_or_none()
            if user is not None:
                user.last_seen_at = _now()
                user.recognition_count = (user.recognition_count or 0) + 1
                s.commit()
    except Exception as exc:  # pragma: no cover
        log.debug("touch_user_seen failed: %s", exc)


def delete_user(name: str) -> bool:
    if _guard() is None:
        return False
    try:
        with get_session() as s:
            user = s.execute(select(User).where(User.name == name)).scalar_one_or_none()
            if user is not None:
                s.delete(user)   # cascades face_embeddings
                s.commit()
                return True
        return False
    except Exception as exc:  # pragma: no cover
        log.warning("delete_user failed: %s", exc)
        return False


def list_users(search: str = "") -> list[dict[str, Any]]:
    if _guard() is None:
        return []
    try:
        with get_session() as s:
            q = select(User)
            if search:
                q = q.where(User.name.ilike(f"%{search}%"))
            q = q.order_by(User.name)
            rows = s.execute(q).scalars().all()
            return [
                {
                    "id": u.id, "name": u.name, "status": u.status,
                    "image_path": u.image_path,
                    "created_at": u.created_at, "last_seen_at": u.last_seen_at,
                    "recognition_count": u.recognition_count or 0,
                }
                for u in rows
            ]
    except Exception as exc:  # pragma: no cover
        log.warning("list_users failed: %s", exc)
        return []


# ---------------------------------------------------------------------------#
# License plates (ANPR) with duplicate suppression
# ---------------------------------------------------------------------------#
def record_plate(
    plate_number: str,
    ocr_confidence: float = 0.0,
    vehicle_type: str = "",
    tracking_id: int = 0,
    camera_source: str = "",
    video_filename: str = "",
    frame_number: int = 0,
    vehicle_image_path: str = "",
    plate_image_path: str = "",
) -> Optional[int]:
    """Insert a plate recognition, suppressing near-duplicates.

    If the same ``plate_number`` + ``tracking_id`` was recorded within
    ``PLATE_DUP_WINDOW_SECONDS``, we increment its recognition_count instead of
    inserting a new row. Returns the row id or None.
    """
    if _guard() is None:
        return None
    try:
        cutoff = _now() - timedelta(seconds=PLATE_DUP_WINDOW_SECONDS)
        with get_session() as s:
            existing = s.execute(
                select(LicensePlate).where(
                    LicensePlate.plate_number == plate_number,
                    LicensePlate.tracking_id == int(tracking_id),
                    LicensePlate.detected_at >= cutoff,
                ).order_by(LicensePlate.detected_at.desc())
            ).scalars().first()
            if existing is not None:
                existing.recognition_count = (existing.recognition_count or 1) + 1
                existing.ocr_confidence = max(existing.ocr_confidence or 0.0, float(ocr_confidence))
                s.commit()
                return existing.id
            plate = LicensePlate(
                plate_number=plate_number,
                ocr_confidence=float(ocr_confidence),
                vehicle_type=vehicle_type,
                tracking_id=int(tracking_id),
                camera_source=camera_source,
                video_filename=video_filename,
                frame_number=int(frame_number),
                vehicle_image_path=vehicle_image_path,
                plate_image_path=plate_image_path,
                recognition_count=1,
            )
            s.add(plate)
            s.commit()
            s.refresh(plate)
            return plate.id
    except Exception as exc:  # pragma: no cover
        log.warning("record_plate failed for %s: %s", plate_number, exc)
        return None


def search_plates(
    query: str = "",
    camera: str = "",
    vehicle_type: str = "",
    min_confidence: float = 0.0,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    newest_first: bool = True,
    limit: int = 500,
) -> list[dict[str, Any]]:
    if _guard() is None:
        return []
    try:
        with get_session() as s:
            q = select(LicensePlate)
            if query:
                q = q.where(LicensePlate.plate_number.ilike(f"%{query}%"))
            if camera:
                q = q.where(LicensePlate.camera_source.ilike(f"%{camera}%"))
            if vehicle_type:
                q = q.where(LicensePlate.vehicle_type.ilike(f"%{vehicle_type}%"))
            if min_confidence > 0:
                q = q.where(LicensePlate.ocr_confidence >= min_confidence)
            if date_from:
                q = q.where(LicensePlate.detected_at >= date_from)
            if date_to:
                q = q.where(LicensePlate.detected_at <= date_to)
            q = q.order_by(LicensePlate.detected_at.desc() if newest_first else LicensePlate.detected_at.asc())
            q = q.limit(limit)
            return [dict(r.__dict__) | {"id": r.id} for r in s.execute(q).scalars().all()]
    except Exception as exc:  # pragma: no cover
        log.warning("search_plates failed: %s", exc)
        return []


def delete_plate(plate_id: int) -> bool:
    if _guard() is None:
        return False
    try:
        with get_session() as s:
            row = s.get(LicensePlate, plate_id)
            if row is not None:
                s.delete(row)
                s.commit()
                return True
        return False
    except Exception as exc:  # pragma: no cover
        log.warning("delete_plate failed: %s", exc)
        return False


# ---------------------------------------------------------------------------#
# Detections
# ---------------------------------------------------------------------------#
def record_detection(
    detection_type: str = "vehicle",
    object_class: str = "",
    confidence: float = 0.0,
    source: str = "",
    tracking_id: int = 0,
    dedupe_window: float = 5.0,
) -> Optional[int]:
    """Record a detection, suppressing high-frequency duplicates."""
    if _guard() is None:
        return None
    try:
        cutoff = _now() - timedelta(seconds=dedupe_window)
        with get_session() as s:
            existing = s.execute(
                select(Detection).where(
                    Detection.object_class == object_class,
                    Detection.source == source,
                    Detection.detected_at >= cutoff,
                )
            ).scalars().first()
            if existing is not None:
                existing.detection_count = (existing.detection_count or 1) + 1
                s.commit()
                return existing.id
            det = Detection(
                detection_type=detection_type,
                object_class=object_class,
                confidence=float(confidence),
                source=source,
                tracking_id=int(tracking_id),
                detection_count=1,
            )
            s.add(det)
            s.commit()
            s.refresh(det)
            return det.id
    except Exception as exc:  # pragma: no cover
        log.debug("record_detection failed: %s", exc)
        return None


def search_detections(
    query: str = "", detection_type: str = "", limit: int = 500,
) -> list[dict[str, Any]]:
    if _guard() is None:
        return []
    try:
        with get_session() as s:
            q = select(Detection)
            if query:
                q = q.where(Detection.object_class.ilike(f"%{query}%"))
            if detection_type:
                q = q.where(Detection.detection_type == detection_type)
            q = q.order_by(Detection.detected_at.desc()).limit(limit)
            return [dict(r.__dict__) | {"id": r.id} for r in s.execute(q).scalars().all()]
    except Exception as exc:  # pragma: no cover
        log.warning("search_detections failed: %s", exc)
        return []


# ---------------------------------------------------------------------------#
# OCR results
# ---------------------------------------------------------------------------#
def record_ocr(text: str, confidence: float = 0.0, source: str = "", image_path: str = "") -> Optional[int]:
    if _guard() is None:
        return None
    try:
        with get_session() as s:
            res = OCRResult(recognized_text=text, confidence=float(confidence),
                            source=source, image_path=image_path)
            s.add(res)
            s.commit()
            s.refresh(res)
            return res.id
    except Exception as exc:  # pragma: no cover
        log.debug("record_ocr failed: %s", exc)
        return None


def search_ocr(query: str = "", limit: int = 200) -> list[dict[str, Any]]:
    if _guard() is None:
        return []
    try:
        with get_session() as s:
            q = select(OCRResult)
            if query:
                q = q.where(OCRResult.recognized_text.ilike(f"%{query}%"))
            q = q.order_by(OCRResult.recognized_at.desc()).limit(limit)
            return [dict(r.__dict__) | {"id": r.id} for r in s.execute(q).scalars().all()]
    except Exception as exc:  # pragma: no cover
        log.warning("search_ocr failed: %s", exc)
        return []


# ---------------------------------------------------------------------------#
# Sessions
# ---------------------------------------------------------------------------#
def start_session(session_id: str, source: str = "") -> Optional[int]:
    if _guard() is None:
        return None
    try:
        with get_session() as s:
            rec = SessionRecord(session_id=session_id, source=source)
            s.add(rec)
            s.commit()
            s.refresh(rec)
            return rec.id
    except Exception as exc:  # pragma: no cover
        log.warning("start_session failed: %s", exc)
        return None


def end_session(session_id: str, processed_frames: int = 0, vehicle_count: int = 0,
                plate_count: int = 0, face_count: int = 0, object_count: int = 0) -> None:
    if _guard() is None:
        return
    try:
        with get_session() as s:
            rec = s.execute(
                select(SessionRecord).where(SessionRecord.session_id == session_id)
            ).scalar_one_or_none()
            if rec is not None:
                rec.end_time = _now()
                rec.processed_frames = processed_frames
                rec.vehicle_count = vehicle_count
                rec.plate_count = plate_count
                rec.face_count = face_count
                rec.object_count = object_count
                s.commit()
    except Exception as exc:  # pragma: no cover
        log.warning("end_session failed: %s", exc)


# ---------------------------------------------------------------------------#
# Application settings
# ---------------------------------------------------------------------------#
def set_setting(key: str, value: str) -> None:
    if _guard() is None:
        return
    try:
        with get_session() as s:
            row = s.execute(select(AppSetting).where(AppSetting.key == key)).scalar_one_or_none()
            if row is None:
                s.add(AppSetting(key=key, value=value))
            else:
                row.value = value
            s.commit()
    except Exception as exc:  # pragma: no cover
        log.debug("set_setting failed: %s", exc)


def get_setting(key: str, default: str = "") -> str:
    if _guard() is None:
        return default
    try:
        with get_session() as s:
            row = s.execute(select(AppSetting).where(AppSetting.key == key)).scalar_one_or_none()
            return row.value if row is not None else default
    except Exception as exc:  # pragma: no cover
        log.debug("get_setting failed: %s", exc)
        return default


def get_all_settings() -> dict[str, str]:
    if _guard() is None:
        return {}
    try:
        with get_session() as s:
            rows = s.execute(select(AppSetting)).scalars().all()
            return {r.key: r.value for r in rows}
    except Exception as exc:  # pragma: no cover
        log.debug("get_all_settings failed: %s", exc)
        return {}


def save_app_state(config: Any) -> None:
    """Persist the salient AppConfig fields into application_settings."""
    if _guard() is None:
        return
    mapping = {
        "camera_index": str(getattr(config, "camera_index", 0)),
        "resolution": f"{getattr(config, 'resolution', (1280, 720))[0]}x{getattr(config, 'resolution', (1280, 720))[1]}",
        "confidence": str(getattr(config, "confidence", 0.45)),
        "yolo_model": getattr(config, "yolo_model", ""),
        "enabled_modules": ",".join(sorted(getattr(config, "enabled_modules", set()))),
        "enabled_modes": ",".join(sorted(getattr(config, "enabled_modes", set()))),
        "theme": getattr(config, "theme", "dark"),
        "source": getattr(config, "source", "webcam"),
        "ip_url": getattr(config, "ip_url", ""),
        "media_path": getattr(config, "media_path", ""),
        "last_opened_page": getattr(config, "last_opened_page", ""),
    }
    for k, v in mapping.items():
        set_setting(k, v)


def load_app_state(config: Any) -> None:
    """Apply persisted settings onto an AppConfig instance (best-effort)."""
    if _guard() is None:
        return
    settings = get_all_settings()
    try:
        if "camera_index" in settings:
            config.camera_index = int(settings["camera_index"])
        if "resolution" in settings:
            w, _, h = settings["resolution"].partition("x")
            config.resolution = (int(w), int(h))
        if "confidence" in settings:
            config.confidence = float(settings["confidence"])
        if "yolo_model" in settings:
            config.yolo_model = settings["yolo_model"]
        if "enabled_modules" in settings:
            config.enabled_modules = set(x for x in settings["enabled_modules"].split(",") if x)
        if "enabled_modes" in settings:
            config.enabled_modes = set(x for x in settings["enabled_modes"].split(",") if x)
        if "theme" in settings:
            config.theme = settings["theme"]
        if "source" in settings:
            config.source = settings["source"]
        if "ip_url" in settings:
            config.ip_url = settings["ip_url"]
        if "media_path" in settings:
            config.media_path = settings["media_path"]
        if "last_opened_page" in settings:
            config.last_opened_page = settings["last_opened_page"]
    except Exception as exc:  # pragma: no cover
        log.warning("load_app_state failed: %s", exc)


# ---------------------------------------------------------------------------#
# System logs
# ---------------------------------------------------------------------------#
def log_system(level: str, message: str, logger: str = "") -> None:
    if _guard() is None:
        return
    try:
        with get_session() as s:
            s.add(SystemLog(level=level.upper(), message=message[:4000], logger=logger))
            s.commit()
    except Exception as exc:  # pragma: no cover
        log.debug("log_system failed: %s", exc)


def recent_logs(limit: int = 200) -> list[dict[str, Any]]:
    if _guard() is None:
        return []
    try:
        with get_session() as s:
            rows = s.execute(select(SystemLog).order_by(SystemLog.created_at.desc()).limit(limit)).scalars().all()
            return [dict(r.__dict__) for r in rows]
    except Exception as exc:  # pragma: no cover
        log.warning("recent_logs failed: %s", exc)
        return []


# ---------------------------------------------------------------------------#
# Dashboard statistics
# ---------------------------------------------------------------------------#
def dashboard_stats() -> dict[str, Any]:
    """Aggregate statistics for the dashboard. Returns zeros if DB unavailable."""
    empty = {
        "total_faces": 0, "total_vehicles": 0, "unique_plates": 0,
        "total_detections": 0, "todays_detections": 0, "total_sessions": 0,
        "most_detected_vehicles": [], "recent_detections": [],
    }
    if _guard() is None:
        return empty
    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with get_session() as s:
            total_faces = s.execute(select(func.count(User.id))).scalar() or 0
            total_plates = s.execute(select(func.count(LicensePlate.id))).scalar() or 0
            unique_plates = s.execute(select(func.count(func.distinct(LicensePlate.plate_number)))).scalar() or 0
            total_detections = s.execute(select(func.count(Detection.id))).scalar() or 0
            todays = s.execute(
                select(func.count(Detection.id)).where(Detection.detected_at >= today)
            ).scalar() or 0
            total_sessions = s.execute(select(func.count(SessionRecord.id))).scalar() or 0

            # Most detected vehicle types (from license_plates.vehicle_type).
            most_rows = s.execute(
                select(LicensePlate.vehicle_type, func.count(LicensePlate.id).label("n"))
                .where(LicensePlate.vehicle_type != "")
                .group_by(LicensePlate.vehicle_type)
                .order_by(func.count(LicensePlate.id).desc())
                .limit(5)
            ).all()
            most_vehicles = [{"type": r[0], "count": r[1]} for r in most_rows]

            recent = s.execute(
                select(Detection).order_by(Detection.detected_at.desc()).limit(8)
            ).scalars().all()
            recent_detections = [
                {"object_class": r.object_class, "confidence": r.confidence,
                 "detected_at": r.detected_at, "source": r.source}
                for r in recent
            ]
            return {
                "total_faces": total_faces,
                "total_vehicles": total_plates,
                "unique_plates": unique_plates,
                "total_detections": total_detections,
                "todays_detections": todays,
                "total_sessions": total_sessions,
                "most_detected_vehicles": most_vehicles,
                "recent_detections": recent_detections,
            }
    except Exception as exc:  # pragma: no cover
        log.warning("dashboard_stats failed: %s", exc)
        return empty
