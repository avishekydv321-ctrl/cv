"""Detection history endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from server.core.config import settings
from server.models.orm_models import DetectionRecord, LicensePlateRecord, OCRRecord, SessionRecord
from server.services.database import get_session, is_available
from server.models.schemas import HistoryItem, HistoryResponse

logger = logging.getLogger("api.history")
router = APIRouter()


def _record_to_item(record: any) -> HistoryItem:
    return HistoryItem(
        id=record.id,
        detection_type=record.detection_type,
        object_class=record.object_class,
        confidence=record.confidence,
        detected_at=record.detected_at.isoformat() if record.detected_at else "",
        source=record.source,
        tracking_id=record.tracking_id,
    )


@router.get("/detections", response_model=HistoryResponse)
async def get_detection_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    query: Optional[str] = Query(None),
):
    """Get detection history."""
    if not is_available():
        return HistoryResponse(items=[], total=0, page=page, page_size=page_size)

    try:
        with get_session() as db:
            stmt = select(DetectionRecord).order_by(DetectionRecord.detected_at.desc())
            if query:
                stmt = stmt.where(DetectionRecord.object_class.ilike(f"%{query}%"))
            total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
            stmt = stmt.offset((page - 1) * page_size).limit(page_size)
            records = db.execute(stmt).scalars().all()
            return HistoryResponse(
                items=[_record_to_item(r) for r in records],
                total=total,
                page=page,
                page_size=page_size,
            )
    except Exception as exc:
        logger.error("Failed to fetch history: %s", exc)
        return HistoryResponse(items=[], total=0, page=page, page_size=page_size)


@router.get("/plates")
async def get_plate_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    query: Optional[str] = Query(None),
):
    """Get ANPR plate history."""
    if not is_available():
        return {"items": [], "total": 0, "page": page, "page_size": page_size}

    try:
        with get_session() as db:
            stmt = select(LicensePlateRecord).order_by(LicensePlateRecord.detected_at.desc())
            if query:
                stmt = stmt.where(LicensePlateRecord.plate_number.ilike(f"%{query}%"))
            total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
            stmt = stmt.offset((page - 1) * page_size).limit(page_size)
            records = db.execute(stmt).scalars().all()
            items = [
                {
                    "id": r.id,
                    "plate_number": r.plate_number,
                    "ocr_confidence": r.ocr_confidence,
                    "vehicle_type": r.vehicle_type,
                    "tracking_id": r.tracking_id,
                    "detected_at": r.detected_at.isoformat() if r.detected_at else "",
                    "camera_source": r.camera_source,
                    "video_filename": r.video_filename,
                    "frame_number": r.frame_number,
                    "recognition_count": r.recognition_count,
                }
                for r in records
            ]
            return {"items": items, "total": total, "page": page, "page_size": page_size}
    except Exception as exc:
        logger.error("Failed to fetch plate history: %s", exc)
        return {"items": [], "total": 0, "page": page, "page_size": page_size}


@router.get("/dashboard")
async def get_dashboard_stats():
    """Get dashboard statistics."""
    if not is_available():
        return {
            "total_detections": 0, "todays_detections": 0,
            "total_faces": 0, "total_vehicles": 0, "unique_plates": 0,
            "total_sessions": 0, "most_detected_vehicles": [], "recent_detections": [],
        }

    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with get_session() as db:
            total_detections = db.execute(select(func.count(DetectionRecord.id))).scalar() or 0
            todays = db.execute(
                select(func.count(DetectionRecord.id)).where(DetectionRecord.detected_at >= today)
            ).scalar() or 0
            total_sessions = db.execute(select(func.count(SessionRecord.id))).scalar() or 0
            total_plates = db.execute(select(func.count(LicensePlateRecord.id))).scalar() or 0
            unique_plates = db.execute(select(func.count(func.distinct(LicensePlateRecord.plate_number)))).scalar() or 0

            recent = db.execute(
                select(DetectionRecord).order_by(DetectionRecord.detected_at.desc()).limit(10)
            ).scalars().all()
            recent_detections = [
                {
                    "object_class": r.object_class,
                    "confidence": r.confidence,
                    "detected_at": r.detected_at.isoformat() if r.detected_at else "",
                    "source": r.source,
                }
                for r in recent
            ]

            return {
                "total_detections": total_detections,
                "todays_detections": todays,
                "total_faces": 0,
                "total_vehicles": total_plates,
                "unique_plates": unique_plates,
                "total_sessions": total_sessions,
                "most_detected_vehicles": [],
                "recent_detections": recent_detections,
            }
    except Exception as exc:
        logger.error("Failed to fetch dashboard stats: %s", exc)
        return {
            "total_detections": 0, "todays_detections": 0,
            "total_faces": 0, "total_vehicles": 0, "unique_plates": 0,
            "total_sessions": 0, "most_detected_vehicles": [], "recent_detections": [],
        }
