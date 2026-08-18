"""VisionSuite — SQLAlchemy ORM models for local PostgreSQL storage.

Clean relational schema covering faces, vehicles, license plates, object
detections, OCR results, sessions, application settings and system logs.

All tables use integer primary keys, proper timestamps, foreign keys where a
relationship exists, indexes on frequently searched columns and unique
constraints where appropriate.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from database import Base


# ---------------------------------------------------------------------------#
# Time helper — SQLAlchemy stores timezone-aware timestamps on PostgreSQL.
# ---------------------------------------------------------------------------#
def utcnow():
    return datetime.utcnow()


class User(Base):
    """Registered user/identity record (face identity anchor)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    status = Column(String(64), default="active")
    image_path = Column(String(1024), default="")
    created_at = Column(DateTime, default=utcnow)
    last_seen_at = Column(DateTime, nullable=True)
    recognition_count = Column(Integer, default=0)

    face_embeddings = relationship("FaceEmbedding", back_populates="user",
                                   cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_name", "name"),
        Index("ix_users_created", "created_at"),
    )


class FaceEmbedding(Base):
    """A user's face embedding vector (128-d), stored as JSON text."""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Text, nullable=False)   # JSON list[float]
    created_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="face_embeddings")

    __table_args__ = (
        Index("ix_face_emb_user", "user_id"),
        Index("ix_face_emb_created", "created_at"),
    )


class LicensePlate(Base):
    """A license plate recognition (ANPR) entry."""

    __tablename__ = "license_plates"

    id = Column(Integer, primary_key=True)
    plate_number = Column(String(64), nullable=False)
    ocr_confidence = Column(Float, default=0.0)
    vehicle_type = Column(String(64), default="")
    tracking_id = Column(Integer, default=0)
    detected_at = Column(DateTime, default=utcnow)
    camera_source = Column(String(512), default="")
    video_filename = Column(String(512), default="")
    frame_number = Column(Integer, default=0)
    vehicle_image_path = Column(String(1024), default="")
    plate_image_path = Column(String(1024), default="")
    recognition_count = Column(Integer, default=1)

    __table_args__ = (
        Index("ix_plates_plate", "plate_number"),
        Index("ix_plates_time", "detected_at"),
        Index("ix_plates_track", "tracking_id"),
        Index("ix_plates_video", "video_filename"),
    )


class Detection(Base):
    """One object detection event (object class, confidence, source...)."""

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    detection_type = Column(String(64), default="vehicle")  # vehicle/face/object...
    object_class = Column(String(128), nullable=False)
    confidence = Column(Float, default=0.0)
    detected_at = Column(DateTime, default=utcnow)
    source = Column(String(512), default="")
    tracking_id = Column(Integer, default=0)
    detection_count = Column(Integer, default=1)

    __table_args__ = (
        Index("ix_detections_type", "detection_type"),
        Index("ix_detections_class", "object_class"),
        Index("ix_detections_time", "detected_at"),
        Index("ix_detections_track", "tracking_id"),
    )


class OCRResult(Base):
    """A recognised piece of text (OCR / QR / codes)."""

    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True)
    recognized_text = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    recognized_at = Column(DateTime, default=utcnow)
    source = Column(String(512), default="")
    image_path = Column(String(1024), default="")

    __table_args__ = (
        Index("ix_ocr_text", "recognized_text"),
        Index("ix_ocr_time", "recognized_at"),
    )


class SessionRecord(Base):
    """One VisionSuite session (start/end, source, frame + object counters)."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(128), unique=True, nullable=False)  # uuid string
    start_time = Column(DateTime, default=utcnow)
    end_time = Column(DateTime, nullable=True)
    source = Column(String(512), default="")
    processed_frames = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    plate_count = Column(Integer, default=0)
    face_count = Column(Integer, default=0)
    object_count = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_sessions_start", "start_time"),
        Index("ix_sessions_source", "source"),
    )


class AppSetting(Base):
    """Persisted application settings (key/value)."""

    __tablename__ = "application_settings"

    id = Column(Integer, primary_key=True)
    key = Column(String(255), nullable=False, unique=True)
    value = Column(Text, default="")

    __table_args__ = (
        Index("ix_settings_key", "key"),
    )


class SystemLog(Base):
    """Important application events / errors for diagnostics."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True)
    level = Column(String(16), default="INFO")
    message = Column(Text, default="")
    logger = Column(String(128), default="")
    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (
        Index("ix_logs_level", "level"),
        Index("ix_logs_time", "created_at"),
    )


# Optional vehicle/plate relationship helper (kept minimal; analytics can join).
VehicleRecord = LicensePlate  # alias for readability in repositories


ALL_MODELS = [
    User, FaceEmbedding, LicensePlate, Detection,
    OCRResult, SessionRecord, AppSetting, SystemLog,
]
