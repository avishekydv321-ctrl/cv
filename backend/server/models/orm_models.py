"""SQLAlchemy ORM models for VisionSuite backend."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    detection_type = Column(String(64), default="object")
    object_class = Column(String(128), nullable=False, index=True)
    confidence = Column(Float, default=0.0)
    detected_at = Column(DateTime, default=func.now(), index=True)
    source = Column(String(512), default="")
    tracking_id = Column(Integer, default=0, index=True)
    detection_count = Column(Integer, default=1)
    extra_data = Column(Text, nullable=True)  # JSON blob for module-specific data


class LicensePlateRecord(Base):
    __tablename__ = "license_plates"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String(64), nullable=False, index=True)
    ocr_confidence = Column(Float, default=0.0)
    vehicle_type = Column(String(64), default="")
    tracking_id = Column(Integer, default=0, index=True)
    detected_at = Column(DateTime, default=func.now(), index=True)
    camera_source = Column(String(512), default="")
    video_filename = Column(String(512), default="")
    frame_number = Column(Integer, default=0)
    vehicle_image_path = Column(String(1024), default="")
    plate_image_path = Column(String(1024), default="")
    recognition_count = Column(Integer, default=1)


class OCRRecord(Base):
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, index=True)
    recognized_text = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    recognized_at = Column(DateTime, default=func.now(), index=True)
    source = Column(String(512), default="")
    image_path = Column(String(1024), default="")


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), unique=True, nullable=False, index=True)
    start_time = Column(DateTime, default=func.now())
    end_time = Column(DateTime, nullable=True)
    source = Column(String(512), default="")
    processed_frames = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    plate_count = Column(Integer, default=0)
    face_count = Column(Integer, default=0)
    object_count = Column(Integer, default=0)


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(512), nullable=False)
    original_name = Column(String(512), nullable=False)
    file_type = Column(String(64), nullable=False)  # image, video
    mime_type = Column(String(128), nullable=False)
    size_bytes = Column(Integer, default=0)
    storage_path = Column(String(1024), nullable=False)
    uploaded_at = Column(DateTime, default=func.now(), index=True)
    processed = Column(Boolean, default=False)
    processing_result = Column(Text, nullable=True)  # JSON


class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(16), default="INFO")
    message = Column(Text, default="")
    logger_name = Column(String(128), default="")
    created_at = Column(DateTime, default=func.now(), index=True)
