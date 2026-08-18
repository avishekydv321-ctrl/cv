"""Pydantic schemas for request/response models."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    label: str
    confidence: float
    box: list[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    source: str = "object"
    color: Optional[list[int]] = None


class ImageDetectionResponse(BaseModel):
    success: bool
    processing_time_ms: float
    detections: list[DetectionResult]
    annotated_image_base64: Optional[str] = None
    info: dict[str, Any] = {}
    error: Optional[str] = None


class VideoDetectionResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    message: str
    processing_time_ms: float = 0.0
    total_frames: int = 0
    detections: list[DetectionResult] = []
    error: Optional[str] = None


class ANPRResponse(BaseModel):
    success: bool
    plate_text: Optional[str] = None
    confidence: Optional[float] = None
    vehicle_box: Optional[list[int]] = None
    plate_box: Optional[list[int]] = None
    timestamp: Optional[str] = None
    processing_time_ms: float
    annotated_image_base64: Optional[str] = None
    error: Optional[str] = None


class OCRResponse(BaseModel):
    success: bool
    text: str = ""
    confidence: float = 0.0
    processing_time_ms: float
    annotated_image_base64: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, str]
    database: str
    device: str


class ModelInfo(BaseModel):
    key: str
    name: str
    available: bool
    status: str
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    yolo_weights: list[str]
    device: str


class HistoryItem(BaseModel):
    id: int
    detection_type: str
    object_class: str
    confidence: float
    detected_at: str
    source: str
    tracking_id: int


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class DashboardStats(BaseModel):
    total_detections: int
    todays_detections: int
    total_faces: int
    total_vehicles: int
    unique_plates: int
    total_sessions: int
    most_detected_vehicles: list[dict[str, Any]]
    recent_detections: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    error: Optional[str] = None


class CameraFrameRequest(BaseModel):
    frame_base64: str = Field(..., description="Base64 encoded image frame")
    modules: list[str] = Field(default_factory=list, description="Enabled AI module keys")
    confidence: float = Field(default=0.45, ge=0.1, le=1.0)


class CameraFrameResponse(BaseModel):
    success: bool
    fps: float
    detections: list[DetectionResult]
    info: dict[str, Any] = {}
    alerts: list[str] = []
    modules: dict[str, float] = {}
    processing_time_ms: float
    annotated_image_base64: Optional[str] = None
    error: Optional[str] = None
