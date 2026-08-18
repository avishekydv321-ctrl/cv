"""Image and video detection endpoints."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server.core.config import settings
from server.models.schemas import ImageDetectionResponse, VideoDetectionResponse
from server.ai.pipeline import get_pipeline
from server.services.storage import storage

logger = logging.getLogger("api.detect")
router = APIRouter()


def _validate_image(file: UploadFile) -> None:
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")
    content = file.file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large")
    return content


def _validate_video(file: UploadFile) -> None:
    allowed = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {file.content_type}")
    content = file.file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Video too large")
    return content


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    modules: str = Form(""),
    confidence: float = Form(0.45),
):
    """Upload an image and run AI detection."""
    t0 = time.perf_counter()
    try:
        content = _validate_image(file)
        module_list = [m.strip() for m in modules.split(",") if m.strip()]
        pipeline = get_pipeline()
        result = pipeline.process_image(content, module_list, confidence)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return ImageDetectionResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image detection failed")
        return ImageDetectionResponse(
            success=False,
            processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            detections=[],
            error=str(exc),
        )


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    modules: str = Form(""),
    confidence: float = Form(0.35),
):
    """Upload a video for processing. Returns immediately with a job ID."""
    t0 = time.perf_counter()
    try:
        content = _validate_video(file)
        file_path = storage.save_upload(content, file.filename or "video.mp4", subdir="videos")
        # For now, process synchronously (first N frames)
        # In production, use Celery for background processing
        module_list = [m.strip() for m in modules.split(",") if m.strip()]
        pipeline = get_pipeline()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        max_frames = min(total_frames, 150)  # Process up to 150 frames
        all_detections = []
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            result = pipeline.process_frame(frame, module_list, confidence)
            all_detections.extend(result.get("detections", []))
            frame_idx += 1

        cap.release()

        processing_time = round((time.perf_counter() - t0) * 1000.0, 1)
        return VideoDetectionResponse(
            success=True,
            message=f"Processed {frame_idx} frames",
            processing_time_ms=processing_time,
            total_frames=total_frames,
            detections=all_detections[:500],  # Limit detections returned
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video detection failed")
        return VideoDetectionResponse(
            success=False,
            message="Video processing failed",
            processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            error=str(exc),
        )


@router.post("/frame")
async def detect_frame(
    frame_base64: str = Form(...),
    modules: str = Form(""),
    confidence: float = Form(0.45),
):
    """Send a base64-encoded frame for real-time detection."""
    t0 = time.perf_counter()
    try:
        import base64
        import numpy as np
        import cv2

        content = base64.b64decode(frame_base64)
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode frame")

        module_list = [m.strip() for m in modules.split(",") if m.strip()]
        pipeline = get_pipeline()
        result = pipeline.process_frame(frame, module_list, confidence)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Frame detection failed")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc), "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1)},
        )
