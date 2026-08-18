"""ANPR (Automatic Number Plate Recognition) endpoints."""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server.core.config import settings
from server.models.schemas import ANPRResponse
from server.ai.pipeline import get_pipeline
from server.services.storage import storage
from server.services.database import get_session
from server.models.orm_models import LicensePlateRecord

logger = logging.getLogger("api.anpr")
router = APIRouter()


def _validate_image(file: UploadFile) -> bytes:
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")
    content = file.file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large")
    return content


@router.post("/image", response_model=ANPRResponse)
async def anpr_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
):
    """Upload an image for ANPR processing."""
    t0 = time.perf_counter()
    try:
        content = _validate_image(file)
        pipeline = get_pipeline()
        result = pipeline.process_anpr_image(content, confidence)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        # Persist to database if successful
        if result.get("success") and result.get("plate_text"):
            try:
                with get_session() as db:
                    record = LicensePlateRecord(
                        plate_number=result["plate_text"],
                        ocr_confidence=result.get("confidence", 0.0),
                        camera_source="api_upload",
                        video_filename=file.filename or "uploaded_image",
                    )
                    db.add(record)
                    db.commit()
            except Exception as exc:
                logger.warning("Failed to persist ANPR record: %s", exc)

        return ANPRResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ANPR image processing failed")
        return ANPRResponse(
            success=False,
            processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            error=str(exc),
        )


@router.post("/video")
async def anpr_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
):
    """Upload a video for ANPR processing. Returns summary of detected plates."""
    t0 = time.perf_counter()
    try:
        content = _validate_image(file)  # reuse validation
        file_path = storage.save_upload(content, file.filename or "video.mp4", subdir="videos")
        pipeline = get_pipeline()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        max_frames = min(total_frames, 300)

        plates_found: list[dict[str, Any]] = []
        frame_idx = 0

        try:
            from modules.anpr_video import VideoANPREngine
            engine = VideoANPREngine(confidence=confidence)
            engine.prepare()
        except Exception as exc:
            logger.warning("ANPR engine prepare failed: %s", exc)
            engine = None

        while frame_idx < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if engine is not None:
                try:
                    result = engine.process(frame)
                    for v in result.vehicles:
                        if v.plate and v.plate.text and v.plate.text not in [p.get("plate_text") for p in plates_found]:
                            plates_found.append({
                                "plate_text": v.plate.text,
                                "confidence": v.plate.confidence,
                                "vehicle_type": v.vehicle_type,
                                "track_id": v.track_id,
                                "frame": frame_idx,
                                "timestamp_s": round(frame_idx / fps, 1),
                            })
                except Exception:
                    pass
            frame_idx += 1

        cap.release()

        processing_time = round((time.perf_counter() - t0) * 1000.0, 1)
        return JSONResponse(content={
            "success": True,
            "processing_time_ms": processing_time,
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "plates_detected": len(plates_found),
            "plates": plates_found[:100],
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ANPR video processing failed")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc), "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1)},
        )
