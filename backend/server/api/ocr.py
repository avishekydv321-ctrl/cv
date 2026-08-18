"""OCR endpoints."""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from server.core.config import settings
from server.models.schemas import OCRResponse
from server.ai.pipeline import get_pipeline
from server.services.storage import storage

logger = logging.getLogger("api.ocr")
router = APIRouter()


def _validate_image(file: UploadFile) -> bytes:
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")
    content = file.file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large")
    return content


@router.post("", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
):
    """Upload an image for OCR text extraction."""
    t0 = time.perf_counter()
    try:
        content = _validate_image(file)
        pipeline = get_pipeline()
        result = pipeline.process_ocr_image(content)
        result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return OCRResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OCR processing failed")
        return OCRResponse(
            success=False,
            processing_time_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            error=str(exc),
        )
