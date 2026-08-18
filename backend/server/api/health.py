"""Health and status endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from sqlalchemy import text

from server.core.config import settings
from server.services.database import get_engine, is_available
from server.ai.pipeline import get_pipeline

logger = logging.getLogger("api.health")
router = APIRouter()


@router.get("/health")
async def health_check():
    """Comprehensive health check."""
    db_status = "connected" if is_available() else "disconnected"
    pipeline = get_pipeline()
    model_status = pipeline.get_model_status()

    return {
        "status": "ok",
        "version": "1.0.0",
        "database": db_status,
        "device": settings.DEVICE,
        "models": model_status,
    }


@router.get("/health/live")
async def health_live():
    return {"status": "alive"}


@router.get("/health/ready")
async def health_ready():
    db_ok = is_available()
    return {
        "status": "ready" if db_ok else "not_ready",
        "database": "connected" if db_ok else "disconnected",
    }
