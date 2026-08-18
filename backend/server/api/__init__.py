"""API routers."""
from fastapi import APIRouter

import server.api.detect as detect
import server.api.anpr as anpr
import server.api.ocr as ocr
import server.api.health as health
import server.api.models as models
import server.api.history as history

api_router = APIRouter()

api_router.include_router(detect.router, prefix="/detect", tags=["detection"])
api_router.include_router(anpr.router, prefix="/anpr", tags=["anpr"])
api_router.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(history.router, prefix="/history", tags=["history"])
