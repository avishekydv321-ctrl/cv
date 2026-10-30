"""VisionSuite FastAPI backend entry point."""
from __future__ import annotations

import logging
import sys
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.core.config import settings
from server.core.logging import configure_logging
from server.api import api_router
from server.services.database import init_db

configure_logging()
logger = logging.getLogger("main")

# Ensure existing visionsuite modules are importable
EXISTING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(EXISTING_ROOT) not in sys.path:
    sys.path.insert(0, str(EXISTING_ROOT))

app = FastAPI(
    title="VisionSuite AI API",
    description="Computer Vision inference backend for VisionSuite",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Include websocket router directly to avoid circular imports
from server.api.websocket.detection import router as websocket_router
app.include_router(websocket_router, tags=["websocket"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check backend logs."},
    )


@app.on_event("startup")
async def startup_event():
    logger.info("VisionSuite backend starting up...")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    if not init_db():
        logger.warning("Database schema initialization failed; API will run without persistence")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("VisionSuite backend shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
