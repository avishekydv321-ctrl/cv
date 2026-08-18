"""WebSocket for live camera detection."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.ai.pipeline import get_pipeline
from server.models.schemas import CameraFrameResponse

logger = logging.getLogger("api.websocket")
router = APIRouter()


@router.websocket("/ws/detection")
async def websocket_detection(websocket: WebSocket):
    """WebSocket endpoint for live camera frame detection.

    Client sends:
        {"frame_base64": "...", "modules": ["objects", "face_recognition"], "confidence": 0.45}

    Server responds:
        {"success": true, "fps": 0.0, "detections": [...], "processing_time_ms": 12.3, ...}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    pipeline = get_pipeline()
    fps_times: list[float] = []
    max_fps_samples = 30

    try:
        while True:
            data_str = await websocket.receive_text()
            t0 = time.perf_counter()

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"success": False, "error": "Invalid JSON"}))
                continue

            frame_b64 = data.get("frame_base64", "")
            modules = data.get("modules", [])
            confidence = float(data.get("confidence", 0.45))

            if not frame_b64:
                await websocket.send_text(json.dumps({"success": False, "error": "No frame provided"}))
                continue

            try:
                content = base64.b64decode(frame_b64)
                nparr = np.frombuffer(content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_text(json.dumps({"success": False, "error": "Could not decode frame"}))
                    continue

                result = pipeline.process_frame(frame, modules, confidence)

                # Calculate FPS
                now = time.perf_counter()
                fps_times.append(now)
                while len(fps_times) > max_fps_samples:
                    fps_times.pop(0)
                if len(fps_times) > 1:
                    span = fps_times[-1] - fps_times[0]
                    result["fps"] = round((len(fps_times) - 1) / span, 1) if span > 0 else 0.0
                else:
                    result["fps"] = 0.0

                result["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                await websocket.send_text(json.dumps(result))

            except Exception as exc:
                logger.error("Frame processing error: %s", exc)
                await websocket.send_text(json.dumps({
                    "success": False,
                    "error": str(exc),
                    "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)


@router.websocket("/ws/camera")
async def websocket_camera_stream(websocket: WebSocket):
    """Simplified WebSocket for camera stream with detection overlay."""
    await websocket.accept()
    logger.info("Camera stream WebSocket connected")

    pipeline = get_pipeline()
    fps_times: list[float] = []

    try:
        while True:
            data_str = await websocket.receive_text()
            t0 = time.perf_counter()

            try:
                data = json.loads(data_str)
                frame_b64 = data.get("frame", "")
                modules = data.get("modules", ["objects"])
                confidence = float(data.get("confidence", 0.45))

                if not frame_b64:
                    continue

                content = base64.b64decode(frame_b64)
                nparr = np.frombuffer(content, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                result = pipeline.process_frame(frame, modules, confidence)

                now = time.perf_counter()
                fps_times.append(now)
                while len(fps_times) > 30:
                    fps_times.pop(0)
                fps = round((len(fps_times) - 1) / (fps_times[-1] - fps_times[0]), 1) if len(fps_times) > 1 else 0.0

                response = {
                    "annotated_frame": result.get("annotated_image_base64", ""),
                    "detections": result.get("detections", []),
                    "fps": fps,
                    "processing_time_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }
                await websocket.send_text(json.dumps(response))

            except Exception as exc:
                logger.error("Camera stream error: %s", exc)
                await websocket.send_text(json.dumps({"error": str(exc)}))

    except WebSocketDisconnect:
        logger.info("Camera stream WebSocket disconnected")
