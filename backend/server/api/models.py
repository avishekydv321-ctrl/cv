"""Models information endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter

from server.core.config import settings
from server.ai.pipeline import get_pipeline
from server.models.schemas import ModelInfo, ModelsResponse

logger = logging.getLogger("api.models")
router = APIRouter()


@router.get("", response_model=ModelsResponse)
async def list_models():
    """List all available AI models and their status."""
    pipeline = get_pipeline()
    status = pipeline.get_model_status()

    models_list = []

    # Base detectors
    for key, name, status_key in [
        ("yolo", "YOLO Object Detection", "yolo"),
        ("face_detector", "Face Detection", "face_detector"),
        ("hand_tracker", "Hand Tracking", "hand_tracker"),
        ("pose_estimator", "Pose Estimation", "pose_estimator"),
        ("code_reader", "QR/Barcode + OCR", "code_reader"),
    ]:
        s = status.get(status_key, {})
        models_list.append(ModelInfo(
            key=key,
            name=name,
            available=s.get("available", False),
            status=s.get("status", "unknown"),
            description=f"{name} model status",
        ))

    # Toggleable modules
    module_display = {
        "face_recognition": "Face Recognition",
        "face_mask": "Face Mask Detection",
        "emotion": "Emotion Detection",
        "age_gender": "Age & Gender Estimation",
        "sign_language": "Sign Language Recognition",
        "fire_smoke": "Fire / Smoke Detection",
        "helmet": "Helmet Detection",
        "ppe": "PPE Detection",
        "vehicle": "Vehicle Tracking",
        "anpr": "Number Plate (ANPR)",
        "currency": "Currency Note Recognition",
        "animal": "Animal Detection",
        "plant_disease": "Plant Disease Classification",
        "document_scan": "Document Scanner",
        "qr_codes": "QR / Barcode Scanning",
    }

    modules_status = status.get("modules", {})
    for key, name in module_display.items():
        s = modules_status.get(key, {"available": False, "status": "not_loaded"})
        models_list.append(ModelInfo(
            key=key,
            name=name,
            available=s.get("available", False),
            status=s.get("status", "not_loaded"),
            description=f"{name} module",
        ))

    yolo_weights = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov11n.pt", "yolov11s.pt", "yolov11m.pt"]

    return ModelsResponse(
        models=models_list,
        yolo_weights=yolo_weights,
        device=settings.DEVICE,
    )
