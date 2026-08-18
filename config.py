"""Application-wide configuration objects."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load environment variables from .env if present (never overrides existing).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except Exception:  # pragma: no cover - python-dotenv optional
    pass

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "640 x 480 (VGA)": (640, 480),
    "1280 x 720 (HD)": (1280, 720),
    "1920 x 1080 (Full HD)": (1920, 1080),
}

# Source types supported by the camera/media pipeline.
SOURCE_TYPES: tuple[str, ...] = ("webcam", "ip", "video_file", "image")

YOLO_MODELS: dict[str, str] = {
    "YOLOv8n (fastest)": "yolov8n.pt",
    "YOLOv8s (balanced)": "yolov8s.pt",
    "YOLOv8m (accurate)": "yolov8m.pt",
    "YOLOv11n (fastest)": "yolov11n.pt",
    "YOLOv11s (balanced)": "yolov11s.pt",
    "YOLOv11m (accurate)": "yolov11m.pt",
}

MODES = ("Objects", "Faces", "Hands", "Pose", "Sign Language",
         "Image Processing", "OCR / Codes")

# Toggleable AI modules. Each key maps to a display label and the module
# identifier used by the modular pipeline in app.py. Modules can run
# simultaneously with the base mode whenever the underlying model is available.
AI_MODULES: dict[str, str] = {
    "face_recognition": "Face Recognition",
    "face_mask": "Face Mask",
    "emotion": "Emotion Detection",
    "age_gender": "Age & Gender",
    "sign_language": "Sign Language",
    "fire_smoke": "Fire / Smoke",
    "helmet": "Helmet Detection",
    "ppe": "PPE Detection",
    "vehicle": "Vehicle Tracking",
    "anpr": "Number Plate (ANPR)",
    "currency": "Currency Note",
    "animal": "Animal Detection",
    "plant_disease": "Plant Disease",
    "document_scan": "Document Scanner",
    "qr_codes": "QR / Barcode",
}

FILTERS = (
    "None", "Grayscale", "Blur", "Canny Edges",
    "Threshold", "Sepia", "Histogram Equalization",
)


@dataclass
class AppConfig:
    """Mutable runtime settings shared between GUI and worker threads."""

    camera_index: int = 0
    resolution: tuple[int, int] = (1280, 720)
    mode: str = "Objects"
    yolo_model: str = "yolov8n.pt"
    confidence: float = 0.45
    image_filter: str = "None"
    detection_enabled: bool = True
    show_fps: bool = True
    mirror: bool = True
    record_csv: bool = False
    # Media source: "webcam", "ip", "video_file", "image".
    source: str = "webcam"
    # IP / RTSP camera stream URL (used when source == "ip").
    ip_url: str = ""
    # Local video file or image path (used when source is "video_file"/"image").
    media_path: str = ""
    # System monitoring (CPU / RAM / GPU) + analytics dashboard.
    system_monitor: bool = True
    analytics_enabled: bool = True
    history_limit: int = 200
    # Optional per-frame cap (0 = unlimited) to reduce CPU on high-FPS feeds.
    fps_limit: int = 0
    # Set of active base modes. Multiple modes can run simultaneously and are
    # merged in the pipeline. Defaults to the first mode in MODES.
    enabled_modes: set[str] = field(default_factory=lambda: {"Objects"})
    # Set of enabled AI module identifiers (see AI_MODULES above).
    enabled_modules: set[str] = field(default_factory=set)
    # Confidence thresholds per feature (labels given on the GUI).
    module_confidences: dict[str, float] = field(default_factory=dict)
    # Emission latency (seconds) for real-time alerts (fire/smoke/unknown face).
    alert_cooldown: float = 3.0

    # -- AI low-quality enhancement preprocessors --
    sharpening: bool = False
    denoising: bool = False
    clahe: bool = False
    exposure_correction: bool = False
    white_balance: bool = False
    stabilization: bool = False

    # -- Native tracking in YOLO mode --
    tracking_enabled: bool = True

    # -- Dark/Light Theme selection --
    theme: str = "dark"

    # -- Last opened page (persisted to PostgreSQL) --
    last_opened_page: str = ""

    # -- Attendance mode configuration --
    attendance_mode: bool = False

    # -- License plate ANPR blacklist --
    blacklist_plates: str = ""

    # -- Smart Alerts notification setup --
    desktop_notifications: bool = True
    sound_alerts: bool = True
    email_alerts_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_sender: str = ""
    email_recipient: str = ""
    email_password: str = ""
