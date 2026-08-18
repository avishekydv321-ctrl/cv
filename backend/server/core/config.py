"""VisionSuite backend configuration."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "VisionSuite AI API"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+psycopg://visionsuite_user:change_me@localhost:5432/visionsuite"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Storage
    STORAGE_URL: str = ""
    STORAGE_KEY: str = ""
    STORAGE_SECRET: str = ""
    STORAGE_BUCKET: str = ""
    UPLOAD_DIR: str = "/tmp/visionsuite_uploads"
    OUTPUT_DIR: str = "/tmp/visionsuite_output"

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # Models
    YOLO_WEIGHTS_DIR: str = "./models"
    PLATE_MODEL_DIR: str = "./models/license_plate"

    # Inference
    DEVICE: str = "cpu"
    CONFIDENCE_THRESHOLD: float = 0.45
    MAX_UPLOAD_SIZE_MB: int = 50
    MAX_VIDEO_DURATION_SECONDS: int = 300

    # Features
    ENABLE_EASYOCR: bool = True
    ENABLE_TESSERACT: bool = False
    ENABLE_FACE_RECOGNITION: bool = False
    ENABLE_GPU: bool = False

    # Auth
    SECRET_KEY: str = "change_me_to_a_strong_random_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
