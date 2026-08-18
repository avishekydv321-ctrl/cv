"""File storage abstraction for VisionSuite backend."""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from server.core.config import settings

logger = logging.getLogger("storage")


class StorageService:
    """Abstract file storage. Currently uses local filesystem; can be extended
    for S3, Cloudinary, Supabase Storage, etc."""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, content: bytes, filename: str, subdir: str = "uploads") -> str:
        """Save uploaded file and return the path."""
        dest_dir = self.upload_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(filename).suffix
        safe_name = f"{uuid.uuid4().hex}{ext}"
        dest_path = dest_dir / safe_name
        dest_path.write_bytes(content)
        logger.info("Saved upload: %s", dest_path)
        return str(dest_path)

    def save_output(self, content: bytes, filename: str, subdir: str = "results") -> str:
        """Save output file and return the path."""
        dest_dir = self.output_dir / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename
        dest_path.write_bytes(content)
        return str(dest_path)

    def get_upload_path(self, filename: str) -> Path:
        return self.upload_dir / filename

    def cleanup(self, path: str) -> None:
        """Remove a temporary file."""
        try:
            p = Path(path)
            if p.exists() and str(p).startswith(str(self.upload_dir)):
                p.unlink()
        except Exception as exc:
            logger.warning("Failed to cleanup %s: %s", path, exc)

    def cleanup_old(self, max_age_hours: int = 24) -> None:
        """Remove files older than max_age_hours from upload directory."""
        import time
        cutoff = time.time() - (max_age_hours * 3600)
        for root, dirs, files in os.walk(str(self.upload_dir)):
            for f in files:
                fp = Path(root) / f
                try:
                    if fp.stat().st_mtime < cutoff:
                        fp.unlink()
                except Exception:
                    pass


storage = StorageService()
