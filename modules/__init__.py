"""AI module registry.

``get_module(key)`` returns a lazily-constructed, cached singleton for each
toggleable AI feature. Modules are imported lazily so the app starts fast and
an optional dependency (e.g. face_recognition, pytesseract) being missing never
breaks anything else.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("modules")

# key -> module class (imported lazily inside get_module)
_REGISTRY: dict[str, tuple[str, str]] = {
    "face_recognition": ("modules.face", "FaceRecognition"),
    "face_mask": ("modules.face", "FaceMask"),
    "emotion": ("modules.face", "EmotionDetector"),
    "age_gender": ("modules.face", "AgeGenderEstimator"),
    "sign_language": ("modules.handsign", "SignLanguage"),
    "fire_smoke": ("modules.safety", "FireSmokeDetector"),
    "helmet": ("modules.safety", "HelmetDetector"),
    "ppe": ("modules.safety", "PPEDetector"),
    "vehicle": ("modules.objects", "VehicleModule"),
    "anpr": ("modules.ocr", "ANPR"),
    "currency": ("modules.objects", "CurrencyModule"),
    "animal": ("modules.objects", "AnimalModule"),
    "plant_disease": ("modules.plants", "PlantDisease"),
    "document_scan": ("modules.ocr", "DocumentScanner"),
    "qr_codes": ("modules.ocr", "QRCodeModule"),
}

_cache: dict[str, Any] = {}


def get_module(key: str) -> Any:
    """Return a cached singleton instance of the AI module with ``key``."""
    if key in _cache:
        return _cache[key]

    if key not in _REGISTRY:
        raise KeyError(f"Unknown AI module: {key}")

    module_name, class_name = _REGISTRY[key]
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        instance = cls()
        _cache[key] = instance
        return instance
    except Exception as exc:  # pragma: no cover
        log.warning("Module %s could not be loaded: %s", key, exc)
        raise


def reset_modules() -> None:
    """Close and drop all cached module singletons."""
    for instance in _cache.values():
        close = getattr(instance, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover
                pass
    _cache.clear()


__all__ = ["get_module", "reset_modules", "_REGISTRY"]
