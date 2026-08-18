"""End-to-end ANPR pipeline test.

Verifies the full chain on a synthetic frame that contains a vehicle-like
region with a clearly rendered license plate:

    vehicle region -> plate detection (dedicated YOLOv8 plate model)
    -> plate crop -> enhancement -> OCR -> normalization -> Plates counter

The test is intentionally deterministic and prints a clear PASS/FAIL summary.
"""
from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("plate_pipeline_test")


def make_plate_patch(text: str = "HR26AB1234", w: int = 360, h: int = 110) -> np.ndarray:
    """Render a realistic Indian-style plate patch with text."""
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (30, 30, 30), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Top line: state + RTO code.
    cv2.putText(img, f"{text[:2]} {text[2:4]}", (w // 2 - 90, 34),
                font, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    # Bottom line: series + number.
    cv2.putText(img, text[4:6], (w // 2 - 60, 82), font, 1.3, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(img, text[6:], (w // 2 + 20, 82), font, 1.3, (20, 20, 20), 3, cv2.LINE_AA)
    # Add a dark border band to look like a plate frame.
    cv2.rectangle(img, (0, h - 12), (w, h), (40, 40, 40), -1)
    return img


def make_vehicle_frame(plate: str = "HR26AB1234") -> np.ndarray:
    """Build a synthetic frame with a car-like body and a visible plate."""
    frame = np.full((480, 720, 3), 120, dtype=np.uint8)
    # Car body (gray rectangle).
    cv2.rectangle(frame, (120, 200), (600, 420), (150, 150, 150), -1)
    # Windows.
    cv2.rectangle(frame, (180, 220), (330, 300), (200, 200, 200), -1)
    cv2.rectangle(frame, (390, 220), (540, 300), (200, 200, 200), -1)
    # Wheels.
    cv2.circle(frame, (200, 420), 40, (40, 40, 40), -1)
    cv2.circle(frame, (520, 420), 40, (40, 40, 40), -1)
    # Plate stuck on the lower bumper.
    plate_patch = make_plate_patch(plate)
    fh, fw = plate_patch.shape[:2]
    frame[350:350 + fh, 260:260 + fw] = plate_patch
    return frame


def main() -> None:
    from modules.anpr_video import VideoANPREngine

    frame = make_vehicle_frame("HR26AB1234")
    cv2.imwrite("/tmp/anpr_vehicle_frame.png", frame)

    engine = VideoANPREngine(confidence=0.35, debug=True)
    # Prepare the dedicated plate model (shows progress + validates).
    try:
        engine.prepare(progress=lambda msg, frac: log.info("  model: %s", msg))
        log.info("Plate model status: %s", engine.plate_model_status)
    except Exception as exc:
        log.error("PLATE MODEL NOT READY: %s", exc)
        sys.exit(1)

    result = engine.process(frame)

    log.info("Vehicles detected: %d", result.vehicle_count)
    log.info("Plates counter:   %d", result.plate_count)
    log.info("Debug model:      %s", result.debug.get("model"))

    ok_vehicle = result.vehicle_count > 0
    ok_plate = result.plate_count > 0

    # Print per-vehicle plates.
    for v in result.vehicles:
        log.info("  vehicle #%s (%s) plate=%s conf=%.3f",
                 v.track_id, v.vehicle_type,
                 v.plate.text if v.plate else None,
                 v.plate.confidence if v.plate else 0.0)

    print("\n===== PLATE PIPELINE TEST =====")
    print(f"Vehicle detected : {'PASS' if ok_vehicle else 'FAIL'}")
    print(f"Plate detected   : {'PASS' if ok_plate else 'FAIL'}")
    ok = ok_vehicle and ok_plate
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
