# VisionSuite models

Place YOLO weights here (auto-downloaded on first use).

## Directory layout
- `models/vehicle/` — YOLO COCO vehicle detector (e.g. `yolov8n.pt`). Used to
  detect vehicles. Downloaded automatically by Ultralytics.
- `models/license_plate/` — dedicated **license-plate** detector for ANPR.
  Stored file: `plate_detector_yolov8n.pt`.

## License-plate detector (ANPR)
A dedicated pretrained YOLOv8n fine-tuned for the `license_plate` class is used
for automatic number plate recognition. It is **separate** from the vehicle
detector.

- Source (public, non-gated, MIT):
  https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt
- Local path: `models/license_plate/plate_detector_yolov8n.pt`
- Auto-downloaded on first ANPR launch if missing; cached and reused after.
- If download/load fails, ANPR reports a clear error naming the model + source
  instead of silently reporting "Plate Not Found".
