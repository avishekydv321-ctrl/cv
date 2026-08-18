# VisionSuite — Real License-Plate ANPR Fix (Dedicated Plate Model + OCR)

## Status: COMPLETE ✅

## Root Cause
- `VideoANPREngine` looked for a nonexistent `models/plate_yolov8n.pt` and silently
  fell back to an unreliable OpenCV **contour** plate detector → "Plate Not Found".
- `_clean_plate_text()` applied destructive global char swaps (O→0, I→1, B→8, S→5,
  Z→2, G→6) that corrupted valid plates.
- No automatic download/validation of a real dedicated plate model.

## Plan
- [x] Validate a real, public, non-gated license-plate detection model (YOLOv8n,
      class `license_plate`) and confirm it loads with the installed Ultralytics.
- [x] Step 1 — New `modules/plate_detector.py` (auto-download + validate + detect).
- [x] Step 2 — Update `modules/anpr_video.py` to use the dedicated plate model as
      primary detector + fix OCR normalization + rich debug output.
- [x] Step 3 — Update `video_anpr.py` worker (progress + debug wiring).
- [x] Step 4 — Update `live_anpr.py` worker (progress + debug wiring).
- [x] Step 5 — Update `anpr_window.py` (ANPR Debug toggle + model status UI).
- [x] Step 6 — Update `live_anpr_window.py` (ANPR Debug toggle + model status UI).
- [x] Step 7 — Update `models/README.md` + `requirements.txt`.
- [x] Step 8 — End-to-end test on a real plate image (vehicle → plate → OCR → text).
      Dedicated plate model downloads, loads (status `loaded`), detects plates, and
      EasyOCR reads `HR26AB1234` @ 0.99 confidence; normalization/validation passes.
- [x] Step 9 — Byte-compile all changed modules + offscreen launch checks.
- [x] Step 10 — Plate-box padding in `_find_plates` (expand each detection ~8% width /
      15% height) so multi-line plates whose digit/state rows are cut off by a tight
      box are fully captured before OCR. Verified on real footage:
      vehicle image → plate `(68, 377, 180, 464)` → OCR `ALBR38462580`
      (matches saved `plate_..._BR38462580.jpg`). All modules compile cleanly.
