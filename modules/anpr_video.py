"""Video ANPR processing engine.

Detects vehicles in a video/live stream, tracks them across frames with stable
unique IDs, locates license plates (within the vehicle region), enhances the
crop for low-quality / night / blurry / tilted / distant / partial plates, and
reads the plate number with OCR.

Pipeline
--------
    Video / live frame
        -> Vehicle detector (YOLO COCO, unchanged)
        -> Vehicle tracking (ByteTrack / centroid)
        -> Vehicle ROI crop
        -> DEDICATED license-plate detector (YOLOv8n, class ``license_plate``)
        -> Plate crop
        -> Image enhancement (grayscale / CLAHE / denoise / sharpen /
           adaptive threshold / perspective)
        -> OCR (EasyOCR primary, Tesseract fallback)
        -> Validation + confidence
        -> Multi-frame voting per tracked vehicle
        -> GUI results

The dedicated plate detector is a completely separate model from the vehicle
detector. If it cannot be downloaded / loaded, a clear :class:`PlateModelError`
is surfaced (never silently reporting "Plate Not Found").
"""
from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from modules.base import get_device
from modules.plate_detector import (
    PLATE_MODEL_FILENAME,
    PlateModelError,
    get_plate_detector,
)
from tracker import CentroidTracker
from utils import Detection

log = logging.getLogger("modules.anpr_video")

# COCO class ids (and names) that count as vehicles.
_VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",  # some framings tag buses/vans oddly
    7: "truck",
    8: "boat",  # occasionally a van is tagged boat in edge models
}

# Cleaner display names.
_VEHICLE_NAMES: dict[str, str] = {
    "bicycle": "Bicycle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "train": "Train",
    "boat": "Boat",
    "van": "Van",
}

# Indian registration pattern (state, district, series, number) — used to
# validate plausible plates WITHOUT hard-coding a particular number.
_INDIAN_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")

ProgressFn = Callable[[str, float], None]


@dataclass
class PlateInfo:
    """Outcome of reading one license plate."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]  # frame-relative x1,y1,x2,y2
    image: Optional[np.ndarray] = None  # enhanced plate crop (BGR/gray)
    raw_text: str = ""  # raw OCR output before normalization


@dataclass
class VehicleDetection:
    """A single tracked vehicle in one frame, with optional plate read."""

    track_id: int
    vehicle_type: str
    confidence: float
    box: tuple[int, int, int, int]  # frame-relative x1,y1,x2,y2
    vehicle_crop: Optional[np.ndarray] = None
    plate: Optional[PlateInfo] = None


@dataclass
class ANPRFrameResult:
    """Everything the worker needs for one processed frame."""

    frame: np.ndarray  # annotated frame (BGR)
    fps: float
    vehicles: list[VehicleDetection]
    vehicle_count: int
    plate_count: int
    debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-track temporal voting state.
# ---------------------------------------------------------------------------
class _TrackState:
    """Accumulates OCR candidates for one tracked vehicle across frames."""

    def __init__(self, track_id: int, vehicle_type: str, first_seen: float) -> None:
        self.track_id = track_id
        self.vehicle_type = vehicle_type
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.candidates: dict[str, list[float]] = {}  # text -> [count, conf_sum]
        self.best_plate: Optional[PlateInfo] = None
        self.emitted = False

    def add_candidate(self, plate: PlateInfo) -> None:
        """Register one OCR read for this track and update the consensus."""
        self.last_seen = time.time()
        if not plate or not plate.text:
            return
        text = plate.text
        if text not in self.candidates:
            self.candidates[text] = [1, plate.confidence]
        else:
            self.candidates[text][0] += 1
            self.candidates[text][1] += plate.confidence
        self._maybe_finalize(plate)

    def _maybe_finalize(self, latest: PlateInfo) -> None:
        """Pick the strongest candidate; finalise when it is consistent enough."""
        if not self.candidates:
            return
        # Score = frequency, then average confidence.
        scored = sorted(
            ((count, conf_sum / count, text)
             for text, (count, conf_sum) in self.candidates.items()),
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )
        count, avg_conf, best_text = scored[0]
        # Finalise when seen multiple times OR high confidence, and not garbage.
        if len(best_text) >= 4 and (count >= 2 or avg_conf >= 0.65):
            if self.best_plate is None or avg_conf > self.best_plate.confidence:
                self.best_plate = PlateInfo(
                    text=best_text,
                    confidence=round(avg_conf, 4),
                    box=latest.box,
                    image=latest.image,
                    raw_text=latest.raw_text,
                )

    def final(self) -> Optional[PlateInfo]:
        """Return the finalised plate for this track (or None)."""
        return self.best_plate


# ---------------------------------------------------------------------------
# Image enhancement helpers.
# ---------------------------------------------------------------------------
def _enhance_plate_crop(crop: np.ndarray) -> list[np.ndarray]:
    """Return several preprocessing variants for OCR to choose from.

    Variants cover: upscaled grayscale, CLAHE, denoise+sharpen, adaptive
    threshold, Otsu binarisation, and (when possible) perspective correction.
    """
    if crop.size == 0:
        return []
    h, w = crop.shape[:2]
    if min(h, w) < 8:
        return []

    # Upscale for fine text recovery.
    scale = 1
    if w < 160:
        scale = 3
    elif w < 320:
        scale = 2
    if scale > 1:
        crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    variants: list[np.ndarray] = []

    # 1. Plain upscaled grayscale.
    variants.append(gray)

    # 2. CLAHE (low-light / shadow equalisation).
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    # 3. Denoise (bilateral) + sharpen.
    denoised = cv2.bilateralFilter(gray, 7, 40, 40)
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)
    variants.append(sharpened)

    # 4. CLAHE applied to the denoised+sharpened result.
    variants.append(clahe.apply(sharpened))

    # 5. Adaptive threshold of the enhanced image.
    enhanced = clahe.apply(sharpened)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15,
    )
    variants.append(thresh)

    # 6. Otsu binarisation of the enhanced image.
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 7. Perspective correction (if a quad can be found).
    warped = _try_perspective_correct(crop)
    if warped is not None:
        variants.append(warped)

    return variants


def _try_perspective_correct(img: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to deskew a tilted plate via a quadrilateral + warp."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blur = cv2.bilateralFilter(gray, 9, 17, 17)
        edged = cv2.Canny(blur, 50, 180)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            return None
        pts = approx.reshape(4, 2).astype("float32")
        if cv2.contourArea(approx) < 0.3 * img.shape[0] * img.shape[1]:
            return None
        # Order points: tl, tr, br, bl.
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        tl, tr, br, bl = (
            pts[np.argmin(s)], pts[np.argmin(diff)],
            pts[np.argmax(s)], pts[np.argmax(diff)],
        )
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 8 or height < 4:
            return None
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                       dtype="float32")
        matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
        return cv2.warpPerspective(img, matrix, (width, height))
    except Exception:  # pragma: no cover
        return None


def _clean_plate_text(text: str) -> str:
    """Normalise OCR output into a plausible plate string.

    Applies **context-aware** corrections: ambiguous characters are fixed based
    on whether they appear in a letter-run or a digit-run, so letters like ``O``
    are never blindly turned into ``0`` (which would corrupt a state code).
    """
    if not text:
        return ""
    s = text.upper()
    cleaned = "".join(c for c in s if c.isalnum())
    if not cleaned:
        return ""

    # Split into letter runs and digit runs.
    runs = re.findall(r"[A-Z]+|\d+", cleaned)
    if not runs:
        return cleaned

    out: list[str] = []
    for run in runs:
        if run.isdigit():
            corr = []
            for c in run:
                if c in "OQ":
                    corr.append("0")
                elif c in "IL":
                    corr.append("1")
                elif c == "B":
                    corr.append("8")
                elif c == "S":
                    corr.append("5")
                elif c == "Z":
                    corr.append("2")
                elif c == "G":
                    corr.append("6")
                elif c == "D":
                    corr.append("0")
                else:
                    corr.append(c)
            out.append("".join(corr))
        else:
            corr = []
            for c in run:
                if c == "0":
                    corr.append("O")
                elif c == "1":
                    corr.append("I")
                elif c == "5":
                    corr.append("S")
                elif c == "8":
                    corr.append("B")
                elif c == "2":
                    corr.append("Z")
                elif c == "6":
                    corr.append("G")
                else:
                    corr.append(c)
            out.append("".join(corr))
    return "".join(out)


def _is_plausible_plate(text: str) -> bool:
    """Validate a plate string without hard-coding a particular number.

    Accepts Indian registration formats (``HR26AB1234``, ``DL01AB1234``,
    ``UP32AB1234``) and generic letter+digit mixes (truck/van plates).
    """
    if not text or len(text) < 4 or len(text) > 12:
        return False
    has_digit = any(c.isdigit() for c in text)
    has_letter = any(c.isalpha() for c in text)
    if not (has_digit and has_letter):
        # Allow pure-digit truck plates (e.g. "1234") only if long enough.
        if not (has_digit and len(text) >= 5):
            return False
    # Reject runs of a single repeated character (e.g. "IIII", "0000").
    if len(set(text)) <= 2:
        return False

    # Indian registration format is strongly preferred.
    if _INDIAN_RE.match(text):
        return True

    # Generic fallback: at least 2 letters and 3 digits with a good mix.
    n_letters = sum(c.isalpha() for c in text)
    n_digits = sum(c.isdigit() for c in text)
    if n_letters >= 2 and n_digits >= 3:
        return True
    return False


def _ocr_easyocr(reader: Any, crop: np.ndarray) -> tuple[str, str, float]:
    """Read a plate with EasyOCR, trying multiple preprocess variants.

    Returns ``(raw_text, normalized_text, confidence)``.
    """
    try:
        variants = _enhance_plate_crop(crop)
        best_text, best_raw, best_conf = "", "", 0.0
        for variant in variants:
            results = reader.readtext(variant, detail=1, paragraph=False)
            if not results:
                continue
            blocks = [r for r in results if r[1] and _clean_plate_text(r[1])]
            if not blocks:
                continue
            raw = " ".join(r[1] for r in blocks)
            text = _clean_plate_text(raw)
            conf = float(np.mean([r[2] for r in blocks]))
            if _is_plausible_plate(text) and conf > best_conf:
                best_text, best_raw, best_conf = text, raw, conf
        return best_raw, best_text, best_conf
    except Exception as exc:  # pragma: no cover
        log.debug("EasyOCR read failed: %s", exc)
        return "", "", 0.0


def _ocr_tesseract(crop: np.ndarray) -> tuple[str, str, float]:
    """Read a plate with Tesseract (raw text + normalized text + confidence)."""
    try:
        import pytesseract

        variants = _enhance_plate_crop(crop)
        best_text, best_raw, best_conf = "", "", 0.0
        for variant in variants:
            data = pytesseract.image_to_data(
                variant, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                output_type=pytesseract.Output.DICT,
            )
            confs = [int(c) for c in data["conf"] if int(c) != -1]
            words = [w for w, c in zip(data["text"], data["conf"]) if int(c) != -1 and w.strip()]
            raw = " ".join(words)
            text = _clean_plate_text(raw)
            conf = float(np.mean(confs) / 100.0) if confs else 0.5
            if _is_plausible_plate(text) and conf > best_conf:
                best_text, best_raw, best_conf = text, raw, conf
        return best_raw, best_text, best_conf
    except Exception as exc:  # pragma: no cover
        log.debug("Tesseract OCR unavailable: %s", exc)
        return "", "", 0.0


# ---------------------------------------------------------------------------
# Plate detection (contour-based, last-resort fallback only).
# ---------------------------------------------------------------------------
def _find_plates_by_contour(roi: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate license-plate rectangles inside a vehicle ROI (fallback)."""
    plates: list[tuple[int, int, int, int]] = []
    h, w = roi.shape[:2]
    if h < 8 or w < 10:
        return plates

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    for lo, hi in ((30, 120), (50, 180), (70, 220)):
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, gw, gh = cv2.boundingRect(cnt)
            is_quad = len(approx) in (4, 5) or (len(approx) >= 4 and cv2.isContourConvex(approx))
            if not is_quad:
                continue
            aspect = gw / float(gh) if gh > 0 else 0
            if not (1.4 <= aspect <= 6.5):
                continue
            if gw < 15 or gh < 8 or gw > w * 0.95 or gh > h * 0.6:
                continue
            area = gw * gh
            if area < 60:
                continue
            plates.append((x, y, x + gw, y + gh))

    # Deduplicate overlapping boxes.
    return _dedup_boxes(plates)


def _dedup_boxes(boxes: list[tuple[int, int, int, int]],
                 iou_thresh: float = 0.4) -> list[tuple[int, int, int, int]]:
    """Deduplicate overlapping boxes by IoU-over-base ratio."""
    deduped: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        duplicate = False
        for (dx1, dy1, dx2, dy2) in deduped:
            ix = min(x2, dx2) - max(x1, dx1)
            iy = min(y2, dy2) - max(y1, dy1)
            if ix > 0 and iy > 0:
                inter = ix * iy
                base = max(1, (x2 - x1) * (y2 - y1))
                if inter / base > iou_thresh:
                    duplicate = True
                    break
        if not duplicate:
            deduped.append(box)
    return deduped


class VideoANPREngine:
    """Detect + track vehicles and read their license plates."""

    def __init__(self, confidence: float = 0.35, debug: bool = False) -> None:
        self.device = get_device()
        self.confidence = confidence
        self.debug = debug
        self._model: Optional[Any] = None
        self._plate: Optional[Any] = None  # dedicated plate detector
        self.plate_model_error: Optional[str] = None
        self.plate_model_status = "not_loaded"
        self._easyocr: Optional[Any] = None
        self._easyocr_attempted = False
        self._centroid = CentroidTracker(max_missed=15, max_distance=120.0)
        self._fps_meter: Optional[Any] = None
        self._track_states: "OrderedDict[int, _TrackState]" = OrderedDict()
        self._frame_count = 0
        self._ocr_interval = 2  # run OCR every N frames per track to save CPU
        log.info("VideoANPREngine initialised on device: %s", self.device)

    # -- lazy back-ends ---------------------------------------------------
    def _ensure_model(self) -> Optional[Any]:
        if self._model is None:
            try:
                from ultralytics import YOLO

                self._model = YOLO("yolov8n.pt")
            except Exception as exc:  # pragma: no cover
                log.warning("YOLO unavailable for video ANPR: %s", exc)
                self._model = None
        return self._model

    def prepare(self, progress: Optional[ProgressFn] = None) -> None:
        """Ensure the dedicated license-plate model is available.

        Raises :class:`PlateModelError` if the model cannot be downloaded or
        loaded, so callers never silently continue with a vehicle-only model
        and report "Plate Not Found".
        """
        try:
            detector = get_plate_detector()
            detector.ensure_available(progress=progress)
            self._plate = detector
            self.plate_model_status = detector.load_status
            self.plate_model_error = None
        except PlateModelError as exc:
            self.plate_model_status = "error"
            self.plate_model_error = str(exc)
            log.error("Plate model error: %s", exc)
            raise

    def _ensure_plate(self) -> Any:
        """Return the dedicated plate detector (lazily prepared)."""
        if self._plate is None:
            self.prepare()
        return self._plate

    @property
    def plate_model_name(self) -> str:
        return PLATE_MODEL_FILENAME

    def _ensure_easyocr(self) -> Optional[Any]:
        """Lazy-load EasyOCR once; never retry after a failure."""
        if self._easyocr_attempted:
            return self._easyocr
        self._easyocr_attempted = True
        try:
            import easyocr

            self._easyocr = easyocr.Reader(["en"], gpu=(self.device == "cuda"), verbose=False)
            log.info("EasyOCR loaded for ANPR.")
        except Exception as exc:  # pragma: no cover
            log.warning("EasyOCR unavailable, will use Tesseract: %s", exc)
            self._easyocr = None
        return self._easyocr

    def reset(self) -> None:
        """Reset tracking and per-track state between runs."""
        self._centroid.reset()
        self._track_states.clear()
        self._frame_count = 0

    # -- vehicle detection ---------------------------------------------
    def _detect_vehicles(self, frame: np.ndarray) -> list[Detection]:
        """Return vehicle detections (label includes track id when tracked)."""
        model = self._ensure_model()
        if model is None:
            return []

        try:
            try:
                results = model.track(
                    frame, conf=self.confidence, persist=True,
                    tracker="bytetrack.yaml", verbose=False, device=self.device,
                )
            except Exception:
                results = model.predict(frame, conf=self.confidence, verbose=False, device=self.device)

            detections: list[Detection] = []
            for result in results:
                names = result.names
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls not in _VEHICLE_CLASSES:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    track_id = int(box.id[0].item()) if box.id is not None else 0
                    conf = float(box.conf[0])
                    label = _VEHICLE_NAMES.get(names[cls], names[cls])
                    label_str = f"{label} #{track_id}" if track_id else label
                    detections.append(
                        Detection(label_str, conf, (x1, y1, x2, y2), "vehicle")
                    )
            return detections
        except Exception as exc:  # pragma: no cover
            log.debug("Vehicle detect error: %s", exc)
            return []

    # -- plate detection ------------------------------------------------
    def _find_plates(self, vehicle_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Find plate boxes within a vehicle crop.

        Primary path is the dedicated YOLOv8 license-plate detector. The
        contour detector is only used as a last-resort fallback and is clearly
        reported via the model status in debug output.
        """
        boxes: list[tuple[int, int, int, int]] = []

        # 1. Dedicated plate model (primary).
        try:
            plate = self._ensure_plate()
            if plate is not None:
                boxes = plate.detect(vehicle_crop, conf=0.25)
        except PlateModelError as exc:
            self.plate_model_status = "error"
            self.plate_model_error = str(exc)
            log.error("Dedicated plate model unavailable: %s", exc)
        except Exception as exc:  # pragma: no cover
            self.plate_model_status = "error"
            self.plate_model_error = str(exc)
            log.debug("Plate model predict error: %s", exc)

        # 2. Contour fallback (best-effort, reported in debug).
        if not boxes:
            boxes.extend(_find_plates_by_contour(vehicle_crop))

        # 3. Expand each plate box slightly so the crop captures the full
        #    plate (including multi-line plates whose bottom digit row or top
        #    state row can be cut off by a tight detection box).
        expanded: list[tuple[int, int, int, int]] = []
        h, w = vehicle_crop.shape[:2]
        for (x1, y1, x2, y2) in boxes:
            pw = max(4, int((x2 - x1) * 0.08))
            ph = max(4, int((y2 - y1) * 0.15))
            nx1, ny1 = max(0, x1 - pw), max(0, y1 - ph)
            nx2, ny2 = min(w, x2 + pw), min(h, y2 + ph)
            expanded.append((nx1, ny1, nx2, ny2))

        # 4. Deduplicate (on the expanded boxes).
        return _dedup_boxes([b for b in expanded if b[2] - b[0] >= 15 and b[3] - b[1] >= 8], 0.5)

    # -- OCR -------------------------------------------------------------
    def _read_plate(self, crop: np.ndarray) -> Optional[PlateInfo]:
        """OCR a plate crop, returning the best valid plate read."""
        best_text, best_raw, best_conf = "", "", 0.0

        # Try EasyOCR first (better for challenging footage / Indian plates).
        reader = self._ensure_easyocr()
        if reader is not None:
            raw, text, conf = _ocr_easyocr(reader, crop)
            if text and conf > best_conf:
                best_text, best_raw, best_conf = text, raw, conf

        # Fall back to / compare with Tesseract.
        if not best_text or best_conf < 0.5:
            raw, text, conf = _ocr_tesseract(crop)
            if text and conf > best_conf:
                best_text, best_raw, best_conf = text, raw, conf

        if _is_plausible_plate(best_text):
            return PlateInfo(
                best_text,
                round(best_conf, 4),
                (0, 0, crop.shape[1], crop.shape[0]),
                crop,
                raw_text=best_raw,
            )
        return None

    # -- per-track plumbing ----------------------------------------------
    def _get_track(self, track_id: int, vehicle_type: str) -> _TrackState:
        if track_id not in self._track_states:
            self._track_states[track_id] = _TrackState(track_id, vehicle_type, time.time())
        return self._track_states[track_id]

    def _prune_tracks(self) -> None:
        """Drop tracks not seen recently to bound memory."""
        stale = [tid for tid, st in self._track_states.items()
                 if time.time() - st.last_seen > 5.0]
        for tid in stale:
            del self._track_states[tid]

    # -- main entry -------------------------------------------------------
    def process(self, frame: np.ndarray,
                progress: Optional[ProgressFn] = None) -> ANPRFrameResult:
        """Process one frame: detect/track vehicles and read their plates."""
        if self._fps_meter is None:
            from utils import FPSMeter

            self._fps_meter = FPSMeter()
        self._frame_count += 1
        self._prune_tracks()

        detections = self._detect_vehicles(frame)
        tracks = self._centroid.update(detections)

        vehicles: list[VehicleDetection] = []
        for det in detections:
            # Determine track id (from label "#N" or centroid tracker).
            track_id = 0
            if " #" in det.label:
                try:
                    track_id = int(det.label.split(" #")[1])
                except ValueError:
                    track_id = 0
            if track_id == 0:
                cx = (det.box[0] + det.box[2]) // 2
                cy = (det.box[1] + det.box[3]) // 2
                for tid, tr in tracks.items():
                    if abs(tr.centroid[0] - cx) < 5 and abs(tr.centroid[1] - cy) < 5:
                        track_id = tid
                        break

            label = det.label.split(" #")[0]
            vehicle_type = _VEHICLE_NAMES.get(label, label)

            x1, y1, x2, y2 = det.box
            hh, ww = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ww, x2), min(hh, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            state = self._get_track(track_id, vehicle_type)

            # Run OCR on a sampling interval to save CPU (CPU-only set-up).
            run_ocr = self._frame_count % self._ocr_interval == 0 or state.final() is None
            best_read: Optional[PlateInfo] = None
            if run_ocr:
                for (px1, py1, px2, py2) in self._find_plates(crop):
                    if px2 - px1 < 15 or py2 - py1 < 8:
                        continue
                    plate_crop = crop[py1:py2, px1:px2]
                    read = self._read_plate(plate_crop)
                    if read is None:
                        continue
                    # Convert crop-relative to frame-relative box.
                    read.box = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                    if best_read is None or read.confidence > best_read.confidence:
                        best_read = read
                if best_read is not None:
                    state.add_candidate(best_read)

            # Attach the finalised plate (kept once consensus is reached).
            final_plate = state.final()
            if final_plate is not None:
                final_plate.image = best_read.image if best_read is not None else final_plate.image
            vehicle = VehicleDetection(track_id, vehicle_type, det.confidence, (x1, y1, x2, y2), crop, final_plate)
            vehicles.append(vehicle)

        result = ANPRFrameResult(
            frame=frame,
            fps=self._fps_meter.tick(),
            vehicles=vehicles,
            vehicle_count=len(vehicles),
            plate_count=sum(1 for v in vehicles if v.plate and v.plate.text),
        )
        result.debug = self._collect_debug(frame, vehicles)
        if self.debug:
            result.debug["model"] = {
                "name": self.plate_model_name,
                "status": self.plate_model_status,
                "error": self.plate_model_error or "",
                "ocr_engine": "EasyOCR" if self._easyocr is not None else "Tesseract",
            }
        return result

    def _collect_debug(self, frame: np.ndarray,
                       vehicles: list[VehicleDetection]) -> dict[str, Any]:
        """Assemble debug info for the current frame (best-effort)."""
        info: dict[str, Any] = {}
        for v in vehicles:
            if v.plate and v.plate.text:
                plate_item = {
                    "plate": v.plate.text,
                    "raw_ocr": v.plate.raw_text,
                    "confidence": v.plate.confidence,
                    "vehicle_type": v.vehicle_type,
                    "track_id": v.track_id,
                    "box": list(v.box),
                    "vehicle_box": list(v.box),
                }
                info[f"track_{v.track_id}"] = plate_item
        return info


def result_placeholder(fps: float):
    """Build an empty ANPRFrameResult for the final 'finished' signal."""
    return ANPRFrameResult(frame=np.zeros((4, 4, 3), dtype=np.uint8), fps=fps,
                           vehicles=[], vehicle_count=0, plate_count=0)
