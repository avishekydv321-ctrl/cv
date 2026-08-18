"""OCR / scanning modules: ANPR, QR & barcode, and document scanner.

All text extraction uses Tesseract (pytesseract) when present, otherwise the
module reports itself unavailable. Document boundary detection uses classic
OpenCV contour detection, so perspective correction works without OCR.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from modules.base import BaseModule, ModuleResult
from utils import Detection

log = logging.getLogger("modules.ocr")


def _ocr_text(frame: np.ndarray) -> str:
    """Run Tesseract on a frame; return '' if unavailable."""
    try:
        import pytesseract

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(gray).strip()
    except Exception as exc:  # pragma: no cover
        log.debug("OCR unavailable: %s", exc)
        return ""


def _enhance_plate_crop(crop: np.ndarray) -> np.ndarray:
    """Apply upscaling, bilateral denoising, CLAHE and Otsu thresholding to plate crops."""
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    # Upscale 3x for OCR clarity
    crop = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Deblur / Sharpen
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    # Adaptive Equalization (CLAHE) for low light / shadows
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    # Binarize
    thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh


def _ocr_plate_with_confidence(crop: np.ndarray) -> tuple[str, float]:
    """OCR the crop and return alphanumeric text and mean confidence."""
    try:
        import pytesseract
        enhanced = _enhance_plate_crop(crop)
        data = pytesseract.image_to_data(enhanced, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data['conf'] if int(c) != -1]
        words = [w for w, c in zip(data['text'], data['conf']) if int(c) != -1 and w.strip()]
        text = "".join(c for c in "".join(words) if c.isalnum()).upper()
        # Clean up commonly misidentified characters
        confidence = float(np.mean(confs) / 100.0) if confs else 0.5
        return text, confidence
    except Exception:
        return "", 0.0


class ANPR(BaseModule):
    """Automatic Number Plate Recognition: detect plate + OCR it."""

    key = "anpr"
    color = (255, 200, 120)

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._cascade = None

    def _ensure(self):
        if self._cascade is None:
            try:
                # Haar cascade for plate detection (OpenCV ships haarcascade_russian_plate_number).
                cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
                import os

                if os.path.exists(cascade_path):
                    self._cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self._cascade = None
        return self._cascade

    def _check_available(self) -> bool:
        try:
            import pytesseract  # noqa: F401
            return True  # Always fallback to contour detector even if cascade is missing
        except Exception:
            return False

    def _find_plates_by_contour(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Contour-based plate locator for distant, angled, or low-contrast plates."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        plates = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, gw, gh = cv2.boundingRect(approx)
                aspect = gw / float(gh) if gh > 0 else 0
                if 2.0 <= aspect <= 5.5 and 45 <= gw <= 350 and 12 <= gh <= 100:
                    overlap = False
                    for (px, py, pw, ph) in plates:
                        if abs(x - px) < 40 and abs(y - py) < 40:
                            overlap = True
                            break
                    if not overlap:
                        plates.append((x, y, gw, gh))
        return plates

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()

        # 1. Try Haar Cascade first
        cascade = self._ensure()
        plates = []
        if cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = list(cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(70, 20)
            ))

        # 2. Supplementary contour-based plates (adds distant/angled plates)
        contours_found = self._find_plates_by_contour(frame)
        for (cx, cy, cw, ch) in contours_found:
            # check if overlaps with cascade detections
            is_new = True
            for (hx, hy, hw, hh) in plates:
                if abs(cx - hx) < 30 and abs(cy - hy) < 30:
                    is_new = False
                    break
            if is_new:
                plates.append((cx, cy, cw, ch))

        # 3. Process crops
        for (x, y, w, h) in plates:
            # Clip bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            text, confidence = _ocr_plate_with_confidence(crop)
            label = text or "plate"
            result.detections.append(
                Detection(f"plate:{label}", confidence, (x1, y1, x2, y2), "anpr", color=self.color)
            )
            if text:
                result.info["plate"] = text

        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class QRCodeModule(BaseModule):
    """Scan QR codes and barcodes (pyzbar with OpenCV fallback)."""

    key = "qr_codes"
    color = (200, 200, 255)

    def __init__(self) -> None:
        super().__init__()
        self._qr = cv2.QRCodeDetector()

    def _check_available(self) -> bool:
        return True  # OpenCV QR fallback always works

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        codes = []
        try:
            from pyzbar import pyzbar

            for code in pyzbar.decode(frame):
                x, y, w, h = code.rect
                data = code.data.decode("utf-8", "ignore")
                result.detections.append(
                    Detection(data, 1.0, (x, y, x + w, y + h), "code", color=self.color)
                )
                codes.append(data)
        except Exception:
            # OpenCV fallback - guard against failures when no code is present
            # (e.g. OpenCV 5.x raises a convexHull assertion error on empty input).
            try:
                data, points, _ = self._qr.detectAndDecode(frame)
            except Exception as exc:  # pragma: no cover
                log.debug("OpenCV QR fallback unavailable: %s", exc)
                data, points = None, None
            if data and points is not None:
                pts = points.reshape(-1, 2).astype(int)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                result.detections.append(
                    Detection(data, 1.0, (int(x1), int(y1), int(x2), int(y2)), "code", color=self.color)
                )
                codes.append(data)
        result.info["codes"] = ", ".join(codes) or "\u2014"
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result


class DocumentScanner(BaseModule):
    """Detect a document, apply perspective correction, extract text."""

    key = "document_scan"
    color = (255, 255, 255)

    def _check_available(self) -> bool:
        return True  # perspective correction is pure OpenCV

    def _largest_quad(self, gray: np.ndarray) -> np.ndarray | None:
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 2000:
                if best is None or cv2.contourArea(approx) > cv2.contourArea(best):
                    best = approx
        return best

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                       dtype="float32")
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad = self._largest_quad(gray)
        if quad is not None:
            pts = quad.reshape(4, 2).astype(int)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            result.detections.append(
                Detection("document", 1.0, (int(x1), int(y1), int(x2), int(y2)), "document", color=self.color)
            )
            h, w = frame.shape[:2]
            result.detections.append(
                Detection("document_scanned", 1.0, (0, 0, w, h), "document", color=self.color)
            )
            text = _ocr_text(frame[y1:y2, x1:x2])
            if text:
                result.info["text"] = text
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def scan_file(self, path: Path) -> str:
        """Perspective-correct and OCR a document image file."""
        frame = cv2.imread(str(path))
        if frame is None:
            return ""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad = self._largest_quad(gray)
        if quad is not None:
            frame = self._four_point_transform(frame, quad.reshape(4, 2))
        return _ocr_text(frame)
