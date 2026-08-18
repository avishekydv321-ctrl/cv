"""Classic image-processing filters applied before/after detection."""
from __future__ import annotations

import cv2
import numpy as np

_SEPIA = np.array(
    [[0.272, 0.534, 0.131],
     [0.349, 0.686, 0.168],
     [0.393, 0.769, 0.189]],
    dtype=np.float32,
)


def apply_filter(frame: np.ndarray, name: str, strength: int = 5) -> np.ndarray:
    """Apply a named filter. Always returns a 3-channel BGR image."""
    if name == "Grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if name == "Blur":
        k = max(3, strength * 2 + 1)
        return cv2.GaussianBlur(frame, (k, k), 0)

    if name == "Canny Edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if name == "Threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if name == "Sepia":
        out = cv2.transform(frame.astype(np.float32), _SEPIA)
        return np.clip(out, 0, 255).astype(np.uint8)

    if name == "Histogram Equalization":
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return frame


def remove_background(frame: np.ndarray, mask: np.ndarray, blur: int = 35) -> np.ndarray:
    """Composite the subject over a blurred copy of the scene."""
    k = blur if blur % 2 else blur + 1
    background = cv2.GaussianBlur(frame, (k, k), 0)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    return np.where(mask3 > 0.5, frame, background).astype(np.uint8)


def histogram_image(frame: np.ndarray, size: tuple[int, int] = (256, 140)) -> np.ndarray:
    """Render a small BGR histogram widget for the analytics panel."""
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for channel, color in enumerate(((255, 80, 80), (80, 255, 120), (80, 140, 255))):
        hist = cv2.calcHist([frame], [channel], None, [width], [0, 256]).flatten()
        hist = cv2.normalize(hist, None, 0, height - 4, cv2.NORM_MINMAX)
        points = np.array([[x, height - int(v)] for x, v in enumerate(hist)], np.int32)
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
    return canvas


class ImageStabilizer:
    """Stabilizes camera jitters by tracking translation motion via optical flow."""

    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        # Find keypoints in the previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=50, qualityLevel=0.01, minDistance=15
        )
        if prev_pts is not None and len(prev_pts) > 5:
            # Track them to the current frame
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
            idx = np.where(status == 1)[0]
            if len(idx) > 5:
                p_pts = prev_pts[idx]
                n_pts = next_pts[idx]
                # Estimate translation motion (rotation/scale ignored for speed)
                M, _ = cv2.estimateAffinePartial2D(n_pts, p_pts)
                if M is not None:
                    stabilized = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                    self.prev_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
                    return stabilized

        self.prev_gray = gray
        return frame

    def reset(self) -> None:
        self.prev_gray = None


class ImageEnhancer:
    """Consolidated image enhancer applying configured CV preprocessing filters."""

    def __init__(self) -> None:
        self.stabilizer = ImageStabilizer()

    def enhance(self, frame: np.ndarray, config: Any) -> np.ndarray:
        out = frame.copy()

        # 1. Image Stabilization
        if getattr(config, "stabilization", False):
            out = self.stabilizer.stabilize(out)
        else:
            self.stabilizer.reset()

        # 2. Auto Exposure Correction
        if getattr(config, "exposure_correction", False):
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            if mean_brightness > 5.0:
                target = 120.0
                factor = np.clip(target / mean_brightness, 0.6, 1.8)
                table = np.array([np.clip(i * factor, 0, 255) for i in range(256)]).astype(np.uint8)
                out = cv2.LUT(out, table)

        # 3. Auto White Balance (Gray World algorithm)
        if getattr(config, "white_balance", False):
            b, g, r = cv2.split(out.astype(np.float32))
            mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
            if mean_b > 5.0 and mean_g > 5.0 and mean_r > 5.0:
                mean_gray = (mean_b + mean_g + mean_r) / 3.0
                b = np.clip(b * (mean_gray / mean_b), 0, 255)
                g = np.clip(g * (mean_gray / mean_g), 0, 255)
                r = np.clip(r * (mean_gray / mean_r), 0, 255)
                out = cv2.merge([b, g, r]).astype(np.uint8)

        # 4. Adaptive Histogram Equalization (CLAHE) for low light / contrast
        if getattr(config, "clahe", False):
            ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # 5. Denoising / Bilateral filtering
        if getattr(config, "denoising", False):
            # Lightweight bilateral filter to preserve edges
            out = cv2.bilateralFilter(out, 5, 30, 30)

        # 6. Image Sharpening / Deblurring
        if getattr(config, "sharpening", False):
            blurred = cv2.GaussianBlur(out, (5, 5), 1.0)
            out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)

        return out

