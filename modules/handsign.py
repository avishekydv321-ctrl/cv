"""Sign-language alphabet recognition using MediaPipe Tasks hand landmarks.

The static hand gesture classifier maps finger states and relative landmark
geometry to ASL alphabet letters. It reuses the shared Tasks hand landmarker
from ``detector.py`` (``_get_hand_landmarker``) so it works on the modern
``mediapipe>=1.0`` Tasks API and runs fully on CPU with no extra model
download beyond the small hand_landmarker.task bundle.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from detector import _get_hand_landmarker
from modules.base import BaseModule, ModuleResult
from utils import Detection

log = logging.getLogger("modules.handsign")

# Finger tip landmarks (MediaPipe hand indices).
_TIP = (4, 8, 12, 16, 20)
_PIP = (3, 6, 10, 14, 18)

# ASL letter -> (thumb_open, index, middle, ring, pinky) finger pattern.
# 1 = extended, 0 = folded. Patterns are heuristic; unknown shapes are reported
# as "?" and can be refined with a trained classifier later.
_ASL_PATTERNS: dict[str, tuple[int, int, int, int, int]] = {
    "A": (0, 0, 0, 0, 0),
    "B": (0, 1, 1, 1, 1),
    "C": (1, 1, 1, 0, 0),
    "D": (0, 1, 0, 0, 0),
    "E": (0, 0, 0, 0, 0),
    "F": (1, 1, 0, 0, 0),
    "G": (0, 1, 0, 0, 0),
    "H": (0, 1, 1, 0, 0),
    "I": (0, 0, 0, 0, 1),
    "K": (0, 1, 1, 0, 0),
    "L": (1, 1, 0, 0, 0),
    "M": (0, 0, 0, 1, 1),
    "N": (0, 0, 0, 0, 1),
    "O": (1, 0, 0, 0, 0),
    "P": (0, 1, 1, 0, 1),
    "Q": (0, 1, 0, 0, 1),
    "R": (0, 1, 1, 0, 0),
    "S": (0, 0, 0, 0, 0),
    "T": (0, 0, 0, 0, 0),
    "U": (0, 1, 1, 0, 0),
    "V": (0, 1, 1, 0, 0),
    "W": (0, 1, 1, 1, 0),
    "Y": (0, 0, 0, 0, 1),
}


class SignLanguage(BaseModule):
    """Recognise static ASL alphabet letters from a video frame."""

    key = "sign_language"
    color = (180, 140, 255)

    def __init__(self) -> None:
        super().__init__()
        self._hands = None

    def _check_available(self) -> bool:
        return self._ensure() is not None

    def _ensure(self):
        if self._hands is None:
            landmarker = _get_hand_landmarker()
            self._hands = landmarker if landmarker.available else None
        return self._hands

    @staticmethod
    def _fingers_up(landmarks, side: str) -> tuple[int, int, int, int, int]:
        # thumb: open if tip is further out from the palm centre along x-axis
        thumb = int(landmarks[4].x > landmarks[3].x if side == "Right" else landmarks[4].x < landmarks[3].x)
        fingers = tuple(
            int(landmarks[t].y < landmarks[p].y) for t, p in zip(_TIP[1:], _PIP[1:])
        )
        return (thumb, *fingers)

    def _classify(self, fingers: tuple[int, int, int, int, int]) -> tuple[str, float]:
        best, best_score = "?", 0.0
        for letter, pattern in _ASL_PATTERNS.items():
            score = sum(a == b for a, b in zip(fingers, pattern)) / len(pattern)
            if score > best_score:
                best, best_score = letter, score
        return best, best_score

    def process(self, frame: np.ndarray, config: Any) -> ModuleResult:
        t0 = time.perf_counter()
        result = ModuleResult()
        landmarker = self._ensure()
        if landmarker is None:
            result.info["available"] = False
            return result

        hands, sides = landmarker.detect(frame)
        letters = []
        for landmarks, side in zip(hands, sides):
            fingers = self._fingers_up(landmarks, side)
            letter, score = self._classify(fingers)
            letters.append(f"{side}:{letter}({score:.0%})")

            # bounding box from landmarks
            xs = [lm.x * frame.shape[1] for lm in landmarks]
            ys = [lm.y * frame.shape[0] for lm in landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            result.detections.append(
                Detection(f"sign:{letter}", score, (x1, y1, x2, y2), "sign", color=self.color)
            )

        result.info["signs"] = ", ".join(letters) or "No hands"
        result.inference_ms = (time.perf_counter() - t0) * 1000.0
        return result
