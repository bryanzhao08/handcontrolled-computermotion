"""
MediaPipe Hands wrapper — uses the new `mediapipe.tasks.vision` API
(mediapipe >= 0.10.13, which removed mp.solutions).

Uses RunningMode.IMAGE (per-frame, no timestamps required) for maximum
compatibility and reliability.

Returns a HandState dataclass per frame with:
  - 21 normalized landmark positions (wrist-relative, scale-invariant)
  - Raw mediapipe coords for drawing
"""
from __future__ import annotations

import dataclasses
import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# New tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# Default model path (relative to repo root)
_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
)


@dataclasses.dataclass
class HandState:
    """Snapshot of one hand in a single frame."""
    landmarks: np.ndarray       # (21, 3) wrist-relative, scale-invariant
    raw_landmarks: np.ndarray   # (21, 3) raw normalized coords in [0, 1]
    handedness: str             # 'left' or 'right'
    confidence: float


class HandTracker:
    """Wraps mediapipe.tasks.vision.HandLandmarker for per-frame hand tracking."""

    WRIST      = 0
    FINGERTIPS = [4, 8, 12, 16, 20]
    MCP_MIDDLE = 9

    # Connections for skeleton drawing
    _CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    def __init__(
        self,
        hand: str = "right",
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
        model_path: str = _DEFAULT_MODEL,
    ):
        self._target = hand.lower()   # 'right', 'left', or 'both'
        self._last_results = None

        abs_model = os.path.abspath(model_path)
        if not os.path.exists(abs_model):
            raise FileNotFoundError(
                f"hand_landmarker.task not found at: {abs_model}\n"
                "Download it with:\n"
                "  curl -L https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
                "-o models/hand_landmarker.task"
            )

        # IMAGE mode: processes each frame independently — no timestamp needed.
        # More reliable than VIDEO mode for live camera feeds.
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=abs_model),
            running_mode=RunningMode.IMAGE,
            num_hands=2,                                       # detect both, filter below
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        print(f"[HandTracker] ready — target={self._target!r}  model={os.path.basename(abs_model)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> Optional[HandState]:
        """Process one BGR frame; return HandState for target hand or None."""
        # Ensure contiguous uint8 RGB array
        rgb = np.ascontiguousarray(
            cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)
        self._last_results = result

        if not result.hand_landmarks:
            return None

        # Pick the first hand that matches the target side
        best_lm = None
        best_label = None
        best_score = 0.0

        for hand_lm, handedness_list in zip(result.hand_landmarks, result.handedness):
            cat = handedness_list[0]
            label = (cat.category_name or cat.display_name or "").lower()
            score = cat.score

            if self._target == "both" or label == self._target:
                if score > best_score:
                    best_lm    = hand_lm
                    best_label = label
                    best_score = score

        # Fallback: if target side wasn't found, take whichever hand was detected
        if best_lm is None and result.hand_landmarks:
            best_lm    = result.hand_landmarks[0]
            cat        = result.handedness[0][0]
            best_label = (cat.category_name or cat.display_name or "unknown").lower()
            best_score = cat.score
            print(f"[HandTracker] target={self._target!r} not found; using {best_label!r} (score={best_score:.2f})")

        if best_lm is None:
            return None

        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in best_lm],
            dtype=np.float32,
        )
        return HandState(
            landmarks=self._normalize(raw),
            raw_landmarks=raw,
            handedness=best_label,
            confidence=best_score,
        )

    def draw(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Draw the most recently detected hand skeleton onto bgr_frame."""
        if self._last_results is None or not self._last_results.hand_landmarks:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        for hand_lm in self._last_results.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            for a, b in self._CONNECTIONS:
                cv2.line(bgr_frame, pts[a], pts[b], (0, 220, 100), 2)
            for pt in pts:
                cv2.circle(bgr_frame, pt, 5, (255, 255, 255), -1)
        return bgr_frame

    @staticmethod
    def _normalize(raw: np.ndarray) -> np.ndarray:
        """Wrist-relative, scale-invariant normalization."""
        shifted = raw - raw[HandTracker.WRIST]
        scale = np.linalg.norm(shifted[HandTracker.MCP_MIDDLE]) + 1e-6
        return shifted / scale

    def close(self) -> None:
        self._landmarker.close()
