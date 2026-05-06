"""
MediaPipe Hands wrapper.

Returns a HandState dataclass per frame with:
  - 21 normalized landmark positions (wrist-relative, scale-invariant)
  - Per-finger bend angles
  - Raw mediapipe coords for drawing
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclasses.dataclass
class HandState:
    """Snapshot of one hand in a single frame."""
    landmarks: np.ndarray       # (21, 3) wrist-relative, scale-invariant
    raw_landmarks: np.ndarray   # (21, 3) raw normalized coords in [0, 1]
    handedness: str             # 'Left' or 'Right'
    confidence: float


class HandTracker:
    """Wraps mediapipe.solutions.hands for per-frame hand tracking."""

    _MP = mp.solutions.hands
    _DRAW = mp.solutions.drawing_utils
    _STYLES = mp.solutions.drawing_styles

    WRIST = 0
    FINGERTIPS = [4, 8, 12, 16, 20]   # thumb → pinky
    MCP_MIDDLE = 9                      # used as scale reference

    def __init__(
        self,
        hand: str = "right",
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        side_map = {"right": "Right", "left": "Left", "both": None}
        self._target = side_map[hand.lower()]
        self._hands = self._MP.Hands(
            static_image_mode=False,
            max_num_hands=2 if hand == "both" else 1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_results = None

    def process(self, bgr_frame: np.ndarray) -> Optional[HandState]:
        """Process one BGR frame; return HandState for target hand or None."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._last_results = self._hands.process(rgb)

        if not self._last_results.multi_hand_landmarks:
            return None

        for hand_lm, hand_info in zip(
            self._last_results.multi_hand_landmarks,
            self._last_results.multi_handedness,
        ):
            label = hand_info.classification[0].label   # 'Left' or 'Right'
            score = hand_info.classification[0].score
            if self._target and label != self._target:
                continue

            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                dtype=np.float32,
            )
            return HandState(
                landmarks=self._normalize(raw),
                raw_landmarks=raw,
                handedness=label,
                confidence=score,
            )
        return None

    def draw(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Draw the most recently detected hand skeleton onto bgr_frame."""
        if self._last_results and self._last_results.multi_hand_landmarks:
            for hand_lm in self._last_results.multi_hand_landmarks:
                self._DRAW.draw_landmarks(
                    bgr_frame,
                    hand_lm,
                    self._MP.HAND_CONNECTIONS,
                    self._STYLES.get_default_hand_landmarks_style(),
                    self._STYLES.get_default_hand_connections_style(),
                )
        return bgr_frame

    @staticmethod
    def _normalize(raw: np.ndarray) -> np.ndarray:
        """Wrist-relative, scale-invariant normalization."""
        shifted = raw - raw[HandTracker.WRIST]
        scale = np.linalg.norm(shifted[HandTracker.MCP_MIDDLE]) + 1e-6
        return shifted / scale

    def close(self) -> None:
        self._hands.close()
