"""
Converts a HandState → a fixed-length numpy feature vector.

Feature layout (all float32):
  [0  : 63]            21 landmarks × 3 (wrist-relative, scale-invariant)
  [63 : 68]            5 per-finger bend angles (cosine)
  [68 : 68 + 63*(H-1)] velocity = frame-to-frame delta for (history_frames-1) steps
                       (only included if use_velocity=True and history_frames > 1)

The same extractor instance must be used at collection time and inference time
to avoid train/serve skew — it carries the frame history buffer internally.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from gesture_engine.hand_tracker import HandState


# ── Finger angle helper ───────────────────────────────────────────────────────

# (mcp_idx, pip_idx, dip_idx, tip_idx) for each finger
_FINGER_JOINTS = [
    (1,  2,  3,  4),   # thumb
    (5,  6,  7,  8),   # index
    (9,  10, 11, 12),  # middle
    (13, 14, 15, 16),  # ring
    (17, 18, 19, 20),  # pinky
]


def _finger_bend_angles(landmarks: np.ndarray) -> np.ndarray:
    """Return cosine of bend angle for each of the 5 fingers (shape: [5,])."""
    angles = []
    for mcp, pip, dip, _ in _FINGER_JOINTS:
        v1 = landmarks[pip] - landmarks[mcp]
        v2 = landmarks[dip] - landmarks[pip]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        cos_a = np.dot(v1, v2) / denom
        angles.append(float(np.clip(cos_a, -1.0, 1.0)))
    return np.array(angles, dtype=np.float32)


# ── Extractor ─────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Stateful extractor — maintains a rolling frame-history buffer.

    Parameters come from the 'model' section of config.yaml:
      use_velocity   (bool)  — include inter-frame velocity features
      history_frames (int)   — number of frames kept for velocity
    """

    def __init__(self, config: dict):
        cfg = config.get("model", {})
        self.use_velocity: bool = cfg.get("use_velocity", True)
        self.history_frames: int = max(1, cfg.get("history_frames", 5))
        self._history: deque[np.ndarray] = deque(maxlen=self.history_frames)

    @classmethod
    def from_config_file(cls, path: str = "config.yaml") -> "FeatureExtractor":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    @property
    def feature_size(self) -> int:
        base = 63 + 5
        if self.use_velocity and self.history_frames > 1:
            base += 63 * (self.history_frames - 1)
        return base

    def extract(self, state: "HandState") -> np.ndarray:
        """Return a 1-D float32 feature vector from a HandState."""
        flat = state.landmarks.flatten()          # 63
        angles = _finger_bend_angles(state.landmarks)  # 5
        self._history.append(flat.copy())

        if self.use_velocity and self.history_frames > 1:
            hist = list(self._history)
            deltas = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
            # Pad with zeros if history not yet full
            while len(deltas) < self.history_frames - 1:
                deltas.insert(0, np.zeros(63, dtype=np.float32))
            vel = np.concatenate(deltas[-(self.history_frames - 1):])
        else:
            vel = np.array([], dtype=np.float32)

        return np.concatenate([flat, angles, vel]).astype(np.float32)

    def reset(self) -> None:
        """Clear history (call when switching gesture labels during collection)."""
        self._history.clear()
