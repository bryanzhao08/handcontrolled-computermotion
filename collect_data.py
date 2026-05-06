"""
Interactive gesture data recorder.

Usage:
    python collect_data.py

Controls (shown in the live window):
  - Press the key shown for each gesture to select that label
  - Hold SPACE to record a batch of frames with the current label
  - Press Q to quit and save all recorded data
"""
from __future__ import annotations

import csv
import os
import time

import cv2
import numpy as np
import yaml

from gesture_engine.hand_tracker import HandTracker
from gesture_engine.feature_extractor import FeatureExtractor


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _count_existing(output_file: str, gesture_names: list[str]) -> dict[str, int]:
    counts = {n: 0 for n in gesture_names}
    if not os.path.exists(output_file):
        return counts
    with open(output_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label", "")
            if lbl in counts:
                counts[lbl] += 1
    return counts


def _draw_hud(
    frame: np.ndarray,
    current_label: str | None,
    recording: bool,
    recorded_this_batch: int,
    samples_per_batch: int,
    label_counts: dict[str, int],
    gesture_keys: dict[str, str],   # key_char → gesture_name
    hand_detected: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    REC_CLR  = (30,  80, 255)
    IDLE_CLR = (50, 200,  80)
    DIM_CLR  = (120, 120, 120)

    # ── Top status bar ──
    cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 15), -1)
    if recording:
        status = f"  ● REC  [{current_label}]"
        clr = REC_CLR
    elif current_label:
        status = f"  READY  [{current_label}]  — hold SPACE to record"
        clr = IDLE_CLR
    else:
        status = "  Select a gesture label with its key"
        clr = DIM_CLR
    cv2.putText(overlay, status, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)

    # ── Key legend (right side) ──
    lx, ly = w - 220, 80
    cv2.putText(overlay, "Gesture Keys:", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    ly += 22
    for key_ch, name in gesture_keys.items():
        active = name == current_label
        c = (100, 220, 255) if active else (160, 160, 160)
        cv2.putText(overlay, f"  [{key_ch.upper()}] {name}", (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        ly += 20

    # ── Sample count bars ──
    bx, by = 15, 80
    cv2.putText(overlay, "Samples collected:", (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    by += 22
    max_bar = 180
    for name, count in label_counts.items():
        fill = min(max_bar, int(count / max(samples_per_batch, 1) * max_bar))
        active = name == current_label
        bar_clr = (100, 220, 255) if active else (70, 130, 70)
        cv2.rectangle(overlay, (bx, by), (bx + fill, by + 16), bar_clr, -1)
        cv2.rectangle(overlay, (bx, by), (bx + max_bar, by + 16), (80, 80, 80), 1)
        cv2.putText(overlay, f" {name}: {count}", (bx + max_bar + 5, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        by += 24

    # ── Recording progress bar ──
    if recording and samples_per_batch > 0:
        prog = recorded_this_batch / samples_per_batch
        cv2.rectangle(overlay, (0, h - 12), (int(w * prog), h), REC_CLR, -1)

    # ── No hand warning ──
    if not hand_detected:
        cv2.putText(overlay, "No hand detected", (w // 2 - 90, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

    return cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)


def main():
    config = load_config()
    cam_cfg     = config.get("camera", {})
    data_cfg    = config.get("data",   {})
    model_cfg   = config.get("model",  {})
    gestures    = config["gestures"]

    gesture_keys: dict[str, str] = {v["key"]: name for name, v in gestures.items()}
    output_file  = data_cfg.get("output_file", "data/gestures.csv")
    samples_per_batch = data_cfg.get("samples_per_gesture", 60)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    tracker   = HandTracker(hand=model_cfg.get("hand", "right"))
    extractor = FeatureExtractor(config)

    cap = cv2.VideoCapture(cam_cfg.get("device_id", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height",  720))

    label_counts   = _count_existing(output_file, list(gestures.keys()))
    current_label: str | None = None
    recording       = False
    recorded_batch  = 0
    new_rows: list[list] = []

    print("\n🤙  Gesture Data Recorder — press Q to quit & save\n")
    for k, n in gesture_keys.items():
        print(f"  [{k.upper()}] → {n}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cam_cfg.get("flip", True):
            frame = cv2.flip(frame, 1)

        state = tracker.process(frame)
        if state is not None:
            tracker.draw(frame)
            features = extractor.extract(state)
            hand_detected = True
        else:
            features = None
            hand_detected = False

        # Recording logic
        if recording:
            if state is not None and features is not None:
                new_rows.append([current_label] + features.tolist())
                recorded_batch += 1
            if recorded_batch >= samples_per_batch:
                recording = False
                label_counts[current_label] = label_counts.get(current_label, 0) + recorded_batch
                print(f"  ✓ {recorded_batch} frames for '{current_label}' "
                      f"(total: {label_counts[current_label]})")
                recorded_batch = 0

        frame = _draw_hud(
            frame, current_label, recording, recorded_batch,
            samples_per_batch, label_counts, gesture_keys, hand_detected,
        )
        cv2.imshow("Gesture Recorder  [Q = quit & save]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if current_label is None:
                print("  ⚠ Select a label first")
            elif not hand_detected:
                print("  ⚠ No hand in frame — position your hand first")
            else:
                recording = True
                recorded_batch = 0
                print(f"  ● Recording '{current_label}'…")
        else:
            ch = chr(key) if key < 256 else ""
            if ch in gesture_keys:
                current_label = gesture_keys[ch]
                extractor.reset()
                recording = False
                recorded_batch = 0
                print(f"  → Label: '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # ── Save ──
    if not new_rows:
        print("\n⚠ No new data recorded.")
        return

    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    feature_cols = [f"f{i}" for i in range(len(new_rows[0]) - 1)]
    with open(output_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["label"] + feature_cols)
        w.writerows(new_rows)
    print(f"\n✅ Saved {len(new_rows)} samples → '{output_file}'")


if __name__ == "__main__":
    main()
