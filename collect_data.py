"""
Interactive gesture data recorder — 3-burst mode.

Usage:
    python collect_data.py

Controls:
  - Press the key shown for each gesture to select that label
  - Press SPACE to start 3 automatic recording bursts (no need to hold)
  - Press Q to quit and save all recorded data

Each SPACE press triggers `rounds_per_press` short bursts separated by a
countdown, forcing you to re-position your hand between rounds for more
varied training data.
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


# ── States ───────────────────────────────────────────────────────────────────

IDLE        = "idle"
COUNTDOWN   = "countdown"
RECORDING   = "recording"
ROUND_DONE  = "round_done"


def _draw_hud(
    frame: np.ndarray,
    state: str,
    current_label: str | None,
    round_idx: int,
    total_rounds: int,
    recorded_this_round: int,
    samples_per_round: int,
    countdown_secs_left: float,
    label_counts: dict[str, int],
    gesture_keys: dict[str, str],
    hand_detected: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    REC_CLR  = (30,  80, 255)
    IDLE_CLR = (50, 200,  80)
    CD_CLR   = (255, 160,  20)
    DIM_CLR  = (120, 120, 120)

    # ── Top status bar ──
    cv2.rectangle(overlay, (0, 0), (w, 56), (15, 15, 15), -1)
    if state == RECORDING:
        status = f"  ● REC  [{current_label}]  round {round_idx}/{total_rounds}"
        clr = REC_CLR
    elif state == COUNTDOWN:
        status = f"  GET READY  [{current_label}]  round {round_idx}/{total_rounds}  — {countdown_secs_left:.1f}s"
        clr = CD_CLR
    elif current_label:
        status = f"  READY  [{current_label}]  — press SPACE to record"
        clr = IDLE_CLR
    else:
        status = "  Select a gesture label with its key"
        clr = DIM_CLR
    cv2.putText(overlay, status, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, clr, 2)

    # ── Round dots ──
    if current_label and state in (RECORDING, COUNTDOWN, ROUND_DONE):
        dot_x = w - 180
        for r in range(total_rounds):
            filled = r < round_idx - 1 or (r == round_idx - 1 and state == ROUND_DONE)
            active = r == round_idx - 1 and state in (RECORDING, COUNTDOWN)
            c = (50, 200, 80) if filled else ((REC_CLR if state == RECORDING else CD_CLR) if active else (60, 60, 60))
            cv2.circle(overlay, (dot_x + r * 28, 36), 9, c, -1 if (filled or active) else 1)

    # ── Key legend ──
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
    target = samples_per_round * total_rounds
    for name, count in label_counts.items():
        fill = min(max_bar, int(count / max(target, 1) * max_bar))
        active = name == current_label
        bar_clr = (100, 220, 255) if active else (70, 130, 70)
        cv2.rectangle(overlay, (bx, by), (bx + fill, by + 16), bar_clr, -1)
        cv2.rectangle(overlay, (bx, by), (bx + max_bar, by + 16), (80, 80, 80), 1)
        cv2.putText(overlay, f" {name}: {count}", (bx + max_bar + 5, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        by += 24

    # ── Recording progress bar ──
    if state == RECORDING and samples_per_round > 0:
        prog = recorded_this_round / samples_per_round
        cv2.rectangle(overlay, (0, h - 12), (int(w * prog), h), REC_CLR, -1)
    elif state == COUNTDOWN:
        # Countdown progress bar (orange, counts down)
        prog = countdown_secs_left / 2.0
        cv2.rectangle(overlay, (0, h - 12), (int(w * prog), h), CD_CLR, -1)

    # ── No hand warning ──
    if not hand_detected:
        cv2.putText(overlay, "No hand detected", (w // 2 - 90, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

    return cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)


def main():
    config      = load_config()
    cam_cfg     = config.get("camera", {})
    data_cfg    = config.get("data",   {})
    model_cfg   = config.get("model",  {})
    gestures    = config["gestures"]

    gesture_keys: dict[str, str] = {v["key"]: name for name, v in gestures.items()}
    output_file      = data_cfg.get("output_file", "data/gestures.csv")
    samples_per_round = data_cfg.get("samples_per_round", 20)
    rounds_per_press  = data_cfg.get("rounds_per_press", 3)
    countdown_secs    = data_cfg.get("countdown_secs", 2.0)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    tracker   = HandTracker(hand=model_cfg.get("hand", "left"))
    extractor = FeatureExtractor(config)

    cap = cv2.VideoCapture(cam_cfg.get("device_id", 1))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height",  720))

    label_counts   = _count_existing(output_file, list(gestures.keys()))
    current_label: str | None = None
    new_rows: list[list] = []

    # State machine
    rec_state        = IDLE
    round_idx        = 0          # 1-indexed, current round number
    recorded_round   = 0          # frames captured this round
    countdown_end    = 0.0        # wall-clock time when countdown ends

    print("\n🤙  Gesture Data Recorder (3-burst mode) — press Q to quit & save\n")
    print(f"  Each SPACE press = {rounds_per_press} rounds × {samples_per_round} frames\n")
    for k, n in gesture_keys.items():
        print(f"  [{k.upper()}] → {n}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cam_cfg.get("flip", True):
            frame = cv2.flip(frame, 1)

        hand_state = tracker.process(frame)
        if hand_state is not None:
            tracker.draw(frame)
            features      = extractor.extract(hand_state)
            hand_detected = True
        else:
            features      = None
            hand_detected = False

        now = time.time()

        # ── State machine ─────────────────────────────────────────────────────
        if rec_state == COUNTDOWN:
            if now >= countdown_end:
                rec_state     = RECORDING
                recorded_round = 0
                print(f"  ● Round {round_idx}/{rounds_per_press} — recording '{current_label}'…")

        elif rec_state == RECORDING:
            if hand_state is not None and features is not None:
                new_rows.append([current_label] + features.tolist())
                recorded_round += 1

            if recorded_round >= samples_per_round:
                label_counts[current_label] = label_counts.get(current_label, 0) + recorded_round
                print(f"  ✓ Round {round_idx} done  ({recorded_round} frames, "
                      f"total {current_label}: {label_counts[current_label]})")
                extractor.reset()

                if round_idx >= rounds_per_press:
                    # All rounds complete
                    rec_state = IDLE
                    round_idx = 0
                    print(f"  🎉 All {rounds_per_press} rounds complete for '{current_label}'!\n")
                else:
                    # Countdown before next round
                    round_idx     += 1
                    rec_state      = COUNTDOWN
                    countdown_end  = now + countdown_secs
                    print(f"  ⏱  Get ready for round {round_idx}…")

        # ── Draw HUD ──────────────────────────────────────────────────────────
        secs_left = max(0.0, countdown_end - now) if rec_state == COUNTDOWN else 0.0
        frame = _draw_hud(
            frame, rec_state, current_label,
            round_idx, rounds_per_press,
            recorded_round, samples_per_round,
            secs_left, label_counts, gesture_keys, hand_detected,
        )
        cv2.imshow("Gesture Recorder  [Q = quit & save]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if rec_state != IDLE:
                pass   # ignore SPACE during active recording
            elif current_label is None:
                print("  ⚠ Select a label first")
            elif not hand_detected:
                print("  ⚠ No hand in frame — position your hand first")
            else:
                round_idx     = 1
                rec_state     = COUNTDOWN
                countdown_end = now + countdown_secs
                extractor.reset()
                print(f"  ⏱  Get ready for round 1 of {rounds_per_press}…")
        else:
            ch = chr(key) if key < 256 else ""
            if ch in gesture_keys:
                current_label  = gesture_keys[ch]
                rec_state      = IDLE
                round_idx      = 0
                extractor.reset()
                print(f"  → Label: '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    # ── Save ──
    if not new_rows:
        print("\n⚠ No new data recorded.")
        return

    file_exists  = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    feature_cols = [f"f{i}" for i in range(len(new_rows[0]) - 1)]
    with open(output_file, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["label"] + feature_cols)
        w.writerows(new_rows)
    print(f"\n✅ Saved {len(new_rows)} samples → '{output_file}'")


if __name__ == "__main__":
    main()
