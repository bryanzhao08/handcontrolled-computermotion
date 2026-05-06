"""
Live gesture recognition → macOS system control.

Usage:
    python main.py

Reads models/ and config.yaml, opens the camera, and dispatches system
actions based on classified hand gestures. Press Q to quit.
"""
from __future__ import annotations

import logging
import pickle
import time
from collections import deque

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")

from gesture_engine.feature_extractor import FeatureExtractor
from gesture_engine.hand_tracker import HandTracker
from system_controls.mac_controls import dispatch, _get_volume
import subprocess


# ── Config / Model loading ─────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(model_dir: str = "models"):
    def _load(name):
        with open(f"{model_dir}/{name}", "rb") as f:
            return pickle.load(f)
    return _load("gesture_model.pkl"), _load("scaler.pkl"), _load("label_encoder.pkl")


# ── HUD rendering ─────────────────────────────────────────────────────────────

def _confidence_color(conf: float, threshold: float) -> tuple:
    if conf >= threshold:
        return (50, 210, 100)      # green — above threshold
    elif conf >= threshold * 0.8:
        return (30, 160, 255)      # orange — close
    return (80, 80, 80)            # grey — too low


def _volume_color(vol: int) -> tuple:
    """Green at high volume, orange mid, red at low."""
    if vol >= 60:
        return (50, 210, 100)   # green
    elif vol >= 30:
        return (30, 160, 255)   # orange
    return (60, 60, 220)        # red


def draw_hud(
    frame: np.ndarray,
    gesture: str | None,
    confidence: float,
    stable_frames: int,
    required_frames: int,
    action_label: str | None,
    last_trigger_ms: float,
    debounce_ms: int,
    threshold: float,
    fps: float,
    hand_present: bool,
    volume: int,
    muted: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Volume widget (right side, full height) ───────────────────────────────
    vbar_w    = 48
    vbar_x    = w - vbar_w - 12
    vbar_top  = 50
    vbar_bot  = h - 120
    vbar_h    = vbar_bot - vbar_top

    # Background track
    cv2.rectangle(overlay, (vbar_x, vbar_top), (vbar_x + vbar_w, vbar_bot), (30, 30, 30), -1)
    cv2.rectangle(overlay, (vbar_x, vbar_top), (vbar_x + vbar_w, vbar_bot), (60, 60, 60), 1)

    if muted:
        # Hatched / red overlay when muted
        cv2.rectangle(overlay, (vbar_x, vbar_top), (vbar_x + vbar_w, vbar_bot), (40, 40, 160), -1)
        cv2.putText(overlay, "MUTE", (vbar_x + 2, vbar_top + vbar_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1)
    else:
        fill_h   = int(vbar_h * volume / 100)
        fill_top = vbar_bot - fill_h
        vol_clr  = _volume_color(volume)
        cv2.rectangle(overlay, (vbar_x, fill_top), (vbar_x + vbar_w, vbar_bot), vol_clr, -1)

    # Volume % label
    vol_text = "MUTE" if muted else f"{volume}%"
    cv2.putText(overlay, vol_text,
                (vbar_x + 4, vbar_bot + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # 🔊 icon-ish label at top
    cv2.putText(overlay, "VOL",
                (vbar_x + 6, vbar_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Tick marks every 25%
    for pct in (25, 50, 75):
        ty = vbar_bot - int(vbar_h * pct / 100)
        cv2.line(overlay, (vbar_x, ty), (vbar_x + vbar_w, ty), (70, 70, 70), 1)
        cv2.putText(overlay, str(pct), (vbar_x - 22, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

    # ── Bottom panel ──
    panel_h = 110
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (12, 12, 12), -1)

    if hand_present and gesture:
        clr = _confidence_color(confidence, threshold)

        # Confidence bar
        bar_max  = int((vbar_x - 32) * 0.9)
        bar_fill = int(bar_max * confidence)
        cv2.rectangle(overlay, (16, h - panel_h + 12), (16 + bar_max, h - panel_h + 36), (50, 50, 50), -1)
        cv2.rectangle(overlay, (16, h - panel_h + 12), (16 + bar_fill, h - panel_h + 36), clr, -1)

        # Threshold marker
        thresh_x = 16 + int(bar_max * threshold)
        cv2.line(overlay, (thresh_x, h - panel_h + 8), (thresh_x, h - panel_h + 40), (255, 200, 0), 2)

        # Gesture name + confidence %
        label_str = f"{gesture}   {confidence * 100:.0f}%"
        cv2.putText(overlay, label_str, (16, h - panel_h + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, clr, 2)

        # ── Stability dots (show confirmation progress) ──
        for i in range(required_frames):
            filled  = i < stable_frames
            dot_clr = clr if filled else (50, 50, 50)
            cx = 16 + i * 14
            cv2.circle(overlay, (cx, h - panel_h + 78), 5, dot_clr, -1)

    else:
        status = "Hand out of frame — idle" if not hand_present else "No hand detected"
        cv2.putText(overlay, status, (16, h - panel_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1)

    # ── Action + cooldown indicator ──
    if action_label:
        since  = time.time() * 1000 - last_trigger_ms
        ratio  = min(1.0, since / debounce_ms)
        cd_clr = (50, 200, 100) if ratio >= 1.0 else (80, 80, 200)
        cd_text = f"→ {action_label}   {'ready' if ratio >= 1.0 else f'{ratio * 100:.0f}% cooldown'}"
        cv2.putText(overlay, cd_text, (16, h - panel_h + 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cd_clr, 1)

    # ── FPS (top-right) ──
    cv2.putText(overlay, f"{fps:.0f} fps", (w - vbar_w - 80, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)



# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    config    = load_config()
    cam_cfg   = config.get("camera",   {})
    model_cfg = config.get("model",    {})
    gestures  = config.get("gestures", {})

    confidence_threshold = model_cfg.get("confidence_threshold", 0.85)
    debounce_ms          = model_cfg.get("debounce_ms",          600)
    # How many consecutive frames the same gesture must appear before firing.
    # Higher = more stable, slightly more latency. 8 frames ≈ 0.25s at 30fps.
    required_stable      = model_cfg.get("required_stable_frames", 8)

    # "idle" is handled automatically by "no hand in frame" — skip it as a
    # trigger even if it still exists in the model classes.
    SKIP_GESTURES = {"idle"}

    try:
        clf, scaler, le = load_model()
    except FileNotFoundError:
        print("❌  Model not found. Run train_model.py first.")
        return

    print(f"✅  Model loaded. Classes: {list(le.classes_)}")
    print(f"   Threshold : {confidence_threshold}")
    print(f"   Debounce  : {debounce_ms}ms")
    print(f"   Stability : {required_stable} frames")

    tracker   = HandTracker(hand=model_cfg.get("hand", "left"))
    extractor = FeatureExtractor(config)

    cap = cv2.VideoCapture(cam_cfg.get("device_id", 1))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height",  720))

    last_trigger_ms = 0.0
    last_action     = None
    prev_t          = time.time()

    # Stability buffer
    gesture_buffer: deque[str] = deque(maxlen=required_stable)

    # Volume state — polled once at start, then updated after each action
    current_volume = _get_volume()
    is_muted       = False

    def _refresh_volume():
        """Re-read volume from macOS (called after any volume/mute action)."""
        nonlocal current_volume, is_muted
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 "set s to get volume settings\n"
                 "return (output volume of s) & \",\" & (output muted of s)"],
                capture_output=True, text=True, timeout=1,
            )
            parts = r.stdout.strip().split(",")
            current_volume = int(parts[0])
            is_muted       = parts[1].strip().lower() == "true"
        except Exception:
            pass

    print("\n🤙  Gesture Controller running — press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cam_cfg.get("flip", True):
            frame = cv2.flip(frame, 1)

        curr_t = time.time()
        fps    = 1.0 / max(curr_t - prev_t, 1e-6)
        prev_t = curr_t

        hand_state = tracker.process(frame)
        hand_present = hand_state is not None

        if hand_present:
            tracker.draw(frame)
        else:
            # Hand left frame → clear buffer so stale gestures don't linger
            gesture_buffer.clear()
            extractor.reset()

        gesture_name  = None
        confidence    = 0.0
        stable_frames = 0

        if hand_present:
            features = extractor.extract(hand_state)
            scaled   = scaler.transform(features.reshape(1, -1))
            proba    = clf.predict_proba(scaled)[0]
            idx      = int(np.argmax(proba))
            confidence   = float(proba[idx])
            gesture_name = le.classes_[idx]

            # ── Stability check ──────────────────────────────────────────────
            gesture_buffer.append(gesture_name)
            stable_frames = sum(1 for g in gesture_buffer if g == gesture_name)
            buffer_full   = len(gesture_buffer) == required_stable
            confirmed     = buffer_full and stable_frames == required_stable

            now_ms = curr_t * 1000
            cooldown_ok = (now_ms - last_trigger_ms) >= debounce_ms

            # Debug line — printed every frame so you can see what's happening
            print(
                f"  gesture={gesture_name:<12s}  conf={confidence:.2f}  "
                f"stable={stable_frames}/{required_stable}  "
                f"confirmed={confirmed}  cooldown={'ok' if cooldown_ok else f'{debounce_ms-(now_ms-last_trigger_ms):.0f}ms'}",
                end="\r",
            )

            # ── Trigger action ───────────────────────────────────────────────
            if (
                confirmed
                and gesture_name not in SKIP_GESTURES
                and confidence >= confidence_threshold
                and cooldown_ok
                and gesture_name in gestures
            ):
                gcfg   = gestures[gesture_name]
                action = gcfg.get("action", "none")
                delta  = gcfg.get("delta", 5)
                if action != "none":
                    dispatch(action, delta=delta)
                    print(f"\n  🔥 {gesture_name:<15s} → {action}  ({confidence * 100:.0f}%)")
                    _refresh_volume()   # update volume display immediately
                last_trigger_ms = now_ms
                last_action     = action

        frame = draw_hud(
            frame,
            gesture_name, confidence,
            stable_frames, required_stable,
            last_action, last_trigger_ms, debounce_ms,
            confidence_threshold, fps,
            hand_present,
            current_volume, is_muted,
        )
        cv2.imshow("Gesture Controller  [Q = quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("\n👋  Stopped.")


if __name__ == "__main__":
    main()
