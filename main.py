"""
Live gesture recognition → macOS system control.

Usage:
    python main.py

Reads models/ and config.yaml, opens the camera, and dispatches system
actions based on classified hand gestures. Press Q to quit.
"""
from __future__ import annotations

import pickle
import time

import cv2
import numpy as np
import yaml

from gesture_engine.feature_extractor import FeatureExtractor
from gesture_engine.hand_tracker import HandTracker
from system_controls.mac_controls import dispatch


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


def draw_hud(
    frame: np.ndarray,
    gesture: str | None,
    confidence: float,
    action_label: str | None,
    last_trigger_ms: float,
    debounce_ms: int,
    threshold: float,
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Bottom panel ──
    panel_h = 100
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (12, 12, 12), -1)

    if gesture:
        clr = _confidence_color(confidence, threshold)

        # Confidence bar
        bar_max = int(w * 0.55)
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

    else:
        cv2.putText(overlay, "No hand detected", (16, h - panel_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 90, 90), 1)

    # ── Action + cooldown indicator ──
    if action_label:
        since = time.time() * 1000 - last_trigger_ms
        ratio  = min(1.0, since / debounce_ms)
        cd_clr = (50, 200, 100) if ratio >= 1.0 else (80, 80, 200)
        # Cooldown arc / text
        cd_text = f"→ {action_label}   {'ready' if ratio >= 1.0 else f'{ratio * 100:.0f}% cooldown'}"
        cv2.putText(overlay, cd_text, (16, h - panel_h + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cd_clr, 1)

    # ── FPS (top-right) ──
    cv2.putText(overlay, f"{fps:.0f} fps", (w - 85, 28),
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

    try:
        clf, scaler, le = load_model()
    except FileNotFoundError:
        print("❌  Model not found. Run train_model.py first.")
        return

    print(f"✅  Model loaded. Classes: {list(le.classes_)}")
    print(f"   Threshold: {confidence_threshold}   Debounce: {debounce_ms}ms")

    tracker   = HandTracker(hand=model_cfg.get("hand", "right"))
    extractor = FeatureExtractor(config)

    cap = cv2.VideoCapture(cam_cfg.get("device_id", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg.get("width",  1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height",  720))

    last_trigger_ms = 0.0
    last_action     = None
    prev_t          = time.time()

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

        state = tracker.process(frame)
        if state is not None:
            tracker.draw(frame)

        gesture_name = None
        confidence   = 0.0

        if state is not None:
            features = extractor.extract(state)
            scaled   = scaler.transform(features.reshape(1, -1))
            proba    = clf.predict_proba(scaled)[0]
            idx      = int(np.argmax(proba))
            confidence   = float(proba[idx])
            gesture_name = le.classes_[idx]

            # ── Trigger action ──
            now_ms = curr_t * 1000
            if (
                confidence >= confidence_threshold
                and (now_ms - last_trigger_ms) >= debounce_ms
                and gesture_name in gestures
            ):
                gcfg = gestures[gesture_name]
                action = gcfg.get("action", "none")
                delta  = gcfg.get("delta", 5)
                if action != "none":
                    dispatch(action, delta=delta)
                    print(f"  {gesture_name:15s} → {action}  ({confidence * 100:.0f}%)")
                last_trigger_ms = now_ms
                last_action     = action

        frame = draw_hud(
            frame,
            gesture_name, confidence,
            last_action, last_trigger_ms, debounce_ms,
            confidence_threshold, fps,
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
