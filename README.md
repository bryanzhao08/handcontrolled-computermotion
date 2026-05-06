# Hand-Controlled Computer Motion 🤙

Vision Pro-inspired hand gesture control for macOS — volume, brightness, and more.
Built on MediaPipe hand landmarks + a fully tunable sklearn MLP classifier.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Record training data (≥ 50 samples per gesture)
python collect_data.py

# 3. Train your model
python train_model.py

# 4. Run live gesture control
python main.py
```

---

## Workflow

```
collect_data.py  →  data/gestures.csv
train_model.py   →  models/gesture_model.pkl
main.py          →  live camera → system actions
```

**To add a new gesture or retrain:** edit `config.yaml`, run `collect_data.py` again (new data appends), then re-run `train_model.py`.

---

## collect_data.py — Recording Controls

| Key | Action |
|-----|--------|
| letter key | Select gesture label (configured in `config.yaml`) |
| `SPACE` | Record a batch of frames with the current label |
| `Q` | Quit and save all recorded data |

Aim for **60+ samples per gesture** across varied lighting and hand positions.

---

## config.yaml — Tunable Parameters

```yaml
gestures:
  swipe_up:
    key: "u"           # hotkey in collect_data.py
    action: volume_up  # which system action to trigger
    delta: 5           # argument passed to the action

model:
  confidence_threshold: 0.85   # raise to reduce false triggers
  debounce_ms: 600             # min ms between consecutive triggers
  history_frames: 5            # frames of velocity history
  use_velocity: true           # include motion features
  hand: right                  # left | right | both
```

### Available actions

| action | effect |
|--------|--------|
| `volume_up` / `volume_down` | adjust system volume by `delta` units |
| `mute_toggle` | toggle mute |
| `brightness_up` / `brightness_down` | adjust display brightness by ~`delta`% |
| `mission_control` | open Mission Control |
| `screenshot` | capture screen to `/tmp/gesture_screenshot.png` |
| `none` | no-op (useful for an "idle" gesture class) |

---

## Project Structure

```
handcontrolled-computermotion/
├── config.yaml              ← all tunable params + gesture mappings
├── collect_data.py          ← interactive data recorder
├── train_model.py           ← model training + evaluation
├── main.py                  ← live gesture control loop
├── requirements.txt
├── gesture_engine/
│   ├── hand_tracker.py      ← MediaPipe wrapper (wrist-relative normalization)
│   └── feature_extractor.py ← landmarks + angles + velocity → feature vector
├── system_controls/
│   └── mac_controls.py      ← volume / brightness / misc macOS actions
├── data/
│   └── gestures.csv         ← your recorded training data
└── models/
    ├── gesture_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

---

## Tips for Good Accuracy

- Record samples in the **same lighting** you'll use it in
- Include an **`idle`** class (open/neutral hand) to prevent false triggers
- Check the confusion matrix from `train_model.py` — if two gestures confuse each other, make them more visually distinct or add more samples
- Lower `confidence_threshold` (e.g. `0.75`) if gestures aren't triggering; raise it if there are false positives
