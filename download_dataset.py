"""
Download and process the HaGRID hand gesture dataset subset into landmark CSV.

HaGRID (Hand Gesture Recognition Image Dataset) contains 552k images across
18 gesture classes. This script downloads a small sample (~200 images per
matching class) directly from the public GitHub mirror, runs MediaPipe
HandLandmarker on each image, and appends extracted landmarks to your
data/gestures.csv — giving you thousands of pre-labeled samples without
manual collection.

Usage:
    python download_dataset.py [--samples N]

Options:
    --samples N   Images to download per gesture class (default: 200)

Gesture mapping (HaGRID → your labels):
    fist          → fist
    stop          → spread
    palm          → idle
    ok            → pinch
    like          → swipe_up
    dislike       → swipe_down
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import urllib.request
import json

import cv2
import numpy as np

# ── Lazy imports (only needed at runtime) ────────────────────────────────────
def _require(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"❌  Missing package: {pkg}. Run: pip install {pkg}")
        sys.exit(1)

# ── Gesture mapping ───────────────────────────────────────────────────────────

# Maps HaGRID class name → your gesture label
GESTURE_MAP = {
    "fist":    "fist",
    "stop":    "spread",
    "palm":    "idle",
    "ok":      "pinch",
    "like":    "swipe_up",
    "dislike": "swipe_down",
}

# HaGRID sample images are hosted on GitHub (public, no auth)
HAGRID_BASE = (
    "https://raw.githubusercontent.com/hukenovs/hagrid/master/hagrid/images"
)

# Fallback: use the annotation JSON from the official repo to find image URLs
HAGRID_ANN_URL = (
    "https://raw.githubusercontent.com/hukenovs/hagrid/master/"
    "hagrid/ann_train_val/{gesture}_train.json"
)

# Direct CDN sample images (from the HuggingFace mirror - public access)
HF_DATASET_URL = (
    "https://huggingface.co/datasets/hagrid/hagrid/resolve/main"
    "/data/{gesture}/{gesture}_{idx:06d}.jpg"
)

# We fall back to scraping the GitHub API for image file paths
GITHUB_API_URL = (
    "https://api.github.com/repos/hukenovs/hagrid/contents"
    "/hagrid/images/{gesture}?per_page=300"
)


def _download_bytes(url: str, timeout: int = 10) -> bytes | None:
    """Download URL → bytes, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gesture-dl/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _fetch_image_urls_github(gesture: str, n: int) -> list[str]:
    """Try to get image URLs via GitHub Contents API."""
    api_url = GITHUB_API_URL.format(gesture=gesture)
    data = _download_bytes(api_url, timeout=15)
    if data is None:
        return []
    try:
        entries = json.loads(data)
        urls = [e["download_url"] for e in entries if e["name"].endswith(".jpg")]
        return urls[:n]
    except Exception:
        return []


def _fetch_image_urls_hf(gesture: str, n: int) -> list[str]:
    """Generate HuggingFace CDN URLs (may require auth, but worth trying)."""
    return [
        HF_DATASET_URL.format(gesture=gesture, idx=i)
        for i in range(n)
    ]


def _decode_image(raw: bytes) -> np.ndarray | None:
    """Decode raw bytes → BGR numpy array."""
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Download HaGRID → gesture landmarks")
    parser.add_argument("--samples", type=int, default=200,
                        help="Images per gesture class (default: 200)")
    parser.add_argument("--output", default="data/gestures.csv",
                        help="Output CSV path (default: data/gestures.csv)")
    parser.add_argument("--config", default="config.yaml",
                        help="config.yaml path (default: config.yaml)")
    args = parser.parse_args()

    # ── Imports ──────────────────────────────────────────────────────────────
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )
    import yaml

    MODEL_PATH = os.path.abspath("models/hand_landmarker.task")
    if not os.path.exists(MODEL_PATH):
        print("❌  models/hand_landmarker.task not found.")
        print("   Run this first:")
        print("   curl -L https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
              "-o models/hand_landmarker.task")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Set up MediaPipe ──────────────────────────────────────────────────────
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # ── Feature extractor (same as collect_data / main) ───────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    from gesture_engine.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(config)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0

    total_saved = 0

    print(f"\n🌐  HaGRID Dataset Downloader")
    print(f"   Gestures : {list(GESTURE_MAP.keys())}")
    print(f"   Samples  : up to {args.samples} per class")
    print(f"   Output   : {args.output}\n")

    feature_cols_written = file_exists  # track if we've written the header yet

    with open(args.output, "a", newline="") as csvfile:
        writer = None  # lazy-init after we know feature size

        for hagrid_cls, your_label in GESTURE_MAP.items():
            print(f"  [{hagrid_cls} → {your_label}]  fetching URLs…", end="", flush=True)

            # Try GitHub API first, then HuggingFace
            urls = _fetch_image_urls_github(hagrid_cls, args.samples)
            if not urls:
                print(" (GitHub API failed, trying HuggingFace)…", end="", flush=True)
                urls = _fetch_image_urls_hf(hagrid_cls, args.samples)

            if not urls:
                print(f"\n  ⚠  Could not fetch URLs for '{hagrid_cls}' — skipping.")
                continue

            print(f" got {len(urls)} URLs", flush=True)

            saved = 0
            failed = 0
            for url in urls:
                raw = _download_bytes(url)
                if raw is None:
                    failed += 1
                    continue

                bgr = _decode_image(raw)
                if bgr is None:
                    failed += 1
                    continue

                rgb = np.ascontiguousarray(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
                )
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = landmarker.detect(mp_img)
                except Exception:
                    failed += 1
                    continue

                if not result.hand_landmarks:
                    failed += 1
                    continue

                # Use first detected hand
                hand_lm = result.hand_landmarks[0]
                raw_lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)

                # Wrist-relative normalization (same as HandTracker._normalize)
                shifted = raw_lm - raw_lm[0]
                scale = np.linalg.norm(shifted[9]) + 1e-6
                normalized = shifted / scale

                from gesture_engine.hand_tracker import HandState
                extractor.reset()
                state = HandState(
                    landmarks=normalized,
                    raw_landmarks=raw_lm,
                    handedness="unknown",
                    confidence=1.0,
                )
                features = extractor.extract(state)

                # Lazy-init CSV writer
                if writer is None:
                    feat_cols = [f"f{i}" for i in range(len(features))]
                    writer = csv.writer(csvfile)
                    if not feature_cols_written:
                        writer.writerow(["label"] + feat_cols)
                        feature_cols_written = True

                writer.writerow([your_label] + features.tolist())
                saved += 1
                total_saved += 1

            print(f"       ✓ saved={saved}  failed/no-hand={failed}")

    landmarker.close()

    print(f"\n✅  Done! {total_saved} landmark samples added to '{args.output}'")
    print("   Now run:  python3 train_model.py")


if __name__ == "__main__":
    main()
