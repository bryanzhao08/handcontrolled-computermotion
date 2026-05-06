"""
Train a gesture classifier from collected landmark data.

Usage:
    python train_model.py

Outputs:
    models/gesture_model.pkl    — trained MLPClassifier
    models/scaler.pkl           — fitted StandardScaler
    models/label_encoder.pkl    — LabelEncoder (index → gesture name)
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    config    = load_config()
    train_cfg = config.get("training", {})
    data_file = config.get("data", {}).get("output_file", "data/gestures.csv")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    print("🏋️  Gesture Model Trainer")
    print("=" * 45)

    # ── Load data ──
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("   Run collect_data.py first.")
        return

    df = pd.read_csv(data_file)
    if df.empty or "label" not in df.columns:
        print("❌ Dataset is empty or malformed.")
        return

    print(f"   Samples : {len(df)}")
    print(f"   Classes : {df['label'].nunique()}")
    print(f"\n   Distribution:\n{df['label'].value_counts().to_string()}\n")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    # ── Encode labels ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── Train/test split ──
    test_size    = train_cfg.get("test_size", 0.2)
    random_state = train_cfg.get("random_state", 42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc,
    )

    # ── Scale ──
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # ── Train ──
    hidden = tuple(train_cfg.get("hidden_layer_sizes", [128, 64]))
    max_iter = train_cfg.get("max_iter", 500)
    print(f"   Architecture : MLP {hidden}  max_iter={max_iter}")
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )
    clf.fit(X_tr, y_tr)

    # ── Evaluate ──
    y_pred = clf.predict(X_te)
    accuracy = (y_pred == y_te).mean()

    print(f"\n📊 Test accuracy: {accuracy * 100:.1f}%\n")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_te, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_df.to_string())

    # ── Save ──
    paths = {
        "gesture_model.pkl":  clf,
        "scaler.pkl":         scaler,
        "label_encoder.pkl":  le,
    }
    for fname, obj in paths.items():
        fpath = os.path.join(model_dir, fname)
        with open(fpath, "wb") as f:
            pickle.dump(obj, f)

    print(f"\n✅ Model saved to '{model_dir}/'")
    print(f"   Classes: {list(le.classes_)}")
    print(f"\n   To retrain with more data: run collect_data.py then this script again.")


if __name__ == "__main__":
    main()
