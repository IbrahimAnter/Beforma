"""
Optional offline trainer.

Use this after you collect labeled sequences from your own gym videos.
Expected dataset format:
    dataset/
      squat/*.json
      push_up/*.json
      bicep_curl/*.json
      shoulder_press/*.json
      jumping_jack/*.json
      lateral_raise/*.json

Each JSON file should be a list of frames, where each frame contains numeric features
such as elbow_angle, knee_angle, wrist_gap, ankle_gap, spine_line_error, etc.

This trainer converts each sequence to statistical features and trains a RandomForest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


NUMERIC_KEYS = [
    "elbow_angle",
    "knee_angle",
    "hip_angle",
    "torso_lean_deg",
    "shoulder_to_wrist_y",
    "wrist_to_shoulder_dist",
    "wrist_to_hip_dist",
    "ankle_gap",
    "wrist_gap",
    "spine_line_error",
]


def sequence_to_vector(sequence: List[Dict]) -> np.ndarray:
    vectors = []
    for key in NUMERIC_KEYS:
        values = [float(frame.get(key, 0.0)) for frame in sequence]
        vectors.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            values[-1] - values[0] if len(values) > 1 else 0.0,
        ])
    return np.array(vectors, dtype=np.float32)


def load_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    root = Path(dataset_dir)
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for file in class_dir.glob("*.json"):
            sequence = json.loads(file.read_text(encoding="utf-8"))
            X.append(sequence_to_vector(sequence))
            y.append(label)
    return np.array(X), np.array(y)


def main(dataset_dir: str = "dataset", out_path: str = "exercise_classifier.joblib"):
    X, y = load_dataset(dataset_dir)
    if len(X) < 20:
        raise RuntimeError("Dataset is too small. Collect more labeled sequences first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
