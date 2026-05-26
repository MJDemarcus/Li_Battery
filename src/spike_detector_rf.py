"""
Random-Forest spike detector — trained on physics-enriched features.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_preprocessor import preprocess_data


FEATURE_COLS = [
    "voltage", "current", "temperature",
    "dT_dt", "dV_dt", "d2T_dt2",
    "T_excess", "norm_rate", "R_internal", "cum_excess",
]


def train_spike_detector(input_path: str = "data/thermal_runaway_data.csv"):
    df, _, _ = preprocess_data(input_path)
    if df is None:
        return None, None

    features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[features].fillna(0.0)
    y = df["label"]

    # Hold-out split for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print("Training Random Forest Spike Detector …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\nHeld-out performance:")
    print(classification_report(y_test, y_pred, digits=3))

    # Precision-recall curve (critical for safety-critical apps)
    proba = rf.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, proba)

    plt.figure(figsize=(8, 5))
    plt.plot(rec, prec, color="#f85149")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RF Spike Detector — Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig("rf_pr_curve.png")
    plt.close()

    return rf, features


if __name__ == "__main__":
    train_spike_detector()
