"""
Multi-model early warning system — wires RF spike detector and CNN-LSTM
trend analyser into a single comparative diagnostic report.

Imports are relative (within the src package) to avoid path issues.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_preprocessor import preprocess_data
from .spike_detector_rf import train_spike_detector, FEATURE_COLS
from .trend_analyzer_lstm import build_cnn_lstm


def run_early_warning_system(input_path: str = "data/thermal_runaway_data.csv"):
    print("Initialising Multi-Model Early Warning System …")

    # ── RF detector ───────────────────────────────────────────────────────
    rf, rf_features = train_spike_detector(input_path)
    if rf is None:
        print("RF training failed — aborting.")
        return

    # ── Sequence data for CNN-LSTM ────────────────────────────────────────
    window_size = 10
    df, X, y = preprocess_data(input_path, window_size=window_size)
    if X is None:
        print("Data load failed — aborting.")
        return

    # ── CNN-LSTM ─────────────────────────────────────────────────────────
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    input_shape = (X.shape[1], X.shape[2])
    lstm = build_cnn_lstm(input_shape)
    lstm.fit(X, y, epochs=20, batch_size=8, verbose=0)

    # ── Inference ────────────────────────────────────────────────────────
    rf_features_present = [c for c in FEATURE_COLS if c in df.columns]
    rf_preds = rf.predict(df[rf_features_present].fillna(0.0))

    lstm_proba = lstm.predict(X, verbose=0).flatten()
    lstm_preds = (lstm_proba > 0.5).astype(int)

    # ── Visualise ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    ax1.plot(df["time"], df["temperature"], color="steelblue", label="Temperature (°C)", alpha=0.6)
    rf_alert = df["time"][rf_preds == 1]
    rf_temp = df["temperature"][rf_preds == 1]
    ax1.scatter(rf_alert, rf_temp, color="red", marker="x", s=60, label="RF alert")
    ax1.axhline(80, color="red", linestyle="--", alpha=0.3, label="80°C threshold")
    ax1.set_title("RF Spike Detector")
    ax1.legend()

    lstm_times = df["time"].values[window_size:]
    ax2.plot(lstm_times, lstm_proba, color="orange", label="CNN-LSTM P(runaway)")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Decision boundary")
    ax2.set_title("CNN-LSTM Trend Analyser")
    ax2.set_xlabel("Time (s)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("early_warning_detection.png")
    print("Detection plot saved to early_warning_detection.png")

    # ── Diagnostic report ────────────────────────────────────────────────
    runaway_time = df.loc[df["temperature"] >= 80, "time"]
    onset = runaway_time.iloc[0] if not runaway_time.empty else None

    rf_first = df["time"][rf_preds == 1].iloc[0] if (rf_preds == 1).any() else None
    lstm_first_idx = np.where(lstm_preds == 1)[0]
    lstm_first = lstm_times[lstm_first_idx[0]] if len(lstm_first_idx) > 0 else None

    print("\n" + "=" * 50)
    print("  THERMAL RUNAWAY DIAGNOSTIC REPORT")
    print("=" * 50)
    if onset:
        print(f"  Runaway onset (T≥80°C):       {onset:.1f}s")
    if rf_first and onset:
        print(f"  RF detector first alert:      {rf_first:.1f}s  ({onset - rf_first:.1f}s lead)")
    if lstm_first and onset:
        print(f"  CNN-LSTM first alert:         {lstm_first:.1f}s  ({onset - lstm_first:.1f}s lead)")
    print("=" * 50)


if __name__ == "__main__":
    run_early_warning_system()
