"""
CNN-LSTM trend analyser — predicts runaway *before* it occurs via lookahead labels.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .data_preprocessor import preprocess_data


def build_cnn_lstm(input_shape: tuple):
    """CNN-LSTM classifier for early thermal runaway prediction."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_trend_analyzer(input_path: str = "data/thermal_runaway_data.csv", lookahead: int = 30):
    window_size = 10
    df, X, y = preprocess_data(input_path, window_size=window_size, lookahead=lookahead)
    if X is None:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm(input_shape)
    model.summary()

    print(f"Training CNN-LSTM (lookahead={lookahead}s) …")
    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1,
        class_weight={0: 1.0, 1: max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))},
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("\nHeld-out performance:")
    print(classification_report(y_test, y_pred, digits=3))

    # Lead-time analysis
    _plot_lead_time(df, model, window_size, lookahead)

    return model


def _plot_lead_time(df, model, window_size, lookahead):
    """Plot detection lead-time vs simple 80°C threshold."""
    import pandas as pd
    features = [c for c in df.columns if c not in ["time", "label"]]
    data_arr = df[features].values
    X_all = np.array([data_arr[i : i + window_size] for i in range(len(data_arr) - window_size)])

    proba = model.predict(X_all, verbose=0).flatten()
    times = df["time"].values[window_size:]

    trigger_threshold = 80.0
    model_warning = times[proba > 0.5][0] if (proba > 0.5).any() else None
    simple_alarm = df.loc[df["temperature"] > trigger_threshold, "time"].iloc[0] if (df["temperature"] > trigger_threshold).any() else None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["time"], df["temperature"], color="steelblue", label="Temperature (°C)", alpha=0.7)
    ax.axhline(trigger_threshold, color="red", linestyle="--", alpha=0.4, label=f"Simple alarm ({trigger_threshold}°C)")
    if model_warning:
        ax.axvline(model_warning, color="orange", linestyle="--", label=f"CNN-LSTM warning (t={model_warning:.1f}s)")
    if simple_alarm and model_warning:
        lead = simple_alarm - model_warning
        ax.set_title(f"CNN-LSTM lead-time advantage: {lead:.1f}s over threshold alarm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("lead_time_comparison.png")
    plt.close()
    print("Lead-time plot saved to lead_time_comparison.png")


if __name__ == "__main__":
    train_trend_analyzer()
