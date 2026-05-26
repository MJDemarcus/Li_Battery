"""
LSTM anomaly detection with stable, training-computed thresholds.

The threshold is derived once from labelled normal-condition data and stored
alongside the model, preventing the per-batch drift that Ken flagged.
"""
import numpy as np
import pandas as pd


def compute_normal_threshold(model, X_normal, threshold_percentile: float = 95) -> float:
    """
    Compute MAE threshold from *normal* (label=0) sequences only.

    Call this once after training and persist the returned scalar with the
    model weights — do NOT recompute on inference batches.
    """
    y_pred = model.predict(X_normal, verbose=0)
    # For regression models predicting next-step value, compare to last step
    y_true = X_normal[:, -1, 0]
    mae = np.mean(np.abs(y_pred.flatten() - y_true), axis=0)
    threshold = float(np.percentile(mae if np.ndim(mae) > 0 else [mae], threshold_percentile))
    return threshold


def detect_anomalies_lstm(model, X_data, threshold: float, y_pred=None):
    """
    Detect anomalies using a pre-computed stable threshold.

    Parameters
    ----------
    model  : trained Keras/mock model
    X_data : (batch, window, features) array
    threshold : scalar from compute_normal_threshold — do NOT derive inline
    y_pred : optional cached predictions

    Returns
    -------
    anomaly_indices, mae_loss_array
    """
    if y_pred is None:
        y_pred = model.predict(X_data, verbose=0)

    y_true = X_data[:, -1, 0]
    mae_loss = np.abs(y_pred.flatten() - y_true)

    anomalies_idx = np.where(mae_loss > threshold)[0]
    return anomalies_idx, mae_loss


def detect_anomalies_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.DataFrame:
    """Fallback IQR anomaly detector — use when model is unavailable."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    return df[(df[column] < lower) | (df[column] > upper)]
