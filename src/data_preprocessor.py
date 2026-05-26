import pandas as pd
import numpy as np
import os


def _compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-grounded features to the dataframe."""
    df = df.copy()
    dt = df["time"].diff().fillna(1.0).clip(lower=1e-6)

    # Smooth temperature before derivative computation
    T_smooth = df["temperature"].rolling(window=5, min_periods=1).mean()

    df["dT_dt"] = T_smooth.diff().fillna(0.0) / dt
    df["dT_dt"] = df["dT_dt"].rolling(window=5, min_periods=1).mean()

    df["d2T_dt2"] = df["dT_dt"].diff().fillna(0.0) / dt
    df["d2T_dt2"] = df["d2T_dt2"].rolling(window=5, min_periods=1).mean()

    df["dV_dt"] = df["voltage"].diff().fillna(0.0) / dt

    # T excess above ambient (first 20 readings used as ambient proxy)
    T_amb = df["temperature"].iloc[:20].min()
    df["T_excess"] = df["temperature"] - T_amb

    # Normalised heating rate: captures Newton ODE residual precursor
    df["norm_rate"] = df["dT_dt"] / (df["T_excess"].clip(lower=0.1))

    # Approximate internal resistance: |ΔV / ΔI|  (proxy — needs current column)
    if "current" in df.columns:
        dI_dt = df["current"].diff().fillna(0.0) / dt
        df["R_internal"] = (df["dV_dt"].abs() / (dI_dt.abs().clip(lower=1e-3)))
        df["R_internal"] = df["R_internal"].clip(upper=df["R_internal"].quantile(0.99))
    else:
        df["R_internal"] = 0.0

    # Cumulative thermal excess energy (integrates early drift)
    df["cum_excess"] = df["T_excess"].clip(lower=0.0).cumsum()

    return df


def preprocess_data(
    input_path: str = "data/thermal_runaway_data.csv",
    window_size: int = 10,
    lookahead: int = 30,
    runaway_temp: float = 80.0,
    rate_threshold: float = 1.0,
) -> tuple:
    """
    Load, engineer, and label battery data.

    Labels use a *lookahead* window so the model must learn precursor signals
    rather than the simultaneous threshold event.  A sequence at index i is
    labelled positive when ANY reading in the next `lookahead` steps breaches
    the thermal-runaway criteria — forcing the model to fire 0–30 steps early.

    Returns (df, X_sequences, y_labels).
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return None, None, None

    df = pd.read_csv(input_path)
    df = df[pd.to_numeric(df["time"], errors="coerce").notnull()].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    df = _compute_physics_features(df)

    # ── Lookahead label ───────────────────────────────────────────────────
    # Positive iff max temperature or max rate in the next `lookahead` steps
    # exceeds threshold.  shift(-lookahead) ensures the label reflects a
    # *future* event, not the current timestep.
    is_runaway = (df["temperature"] > runaway_temp) | (df["dT_dt"] > rate_threshold)
    df["label"] = (
        is_runaway.rolling(lookahead, min_periods=1).max().shift(-lookahead)
        .fillna(0.0).astype(int)
    )

    # ── Sequences ────────────────────────────────────────────────────────
    feature_cols = [
        "voltage", "current", "temperature",
        "dT_dt", "dV_dt", "d2T_dt2",
        "T_excess", "norm_rate", "R_internal", "cum_excess",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    data_arr = df[feature_cols].values
    labels_arr = df["label"].values

    X_seq, y_seq = [], []
    for i in range(len(data_arr) - window_size):
        X_seq.append(data_arr[i : i + window_size])
        y_seq.append(labels_arr[i + window_size])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print(
        f"Preprocessed {len(df)} rows → {len(X_seq)} sequences "
        f"(lookahead={lookahead}s, {y_seq.sum()} positive)"
    )
    return df, X_seq, y_seq


if __name__ == "__main__":
    df, X, y = preprocess_data()
    if df is not None:
        print(df[["time", "temperature", "dT_dt", "d2T_dt2", "T_excess", "label"]].tail(15))
