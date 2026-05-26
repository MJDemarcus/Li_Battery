"""
Thermal runaway prediction pipeline — end-to-end working inference.

Uses the four-trigger / five-state architecture from hybrid_evaluator.py,
not hardcoded stubs.  Call predict_status() with a DataFrame window to get
a real state assessment.
"""
import pandas as pd
import numpy as np
from .hybrid_evaluator import run_hybrid_heuristic, get_warning_metrics


# Five-state labels (0=Stable … 4=Critical)
STATE_LABELS = {
    0: "Stable",
    1: "Explanation Needed",
    2: "Watching Brief",
    3: "High Risk",
    4: "Critical Warning",
}


def predict_status(df: pd.DataFrame) -> dict:
    """
    Run the four-trigger engine on a window of battery readings.

    Parameters
    ----------
    df : DataFrame with columns: time, temperature, (voltage, current optional)

    Returns
    -------
    dict with keys:
        status      : state label string
        state_code  : int 0-4
        probability : int 0-100 (trigger count * 25)
        triggers    : dict of A/B/C/D booleans
        lead_time   : seconds of warning lead (None if not yet triggered)
        health_score: float 0-1 (1 = fully healthy)
    """
    if df is None or len(df) < 20:
        return {
            "status": STATE_LABELS[0],
            "state_code": 0,
            "probability": 0,
            "triggers": {"A": False, "B": False, "C": False, "D": False},
            "lead_time": None,
            "health_score": 1.0,
        }

    result = run_hybrid_heuristic(df.copy())

    last = result.iloc[-1]
    trigger_sum = int(last.get("trigger_sum", 0))
    state_code = min(trigger_sum, 4)
    probability = trigger_sum * 25

    triggers = {
        "A": bool(last.get("trigger_A", 0)),
        "B": bool(last.get("trigger_B", 0)),
        "C": bool(last.get("trigger_C", 0)),
        "D": bool(last.get("trigger_D", 0)),
    }

    warning_time, runaway_time, lead_time = get_warning_metrics(result)

    health_score = max(0.0, 1.0 - probability / 100.0)

    return {
        "status": STATE_LABELS[state_code],
        "state_code": state_code,
        "probability": probability,
        "triggers": triggers,
        "lead_time": lead_time,
        "health_score": round(health_score, 3),
    }


def evaluate_on_dataset(
    df: pd.DataFrame,
    window_seconds: int = 120,
    step_seconds: int = 10,
    runaway_temp: float = 80.0,
) -> pd.DataFrame:
    """
    Slide a window over the dataset and compute predictions + lead time.

    Useful for generating the precision-recall and lead-time distribution
    metrics Ken recommended for the submission.
    """
    results = []
    times = df["time"].values
    t_start = times[0]
    t_end = times[-1]

    runaway_times = df.loc[df["temperature"] >= runaway_temp, "time"]
    actual_onset = float(runaway_times.iloc[0]) if not runaway_times.empty else None

    t = t_start + window_seconds
    while t <= t_end:
        window = df[df["time"].between(t - window_seconds, t)]
        if len(window) < 20:
            t += step_seconds
            continue

        pred = predict_status(window)
        lead = (actual_onset - t) if (actual_onset and t < actual_onset) else None

        results.append({
            "time": t,
            "state_code": pred["state_code"],
            "status": pred["status"],
            "probability": pred["probability"],
            "trigger_A": pred["triggers"]["A"],
            "trigger_B": pred["triggers"]["B"],
            "trigger_C": pred["triggers"]["C"],
            "trigger_D": pred["triggers"]["D"],
            "health_score": pred["health_score"],
            "seconds_to_runaway": lead,
        })
        t += step_seconds

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/thermal_runaway_data.csv"
    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    result = predict_status(df)
    print("\nPrediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
