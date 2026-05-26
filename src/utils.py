"""Shared utilities for the Li_Battery package."""
import numpy as np
import pandas as pd


def compute_lead_time_distribution(eval_df: pd.DataFrame, trigger_col: str = "trigger_D") -> dict:
    """
    From evaluate_on_dataset output, compute lead-time stats for a given trigger.
    Returns dict with mean, median, min, max (seconds before runaway).
    """
    fired = eval_df[eval_df[trigger_col] & eval_df["seconds_to_runaway"].notna()]
    if fired.empty:
        return {"mean": None, "median": None, "min": None, "max": None}
    leads = fired["seconds_to_runaway"].values
    return {
        "mean": round(float(np.mean(leads)), 1),
        "median": round(float(np.median(leads)), 1),
        "min": round(float(np.min(leads)), 1),
        "max": round(float(np.max(leads)), 1),
    }


def precision_recall_at_threshold(eval_df: pd.DataFrame, state_threshold: int = 2) -> dict:
    """
    Treat state_code >= state_threshold as positive prediction.
    Compare against whether seconds_to_runaway is finite (actual runaway series).
    """
    eval_df = eval_df.copy()
    eval_df["pred_pos"] = eval_df["state_code"] >= state_threshold
    eval_df["actual_pos"] = eval_df["seconds_to_runaway"].notna()

    tp = (eval_df["pred_pos"] & eval_df["actual_pos"]).sum()
    fp = (eval_df["pred_pos"] & ~eval_df["actual_pos"]).sum()
    fn = (~eval_df["pred_pos"] & eval_df["actual_pos"]).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }
