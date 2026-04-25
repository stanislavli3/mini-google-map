"""
calibrate_submission.py
=======================

Quantile-match the predicted durations in `submission_final.csv` to the
distribution of `kaggle_like_val.csv` durations. If the model ranks trips
correctly but compresses their magnitudes (predicts 8 min for a trip that's
really 25 min, 10 min for one that's really 35 min), quantile calibration
stretches the output to match the target distribution while preserving the
rank order.

Writes submission_calibrated.csv.
"""

from __future__ import annotations
# --- src/ path bootstrap (cross-subdir sibling imports) ---
import os as _os, sys as _sys
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _SRC)
import _path_bootstrap  # noqa: F401  (registers all src/<subdir>/ on sys.path)
PROJECT_ROOT = _path_bootstrap.PROJECT_ROOT
del _os, _sys, _SRC
# --- end bootstrap ---


import numpy as np
import pandas as pd


def quantile_calibrate(pred_values: np.ndarray, target_values: np.ndarray) -> np.ndarray:
    """Map each pred to the value at the same percentile in `target_values`."""
    sorted_target = np.sort(target_values)
    # Convert pred to ranks in [0, 1]
    ranks_pct = pd.Series(pred_values).rank(pct=True, method="average").to_numpy()
    # Pull the value at that percentile from the target distribution
    idx = np.clip(
        (ranks_pct * (len(sorted_target) - 1)).round().astype(int),
        0, len(sorted_target) - 1,
    )
    return sorted_target[idx]


def main():
    sub = pd.read_csv("submission_final.csv")
    val = pd.read_csv("kaggle_like_val.csv")

    print("=" * 60)
    print("Quantile calibration")
    print("=" * 60)
    print(f"\nsubmission_final.csv ({len(sub)} rows):")
    print(sub["duration_min"].describe().round(2).to_string())

    print(f"\nkaggle_like_val.csv ({len(val)} rows):")
    print(val["duration_min"].describe().round(2).to_string())

    calibrated = quantile_calibrate(sub["duration_min"].values,
                                     val["duration_min"].values)

    # Keep it inside a reasonable global band
    calibrated = np.clip(calibrated, 0.5, 120.0)

    out = sub.copy()
    out["duration_min"] = calibrated

    print(f"\ncalibrated predictions:")
    print(out["duration_min"].describe().round(2).to_string())

    # How much did individual predictions move?
    delta = out["duration_min"].values - sub["duration_min"].values
    print(f"\nPer-row delta (calibrated - original):")
    print(f"  mean:   {delta.mean():+.2f}")
    print(f"  median: {float(np.median(delta)):+.2f}")
    print(f"  p5:     {float(np.percentile(delta, 5)):+.2f}")
    print(f"  p95:    {float(np.percentile(delta, 95)):+.2f}")
    print(f"  max abs:{float(np.abs(delta).max()):.2f}")

    out_path = "submission_calibrated.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
