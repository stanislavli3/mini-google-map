"""
build_kaggle_like_val.py
========================

Phase 2A of the Round-2 plan. Our current Stage 3 val set is the last 3 days
of `Trajectories/` data — a different distribution from the Kaggle test set,
which samples (source, dest, source_time) queries on the 50 Test Cases
vehicles. Local val RMSE therefore under-estimates test error badly.

This helper builds a *held-out* synthetic val set in the same shape as the
Kaggle test CSV (same columns, same vehicles, trips drawn from Test Cases/).
Each row has a known true `duration_min` so local iteration is honest.

USAGE
-----
    python build_kaggle_like_val.py --per-vehicle 8 --out kaggle_like_val.csv

Then pass `--kaggle-val kaggle_like_val.csv` to `stage3_eta_prediction.py` (or
set it via env-var / default in main()).
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from map_matching_solution import load_trajectory, great_circle_distance


def _select_interior_pair(
    traj,
    min_gap_s: int = 60,
    max_gap_s: int = 120 * 60,  # 120-min window. The 30-min cap was wrong:
                                 # LB feedback showed test trips have a heavy
                                 # tail (real taxi trips with traffic / waits
                                 # can run 60+ min on small haversine), and
                                 # the cap was making early-stopping pick a
                                 # model that ignored that tail.
    rng: random.Random = None,
) -> Tuple[int, int]:
    """
    Pick (src_idx, dst_idx) from a trajectory such that the elapsed time
    between them is in [min_gap_s, max_gap_s]. Returns (-1, -1) if the
    trajectory doesn't contain a valid pair.

    Strategy: pick a random source index in the first 90% of the trajectory,
    then find the earliest index whose timestamp is at least min_gap_s later
    and the latest whose timestamp is at most max_gap_s later, and sample
    uniformly inside that window.
    """
    rng = rng or random.Random()
    n = len(traj)
    if n < 10:
        return -1, -1

    for _ in range(30):
        s = rng.randint(0, max(1, int(n * 0.9)) - 1)
        src_ts = traj[s].timestamp

        # Find indices whose timestamps fall inside the gap window
        valid = [
            j for j in range(s + 1, n)
            if min_gap_s <= (traj[j].timestamp - src_ts) <= max_gap_s
        ]
        if not valid:
            continue
        d = rng.choice(valid)
        return s, d
    return -1, -1


def build_synthetic_val(
    test_cases_dir: str = "Test Cases",
    per_vehicle: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthesize queries from each Test Cases trajectory. Returns a DataFrame
    with the Kaggle test schema + a `duration_min` ground-truth column.

    Columns:
      id, vehicle_id, source_lat, source_lon, source_time, dest_lat, dest_lon,
      duration_min
    `source_time` here is a unix-seconds integer (not a human string), to
    simplify downstream consumption.
    """
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(test_cases_dir, "*.txt")))
    rows = []

    for fpath in files:
        vehicle_id = os.path.basename(fpath)  # keep .txt suffix, like Kaggle test CSV
        try:
            traj = load_trajectory(fpath)
        except Exception:
            continue
        if len(traj) < 20:
            continue

        kept = 0
        attempts = 0
        while kept < per_vehicle and attempts < per_vehicle * 5:
            attempts += 1
            s, d = _select_interior_pair(traj, rng=rng)
            if s < 0:
                continue
            src, dst = traj[s], traj[d]
            # Skip degenerate pairs where src and dst are nearly co-located
            hav = great_circle_distance(src.lat, src.lon, dst.lat, dst.lon)
            if hav < 100:  # less than 100m makes the duration noisy
                continue
            rows.append({
                "vehicle_id": vehicle_id,
                "source_lat": src.lat,
                "source_lon": src.lon,
                "source_time": int(src.timestamp),
                "dest_lat": dst.lat,
                "dest_lon": dst.lon,
                "duration_min": (dst.timestamp - src.timestamp) / 60.0,
            })
            kept += 1

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-vehicle", type=int, default=8,
                        help="How many synthetic queries per Test Cases vehicle (default 8)")
    parser.add_argument("--test-cases-dir", default="Test Cases")
    parser.add_argument("--out", default="kaggle_like_val.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = build_synthetic_val(
        test_cases_dir=args.test_cases_dir,
        per_vehicle=args.per_vehicle,
        seed=args.seed,
    )
    print(f"Built {len(df)} synthetic val rows across "
          f"{df['vehicle_id'].nunique()} vehicles.")
    print(f"Duration distribution (min):")
    for p in (5, 25, 50, 75, 95):
        print(f"  p{p}: {np.percentile(df['duration_min'], p):.2f}")
    print(f"  mean: {df['duration_min'].mean():.2f}  max: {df['duration_min'].max():.2f}")

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
