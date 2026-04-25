"""
enhance_vehicle_features.py
===========================

Augment an existing `vehicle_features.pkl` with 24 per-hour median-speed
columns (`v_hour_speed_h00` … `v_hour_speed_h23`) computed directly from
raw GPS pairs in `Test Cases/*.txt`. Avoids having to re-run the expensive
map matching in `process_test_cases.py`.

The per-hour speed is a rough proxy — it uses straight-line speed between
consecutive GPS points, not matched-road speed — but for a driver-level
hourly aggregate that distinction is noise.
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


import glob
import os
import pickle
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from map_matching_solution import load_trajectory, great_circle_distance


TEST_CASES_DIR = "Test Cases"
VEHICLE_FEATURES_PATH = "vehicle_features.pkl"
MAX_REASONABLE_SPEED = 50.0   # m/s, dropped as matcher artifact


def main():
    with open(VEHICLE_FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    print(f"Loaded vehicle_features.pkl with {len(features)} entries")

    already_enhanced = sum(
        1 for v in features.values()
        if "v_hour_speed_h00" in v
    )
    if already_enhanced == len(features):
        print("Already enhanced. Nothing to do.")
        return

    files = sorted(glob.glob(os.path.join(TEST_CASES_DIR, "*.txt")))
    print(f"Processing {len(files)} trajectory files...")

    for i, fpath in enumerate(files):
        vid = os.path.basename(fpath)   # keep .txt, matches feature dict keys
        if vid not in features:
            continue

        try:
            traj = load_trajectory(fpath)
        except Exception:
            continue

        per_hour_speeds = defaultdict(list)
        for a, b in zip(traj, traj[1:]):
            dt = abs(b.timestamp - a.timestamp)
            if dt == 0:
                continue
            d = great_circle_distance(a.lat, a.lon, b.lat, b.lon)
            if d == 0:
                continue
            s = d / dt
            if s > MAX_REASONABLE_SPEED:
                continue
            hour = datetime.fromtimestamp(a.timestamp, tz=timezone.utc).hour
            per_hour_speeds[hour].append(s)

        all_speeds = [s for speeds in per_hour_speeds.values() for s in speeds]
        overall_median = (
            float(np.median(all_speeds))
            if all_speeds
            else features[vid].get("v_median_speed_ms", 10.0)
        )

        for h in range(24):
            col = f"v_hour_speed_h{h:02d}"
            if per_hour_speeds[h]:
                features[vid][col] = float(np.median(per_hour_speeds[h]))
            else:
                features[vid][col] = overall_median

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]")

    # Back up the old file, write the enhanced one
    bak = VEHICLE_FEATURES_PATH + ".before_per_hour.bak"
    if not os.path.exists(bak):
        os.replace(VEHICLE_FEATURES_PATH, bak)
        with open(VEHICLE_FEATURES_PATH, "wb") as f:
            pickle.dump(features, f)
    else:
        # Bak already exists from a prior run — don't overwrite it.
        with open(VEHICLE_FEATURES_PATH, "wb") as f:
            pickle.dump(features, f)

    # Quick sanity print
    sample_vid = next(iter(features))
    sample = features[sample_vid]
    hourly_keys = [k for k in sample if k.startswith("v_hour_speed_h")]
    print(f"\nEnhanced {len(features)} vehicles.")
    print(f"New columns per vehicle: {len(hourly_keys)} "
          f"(expected 24)")
    if hourly_keys:
        vals = [sample[k] for k in sorted(hourly_keys)]
        print(f"Sample vehicle ({sample_vid}) hourly medians (m/s):")
        for i, v in enumerate(vals):
            print(f"  h{i:02d}: {v:.2f}", end="  ")
            if (i + 1) % 6 == 0:
                print()


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
