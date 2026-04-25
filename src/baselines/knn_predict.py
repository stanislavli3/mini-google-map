"""
knn_predict.py
==============
Per-vehicle k-NN predictor. Standalone script — does not need the
CatBoost model, the physics ETA, or the map-matched speed dictionary.
For each test row (vehicle_id, src, dst, source_time):
  1. Look at that vehicle's full historical trip set from Test Cases/.
  2. Filter to trips within ±hr_radius hours of the test hour-of-day.
  3. Score historical trips by haversine(hist_src, test_src) +
     haversine(hist_dst, test_dst).
  4. Return median duration of the top-k most similar trips.

Output: submission_knn.csv (or whatever --out you pass).

USAGE
-----
    python knn_predict.py --k 10 --hr-radius 2 --out submission_knn.csv
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


import argparse
import glob
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from map_matching_solution import load_trajectory, great_circle_distance


def hav_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    dp = p2 - p1
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def identify_trips(traj):
    if any(p.flag == 1 for p in traj):
        trips = []
        start = None
        for i, p in enumerate(traj):
            if p.flag == 1 and start is None:
                start = i
            elif p.flag == 0 and start is not None:
                if i - 1 - start >= 3:
                    trips.append((start, i - 1))
                start = None
        if start is not None and len(traj) - 1 - start >= 3:
            trips.append((start, len(traj) - 1))
        return trips
    GAP = 5 * 60
    trips = []
    start = 0
    for i in range(1, len(traj)):
        if traj[i].timestamp - traj[i - 1].timestamp > GAP:
            if i - 1 - start >= 3:
                trips.append((start, i - 1))
            start = i
    if len(traj) - 1 - start >= 3:
        trips.append((start, len(traj) - 1))
    return trips


def build_vehicle_history(test_cases_dir):
    vehicle_trips = {}
    for fpath in sorted(glob.glob(os.path.join(test_cases_dir, "*.txt"))):
        vid = os.path.basename(fpath)
        try:
            traj = load_trajectory(fpath)
        except Exception:
            continue
        trips = identify_trips(traj)
        rows = []
        for s, e in trips:
            dur_s = traj[e].timestamp - traj[s].timestamp
            if dur_s < 60 or dur_s > 2 * 3600:
                continue
            rows.append({
                'src_lat': traj[s].lat, 'src_lon': traj[s].lon,
                'dst_lat': traj[e].lat, 'dst_lon': traj[e].lon,
                'hour': datetime.fromtimestamp(traj[s].timestamp, tz=timezone.utc).hour,
                'dur_min': dur_s / 60,
            })
        if rows:
            vehicle_trips[vid] = pd.DataFrame(rows)
    return vehicle_trips


def predict_knn(test_row, vid_history, k=10, hr_radius=2):
    if vid_history is None or len(vid_history) == 0:
        return None
    h = vid_history
    hr_diff = np.minimum(np.abs(h.hour - test_row.hour),
                          24 - np.abs(h.hour - test_row.hour))
    h = h[hr_diff <= hr_radius]
    if len(h) == 0:
        return None
    src_d = hav_km(h.src_lat.values, h.src_lon.values,
                    test_row.source_lat, test_row.source_lon)
    dst_d = hav_km(h.dst_lat.values, h.dst_lon.values,
                    test_row.dest_lat, test_row.dest_lon)
    h = h.assign(score=src_d + dst_d).sort_values('score').head(k)
    return float(h.dur_min.median())


def parse_test_hour(s):
    return datetime.strptime(s, "%m/%d/%y %H:%M").hour


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test-csv', default='kaggle-test-file-minute.csv')
    p.add_argument('--test-cases-dir', default='Test Cases')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--hr-radius', type=int, default=2)
    p.add_argument('--fallback-kph', type=float, default=12.0)
    p.add_argument('--out', default='submission_knn.csv')
    args = p.parse_args()

    test = pd.read_csv(args.test_csv)
    test['hour'] = test.source_time.apply(parse_test_hour)
    test['hav_km'] = hav_km(test.source_lat, test.source_lon,
                             test.dest_lat, test.dest_lon)

    print(f"Building per-vehicle history from {args.test_cases_dir}/...")
    history = build_vehicle_history(args.test_cases_dir)
    print(f"  {len(history)} vehicles, "
          f"{sum(len(v) for v in history.values())} total trips")

    preds = []
    n_fallback = 0
    for _, row in test.iterrows():
        v = predict_knn(row, history.get(row.vehicle_id),
                         k=args.k, hr_radius=args.hr_radius)
        if v is None:
            v = max(0.5, row.hav_km / args.fallback_kph * 60)
            n_fallback += 1
        preds.append(v)
    preds = np.clip(preds, 0.5, 240)
    print(f"  fallback used for {n_fallback} rows")
    print(f"  predictions: mean={preds.mean():.2f} median={np.median(preds):.2f} "
          f"std={preds.std():.2f} max={preds.max():.2f}")

    pd.DataFrame({'id': test.id, 'duration_min': preds}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)
    main()
