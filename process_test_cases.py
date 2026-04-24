"""
process_test_cases.py
=====================

Two jobs:

1. Map-match the 50 trajectory files in "Test Cases/" and merge their
   observed speeds into complete_speeds.pkl. These vehicles are the exact
   ones queried in the Kaggle test set, so their historical speeds are
   directly relevant.

2. Extract per-vehicle feature aggregates from those same trajectories.
   For each of the 50 test vehicles, compute historical stats that can be
   joined into the test CSV before prediction.

The vehicle_id column in kaggle-test-file-minute.csv contains filenames WITH
the .txt extension (e.g., 'e270309b-0ccc-48a9-a4f5-2c7e54e57f73.txt'), so
this script preserves that format for join compatibility.

OUTPUTS
-------
  complete_speeds.pkl       (updated in place — previous version backed up
                             to complete_speeds.before_test_cases.pkl)
  vehicle_features.pkl      (dict mapping vehicle_id -> feature dict)

USAGE
-----
    python process_test_cases.py
"""

from __future__ import annotations

import os
import glob
import math
import pickle
import shutil
import time
import warnings
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

from map_matching_solution import (
    load_road_network,
    load_trajectory,
    great_circle_distance,
)
# Use the fast matcher we wrote earlier.
from map_matching_fast import (
    hmm_map_match_fast as hmm_map_match,
    iter_matched_pairs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPHML_PATH = "sf_road_network.graphml"
TEST_CASES_DIR = "Test Cases"
EXISTING_SPEEDS_PATH = "complete_speeds.pkl"
BACKUP_SPEEDS_PATH = "complete_speeds.before_test_cases.pkl"
VEHICLE_FEATURES_PATH = "vehicle_features.pkl"

TIME_INTERVAL = 30 * 60
MAX_POINTS_PER_FILE = 400  # reduced from 2000 for 5x Stage 5 speedup;
                            # vehicle aggregates (mean/median/percentiles)
                            # remain representative with a 400-point slice
MAX_REASONABLE_SPEED = 50.0


def timestamp_to_time_bin(ts, interval=TIME_INTERVAL):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    return (date_str, sec // interval)


# ---------------------------------------------------------------------------
# Part 1: map-match + merge into Stage 2 dict
# ---------------------------------------------------------------------------
def compute_segment_speeds(trajectory, matched):
    """Aggregate observed speeds. Skips pairs where either matched entry is
    None (dropped in preprocessing or past a path break)."""
    segment_speeds = defaultdict(list)
    for pt_i, pt_next, match_i, match_next in iter_matched_pairs(trajectory, matched):
        t_i = pt_i.timestamp
        t_next = pt_next.timestamp
        edge_i, (lat_i, lon_i) = match_i
        edge_next, (lat_next, lon_next) = match_next

        avg_ts = (t_i + t_next) // 2
        time_bin = timestamp_to_time_bin(avg_ts)

        distance = great_circle_distance(lat_i, lon_i, lat_next, lon_next)
        time_delta = abs(t_next - t_i)
        if time_delta == 0 or distance == 0:
            continue

        speed = distance / time_delta
        if speed > MAX_REASONABLE_SPEED:
            continue

        # Confidence weighting: occupied (flag=1) pairs are more reliable
        # because the cab has a passenger and a destination. Weight 2x.
        weight = 2 if (pt_i.flag == 1 and pt_next.flag == 1) else 1
        edges_touched = [edge_i] if edge_i == edge_next else [edge_i, edge_next]
        for e in edges_touched:
            segment_speeds[(e, time_bin)].extend([speed] * weight)

    return dict(segment_speeds)


def match_test_cases(G, edges_gdf, kdtree, edge_index):
    """
    Map-match every file in Test Cases, compute observed speeds.

    Returns:
        new_segment_speeds: dict[(edge, time_bin)] -> list[speed]
        per_vehicle_matched: dict[vehicle_id] -> (trajectory, matched)
                             (the matched trajectories, for feature extraction)
    """
    files = sorted(glob.glob(os.path.join(TEST_CASES_DIR, "*.txt")))
    print(f"Map-matching {len(files)} Test Cases files...")

    new_segment_speeds = defaultdict(list)
    per_vehicle_matched = {}

    t0 = time.time()
    for i, fpath in enumerate(files):
        vehicle_id = os.path.basename(fpath)  # keep .txt extension for join
        print(f"\n[{i+1}/{len(files)}] {vehicle_id}")

        traj = load_trajectory(fpath)
        if len(traj) > MAX_POINTS_PER_FILE:
            traj = traj[:MAX_POINTS_PER_FILE]

        matched = hmm_map_match(
            traj, G, edges_gdf, kdtree, edge_index,
            sigma_z=4.07, beta=3.0,
        )
        if not matched or len(matched) < 2:
            print(f"  skipped (insufficient matches)")
            continue

        per_vehicle_matched[vehicle_id] = (traj, matched)

        seg_speeds = compute_segment_speeds(traj, matched)
        for key, speeds in seg_speeds.items():
            new_segment_speeds[key].extend(speeds)

    elapsed = time.time() - t0
    print(f"\nMatched in {elapsed/60:.1f} min. "
          f"New (edge, time_bin) observations: {len(new_segment_speeds):,}")
    return dict(new_segment_speeds), per_vehicle_matched


def merge_into_complete_speeds(new_segment_speeds):
    """
    Load existing complete_speeds.pkl, merge new observations via median,
    write back. Backs up the original first.
    """
    if not os.path.exists(EXISTING_SPEEDS_PATH):
        print(f"No existing {EXISTING_SPEEDS_PATH} — creating fresh.")
        existing = {}
    else:
        shutil.copy(EXISTING_SPEEDS_PATH, BACKUP_SPEEDS_PATH)
        print(f"Backed up existing speeds to {BACKUP_SPEEDS_PATH}")
        with open(EXISTING_SPEEDS_PATH, "rb") as f:
            existing = pickle.load(f)
        print(f"Loaded {len(existing):,} existing (edge, time_bin) entries")

    # Existing is already aggregated (single median per key).
    # For the new observations, we have raw lists.
    # Combine: where new data exists, compute median of
    # [existing_value] + new_observations. Where new is absent, keep existing.

    updated = dict(existing)
    n_new = 0
    n_merged = 0
    for key, new_speeds in new_segment_speeds.items():
        if not new_speeds:
            continue
        if key in updated:
            # Re-median with existing aggregated value + all new obs.
            # Technically biased toward the existing median because it's
            # one value vs many raw observations. For our purposes this is
            # acceptable given we don't have the raw pre-aggregation data.
            combined = new_speeds + [updated[key]]
            updated[key] = float(np.median(combined))
            n_merged += 1
        else:
            updated[key] = float(np.median(new_speeds))
            n_new += 1

    with open(EXISTING_SPEEDS_PATH, "wb") as f:
        pickle.dump(updated, f)

    print(f"\nMerged complete_speeds.pkl:")
    print(f"  New entries added:      {n_new:,}")
    print(f"  Existing entries merged: {n_merged:,}")
    print(f"  Total entries now:       {len(updated):,}")
    return updated


# ---------------------------------------------------------------------------
# Part 2: per-vehicle features
# ---------------------------------------------------------------------------
def compute_vehicle_features(per_vehicle_matched, edges_gdf):
    """
    For each vehicle, compute historical aggregates from its matched trips.

    Features per vehicle:
      - v_mean_speed_ms             mean instantaneous speed (all pairs)
      - v_median_speed_ms           median ditto
      - v_p20_speed_ms / v_p80_speed_ms   speed spread
      - v_total_km                  historical distance driven (rough)
      - v_n_points                  number of GPS points
      - v_hour_dominant             most common hour of activity
      - v_hour_entropy              how concentrated they are in time
      - v_pct_highway               fraction of matched edges classified as motorway*
      - v_typical_trip_min          median trip duration (flag-based trips)
    """
    # Build road-type lookup
    rtype = {}
    for _, row in edges_gdf.iterrows():
        et = (row["u"], row["v"], row["key"])
        rt = row.get("highway", "unclassified")
        if isinstance(rt, list):
            rt = rt[0]
        rtype[et] = rt or "unclassified"

    def is_highway(rt):
        return rt and ("motorway" in rt or "trunk" in rt)

    features = {}

    for vehicle_id, (traj, matched) in per_vehicle_matched.items():
        speeds = []
        speeds_by_hour = defaultdict(list)  # hour -> list of instantaneous m/s
        total_dist = 0.0
        hwy_count = 0
        total_matches = 0
        hours = Counter()

        for pt_i, pt_next, _match_i, match_next in iter_matched_pairs(traj, matched):
            t_i = pt_i.timestamp
            t_next = pt_next.timestamp
            _, (lat_i, lon_i) = _match_i
            edge_next, (lat_next, lon_next) = match_next

            d = great_circle_distance(lat_i, lon_i, lat_next, lon_next)
            dt = abs(t_next - t_i)
            if dt == 0 or d == 0:
                continue
            s = d / dt
            if s > MAX_REASONABLE_SPEED:
                continue

            speeds.append(s)
            total_dist += d
            if is_highway(rtype.get(edge_next, "")):
                hwy_count += 1
            total_matches += 1

            hour = datetime.fromtimestamp(t_i, tz=timezone.utc).hour
            hours[hour] += 1
            speeds_by_hour[hour].append(s)

        if total_matches < 2:
            continue

        # Trip durations via flag-based segmentation
        trip_durations = []
        if any(p.flag == 1 for p in traj):
            start = None
            for i, p in enumerate(traj):
                if p.flag == 1 and start is None:
                    start = i
                elif p.flag == 0 and start is not None:
                    dur = traj[i - 1].timestamp - traj[start].timestamp
                    if 60 <= dur <= 2 * 3600:
                        trip_durations.append(dur / 60.0)
                    start = None
            if start is not None:
                dur = traj[-1].timestamp - traj[start].timestamp
                if 60 <= dur <= 2 * 3600:
                    trip_durations.append(dur / 60.0)

        if not speeds:
            continue

        speeds_arr = np.asarray(speeds)

        # Hour entropy (normalized)
        total_hour_obs = sum(hours.values())
        h_entropy = 0.0
        if total_hour_obs > 0:
            for c in hours.values():
                p = c / total_hour_obs
                if p > 0:
                    h_entropy -= p * math.log(p)
        h_entropy_norm = h_entropy / math.log(24)  # max entropy is log(24)

        overall_median = float(np.median(speeds_arr))
        # Per-hour median speed for this vehicle; missing hours fall back to
        # the overall median so the feature is never NaN.
        per_hour_median = {}
        for h in range(24):
            if speeds_by_hour[h]:
                per_hour_median[h] = float(np.median(speeds_by_hour[h]))
            else:
                per_hour_median[h] = overall_median

        row = {
            "v_mean_speed_ms": float(speeds_arr.mean()),
            "v_median_speed_ms": overall_median,
            "v_p20_speed_ms": float(np.percentile(speeds_arr, 20)),
            "v_p80_speed_ms": float(np.percentile(speeds_arr, 80)),
            "v_total_km": total_dist / 1000.0,
            "v_n_points": len(traj),
            "v_hour_dominant": int(hours.most_common(1)[0][0]) if hours else 12,
            "v_hour_entropy": float(h_entropy_norm),
            "v_pct_highway": (hwy_count / total_matches) if total_matches else 0.0,
            "v_typical_trip_min": (
                float(np.median(trip_durations)) if trip_durations else 10.0
            ),
            "v_n_trips": len(trip_durations),
        }
        for h in range(24):
            row[f"v_hour_speed_h{h:02d}"] = per_hour_median[h]

        features[vehicle_id] = row

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PROCESS TEST CASES")
    print("=" * 70)

    print("\nLoading road network...")
    G, edges_gdf, kdtree, edge_index = load_road_network(GRAPHML_PATH)

    # Part 1: map-match + merge
    new_segment_speeds, per_vehicle_matched = match_test_cases(
        G, edges_gdf, kdtree, edge_index
    )

    merge_into_complete_speeds(new_segment_speeds)

    # Part 2: per-vehicle features
    print("\n" + "=" * 70)
    print("EXTRACTING PER-VEHICLE FEATURES")
    print("=" * 70)
    features = compute_vehicle_features(per_vehicle_matched, edges_gdf)
    print(f"Extracted features for {len(features)} vehicles")

    # Sample: print one
    if features:
        sample_vid = list(features.keys())[0]
        print(f"\nSample vehicle: {sample_vid}")
        for k, v in features[sample_vid].items():
            print(f"  {k}: {v}")

    with open(VEHICLE_FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)
    print(f"\nSaved to {VEHICLE_FEATURES_PATH}")

    print("\nDone. Next: update stage3_eta_prediction.py to join these features.")


if __name__ == "__main__":
    main()
