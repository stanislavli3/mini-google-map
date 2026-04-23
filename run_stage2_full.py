"""
run_stage2_full.py
==================

Self-contained runner for Stage 2: map-match every trajectory, compute
observed segment speeds per (date, time_bin), then propagate to nearby
uncovered edges. Saves `complete_speeds.pkl` for Stage 3.

Two phases:
  Phase A — Map-match every trajectory, compute observed speeds.
            Checkpoints after every 50 files.
  Phase B — Propagate observed speeds to neighboring uncovered edges via
            similarity-weighted iteration (like your original
            `estimate_missing_speeds`, but tuned for runtime).

Both phases save separately:
  observed_speeds.pkl   — raw observations (phase A output)
  complete_speeds.pkl   — observations + propagated estimates (phase B output)

Stage 3 should load `complete_speeds.pkl`. If phase B crashes for any
reason, you can still use `observed_speeds.pkl` as a fallback.

USAGE
-----
    source .venv/bin/activate
    python run_stage2_full.py

    # Skip re-running phase A if observed_speeds.pkl already exists:
    python run_stage2_full.py --skip-phase-a

    # Only run phase A (if you want to try Stage 3 without propagation first):
    python run_stage2_full.py --skip-phase-b
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
import math
import pickle
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

warnings.filterwarnings("ignore")

from map_matching_solution import (
    GPSPoint,
    load_road_network,
    load_trajectory,
    great_circle_distance,
)
from map_matching_fast import hmm_map_match_fast as hmm_map_match


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPHML_PATH = "sf_road_network.graphml"
TRAJ_DIR = "Trajectories"

OBSERVED_PATH = "observed_speeds.pkl"   # phase A output
COMPLETE_PATH = "complete_speeds.pkl"   # phase B output (what Stage 3 loads)
PARTIAL_PATH = "observed_speeds.partial.pkl"

TIME_INTERVAL = 30 * 60
MAX_POINTS_PER_FILE = 150
LOG_EVERY = 10
CHECKPOINT_EVERY = 50
MAX_REASONABLE_SPEED = 50.0

# Phase B tuning
PROPAGATION_ITERATIONS = 3
# (3 hops captures most of the useful signal; iterations 4-5 mostly smear noise)

ROAD_TYPE_DEFAULT_SPEED = {
    "motorway": 29.1, "motorway_link": 22.4,
    "trunk": 20.1, "trunk_link": 15.6,
    "primary": 15.6, "primary_link": 13.4,
    "secondary": 13.4, "secondary_link": 11.2,
    "tertiary": 11.2, "tertiary_link": 11.2,
    "residential": 11.2, "living_street": 6.7,
    "service": 6.7, "unclassified": 11.2,
}
DEFAULT_SPEED = 11.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def timestamp_to_time_bin(ts, interval=TIME_INTERVAL):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    return (date_str, sec // interval)


def format_time(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# PHASE A: observation
# ---------------------------------------------------------------------------
def compute_segment_speeds(trajectory, matched, interval=TIME_INTERVAL):
    """Same logic as your Stage 2 notebook."""
    segment_speeds = defaultdict(list)
    n_pairs = min(len(trajectory), len(matched)) - 1

    for i in range(n_pairs):
        t_i = trajectory[i].timestamp
        t_next = trajectory[i + 1].timestamp
        edge_i, (lat_i, lon_i) = matched[i]
        edge_next, (lat_next, lon_next) = matched[i + 1]

        avg_ts = (t_i + t_next) // 2
        time_bin = timestamp_to_time_bin(avg_ts, interval)

        distance = great_circle_distance(lat_i, lon_i, lat_next, lon_next)
        time_delta = abs(t_next - t_i)

        if time_delta == 0 or distance == 0:
            continue

        speed = distance / time_delta
        if speed > MAX_REASONABLE_SPEED:
            continue

        if edge_i == edge_next:
            segment_speeds[(edge_i, time_bin)].append(speed)
        else:
            segment_speeds[(edge_i, time_bin)].append(speed)
            segment_speeds[(edge_next, time_bin)].append(speed)

    return dict(segment_speeds)


def aggregate_speeds(all_segment_speeds):
    return {
        k: float(np.median(v))
        for k, v in all_segment_speeds.items()
        if v
    }


def run_phase_a(G, edges_gdf, kdtree, edge_index, files):
    """Map-match every trajectory and collect observed speeds."""
    print("\n" + "=" * 70)
    print("PHASE A: OBSERVATION")
    print("=" * 70)
    print(f"Processing {len(files)} trajectory files...")

    all_segment_speeds = defaultdict(list)
    t0 = time.time()
    n_success = n_failed = n_skipped = 0

    for i, fpath in enumerate(files):
        print(f"[{i+1}/{len(files)}] {os.path.basename(fpath)}")
        try:
            traj = load_trajectory(fpath)
        except Exception as e:
            print(f"  [{i+1}] {os.path.basename(fpath)} — load failed: {e}")
            n_failed += 1
            continue

        if len(traj) < 4:
            n_skipped += 1
            continue

        if len(traj) > MAX_POINTS_PER_FILE:
            traj = traj[:MAX_POINTS_PER_FILE]

        try:
            matched = hmm_map_match(
                traj, G, edges_gdf, kdtree, edge_index,
                sigma_z=4.07, beta=3.0,
            )
        except Exception as e:
            print(f"  [{i+1}] {os.path.basename(fpath)} — match failed: {e}")
            n_failed += 1
            continue

        if not matched or len(matched) < 2:
            n_skipped += 1
            continue

        try:
            seg_speeds = compute_segment_speeds(traj, matched)
            for key, speeds in seg_speeds.items():
                all_segment_speeds[key].extend(speeds)
            n_success += 1
        except Exception as e:
            print(f"  [{i+1}] {os.path.basename(fpath)} — speed compute failed: {e}")
            n_failed += 1

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(files) - (i + 1)) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(files)}] "
                f"ok={n_success} fail={n_failed} skip={n_skipped} | "
                f"{len(all_segment_speeds):,} keys | "
                f"{rate:.2f} files/s | "
                f"elapsed={format_time(elapsed)} | "
                f"ETA={format_time(eta)}"
            )

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  -> checkpoint...")
            partial = aggregate_speeds(dict(all_segment_speeds))
            with open(PARTIAL_PATH, "wb") as fh:
                pickle.dump(partial, fh)

    observed_speeds = aggregate_speeds(dict(all_segment_speeds))

    unique_bins = {tb for (_, tb) in observed_speeds}
    unique_dates = {tb[0] for tb in unique_bins}
    print(f"\nPhase A complete:")
    print(f"  Files: ok={n_success} fail={n_failed} skip={n_skipped}")
    print(f"  (edge, time_bin) pairs: {len(observed_speeds):,}")
    print(f"  Unique time bins: {len(unique_bins):,}")
    if unique_dates:
        print(f"  Date range: {min(unique_dates)} to {max(unique_dates)}")
    print(f"  Elapsed: {format_time(time.time() - t0)}")

    with open(OBSERVED_PATH, "wb") as fh:
        pickle.dump(observed_speeds, fh)
    print(f"  Saved to {OBSERVED_PATH}")

    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    return observed_speeds


# ---------------------------------------------------------------------------
# PHASE B: propagation
# ---------------------------------------------------------------------------
def build_edge_metadata(edges_gdf):
    edge_rtype = {}
    edge_length = {}
    for _, row in edges_gdf.iterrows():
        et = (row["u"], row["v"], row["key"])
        rt = row.get("highway", "unclassified")
        if isinstance(rt, list):
            rt = rt[0]
        edge_rtype[et] = rt if rt else "unclassified"
        edge_length[et] = float(row.get("length", 0.0))
    return edge_rtype, edge_length


def build_adjacency(G, all_edges):
    """edge -> set of neighboring edges (sharing any endpoint)."""
    edge_set = set(all_edges)
    adj = defaultdict(set)
    for u, v, k in all_edges:
        et = (u, v, k)
        for neighbor in G.out_edges(v, keys=True):
            if tuple(neighbor) in edge_set:
                adj[et].add(tuple(neighbor))
        for neighbor in G.in_edges(u, keys=True):
            if tuple(neighbor) in edge_set:
                adj[et].add(tuple(neighbor))
        for neighbor in G.in_edges(v, keys=True):
            if tuple(neighbor) in edge_set:
                adj[et].add(tuple(neighbor))
        for neighbor in G.out_edges(u, keys=True):
            if tuple(neighbor) in edge_set:
                adj[et].add(tuple(neighbor))
        adj[et].discard(et)
    return dict(adj)


def run_phase_b(G, edges_gdf, observed_speeds):
    """Propagate observed speeds to nearby uncovered edges."""
    print("\n" + "=" * 70)
    print("PHASE B: PROPAGATION")
    print("=" * 70)

    all_edges = [
        (row["u"], row["v"], row["key"])
        for _, row in edges_gdf.iterrows()
    ]
    print(f"  Network: {len(all_edges):,} edges")

    edge_rtype, edge_length = build_edge_metadata(edges_gdf)

    print("  Building edge adjacency (one-time)...")
    t0 = time.time()
    adj = build_adjacency(G, all_edges)
    print(f"    done in {format_time(time.time() - t0)}")

    # Group observed speeds by time_bin
    bin_to_observed = defaultdict(dict)
    for (edge_id, tbin), speed in observed_speeds.items():
        bin_to_observed[tbin][edge_id] = speed

    active_bins = sorted(bin_to_observed.keys())
    print(f"  Active time bins: {len(active_bins):,}")

    def road_type_base(rt):
        return rt.replace("_link", "")

    def road_type_similarity(a, b):
        if a == b:
            return 1.0
        if road_type_base(a) == road_type_base(b):
            return 0.5
        return 0.2

    def length_similarity(la, lb):
        m = max(la, lb)
        if m == 0:
            return 1.0
        return math.exp(-abs(la - lb) / m)

    complete_speeds = dict(observed_speeds)
    observed_keys = set(observed_speeds.keys())

    total_bins = len(active_bins)
    t_start = time.time()

    for bin_idx, tbin in enumerate(active_bins):
        current_bin_speeds = dict(bin_to_observed[tbin])

        for iteration in range(PROPAGATION_ITERATIONS):
            hop_decay = 0.8 ** iteration

            # Only look at edges adjacent to currently-known edges
            # (massive speedup vs. iterating over all 27k edges every pass)
            frontier = set()
            for edge in current_bin_speeds.keys():
                frontier.update(adj.get(edge, []))
            frontier -= set(current_bin_speeds.keys())

            newly_estimated = {}
            for target_edge in frontier:
                target_rt = edge_rtype.get(target_edge, "unclassified")
                target_len = edge_length.get(target_edge, 0.0)

                weighted_sum = 0.0
                weight_total = 0.0

                for neighbor in adj.get(target_edge, []):
                    if neighbor not in current_bin_speeds:
                        continue

                    neighbor_speed = current_bin_speeds[neighbor]
                    neighbor_rt = edge_rtype.get(neighbor, "unclassified")
                    neighbor_len = edge_length.get(neighbor, 0.0)

                    w_road = road_type_similarity(target_rt, neighbor_rt)
                    w_len = length_similarity(target_len, neighbor_len)
                    w_obs = 1.5 if (neighbor, tbin) in observed_keys else 1.0
                    w = w_road * w_len * w_obs * hop_decay

                    weighted_sum += w * neighbor_speed
                    weight_total += w

                if weight_total > 0:
                    newly_estimated[target_edge] = weighted_sum / weight_total

            current_bin_speeds.update(newly_estimated)
            if not newly_estimated:
                break

        for edge, speed in current_bin_speeds.items():
            complete_speeds[(edge, tbin)] = speed

        if (bin_idx + 1) % 50 == 0 or (bin_idx + 1) == total_bins:
            elapsed = time.time() - t_start
            rate = (bin_idx + 1) / elapsed
            eta = (total_bins - (bin_idx + 1)) / rate if rate > 0 else 0
            print(
                f"  [{bin_idx+1}/{total_bins}] bins | "
                f"{len(complete_speeds):,} total keys | "
                f"elapsed={format_time(elapsed)} | "
                f"ETA={format_time(eta)}"
            )

    # Road-type default for any remaining gaps
    print("  Filling remaining gaps with road-type defaults...")
    n_filled = 0
    for tbin in active_bins:
        for edge in all_edges:
            key = (edge, tbin)
            if key not in complete_speeds:
                rt = edge_rtype.get(edge, "unclassified")
                complete_speeds[key] = ROAD_TYPE_DEFAULT_SPEED.get(rt, DEFAULT_SPEED)
                n_filled += 1

    print(f"\nPhase B complete:")
    print(f"  Total keys: {len(complete_speeds):,}")
    print(f"  Observed: {len(observed_keys):,}")
    print(f"  Propagated: {len(complete_speeds) - len(observed_keys) - n_filled:,}")
    print(f"  Road-type default: {n_filled:,}")
    print(f"  Elapsed: {format_time(time.time() - t_start)}")

    with open(COMPLETE_PATH, "wb") as fh:
        pickle.dump(complete_speeds, fh)
    size_mb = os.path.getsize(COMPLETE_PATH) / (1024 * 1024)
    print(f"  Saved to {COMPLETE_PATH} ({size_mb:.1f} MB)")

    return complete_speeds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phase-a", action="store_true",
                        help="Skip phase A; load existing observed_speeds.pkl")
    parser.add_argument("--skip-phase-b", action="store_true",
                        help="Skip phase B; stop after phase A")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 2 — FULL COVERAGE RUN")
    print("=" * 70)
    print(f"\nLoading road network from {GRAPHML_PATH}...")
    G, edges_gdf, kdtree, edge_index = load_road_network(GRAPHML_PATH)

    if args.skip_phase_a:
        if not os.path.exists(OBSERVED_PATH):
            print(f"ERROR: --skip-phase-a but {OBSERVED_PATH} doesn't exist.")
            sys.exit(1)
        print(f"\nLoading existing {OBSERVED_PATH}...")
        with open(OBSERVED_PATH, "rb") as fh:
            observed_speeds = pickle.load(fh)
        print(f"  {len(observed_speeds):,} observed keys")
    else:
        files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.txt")))
        if not files:
            print(f"ERROR: no .txt files in {TRAJ_DIR}")
            sys.exit(1)
        observed_speeds = run_phase_a(G, edges_gdf, kdtree, edge_index, files)

    if args.skip_phase_b:
        print("\n--skip-phase-b set; stopping after phase A.")
        print(f"If you want Stage 3 to consume the observed-only file, rename:")
        print(f"    mv {OBSERVED_PATH} {COMPLETE_PATH}")
        return

    run_phase_b(G, edges_gdf, observed_speeds)

    print("\nDone.")
    print("=" * 70)


if __name__ == "__main__":
    main()
