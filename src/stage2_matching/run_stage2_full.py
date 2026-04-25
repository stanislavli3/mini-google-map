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
# --- src/ path bootstrap (cross-subdir sibling imports) ---
import os as _os, sys as _sys
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _SRC)
import _path_bootstrap  # noqa: F401  (registers all src/<subdir>/ on sys.path)
PROJECT_ROOT = _path_bootstrap.PROJECT_ROOT
del _os, _sys, _SRC
# --- end bootstrap ---


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
from map_matching_fast import (
    hmm_map_match_fast as hmm_map_match,
    iter_matched_pairs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPHML_PATH = "sf_road_network.graphml"
TRAJ_DIR = "Trajectories"

OBSERVED_PATH = "observed_speeds.pkl"   # phase A output
COMPLETE_PATH = "complete_speeds.pkl"   # phase B output (what Stage 3 loads)
PARTIAL_PATH = "observed_speeds.partial.pkl"
MATCHED_CACHE_DIR = "matched_cache"     # per-file matched pickles, reused across runs
PARAMS_PATH = "matching_params.pkl"     # optional, written by calibrate_params.py

TIME_INTERVAL = 30 * 60
MAX_POINTS_PER_FILE = 150
LOG_EVERY = 10
CHECKPOINT_EVERY = 50
MAX_REASONABLE_SPEED = 50.0
DEFAULT_SIGMA_Z = 4.07
DEFAULT_BETA = 3.0

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
# Road-class-aware per-edge speed caps (m/s). Anything above the cap for
# either endpoint edge is almost certainly a map-matching artefact (e.g. a
# wrong parallel street), so we drop it before it contaminates the
# aggregate. Residential 200 km/h "speeds" were a real source of bad
# physics ETAs before this filter.
SPEED_CAP_MS = {
    "motorway": 42.0, "motorway_link": 30.0,
    "trunk": 30.0, "trunk_link": 25.0,
    "primary": 25.0, "primary_link": 22.0,
    "secondary": 22.0, "secondary_link": 20.0,
    "tertiary": 20.0, "tertiary_link": 20.0,
    "residential": 18.0, "living_street": 10.0,
    "service": 10.0, "unclassified": 20.0,
}


def _edge_speed_cap(edge_id, edge_rtype):
    if edge_rtype is None:
        return MAX_REASONABLE_SPEED
    rt = edge_rtype.get(edge_id, "unclassified")
    return SPEED_CAP_MS.get(rt, 20.0)


def compute_segment_speeds(trajectory, matched, interval=TIME_INTERVAL, edge_rtype=None):
    """Aggregate observed speeds per (edge, time_bin).

    `matched` is aligned with `trajectory`; entries may be None where the
    matcher dropped a point (preprocessing, no candidates, path break).
    Pairs touching None are skipped. `edge_rtype` (optional) enables a
    road-class-aware speed cap on top of the global MAX_REASONABLE_SPEED.
    """
    segment_speeds = defaultdict(list)
    for pt_i, pt_next, match_i, match_next in iter_matched_pairs(trajectory, matched):
        t_i = pt_i.timestamp
        t_next = pt_next.timestamp
        edge_i, (lat_i, lon_i) = match_i
        edge_next, (lat_next, lon_next) = match_next

        avg_ts = (t_i + t_next) // 2
        time_bin = timestamp_to_time_bin(avg_ts, interval)

        distance = great_circle_distance(lat_i, lon_i, lat_next, lon_next)
        time_delta = abs(t_next - t_i)
        if time_delta == 0 or distance == 0:
            continue

        speed = distance / time_delta
        if speed > MAX_REASONABLE_SPEED:
            continue
        cap = max(_edge_speed_cap(edge_i, edge_rtype),
                  _edge_speed_cap(edge_next, edge_rtype))
        if speed > cap:
            continue

        # Confidence weighting: occupied (flag=1) pairs are more reliable
        # because the cab has a passenger and a destination. Give them 2x
        # weight in the median aggregate by duplicating the observation.
        weight = 2 if (pt_i.flag == 1 and pt_next.flag == 1) else 1
        edges_touched = [edge_i] if edge_i == edge_next else [edge_i, edge_next]
        for e in edges_touched:
            segment_speeds[(e, time_bin)].extend([speed] * weight)

    return dict(segment_speeds)


def _build_edge_rtype(edges_gdf):
    """Minimal edge -> highway-tag dict, used for road-class speed caps."""
    rt = {}
    for _, row in edges_gdf.iterrows():
        tag = row.get("highway", "unclassified")
        if isinstance(tag, list):
            tag = tag[0]
        rt[(row["u"], row["v"], row["key"])] = tag if tag else "unclassified"
    return rt


def aggregate_speeds(all_segment_speeds):
    return {
        k: float(np.median(v))
        for k, v in all_segment_speeds.items()
        if v
    }


def _log_progress(done, total, t0, n_success, n_failed, n_skipped, all_segment_speeds):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"  [{done}/{total}] "
        f"ok={n_success} fail={n_failed} skip={n_skipped} | "
        f"{len(all_segment_speeds):,} keys | "
        f"{rate:.2f} files/s | "
        f"elapsed={format_time(elapsed)} | "
        f"ETA={format_time(eta)}"
    )


def _save_partial(all_segment_speeds):
    partial = aggregate_speeds(dict(all_segment_speeds))
    with open(PARTIAL_PATH, "wb") as fh:
        pickle.dump(partial, fh)


def _load_matching_params():
    """Load calibrated (sigma_z, beta) from PARAMS_PATH if present."""
    if not os.path.exists(PARAMS_PATH):
        return DEFAULT_SIGMA_Z, DEFAULT_BETA
    try:
        with open(PARAMS_PATH, "rb") as fh:
            p = pickle.load(fh)
        sz = float(p.get("sigma_z", DEFAULT_SIGMA_Z))
        bt = float(p.get("beta", DEFAULT_BETA))
        print(f"  Loaded calibrated params: sigma_z={sz:.2f}m, beta={bt:.2f}m")
        return sz, bt
    except Exception as e:
        print(f"  WARNING: failed to read {PARAMS_PATH} ({e}); using defaults")
        return DEFAULT_SIGMA_Z, DEFAULT_BETA


def _match_one(fpath, sigma_z, beta, cache_dir):
    """Match a single trajectory file, caching the matched pickle to disk.

    Returns (trajectory, matched) or (None, None) on failure / skip.
    Uses `cache_dir/<stem>.pkl` as a per-file cache so re-runs of phase A
    (e.g. to re-tune the aggregator) don't pay the Viterbi cost again.
    """
    stem = os.path.splitext(os.path.basename(fpath))[0]
    cache_path = os.path.join(cache_dir, f"{stem}.pkl")

    try:
        traj = load_trajectory(fpath)
    except Exception:
        return None, None
    if len(traj) < 4:
        return None, None
    if len(traj) > MAX_POINTS_PER_FILE:
        traj = traj[:MAX_POINTS_PER_FILE]

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                matched = pickle.load(fh)
            if matched and len(matched) == len(traj):
                return traj, matched
        except Exception:
            pass  # fall through and re-match

    try:
        matched = hmm_map_match(
            traj, _WORKER_G, _WORKER_EDGES_GDF, _WORKER_KDTREE, _WORKER_EDGE_INDEX,
            sigma_z=sigma_z, beta=beta,
        )
    except Exception:
        return None, None

    if not matched:
        return None, None

    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(matched, fh)
    except Exception:
        pass  # cache is advisory; don't fail the match
    return traj, matched


# Module-level worker state. In the single-process path we populate these
# in run_phase_a. In the multiprocessing path, _worker_init does it per
# worker. Same state either way means _match_one is agnostic.
_WORKER_G = None
_WORKER_EDGES_GDF = None
_WORKER_KDTREE = None
_WORKER_EDGE_INDEX = None


def _worker_init(graphml_path):
    """Pool initializer: load the road network once per worker."""
    global _WORKER_G, _WORKER_EDGES_GDF, _WORKER_KDTREE, _WORKER_EDGE_INDEX
    _WORKER_G, _WORKER_EDGES_GDF, _WORKER_KDTREE, _WORKER_EDGE_INDEX = (
        load_road_network(graphml_path)
    )


def _match_one_for_pool(args):
    """Pool.imap wrapper — takes a tuple because imap takes a single arg."""
    fpath, sigma_z, beta, cache_dir = args
    return fpath, _match_one(fpath, sigma_z, beta, cache_dir)


def run_phase_a(G, edges_gdf, kdtree, edge_index, files, workers=1):
    """Map-match every trajectory and collect observed speeds.

    Per-file checkpointing: each matched sequence is pickled under
    MATCHED_CACHE_DIR. On re-runs we hit the cache and skip Viterbi.

    `workers` > 1 enables a multiprocessing.Pool with per-worker road-
    network loading. Use 1 on memory-constrained machines (the network
    is ~100MB per worker).
    """
    print("\n" + "=" * 70)
    print("PHASE A: OBSERVATION")
    print("=" * 70)
    os.makedirs(MATCHED_CACHE_DIR, exist_ok=True)

    sigma_z, beta = _load_matching_params()
    edge_rtype = _build_edge_rtype(edges_gdf)

    print(f"Processing {len(files)} trajectory files with {workers} worker(s)...")

    # Populate single-process worker state so _match_one works without a Pool
    global _WORKER_G, _WORKER_EDGES_GDF, _WORKER_KDTREE, _WORKER_EDGE_INDEX
    _WORKER_G, _WORKER_EDGES_GDF, _WORKER_KDTREE, _WORKER_EDGE_INDEX = (
        G, edges_gdf, kdtree, edge_index
    )

    all_segment_speeds = defaultdict(list)
    t0 = time.time()
    n_success = n_failed = n_skipped = 0

    def _consume(fpath, traj, matched, idx):
        nonlocal n_success, n_failed, n_skipped
        if traj is None or matched is None:
            n_failed += 1
            return
        non_none = sum(1 for m in matched if m is not None)
        if non_none < 2:
            n_skipped += 1
            return
        try:
            seg_speeds = compute_segment_speeds(traj, matched, edge_rtype=edge_rtype)
            for key, speeds in seg_speeds.items():
                all_segment_speeds[key].extend(speeds)
            n_success += 1
        except Exception as e:
            print(f"  [{idx+1}] {os.path.basename(fpath)} — speed compute failed: {e}")
            n_failed += 1

    if workers <= 1:
        for i, fpath in enumerate(files):
            traj, matched = _match_one(fpath, sigma_z, beta, MATCHED_CACHE_DIR)
            _consume(fpath, traj, matched, i)

            if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(files):
                _log_progress(i + 1, len(files), t0, n_success, n_failed,
                              n_skipped, all_segment_speeds)
            if (i + 1) % CHECKPOINT_EVERY == 0:
                _save_partial(all_segment_speeds)
    else:
        from multiprocessing import Pool
        tasks = [(f, sigma_z, beta, MATCHED_CACHE_DIR) for f in files]
        with Pool(workers, initializer=_worker_init, initargs=(GRAPHML_PATH,)) as pool:
            for i, (fpath, (traj, matched)) in enumerate(
                pool.imap_unordered(_match_one_for_pool, tasks, chunksize=4)
            ):
                _consume(fpath, traj, matched, i)
                if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(files):
                    _log_progress(i + 1, len(files), t0, n_success, n_failed,
                                  n_skipped, all_segment_speeds)
                if (i + 1) % CHECKPOINT_EVERY == 0:
                    _save_partial(all_segment_speeds)

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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for phase A "
                             "(each loads its own road network, ~100MB/worker)")
    parser.add_argument("--clear-match-cache", action="store_true",
                        help=f"Delete {MATCHED_CACHE_DIR}/ before running. Use "
                             "after changing sigma_z/beta via calibration.")
    args = parser.parse_args()

    if args.clear_match_cache and os.path.exists(MATCHED_CACHE_DIR):
        import shutil
        print(f"Clearing {MATCHED_CACHE_DIR}/ ...")
        shutil.rmtree(MATCHED_CACHE_DIR)

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
        observed_speeds = run_phase_a(
            G, edges_gdf, kdtree, edge_index, files, workers=args.workers
        )

    if args.skip_phase_b:
        print("\n--skip-phase-b set; stopping after phase A.")
        print(f"If you want Stage 3 to consume the observed-only file, rename:")
        print(f"    mv {OBSERVED_PATH} {COMPLETE_PATH}")
        return

    run_phase_b(G, edges_gdf, observed_speeds)

    print("\nDone.")
    print("=" * 70)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
