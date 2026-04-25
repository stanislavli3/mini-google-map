"""
baseline_metrics.py
===================

Phase 0 of the map-matching improvement plan: run current matcher on a
sample of trajectory files and write a reference-metrics file so future
runs can be compared against it.

Metrics captured:
  - wall-clock time (total, per-file median, per-file p90)
  - number of (segment, time_bin) speed observations
  - number of unique edges with at least one observation
  - % of edges in the road network covered
  - median # observations per covered edge

Output: baseline_metrics.txt in the project root.

USAGE
-----
    python baseline_metrics.py --files 20
    python baseline_metrics.py --files 100 --workers 4
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
import pickle
import time
from collections import defaultdict

import numpy as np

from map_matching_solution import load_road_network, load_trajectory
from map_matching_fast import hmm_map_match_fast as hmm_map_match
from run_stage2_full import (
    compute_segment_speeds, _build_edge_rtype,
    GRAPHML_PATH, TRAJ_DIR, MAX_POINTS_PER_FILE,
    DEFAULT_SIGMA_Z, DEFAULT_BETA,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=20,
                        help="How many trajectory files to sample (default 20)")
    parser.add_argument("--out", type=str, default="baseline_metrics.txt")
    args = parser.parse_args()

    print(f"Loading road network from {GRAPHML_PATH}...")
    G, edges_gdf, kdtree, edge_index = load_road_network(GRAPHML_PATH)
    total_edges = len(edges_gdf)
    edge_rtype = _build_edge_rtype(edges_gdf)
    print(f"  {total_edges} edges in network")

    all_files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.txt")))
    if not all_files:
        raise SystemExit(f"No .txt files in {TRAJ_DIR}")
    files = all_files[: args.files]
    print(f"Running matcher on {len(files)} files...")

    per_file_seconds = []
    all_segment_speeds = defaultdict(list)
    n_ok = n_fail = 0

    for i, fpath in enumerate(files):
        t0 = time.time()
        try:
            traj = load_trajectory(fpath)
            if len(traj) > MAX_POINTS_PER_FILE:
                traj = traj[:MAX_POINTS_PER_FILE]
            if len(traj) < 4:
                n_fail += 1
                continue
            matched = hmm_map_match(
                traj, G, edges_gdf, kdtree, edge_index,
                sigma_z=DEFAULT_SIGMA_Z, beta=DEFAULT_BETA,
            )
            non_none = sum(1 for m in matched if m is not None)
            if non_none < 2:
                n_fail += 1
                continue
            seg_speeds = compute_segment_speeds(traj, matched, edge_rtype=edge_rtype)
            for k, v in seg_speeds.items():
                all_segment_speeds[k].extend(v)
            n_ok += 1
        except Exception as e:
            print(f"  [{i+1}] {os.path.basename(fpath)} — {e}")
            n_fail += 1
        per_file_seconds.append(time.time() - t0)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] ok={n_ok} fail={n_fail}")

    # Metrics
    n_observations = sum(len(v) for v in all_segment_speeds.values())
    covered_edges = {eid for (eid, _tb) in all_segment_speeds.keys()}
    n_covered = len(covered_edges)
    coverage_pct = 100.0 * n_covered / total_edges if total_edges else 0.0
    per_edge_counts = [len(v) for v in all_segment_speeds.values()]
    median_per_edge = float(np.median(per_edge_counts)) if per_edge_counts else 0.0

    total_sec = sum(per_file_seconds)
    med_sec = float(np.median(per_file_seconds)) if per_file_seconds else 0.0
    p90_sec = float(np.percentile(per_file_seconds, 90)) if per_file_seconds else 0.0

    report = [
        "=" * 60,
        "BASELINE METRICS",
        "=" * 60,
        f"files processed        {len(files)} (ok={n_ok}, fail={n_fail})",
        f"wall-clock total       {total_sec:.1f}s",
        f"per-file median        {med_sec:.2f}s",
        f"per-file p90           {p90_sec:.2f}s",
        f"",
        f"edges in network       {total_edges:,}",
        f"edges covered          {n_covered:,}  ({coverage_pct:.2f}%)",
        f"(edge, tbin) keys      {len(all_segment_speeds):,}",
        f"total observations     {n_observations:,}",
        f"median obs / edge      {median_per_edge:.1f}",
        "",
        f"sigma_z                {DEFAULT_SIGMA_Z}",
        f"beta                   {DEFAULT_BETA}",
    ]
    text = "\n".join(report)
    print("\n" + text)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
