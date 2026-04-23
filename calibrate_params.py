"""
calibrate_params.py
===================

Phase 2.1 + 2.2 of the map-matching improvement plan: estimate sigma_z
and beta from real matched trajectories rather than using the N&K paper's
defaults (which were fit to 1-Hz European data, not 2008 SF cab pings).

Procedure:
  1. Run the current matcher on N sample files with the default parameters.
  2. Collect (candidate_distance, route_mismatch) statistics from every
     matched point / consecutive pair.
  3. sigma_z  = median(candidate_distance) / 0.6745   (MAD-robust estimate)
     beta     = mean(|gc_dist - route_dist|)
  4. Write both to matching_params.pkl.

run_stage2_full.py picks this file up automatically on the next run; pass
`--clear-match-cache` there so previously-cached matched sequences are
regenerated under the new parameters.

USAGE
-----
    python calibrate_params.py --files 50
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import time

import numpy as np
import networkx as nx

from map_matching_solution import load_road_network, load_trajectory, great_circle_distance
from map_matching_fast import (
    hmm_map_match_fast as hmm_map_match,
    _sssp_distance,
    iter_matched_pairs,
)
from run_stage2_full import (
    GRAPHML_PATH, TRAJ_DIR, MAX_POINTS_PER_FILE,
    DEFAULT_SIGMA_Z, DEFAULT_BETA, PARAMS_PATH,
)


def _route_dist(G, u, v):
    if u == v:
        return 0.0
    return _sssp_distance(G, u).get(v, float("inf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=50,
                        help="How many files to sample for calibration")
    parser.add_argument("--out", type=str, default=PARAMS_PATH)
    args = parser.parse_args()

    print(f"Loading road network from {GRAPHML_PATH}...")
    G, edges_gdf, kdtree, edge_index = load_road_network(GRAPHML_PATH)

    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.txt")))[: args.files]
    print(f"Running matcher on {len(files)} files with defaults "
          f"sigma_z={DEFAULT_SIGMA_Z}, beta={DEFAULT_BETA}...")

    cand_distances: list[float] = []  # GPS -> matched proj point, per matched step
    dt_values: list[float] = []       # |gc_dist - route_dist|, per matched pair

    t0 = time.time()
    for i, fpath in enumerate(files):
        try:
            traj = load_trajectory(fpath)
            if len(traj) > MAX_POINTS_PER_FILE:
                traj = traj[:MAX_POINTS_PER_FILE]
            if len(traj) < 4:
                continue
            matched = hmm_map_match(
                traj, G, edges_gdf, kdtree, edge_index,
                sigma_z=DEFAULT_SIGMA_Z, beta=DEFAULT_BETA,
            )
        except Exception as e:
            print(f"  [{i+1}] {os.path.basename(fpath)} — {e}")
            continue

        # Per-point candidate distances: GPS -> matched projection
        for pt, m in zip(traj, matched):
            if m is None:
                continue
            _eid, (plat, plon) = m
            cand_distances.append(great_circle_distance(pt.lat, pt.lon, plat, plon))

        # Per-pair d_t = |gc - route_dist|
        for pt_i, pt_next, m_i, m_next in iter_matched_pairs(traj, matched):
            gc = great_circle_distance(pt_i.lat, pt_i.lon, pt_next.lat, pt_next.lon)
            u_end = m_i[0][1]     # v-node of edge_i
            v_start = m_next[0][0]  # u-node of edge_next
            rd = _route_dist(G, u_end, v_start)
            if np.isinf(rd):
                continue
            dt_values.append(abs(gc - rd))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] "
                  f"samples so far: cand={len(cand_distances):,} "
                  f"pairs={len(dt_values):,}")

    elapsed = time.time() - t0
    if not cand_distances or not dt_values:
        raise SystemExit("No matched samples collected; cannot calibrate.")

    cand = np.asarray(cand_distances)
    dt_arr = np.asarray(dt_values)

    sigma_z = float(np.median(cand) / 0.6745)       # MAD -> Gaussian sigma
    beta = float(np.mean(dt_arr))

    # Clip to sane bands. If these clip heavily, the matcher is producing
    # garbage that no parameter tuning will fix — you probably have a
    # bearing / radius / break issue first.
    sigma_z_clipped = float(np.clip(sigma_z, 3.0, 40.0))
    beta_clipped = float(np.clip(beta, 1.0, 30.0))

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  candidate distances: n={len(cand):,}  "
          f"median={np.median(cand):.2f}m  p90={np.percentile(cand, 90):.2f}m")
    print(f"  pair d_t values:     n={len(dt_arr):,}  "
          f"mean={np.mean(dt_arr):.2f}m  median={np.median(dt_arr):.2f}m")
    print(f"  sigma_z  estimate: {sigma_z:.2f}m    (default {DEFAULT_SIGMA_Z})")
    print(f"  beta     estimate: {beta:.2f}m      (default {DEFAULT_BETA})")
    if sigma_z != sigma_z_clipped or beta != beta_clipped:
        print(f"  (clipped to: sigma_z={sigma_z_clipped:.2f}, "
              f"beta={beta_clipped:.2f})")
    print(f"  calibration elapsed: {elapsed:.1f}s")

    params = {
        "sigma_z": sigma_z_clipped,
        "beta": beta_clipped,
        "sigma_z_raw": sigma_z,
        "beta_raw": beta,
        "n_samples_candidates": len(cand),
        "n_samples_pairs": len(dt_arr),
        "files_used": len(files),
    }
    with open(args.out, "wb") as fh:
        pickle.dump(params, fh)
    print(f"\nWrote {args.out}. Next run of run_stage2_full.py will use these.")
    print("Remember to pass --clear-match-cache so per-file caches don't")
    print("reuse matches computed under the old parameters.")


if __name__ == "__main__":
    main()
