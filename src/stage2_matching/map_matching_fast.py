"""
map_matching_fast.py
====================

Drop-in faster HMM map matcher for Stage 1/2.

Keeps all optimizations of the previous version:
  - (u,v,k) -> geometry dict instead of DataFrame linear scans
  - Single-source Dijkstra with in-memory cache

Adds robustness fixes from the Phase 2/3 review:
  - Dedup threshold locked at 8m (was 2*sigma_z, which grew with tuning)
  - Adaptive candidate-search radius that scales with GPS time gap
  - Bearing term in the emission probability (prevents "wrong parallel
    street" errors in grid cities)
  - Path-break detection: if transition probabilities collapse between
    steps, the chain is split and each segment is Viterbi'd independently
  - Output is aligned with the *raw* input trajectory. Dropped points
    (dedup, no candidates, past a path break) are represented by None so
    downstream code can still use the original timestamps.
    Consumers must skip None entries before unpacking.

USAGE
-----
In run_stage2_full.py / process_test_cases.py:

    from map_matching_fast import hmm_map_match_fast as hmm_map_match

    matched = hmm_map_match(traj, G, edges_gdf, kdtree, edge_index)
    # len(matched) == len(traj); entries are None where the point wasn't matched
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


import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import geopandas as gpd
from scipy.spatial import KDTree

from map_matching_solution import (
    GPSPoint, Candidate, great_circle_distance,
    emission_probability, transition_probability,
    viterbi, _project_point_on_line,
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEDUP_THRESHOLD_M = 8.0
# Stationary-point filter in preprocessing. Locked at 8m regardless of
# sigma_z so tuning sigma_z upward doesn't silently delete more points.

MIN_SEARCH_RADIUS_M = 200.0
MAX_SEARCH_RADIUS_M = 350.0
URBAN_SPEED_SCALE_MS = 10.0
# Adaptive-radius formula: min(MAX, max(MIN, 2*sigma_z + 10*dt_seconds)).
# The original version used 35 m/s (freeway max) and no upper cap, which
# produced 3km search balls for 90-second gaps — hundreds of candidate
# edges per point, unusable runtime. 10 m/s (urban average) with a 350m
# ceiling gives ~4x the 200m baseline area only for gaps above ~20s.

BEARING_SIGMA_DEG = 45.0
# Gaussian std for the emission-bearing penalty. 45° means a 90° mismatch
# costs ~exp(-2) = ~0.14; a 180° mismatch (going the wrong way on a one-way)
# costs ~exp(-8) = ~3e-4.

PATH_BREAK_THRESHOLD = 1e-8
# If the max transition probability across all (i,j) pairs at some step t
# falls below this, the chain is split between t and t+1. The two sides
# are Viterbi'd independently; no transition is forced across the gap.


# ---------------------------------------------------------------------------
# Global caches — built lazily on first use
# ---------------------------------------------------------------------------
_edge_geom_cache: Dict = {}   # id(edges_gdf) -> {(u,v,k): geometry}
_node_coord_cache: Dict = {}  # id(G) -> {node: (lat, lon)}
_sssp_cache: Dict = {}        # (id(G), source) -> {target: distance}


def _build_edge_geom_cache(edges_gdf):
    cache_key = id(edges_gdf)
    if cache_key in _edge_geom_cache:
        return _edge_geom_cache[cache_key]
    d = {}
    for _, row in edges_gdf.iterrows():
        d[(row["u"], row["v"], row["key"])] = row["geometry"]
    _edge_geom_cache[cache_key] = d
    return d


def _build_node_coord_cache(G):
    cache_key = id(G)
    if cache_key in _node_coord_cache:
        return _node_coord_cache[cache_key]
    d = {n: (data["y"], data["x"]) for n, data in G.nodes(data=True)}
    _node_coord_cache[cache_key] = d
    return d


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _bearing_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


# ---------------------------------------------------------------------------
# Fast candidate finder
# ---------------------------------------------------------------------------
def get_candidates_fast(
    lat, lon, edges_gdf, kdtree, edge_index, radius=MIN_SEARCH_RADIUS_M,
):
    """Fast candidate finder. Output format identical to the original."""
    geom_by_edge = _build_edge_geom_cache(edges_gdf)

    radius_rad = radius / 6_371_000
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    idxs = kdtree.query_ball_point([lat_rad, lon_rad], radius_rad)

    if not idxs:
        return []

    seen = set()
    edge_ids = []
    for i in idxs:
        u, v, k = int(edge_index[i][0]), int(edge_index[i][1]), int(edge_index[i][2])
        eid = (u, v, k)
        if eid not in seen:
            seen.add(eid)
            edge_ids.append(eid)

    candidates = []
    for eid in edge_ids:
        geom = geom_by_edge.get(eid)
        if geom is None:
            continue
        proj_lat, proj_lon, dist = _project_point_on_line(lat, lon, geom)
        if dist <= radius:
            candidates.append(Candidate(eid, (proj_lat, proj_lon), dist))

    candidates.sort(key=lambda c: c.distance)
    return candidates


# ---------------------------------------------------------------------------
# Cached single-source Dijkstra
# ---------------------------------------------------------------------------
def _sssp_distance(G, source):
    """Return {target: shortest_path_length} from `source`, cached."""
    key = (id(G), source)
    cached = _sssp_cache.get(key)
    if cached is not None:
        return cached

    try:
        dists = nx.single_source_dijkstra_path_length(G, source, weight="length")
    except nx.NodeNotFound:
        dists = {}

    if len(_sssp_cache) >= 2000:
        keys = list(_sssp_cache.keys())
        for k in keys[: len(keys) // 2]:
            del _sssp_cache[k]

    _sssp_cache[key] = dists
    return dists


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
MatchedEntry = Optional[Tuple[Tuple[int, int, int], Tuple[float, float]]]


def hmm_map_match_fast(
    trajectory: List[GPSPoint],
    G: nx.MultiDiGraph,
    edges_gdf: gpd.GeoDataFrame,
    kdtree: KDTree,
    edge_index: np.ndarray,
    sigma_z: float = 4.07,
    beta: float = 3.0,
    use_bearing: bool = True,
    detect_breaks: bool = True,
) -> List[MatchedEntry]:
    """
    HMM map matching with adaptive radius, bearing-aware emission, and
    path-break detection.

    Returns a list of length `len(trajectory)`. result[i] is either
    `(edge_id, (proj_lat, proj_lon))` or None if point i was dropped
    (dedup / no candidates / past a path break).
    """
    T_raw = len(trajectory)
    result: List[MatchedEntry] = [None] * T_raw
    if T_raw < 2:
        return result

    node_coords = _build_node_coord_cache(G) if use_bearing else None

    # 1. Preprocess with locked dedup threshold
    kept_raw_idx: List[int] = [0]
    filtered: List[GPSPoint] = [trajectory[0]]
    for i in range(1, T_raw):
        pt = trajectory[i]
        prev = filtered[-1]
        if great_circle_distance(prev.lat, prev.lon, pt.lat, pt.lon) > DEDUP_THRESHOLD_M:
            kept_raw_idx.append(i)
            filtered.append(pt)
    if len(filtered) < 2:
        return result

    # 2. Candidates with adaptive radius (larger radius for larger time gaps)
    candidates_per_step: List[List[Candidate]] = []
    for t, pt in enumerate(filtered):
        if t == 0:
            radius = MIN_SEARCH_RADIUS_M
        else:
            dt = max(1.0, float(abs(pt.timestamp - filtered[t - 1].timestamp)))
            radius = min(
                MAX_SEARCH_RADIUS_M,
                max(MIN_SEARCH_RADIUS_M,
                    2.0 * sigma_z + URBAN_SPEED_SCALE_MS * dt),
            )
        cands = get_candidates_fast(
            pt.lat, pt.lon, edges_gdf, kdtree, edge_index, radius=radius
        )
        candidates_per_step.append(cands)

    valid_local = [t for t, c in enumerate(candidates_per_step) if c]
    if len(valid_local) < 2:
        return result

    filt_traj = [filtered[t] for t in valid_local]
    filt_cands = [candidates_per_step[t] for t in valid_local]
    # Original-trajectory index for each surviving local step
    local_to_raw = [kept_raw_idx[valid_local[t]] for t in range(len(valid_local))]
    T = len(filt_cands)

    # 3. Emission probabilities (+ optional bearing term)
    emission: List[List[float]] = []
    for t in range(T):
        step = []
        gps_bearing = None
        if use_bearing and t > 0:
            prev_pt, curr_pt = filt_traj[t - 1], filt_traj[t]
            gc = great_circle_distance(
                prev_pt.lat, prev_pt.lon, curr_pt.lat, curr_pt.lon
            )
            # Skip bearing term when points are basically co-located: the
            # bearing is numerical noise there.
            if gc >= 5.0:
                gps_bearing = _bearing_deg(
                    prev_pt.lat, prev_pt.lon, curr_pt.lat, curr_pt.lon
                )

        for c in filt_cands[t]:
            p = emission_probability(c.distance, sigma_z)
            if gps_bearing is not None:
                u, v, _k = c.edge_id
                u_coord = node_coords.get(u)
                v_coord = node_coords.get(v)
                if u_coord is not None and v_coord is not None:
                    edge_br = _bearing_deg(
                        u_coord[0], u_coord[1], v_coord[0], v_coord[1]
                    )
                    delta = _bearing_delta(gps_bearing, edge_br)
                    p *= math.exp(
                        -(delta * delta) / (2.0 * BEARING_SIGMA_DEG * BEARING_SIGMA_DEG)
                    )
            step.append(p)
        emission.append(step)

    # 4. Transition probabilities, per-step max tracked for break detection
    break_after = [False] * (T - 1)
    transition: List[List[List[float]]] = []
    for t in range(T - 1):
        pt_curr = filt_traj[t]
        pt_next = filt_traj[t + 1]
        gc_dist = great_circle_distance(
            pt_curr.lat, pt_curr.lon, pt_next.lat, pt_next.lon
        )

        src_nodes = {c.edge_id[1] for c in filt_cands[t]}
        for s in src_nodes:
            _sssp_distance(G, s)

        trans_matrix = []
        step_max = 0.0
        for ci in filt_cands[t]:
            dists_from_i = _sssp_distance(G, ci.edge_id[1])
            row = []
            for cj in filt_cands[t + 1]:
                node_j = cj.edge_id[0]
                if ci.edge_id[1] == node_j:
                    route_dist = 0.0
                else:
                    route_dist = dists_from_i.get(node_j, float("inf"))
                d_t = abs(gc_dist - route_dist)
                p = transition_probability(d_t, beta)
                row.append(p)
                if p > step_max:
                    step_max = p
            trans_matrix.append(row)
        transition.append(trans_matrix)
        if detect_breaks and step_max < PATH_BREAK_THRESHOLD:
            break_after[t] = True

    # 5. Slice into unbroken segments and run Viterbi per segment
    segments: List[Tuple[int, int]] = []
    seg_start = 0
    for t in range(T - 1):
        if break_after[t]:
            segments.append((seg_start, t))
            seg_start = t + 1
    segments.append((seg_start, T - 1))

    for a, b in segments:
        if a > b:
            continue
        if a == b:
            # Singleton segment — pick highest-emission candidate.
            cands = filt_cands[a]
            probs = emission[a]
            best = max(range(len(cands)), key=lambda i: probs[i])
            c = cands[best]
            result[local_to_raw[a]] = (c.edge_id, c.proj_point)
            continue
        seg_cands = filt_cands[a : b + 1]
        seg_em = emission[a : b + 1]
        seg_tr = transition[a:b]  # length (b - a)
        matched_seg = viterbi(seg_cands, seg_em, seg_tr)
        for k, entry in enumerate(matched_seg):
            result[local_to_raw[a + k]] = entry

    return result


def compact_matched(matched: List[MatchedEntry]):
    """Strip None entries. Use when a downstream consumer (e.g. the folium
    visualizer) can't handle gaps."""
    return [m for m in matched if m is not None]


def iter_matched_pairs(trajectory, matched):
    """
    Yield (traj_i, traj_next, match_i, match_next) for each consecutive
    pair where BOTH matched entries are non-None. This is what you want
    for computing segment speeds / trip features after path breaks.
    """
    n = min(len(trajectory), len(matched))
    for i in range(n - 1):
        m_i, m_next = matched[i], matched[i + 1]
        if m_i is None or m_next is None:
            continue
        yield trajectory[i], trajectory[i + 1], m_i, m_next


def clear_caches():
    """Call between independent runs if memory is tight."""
    _sssp_cache.clear()
