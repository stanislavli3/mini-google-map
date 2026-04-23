"""
map_matching_fast.py
====================

Drop-in replacement for the slow parts of map_matching_solution.py.

Two optimizations, each ~10-50x:

1. `get_candidates_fast` — uses a pre-built dict instead of DataFrame linear
   scans for every KDTree hit. The original scans 27k rows per candidate.

2. `hmm_map_match_fast` — uses single-source Dijkstra with caching instead
   of running pairwise Dijkstra thousands of times. Same algorithm,
   correct output, massively less redundant work.

USAGE
-----
In run_stage2_full.py, replace:

    from map_matching_solution import (
        GPSPoint, load_road_network, load_trajectory,
        great_circle_distance, hmm_map_match,
    )

with:

    from map_matching_solution import (
        GPSPoint, load_road_network, load_trajectory, great_circle_distance,
    )
    from map_matching_fast import hmm_map_match_fast as hmm_map_match

Nothing else changes. The API is identical.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import geopandas as gpd
from scipy.spatial import KDTree

# Import everything from the original so we can reuse helpers
from map_matching_solution import (
    GPSPoint, Candidate, great_circle_distance,
    emission_probability, transition_probability,
    preprocess_trajectory, viterbi, _project_point_on_line,
)


# ---------------------------------------------------------------------------
# Global caches — built lazily on first use
# ---------------------------------------------------------------------------
_edge_geom_cache: Dict = {}  # (u,v,k) -> geometry
_sssp_cache: Dict = {}       # (id(G), source_node) -> {target: distance}


def _build_edge_geom_cache(edges_gdf):
    """Build (u,v,k) -> geometry dict. One-time per edges_gdf."""
    cache_key = id(edges_gdf)
    if cache_key in _edge_geom_cache:
        return _edge_geom_cache[cache_key]

    d = {}
    for _, row in edges_gdf.iterrows():
        d[(row["u"], row["v"], row["key"])] = row["geometry"]
    _edge_geom_cache[cache_key] = d
    return d


def get_candidates_fast(
    lat, lon, edges_gdf, kdtree, edge_index, radius=200.0,
):
    """
    Fast candidate finder. Same output as the original get_candidates.
    Replaces the full-DataFrame linear scan with a dict lookup.
    """
    geom_by_edge = _build_edge_geom_cache(edges_gdf)

    # KDTree query
    radius_rad = radius / 6_371_000
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    idxs = kdtree.query_ball_point([lat_rad, lon_rad], radius_rad)

    if not idxs:
        return []

    # Dedup edges
    seen = set()
    edge_ids = []
    for i in idxs:
        u, v, k = int(edge_index[i][0]), int(edge_index[i][1]), int(edge_index[i][2])
        eid = (u, v, k)
        if eid not in seen:
            seen.add(eid)
            edge_ids.append(eid)

    # Project onto each edge geometry (O(1) lookup instead of O(n) scan!)
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
    """
    Return a dict {target_node: shortest_path_length} from `source`.
    Cached in-memory. First call per (G, source) is expensive, subsequent
    calls are O(1) lookups.
    """
    key = (id(G), source)
    cached = _sssp_cache.get(key)
    if cached is not None:
        return cached

    try:
        dists = nx.single_source_dijkstra_path_length(G, source, weight="length")
    except nx.NodeNotFound:
        dists = {}

    # Cap cache size: drop oldest half when too big
    if len(_sssp_cache) >= 2000:
        keys = list(_sssp_cache.keys())
        for k in keys[: len(keys) // 2]:
            del _sssp_cache[k]

    _sssp_cache[key] = dists
    return dists


def _route_dist(G, u, v):
    """Fast cached shortest-path distance from u to v."""
    if u == v:
        return 0.0
    dists = _sssp_distance(G, u)
    return dists.get(v, float("inf"))


# ---------------------------------------------------------------------------
# Fast HMM map matching
# ---------------------------------------------------------------------------
def hmm_map_match_fast(
    trajectory: List[GPSPoint],
    G: nx.MultiDiGraph,
    edges_gdf: gpd.GeoDataFrame,
    kdtree: KDTree,
    edge_index: np.ndarray,
    sigma_z: float = 4.07,
    beta: float = 3.0,
):
    """
    Same output as hmm_map_match but dramatically faster due to:
      - Cached single-source Dijkstra instead of pairwise calls
      - Dict-lookup candidate finder instead of DataFrame scans
    """
    # Preprocess
    trajectory = preprocess_trajectory(trajectory, sigma_z)
    print(f"  After preprocessing: {len(trajectory)} GPS points")

    if len(trajectory) < 2:
        print("  Too few points after preprocessing.")
        return []

    # Find candidates
    print("  Finding candidate road segments...")
    candidates_per_step = []
    valid_steps = []

    for t, pt in enumerate(trajectory):
        cands = get_candidates_fast(
            pt.lat, pt.lon, edges_gdf, kdtree, edge_index, radius=200.0,
        )
        if cands:
            candidates_per_step.append(cands)
            valid_steps.append(t)
        else:
            candidates_per_step.append([])

    filtered_trajectory = [trajectory[t] for t in valid_steps]
    filtered_candidates = [candidates_per_step[t] for t in valid_steps]

    if len(filtered_candidates) < 2:
        print("  Too few points with candidates.")
        return []

    T = len(filtered_candidates)
    print(f"  {T} steps with candidates")

    # Emission probabilities
    emission_probs = []
    for t in range(T):
        step_probs = [emission_probability(c.distance, sigma_z)
                      for c in filtered_candidates[t]]
        emission_probs.append(step_probs)

    # Transition probabilities — with SSSP caching
    print("  Computing transition probabilities (with SSSP cache)...")
    transition_probs = []

    for t in range(T - 1):
        pt_curr = filtered_trajectory[t]
        pt_next = filtered_trajectory[t + 1]

        gc_dist = great_circle_distance(
            pt_curr.lat, pt_curr.lon, pt_next.lat, pt_next.lon
        )

        # Pre-compute SSSP for each unique source node in step t
        # (This is where the magic happens: one Dijkstra per source, not per pair.)
        src_nodes = {c.edge_id[1] for c in filtered_candidates[t]}  # 'v' nodes
        # Warm the cache
        for src in src_nodes:
            _sssp_distance(G, src)

        trans_matrix = []
        for i, cand_i in enumerate(filtered_candidates[t]):
            row = []
            node_i = cand_i.edge_id[1]
            dists_from_i = _sssp_distance(G, node_i)

            for j, cand_j in enumerate(filtered_candidates[t + 1]):
                node_j = cand_j.edge_id[0]
                if node_i == node_j:
                    route_dist = 0.0
                else:
                    route_dist = dists_from_i.get(node_j, float("inf"))

                d_t = abs(gc_dist - route_dist)
                prob = transition_probability(d_t, beta)
                row.append(prob)
            trans_matrix.append(row)
        transition_probs.append(trans_matrix)

    # Viterbi
    print("  Running Viterbi algorithm...")
    matched = viterbi(filtered_candidates, emission_probs, transition_probs)

    print(f"  Matched {len(matched)} points to road segments.")
    return matched


def clear_caches():
    """Call this between independent runs if memory is tight."""
    global _sssp_cache
    _sssp_cache.clear()
