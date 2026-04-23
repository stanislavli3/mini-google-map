"""
Stage 3: Taxi Travel Time Prediction
====================================

Given (source_lat, source_lon, source_time, dest_lat, dest_lon, vehicle_id),
predict travel time in minutes.

Strategy: Hybrid physics + CatBoost.
  1. Build a supervised training set by splitting full trajectories into
     (origin, dest, start_time, vehicle, true_duration) rows.
  2. For each query, compute a physics-based ETA by routing on the SF road
     network and summing length/speed over edges, using the Stage 2 speed
     dictionary (falling back to road-type defaults).
  3. Feed physics_eta + tabular features into CatBoost. Train on log(duration).
  4. Validate chronologically (last N days held out).
"""

from __future__ import annotations

import os
import glob
import math
import pickle
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold

# Your existing modules
from map_matching_solution import (
    GPSPoint,
    load_road_network,
    load_trajectory,
    great_circle_distance,
    hmm_map_match,
)
from weather_features import load_sf_weather, floor_to_hour_utc

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIME_INTERVAL = 30 * 60  # 30-min bins, matches Stage 2

# Fallback free-flow speeds (m/s). Identical to Stage 2 so the pipeline
# stays consistent.
ROAD_TYPE_DEFAULT_SPEED = {
    "motorway": 29.1, "motorway_link": 22.4,
    "trunk": 20.1, "trunk_link": 15.6,
    "primary": 15.6, "primary_link": 13.4,
    "secondary": 13.4, "secondary_link": 11.2,
    "tertiary": 11.2, "tertiary_link": 11.2,
    "residential": 11.2, "living_street": 6.7,
    "service": 6.7, "unclassified": 11.2,
}
DEFAULT_SPEED_MS = 11.2  # ~25 mph fallback


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def timestamp_to_time_bin(ts: int, interval: int = TIME_INTERVAL) -> Tuple[str, int]:
    """Same as Stage 2 — (date_str, bin_index)."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    date_str = dt.strftime("%Y-%m-%d")
    sec = dt.hour * 3600 + dt.minute * 60 + dt.second
    return (date_str, sec // interval)


def time_of_day_bin(ts: int, interval: int = TIME_INTERVAL) -> int:
    """Time-of-day bin only (ignores date) — useful for day-agnostic lookups."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return (dt.hour * 3600 + dt.minute * 60 + dt.second) // interval


def parse_test_time(time_str: str, reference_year: int = 2008) -> int:
    """
    Parse the test CSV's source_time (e.g. '6/8/08 21:35') into a unix timestamp.
    We assume UTC to stay consistent with the trajectory timestamps.
    """
    # The format in the screenshot is 'M/D/YY HH:MM'
    dt = datetime.strptime(time_str, "%m/%d/%y %H:%M")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


# ---------------------------------------------------------------------------
# 1. Build supervised training set from full trajectories
# ---------------------------------------------------------------------------
def identify_trip_segments(traj: List[GPSPoint]) -> List[Tuple[int, int]]:
    """
    A trajectory file often contains many trips back-to-back. The `flag`
    field toggles between 0 (empty) and 1 (occupied). A taxi "trip" is a
    continuous span of flag=1 points — that's when a passenger is in the car.

    Returns (start_idx, end_idx) index pairs (inclusive) for each trip.

    If the file has no flag=1 points at all (some datasets don't use the
    flag), we fall back to treating the whole file as one trip split by
    long time gaps.
    """
    trips: List[Tuple[int, int]] = []

    if any(p.flag == 1 for p in traj):
        # Flag-based splitting
        start = None
        for i, p in enumerate(traj):
            if p.flag == 1 and start is None:
                start = i
            elif p.flag == 0 and start is not None:
                if i - 1 - start >= 3:   # need at least 4 points
                    trips.append((start, i - 1))
                start = None
        if start is not None and len(traj) - 1 - start >= 3:
            trips.append((start, len(traj) - 1))
    else:
        # Gap-based splitting: a gap > 5 minutes = new trip
        GAP_THRESHOLD = 5 * 60
        start = 0
        for i in range(1, len(traj)):
            if traj[i].timestamp - traj[i - 1].timestamp > GAP_THRESHOLD:
                if i - 1 - start >= 3:
                    trips.append((start, i - 1))
                start = i
        if len(traj) - 1 - start >= 3:
            trips.append((start, len(traj) - 1))

    return trips


def build_training_rows_from_trajectories(
    traj_dir: str,
    max_files: Optional[int] = None,
    prefix_samples_per_trip: int = 5,
    min_trip_duration_s: int = 60,
    max_trip_duration_s: int = 2 * 3600,
) -> pd.DataFrame:
    """
    Walk every trajectory file, split into trips, and emit training rows.

    For each trip, we emit (prefix_samples_per_trip + 1) rows:
      - The full trip (source=first point, dest=last point)
      - `prefix_samples_per_trip` random partial trips whose "dest" is an
        intermediate GPS point. This augmentation is directly from the
        Porto paper (Section 2.1) — it forces the model to handle queries
        of every length, not just full trips.

    Each row is labeled with `duration_min` = (dest_ts - source_ts) / 60.
    """
    files = sorted(glob.glob(os.path.join(traj_dir, "*.txt")))
    if max_files:
        files = files[:max_files]

    rng = np.random.default_rng(42)
    rows = []

    for i, fpath in enumerate(files):
        vehicle_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            traj = load_trajectory(fpath)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {vehicle_id}: failed to load ({e})")
            continue

        if len(traj) < 4:
            continue

        segments = identify_trip_segments(traj)
        if not segments:
            continue

        for (s, e) in segments:
            trip = traj[s:e + 1]
            duration = trip[-1].timestamp - trip[0].timestamp
            if duration < min_trip_duration_s or duration > max_trip_duration_s:
                continue

            # Full-trip row
            rows.append({
                "vehicle_id": vehicle_id,
                "source_lat": trip[0].lat, "source_lon": trip[0].lon,
                "source_time": trip[0].timestamp,
                "dest_lat": trip[-1].lat, "dest_lon": trip[-1].lon,
                "duration_min": duration / 60.0,
            })

            # Partial-trip augmentation: pick random interior endpoints
            if len(trip) > 3 and prefix_samples_per_trip > 0:
                k = min(prefix_samples_per_trip, len(trip) - 2)
                # Pick random split points, not too close to the start
                candidates = list(range(2, len(trip) - 1))
                picks = rng.choice(
                    candidates, size=min(k, len(candidates)), replace=False
                )
                for p in picks:
                    sub_dur = trip[p].timestamp - trip[0].timestamp
                    if sub_dur < min_trip_duration_s:
                        continue
                    rows.append({
                        "vehicle_id": vehicle_id,
                        "source_lat": trip[0].lat, "source_lon": trip[0].lon,
                        "source_time": trip[0].timestamp,
                        "dest_lat": trip[p].lat, "dest_lon": trip[p].lon,
                        "duration_min": sub_dur / 60.0,
                    })

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(files)} files, {len(rows)} rows so far")

    df = pd.DataFrame(rows)
    print(f"\nBuilt {len(df)} training rows from {len(files)} files.")
    return df


# ---------------------------------------------------------------------------
# 2. Physics-based ETA
# ---------------------------------------------------------------------------
class PhysicsETA:
    """
    Routes the shortest path on the road network and sums estimated
    travel times over edges.

    `speed_lookup(edge_id, time_bin) -> speed_ms` is the interface to your
    Stage 2 output. If the lookup returns None, we fall back to the road
    type default.
    """

    def __init__(
        self,
        G: nx.MultiDiGraph,
        edges_gdf: gpd.GeoDataFrame,
        speed_lookup,
    ):
        self.G = G
        self.edges_gdf = edges_gdf
        self.speed_lookup = speed_lookup

        # Precompute: node_id -> (lat, lon) for fast nearest-node lookup
        self.node_coords = np.array([
            [G.nodes[n]["y"], G.nodes[n]["x"]] for n in G.nodes
        ])
        self.node_list = list(G.nodes)
        self._node_tree = cKDTree(self.node_coords)

        # Edge attribute caches
        self._edge_length: Dict[Tuple, float] = {}
        self._edge_rtype: Dict[Tuple, str] = {}
        for _, row in edges_gdf.iterrows():
            et = (row["u"], row["v"], row["key"])
            self._edge_length[et] = float(row.get("length", 0.0))
            rt = row.get("highway", "unclassified")
            if isinstance(rt, list):
                rt = rt[0]
            self._edge_rtype[et] = rt if rt else "unclassified"

        self._edge_weights_cache: Dict[Tuple, Dict] = {}
        # Build node→index mapping for scipy sparse Dijkstra
        self._node_to_idx: Dict[int, int] = {n: i for i, n in enumerate(self.node_list)}

    def _get_edge_weights(self, tbin) -> Dict[Tuple, float]:
        """Precompute and cache {(u,v,key): travel_time_s} for each time bin."""
        if tbin not in self._edge_weights_cache:
            weights: Dict[Tuple, float] = {}
            for et, length in self._edge_length.items():
                if length <= 0:
                    weights[et] = 0.0
                    continue
                speed = self.speed_lookup(et, tbin)
                if speed is None or speed <= 0.1:
                    rt = self._edge_rtype.get(et, "unclassified")
                    speed = ROAD_TYPE_DEFAULT_SPEED.get(rt, DEFAULT_SPEED_MS)
                weights[et] = length / speed
            self._edge_weights_cache[tbin] = weights
        return self._edge_weights_cache[tbin]

    def nearest_node(self, lat: float, lon: float) -> int:
        _, idx = self._node_tree.query([lat, lon], k=1)
        return self.node_list[idx]

    def edge_travel_time_s(self, edge_id: Tuple, time_bin) -> float:
        """Seconds to traverse this edge at the given time bin."""
        length = self._edge_length.get(edge_id, 0.0)
        if length <= 0:
            return 0.0
        speed = self.speed_lookup(edge_id, time_bin)
        if speed is None or speed <= 0.1:
            rt = self._edge_rtype.get(edge_id, "unclassified")
            speed = ROAD_TYPE_DEFAULT_SPEED.get(rt, DEFAULT_SPEED_MS)
        return length / speed

    def predict_minutes(
        self,
        source_lat: float, source_lon: float,
        dest_lat: float, dest_lon: float,
        start_ts: int,
    ) -> Tuple[float, float, int]:
        """
        Returns (eta_minutes, route_length_m, n_edges_in_route).
        Falls back to haversine/default-speed if no path exists.
        """
        src = self.nearest_node(source_lat, source_lon)
        dst = self.nearest_node(dest_lat, dest_lon)

        if src == dst:
            # Same node — just return a small travel time proportional to distance
            d = great_circle_distance(source_lat, source_lon,
                                      dest_lat, dest_lon)
            return (d / DEFAULT_SPEED_MS) / 60.0, d, 0

        # Weight edges by length / current-speed. We use a snapshot time bin
        # (we don't re-bin as we progress along the route — a simplification).
        tbin = timestamp_to_time_bin(start_ts)

        # We call networkx's shortest path with a weight function. We can't
        # precompute weights easily because they depend on tbin.
        def weight(u, v, data):
            # data is the multi-edge attribute dict; pick min-length key
            key = min(data.keys(), key=lambda k: data[k].get("length", 1e9))
            edge_id = (u, v, key)
            return self.edge_travel_time_s(edge_id, tbin)

        try:
            path = nx.shortest_path(self.G, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            d = great_circle_distance(source_lat, source_lon,
                                      dest_lat, dest_lon)
            return (d / DEFAULT_SPEED_MS) / 60.0, d, 0

        total_s = 0.0
        total_len = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edges_uv = self.G.get_edge_data(u, v)
            if not edges_uv:
                continue
            # Pick the shortest parallel edge
            key = min(edges_uv.keys(),
                      key=lambda k: edges_uv[k].get("length", 1e9))
            edge_id = (u, v, key)
            total_s += self.edge_travel_time_s(edge_id, tbin)
            total_len += self._edge_length.get(edge_id, 0.0)

        return total_s / 60.0, total_len, len(path) - 1


def make_speed_lookup_from_stage2(
    complete_speeds: Dict[Tuple, float],
    fallback_to_time_of_day: bool = True,
):
    """
    Wraps Stage 2 `complete_speeds` dict into a callable.
    `complete_speeds` is keyed by (edge_id, (date_str, bin_index)).

    If a query's exact (date, bin) isn't present, we optionally fall back
    to the median speed across all observed days for that same time-of-day
    bin — that's usually what you want for test queries whose date isn't
    in the training range.
    """
    # Precompute day-agnostic fallback: edge -> {bin_index -> [speeds]}
    tod_bucket: Dict[Tuple, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (edge_id, (date_str, bin_idx)), speed in complete_speeds.items():
        tod_bucket[edge_id][bin_idx].append(speed)

    tod_median: Dict[Tuple[Tuple, int], float] = {}
    for edge_id, bins in tod_bucket.items():
        for bin_idx, speeds in bins.items():
            tod_median[(edge_id, bin_idx)] = float(np.median(speeds))

    def lookup(edge_id, time_bin):
        if (edge_id, time_bin) in complete_speeds:
            return complete_speeds[(edge_id, time_bin)]
        if fallback_to_time_of_day:
            bin_idx = time_bin[1]
            return tod_median.get((edge_id, bin_idx))
        return None

    return lookup


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    return great_circle_distance(lat1, lon1, lat2, lon2) / 1000.0


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure geometric + temporal features — no network, no physics yet."""
    df = df.copy()

    # Ensure source_time is an int timestamp
    if df["source_time"].dtype == object:
        df["source_time"] = df["source_time"].apply(
            lambda x: parse_test_time(x) if isinstance(x, str) else int(x)
        )

    df["haversine_km"] = df.apply(
        lambda r: haversine_km(r.source_lat, r.source_lon,
                               r.dest_lat, r.dest_lon), axis=1
    )
    df["bearing"] = df.apply(
        lambda r: bearing_deg(r.source_lat, r.source_lon,
                              r.dest_lat, r.dest_lon), axis=1
    )
    df["manhattan_km"] = (
        abs(df["source_lat"] - df["dest_lat"]) * 111.0
        + abs(df["source_lon"] - df["dest_lon"])
          * 111.0 * np.cos(np.radians(df["source_lat"]))
    )

    # Time features
    dts = df["source_time"].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc)
    )
    df["hour"] = dts.apply(lambda d: d.hour)
    df["minute_of_day"] = dts.apply(lambda d: d.hour * 60 + d.minute)
    df["day_of_week"] = dts.apply(lambda d: d.weekday())
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

    return df


def add_cluster_features(
    df: pd.DataFrame,
    kmeans_src=None,
    kmeans_dst=None,
    n_clusters: int = 100,
) -> Tuple[pd.DataFrame, object, object]:
    """KMeans on lat/lon → categorical cluster IDs."""
    from sklearn.cluster import MiniBatchKMeans

    df = df.copy()

    if kmeans_src is None:
        kmeans_src = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, n_init="auto"
        )
        kmeans_src.fit(df[["source_lat", "source_lon"]].values)
    if kmeans_dst is None:
        kmeans_dst = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=43, n_init="auto"
        )
        kmeans_dst.fit(df[["dest_lat", "dest_lon"]].values)

    df["src_cluster"] = kmeans_src.predict(
        df[["source_lat", "source_lon"]].values
    )
    df["dst_cluster"] = kmeans_dst.predict(
        df[["dest_lat", "dest_lon"]].values
    )
    df["od_pair"] = df["src_cluster"].astype(str) + "_" + df["dst_cluster"].astype(str)
    return df, kmeans_src, kmeans_dst


ROAD_CLASS_BUCKETS = {
    "motorway": "motorway", "motorway_link": "motorway",
    "trunk": "trunk", "trunk_link": "trunk",
    "primary": "primary", "primary_link": "primary",
    "secondary": "secondary", "secondary_link": "secondary",
    "tertiary": "other", "tertiary_link": "other",
    "residential": "residential", "living_street": "residential",
    "service": "other", "unclassified": "other",
}


def add_physics_features(df: pd.DataFrame, physics: PhysicsETA) -> pd.DataFrame:
    """
    Route every row using batched scipy Dijkstra (C-compiled, ~100x faster than NX).
    We capture predecessors so we can reconstruct the shortest path per row and
    derive:
      - route_n_edges (no longer stubbed to 0)
      - route_n_turns (bearing-change count)
      - route_pct_{motorway,trunk,primary,secondary,residential,other}
      - a rolling-time-bin ETA that re-bins as we walk the path (for trips
        that cross 30-min boundaries). The original start-time-snapshot ETA
        is retained as physics_eta_snapshot_min so CatBoost can pick either.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    df = df.copy()
    n = len(physics.node_list)
    node_to_idx = physics._node_to_idx

    src_idxs = physics._node_tree.query(df[["source_lat", "source_lon"]].values)[1]
    dst_idxs = physics._node_tree.query(df[["dest_lat", "dest_lon"]].values)[1]

    tbins = df["source_time"].apply(lambda ts: timestamp_to_time_bin(int(ts)))
    start_ts_arr = df["source_time"].astype("int64").values

    etas_snapshot = np.zeros(len(df))
    etas_rolling = np.zeros(len(df))
    lens = np.zeros(len(df))
    n_edges_arr = np.zeros(len(df), dtype=np.int32)
    n_turns_arr = np.zeros(len(df), dtype=np.int32)
    pct_motorway = np.zeros(len(df))
    pct_trunk = np.zeros(len(df))
    pct_primary = np.zeros(len(df))
    pct_secondary = np.zeros(len(df))
    pct_residential = np.zeros(len(df))
    pct_other = np.zeros(len(df))

    for tbin in tbins.unique():
        mask_positions = np.where((tbins == tbin).values)[0]

        weights = physics._get_edge_weights(tbin)
        rows_g, cols_g, t_data, d_data = [], [], [], []
        for (u, v, key), tt in weights.items():
            if tt <= 0:
                continue
            ui = node_to_idx.get(u)
            vi = node_to_idx.get(v)
            if ui is None or vi is None:
                continue
            rows_g.append(ui)
            cols_g.append(vi)
            t_data.append(tt)
            d_data.append(physics._edge_length.get((u, v, key), 0.0))

        T_mat = csr_matrix((t_data, (rows_g, cols_g)), shape=(n, n))
        D_mat = csr_matrix((d_data, (rows_g, cols_g)), shape=(n, n))

        unique_srcs = np.unique(src_idxs[mask_positions])
        t_dists, t_preds = sp_dijkstra(
            T_mat, indices=unique_srcs, directed=True, return_predecessors=True
        )
        d_dists = sp_dijkstra(D_mat, indices=unique_srcs, directed=True)
        src_row = {int(s): i for i, s in enumerate(unique_srcs)}

        for pos in mask_positions:
            si, di = int(src_idxs[pos]), int(dst_idxs[pos])
            r = src_row[si]
            snap_t = t_dists[r, di]
            total_d = d_dists[r, di]

            if np.isinf(snap_t):
                row = df.iloc[pos]
                d = great_circle_distance(
                    row.source_lat, row.source_lon,
                    row.dest_lat, row.dest_lon,
                )
                snap_t, total_d = d / DEFAULT_SPEED_MS, d
                etas_snapshot[pos] = snap_t / 60.0
                etas_rolling[pos] = snap_t / 60.0
                lens[pos] = total_d
                pct_other[pos] = 1.0
                continue

            # Reconstruct path: walk predecessors from dst back to src
            path_nodes = [di]
            cur = di
            guard = 0
            while cur != si and guard < 20000:
                prev = int(t_preds[r, cur])
                if prev < 0:
                    break
                path_nodes.append(prev)
                cur = prev
                guard += 1
            path_nodes.reverse()

            start_ts = int(start_ts_arr[pos])
            running_s = 0.0
            cur_tbin = tbin
            cur_weights = weights

            bucket_counts = {"motorway": 0, "trunk": 0, "primary": 0,
                             "secondary": 0, "residential": 0, "other": 0}
            n_turns = 0
            last_bearing = None
            n_edges = 0

            for i in range(len(path_nodes) - 1):
                u_idx, v_idx = path_nodes[i], path_nodes[i + 1]
                u_node = physics.node_list[u_idx]
                v_node = physics.node_list[v_idx]
                edges_uv = physics.G.get_edge_data(u_node, v_node)
                if not edges_uv:
                    continue
                key = min(edges_uv.keys(),
                          key=lambda k: edges_uv[k].get("length", 1e9))
                edge_id = (u_node, v_node, key)

                # Rolling bin: re-bucket when we cross a 30-min boundary
                new_tbin = timestamp_to_time_bin(start_ts + int(running_s))
                if new_tbin != cur_tbin:
                    cur_tbin = new_tbin
                    cur_weights = physics._get_edge_weights(cur_tbin)
                tt = cur_weights.get(edge_id)
                if tt is None or tt <= 0:
                    tt = physics.edge_travel_time_s(edge_id, cur_tbin)
                running_s += tt

                rt = physics._edge_rtype.get(edge_id, "unclassified")
                bucket_counts[ROAD_CLASS_BUCKETS.get(rt, "other")] += 1

                u_lat, u_lon = physics.node_coords[u_idx]
                v_lat, v_lon = physics.node_coords[v_idx]
                br = bearing_deg(u_lat, u_lon, v_lat, v_lon)
                if last_bearing is not None:
                    delta = abs(br - last_bearing)
                    if delta > 180:
                        delta = 360 - delta
                    if delta > 30:
                        n_turns += 1
                last_bearing = br
                n_edges += 1

            etas_snapshot[pos] = snap_t / 60.0
            etas_rolling[pos] = (running_s / 60.0) if n_edges > 0 else snap_t / 60.0
            lens[pos] = total_d
            n_edges_arr[pos] = n_edges
            n_turns_arr[pos] = n_turns
            if n_edges > 0:
                pct_motorway[pos] = bucket_counts["motorway"] / n_edges
                pct_trunk[pos] = bucket_counts["trunk"] / n_edges
                pct_primary[pos] = bucket_counts["primary"] / n_edges
                pct_secondary[pos] = bucket_counts["secondary"] / n_edges
                pct_residential[pos] = bucket_counts["residential"] / n_edges
                pct_other[pos] = bucket_counts["other"] / n_edges

    df["physics_eta_min"] = etas_rolling
    df["physics_eta_snapshot_min"] = etas_snapshot
    df["route_length_km"] = lens / 1000.0
    df["route_n_edges"] = n_edges_arr
    df["route_n_turns"] = n_turns_arr
    df["route_pct_motorway"] = pct_motorway
    df["route_pct_trunk"] = pct_trunk
    df["route_pct_primary"] = pct_primary
    df["route_pct_secondary"] = pct_secondary
    df["route_pct_residential"] = pct_residential
    df["route_pct_other"] = pct_other
    df["route_detour_ratio"] = df["route_length_km"] / df["haversine_km"].clip(lower=0.01)
    return df


def _oof_target_encode(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """
    Out-of-fold target encoding for hist_duration_* on the training set.
    Each row's encoded value is computed from rows in the other folds only,
    so the label never appears in its own feature.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    hist_min = np.full(n, np.nan)
    hist_od = np.full(n, np.nan)
    hist_hour = np.full(n, np.nan)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_pos, val_pos in kf.split(df):
        base = df.iloc[train_pos]
        fold = df.iloc[val_pos]

        agg = base.groupby(["src_cluster", "dst_cluster", "hour"])["duration_min"].median()
        agg2 = base.groupby(["src_cluster", "dst_cluster"])["duration_min"].median()
        agg3 = base.groupby(["hour"])["duration_min"].median()

        keys1 = list(zip(fold["src_cluster"], fold["dst_cluster"], fold["hour"]))
        keys2 = list(zip(fold["src_cluster"], fold["dst_cluster"]))
        keys3 = list(fold["hour"])

        hist_min[val_pos] = [agg.get(k, np.nan) for k in keys1]
        hist_od[val_pos] = [agg2.get(k, np.nan) for k in keys2]
        hist_hour[val_pos] = [agg3.get(k, np.nan) for k in keys3]

    global_median = float(df["duration_min"].median())
    hist_min = np.where(np.isnan(hist_min), hist_od, hist_min)
    hist_min = np.where(np.isnan(hist_min), hist_hour, hist_min)
    hist_min = np.where(np.isnan(hist_min), global_median, hist_min)
    hist_od = np.where(np.isnan(hist_od), hist_hour, hist_od)
    hist_od = np.where(np.isnan(hist_od), global_median, hist_od)
    hist_hour = np.where(np.isnan(hist_hour), global_median, hist_hour)

    df["hist_duration_min"] = hist_min
    df["hist_duration_od"] = hist_od
    df["hist_duration_hour"] = hist_hour
    return df


def add_historical_features(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Historical target-encoded duration features.

    - If `train_df` is None, we're encoding the training set itself, so we
      use out-of-fold means (KFold with n_folds). No row sees its own label.
    - If `train_df` is provided, we're encoding val/test, so we just look
      up medians from the full training set (clean — none of these rows
      were used to compute those medians).
    """
    if train_df is None:
        return _oof_target_encode(df, n_folds=n_folds)

    df = df.copy()
    base = train_df

    agg = (
        base.groupby(["src_cluster", "dst_cluster", "hour"])["duration_min"]
        .median()
        .rename("hist_duration_min")
        .reset_index()
    )
    df = df.merge(agg, on=["src_cluster", "dst_cluster", "hour"], how="left")

    agg2 = (
        base.groupby(["src_cluster", "dst_cluster"])["duration_min"]
        .median()
        .rename("hist_duration_od")
        .reset_index()
    )
    df = df.merge(agg2, on=["src_cluster", "dst_cluster"], how="left")

    agg3 = (
        base.groupby(["hour"])["duration_min"]
        .median()
        .rename("hist_duration_hour")
        .reset_index()
    )
    df = df.merge(agg3, on=["hour"], how="left")

    global_median = float(base["duration_min"].median())
    df["hist_duration_min"] = (
        df["hist_duration_min"]
        .fillna(df["hist_duration_od"])
        .fillna(df["hist_duration_hour"])
        .fillna(global_median)
    )
    df["hist_duration_od"] = (
        df["hist_duration_od"].fillna(df["hist_duration_hour"]).fillna(global_median)
    )
    df["hist_duration_hour"] = df["hist_duration_hour"].fillna(global_median)
    return df


VEHICLE_FEATURE_COLS = [
    "v_mean_speed_ms", "v_median_speed_ms", "v_p20_speed_ms", "v_p80_speed_ms",
    "v_total_km", "v_n_points", "v_hour_dominant", "v_hour_entropy",
    "v_pct_highway", "v_typical_trip_min", "v_n_trips",
]


def add_vehicle_features(
    df: pd.DataFrame,
    vehicle_features: Dict,
    fallback_medians: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge per-vehicle historical aggregates computed by
    process_test_cases.compute_vehicle_features.

    Keys in `vehicle_features` include the `.txt` suffix (filename-based);
    df["vehicle_id"] is the stem. We normalize both to the stem form before
    joining. Unseen vehicles get column-wise medians — passing a
    `fallback_medians` dict ensures train/val/test use the same fill values.
    Returns (df, medians_dict) so the caller can reuse medians across splits.
    """
    df = df.copy()
    normalized = {
        (k[:-4] if isinstance(k, str) and k.endswith(".txt") else k): v
        for k, v in vehicle_features.items()
    }
    if normalized:
        vf_df = pd.DataFrame.from_dict(normalized, orient="index").reset_index()
        vf_df = vf_df.rename(columns={"index": "_vid_stem"})
    else:
        vf_df = pd.DataFrame(columns=["_vid_stem"] + VEHICLE_FEATURE_COLS)

    for c in VEHICLE_FEATURE_COLS:
        if c not in vf_df.columns:
            vf_df[c] = np.nan

    df["_vid_stem"] = (
        df["vehicle_id"].astype(str).str.replace(".txt", "", regex=False)
    )
    df = df.merge(
        vf_df[["_vid_stem"] + VEHICLE_FEATURE_COLS], on="_vid_stem", how="left"
    )
    df = df.drop(columns=["_vid_stem"])

    medians = fallback_medians or {}
    for c in VEHICLE_FEATURE_COLS:
        if c in medians:
            fill_val = medians[c]
        else:
            series = vf_df[c].dropna()
            if len(series) == 0:
                fill_val = 12 if c == "v_hour_dominant" else 0.0
            else:
                fill_val = (
                    int(series.median()) if c == "v_hour_dominant"
                    else float(series.median())
                )
            medians[c] = fill_val
        df[c] = df[c].fillna(fill_val)

    return df, medians


def add_weather_features(
    df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join hourly SF weather onto each row by flooring source_time to the hour.
    `weather_df` is the output of weather_features.load_sf_weather.
    """
    df = df.copy()
    df["_hour_ts"] = df["source_time"].apply(
        lambda ts: floor_to_hour_utc(int(ts))
    )
    w = weather_df.rename(columns={"timestamp_hour": "_hour_ts"})
    df = df.merge(
        w[["_hour_ts", "temp_c", "precip_mm", "wind_kph", "condition"]],
        on="_hour_ts", how="left",
    )
    df = df.drop(columns=["_hour_ts"])

    for c in ("temp_c", "wind_kph", "precip_mm"):
        med = float(weather_df[c].median()) if not weather_df[c].isna().all() else 0.0
        df[c] = df[c].fillna(med)
    df["condition"] = df["condition"].fillna("0").astype(str)
    df["weather_is_rain"] = (df["precip_mm"] > 0.1).astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. CatBoost training
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "haversine_km", "manhattan_km", "bearing",
    "hour", "minute_of_day", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos",
    "src_cluster", "dst_cluster", "od_pair",
    "physics_eta_min", "physics_eta_snapshot_min",
    "route_length_km", "route_n_edges", "route_n_turns", "route_detour_ratio",
    "route_pct_motorway", "route_pct_trunk", "route_pct_primary",
    "route_pct_secondary", "route_pct_residential", "route_pct_other",
    "hist_duration_min", "hist_duration_od", "hist_duration_hour",
    "vehicle_id",
    "v_mean_speed_ms", "v_median_speed_ms", "v_p20_speed_ms", "v_p80_speed_ms",
    "v_total_km", "v_n_points", "v_hour_dominant", "v_hour_entropy",
    "v_pct_highway", "v_typical_trip_min", "v_n_trips",
    "temp_c", "precip_mm", "wind_kph", "weather_is_rain", "condition",
]
CAT_COLS = ["src_cluster", "dst_cluster", "od_pair", "vehicle_id",
            "hour", "day_of_week", "is_weekend",
            "v_hour_dominant", "condition"]


def train_catboost(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    use_log_target: bool = True,
):
    """
    Train a CatBoost regressor. `train_df` must contain FEATURE_COLS and
    a 'duration_min' column.
    """
    from catboost import CatBoostRegressor, Pool

    X_train = train_df[FEATURE_COLS].copy()
    y_train = np.log1p(train_df["duration_min"]) if use_log_target \
        else train_df["duration_min"]

    # Ensure categorical columns are strings (CatBoost is picky)
    for c in CAT_COLS:
        X_train[c] = X_train[c].astype(str)

    train_pool = Pool(X_train, y_train, cat_features=CAT_COLS)

    val_pool = None
    if val_df is not None and len(val_df) > 0:
        X_val = val_df[FEATURE_COLS].copy()
        y_val = np.log1p(val_df["duration_min"]) if use_log_target \
            else val_df["duration_min"]
        for c in CAT_COLS:
            X_val[c] = X_val[c].astype(str)
        val_pool = Pool(X_val, y_val, cat_features=CAT_COLS)

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        od_type="Iter",
        od_wait=300,
        verbose=200,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=val_pool is not None)
    return model


def apply_sanity_fallback(pred_min, df: pd.DataFrame) -> np.ndarray:
    """
    Protect against model blow-outs by enforcing distance-based bounds and
    blending wildly off predictions toward the physics ETA.

    - Hard floor: route_length at 40 m/s (~144 km/h, absolute ceiling speed)
    - Hard ceiling: route_length at 2.5 m/s (~9 km/h, gridlock crawl)
    - Soft blend toward physics_eta when |pred / physics| is outside [1/3, 3]

    The physics ETA is conservative — higher RMSE than the model on average
    — but never absurd, which makes it the right fallback for tail cases.
    """
    pred = np.asarray(pred_min, dtype=float).copy()
    physics = df["physics_eta_min"].values
    route_km = np.asarray(df["route_length_km"].values, dtype=float)
    haversine = np.asarray(df["haversine_km"].values, dtype=float)

    # Use max(route, haversine) so a broken physics route (route_km ≈ 0)
    # doesn't floor the prediction at 0 minutes.
    length_km = np.maximum(route_km, haversine).clip(min=0.05)
    t_floor = length_km * 1000.0 / 40.0 / 60.0
    t_ceil = length_km * 1000.0 / 2.5 / 60.0
    pred = np.clip(pred, t_floor, t_ceil)

    safe_physics = np.clip(physics, 0.5, 120.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pred / safe_physics
    extreme = (ratio < 1.0 / 3.0) | (ratio > 3.0) | ~np.isfinite(ratio)
    pred = np.where(extreme, 0.4 * pred + 0.6 * safe_physics, pred)

    return np.clip(pred, 0.5, 120.0)


def _train_stratified(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      min_subset_rows: int = 1000) -> Dict:
    """
    Train weekday vs. weekend models if each subset has enough rows;
    otherwise fall back to a single model. Weekday = Mon-Fri, weekend = Sat-Sun
    (driven off the `is_weekend` column already in the feature set).

    Returns a dict with key either 'single' or both 'weekday' and 'weekend'.
    """
    wk_mask = (train_df["is_weekend"] == 0).values
    we_mask = (train_df["is_weekend"] == 1).values
    n_wk, n_we = int(wk_mask.sum()), int(we_mask.sum())

    if n_wk < min_subset_rows or n_we < min_subset_rows:
        print(f"  Stratification skipped (weekday={n_wk}, weekend={n_we}, "
              f"min={min_subset_rows}); training one model.")
        return {"single": train_catboost(train_df, val_df)}

    wk_val = val_df[val_df["is_weekend"] == 0] if len(val_df) else val_df
    we_val = val_df[val_df["is_weekend"] == 1] if len(val_df) else val_df
    print(f"  Weekday model: train={n_wk}, val={len(wk_val)}")
    m_wk = train_catboost(train_df[wk_mask], wk_val if len(wk_val) > 0 else None)
    print(f"  Weekend model: train={n_we}, val={len(we_val)}")
    m_we = train_catboost(train_df[we_mask], we_val if len(we_val) > 0 else None)
    return {"weekday": m_wk, "weekend": m_we}


def _predict_stratified(models: Dict, df: pd.DataFrame) -> np.ndarray:
    """Route each row to the matching stratified model."""
    if "single" in models:
        return predict_catboost(models["single"], df)
    pred = np.zeros(len(df), dtype=float)
    is_we = (df["is_weekend"] == 1).values
    if (~is_we).any():
        pred[~is_we] = predict_catboost(models["weekday"], df[~is_we])
    if is_we.any():
        pred[is_we] = predict_catboost(models["weekend"], df[is_we])
    return pred


def predict_catboost(model, df: pd.DataFrame, use_log_target: bool = True):
    X = df[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    pred = model.predict(X)
    return np.expm1(pred) if use_log_target else pred


def quick_diagnostics(
    graphml_path: str = "sf_road_network.graphml",
    traj_dir: str = "Trajectories",
    stage2_speeds_path: str = "complete_speeds.pkl",
    vehicle_features_path: str = "vehicle_features.pkl",
    weather_cache_path: str = "sf_weather.pkl",
    test_csv: str = "kaggle-test-file-minute.csv",
    sample_train_files: int = 50,
):
    """Smoke-test the full feature pipeline on a small sample.

    Asserts:
      a) all expected feature columns are present and non-null after fill
      b) route_n_edges is not a constant (predecessor reconstruction works)
      c) weather-column hit rate >= 95% of sampled rows
    """

    print("=" * 60)
    print("QUICK DIAGNOSTICS (no training)")
    print("=" * 60)

    print("\n1. Loading road network...")
    G, edges_gdf, kdtree, edge_index = load_road_network(graphml_path)

    print("2. Loading Stage 2 speeds...")
    with open(stage2_speeds_path, "rb") as f:
        complete_speeds = pickle.load(f)
    speed_lookup = make_speed_lookup_from_stage2(complete_speeds)
    physics = PhysicsETA(G, edges_gdf, speed_lookup)

    vehicle_features: Dict = {}
    if os.path.exists(vehicle_features_path):
        with open(vehicle_features_path, "rb") as f:
            vehicle_features = pickle.load(f)
    print(f"   vehicle_features entries: {len(vehicle_features)}")

    try:
        weather_df = load_sf_weather(cache_path=weather_cache_path)
    except Exception as e:
        print(f"   WARNING: weather fetch failed ({e}); using empty frame")
        weather_df = pd.DataFrame(
            columns=["timestamp_hour", "temp_c", "precip_mm", "wind_kph", "condition"]
        )

    print(f"3. Building training sample ({sample_train_files} files)...")
    train_raw = build_training_rows_from_trajectories(
        traj_dir, max_files=sample_train_files, prefix_samples_per_trip=2
    )

    print("4. Feature engineering (train sample)...")
    train_df = add_basic_features(train_raw)
    train_df, km_src, km_dst = add_cluster_features(train_df, n_clusters=80)

    print("   Physics features (1000 sample rows)...")
    train_sample = train_df.sample(
        min(1000, len(train_df)), random_state=42
    ).reset_index(drop=True)
    train_sample = add_physics_features(train_sample, physics)
    train_sample = add_historical_features(train_sample)
    train_sample, v_medians = add_vehicle_features(train_sample, vehicle_features)
    train_sample = add_weather_features(train_sample, weather_df)

    # --- Assertions ---
    print("\nAssertions:")
    missing = [c for c in FEATURE_COLS if c not in train_sample.columns]
    assert not missing, f"Missing feature columns: {missing}"
    print(f"   (a) all {len(FEATURE_COLS)} feature columns present — OK")

    null_cols = [c for c in FEATURE_COLS if train_sample[c].isna().any()]
    assert not null_cols, f"Columns with NaNs after fill: {null_cols}"
    print("   (a) no NaNs in feature columns — OK")

    n_unique_edges = train_sample["route_n_edges"].nunique()
    assert n_unique_edges > 1, (
        f"route_n_edges is constant ({n_unique_edges} unique values) — "
        "predecessor reconstruction is broken"
    )
    print(f"   (b) route_n_edges has {n_unique_edges} unique values — OK")

    if len(weather_df) > 0:
        hit_rate = 1.0 - train_sample["temp_c"].isna().mean()
        # After fill there shouldn't be NaNs, so measure pre-fill by re-merging
        probe = train_sample[["source_time"]].copy()
        probe["_h"] = probe["source_time"].apply(lambda ts: floor_to_hour_utc(int(ts)))
        merged = probe.merge(
            weather_df[["timestamp_hour", "temp_c"]].rename(
                columns={"timestamp_hour": "_h"}
            ),
            on="_h", how="left",
        )
        hit_rate = 1.0 - merged["temp_c"].isna().mean()
        print(f"   (c) weather merge hit rate: {hit_rate:.1%}")
        assert hit_rate >= 0.95, f"Weather hit rate too low: {hit_rate:.1%}"

    print("\n5. Loading test set...")
    test_df = pd.read_csv(test_csv)
    test_df["source_time"] = test_df["source_time"].apply(parse_test_time)
    test_df["duration_min"] = 0.0

    print("6. Feature engineering (test)...")
    test_df = add_basic_features(test_df)
    test_df, _, _ = add_cluster_features(test_df, km_src, km_dst)

    test_sample = test_df.sample(
        min(1000, len(test_df)), random_state=42
    ).reset_index(drop=True)
    test_sample = add_physics_features(test_sample, physics)
    test_sample = add_historical_features(test_sample, train_df=train_sample)
    test_sample, _ = add_vehicle_features(
        test_sample, vehicle_features, fallback_medians=v_medians
    )
    test_sample = add_weather_features(test_sample, weather_df)
    print(f"   test sample columns OK: {len(test_sample.columns)}")


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main(
    graphml_path: str = "sf_road_network.graphml",
    traj_dir: str = "Trajectories",
    stage2_speeds_path: str = "complete_speeds.pkl",
    vehicle_features_path: str = "vehicle_features.pkl",
    weather_cache_path: str = "sf_weather.pkl",
    test_csv: str = "kaggle-test-file-minute.csv",
    submission_csv: str = "submission.csv",
    max_train_files: Optional[int] = None,
    val_days: int = 3,
):
    """End-to-end pipeline. Assumes Stage 1 and Stage 2 have already run
    and produced `complete_speeds.pkl` (pickle of the dict), and that
    process_test_cases.py has produced `vehicle_features.pkl`."""

    # --- Load network ---
    print("=" * 60)
    print("Loading road network...")
    G, edges_gdf, kdtree, edge_index = load_road_network(graphml_path)

    # --- Load Stage 2 speeds ---
    print("\nLoading Stage 2 speed dictionary...")
    with open(stage2_speeds_path, "rb") as f:
        complete_speeds = pickle.load(f)
    print(f"  {len(complete_speeds)} (edge, time_bin) entries")
    speed_lookup = make_speed_lookup_from_stage2(complete_speeds)

    # --- Load vehicle features (may be empty dict if process_test_cases hasn't run) ---
    vehicle_features: Dict = {}
    if os.path.exists(vehicle_features_path):
        with open(vehicle_features_path, "rb") as f:
            vehicle_features = pickle.load(f)
        print(f"  Loaded {len(vehicle_features)} vehicle feature entries")
    else:
        print(f"  WARNING: {vehicle_features_path} not found — v_* features will be medians only")

    # --- Load SF weather ---
    print("\nLoading SF weather...")
    try:
        weather_df = load_sf_weather(cache_path=weather_cache_path)
        print(f"  {len(weather_df)} hourly weather rows")
    except Exception as e:
        print(f"  WARNING: weather fetch failed ({e}); using empty frame")
        weather_df = pd.DataFrame(
            columns=["timestamp_hour", "temp_c", "precip_mm", "wind_kph", "condition"]
        )

    # --- Build physics engine ---
    physics = PhysicsETA(G, edges_gdf, speed_lookup)

    # --- Build training set ---
    print("\nBuilding training set from trajectories...")
    train_raw = build_training_rows_from_trajectories(
        traj_dir, max_files=max_train_files, prefix_samples_per_trip=5
    )

    # --- Chronological train/val split ---
    train_raw = train_raw.sort_values("source_time").reset_index(drop=True)
    cutoff_ts = train_raw["source_time"].max() - val_days * 86400
    train_df = train_raw[train_raw["source_time"] <= cutoff_ts].copy()
    val_df = train_raw[train_raw["source_time"] > cutoff_ts].copy()
    print(f"  Train: {len(train_df)}   Val: {len(val_df)}")

    # --- Feature engineering (train) ---
    print("\nFeature engineering (train)...")
    train_df = add_basic_features(train_df)
    train_df, km_src, km_dst = add_cluster_features(train_df, n_clusters=80)
    print("  Adding physics features (this is the slow step)...")
    train_df = add_physics_features(train_df, physics)
    train_df = add_historical_features(train_df)
    train_df, v_medians = add_vehicle_features(train_df, vehicle_features)
    train_df = add_weather_features(train_df, weather_df)

    # --- Feature engineering (val) ---
    print("\nFeature engineering (val)...")
    val_df = add_basic_features(val_df)
    val_df, _, _ = add_cluster_features(val_df, km_src, km_dst)
    val_df = add_physics_features(val_df, physics)
    val_df = add_historical_features(val_df, train_df=train_df)
    val_df, _ = add_vehicle_features(val_df, vehicle_features, fallback_medians=v_medians)
    val_df = add_weather_features(val_df, weather_df)

    # --- Train model (weekday/weekend stratified when data supports it) ---
    print("\nTraining CatBoost (stratified)...")
    models = _train_stratified(train_df, val_df)

    # --- Score val set ---
    val_pred = _predict_stratified(models, val_df)
    rmse = float(np.sqrt(np.mean((val_pred - val_df["duration_min"]) ** 2)))
    print(f"\nValidation RMSE (minutes): {rmse:.4f}")

    # Weekday/weekend RMSE split so we can see whether stratification helped
    val_is_we = (val_df["is_weekend"] == 1).values
    if val_is_we.any() and (~val_is_we).any():
        rmse_wk = float(np.sqrt(np.mean(
            (val_pred[~val_is_we] - val_df["duration_min"].values[~val_is_we]) ** 2
        )))
        rmse_we = float(np.sqrt(np.mean(
            (val_pred[val_is_we] - val_df["duration_min"].values[val_is_we]) ** 2
        )))
        print(f"  weekday RMSE: {rmse_wk:.4f}  (n={int((~val_is_we).sum())})")
        print(f"  weekend RMSE: {rmse_we:.4f}  (n={int(val_is_we.sum())})")

    # Physics-only baselines for comparison
    phys_rmse = float(np.sqrt(np.mean(
        (val_df["physics_eta_min"] - val_df["duration_min"]) ** 2
    )))
    phys_snap_rmse = float(np.sqrt(np.mean(
        (val_df["physics_eta_snapshot_min"] - val_df["duration_min"]) ** 2
    )))
    print(f"Physics-only (rolling) baseline RMSE: {phys_rmse:.4f}")
    print(f"Physics-only (snapshot) baseline RMSE: {phys_snap_rmse:.4f}")

    # --- Predict on test ---
    print("\nLoading test set...")
    test_df = pd.read_csv(test_csv)
    test_df["source_time"] = test_df["source_time"].apply(parse_test_time)
    test_df["duration_min"] = 0.0  # dummy for feature funcs

    test_df = add_basic_features(test_df)
    test_df, _, _ = add_cluster_features(test_df, km_src, km_dst)
    print("  Physics features for test set...")
    test_df = add_physics_features(test_df, physics)
    test_df = add_historical_features(test_df, train_df=train_df)
    test_df, _ = add_vehicle_features(test_df, vehicle_features, fallback_medians=v_medians)
    test_df = add_weather_features(test_df, weather_df)

    pred_raw = _predict_stratified(models, test_df)
    pred_min = apply_sanity_fallback(pred_raw, test_df)

    # Visibility: how many predictions were adjusted by the fallback
    n_adjusted = int(np.sum(np.abs(pred_raw - pred_min) > 0.5))
    print(f"  sanity fallback adjusted {n_adjusted} / {len(pred_min)} predictions "
          f"by more than 0.5 min")

    submission = pd.DataFrame({
        "id": test_df["id"],
        "duration_min": pred_min,
    })
    submission.to_csv(submission_csv, index=False)
    print(f"\nSubmission written to {submission_csv}  ({len(submission)} rows)")

    return models, train_df, val_df, test_df


if __name__ == "__main__":
    main()
