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

# Your existing modules
from map_matching_solution import (
    GPSPoint,
    load_road_network,
    load_trajectory,
    great_circle_distance,
    hmm_map_match,
)

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


def add_physics_features(df: pd.DataFrame, physics: PhysicsETA) -> pd.DataFrame:
    """Route every row using batched scipy Dijkstra (C-compiled, ~100x faster than NX)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    df = df.copy()
    n = len(physics.node_list)
    node_to_idx = physics._node_to_idx

    # Vectorized nearest-node lookup for all rows at once
    src_idxs = physics._node_tree.query(df[["source_lat", "source_lon"]].values)[1]
    dst_idxs = physics._node_tree.query(df[["dest_lat", "dest_lon"]].values)[1]

    tbins = df["source_time"].apply(lambda ts: timestamp_to_time_bin(int(ts)))

    etas = np.zeros(len(df))
    lens = np.zeros(len(df))

    for tbin in tbins.unique():
        mask_positions = np.where((tbins == tbin).values)[0]

        # Build sparse travel-time and distance matrices (cached per tbin)
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

        # One scipy Dijkstra per unique source node — covers all destinations for free
        unique_srcs = np.unique(src_idxs[mask_positions])
        t_dists = sp_dijkstra(T_mat, indices=unique_srcs, directed=True)
        d_dists = sp_dijkstra(D_mat, indices=unique_srcs, directed=True)
        src_row = {int(s): i for i, s in enumerate(unique_srcs)}

        for pos in mask_positions:
            si, di = int(src_idxs[pos]), int(dst_idxs[pos])
            r = src_row[si]
            total_t = t_dists[r, di]
            total_d = d_dists[r, di]
            if np.isinf(total_t):
                row = df.iloc[pos]
                d = great_circle_distance(
                    row.source_lat, row.source_lon,
                    row.dest_lat, row.dest_lon,
                )
                total_t, total_d = d / DEFAULT_SPEED_MS, d
            etas[pos] = total_t / 60.0
            lens[pos] = total_d

    df["physics_eta_min"] = etas
    df["route_length_km"] = lens / 1000.0
    # route_n_edges not reconstructed from scipy predecessors; set to 0 (constant, CatBoost ignores)
    df["route_n_edges"] = 0
    df["route_detour_ratio"] = df["route_length_km"] / df["haversine_km"].clip(lower=0.01)
    return df


def add_historical_features(
    df: pd.DataFrame, train_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    For each (src_cluster, dst_cluster, hour) bucket in `train_df`, store
    the median duration. Then look up that value as a feature. If
    train_df is None, compute from df itself (for training) but use
    leave-one-out to avoid leakage.

    For simplicity here, we compute on all of train_df. If you want to be
    strict, swap for a k-fold target encoding.
    """
    df = df.copy()
    base = train_df if train_df is not None else df

    agg = (
        base.groupby(["src_cluster", "dst_cluster", "hour"])["duration_min"]
        .median()
        .rename("hist_duration_min")
        .reset_index()
    )
    df = df.merge(agg, on=["src_cluster", "dst_cluster", "hour"], how="left")

    # Broader fallbacks for unseen OD+hour combos
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

    df["hist_duration_min"] = (
        df["hist_duration_min"]
        .fillna(df["hist_duration_od"])
        .fillna(df["hist_duration_hour"])
        .fillna(base["duration_min"].median())
    )
    return df


# ---------------------------------------------------------------------------
# 4. CatBoost training
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "haversine_km", "manhattan_km", "bearing",
    "hour", "minute_of_day", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos",
    "src_cluster", "dst_cluster", "od_pair",
    "physics_eta_min", "route_length_km", "route_n_edges", "route_detour_ratio",
    "hist_duration_min", "hist_duration_od", "hist_duration_hour",
    "vehicle_id",
]
CAT_COLS = ["src_cluster", "dst_cluster", "od_pair", "vehicle_id",
            "hour", "day_of_week", "is_weekend"]


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
    test_csv: str = "kaggle-test-file-minute.csv",
    sample_train_files: int = 50,  # Just sample to save time
):
    """Run diagnostics without training - takes ~5 minutes instead of 30"""
    
    print("="*60)
    print("QUICK DIAGNOSTICS (no training)")
    print("="*60)
    
    # --- Load network ---
    print("\n1. Loading road network...")
    G, edges_gdf, kdtree, edge_index = load_road_network(graphml_path)
    
    # --- Load Stage 2 speeds ---
    print("2. Loading Stage 2 speeds...")
    with open(stage2_speeds_path, "rb") as f:
        complete_speeds = pickle.load(f)
    speed_lookup = make_speed_lookup_from_stage2(complete_speeds)
    physics = PhysicsETA(G, edges_gdf, speed_lookup)
    
    # --- Build SMALL training sample ---
    print(f"3. Building training sample ({sample_train_files} files)...")
    train_raw = build_training_rows_from_trajectories(
        traj_dir, max_files=sample_train_files, prefix_samples_per_trip=2
    )
    
    # --- Feature engineering on train sample ---
    print("4. Feature engineering (train sample)...")
    train_df = add_basic_features(train_raw)
    train_df, km_src, km_dst = add_cluster_features(train_df, n_clusters=80)
    
    # Only do physics on a small subset to save time
    print("   Physics features (1000 sample rows)...")
    train_sample = train_df.sample(min(1000, len(train_df)), random_state=42)
    train_sample = add_physics_features(train_sample, physics)
    
    # --- Load and process test set ---
    print("5. Loading test set...")
    test_df = pd.read_csv(test_csv)
    test_df["source_time"] = test_df["source_time"].apply(parse_test_time)
    test_df["duration_min"] = 0.0  # Dummy
    
    print("6. Feature engineering (test)...")
    test_df = add_basic_features(test_df)
    test_df, _, _ = add_cluster_features(test_df, km_src, km_dst)
    
    # Physics on sample of test
    print("   Physics features (1000 test rows)...")
    test_sample = test_df.sample(min(1000, len(test_df)), random_state=42)
    test_sample = add_physics_features(test_sample, physics)


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------
def main(
    graphml_path: str = "sf_road_network.graphml",
    traj_dir: str = "Trajectories",
    stage2_speeds_path: str = "complete_speeds.pkl",
    test_csv: str = "kaggle-test-file-minute.csv",
    submission_csv: str = "submission.csv",
    max_train_files: Optional[int] = None,
    val_days: int = 3,
):
    """End-to-end pipeline. Assumes Stage 1 and Stage 2 have already run
    and produced `complete_speeds.pkl` (pickle of the dict)."""

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

    # --- Feature engineering ---
    print("\nFeature engineering (train)...")
    train_df = add_basic_features(train_df)
    train_df, km_src, km_dst = add_cluster_features(train_df, n_clusters=80)
    print("  Adding physics features (this is the slow step)...")
    train_df = add_physics_features(train_df, physics)
    train_df = add_historical_features(train_df)

    print("\nFeature engineering (val)...")
    val_df = add_basic_features(val_df)
    val_df, _, _ = add_cluster_features(val_df, km_src, km_dst)
    val_df = add_physics_features(val_df, physics)
    val_df = add_historical_features(val_df, train_df=train_df)

    # --- Train model ---
    print("\nTraining CatBoost...")
    model = train_catboost(train_df, val_df)

    # --- Score val set ---
    val_pred = predict_catboost(model, val_df)
    rmse = float(np.sqrt(np.mean((val_pred - val_df["duration_min"]) ** 2)))
    print(f"\nValidation RMSE (minutes): {rmse:.4f}")

    # Also report physics-only baseline for comparison
    phys_rmse = float(np.sqrt(np.mean(
        (val_df["physics_eta_min"] - val_df["duration_min"]) ** 2
    )))
    print(f"Physics-only baseline RMSE: {phys_rmse:.4f}")

    # --- Predict on test ---
    print("\nLoading test set...")
    test_df = pd.read_csv(test_csv)
    test_df["source_time"] = test_df["source_time"].apply(parse_test_time)
    # Test set has no duration_min; we add a dummy so the feature funcs don't break
    test_df["duration_min"] = 0.0

    test_df = add_basic_features(test_df)
    test_df, _, _ = add_cluster_features(test_df, km_src, km_dst)
    print("  Physics features for test set...")
    test_df = add_physics_features(test_df, physics)
    test_df = add_historical_features(test_df, train_df=train_df)

    pred_min = predict_catboost(model, test_df)
    pred_min = np.clip(pred_min, 0.5, 120.0)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "duration_min": pred_min,
    })
    submission.to_csv(submission_csv, index=False)
    print(f"\nSubmission written to {submission_csv}  ({len(submission)} rows)")

    return model, train_df, val_df, test_df


if __name__ == "__main__":
    main()
