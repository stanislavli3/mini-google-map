# SF Taxi ETA — Stage 2/3 Upgrades

A three-stage pipeline for predicting taxi travel times in San Francisco from raw
2008 cab GPS trajectories. This README documents the **enhancements** layered on
top of the starting codebase: stronger feature engineering for the ETA model,
robustness fixes for the HMM map matcher, and end-to-end pipeline orchestration
that makes the whole thing reproducible from a single command.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Summary of additions](#summary-of-additions)
- [Stage 2 — map matching](#stage-2--map-matching)
  - [Robustness upgrades in `map_matching_fast.py`](#robustness-upgrades-in-map_matching_fastpy)
  - [Parameter calibration (`calibrate_params.py`)](#parameter-calibration-calibrate_paramspy)
  - [Speed aggregation improvements](#speed-aggregation-improvements)
  - [Runtime improvements](#runtime-improvements)
- [Stage 3 — ETA model](#stage-3--eta-model)
  - [Feature engineering](#feature-engineering)
  - [Out-of-fold target encoding](#out-of-fold-target-encoding)
  - [Weekday/weekend stratified models](#weekdayweekend-stratified-models)
  - [Outlier-safe predictions](#outlier-safe-predictions)
- [New helper modules](#new-helper-modules)
- [How to run](#how-to-run)
- [File reference](#file-reference)
- [Implementation notes](#implementation-notes)

---

## Pipeline overview

```
  Trajectories/*.txt ──► Stage 2 ──► complete_speeds.pkl ──► Stage 3 ──► submission.csv
  (cab GPS logs)         (map        (per-edge              (feature    (Kaggle
                         matching +   time-bucketed          matrix +    predictions)
                         speed        speeds)                CatBoost)
                         aggregation)
```

Stage 1 is the graph-construction step (`build_graphml.py` downloads the SF
drivable road network from OpenStreetMap into `sf_road_network.graphml`). Stage 2
map-matches every cab trajectory against that graph and produces a lookup table
of median speeds per `(edge, time_bucket)`. Stage 3 builds a feature matrix from
the Kaggle test CSV, trains CatBoost regressors, and writes the submission.

Everything below describes what was added or changed on top of the existing
code.

---

## Summary of additions

| Area | Changes |
|---|---|
| Feature matrix | 20 features → **44 features** across six groups |
| Map-matching HMM | Bearing-aware emission, path-break detection, adaptive candidate radius, locked dedup threshold |
| Parameter calibration | New script that estimates `σ_z` and `β` empirically from data (replaces hard-coded paper defaults) |
| Speed aggregation | Road-class-aware speed caps; passenger-flag confidence weighting |
| Model training | Weekday/weekend stratified CatBoost models (with data-sufficiency guard) |
| Outlier handling | Distance-based floor/ceiling + soft blend to physics ETA |
| Leakage | 5-fold out-of-fold target encoding for historical duration features |
| Runtime | Multiprocessing in Phase A (4+ workers); per-file matched-sequence disk cache |
| Orchestration | `run_pipeline.sh` + `build_graphml.py` + `baseline_metrics.py` scripts |
| External signal | SF hourly weather joined via `meteostat` |

**Aggregate code delta:** 12 files changed, ~1,500 net lines added across
feature engineering, robustness fixes, and orchestration.

---

## Stage 2 — map matching

### Robustness upgrades in `map_matching_fast.py`

The starting map matcher used the parameters from the Newson & Krumm (2009)
paper, which were fitted on 1-Hz European GPS data. For 2008 SF cab data with
60–120 second gaps between pings, those defaults are miscalibrated and the
matcher makes several classes of errors that silently corrupt downstream speeds.
Four additions close those gaps.

#### Locked preprocessing dedup threshold

The stationary-point filter in `preprocess_trajectory` used to be `2 * σ_z`.
Raising `σ_z` during calibration therefore silently deleted more points from
every trajectory. Dedup threshold is now pinned at **8 meters**, independent of
`σ_z`.

```python
# map_matching_fast.py
DEDUP_THRESHOLD_M = 8.0
```

#### Adaptive candidate search radius (with cap)

Fixed-200m candidate search fails on the long gaps in 2008 taxi data because
GPS noise combined with time-between-pings can put the correct road outside the
ball. The radius now scales with the gap:

```python
radius = min(MAX_SEARCH_RADIUS_M,
             max(MIN_SEARCH_RADIUS_M,
                 2.0 * sigma_z + URBAN_SPEED_SCALE_MS * dt))
# MIN = 200, MAX = 350, URBAN_SPEED_SCALE_MS = 10
```

The upper cap at 350m matters: without it, a 90-second gap would produce a 3km
search ball (hundreds of candidate edges per point, making the Viterbi lattice
unusably large).

#### Bearing-aware emission probability

San Francisco's grid means two parallel one-way streets are often both within
the search radius and equidistant from a GPS ping — the original matcher picks
arbitrarily, producing bad edge assignments on straight segments. The emission
probability now incorporates a bearing-match penalty:

```
emission_p *= exp(- Δbearing² / (2 · 45°²))
```

Where `Δbearing` is the angle between (a) the heading implied by the previous
GPS point → current GPS point and (b) the edge's u→v direction. A 180°
mismatch (wrong way on a one-way) gets penalized by ~exp(-8) ≈ 3·10⁻⁴, enough
for the Viterbi to pick the correctly-directed edge. The term is skipped when
consecutive GPS points are <5m apart (bearing is numerical noise there).

#### Path-break detection

If the maximum transition probability across all candidate pairs at some step
`t` falls below a threshold (`1e-8`), the trajectory is split at that point and
each unbroken segment runs Viterbi independently. This prevents the previous
behavior of forcing a transition across a genuine GPS drop-out, which produced
garbage matched sequences that then corrupted the speed aggregate.

```python
PATH_BREAK_THRESHOLD = 1e-8
# If step_max < threshold, break_after[t] = True.
# Segments are Viterbi'd independently; no edge speed is computed across the gap.
```

#### Output alignment with raw trajectory

The previous matched output was aligned with the post-preprocessing
trajectory, but downstream consumers (`compute_segment_speeds`,
`compute_vehicle_features`) indexed into the raw trajectory for timestamps —
a latent misalignment bug whenever preprocessing dropped any point. The new
contract:

```python
# matched[i] is aligned with trajectory[i].
# Either (edge_id, (proj_lat, proj_lon)) or None (dropped / past a break).
matched: List[Optional[Tuple[Tuple[int, int, int], Tuple[float, float]]]]
```

Three helpers support the contract:
- `iter_matched_pairs(trajectory, matched)` — yields `(pt_i, pt_next, match_i, match_next)` for each consecutive pair where both matched entries are non-None.
- `compact_matched(matched)` — strips `None` entries (for visualizers that can't handle gaps).
- `clear_caches()` — evicts the module-level SSSP cache between independent runs.

All downstream consumers (`run_stage2_full.compute_segment_speeds`,
`process_test_cases.compute_segment_speeds`, `process_test_cases.compute_vehicle_features`,
`map_matching_solution.visualize_matching`) were updated to skip `None` entries.

---

### Parameter calibration (`calibrate_params.py`)

A new standalone script that estimates the two HMM parameters from data rather
than using the paper's defaults:

```
σ_z  = median(GPS-to-projection-distance)   /   0.6745       (MAD-based)
β    = mean(|great_circle_distance  −  graph_route_distance|)
```

Usage:

```bash
python calibrate_params.py --files 50
```

Typical output for this dataset:

```
candidate distances: n=5,884  median=8.19m  p90=43.17m
pair d_t values:     n=5,282  mean=76.38m   median=24.64m

sigma_z  estimate: 12.15m    (default 4.07)
beta     estimate: 76.38m    clipped → 30.00m (default 3.0)
```

The σ_z estimate of 12m is ~3× the paper's 4m default, confirming 2008 SF cab
GPS is considerably noisier than 1-Hz European data.

Both values are clipped to a sanity band (`σ_z ∈ [3, 40]`, `β ∈ [1, 30]`) and
written to `matching_params.pkl`. `run_stage2_full.py` loads this file
automatically on its next run. After re-calibrating, pass
`--clear-match-cache` to `run_stage2_full.py` so the previously-cached matched
sequences (computed under the old parameters) are regenerated.

---

### Speed aggregation improvements

#### Road-class-aware speed caps

The original filter applied a single `MAX_REASONABLE_SPEED = 50 m/s` cap for
every observation. A 35 m/s "speed" on a residential edge is essentially
always a map-matching artefact but would sail past that cap. Per-class caps:

| Road class | Cap (m/s) |
|---|---:|
| motorway | 42 |
| motorway_link / trunk | 30 |
| primary / primary_link | 22–25 |
| secondary / tertiary | 20 |
| residential | 18 |
| living_street / service | 10 |

Implemented in `run_stage2_full._edge_speed_cap` with an `edge_rtype` dict
built once at Phase A startup via `_build_edge_rtype`.

#### Passenger-flag confidence weighting

Pairs of GPS points where both observations have occupied flag = 1 (passenger
in cab, driver going to a known destination) are weighted **2×** in the median
aggregate. Mixed-flag and empty-cab pairs keep their original weight of 1.
Weighting is implemented via simple duplication of the speed observation in
the list that `aggregate_speeds` medians:

```python
weight = 2 if (pt_i.flag == 1 and pt_next.flag == 1) else 1
edges_touched = [edge_i] if edge_i == edge_next else [edge_i, edge_next]
for e in edges_touched:
    segment_speeds[(e, time_bin)].extend([speed] * weight)
```

Dropping flag-0 pairs entirely was considered but rejected — it hurt coverage
more than it helped signal quality.

---

### Runtime improvements

#### Per-file matched-sequence cache

Every successful match result is pickled to `matched_cache/<file_stem>.pkl`.
On re-runs (to re-tune the aggregator or filter thresholds), Phase A
reads from the cache and skips Viterbi entirely. Invalidated with
`--clear-match-cache` (needed after parameter recalibration).

#### Multiprocessing Phase A

New `--workers N` flag spawns `N` `multiprocessing.Pool` workers, each with its
own copy of the road network (via an initializer). Files are distributed via
`imap_unordered(chunksize=4)`. Each worker maintains its own SSSP cache
(cold-start penalty the first few files, then steady-state).

```bash
python run_stage2_full.py --workers 4 --clear-match-cache
```

4 workers on an 8-core Mac is the sweet spot; memory footprint is ~100MB per
worker (the road network).

---

## Stage 3 — ETA model

### Feature engineering

The CatBoost feature matrix grew from 20 to **44 columns** grouped as follows:

#### Geometric / temporal (9 columns)

Computed from `source_time`, source/destination coordinates. No change in
scope versus the starting code — just listed here for completeness.

`haversine_km`, `manhattan_km`, `bearing`, `hour`, `minute_of_day`,
`day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`

#### Spatial clusters (3 columns)

`src_cluster`, `dst_cluster`, `od_pair` — KMeans-80 over source/destination
coordinates plus their concatenated categorical.

#### Physics / routing (10 columns) — `add_physics_features`

The routing function was substantially expanded. It now captures `return_predecessors=True`
from scipy Dijkstra, reconstructs the per-row shortest path, and uses it to
produce:

| Column | Meaning |
|---|---|
| `physics_eta_min` | Rolling-bin ETA: walks the path, re-bins speeds at each 30-min boundary crossed |
| `physics_eta_snapshot_min` | Old start-time-snapshot ETA (kept so CatBoost can learn when the two disagree) |
| `route_length_km` | Graph distance along the chosen path |
| `route_n_edges` | Real edge count (was stubbed to 0) |
| `route_n_turns` | Bearing changes > 30° along the path |
| `route_detour_ratio` | `route_length_km / haversine_km` |
| `route_pct_motorway` | Fraction of path edges classified as motorway/motorway_link |
| `route_pct_trunk` | Trunk + trunk_link |
| `route_pct_primary` | Primary |
| `route_pct_secondary` | Secondary |
| `route_pct_residential` | Residential + living_street |
| `route_pct_other` | Everything else |

The rolling-bin ETA matters for long trips that cross rush-hour boundaries —
the old snapshot approach assumed whatever speeds applied at `source_time`
held for the entire trip.

#### Historical target-encoded duration (3 columns) — `add_historical_features`

`hist_duration_min`, `hist_duration_od`, `hist_duration_hour` — median
historical duration at three levels of OD/hour granularity, with a fallback
cascade from most-specific to least-specific. See
[Out-of-fold target encoding](#out-of-fold-target-encoding) below for the
leakage fix.

#### Per-vehicle history (12 columns) — `add_vehicle_features`

Merged from `vehicle_features.pkl` (produced by `process_test_cases.py`):

| Column | Meaning |
|---|---|
| `v_mean_speed_ms` | Driver's historical mean speed |
| `v_median_speed_ms` | Median |
| `v_p20_speed_ms`, `v_p80_speed_ms` | Speed distribution spread |
| `v_total_km` | Total historical distance |
| `v_n_points` | GPS point count |
| `v_hour_dominant` | Most common hour of activity (categorical) |
| `v_hour_entropy` | How concentrated the driver's hours are |
| `v_pct_highway` | Fraction of matched edges that are motorway/trunk |
| `v_typical_trip_min` | Median trip duration in flag-segmented trips |
| `v_n_trips` | Number of detected trips |
| `vehicle_id` | Raw categorical (also still passed in) |

Before the merge, `vehicle_features.pkl` was being produced but never joined
into the feature matrix — `vehicle_id` was only used as a raw categorical,
throwing away all the aggregates. The merge handles the `.txt`-suffix
mismatch between the pickle keys (with suffix) and the Kaggle CSV's
`vehicle_id` field, and fills unseen vehicles with column-wise medians
computed on the training set and reused across val/test.

#### Weather (5 columns) — `add_weather_features`

New external signal. `weather_features.py` queries the `meteostat` Python
package for hourly observations at KSFO (San Francisco Intl.) station for
the 2008 span, caches to `sf_weather.pkl`, and joins by floored
`source_time` hour:

| Column | Source |
|---|---|
| `temp_c` | Meteostat `temp` |
| `precip_mm` | Meteostat `prcp`, filled 0 where null |
| `wind_kph` | Meteostat `wspd` |
| `weather_is_rain` | Derived: `precip_mm > 0.1` |
| `condition` | Meteostat `coco` code as string (categorical) |

Missing sensor readings are filled with a 3-hour rolling median before falling
back to the global median. Rain and wind both have reasonable physical stories
for ETA variance.

#### CatBoost categorical columns (9)

```python
CAT_COLS = ["src_cluster", "dst_cluster", "od_pair", "vehicle_id",
            "hour", "day_of_week", "is_weekend",
            "v_hour_dominant", "condition"]
```

---

### Out-of-fold target encoding

The original `add_historical_features` had a latent leakage bug: when called
during training (no `train_df` passed), it computed group medians from the
entire training set, so every row saw its own `duration_min` in the aggregate
it joined back onto itself. The leakage was bounded (medians dilute one value
across a whole group) but real.

The new implementation splits into two paths:

- **Training (`train_df is None`)** — runs `_oof_target_encode` with `KFold(n_splits=5, shuffle=True, random_state=42)`. Each fold's rows receive target-encoded values computed from **the other four folds only**. No row ever sees its own label in its own feature.
- **Val/test (`train_df is not None`)** — looks up stats from the full training set directly. No leakage concern since val/test rows were never in `train_df`.

The fallback cascade (OD+hour → OD → hour → global median) was preserved.

---

### Weekday/weekend stratified models

From the Kaggle course guidance: weekday and weekend traffic patterns differ
enough that separate models can outperform a single model with `is_weekend` as
one feature among many. The pipeline now trains two CatBoost regressors when
each subset has sufficient data:

```python
def _train_stratified(train_df, val_df, min_subset_rows=1000):
    if either subset has < min_subset_rows:
        return {"single": train_catboost(train_df, val_df)}
    return {
        "weekday": train_catboost(train_df[is_weekend == 0], val_df[is_weekend == 0]),
        "weekend": train_catboost(train_df[is_weekend == 1], val_df[is_weekend == 1]),
    }
```

Each model gets its own matching validation subset for early stopping.
`_predict_stratified` routes each input row to the matching model based on
its `is_weekend` flag. Per-subset RMSE is reported alongside the overall RMSE
so the user can see whether stratification helped. The `min_subset_rows=1000`
guard prevents overfitting the weekend model on sparse weekend data
(training on 300 rows and early-stopping would typically memorize).

The fallback to single-model preserves existing behavior for small runs
(e.g. `quick_diagnostics`) where the stratification would overfit.

---

### Outlier-safe predictions

RMSE squares the error, so a single 30-min outlier contributes 900 to the sum
while twenty 3-min errors contribute 180. The original `np.clip(pred, 0.5, 120)`
doesn't prevent the worst outliers — it only caps the absolute maximum. The
new `apply_sanity_fallback` applies three guards sequentially:

1. **Distance-based hard floor**: route cannot be faster than 40 m/s (~144 km/h absolute ceiling speed) times the greater of `route_length_km` or `haversine_km` (the max guards against broken physics routes where `route_length_km ≈ 0`).
2. **Distance-based hard ceiling**: same logic at 2.5 m/s (~9 km/h, gridlock crawl).
3. **Soft blend to physics ETA**: if `|pred / physics_eta|` falls outside `[1/3, 3]`, the prediction is blended `0.4 · pred + 0.6 · physics_eta`. Physics ETA is less accurate on average but never absurd, which makes it the right fallback for tail cases.

The three guards are applied to test predictions only (not to val RMSE
reporting, so model quality stays visible). The number of rows that got
adjusted by more than 0.5 minutes is logged.

---

## New helper modules

| Module | Purpose |
|---|---|
| `weather_features.py` | Meteostat SF weather loader + hour-flooring helper, pickled to `sf_weather.pkl` |
| `build_graphml.py` | One-time `osmnx.graph_from_place("San Francisco, California, USA")` → `sf_road_network.graphml` |
| `baseline_metrics.py` | Reference-metrics script. Runs the current matcher on N files and writes `baseline_metrics.txt` (wall-clock, coverage %, observations/edge) |
| `calibrate_params.py` | Empirical σ_z / β estimation. Writes `matching_params.pkl` consumed by `run_stage2_full.py` |
| `run_pipeline.sh` | End-to-end orchestration: builds graphml → baseline → calibrate → Phase A+B → vehicle features → Stage 3 diagnostics → Stage 3 full training |

---

## How to run

### Prerequisites

- Python 3.11 or 3.12
- `.venv` with dependencies from `requirements.txt` (notably `osmnx`, `catboost`, `scikit-learn`, `meteostat`)
- `Trajectories/` extracted from `Trajectories.zip`
- `Test Cases/` extracted from `Test cases.zip`
- `kaggle-test-file-minute.csv` at project root (user must obtain from Kaggle; excluded from `.gitignore`)

### One-shot end-to-end

```bash
# Long-running; use nohup + caffeinate to survive lid-close on macOS.
nohup caffeinate -i bash run_pipeline.sh > pipeline.log 2>&1 &
tail -f pipeline.log
```

Stages executed by `run_pipeline.sh` (paths shown relative to repo root):

1. `src/stage1_graph/build_graphml.py` — downloads SF road network (skipped if file exists)
2. `src/stage2_matching/baseline_metrics.py --files 20` — reference measurements
3. `src/stage2_matching/calibrate_params.py --files 50` — writes `matching_params.pkl`
4. `src/stage2_matching/run_stage2_full.py --workers 4 --clear-match-cache` — Phase A (map matching) + Phase B (propagation) → `complete_speeds.pkl`
5. `src/stage2_matching/process_test_cases.py` — writes `vehicle_features.pkl` and merges Test-Cases observations into `complete_speeds.pkl`
6. `quick_diagnostics()` — Stage 3 feature-matrix smoke test with assertions
7. `src/stage3_eta/stage3_eta_prediction.py` — Stage 3 full training + test prediction → `submission.csv`
8. Submission verification (row count + schema)

### Running stages individually

```bash
# Rebuild just the road network
python src/stage1_graph/build_graphml.py --force

# Re-calibrate parameters from a different sample size
python src/stage2_matching/calibrate_params.py --files 100

# Re-run Stage 2 with 8 workers on a cached Phase A
python src/stage2_matching/run_stage2_full.py --workers 8                   # reuses matched_cache/
python src/stage2_matching/run_stage2_full.py --workers 8 --clear-match-cache   # forces re-match

# Re-run Stage 3 against a locked complete_speeds.pkl
python src/stage3_eta/stage3_eta_prediction.py

# Build a held-out val set for honest local iteration
python src/utils/build_kaggle_like_val.py --per-vehicle 8 --out kaggle_like_val.csv

# Standalone alternate predictor (no CatBoost, no physics ETA)
python src/baselines/knn_predict.py --k 10 --hr-radius 2 --out submission_knn.csv
```

### Tuning notes

- `matched_cache/` is keyed by filename stem only. If you change
  `map_matching_fast.py` logic (not just parameters), pass
  `--clear-match-cache` to force regeneration.
- Phase A is the long pole (~90 min on 486 files with 4 workers under
  moderate system load). Close competing apps (Zoom, VMs, Spotlight
  indexing) to get full CPU utilization.

---

## Repository layout

All Python source lives under `src/`, grouped by pipeline stage. Generated
data, submissions, caches, and trajectory inputs are git-ignored — they
live alongside the source on disk but never enter the repo.

```
mini-google-map/
├── README.md
├── requirements.txt
├── run_pipeline.sh                       # end-to-end orchestration (Stage 1→3)
├── run_resume.sh                         # resume from Stage 5 (after Phase A+B)
├── .gitignore
└── src/
    ├── _path_bootstrap.py                # makes every src/<subdir>/ importable as a flat namespace
    │
    ├── stage1_graph/
    │   └── build_graphml.py              # `osmnx.graph_from_place(...)` → sf_road_network.graphml
    │
    ├── stage2_matching/                  # HMM map matching + speed aggregation
    │   ├── map_matching_solution.py      # base HMM (emission, transition, Viterbi, projection)
    │   ├── map_matching_fast.py          # faster matcher: bearing emission, adaptive radius, path-break detection, sibling-import contract
    │   ├── calibrate_params.py           # MAD-based σ_z + mean β estimation → matching_params.pkl
    │   ├── baseline_metrics.py           # Phase 0 — wall-clock + coverage on a sample
    │   ├── run_stage2_full.py            # Phase A (match) + Phase B (propagate) → complete_speeds.pkl
    │   └── process_test_cases.py         # match Test Cases trajectories, build vehicle_features.pkl
    │
    ├── stage3_eta/                       # Feature matrix + CatBoost/LightGBM
    │   ├── stage3_eta_prediction.py      # ETA model: 48-feature matrix, residual target, ensemble, sanity fallback → submission.csv
    │   ├── weather_features.py           # Meteostat hourly SF weather loader → sf_weather.pkl
    │   └── enhance_vehicle_features.py   # appends per-hour median-speed columns to vehicle_features.pkl
    │
    ├── utils/                            # eval + post-processing
    │   ├── build_kaggle_like_val.py      # synthesize a held-out val set in Kaggle test schema
    │   └── calibrate_submission.py       # quantile-match a submission to a target distribution
    │
    └── baselines/                        # alternate, completely-different predictors
        └── knn_predict.py                # per-vehicle k-NN over historical trip patterns
```

### Cross-subdir imports

Every file in `src/<subdir>/` starts with a small bootstrap snippet that
adds *all* `src/<subdir>/` directories to `sys.path` and exposes
`PROJECT_ROOT`. Entry-point scripts then `os.chdir(PROJECT_ROOT)` in
their `__main__` block, so the relative paths in the code (e.g.
`'sf_road_network.graphml'`, `'Trajectories'`) resolve consistently no
matter where the script was invoked from.

### Generated artifacts (git-ignored)

| Artifact | Producer | Consumer |
|---|---|---|
| `sf_road_network.graphml` | `build_graphml.py` | Stages 2 and 3 |
| `matching_params.pkl` | `calibrate_params.py` | `run_stage2_full.py` |
| `matched_cache/*.pkl` | `run_stage2_full.py` | `run_stage2_full.py` (self-cache) |
| `observed_speeds.pkl` | Phase A | Phase B |
| `complete_speeds.pkl` | Phase B | Stage 3 |
| `vehicle_features.pkl` | `process_test_cases.py` | Stage 3 |
| `sf_weather.pkl` | `weather_features.py` | Stage 3 |
| `baseline_metrics.txt` | `baseline_metrics.py` | Humans |
| `kaggle_like_val.csv` | `build_kaggle_like_val.py` | Stage 3 (val set) |
| `submission*.csv` | `stage3_eta_prediction.py` + variants | Kaggle |

---

## Implementation notes

### On the HMM calibration values

For this dataset, calibration typically produces `σ_z ≈ 12m` and the raw `β`
estimate is large enough to trigger the sanity clip (β ≥ 30m). A large raw β
suggests a subset of trajectories have path-break or wrong-street matches
where the graph route diverges substantially from the straight-line distance.
The clip prevents transition probabilities from becoming near-uniform, which
would let Viterbi lose discriminating power. If calibration is rerun and
β no longer needs clipping, it means the path-break detection and bearing
improvements reduced the tail of wrongly-matched pairs.

### On CatBoost stratification

Stratification is only helpful when both subsets have enough samples to fit
deep-tree structures. For the full dataset (~500 cab files, ~15k training
rows after sampling), weekday and weekend each have >1000 rows and
stratification should be a clean win. The `min_subset_rows=1000` guard
prevents regression on small experimental runs. The fallback path ensures the
pipeline still works end-to-end on a 50-file sample.

### On the weather join

Meteostat returns UTC-indexed observations. `source_time` is stored as a UTC
Unix timestamp. `floor_to_hour_utc` aligns both to the same hour boundary.
Missing weather hours (a handful in 2008 data) are filled with a rolling
3-hour median before falling back to the global median. This matters
because CatBoost doesn't handle NaN in numerical features gracefully under
all settings.

### On map-matched output contract

Older code in this repo assumes `matched[i]` is always a valid tuple and
indexes directly. After the Stage 2 refactor, `matched[i]` can be `None`
(dropped point). All downstream consumers (`compute_segment_speeds`,
`compute_vehicle_features`, `visualize_matching`) were updated to use
`iter_matched_pairs` or an explicit `is None` check. If you write new code
that consumes matched output, use `iter_matched_pairs(trajectory, matched)`
rather than manual indexing.

---
