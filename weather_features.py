"""
weather_features.py
===================

Pull historical hourly weather for San Francisco from Meteostat and cache it
to disk. Stage 3 joins this to each training/test row by source-time hour so
CatBoost can pick up rain/wind/temperature effects on travel time.

Meteostat queries the nearest KSFO (San Francisco Intl.) ICAO station and
returns hourly observations in UTC.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# pandas 3 compatibility shim for meteostat
# ---------------------------------------------------------------------------
# meteostat 1.6.8's Hourly/Daily/Monthly classes call pd.read_csv with
# parse_dates={"time": [0, 1]} — a pandas 2 feature for combining multiple
# columns into a single datetime. pandas 3 rejects that form:
#   TypeError: Only booleans and lists are accepted for the 'parse_dates' parameter
# We patch pd.read_csv once at import so meteostat's calls succeed.
_ORIGINAL_READ_CSV = pd.read_csv


def _read_csv_pd3_compat(*args, **kwargs):
    parse_dates = kwargs.get("parse_dates")
    if isinstance(parse_dates, dict):
        combine_spec = dict(parse_dates)
        kwargs["parse_dates"] = False
        df = _ORIGINAL_READ_CSV(*args, **kwargs)
        for new_col, cols in combine_spec.items():
            # cols can be positional indices (the meteostat case) or names
            if all(isinstance(c, int) for c in cols):
                src_names = [df.columns[i] for i in cols]
            else:
                src_names = list(cols)
            combined = df[src_names].astype(str).agg(" ".join, axis=1)
            df[new_col] = pd.to_datetime(combined, errors="coerce")
            df = df.drop(columns=src_names)
        # Reorder so the new datetime column(s) come first — matches the
        # positional semantics downstream meteostat code relies on.
        first = [k for k in combine_spec]
        others = [c for c in df.columns if c not in first]
        df = df[first + others]
        return df
    return _ORIGINAL_READ_CSV(*args, **kwargs)


pd.read_csv = _read_csv_pd3_compat


# pandas 3 renamed frequency strings: "H" → "h" (hourly), "T" → "min", etc.
# meteostat 1.6.8 uses the old uppercase aliases. We fix them by mutating
# the class attribute before any instance is created.
def _apply_meteostat_freq_patch():
    try:
        from meteostat.interface.hourly import Hourly
        if getattr(Hourly, "_freq", None) == "1H":
            Hourly._freq = "1h"
    except Exception:
        pass
    try:
        from meteostat.interface.daily import Daily
        if getattr(Daily, "_freq", None) == "1D":
            Daily._freq = "1D"  # pandas 3 still accepts "1D"; no-op
    except Exception:
        pass


_apply_meteostat_freq_patch()


SF_LAT = 37.7749
SF_LON = -122.4194
CACHE_PATH_DEFAULT = "sf_weather.pkl"


def load_sf_weather(
    start_year: int = 2008,
    end_year: int = 2009,
    cache_path: str = CACHE_PATH_DEFAULT,
) -> pd.DataFrame:
    """
    Return a DataFrame indexed implicitly by `timestamp_hour` (int unix ts,
    floored to the hour, UTC) with columns:
      timestamp_hour, temp_c, precip_mm, wind_kph, condition

    `condition` is the meteostat coco code as a string ("1".."27"); Stage 3
    treats it as a CatBoost categorical.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from meteostat import Point, Hourly

    location = Point(SF_LAT, SF_LON)
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31, 23, 59)
    hourly = Hourly(location, start, end)
    data = hourly.fetch()

    if data.empty:
        raise RuntimeError(
            f"Meteostat returned no data for SF {start_year}-{end_year}. "
            "Check network connectivity."
        )

    # meteostat returns a tz-naive DatetimeIndex in UTC — make it explicit
    data.index = data.index.tz_localize("UTC") if data.index.tz is None else data.index

    # pandas 3's .astype('int64') on DatetimeIndex now returns microseconds,
    # not nanoseconds, so the `// 10**9` pattern gives wrong unix seconds.
    # Convert to naive UTC then cast to second-resolution datetime64.
    naive_utc = data.index.tz_convert("UTC").tz_localize(None)
    ts_sec = naive_utc.astype("datetime64[s]").astype("int64")
    df = pd.DataFrame({
        "timestamp_hour": ts_sec,
        "temp_c": data["temp"].values,
        "precip_mm": data["prcp"].fillna(0.0).values,
        "wind_kph": data["wspd"].values,
        "condition": data["coco"].fillna(0).astype("int64").astype(str).values,
    })

    # Fill sensor gaps with a rolling 3-hour median before falling back to
    # overall means — avoids NaNs leaking into CatBoost for a handful of
    # hours without reports.
    df = df.sort_values("timestamp_hour").reset_index(drop=True)
    for col in ("temp_c", "wind_kph"):
        df[col] = df[col].fillna(df[col].rolling(3, min_periods=1).median())
        df[col] = df[col].fillna(df[col].median())

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    return df


def floor_to_hour_utc(ts: int) -> int:
    """Floor a unix timestamp to the hour boundary, UTC."""
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return int(dt.timestamp())
