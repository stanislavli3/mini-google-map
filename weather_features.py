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

    df = pd.DataFrame({
        "timestamp_hour": (data.index.astype("int64") // 10**9).astype("int64"),
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
