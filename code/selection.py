"""Selection helpers for mapping clicked points back to source profiles."""

from __future__ import annotations

from typing import Any

import pandas as pd


def nearest_track_point(
    lon: float,
    lat: float,
    track: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    profile_col: str = "profile",
) -> dict[str, Any]:
    """Find the nearest track point to a clicked lon/lat pair.

    This prototype uses squared distance in lon/lat degrees. The returned
    dictionary includes the original profile value, so a thinned plotted track
    can still map back to the full dataset.
    """
    required = {lon_col, lat_col, profile_col}
    missing = required.difference(track.columns)
    if missing:
        raise KeyError(f"Track is missing required columns: {sorted(missing)}")
    if track.empty:
        raise ValueError("track must contain at least one point.")

    distances = (track[lon_col] - lon) ** 2 + (track[lat_col] - lat) ** 2
    idx = distances.idxmin()
    row = track.loc[idx]

    return {
        "index": idx,
        "profile": row[profile_col],
        "lon": row[lon_col],
        "lat": row[lat_col],
        "distance2": distances.loc[idx],
    }
