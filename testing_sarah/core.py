"""Pure data operations for flight-track extraction and dataset cutting."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd
import xarray as xr


def _values_for_name(ds: xr.Dataset, name: str) -> pd.Series:
    """Return a 1D pandas Series for a dataset coordinate or data variable."""
    if name in ds:
        values = ds[name].values
    elif name in ds.coords:
        values = ds.coords[name].values
    else:
        raise KeyError(f"Dataset does not contain {name!r}.")

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name!r} must be one-dimensional for track extraction.")
    return pd.Series(array)


def extract_track(
    ds: xr.Dataset,
    lat: str = "lat",
    lon: str = "lon",
    profile_dim: Hashable = "nprofile",
    time: str | None = "time",
) -> pd.DataFrame:
    """Extract a flight track from an xarray dataset.

    Parameters
    ----------
    ds:
        Source dataset containing latitude and longitude along a profile dimension.
    lat, lon:
        Names of latitude and longitude variables or coordinates.
    profile_dim:
        Name of the profile dimension. If a matching coordinate exists, its values
        are used; otherwise integer positions are used.
    time:
        Optional time variable or coordinate name. Missing time is ignored.

    Returns
    -------
    pandas.DataFrame
        DataFrame with at least ``lat``, ``lon``, and ``profile`` columns. A
        ``time`` column is included when available.
    """
    if profile_dim not in ds.dims and profile_dim not in ds.coords:
        raise KeyError(f"Dataset does not contain profile dimension {profile_dim!r}.")

    lat_values = _values_for_name(ds, lat)
    lon_values = _values_for_name(ds, lon)
    if len(lat_values) != len(lon_values):
        raise ValueError("Latitude and longitude arrays must have the same length.")

    if profile_dim in ds.coords:
        profile_values = pd.Series(ds.coords[profile_dim].values)
    else:
        profile_values = pd.Series(range(len(lat_values)))

    if len(profile_values) != len(lat_values):
        raise ValueError(
            "Profile coordinate length must match latitude/longitude length."
        )

    track = pd.DataFrame(
        {
            "lat": lat_values.to_numpy(),
            "lon": lon_values.to_numpy(),
            "profile": profile_values.to_numpy(),
        }
    )

    if time and (time in ds or time in ds.coords):
        time_values = _values_for_name(ds, time)
        if len(time_values) == len(track):
            track["time"] = time_values.to_numpy()

    return track


def thin_track(df: pd.DataFrame, step: int = 20) -> pd.DataFrame:
    """Return every nth row of a track DataFrame for efficient plotting."""
    if step < 1:
        raise ValueError("step must be at least 1.")
    return df.iloc[::step].copy().reset_index(drop=True)


def cut_dataset(
    ds: xr.Dataset,
    start_profile: int | float,
    end_profile: int | float,
    profile_dim: Hashable = "nprofile",
) -> xr.Dataset:
    """Cut a dataset between two profile coordinate values or positions.

    The order of ``start_profile`` and ``end_profile`` does not matter. If
    ``profile_dim`` has a coordinate, inclusive coordinate slicing is used.
    Otherwise inclusive integer-position slicing is used.
    """
    start, end = sorted((start_profile, end_profile))

    if profile_dim in ds.coords:
        return ds.sel({profile_dim: slice(start, end)})

    if profile_dim not in ds.dims:
        raise KeyError(f"Dataset does not contain profile dimension {profile_dim!r}.")

    return ds.isel({profile_dim: slice(int(start), int(end) + 1)})

def save_dataset(
    ds: xr.Dataset,
    outdir: str):
    ds.to_netcdf(outdir+"selected_flight_section.nc")
    return True