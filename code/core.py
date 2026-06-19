"""Pure data operations for flight-track extraction and dataset cutting."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _values_for_name(ds: xr.Dataset, name: str, profile_dim: Hashable = "nprofile") -> pd.Series:
    """Return one value per profile for a coordinate or data variable."""
    if name in ds:
        data = ds[name]
    elif name in ds.coords:
        data = ds.coords[name]
    else:
        raise KeyError(f"Dataset does not contain {name!r}.")

    if profile_dim not in data.dims:
        raise ValueError(f"{name!r} must use profile dimension {profile_dim!r}.")

    array = data.transpose(profile_dim, ...).values
    array = array.reshape(ds.sizes[profile_dim], -1)
    values = np.array(
        [
            row[np.isfinite(row)][0] if np.issubdtype(row.dtype, np.number) and np.isfinite(row).any()
            else row[0]
            for row in array
        ]
    )
    return pd.Series(values)


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

    lat_values = _values_for_name(ds, lat, profile_dim=profile_dim)
    lon_values = _values_for_name(ds, lon, profile_dim=profile_dim)
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
        time_values = _values_for_name(ds, time, profile_dim=profile_dim)
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

def save_dataset(ds: xr.Dataset, outdir: str | Path, filename: str = "selected_flight_section.nc") -> Path:
    """Save the selected flight section to the processed data folder."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / filename
    ds.to_netcdf(output_path)
    return output_path
