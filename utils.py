"""Small utilities for examples and prototyping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def create_dummy_dataset(nprofile: int = 240, nheight: int = 80) -> xr.Dataset:
    """Create a synthetic flight/lidar dataset for demos and tests."""
    profiles = np.arange(nprofile)
    heights = np.linspace(0, 12_000, nheight)
    lat = 47.5 + 0.6 * np.sin(np.linspace(0, 2.8, nprofile))
    lon = 10.5 + 1.0 * np.cos(np.linspace(0, 2.8, nprofile))
    time = pd.date_range("2026-06-08T09:00:00", periods=nprofile, freq="30s")
    altitude = 9_000 + 800 * np.sin(np.linspace(0, 5, nprofile))

    signal = np.exp(-((heights[None, :] - altitude[:, None]) ** 2) / 5_000_000)
    signal += 0.05 * np.random.default_rng(7).normal(size=(nprofile, nheight))

    return xr.Dataset(
        data_vars={
            "lat": ("nprofile", lat),
            "lon": ("nprofile", lon),
            "time": ("nprofile", time),
            "altitude": ("nprofile", altitude),
            "backscatter": (("nprofile", "height"), signal),
        },
        coords={
            "nprofile": profiles,
            "height": heights,
        },
        attrs={"description": "Synthetic dataset for teamx_proto demos."},
    )


def create_large_dummy_dataset() -> xr.Dataset:
    """Create a 10,000-profile dummy dataset for map performance testing."""
    return create_dummy_dataset(nprofile=10_000, nheight=80)
