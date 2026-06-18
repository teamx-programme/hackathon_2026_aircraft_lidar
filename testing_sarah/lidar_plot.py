"""Early placeholder plotting helpers for lidar curtain views."""

from __future__ import annotations

from collections.abc import Hashable

import matplotlib.pyplot as plt
import xarray as xr


def plot_lidar_curtain(
    ds: xr.Dataset,
    x: Hashable = "nprofile",
    z: Hashable = "height",
    variable: Hashable | None = None,
):
    """Plot a simple lidar curtain from a dataset.

    Parameters
    ----------
    ds:
        Dataset containing a 2D variable with profile and vertical dimensions.
    x, z:
        Horizontal/profile and vertical coordinate names.
    variable:
        Variable to plot. If omitted, the first 2D data variable containing both
        ``x`` and ``z`` dimensions is used.
    """
    if variable is None:
        for name, data_array in ds.data_vars.items():
            if x in data_array.dims and z in data_array.dims:
                variable = name
                break
    if variable is None:
        raise ValueError("No suitable 2D variable found for lidar curtain plotting.")

    data = ds[variable]
    fig, ax = plt.subplots()
    data.plot(x=x, y=z, ax=ax)
    ax.set_title(str(variable))
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(z))
    return fig, ax
