"""Self-contained 2D TEAMx lidar curtain plotting.

This file is meant for the hackathon repository layout where ``app.py``
hands a cut-down ``*_pro``-style NetCDF file to the plotting code.  It keeps
all helper functions in this file so it can be copied next to ``app.py`` and
``plot3d.py`` without depending on the thesis package structure.
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer


ALLOWED_WIND_COMPONENTS = ("u", "v", "w")
DEFAULT_GROUP = "combined"
DEFAULT_LON_VAR = "AER_lon_avg"
DEFAULT_LAT_VAR = "AER_lat_avg"
DEFAULT_ALT_VAR = "GEO_alt_avg"
DEFAULT_PROFILE_DIM = "nprofile"


def open_lidar_file(lidar_file: str | Path, group: str | None = DEFAULT_GROUP) -> xr.Dataset:
    """Open a TEAMx lidar NetCDF file, falling back from ``combined`` to root."""
    path = Path(lidar_file)
    if not path.exists():
        raise FileNotFoundError(f"Lidar file not found: {path}")
    if path.suffix.lower() != ".nc":
        raise ValueError(f"Expected a .nc file, got: {path.name}")

    if group is None:
        return xr.open_dataset(path)

    try:
        return xr.open_dataset(path, group=group)
    except (OSError, ValueError):
        return xr.open_dataset(path)


def load_lidar_dataset(
    lidar_source: str | Path | xr.Dataset,
    group: str | None = DEFAULT_GROUP,
) -> xr.Dataset:
    """Return an in-memory dataset from a path or an already opened dataset."""
    if isinstance(lidar_source, xr.Dataset):
        return lidar_source

    ds = open_lidar_file(lidar_source, group=group)
    try:
        return ds.load()
    finally:
        ds.close()


def available_wind_components(ds: xr.Dataset) -> list[str]:
    """Return available wind components among ``u``, ``v``, and ``w``."""
    return [name for name in ALLOWED_WIND_COMPONENTS if name in ds]


def validate_wind_component(ds: xr.Dataset, variable: str) -> str:
    """Validate that a requested wind component exists in the dataset."""
    variable = str(variable).lower()
    if variable not in ALLOWED_WIND_COMPONENTS:
        raise ValueError("variable must be one of: u, v, w")
    if variable not in ds:
        available = ", ".join(available_wind_components(ds)) or "none"
        raise KeyError(f"Dataset has no {variable!r} variable. Available: {available}")
    return variable


def validate_dataset_for_plotting(
    ds: xr.Dataset,
    variable: str,
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    alt_var: str = DEFAULT_ALT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
) -> str:
    """Check the minimum fields needed for a 2D curtain plot."""
    variable = validate_wind_component(ds, variable)
    required = [lon_var, lat_var, alt_var, variable]
    missing = [name for name in required if name not in ds and name not in ds.coords]
    if missing:
        raise KeyError(f"Dataset is missing required variables: {missing}")
    if profile_dim not in ds.dims and profile_dim not in ds.coords:
        raise KeyError(f"Dataset has no profile dimension {profile_dim!r}.")
    return variable


def plot_2d(
    lidar_source: str | Path | xr.Dataset,
    dem_path: str | Path,
    variable: str = "w",
    group: str | None = DEFAULT_GROUP,
    output_path: str | Path | None = None,
    show: bool = False,
    **kwargs: Any,
):
    """Load a cut NetCDF file and create a 2D wind-component curtain plot.

    Parameters
    ----------
    lidar_source:
        Path to a cut ``.nc`` file or an in-memory ``xarray.Dataset``.
    dem_path:
        DEM GeoTIFF used for the terrain line below the curtain.
    variable:
        Wind component to color by: ``"u"``, ``"v"``, or ``"w"``.
    output_path:
        Optional image path, for example ``"plot2d.png"``.
    show:
        If ``True``, call ``matplotlib.pyplot.show()`` after plotting.
    kwargs:
        Forwarded to :func:`plot_selected_2d`.
    """
    ds = load_lidar_dataset(lidar_source, group=group)
    fig, ax = plot_selected_2d(ds, dem_path, variable=variable, **kwargs)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_selected_2d(
    ds: xr.Dataset,
    dem_path: str | Path,
    variable: str = "w",
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    alt_var: str = DEFAULT_ALT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
    alt_dim: Hashable | None = None,
    z_min: float = 500,
    z_max: float = 3200,
    dz: float = 25,
    wind_every_n: int = 2,
    vertical_every_n: int = 1,
    colour_levels: Sequence[float] | None = None,
    cmap: str = "bwr",
):
    """Plot one selected wind component as a 2D curtain."""
    variable = validate_dataset_for_plotting(
        ds,
        variable,
        lon_var=lon_var,
        lat_var=lat_var,
        alt_var=alt_var,
        profile_dim=profile_dim,
    )
    _validate_existing_path(dem_path, "DEM")
    _validate_step(wind_every_n, "wind_every_n")
    _validate_step(vertical_every_n, "vertical_every_n")

    ds_track, track_info = add_straight_track_coordinates(
        ds,
        lon_var=lon_var,
        lat_var=lat_var,
        profile_dim=profile_dim,
    )
    has_uv_wind = {"u", "v"}.issubset(ds_track.data_vars)
    if has_uv_wind:
        ds_track = add_track_relative_wind(ds_track, track_info)

    alt_dim = _infer_alt_dim(ds_track, alt_var, profile_dim, alt_dim)
    topo = sample_topography_from_dem_2d(
        ds_track,
        dem_path,
        lon_var=lon_var,
        lat_var=lat_var,
        profile_dim=profile_dim,
        every_n=1,
    )
    topo = topo.assign(along_track_m=ds_track["along_track_m"])

    z_grid = np.arange(z_min, z_max + dz, dz)
    along = ds_track["along_track_m"].values
    altitude = _altitude_values(ds_track, alt_var, profile_dim, alt_dim)
    values = _curtain_values(ds_track, variable, profile_dim, alt_dim)
    curtain = _interpolate_to_regular_altitude_grid(altitude, values, z_grid)

    x_grid, z_mesh = np.meshgrid(along, z_grid)
    fig, ax = plt.subplots(figsize=(24, 10))
    levels = (
        np.asarray(colour_levels, dtype=float)
        if colour_levels is not None
        else _default_wind_component_levels(curtain)
    )

    filled = ax.contourf(
        x_grid,
        z_mesh,
        curtain,
        levels=levels,
        cmap=cmap,
        extend="both",
    )
    fig.colorbar(filled, ax=ax, label=f"{variable} [m s$^{{-1}}$]")
    ax.contour(x_grid, z_mesh, curtain, levels=levels, colors="k", linewidths=0.4)

    if has_uv_wind and "w" in ds_track:
        _add_wind_quiver(
            ax,
            ds_track,
            alt_var=alt_var,
            w_var="w",
            profile_dim=profile_dim,
            alt_dim=alt_dim,
            horizontal_every_n=wind_every_n,
            vertical_every_n=vertical_every_n,
        )

    _plot_topography(ax, topo, z_min)
    ax.set_xlabel("Distance along selected flight section [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(f"2D curtain plot of wind component {variable}")
    ax.set_ylim(z_min, z_max)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig, ax


def plot_wind_curtain_panels(
    lidar_source: str | Path | xr.Dataset,
    dem_path: str | Path,
    group: str | None = DEFAULT_GROUP,
    alt_var: str = DEFAULT_ALT_VAR,
    w_var: str = "w",
    u_var: str = "u",
    v_var: str = "v",
    ff_h_var: str = "ff_h",
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
    alt_dim: Hashable | None = None,
    z_min: float = 500,
    z_max: float = 3200,
    dz: float = 25,
    vertical_every_n: int = 1,
    horizontal_every_n: int = 2,
    barb_length: float = 5.5,
    panels: Sequence[str] = ("projected_wind", "horizontal_wind"),
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Create the thesis-style stacked 2D wind curtain panels."""
    ds = load_lidar_dataset(lidar_source, group=group)
    _validate_step(vertical_every_n, "vertical_every_n")
    _validate_step(horizontal_every_n, "horizontal_every_n")
    if not panels:
        raise ValueError("At least one panel must be requested.")

    for name in [lon_var, lat_var, alt_var, u_var, v_var, w_var]:
        if name not in ds and name not in ds.coords:
            raise KeyError(f"Dataset is missing required variable {name!r}.")

    ds_track, track_info = add_straight_track_coordinates(
        ds,
        lon_var=lon_var,
        lat_var=lat_var,
        profile_dim=profile_dim,
    )
    ds_track = add_track_relative_wind(ds_track, track_info, u_var=u_var, v_var=v_var)
    alt_dim = _infer_alt_dim(ds_track, alt_var, profile_dim, alt_dim)

    topo = sample_topography_from_dem_2d(
        ds_track,
        dem_path,
        lon_var=lon_var,
        lat_var=lat_var,
        profile_dim=profile_dim,
        every_n=1,
    )
    topo = topo.assign(along_track_m=ds_track["along_track_m"])
    z_grid = np.arange(z_min, z_max + dz, dz)

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(24, 5 * len(panels)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    plotters = {
        "projected_wind": _plot_projected_wind_panel,
        "horizontal_wind": _plot_horizontal_wind_panel,
    }
    panel_context = {
        "fig": fig,
        "ds_track": ds_track,
        "topo": topo,
        "alt_var": alt_var,
        "w_var": w_var,
        "u_var": u_var,
        "v_var": v_var,
        "ff_h_var": ff_h_var,
        "profile_dim": profile_dim,
        "alt_dim": alt_dim,
        "z_grid": z_grid,
        "z_min": z_min,
        "z_max": z_max,
        "vertical_every_n": vertical_every_n,
        "horizontal_every_n": horizontal_every_n,
        "barb_length": barb_length,
    }

    for ax, panel in zip(axes, panels):
        if panel not in plotters:
            known = ", ".join(plotters)
            raise ValueError(f"Unknown panel {panel!r}. Choose from: {known}.")
        plotters[panel](ax=ax, **panel_context)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def local_utm_crs(lon: np.ndarray, lat: np.ndarray) -> CRS:
    """Choose a local UTM CRS from longitude and latitude values."""
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    zone = int(np.floor((lon_med + 180) / 6) + 1)
    epsg = 32600 + zone if lat_med >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def add_straight_track_coordinates(
    ds: xr.Dataset,
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
):
    """Add along-track and cross-track coordinates from a mean straight track."""
    lon = _profile_values(ds, lon_var, profile_dim)
    lat = _profile_values(ds, lat_var, profile_dim)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        raise ValueError("At least two finite lon/lat profiles are required.")

    crs = local_utm_crs(lon[valid], lat[valid])
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon[valid], lat[valid])
    xy = np.column_stack([x, y])

    center = xy.mean(axis=0)
    _, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    along_unit = vh[0]
    if np.dot(along_unit, xy[-1] - xy[0]) < 0:
        along_unit = -along_unit
    cross_unit = np.array([-along_unit[1], along_unit[0]])

    along_raw = (xy - center) @ along_unit
    along = along_raw - along_raw[0]
    cross = (xy - center) @ cross_unit

    along_full = np.full(ds.sizes[profile_dim], np.nan)
    cross_full = np.full(ds.sizes[profile_dim], np.nan)
    along_full[valid] = along
    cross_full[valid] = cross

    out = ds.assign(
        along_track_m=(profile_dim, along_full),
        cross_track_m=(profile_dim, cross_full),
    )
    track_info = {
        "crs": crs.to_string(),
        "center_x_m": center[0],
        "center_y_m": center[1],
        "along_start_m": along_raw[0],
        "along_unit_east_north": along_unit,
        "cross_unit_east_north": cross_unit,
    }
    return out, track_info


def add_track_relative_wind(
    ds: xr.Dataset,
    track_info: dict[str, Any],
    u_var: str = "u",
    v_var: str = "v",
) -> xr.Dataset:
    """Rotate east/north wind components into along-track coordinates."""
    along_unit = np.asarray(track_info["along_unit_east_north"])
    cross_unit = np.asarray(track_info["cross_unit_east_north"])
    return ds.assign(
        wind_along_track=ds[u_var] * along_unit[0] + ds[v_var] * along_unit[1],
        wind_cross_track=ds[u_var] * cross_unit[0] + ds[v_var] * cross_unit[1],
    )


def sample_topography_from_dem_2d(
    ds: xr.Dataset,
    dem_path: str | Path,
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
    every_n: int = 1,
) -> xr.Dataset:
    """Sample DEM elevation along the flight track."""
    _validate_step(every_n, "every_n")
    track = ds[[lon_var, lat_var]].isel({profile_dim: slice(None, None, every_n)})
    lon = _profile_values(track, lon_var, profile_dim)
    lat = _profile_values(track, lat_var, profile_dim)
    topo = _sample_dem(dem_path, lon, lat)
    profile_coord = (
        track[profile_dim].values
        if profile_dim in track.coords
        else np.arange(ds.sizes[profile_dim])[::every_n]
    )

    return xr.Dataset(
        data_vars={
            lon_var: (profile_dim, lon),
            lat_var: (profile_dim, lat),
            "topography_alt": (profile_dim, topo),
        },
        coords={profile_dim: profile_coord},
    )


def _sample_dem(dem_path: str | Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample DEM values at longitude/latitude points."""
    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError("DEM has no CRS information.")
        if dem.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs("EPSG:4326", dem.crs, always_xy=True)
            xs, ys = transformer.transform(lon, lat)
        else:
            xs, ys = lon, lat

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        valid = (
            np.isfinite(xs)
            & np.isfinite(ys)
            & (xs >= dem.bounds.left)
            & (xs <= dem.bounds.right)
            & (ys >= dem.bounds.bottom)
            & (ys <= dem.bounds.top)
        )

        topo = np.full(len(xs), np.nan)
        coords = list(zip(xs[valid], ys[valid]))
        if coords:
            sampled = np.array([value[0] for value in dem.sample(coords)], dtype=float)
            if dem.nodata is not None:
                sampled = np.where(sampled == dem.nodata, np.nan, sampled)
            topo[valid] = sampled

    return topo


def _plot_projected_wind_panel(
    ax,
    fig,
    ds_track,
    topo,
    alt_var,
    w_var,
    profile_dim,
    alt_dim,
    z_grid,
    z_min,
    z_max,
    vertical_every_n,
    horizontal_every_n,
    **_,
):
    along = ds_track["along_track_m"].values
    altitude = _altitude_values(ds_track, alt_var, profile_dim, alt_dim)
    w_values = _curtain_values(ds_track, w_var, profile_dim, alt_dim)
    curtain = _interpolate_to_regular_altitude_grid(altitude, w_values, z_grid)
    x_grid, z_mesh = np.meshgrid(along, z_grid)
    levels = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])

    filled = ax.contourf(
        x_grid,
        z_mesh,
        curtain,
        levels=levels,
        cmap="bwr",
        extend="both",
    )
    fig.colorbar(filled, ax=ax, label="Vertical velocity w [m s$^{-1}$]")
    ax.contour(x_grid, z_mesh, curtain, levels=levels, colors="k", linewidths=0.4)
    _add_wind_quiver(
        ax,
        ds_track,
        alt_var=alt_var,
        w_var=w_var,
        profile_dim=profile_dim,
        alt_dim=alt_dim,
        horizontal_every_n=horizontal_every_n,
        vertical_every_n=vertical_every_n,
    )
    _plot_topography(ax, topo, z_min)
    _style_axis(ax, z_min, z_max, "Projected along-track wind and vertical velocity")


def _plot_horizontal_wind_panel(
    ax,
    fig,
    ds_track,
    topo,
    alt_var,
    u_var,
    v_var,
    ff_h_var,
    profile_dim,
    alt_dim,
    z_grid,
    z_min,
    z_max,
    vertical_every_n,
    horizontal_every_n,
    barb_length,
    **_,
):
    ds_speed = ds_track.assign(
        _horizontal_speed=_horizontal_speed(ds_track, ff_h_var, u_var, v_var)
    )
    along = ds_speed["along_track_m"].values
    altitude = _altitude_values(ds_speed, alt_var, profile_dim, alt_dim)
    speed = _curtain_values(ds_speed, "_horizontal_speed", profile_dim, alt_dim)
    curtain = _interpolate_to_regular_altitude_grid(altitude, speed, z_grid)
    x_grid, z_mesh = np.meshgrid(along, z_grid)
    levels = _positive_levels(curtain)

    filled = ax.contourf(
        x_grid,
        z_mesh,
        curtain,
        levels=levels,
        cmap="viridis",
        extend="max",
    )
    fig.colorbar(filled, ax=ax, label="Horizontal wind speed [m s$^{-1}$]")
    ax.contour(x_grid, z_mesh, curtain, levels=levels, colors="k", linewidths=0.35)

    barb = _thin_grid(ds_speed, profile_dim, alt_dim, horizontal_every_n, vertical_every_n)
    barb_altitude = _altitude_values(barb, alt_var, profile_dim, alt_dim)
    barb_u = _curtain_values(barb, u_var, profile_dim, alt_dim)
    barb_v = _curtain_values(barb, v_var, profile_dim, alt_dim)
    ax.barbs(
        np.repeat(barb["along_track_m"].values, barb.sizes[alt_dim]),
        barb_altitude.ravel(),
        _mps_to_knots(barb_u).ravel(),
        _mps_to_knots(barb_v).ravel(),
        color="black",
        alpha=0.75,
        length=barb_length,
        linewidth=0.7,
        pivot="middle",
        sizes={"emptybarb": 0.06},
        barb_increments={"half": 5, "full": 10, "flag": 50},
    )

    _plot_topography(ax, topo, z_min)
    _style_axis(ax, z_min, z_max, "Horizontal wind speed and direction barbs")


def _add_wind_quiver(
    ax,
    ds_track: xr.Dataset,
    alt_var: str,
    w_var: str,
    profile_dim: Hashable,
    alt_dim: Hashable,
    horizontal_every_n: int,
    vertical_every_n: int,
) -> None:
    arrow = _thin_grid(ds_track, profile_dim, alt_dim, horizontal_every_n, vertical_every_n)
    arrow_altitude = _altitude_values(arrow, alt_var, profile_dim, alt_dim)
    arrow_vertical = _curtain_values(arrow, w_var, profile_dim, alt_dim)
    ax.quiver(
        np.repeat(arrow["along_track_m"].values, arrow.sizes[alt_dim]),
        arrow_altitude.ravel(),
        _curtain_values(arrow, "wind_along_track", profile_dim, alt_dim).ravel(),
        arrow_vertical.ravel(),
        color="black",
        alpha=0.7,
        width=0.001,
        scale=200,
    )


def _infer_alt_dim(
    ds: xr.Dataset,
    alt_var: str,
    profile_dim: Hashable,
    alt_dim: Hashable | None = None,
) -> Hashable:
    if alt_dim is not None:
        if alt_dim not in ds.dims:
            raise KeyError(f"Dataset has no altitude dimension {alt_dim!r}.")
        return alt_dim

    candidates = [dim for dim in ds[alt_var].dims if dim != profile_dim]
    if len(candidates) == 1:
        return candidates[0]

    component_dims = []
    for component in ALLOWED_WIND_COMPONENTS:
        if component in ds:
            component_dims.extend(dim for dim in ds[component].dims if dim != profile_dim)
    unique = list(dict.fromkeys(component_dims))
    if len(unique) == 1:
        return unique[0]

    raise ValueError(
        f"Could not infer altitude dimension from {alt_var!r}. "
        "Pass alt_dim explicitly."
    )


def _altitude_values(
    ds: xr.Dataset,
    alt_var: str,
    profile_dim: Hashable,
    alt_dim: Hashable,
) -> np.ndarray:
    data = ds[alt_var]
    if profile_dim in data.dims and alt_dim in data.dims:
        return _curtain_values(ds, alt_var, profile_dim, alt_dim)
    if data.dims == (alt_dim,):
        z = data.values
        return np.tile(z, (ds.sizes[profile_dim], 1))
    raise ValueError(
        f"{alt_var!r} must have dimensions ({profile_dim!r}, {alt_dim!r}) "
        f"or ({alt_dim!r},)."
    )


def _curtain_values(
    ds: xr.Dataset,
    var_name: str,
    profile_dim: Hashable,
    alt_dim: Hashable,
) -> np.ndarray:
    data = ds[var_name]
    if profile_dim not in data.dims or alt_dim not in data.dims:
        raise ValueError(f"{var_name!r} must use {profile_dim!r} and {alt_dim!r}.")
    extra_dims = [dim for dim in data.dims if dim not in {profile_dim, alt_dim}]
    if extra_dims:
        raise ValueError(f"{var_name!r} has extra dimensions {extra_dims}.")
    return data.transpose(profile_dim, alt_dim).values


def _interpolate_to_regular_altitude_grid(
    altitude: np.ndarray,
    values: np.ndarray,
    z_grid: np.ndarray,
) -> np.ndarray:
    curtain = np.full((len(z_grid), altitude.shape[0]), np.nan)
    for idx in range(altitude.shape[0]):
        z = altitude[idx]
        profile_values = values[idx]
        valid = np.isfinite(z) & np.isfinite(profile_values)
        if valid.sum() < 2:
            continue

        z_unique, unique_idx = np.unique(z[valid], return_index=True)
        values_unique = profile_values[valid][unique_idx]
        inside = (z_grid >= z_unique.min()) & (z_grid <= z_unique.max())
        curtain[inside, idx] = np.interp(z_grid[inside], z_unique, values_unique)

    if not np.isfinite(curtain).any():
        raise ValueError("No finite data found inside the selected altitude range.")
    return curtain


def _profile_values(ds: xr.Dataset, var_name: str, profile_dim: Hashable) -> np.ndarray:
    data = ds[var_name]
    if profile_dim not in data.dims:
        raise ValueError(f"{var_name!r} must use profile dimension {profile_dim!r}.")
    extra_dims = [dim for dim in data.dims if dim != profile_dim]
    if extra_dims:
        squeezed = data.squeeze(drop=True)
        if squeezed.dims != (profile_dim,):
            raise ValueError(f"{var_name!r} must be one-dimensional over {profile_dim!r}.")
        data = squeezed
    return np.asarray(data.values)


def _default_wind_component_levels(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-1, 0, 1])
    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs == 0:
        max_abs = 1.0
    max_abs = max(1.0, np.ceil(max_abs))
    return np.linspace(-max_abs, max_abs, 13)


def _horizontal_speed(ds: xr.Dataset, ff_h_var: str, u_var: str, v_var: str):
    if ff_h_var in ds:
        return ds[ff_h_var]
    return xr.apply_ufunc(np.hypot, ds[u_var], ds[v_var])


def _positive_levels(values: np.ndarray, n_levels: int = 13) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0, 1, n_levels)
    max_value = max(1.0, float(np.nanmax(finite)))
    return np.linspace(0, np.ceil(max_value), n_levels)


def _thin_grid(
    ds: xr.Dataset,
    profile_dim: Hashable,
    alt_dim: Hashable,
    horizontal_every_n: int,
    vertical_every_n: int,
) -> xr.Dataset:
    return ds.isel(
        {
            profile_dim: slice(None, None, horizontal_every_n),
            alt_dim: slice(None, None, vertical_every_n),
        }
    )


def _mps_to_knots(values: np.ndarray) -> np.ndarray:
    return values * 1.9438444924406


def _plot_topography(ax, topo: xr.Dataset, z_min: float) -> None:
    ax.plot(
        topo["along_track_m"].values,
        topo["topography_alt"].values,
        color="black",
        linewidth=1,
        alpha=0.5,
        label="Topography",
    )
    ax.fill_between(
        topo["along_track_m"].values,
        topo["topography_alt"].values,
        y2=z_min,
        color="black",
        alpha=0.15,
    )


def _style_axis(ax, z_min: float, z_max: float, title: str) -> None:
    ax.set_xlabel("Distance along selected flight section [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(title)
    ax.set_ylim(z_min, z_max)
    ax.legend(loc="upper right")


def _validate_existing_path(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _validate_step(value: int, name: str) -> None:
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a 2D TEAMx lidar curtain.")
    parser.add_argument("lidar_nc", help="Cut *_pro-style NetCDF file.")
    parser.add_argument("dem_tif", help="DEM GeoTIFF.")
    parser.add_argument("--variable", default="w", choices=ALLOWED_WIND_COMPONENTS)
    parser.add_argument("--group", default=DEFAULT_GROUP, help="NetCDF group; use 'none' for root.")
    parser.add_argument("--output", help="Optional output image path.")
    parser.add_argument("--z-min", type=float, default=500)
    parser.add_argument("--z-max", type=float, default=3200)
    parser.add_argument("--dz", type=float, default=25)
    parser.add_argument("--wind-every-n", type=int, default=2)
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    group = None if str(args.group).lower() == "none" else args.group
    plot_2d(
        args.lidar_nc,
        args.dem_tif,
        variable=args.variable,
        group=group,
        output_path=args.output,
        show=not args.no_show,
        z_min=args.z_min,
        z_max=args.z_max,
        dz=args.dz,
        wind_every_n=args.wind_every_n,
    )


plot_in_2d = plot_2d
plot_from_netcdf = plot_2d


if __name__ == "__main__":
    main()
