"""2-panel TEAMx lidar curtain plotting."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer


DEFAULT_LON_VAR = "AER_lon_avg"
DEFAULT_LAT_VAR = "AER_lat_avg"
DEFAULT_ALT_VAR = "GEO_alt_avg"
DEFAULT_PROFILE_DIM = "nprofile"


def open_lidar_file(lidar_file: str | Path, group: str | None = None) -> xr.Dataset:
    """Open a TEAMx NetCDF file from the root group or a named group."""
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
    group: str | None = None,
) -> xr.Dataset:
    """Load a NetCDF path or return an already opened dataset."""
    if isinstance(lidar_source, xr.Dataset):
        return lidar_source

    ds = open_lidar_file(lidar_source, group=group)
    try:
        return ds.load()
    finally:
        ds.close()


def plot_wind_curtain_panels(
    lidar_source: str | Path | xr.Dataset,
    dem_path: str | Path | None = None,
    group: str | None = None,
    alt_var: str = DEFAULT_ALT_VAR,
    topography_var: str = "TOP_avg",
    time_var: str | None = None,
    first_panel_var: str = "w",
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
    vertical_every_n: int = 2,
    wind_every_n: int = 3,
    barb_length: float = 5.5,
    use_dem_topography: bool = True,
    output_path: str | Path | None = None,
    show: bool = False,
):
    """Create the two stacked 2D curtain panels from a cut TEAMx NetCDF file."""
    ds = load_lidar_dataset(lidar_source, group=group)
    _validate_step(vertical_every_n, "vertical_every_n")
    _validate_step(wind_every_n, "wind_every_n")

    for name in [lon_var, lat_var, alt_var, topography_var, u_var, v_var, w_var]:
        if name not in ds and name not in ds.coords:
            raise KeyError(f"Dataset is missing required variable {name!r}.")

    ds_track, track_info = rotate_coordinate_system(
        ds,
        lon_var=lon_var,
        lat_var=lat_var,
        profile_dim=profile_dim,
    )
    ds_track = rotate_wind_into_rotated_coordinate_system(
        ds_track,
        track_info,
        u_var=u_var,
        v_var=v_var,
    )
    if first_panel_var not in ds_track and first_panel_var not in ds_track.coords:
        raise KeyError(f"Dataset is missing first-panel variable {first_panel_var!r}.")
    alt_dim = _infer_alt_dim(ds_track, alt_var, profile_dim, alt_dim)

    if use_dem_topography and dem_path is not None and Path(dem_path).exists():
        topo = sample_topography_from_dem_2d(
            ds_track,
            dem_path,
            lon_var=lon_var,
            lat_var=lat_var,
            profile_dim=profile_dim,
            every_n=1,
        )
        topo = topo.assign(along_track_m=ds_track["along_track_m"])
    else:
        topo_data = ds_track[topography_var]
        if profile_dim not in topo_data.dims:
            raise ValueError(f"{topography_var!r} must use profile dimension {profile_dim!r}.")
        topo_values = topo_data.transpose(profile_dim, ...).values
        topo_values = topo_values.reshape(ds_track.sizes[profile_dim], -1)
        topo_values = np.array(
            [
                row[np.isfinite(row)][0] if np.isfinite(row).any() else np.nan
                for row in topo_values
            ]
        )

        topo = xr.Dataset(
            data_vars={
                "topography_alt": (
                    profile_dim,
                    topo_values,
                ),
                lon_var: (
                    profile_dim,
                    _profile_values(ds_track, lon_var, profile_dim),
                ),
                lat_var: (
                    profile_dim,
                    _profile_values(ds_track, lat_var, profile_dim),
                ),
                "along_track_m": (
                    profile_dim,
                    ds_track["along_track_m"].values,
                ),
            },
            coords={
                profile_dim: (
                    ds_track[profile_dim].values
                    if profile_dim in ds_track.coords
                    else np.arange(ds_track.sizes[profile_dim])
                )
            },
        )
    if time_var is None:
        for candidate in ("time", "AER_posixtime", "posixtime", "datenum", "datenumber"):
            if candidate in ds_track or candidate in ds_track.coords:
                time_var = candidate
                break
    if time_var is None or (time_var not in ds_track and time_var not in ds_track.coords):
        raise KeyError(
            "Dataset is missing a sampling-time variable. "
            "Tried: time, AER_posixtime, posixtime, datenum, datenumber."
        )

    time_data = ds_track[time_var]
    if profile_dim not in time_data.dims:
        raise ValueError(f"{time_var!r} must use profile dimension {profile_dim!r}.")
    time_values = time_data.transpose(profile_dim, ...).values
    time_values = time_values.reshape(ds_track.sizes[profile_dim], -1)
    if np.issubdtype(time_values.dtype, np.datetime64):
        sampling_time = np.array(
            [
                row[~np.isnat(row)][0] if (~np.isnat(row)).any() else np.datetime64("NaT")
                for row in time_values
            ],
            dtype="datetime64[ns]",
        )
    else:
        raw_time = np.array(
            [
                row[np.isfinite(row)][0] if np.isfinite(row).any() else np.nan
                for row in time_values.astype(float)
            ]
        )
        lower_time_var = str(time_var).lower()
        if "datenum" in lower_time_var:
            sampling_datetime = raw_time - 719163
        else:
            finite_time = raw_time[np.isfinite(raw_time)]
            max_time = np.nanmax(np.abs(finite_time)) if finite_time.size else 0
            if max_time > 1e17:
                sampling_time = raw_time.astype("datetime64[ns]")
            elif max_time > 1e14:
                sampling_time = raw_time.astype("datetime64[us]")
            elif max_time > 1e11:
                sampling_time = raw_time.astype("datetime64[ms]")
            else:
                sampling_time = raw_time.astype("datetime64[s]")
            sampling_datetime = mdates.date2num(
                sampling_time.astype("datetime64[ms]").astype(object)
            )
    if np.issubdtype(time_values.dtype, np.datetime64):
        sampling_datetime = mdates.date2num(
            sampling_time.astype("datetime64[ms]").astype(object)
        )
    ds_track = ds_track.assign(sampling_datetime=(profile_dim, sampling_datetime))
    topo = topo.assign(sampling_datetime=(profile_dim, sampling_datetime))
    z_grid = np.arange(z_min, z_max + 25, 25)
    width_multiplier = max(1, int(np.ceil(ds_track.sizes[profile_dim] / 200)))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(24 * width_multiplier, 13 * width_multiplier),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    panel_context = {
        "fig": fig,
        "ds_track": ds_track,
        "topo": topo,
        "alt_var": alt_var,
        "first_panel_var": first_panel_var,
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
        "horizontal_every_n": wind_every_n,
        "barb_length": barb_length,
    }

    _plot_projected_wind_panel(ax=axes[0], **panel_context)
    _plot_horizontal_wind_panel(ax=axes[1], **panel_context)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def local_utm_crs(lon: np.ndarray, lat: np.ndarray) -> CRS:
    """Choose the local UTM coordinate system for the flight track."""
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    zone = int(np.floor((lon_med + 180) / 6) + 1)
    epsg = 32600 + zone if lat_med >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def rotate_coordinate_system(
    ds: xr.Dataset,
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
):
    """Convert lon/lat into along-track and cross-track meter coordinates."""
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


def rotate_wind_into_rotated_coordinate_system(
    ds: xr.Dataset,
    track_info: dict[str, Any],
    u_var: str = "u",
    v_var: str = "v",
) -> xr.Dataset:
    """Project u/v wind onto the along-track and cross-track axes."""
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
    """Read topography from a DEM at each flight-track point."""
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
    """Return DEM heights at lon/lat sample points."""
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
    first_panel_var,
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
    """Draw the first panel: curtain colour plus projected wind arrows."""
    along = np.arange(ds_track.sizes[profile_dim])
    altitude = _altitude_values(ds_track, alt_var, profile_dim, alt_dim)
    values = _curtain_values(ds_track, first_panel_var, profile_dim, alt_dim)
    curtain = _interpolate_to_regular_altitude_grid(altitude, values, z_grid)
    x_grid, z_mesh = np.meshgrid(along, z_grid)
    levels = _default_wind_component_levels(curtain)

    filled = ax.contourf(
        x_grid,
        z_mesh,
        curtain,
        levels=levels,
        cmap="bwr",
        extend="both",
    )
    colorbar = fig.colorbar(
        filled,
        ax=ax,
        label=f"{first_panel_var} [m s$^{{-1}}$]",
        format="%.1f",
    )
    colorbar.ax.tick_params(labelsize=34)
    colorbar.set_label(f"{first_panel_var} [m s$^{{-1}}$]", fontsize=34)
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
    _style_axis(
        ax,
        z_min,
        z_max,
        f"Wind projected along curtain, coloured by {first_panel_var}",
        ds_track=ds_track,
    )


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
    """Draw the second panel: horizontal speed colours plus wind barbs."""
    ds_speed = ds_track.assign(
        _horizontal_speed=_horizontal_speed(ds_track, ff_h_var, u_var, v_var)
    )
    along = np.arange(ds_speed.sizes[profile_dim])
    altitude = _altitude_values(ds_speed, alt_var, profile_dim, alt_dim)
    speed = _curtain_values(ds_speed, "_horizontal_speed", profile_dim, alt_dim)
    curtain = _interpolate_to_regular_altitude_grid(altitude, speed, z_grid)
    x_grid, z_mesh = np.meshgrid(along, z_grid)
    levels = np.round(_positive_levels(curtain), 1)

    filled = ax.contourf(
        x_grid,
        z_mesh,
        curtain,
        levels=levels,
        cmap="YlGnBu",
        extend="max",
        alpha=0.8,
    )
    colorbar = fig.colorbar(
        filled,
        ax=ax,
        label="Horizontal wind speed [m s$^{-1}$]",
        format="%.1f",
    )
    colorbar.ax.tick_params(labelsize=34)
    colorbar.set_label("Horizontal wind speed [m s$^{-1}$]", fontsize=34)
    ax.contour(x_grid, z_mesh, curtain, levels=levels, colors="k", linewidths=0.35)

    barb = _thin_grid(ds_speed, profile_dim, alt_dim, horizontal_every_n, vertical_every_n)
    barb_altitude = _altitude_values(barb, alt_var, profile_dim, alt_dim)
    barb_u = _curtain_values(barb, u_var, profile_dim, alt_dim)
    barb_v = _curtain_values(barb, v_var, profile_dim, alt_dim)
    ax.barbs(
        np.repeat(
            np.arange(0, ds_speed.sizes[profile_dim], horizontal_every_n),
            barb.sizes[alt_dim],
        ),
        barb_altitude.ravel(),
        _mps_to_knots(barb_u).ravel(),
        _mps_to_knots(barb_v).ravel(),
        color="black",
        alpha=1,
        length=barb_length * 1.3,
        linewidth=1,
        pivot="middle",
        sizes={"emptybarb": 0.06},
        barb_increments={"half": 5, "full": 10, "flag": 50},
    )

    _plot_topography(ax, topo, z_min)
    _style_axis(
        ax,
        z_min,
        z_max,
        "Horizontal wind speed and direction, meteorological convention",
        ds_track=ds_speed,
    )


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
    """Add along-track/vertical wind arrows to the first panel."""
    arrow = _thin_grid(ds_track, profile_dim, alt_dim, horizontal_every_n, vertical_every_n)
    arrow_altitude = _altitude_values(arrow, alt_var, profile_dim, alt_dim)
    arrow_vertical = _curtain_values(arrow, w_var, profile_dim, alt_dim)
    ax.quiver(
        np.repeat(
            np.arange(0, ds_track.sizes[profile_dim], horizontal_every_n),
            arrow.sizes[alt_dim],
        ),
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
    """Find the non-profile dimension used as altitude."""
    if alt_dim is not None:
        if alt_dim not in ds.dims:
            raise KeyError(f"Dataset has no altitude dimension {alt_dim!r}.")
        return alt_dim

    candidates = [dim for dim in ds[alt_var].dims if dim != profile_dim]
    if len(candidates) == 1:
        return candidates[0]

    component_dims = []
    for component in ("u", "v", "w"):
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
    """Return altitude as a profile-by-altitude array."""
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
    """Return a 2D variable as profile-by-altitude values."""
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
    """Interpolate each profile onto a regular altitude grid."""
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
    """Return one value per profile for a profile-coordinate variable."""
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
    """Build symmetric colour levels around zero."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-1, 0, 1])
    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs == 0:
        max_abs = 1.0
    max_abs = max(1.0, np.ceil(max_abs))
    return np.linspace(-max_abs, max_abs, 13)


def _horizontal_speed(ds: xr.Dataset, ff_h_var: str, u_var: str, v_var: str):
    """Use stored horizontal speed or compute it from u/v."""
    if ff_h_var in ds:
        return ds[ff_h_var]
    return xr.apply_ufunc(np.hypot, ds[u_var], ds[v_var])


def _positive_levels(values: np.ndarray, n_levels: int = 13) -> np.ndarray:
    """Build positive colour levels for wind speed."""
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
    """Thin profiles and altitude levels for arrows/barbs."""
    return ds.isel(
        {
            profile_dim: slice(None, None, horizontal_every_n),
            alt_dim: slice(None, None, vertical_every_n),
        }
    )


def _mps_to_knots(values: np.ndarray) -> np.ndarray:
    """Convert metres per second to knots for Matplotlib wind barbs."""
    return values * 1.9438444924406


def _plot_topography(ax, topo: xr.Dataset, z_min: float) -> None:
    """Draw the topography line and shaded ground."""
    topo_dim = topo["topography_alt"].dims[0]
    x_values = np.arange(topo.sizes[topo_dim])
    ax.plot(
        x_values,
        topo["topography_alt"].values,
        color="black",
        linewidth=1,
        alpha=0.5,
    )
    ax.fill_between(
        x_values,
        topo["topography_alt"].values,
        y2=z_min,
        color="black",
        alpha=0.15,
    )


def _style_axis(
    ax,
    z_min: float,
    z_max: float,
    title: str,
    ds_track: xr.Dataset | None = None,
) -> None:
    """Apply titles, labels, and combined time/distance tick labels."""
    ax.set_xlabel("")
    ax.set_ylabel("Altitude [m]", fontsize=28)
    ax.set_title(title, fontsize=34)
    ax.set_ylim(z_min, z_max)
    ax.tick_params(axis="both", labelsize=25)
    ax.tick_params(axis="x", labelbottom=True)
    if (
        ds_track is not None
        and "sampling_datetime" in ds_track
        and "along_track_m" in ds_track
    ):
        times = ds_track["sampling_datetime"].values
        distance_km = ds_track["along_track_m"].values / 1000

        def tick_label(value, _):
            idx = int(round(value))
            if idx < 0 or idx >= len(times):
                return ""
            time_text = mdates.num2date(times[idx]).strftime("%H:%M:%S")
            return f"{time_text}\n{distance_km[idx]:.1f} km"

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(tick_label))


def _validate_step(value: int, name: str) -> None:
    """Require a positive integer thinning step."""
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
