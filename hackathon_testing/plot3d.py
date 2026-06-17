"""Self-contained 3D TEAMx lidar curtain plotting.

This file is designed to live directly next to ``app.py`` and ``plot2d.py`` in
the hackathon repository's ``code`` folder.  It expects a cut-down ``*_pro``-
style NetCDF file and keeps all helper functions local to this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.windows import from_bounds

try:
    import pyvista as pv
except ImportError:  # pragma: no cover - handled when plotting is requested
    pv = None


ALLOWED_WIND_COMPONENTS = ("u", "v", "w")
DEFAULT_GROUP = "combined"
DEFAULT_LON_VAR = "AER_lon_avg"
DEFAULT_LAT_VAR = "AER_lat_avg"
DEFAULT_ALT_VAR = "GEO_alt_avg"
DEFAULT_PROFILE_DIM = "nprofile"

WIND_COLOUR_REFERENCE = 4.0
WIND_COLOUR_GREEN = "#1a9850"
WIND_COLOUR_BLUE = "#2166ac"
WIND_COLOUR_WHITE = "#ffffff"
WIND_COLOUR_RED = "#d73027"
WIND_COLOUR_PURPLE = "#762a83"


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
    """Check the minimum fields needed for a 3D curtain plot."""
    variable = validate_wind_component(ds, variable)
    required = [lon_var, lat_var, alt_var, variable]
    missing = [name for name in required if name not in ds and name not in ds.coords]
    if missing:
        raise KeyError(f"Dataset is missing required variables: {missing}")
    if profile_dim not in ds.dims and profile_dim not in ds.coords:
        raise KeyError(f"Dataset has no profile dimension {profile_dim!r}.")
    return variable


def plot_3d(
    lidar_source: str | Path | xr.Dataset,
    dem_path: str | Path,
    satellite_path: str | Path | None = None,
    variable: str = "w",
    group: str | None = DEFAULT_GROUP,
    show: bool = True,
    **kwargs: Any,
):
    """Load a cut NetCDF file and create a 3D topography and wind curtain plot."""
    ds = load_lidar_dataset(lidar_source, group=group)
    return plot_selected_3d(
        ds,
        dem_path,
        satellite_path=satellite_path,
        colour_var=variable,
        show=show,
        **kwargs,
    )


def plot_selected_3d(
    ds: xr.Dataset,
    dem_path: str | Path,
    satellite_path: str | Path | None = None,
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    alt_var: str = DEFAULT_ALT_VAR,
    colour_var: str = "w",
    across_width_m: float = 1000,
    z_min: float = 500,
    z_max: float = 3000,
    dz: float = 25,
    wind_every_n: int = 1,
    wind_scale: float = 50,
    colour_clim: Sequence[float] | None = None,
    colour_cmap: str | Sequence[str] = "bwr",
    show_wind_arrows: bool = True,
    player: str = "window",
    show: bool = True,
):
    """Plot a 3D DEM swath with a vertical wind-component curtain."""
    pv_mod = _require_pyvista()
    u_var = "u"
    v_var = "v"
    w_var = "w"
    profile_dim = DEFAULT_PROFILE_DIM
    alt_dim = None
    track_step = 10
    colour_mode = None
    colour_log = False

    colour_var = validate_dataset_for_plotting(
        ds,
        colour_var,
        lon_var=lon_var,
        lat_var=lat_var,
        alt_var=alt_var,
        profile_dim=profile_dim,
    )
    _validate_existing_path(dem_path, "DEM")
    if satellite_path is not None:
        _validate_existing_path(satellite_path, "Satellite")
    _validate_step(wind_every_n, "wind_every_n")

    topo = sample_topography_from_dem_3d(
        ds,
        dem_path,
        lon_var=lon_var,
        lat_var=lat_var,
        profile_dim=profile_dim,
        across_width_m=across_width_m,
        track_step=track_step,
    )
    topo_e = topo["east_m"].values
    topo_n = topo["north_m"].values
    topo_z = topo["topography_alt"].values
    if not np.isfinite(topo_z).any():
        raise ValueError("No DEM cells found inside the selected flight-track corridor.")

    if satellite_path is not None:
        satellite_texture, texture_coords = satellite_texture_and_coords(
            satellite_path,
            topo["lon"].values,
            topo["lat"].values,
        )
    else:
        satellite_texture = None
        texture_coords = None

    topography = _terrain_surface_from_grid(
        topo_e,
        topo_n,
        topo_z,
        texture_coords=texture_coords,
    )

    east, north = _profile_local_xy(ds, lon_var, lat_var, profile_dim)
    alt_dim = _infer_alt_dim(ds, alt_var, profile_dim, alt_dim)
    z_grid = np.arange(z_min, z_max + dz, dz)
    altitude = _altitude_values(ds, alt_var, profile_dim, alt_dim)
    values = _curtain_values(ds, colour_var, profile_dim, alt_dim)
    curtain_values = _interpolate_to_regular_altitude_grid(altitude, values, z_grid)

    curtain_e, curtain_z = np.meshgrid(east, z_grid)
    curtain_n, _ = np.meshgrid(north, z_grid)
    curtain = pv_mod.StructuredGrid(curtain_e, curtain_n, curtain_z)
    curtain[colour_var] = curtain_values.ravel(order="F")

    colour_cmap, colour_clim = _resolve_colour_options(
        curtain_values,
        colour_var=colour_var,
        colour_clim=colour_clim,
        colour_cmap=colour_cmap,
        colour_mode=colour_mode,
        colour_log=colour_log,
        u_var=u_var,
        v_var=v_var,
        w_var=w_var,
    )

    notebook = player in {"notebook", "trame"}
    plotter = pv_mod.Plotter(notebook=notebook)
    if satellite_path is None:
        plotter.add_mesh(topography, color="lightgrey", smooth_shading=False, opacity=1.0)
    else:
        plotter.add_mesh(
            topography,
            texture=satellite_texture,
            show_scalar_bar=False,
            opacity=1.0,
            smooth_shading=False,
        )

    plotter.add_mesh(
        curtain,
        scalars=colour_var,
        cmap=colour_cmap,
        clim=colour_clim,
        log_scale=colour_log,
        opacity=0.8,
        nan_opacity=0.0,
        lighting=False,
        scalar_bar_args={"label_font_size": 36, "title_font_size": 36},
    )

    if show_wind_arrows and {u_var, v_var, w_var}.issubset(ds.data_vars):
        wind_arrows = _wind_arrow_glyphs(
            ds,
            east=east,
            north=north,
            alt_var=alt_var,
            u_var=u_var,
            v_var=v_var,
            w_var=w_var,
            profile_dim=profile_dim,
            alt_dim=alt_dim,
            wind_every_n=wind_every_n,
            wind_scale=wind_scale,
        )
        if wind_arrows is not None:
            plotter.add_mesh(wind_arrows, color="black", opacity=0.75)

    plotter.add_axes()
    plotter.show_grid(xtitle="east", ytitle="north", ztitle="alt", font_size=18)
    plotter.camera_position = "iso"

    if player == "window":
        plotter.disable_anti_aliasing()
        plotter.disable_depth_peeling()
        plotter.enable_lightkit()
        if plotter.iren is not None:
            plotter.iren.interactor.SetDesiredUpdateRate(30)
            plotter.iren.interactor.SetStillUpdateRate(0.1)

    if show:
        if player == "notebook":
            plotter.show(jupyter_backend="html")
        elif player == "trame":
            plotter.show(jupyter_backend="trame")
        else:
            plotter.show(interactive=True)

    return plotter


def local_utm_crs(lon: np.ndarray, lat: np.ndarray) -> CRS:
    """Choose a local UTM CRS from longitude and latitude values."""
    lon_med = float(np.nanmedian(lon))
    lat_med = float(np.nanmedian(lat))
    zone = int(np.floor((lon_med + 180) / 6) + 1)
    epsg = 32600 + zone if lat_med >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def sample_topography_from_dem_3d(
    ds: xr.Dataset,
    dem_path: str | Path,
    lon_var: str = DEFAULT_LON_VAR,
    lat_var: str = DEFAULT_LAT_VAR,
    profile_dim: Hashable = DEFAULT_PROFILE_DIM,
    across_width_m: float = 1000,
    track_step: int = 10,
) -> xr.Dataset:
    """Clip DEM to a corridor around the flight track without resampling."""
    lon = _profile_values(ds, lon_var, profile_dim)
    lat = _profile_values(ds, lat_var, profile_dim)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        raise ValueError("At least two finite lon/lat profiles are required.")
    lon = lon[valid]
    lat = lat[valid]

    local_crs = local_utm_crs(lon, lat)
    to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    track_x, track_y = to_local.transform(lon, lat)
    track_xy_full = np.column_stack([track_x, track_y])
    track_xy = track_xy_full[::track_step]
    origin = track_xy_full[0]

    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError("DEM has no CRS information.")
        local_bounds = (
            np.nanmin(track_xy_full[:, 0]) - across_width_m,
            np.nanmin(track_xy_full[:, 1]) - across_width_m,
            np.nanmax(track_xy_full[:, 0]) + across_width_m,
            np.nanmax(track_xy_full[:, 1]) + across_width_m,
        )
        corners_x = [local_bounds[0], local_bounds[0], local_bounds[2], local_bounds[2]]
        corners_y = [local_bounds[1], local_bounds[3], local_bounds[1], local_bounds[3]]
        to_dem = Transformer.from_crs(local_crs, dem.crs, always_xy=True)
        dem_x, dem_y = to_dem.transform(corners_x, corners_y)
        window = from_bounds(
            min(dem_x),
            min(dem_y),
            max(dem_x),
            max(dem_y),
            dem.transform,
        ).round_offsets().round_lengths()

        z = dem.read(1, window=window).astype(float)
        transform = dem.window_transform(window)
        rows, cols = np.indices(z.shape)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        xs = np.asarray(xs).reshape(z.shape)
        ys = np.asarray(ys).reshape(z.shape)

        to_local_from_dem = Transformer.from_crs(dem.crs, local_crs, always_xy=True)
        local_x, local_y = to_local_from_dem.transform(xs, ys)
        to_lonlat = Transformer.from_crs(dem.crs, "EPSG:4326", always_xy=True)
        grid_lon, grid_lat = to_lonlat.transform(xs, ys)

        if dem.nodata is not None:
            z = np.where(z == dem.nodata, np.nan, z)

    distance = _distance_to_track_m(local_x, local_y, track_xy)
    z = np.where(distance <= across_width_m, z, np.nan)

    return xr.Dataset(
        data_vars={
            "lon": (("y", "x"), grid_lon),
            "lat": (("y", "x"), grid_lat),
            "east_m": (("y", "x"), local_x - origin[0]),
            "north_m": (("y", "x"), local_y - origin[1]),
            "topography_alt": (("y", "x"), z),
            "distance_to_track_m": (("y", "x"), distance),
        },
        coords={"y": np.arange(z.shape[0]), "x": np.arange(z.shape[1])},
    )


def satellite_texture_and_coords(
    satellite_path: str | Path,
    lon: np.ndarray,
    lat: np.ndarray,
):
    """Create a PyVista texture and texture coordinates for lon/lat points."""
    pv_mod = _require_pyvista()
    with rasterio.open(satellite_path) as src:
        if src.crs is None:
            raise ValueError("Satellite raster has no CRS information.")
        bands = [1, 2, 3] if src.count >= 3 else [1]
        image = np.moveaxis(src.read(bands), 0, -1)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        image = np.clip(image[:, :, :3] * 1.3, 0, 255).astype(np.uint8)

        if src.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(lon, lat)
        else:
            xs, ys = lon, lat

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        texture_x = (xs - src.bounds.left) / (src.bounds.right - src.bounds.left)
        texture_y = (ys - src.bounds.bottom) / (src.bounds.top - src.bounds.bottom)
        texture_coords = np.column_stack([texture_x.ravel(), texture_y.ravel()]).reshape(
            lon.shape + (2,)
        )

    return pv_mod.Texture(image), texture_coords


def _terrain_surface_from_grid(
    east: np.ndarray,
    north: np.ndarray,
    altitude: np.ndarray,
    texture_coords: np.ndarray | None = None,
):
    pv_mod = _require_pyvista()
    valid = np.isfinite(altitude)
    valid_cells = (
        valid[:-1, :-1]
        & valid[1:, :-1]
        & valid[1:, 1:]
        & valid[:-1, 1:]
    )

    used = np.zeros(valid.shape, dtype=bool)
    rows, cols = np.where(valid_cells)
    used[rows, cols] = True
    used[rows + 1, cols] = True
    used[rows + 1, cols + 1] = True
    used[rows, cols + 1] = True

    point_ids = np.full(valid.shape, -1, dtype=int)
    point_ids[used] = np.arange(used.sum())
    points = np.column_stack([east[used], north[used], altitude[used]])

    faces = []
    for row, col in zip(rows, cols):
        faces.extend(
            [
                4,
                point_ids[row, col],
                point_ids[row + 1, col],
                point_ids[row + 1, col + 1],
                point_ids[row, col + 1],
            ]
        )

    surface = pv_mod.PolyData(points, np.asarray(faces))
    surface["topography_alt"] = altitude[used]
    if texture_coords is not None:
        surface.active_texture_coordinates = texture_coords[used]
    return surface


def _wind_arrow_glyphs(
    ds: xr.Dataset,
    east: np.ndarray,
    north: np.ndarray,
    alt_var: str,
    u_var: str,
    v_var: str,
    w_var: str,
    profile_dim: Hashable,
    alt_dim: Hashable,
    wind_every_n: int,
    wind_scale: float,
):
    pv_mod = _require_pyvista()
    wind = ds.isel({profile_dim: slice(None, None, wind_every_n)})
    wind_east = east[::wind_every_n]
    wind_north = north[::wind_every_n]
    wind_altitude = _altitude_values(wind, alt_var, profile_dim, alt_dim)
    wind_u = _curtain_values(wind, u_var, profile_dim, alt_dim)
    wind_v = _curtain_values(wind, v_var, profile_dim, alt_dim)
    wind_w = _curtain_values(wind, w_var, profile_dim, alt_dim)

    wind_points = np.column_stack(
        [
            np.repeat(wind_east, wind.sizes[alt_dim]),
            np.repeat(wind_north, wind.sizes[alt_dim]),
            wind_altitude.ravel(),
        ]
    )
    wind_vectors = np.column_stack([wind_u.ravel(), wind_v.ravel(), wind_w.ravel()])
    valid_wind = np.isfinite(wind_points).all(axis=1) & np.isfinite(wind_vectors).all(axis=1)
    if not valid_wind.any():
        return None

    wind_cloud = pv_mod.PolyData(wind_points[valid_wind])
    wind_cloud["wind"] = wind_vectors[valid_wind]
    wind_cloud["wind_speed"] = np.linalg.norm(wind_vectors[valid_wind], axis=1)
    return wind_cloud.glyph(
        orient="wind",
        scale="wind_speed",
        factor=wind_scale,
        geom=pv_mod.Arrow(tip_length=0.25, tip_radius=0.08, shaft_radius=0.03),
    )


def _profile_local_xy(
    ds: xr.Dataset,
    lon_var: str,
    lat_var: str,
    profile_dim: Hashable,
) -> tuple[np.ndarray, np.ndarray]:
    lon = _profile_values(ds, lon_var, profile_dim)
    lat = _profile_values(ds, lat_var, profile_dim)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        raise ValueError("At least two finite lon/lat profiles are required.")

    crs = local_utm_crs(lon[valid], lat[valid])
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    origin_x, origin_y = transformer.transform(lon[valid][0], lat[valid][0])
    return np.asarray(x) - origin_x, np.asarray(y) - origin_y


def _local_transform_from_profile(
    ds: xr.Dataset,
    lon_var: str,
    lat_var: str,
) -> tuple[Transformer, float, float]:
    lon = np.asarray(ds[lon_var].values)
    lat = np.asarray(ds[lat_var].values)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 1:
        raise ValueError("At least one finite lon/lat profile is required.")

    crs = local_utm_crs(lon[valid], lat[valid])
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    origin_x, origin_y = transformer.transform(lon[valid][0], lat[valid][0])
    return transformer, origin_x, origin_y


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
        raise ValueError(
            f"{var_name!r} must have dimensions {profile_dim!r} and {alt_dim!r} "
            "to be plotted as a vertical curtain."
        )
    extra_dims = [dim for dim in data.dims if dim not in {profile_dim, alt_dim}]
    if extra_dims:
        raise ValueError(
            f"{var_name!r} has extra dimensions {extra_dims}. "
            "Select one slice before calling plot_selected_3d."
        )
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


def _distance_to_track_m(x: np.ndarray, y: np.ndarray, track_xy: np.ndarray) -> np.ndarray:
    points = np.column_stack([x.ravel(), y.ravel()])
    min_dist2 = np.full(len(points), np.inf)
    for start, end in zip(track_xy[:-1], track_xy[1:]):
        segment = end - start
        segment_len2 = np.dot(segment, segment)
        if segment_len2 == 0:
            continue
        t = ((points - start) @ segment) / segment_len2
        t = np.clip(t, 0, 1)
        closest = start + t[:, None] * segment
        dist2 = np.sum((points - closest) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return np.sqrt(min_dist2).reshape(x.shape)


def _resolve_colour_options(
    curtain_values: np.ndarray,
    colour_var: str,
    colour_clim: Sequence[float] | None,
    colour_cmap: str | Sequence[str],
    colour_mode: str | None,
    colour_log: bool,
    u_var: str,
    v_var: str,
    w_var: str,
) -> tuple[str | Sequence[str], list[float]]:
    is_wind_component = colour_var in {u_var, v_var, w_var, "u", "v", "w"}
    use_centered_wind_colours = (
        is_wind_component
        and not colour_log
        and colour_mode != "linear"
    )
    use_rb_colour_map = use_centered_wind_colours and colour_mode in {None, "rb"}
    finite_values = curtain_values[np.isfinite(curtain_values)]
    if finite_values.size == 0:
        raise ValueError(f"No finite values found for colour_var={colour_var!r}.")

    if colour_clim is None:
        if use_centered_wind_colours:
            colour_abs_max = float(np.nanmax(np.abs(finite_values)))
            if colour_abs_max == 0:
                colour_abs_max = 1.0
            colour_clim = [-colour_abs_max, colour_abs_max]
        else:
            colour_clim = [float(finite_values.min()), float(finite_values.max())]
    else:
        colour_clim = [float(colour_clim[0]), float(colour_clim[1])]

    if use_centered_wind_colours:
        colour_abs_max = float(np.nanmax(np.abs(colour_clim)))
        if colour_abs_max == 0:
            colour_abs_max = 1.0
        colour_clim = [-colour_abs_max, colour_abs_max]
        if use_rb_colour_map:
            colour_cmap = _wind_component_colour_map(colour_abs_max)

    if colour_log:
        finite_positive = curtain_values[np.isfinite(curtain_values) & (curtain_values > 0)]
        if finite_positive.size == 0:
            raise ValueError(f"Log colour scale needs positive values for colour_var={colour_var!r}.")
        colour_clim = [
            max(float(colour_clim[0]), float(finite_positive.min())),
            float(colour_clim[1]),
        ]

    return colour_cmap, colour_clim


def _wind_component_colour_map(colour_abs_max: float, n_colours: int = 256) -> list[str]:
    values = np.linspace(-colour_abs_max, colour_abs_max, n_colours)
    colours = []
    for value in values:
        if value < -WIND_COLOUR_REFERENCE:
            colour = _interpolate_colour(
                value,
                -colour_abs_max,
                WIND_COLOUR_GREEN,
                -WIND_COLOUR_REFERENCE,
                WIND_COLOUR_BLUE,
            )
        elif value < 0:
            colour = _interpolate_colour(
                value,
                -WIND_COLOUR_REFERENCE,
                WIND_COLOUR_BLUE,
                0,
                WIND_COLOUR_WHITE,
            )
        elif value <= WIND_COLOUR_REFERENCE:
            colour = _interpolate_colour(
                value,
                0,
                WIND_COLOUR_WHITE,
                WIND_COLOUR_REFERENCE,
                WIND_COLOUR_RED,
            )
        else:
            colour = _interpolate_colour(
                value,
                WIND_COLOUR_REFERENCE,
                WIND_COLOUR_RED,
                colour_abs_max,
                WIND_COLOUR_PURPLE,
            )
        colours.append(colour)
    return colours


def _interpolate_colour(
    value: float,
    left_value: float,
    left_colour: str,
    right_value: float,
    right_colour: str,
) -> str:
    if right_value == left_value:
        return right_colour
    weight = (value - left_value) / (right_value - left_value)
    rgb = (1 - weight) * _hex_to_rgb(left_colour) + weight * _hex_to_rgb(right_colour)
    return _rgb_to_hex(rgb)


def _hex_to_rgb(colour: str) -> np.ndarray:
    colour = colour.lstrip("#")
    return np.asarray([int(colour[idx:idx + 2], 16) for idx in (0, 2, 4)], dtype=float)


def _rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = np.clip(np.round(rgb), 0, 255).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _require_pyvista():
    if pv is None:
        raise ImportError(
            "pyvista is required for 3D plotting. Install it with the project "
            "requirements before using plot3d.py."
        )
    return pv


def _validate_existing_path(path: str | Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def _validate_step(value: int, name: str) -> None:
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a 3D TEAMx lidar curtain.")
    parser.add_argument("lidar_nc", help="Cut *_pro-style NetCDF file.")
    parser.add_argument("dem_tif", help="DEM GeoTIFF.")
    parser.add_argument("--satellite", help="Optional satellite GeoTIFF texture.")
    parser.add_argument("--variable", default="w", choices=ALLOWED_WIND_COMPONENTS)
    parser.add_argument("--group", default=DEFAULT_GROUP, help="NetCDF group; use 'none' for root.")
    parser.add_argument("--z-min", type=float, default=500)
    parser.add_argument("--z-max", type=float, default=3000)
    parser.add_argument("--dz", type=float, default=25)
    parser.add_argument("--across-width-m", type=float, default=1000)
    parser.add_argument("--wind-every-n", type=int, default=1)
    parser.add_argument("--wind-scale", type=float, default=50)
    parser.add_argument("--player", default="window", choices=["window", "notebook", "trame"])
    parser.add_argument("--no-show", action="store_true", help="Build but do not open the plot.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    group = None if str(args.group).lower() == "none" else args.group
    plot_3d(
        args.lidar_nc,
        args.dem_tif,
        satellite_path=args.satellite,
        variable=args.variable,
        group=group,
        show=not args.no_show,
        z_min=args.z_min,
        z_max=args.z_max,
        dz=args.dz,
        across_width_m=args.across_width_m,
        wind_every_n=args.wind_every_n,
        wind_scale=args.wind_scale,
        player=args.player,
    )


plot_in_3d = plot_3d
plot_from_netcdf = plot_3d
plot_curtain_3d = plot_selected_3d


if __name__ == "__main__":
    main()
