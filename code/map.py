"""Panel/HoloViews app for interactively selecting and cutting flight tracks."""

from __future__ import annotations

import argparse
import configparser
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from core import cut_dataset, save_dataset, extract_track, thin_track
from selection import nearest_track_point


class FlightMapApp:
    """Interactive 2D flight-track selector and dataset cutter."""

    def __init__(
        self,
        ds: xr.Dataset,
        lat: str = "lat",
        lon: str = "lon",
        profile_dim: Hashable = "nprofile",
        time: str | None = "time",
        hover_columns: Sequence[str] | None = None,
        thinning_step: int = 1,
        basemap: bool = True,
        output_dir: str | Path = "data/processed",
        dem_path: str | Path | None = None,
        satellite_path: str | Path | None = None,
        results_dir: str | Path = "results",
        plot2d_options: dict[str, Any] | None = None,
    ) -> None:
        self.ds = ds
        self.lat = lat
        self.lon = lon
        self.profile_dim = profile_dim
        self.time = time
        self.hover_columns = list(hover_columns or [])
        self.thinning_step = thinning_step
        self.basemap = basemap
        self.output_dir = Path(output_dir)
        self.dem_path = Path(dem_path) if dem_path is not None else None
        self.satellite_path = Path(satellite_path) if satellite_path is not None else None
        self.results_dir = Path(results_dir)
        self.plot2d_options = plot2d_options or {}

        self.track = extract_track(ds, lat=lat, lon=lon, profile_dim=profile_dim, time=time)
        self.track = self.track.dropna(subset=["lat", "lon", "profile"]).reset_index(drop=True)
        self._add_hover_metadata()
        self.track_plot = thin_track(self.track, step=thinning_step)
        self.track_plot["track_order"] = range(len(self.track_plot))
        self.start_profile: Any | None = None
        self.end_profile: Any | None = None
        self.ds_cut: xr.Dataset | None = None
        self.save_status: bool | None = None
        self.plot2d_status: str | None = None
        self._selection_mode: str | None = None

        self._panel = None
        self._status = None
        self._select_start_button = None
        self._select_end_button = None
        self._cut_button = None
        self._save_button = None
        self._plot2d_button = None
        self._plot3d_button = None
        self._plot2d_output = None
        self._map_pane = None
        self._tap_stream = None

    def _add_hover_metadata(self) -> None:
        for column in self.hover_columns:
            if column in self.track.columns:
                continue
            if column in self.ds:
                values = self.ds[column].values
            elif column in self.ds.coords:
                values = self.ds.coords[column].values
            else:
                continue

            array = values
            if getattr(array, "ndim", None) == 1:
                series = pd.Series(array)
                if len(series) == len(self.track):
                    self.track[column] = series.to_numpy()

    @property
    def panel(self):
        """Panel layout for display in notebooks or apps."""
        if self._panel is None:
            self._panel = self._build_panel()
        return self._panel

    def show(self):
        """Return the Panel layout for display."""
        return self.panel

    def cut(self) -> xr.Dataset | None:
        """Return the selected cut dataset, creating it if both picks exist."""
        if self.start_profile is None or self.end_profile is None:
            return self.ds_cut
        self.ds_cut = cut_dataset(
            self.ds,
            self.start_profile,
            self.end_profile,
            profile_dim=self.profile_dim,
        )
        return self.ds_cut

    def _cut_filename(self) -> str:
        """Return the processed filename for the current profile selection."""
        if self.start_profile is None or self.end_profile is None:
            raise ValueError("Select start and end profiles before saving.")

        start, end = sorted((self.start_profile, self.end_profile))
        return f"cut_flight_track_{start:g}_{end:g}.nc"

    def _save_cut_dataset(self, ds_cut: xr.Dataset) -> Path:
        return save_dataset(
            ds_cut,
            self.output_dir,
            filename=self._cut_filename(),
        )

    def save(self) -> xr.Dataset | None:
        """Save the selected cut dataset, creating if it has been cut."""
        if self.ds_cut is None:
            raise Exception("Dataset needs to be cut before saving.") 
        self.save_status = self._save_cut_dataset(self.ds_cut)
        return self.save_status

    
    def _build_panel(self):
        try:
            import holoviews as hv
            import hvplot.pandas  # noqa: F401
            import panel as pn
        except ImportError as exc:
            missing = exc.name or "one of the interactive plotting packages"
            raise ImportError(
                "FlightMapApp.show() needs the interactive map dependencies. "
                "Install the packages listed in TEAMx_prototype/requirements.txt "
                f"and try again. Missing package: {missing!r}."
            ) from exc

        pn.extension("bokeh")
        hv.extension("bokeh")

        plot = self._make_track_plot()
        self._tap_stream = hv.streams.Tap(source=plot, x=None, y=None)
        self._tap_stream.add_subscriber(self._handle_tap)
        self._map_pane = pn.pane.HoloViews(plot)

        self._status = pn.pane.Markdown(self._status_text())
        self._select_start_button = pn.widgets.Button(
            name="Select starting point",
            button_type="default",
        )
        self._select_end_button = pn.widgets.Button(
            name="Select end point",
            button_type="default",
        )
        self._cut_button = pn.widgets.Button(name="Cut dataset", button_type="primary")
        self._save_button = pn.widgets.Button(name="Save dataset", button_type="primary")
        self._plot2d_button = pn.widgets.Button(name="Plot 2D", button_type="success")
        self._plot3d_button = pn.widgets.Button(name="Plot 3D", button_type="success")

        self._select_start_button.on_click(self._handle_select_start_button)
        self._select_end_button.on_click(self._handle_select_end_button)
        self._cut_button.on_click(self._handle_cut_button)
        self._save_button.on_click(self._handle_save_button)
        self._plot3d_button.on_click(self._handle_plot3d_button)
        self._plot2d_output = pn.bind(
            self._render_plot2d,
            clicks=self._plot2d_button.param.clicks,
        )

        header = pn.pane.Markdown(
            "### TEAMx flight-track selector\n"
            "Select a starting point, click the track, select an end point, click the track again, "
            "then cut the dataset. Use `Plot 2D` to render and save the cut section below.",
            width=850,
        )

        return pn.Column(
            header,
            self._map_pane,
            pn.Row(
                self._select_start_button,
                self._select_end_button,
                self._cut_button,
                self._save_button,
                self._plot2d_button,
                self._plot3d_button,
            ),
            self._status,
            self._plot2d_output,
            sizing_mode="stretch_width",
        )

    def _make_track_plot(self):
        """Create the selector map."""
        hover_cols = ["profile", "lat", "lon"]
        if "time" in self.track_plot.columns:
            hover_cols.append("time")
        hover_cols.extend(col for col in self.hover_columns if col in self.track_plot.columns)

        return self.track_plot.hvplot.points(
            x="lon",
            y="lat",
            geo=True,
            tiles="OSM" if self.basemap else None,
            hover_cols=hover_cols + ["track_order"],
            size=40,
            c="track_order",
            cmap="viridis",
            colorbar=True,
            frame_width=850,
            frame_height=550,
            title="Flight track selector",
            tools=["hover", "tap"],
        )

    def _handle_tap(self, x: float | None = None, y: float | None = None) -> None:
        if x is None or y is None:
            return
        if self._selection_mode is None:
            self._refresh_status()
            return

        lon, lat = self._tap_to_lon_lat(x, y)
        selected = nearest_track_point(lon, lat, self.track_plot)
        profile = selected["profile"]

        if self._selection_mode == "start":
            self.start_profile = profile
            self.ds_cut = None
        elif self._selection_mode == "end":
            self.end_profile = profile
            self.ds_cut = None

        self._selection_mode = None
        self._refresh_status()

    def _handle_select_start_button(self, _event) -> None:
        self._selection_mode = "start"
        self._refresh_status()

    def _handle_select_end_button(self, _event) -> None:
        self._selection_mode = "end"
        self._refresh_status()

    def _handle_cut_button(self, _event) -> None:
        self.cut()
        self._refresh_status()
        
    def _handle_save_button(self, _event) -> None:
        self.save()
        self._refresh_status()

    def _plot2d_from_dataset(self, ds_cut: xr.Dataset):
        """Save and render the current cut dataset as the 2D panel plot."""
        import panel as pn
        import plot2d

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.results_dir / "plot2d_panels.png"
        self._save_cut_dataset(ds_cut)

        fig, _axes = plot2d.plot_wind_curtain_panels(
            ds_cut,
            dem_path=self.dem_path,
            output_path=output_path,
            show=False,
            **self.plot2d_options,
        )
        self.plot2d_status = f"Saved {output_path}"
        self._refresh_status()
        return pn.pane.Matplotlib(fig, tight=True, dpi=120, width=680)

    def _render_plot2d(self, clicks: int):
        """Panel callback bound to the Plot 2D button."""
        import panel as pn

        if clicks == 0:
            return pn.pane.Markdown("")

        if self.ds_cut is None:
            self.cut()
        if self.ds_cut is None:
            self.plot2d_status = "Select start and end profiles before plotting."
            self._refresh_status()
            return pn.pane.Markdown("Select start and end profiles before plotting.")

        return self._plot2d_from_dataset(self.ds_cut)
        
    def _handle_plot3d_button(self, _event) -> None:
        if self.ds_cut is None:
            self.cut()
        if self.ds_cut is None:
            self.plot2d_status = "Select start and end profiles before plotting."
            self._refresh_status()
            return None

        import plot3d

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_cut_dataset(self.ds_cut)

        plot3d.plot_3d(
            self.ds_cut,
            self.dem_path,
            satellite_path=self.satellite_path,
            variable="w",
            group=None,
            show=True,
            z_min=500,
            z_max=3000,
            dz=25,
            across_width_m=10000,
            wind_every_n=1,
            wind_scale=50,
        )
        self._refresh_status()
        return None
        
    def _tap_to_lon_lat(self, x: float, y: float) -> tuple[float, float]:
        if -180 <= x <= 180 and -90 <= y <= 90:
            return x, y

        try:
            from pyproj import Transformer

            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
            return float(lon), float(lat)
        except Exception:
            return x, y

    def _status_text(self) -> str:
        cut_text = "not cut yet" if self.ds_cut is None else "ready"
        save_text = (
            "nothing saved yet"
            if self.save_status is None
            else f"Dataset saved to {self.save_status}"
        )
        plot2d_text = self.plot2d_status or "not plotted yet"
        mode_text = self._selection_mode or "none"
        return (
            f"Selection mode: `{mode_text}`  \n"
            f"Start profile: `{self.start_profile}`  \n"
            f"End profile: `{self.end_profile}`  \n"
            f"Cut dataset: `{cut_text}`   \n"
            f"Save dataset: `{save_text}`  \n"
            f"Plot 2D: `{plot2d_text}`"
        )

    def _refresh_status(self) -> None:
        if self._status is not None:
            self._status.object = self._status_text()


def resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve settings values relative to the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def read_settings(settings_path: Path) -> tuple[configparser.ConfigParser, Path]:
    """Read settings.ini and return it with the repository root."""
    settings_path = settings_path.resolve()
    config = configparser.ConfigParser()
    config.read(settings_path, encoding="utf-8")
    return config, settings_path.parent


def configured_file(config: configparser.ConfigParser, base_dir: Path, folder_key: str, file_key: str) -> Path:
    """Resolve one configured input file from [paths] and [files]."""
    folder = resolve_path(config["paths"][folder_key], base_dir)
    filename = config["files"][file_key].strip()
    path = resolve_path(filename, folder)
    if path.exists() or path.suffix:
        return path

    nc_path = path.with_suffix(".nc")
    if file_key == "pro_file" and nc_path.exists():
        return nc_path

    return path


def load_lidar_dataset(raw_file: Path) -> xr.Dataset:
    """Load a TEAMx lidar NetCDF file, preferring the combined group."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Lidar file not found: {raw_file}")
    if raw_file.suffix.lower() != ".nc":
        raise ValueError(f"Expected a .nc lidar file, got: {raw_file.name}")

    try:
        ds = xr.open_dataset(raw_file, group="combined")
    except (OSError, ValueError):
        ds = xr.open_dataset(raw_file)

    try:
        return ds.load()
    finally:
        ds.close()


def build_app(settings_path: Path) -> FlightMapApp:
    """Create the interactive map app from settings.ini."""
    config, base_dir = read_settings(settings_path)

    raw_file = configured_file(config, base_dir, "raw_dir", "pro_file")
    dem_path = configured_file(config, base_dir, "topography_dir", "topography_file")
    satellite_path = configured_file(config, base_dir, "satellite_dir", "satellite_file")
    processed_dir = resolve_path(config["paths"]["processed_dir"], base_dir)
    results_dir = resolve_path(config["paths"]["results_dir"], base_dir)

    ds = load_lidar_dataset(raw_file)

    plot2d_options = {
        "z_min": config.getfloat("plot2d", "z_min", fallback=500),
        "z_max": config.getfloat("plot2d", "z_max", fallback=3200),
        "vertical_every_n": config.getint("plot2d", "vertical_every_n", fallback=2),
        "wind_every_n": config.getint("plot2d", "wind_every_n", fallback=3),
        "use_dem_topography": config.getboolean("plot2d", "use_dem_topography", fallback=True),
        "first_panel_var": config.get("plot2d", "first_panel_var", fallback="w"),
    }

    return FlightMapApp(
        ds,
        lat="AER_lat_avg",
        lon="AER_lon_avg",
        profile_dim="nprofile",
        time="AER_posixtime",
        hover_columns=["AER_alt_avg"],
        output_dir=processed_dir,
        dem_path=dem_path,
        satellite_path=satellite_path,
        results_dir=results_dir,
        plot2d_options=plot2d_options,
    )


def main() -> None:
    """Launch the Panel selector app."""
    parser = argparse.ArgumentParser(description="Open TEAMx flight-track selector.")
    parser.add_argument(
        "--settings",
        default=Path(__file__).resolve().parents[1] / "settings.ini",
        type=Path,
    )
    args = parser.parse_args()

    app = build_app(args.settings)

    import panel as pn

    pn.serve(app.panel, title="TEAMx flight-track selector", show=True)


if __name__ == "__main__":
    main()
