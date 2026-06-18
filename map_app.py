"""Panel/HoloViews app for interactively selecting and cutting flight tracks."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import pandas as pd
import xarray as xr

from teamx_proto.core import cut_dataset, save_dataset, extract_track, thin_track
from teamx_proto.selection import nearest_track_point


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
        thinning_step: int = 20,
        basemap: bool = True,
    ) -> None:
        self.ds = ds
        self.lat = lat
        self.lon = lon
        self.profile_dim = profile_dim
        self.time = time
        self.hover_columns = list(hover_columns or [])
        self.thinning_step = thinning_step
        self.basemap = basemap

        self.track = extract_track(ds, lat=lat, lon=lon, profile_dim=profile_dim, time=time)
        self._add_hover_metadata()
        self.track_plot = thin_track(self.track, step=thinning_step)
        self.start_profile: Any | None = None
        self.end_profile: Any | None = None
        self.ds_cut: xr.Dataset | None = None
        self.save_status: bool | None = None
        self._selection_mode: str | None = None

        self._panel = None
        self._status = None
        self._select_start_button = None
        self._select_end_button = None
        self._cut_button = None
        self._save_button = None
        self._plot2d_button = None
        self._plot3d_button = None

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

    def save(self) -> xr.Dataset | None:
        """Save the selected cut dataset, creating if it has been cut."""
        if self.ds_cut is None:
            raise Exception("Dataset needs to be cut before saving.") 
        self.save_status = save_dataset(
            self.ds_cut,
            "/home/sarah/TEAMx_hackathon/" # just for testing, change later
        )
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

        hover_cols = ["profile", "lat", "lon"]
        if "time" in self.track_plot.columns:
            hover_cols.append("time")
        hover_cols.extend(col for col in self.hover_columns if col in self.track_plot.columns)

        plot = self.track_plot.hvplot.points(
            x="lon",
            y="lat",
            geo=True,
            tiles="OSM" if self.basemap else None,
            hover_cols=hover_cols,
            size=8,
            #color="profile",
            #cmap="viridis",
            frame_width=850,
            frame_height=550,
            title="Flight track selector",
            tools=["hover", "tap"],
        )

        tap_stream = hv.streams.Tap(source=plot, x=None, y=None)
        tap_stream.add_subscriber(self._handle_tap)

        self._status = pn.pane.Markdown(self._status_text())
        self._select_start_button = pn.widgets.Button(
            name="Select starting point",
            button_type="default",
        )
        self._select_end_button = pn.widgets.Button(
            name="Create end point",
            button_type="default",
        )
        self._cut_button = pn.widgets.Button(name="Cut dataset", button_type="primary")
        self._save_button = pn.widgets.Button(name="Save dataset", button_type="primary")
        self._plot2d_button = pn.widgets.Button(name="Plot selection in 2D", button_type="success")
        self._plot3d_button = pn.widgets.Button(name="Plot selection in 3D", button_type="success")

        self._select_start_button.on_click(self._handle_select_start_button)
        self._select_end_button.on_click(self._handle_select_end_button)
        self._cut_button.on_click(self._handle_cut_button)
        self._save_button.on_click(self._handle_save_button)
        self._plot2d_button.on_click(self._handle_plot2d_button)
        self._plot3d_button.on_click(self._handle_plot3d_button)

        
        return pn.Column(
            plot,
            pn.Row(
                self._select_start_button,
                self._select_end_button,
                self._cut_button,
                self._save_button,
                self._plot2d_button,
                self._plot3d_button,
            ),
            self._status,
            sizing_mode="stretch_width",
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

    def _handle_plot2d_button(self, _event) -> None:
        ###TO BE FILLED

    def _handle_plot3d_button(self, _event) -> None:
        ###TO BE FILLED
        
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
        save_text = "nothing saved yet" if self.save_status is None else "Dataset saved."
        mode_text = self._selection_mode or "none"
        return (
            f"Selection mode: `{mode_text}`  \n"
            f"Start profile: `{self.start_profile}`  \n"
            f"End profile: `{self.end_profile}`  \n"
            f"Cut dataset: `{cut_text}`   \n"
            f"Save dataset: `{save_text}`"
        )

    def _refresh_status(self) -> None:
        if self._status is not None:
            self._status.object = self._status_text()
