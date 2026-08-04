"""Docked interactive PITC / PWV diagnostic plots."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.viz.cross_section_panel import DOCK_OBJECT_NAME as XS_DOCK_OBJECT_NAME
from nvitk.gui.viz.left_dock import attach_left_inspection_dock

DOCK_OBJECT_NAME = "nvitk_hemodynamics_plot_dock"

_CMAPS = (
    "viridis",
    "magma",
    "plasma",
    "inferno",
    "turbo",
    "cividis",
    "coolwarm",
    "RdYlBu_r",
    "gray",
)


class HemodynamicsPlotPanel(QWidget):
    """Interactive Matplotlib panel for PITC / PWV / Bjornfoot figures."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the plot-type selector, station coloring controls, and Matplotlib canvas/toolbar."""
        super().__init__(parent)
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._canvas = None
        self._toolbar = None
        self._fig = None
        self._region_plot_data: dict[str, dict[str, Any]] = {}
        self._saved_plot_paths: dict[str, Any] = {}
        self._station_layer: Any | None = None
        self._auto_limits: tuple[float, float] | None = None

        layout = QVBoxLayout()
        layout.addWidget(self._title)

        self._plot_selector = QComboBox()
        self._plot_selector.addItem("PITC: quality + PI vs distance", "pitc")
        self._plot_selector.addItem("PWV timing + weights", "pwv")
        self._plot_selector.addItem(
            "Bjornfoot fit QC (residuals, delay, correlation)", "bjornfoot"
        )
        self._plot_selector.currentIndexChanged.connect(self._render_current_plot)
        layout.addWidget(self._plot_selector)

        color_row = QHBoxLayout()
        self._color_label = QLabel("Color stations by")
        self._color_selector = QComboBox()
        self._color_selector.setToolTip(
            "Recolor the Napari stations layer by a numeric station feature."
        )
        self._color_selector.currentIndexChanged.connect(self._apply_station_colors)
        color_row.addWidget(self._color_label)
        color_row.addWidget(self._color_selector, stretch=1)
        layout.addLayout(color_row)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap"))
        self._cmap_selector = QComboBox()
        for name in _CMAPS:
            self._cmap_selector.addItem(name, name)
        self._cmap_selector.setCurrentIndex(_CMAPS.index("viridis"))
        self._cmap_selector.currentIndexChanged.connect(self._apply_station_colors)
        cmap_row.addWidget(self._cmap_selector, stretch=1)
        layout.addLayout(cmap_row)

        range_row = QHBoxLayout()
        self._auto_range = QCheckBox("Auto range")
        self._auto_range.setChecked(True)
        self._auto_range.toggled.connect(self._on_auto_range_toggled)
        range_row.addWidget(self._auto_range)
        range_row.addWidget(QLabel("lo"))
        self._vmin = QDoubleSpinBox()
        self._vmin.setDecimals(4)
        self._vmin.setRange(-1e6, 1e6)
        self._vmin.setSingleStep(0.1)
        self._vmin.valueChanged.connect(self._apply_station_colors)
        range_row.addWidget(self._vmin)
        range_row.addWidget(QLabel("hi"))
        self._vmax = QDoubleSpinBox()
        self._vmax.setDecimals(4)
        self._vmax.setRange(-1e6, 1e6)
        self._vmax.setSingleStep(0.1)
        self._vmax.valueChanged.connect(self._apply_station_colors)
        range_row.addWidget(self._vmax)
        layout.addLayout(range_row)
        self._set_range_widgets_enabled(False)

        legend_row = QHBoxLayout()
        self._show_legend = QCheckBox("Show plot legend")
        self._show_legend.setChecked(False)
        self._show_legend.toggled.connect(self._render_current_plot)
        legend_row.addWidget(self._show_legend)
        legend_row.addStretch(1)
        layout.addLayout(legend_row)

        self._color_label.setVisible(False)
        self._color_selector.setVisible(False)
        self._cmap_selector.setVisible(False)
        self._auto_range.setVisible(False)
        self._vmin.setVisible(False)
        self._vmax.setVisible(False)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(5.0, 4.5), dpi=96)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setMinimumHeight(220)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, stretch=1)
        except Exception as exc:
            err = QLabel(f"Matplotlib unavailable: {exc}")
            err.setWordWrap(True)
            layout.addWidget(err)
        self.setLayout(layout)

    def _set_range_widgets_enabled(self, enabled: bool) -> None:
        """Enable/disable the manual vmin/vmax spin boxes."""
        self._vmin.setEnabled(enabled)
        self._vmax.setEnabled(enabled)

    def _on_auto_range_toggled(self, checked: bool) -> None:
        """When auto-range is enabled, disable the manual spin boxes and snap them to the last
        computed auto limits; re-applies station coloring either way."""
        self._set_range_widgets_enabled(not checked)
        if checked and self._auto_limits is not None:
            self._vmin.blockSignals(True)
            self._vmax.blockSignals(True)
            self._vmin.setValue(self._auto_limits[0])
            self._vmax.setValue(self._auto_limits[1])
            self._vmin.blockSignals(False)
            self._vmax.blockSignals(False)
        self._apply_station_colors()

    def bind_station_layer(
        self,
        layer: Any | None,
        *,
        mode: str = "hemo",
        default_feature: str | None = None,
        default_cmap: str | None = None,
    ) -> None:
        """Attach the stations Points layer and populate coloring controls."""
        from nvitk.gui.viz.hemo_geometry import station_feature_choices

        _ = mode
        self._station_layer = layer
        self._color_selector.blockSignals(True)
        self._color_selector.clear()
        has_layer = layer is not None
        for w in (
            self._color_label,
            self._color_selector,
            self._cmap_selector,
            self._auto_range,
            self._vmin,
            self._vmax,
        ):
            w.setVisible(has_layer)
        if layer is None:
            self._color_selector.blockSignals(False)
            return
        choices = station_feature_choices(layer)
        for name in choices:
            self._color_selector.addItem(name, name)
        self._color_label.setVisible(bool(choices))
        self._color_selector.setVisible(bool(choices))
        preferred = str(default_feature or "quality")
        if preferred in choices:
            self._color_selector.setCurrentIndex(choices.index(preferred))
        elif choices:
            self._color_selector.setCurrentIndex(0)
        cmap = str(default_cmap or "viridis")
        idx = self._cmap_selector.findData(cmap)
        if idx >= 0:
            self._cmap_selector.setCurrentIndex(idx)
        self._color_selector.blockSignals(False)
        if self._color_selector.count():
            self._apply_station_colors()

    def _apply_station_colors(self, _index: int | None = None) -> None:
        """Recolor the bound stations layer by the selected feature/colormap/range, updating the
        auto-range limits and spin-box values from the result."""
        if self._station_layer is None or self._color_selector.count() == 0:
            return
        feature = self._color_selector.currentData() or self._color_selector.currentText()
        if not feature:
            return
        cmap = self._cmap_selector.currentData() or self._cmap_selector.currentText() or "viridis"
        from nvitk.gui.viz.hemo_geometry import color_stations_by_feature

        try:
            lo_hi = color_stations_by_feature(
                self._station_layer,
                str(feature),
                cmap_name=str(cmap),
                vmin=None if self._auto_range.isChecked() else float(self._vmin.value()),
                vmax=None if self._auto_range.isChecked() else float(self._vmax.value()),
            )
            if lo_hi is not None:
                self._auto_limits = lo_hi
                if self._auto_range.isChecked():
                    self._vmin.blockSignals(True)
                    self._vmax.blockSignals(True)
                    self._vmin.setValue(lo_hi[0])
                    self._vmax.setValue(lo_hi[1])
                    self._vmin.blockSignals(False)
                    self._vmax.blockSignals(False)
        except Exception:
            pass

    def show_plot(
        self,
        region_plot_data: dict[str, dict[str, Any]],
        *,
        mode: str = "hemo",
        initial_plot: str = "pitc",
        saved_plot_paths: dict[str, Any] | None = None,
    ) -> None:
        """Render a live measurement figure with zoom/pan controls."""
        self._region_plot_data = region_plot_data
        self._saved_plot_paths = dict(saved_plot_paths or {})
        _ = mode
        idx = self._plot_selector.findData(str(initial_plot))
        self._plot_selector.blockSignals(True)
        self._plot_selector.setCurrentIndex(idx if idx >= 0 else 0)
        self._plot_selector.blockSignals(False)
        self._render_current_plot()

    def _show_saved_png(self, kind: str, label: str) -> bool:
        """Display a stage-6 PNG when interactive arrays are unavailable."""
        path = self._saved_plot_paths.get(str(kind))
        if path is None or self._fig is None or self._canvas is None:
            return False
        try:
            from matplotlib.image import imread
            from pathlib import Path

            p = Path(path)
            if not p.is_file():
                return False
            img = imread(str(p))
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.imshow(img)
            ax.set_axis_off()
            self._fig.tight_layout()
            self._title.setText(f"{label}\n(saved stage-6 figure)")
            if self._toolbar is not None:
                self._toolbar.update()
            self._canvas.draw_idle()
            return True
        except Exception:
            return False

    def _has_interactive_arrays(self, kind: str) -> bool:
        """True if any loaded region has the array data needed to draw *kind* (``"pitc"`` needs
        ``distance_mm``; PWV/Bjornfoot need ``pwv_distance_mm``) interactively rather than from a
        saved static figure."""
        if not self._region_plot_data:
            return False
        if kind == "pitc":
            return any(
                np.asarray(d.get("distance_mm", [])).size
                for d in self._region_plot_data.values()
                if isinstance(d, dict)
            )
        # PWV / Bjornfoot need timing (or residual) arrays from the viz bundle.
        return any(
            np.asarray(d.get("pwv_distance_mm", [])).size
            for d in self._region_plot_data.values()
            if isinstance(d, dict)
        )

    def _render_current_plot(self, _index: int | None = None) -> None:
        """Render the selected PITC, PWV, or Bjornfoot QC figure."""
        kind = self._plot_selector.currentData() or "pitc"
        if kind == "pitc":
            from nvitk.pipes.qvtpy.util.hemodynamics.measure_plots import make_pitc_figure

            make_fn = make_pitc_figure
            label = "PITC: quality and pulsatility index vs distance"
        elif kind == "bjornfoot":
            from nvitk.pipes.qvtpy.util.hemodynamics.measure_plots import make_bjornfoot_qc_figure

            make_fn = make_bjornfoot_qc_figure
            label = (
                "Bjornfoot QC: weighted residual RMS, XCor−model delay, "
                "and waveform correlation"
            )
        else:
            from nvitk.pipes.qvtpy.util.hemodynamics.measure_plots import make_pwv_figure

            make_fn = make_pwv_figure
            label = "PWV: XCor delay, time-to-upstroke, and weights vs distance"
        self._title.setText(label)
        if self._canvas is None or self._fig is None:
            return
        fig = None
        if self._has_interactive_arrays(str(kind)):
            fig = make_fn(
                self._region_plot_data,
                show_legend=bool(self._show_legend.isChecked()),
                fig=self._fig,
            )
        if fig is None:
            if self._show_saved_png(str(kind), label):
                return
            self._fig.clear()
            self._title.setText(f"{label}\n(No plot data available.)")
            self._canvas.draw_idle()
            return
        if self._toolbar is not None:
            self._toolbar.update()
        self._canvas.draw_idle()


def attach_hemodynamics_plot_dock(
    viewer: Any,
    panel: HemodynamicsPlotPanel,
    *,
    mode: str = "hemo",
) -> Any:
    """Attach or replace the hemodynamics plot dock on the left."""
    _ = mode
    return attach_left_inspection_dock(
        viewer,
        panel,
        object_name=DOCK_OBJECT_NAME,
        title="PITC / PWV diagnostics",
        tabify_with=[XS_DOCK_OBJECT_NAME],
        minimum_width=360,
    )


def show_hemodynamics_plot(
    viewer: Any,
    region_plot_data: dict[str, dict[str, Any]],
    *,
    mode: str = "hemo",
    station_layer: Any | None = None,
    default_face_key: str | None = None,
    initial_plot: str = "pitc",
    saved_plot_paths: dict[str, Any] | None = None,
) -> Any:
    """Generate and show the merged PITC/PWV diagnostics dock."""
    panel = HemodynamicsPlotPanel()
    panel.show_plot(
        region_plot_data,
        mode=mode,
        initial_plot=initial_plot,
        saved_plot_paths=saved_plot_paths,
    )
    state = getattr(viewer, "_nvitk_hemo_overlay_state", None)
    layer = station_layer
    face_key = default_face_key
    if layer is None and state is not None:
        layer = getattr(state, "station_layer", None)
        face_key = face_key or getattr(state, "default_face_key", None)
    panel.bind_station_layer(layer, mode=mode, default_feature=face_key)
    return attach_hemodynamics_plot_dock(viewer, panel, mode=mode)


__all__ = [
    "DOCK_OBJECT_NAME",
    "HemodynamicsPlotPanel",
    "attach_hemodynamics_plot_dock",
    "show_hemodynamics_plot",
]
