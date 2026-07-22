"""Docked interactive PITC and PWV diagnostic plots."""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nvitk.gui.viz.cross_section_panel import DOCK_OBJECT_NAME as XS_DOCK_OBJECT_NAME
from nvitk.gui.viz.left_dock import attach_left_inspection_dock

DOCK_OBJECT_NAME = "nvitk_hemodynamics_plot_dock"


def _dock_object_name(mode: str) -> str:
    """Per-mode dock id so PITC and PWV diagnostics coexist as separate tabs."""
    return f"nvitk_hemodynamics_{str(mode).lower()}_dock"


class HemodynamicsPlotPanel(QWidget):
    """Interactive Matplotlib panel for a generated PITC or PWV figure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._canvas = None
        self._toolbar = None
        self._fig = None
        self._region_plot_data: dict[str, dict[str, Any]] = {}
        self._mode = "pwv"
        self._station_layer: Any | None = None
        layout = QVBoxLayout()
        layout.addWidget(self._title)

        color_row = QHBoxLayout()
        self._color_label = QLabel("Color stations by")
        self._color_selector = QComboBox()
        self._color_selector.setToolTip(
            "Napari no longer exposes Points feature-colormap controls; "
            "use this to recolor the stations layer after a run."
        )
        self._color_selector.currentIndexChanged.connect(self._on_color_feature_changed)
        color_row.addWidget(self._color_label)
        color_row.addWidget(self._color_selector, stretch=1)
        layout.addLayout(color_row)
        self._color_label.setVisible(False)
        self._color_selector.setVisible(False)

        self._plot_selector = QComboBox()
        self._plot_selector.addItem("PWV timing + weights", "pwv")
        self._plot_selector.addItem("Bjornfoot fit QC (residuals, delay, correlation)", "bjornfoot")
        self._plot_selector.currentIndexChanged.connect(self._render_current_plot)
        self._plot_selector.setVisible(False)
        layout.addWidget(self._plot_selector)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(5.0, 4.5), dpi=96)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            layout.addWidget(self._toolbar)
            layout.addWidget(self._canvas, stretch=1)
        except Exception as exc:
            err = QLabel(f"Matplotlib unavailable: {exc}")
            err.setWordWrap(True)
            layout.addWidget(err)
        self.setLayout(layout)

    def bind_station_layer(
        self,
        layer: Any | None,
        *,
        mode: str,
        default_feature: str | None = None,
    ) -> None:
        """Attach the stations Points layer and populate the color-by feature list."""
        from nvitk.gui.viz.hemo_geometry import station_feature_choices

        self._station_layer = layer
        self._mode = str(mode)
        self._color_selector.blockSignals(True)
        self._color_selector.clear()
        if layer is None:
            self._color_label.setVisible(False)
            self._color_selector.setVisible(False)
            self._color_selector.blockSignals(False)
            return
        choices = station_feature_choices(layer)
        for name in choices:
            self._color_selector.addItem(name, name)
        self._color_label.setVisible(bool(choices))
        self._color_selector.setVisible(bool(choices))
        preferred = str(
            default_feature or ("quality" if mode == "pitc" else "pwv_weight_area")
        )
        if preferred in choices:
            self._color_selector.setCurrentIndex(choices.index(preferred))
        elif choices:
            self._color_selector.setCurrentIndex(0)
        self._color_selector.blockSignals(False)
        if self._color_selector.count():
            self._on_color_feature_changed()

    def _on_color_feature_changed(self, _index: int | None = None) -> None:
        if self._station_layer is None or self._color_selector.count() == 0:
            return
        feature = self._color_selector.currentData()
        if not feature:
            feature = self._color_selector.currentText()
        if not feature:
            return
        from nvitk.gui.viz.hemo_geometry import color_stations_by_feature

        try:
            color_stations_by_feature(
                self._station_layer,
                str(feature),
                mode=self._mode,
            )
        except Exception:
            pass

    def show_plot(
        self,
        region_plot_data: dict[str, dict[str, Any]],
        *,
        mode: str,
    ) -> None:
        """Render a live measurement figure with zoom/pan controls."""
        self._region_plot_data = region_plot_data
        self._mode = mode
        self._plot_selector.setVisible(mode == "pwv")
        self._render_current_plot()

    def _render_current_plot(self, _index: int | None = None) -> None:
        """Render the selected PITC, PWV, or Bjornfoot QC figure."""
        if self._mode == "pitc":
            from nvitk.pipes.qvtpy.util.measure_plots import make_pitc_figure

            make_fn = make_pitc_figure
            label = "PITC: quality and pulsatility index vs distance"
        elif self._plot_selector.currentData() == "bjornfoot":
            from nvitk.pipes.qvtpy.util.measure_plots import make_bjornfoot_qc_figure

            make_fn = make_bjornfoot_qc_figure
            label = (
                "Bjornfoot QC: weighted residual RMS, XCor−model delay, "
                "and waveform correlation"
            )
        else:
            from nvitk.pipes.qvtpy.util.measure_plots import make_pwv_figure

            make_fn = make_pwv_figure
            label = "PWV: XCor delay, time-to-upstroke, and weights vs distance"
        self._title.setText(label)
        if self._canvas is None:
            return
        fig = make_fn(self._region_plot_data, show_legend=False)
        if fig is None:
            self._canvas.figure.clear()
            self._title.setText(f"{label}\n(No plot data available.)")
            self._canvas.draw_idle()
            return
        self._fig = fig
        self._canvas.figure = fig
        if self._toolbar is not None:
            self._toolbar.update()
        self._canvas.draw_idle()


def attach_hemodynamics_plot_dock(
    viewer: Any,
    panel: HemodynamicsPlotPanel,
    *,
    mode: str,
) -> Any:
    """Attach or replace the hemodynamics plot dock on the left.

    PITC and PWV use distinct dock ids so both remain available as tabs (alongside
    the cross-section dock); re-running the same tool replaces its own dock.
    """
    other_mode = "pwv" if str(mode).lower() == "pitc" else "pitc"
    return attach_left_inspection_dock(
        viewer,
        panel,
        object_name=_dock_object_name(mode),
        title=f"{mode.upper()} diagnostics",
        tabify_with=[XS_DOCK_OBJECT_NAME, _dock_object_name(other_mode)],
        minimum_width=360,
    )


def show_hemodynamics_plot(
    viewer: Any,
    region_plot_data: dict[str, dict[str, Any]],
    *,
    mode: str,
    station_layer: Any | None = None,
    default_face_key: str | None = None,
) -> Any:
    """Generate and show a PITC/PWV diagnostics dock."""
    panel = HemodynamicsPlotPanel()
    panel.show_plot(region_plot_data, mode=mode)
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
