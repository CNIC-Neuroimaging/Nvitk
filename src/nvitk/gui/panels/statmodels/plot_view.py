"""
Plot pane — Matplotlib canvas, group include/exclude checklist, and axis-limit sliders.

Description
-----------
Three controls sit around the figure:

*Groups*      one checkbox per level of the grouping column. Unchecking a level removes both its
              model curve and its raw points, and lets the axes rescale to what remains. The fit is
              untouched — the black fixed-effect line stays the all-group estimate — so this is a
              display filter, not a re-analysis. Use a ``territory`` filter on the analysis frame
              when the levels should actually leave the model.
*Axis limits* sliders that rescale the current figure without recomputing anything.
*Show legend* toggles the legend in place.

The figure itself is always drawn white (see :func:`~.theme.whiten_figure`) so it stays readable
against the dark chrome and exports cleanly.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger

from .constants import AXIS_SLIDER_MARGIN, AXIS_SLIDER_STEPS
from .theme import COLOR_ERROR, muted_label_style, whiten_figure

log = Logger()


class PlotPanel(QGroupBox):
    """
    The plot pane: canvas + group checklist + axis controls.

    Signals
    -------
    optionsChanged
        A control that requires the plot to be *redrawn* changed (the group selection). Axis sliders
        and the legend toggle act on the existing figure and do not emit.
    """

    optionsChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the canvas host, the Groups box and the axis-limit controls."""
        super().__init__("Plots", parent)
        self._canvas = None
        self._fig = None
        self._axes = None
        self._axis_base: dict[str, tuple[float, float]] = {}
        self._axis_span: dict[str, tuple[float, float]] = {}
        self._group_column = ""
        self._group_boxes: dict[str, QCheckBox] = {}
        self._suspend_group_signal = False

        lay = QVBoxLayout(self)
        # The figure is the point of this pane — keep the chrome around it thin.
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        # Options supplied by the window (plot mode / x / points) sit on one compact row with the
        # figure picker, so plot controls live next to the plot instead of in the formulation box.
        self._top_row = QWidget()
        top_lay = QHBoxLayout(self._top_row)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(6)
        self._options_slot = QHBoxLayout()
        top_lay.addLayout(self._options_slot, stretch=1)

        self._kind_row = QWidget()
        kind_lay = QHBoxLayout(self._kind_row)
        kind_lay.setContentsMargins(0, 0, 0, 0)
        kind_lay.addWidget(QLabel("Figure"))
        self._kind = QComboBox()
        kind_lay.addWidget(self._kind)
        self._kind_row.setVisible(False)
        top_lay.addWidget(self._kind_row)
        lay.addWidget(self._top_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(muted_label_style())
        self._status.setVisible(False)
        lay.addWidget(self._status)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_canvas_host())
        split.addWidget(self._build_groups_box())
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 190])
        lay.addWidget(split, stretch=1)

        lay.addWidget(self._build_axis_controls())

    # ---- construction ---------------------------------------------------------
    def _build_canvas_host(self) -> QWidget:
        """Container the Matplotlib canvas is swapped into."""
        self._host = QWidget()
        self._host_layout = QVBoxLayout(self._host)
        self._host_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Parameter / EMM plots appear here after fitting.")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(muted_label_style())
        self._host_layout.addWidget(hint)
        return self._host

    def _build_groups_box(self) -> QWidget:
        """Scrollable checklist of the grouping column's levels."""
        box = QGroupBox("Groups")
        box.setToolTip(
            "Show or hide a group in the plot. This hides its model curve and its raw points and "
            "rescales the axes — it does not refit the model, so the fixed-effect line stays the "
            "all-group estimate. To drop a group from the model itself, filter it out of the "
            "analysis dataframe."
        )
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)

        row = QHBoxLayout()
        for label, state in (("All", True), ("None", False)):
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _=False, s=state: self._set_all_groups(s))
            row.addWidget(btn)
        lay.addLayout(row)

        self._groups_scroll = QScrollArea()
        self._groups_scroll.setWidgetResizable(True)
        self._groups_scroll.setFrameShape(QScrollArea.NoFrame)
        self._groups_host = QWidget()
        self._groups_layout = QVBoxLayout(self._groups_host)
        self._groups_layout.setContentsMargins(0, 0, 0, 0)
        self._groups_layout.setSpacing(2)
        self._groups_scroll.setWidget(self._groups_host)
        lay.addWidget(self._groups_scroll, stretch=1)

        self._groups_box = box
        return box

    def _build_axis_controls(self) -> QWidget:
        """Legend toggle plus sliders that rescale the current figure without recomputing."""
        box = QGroupBox("Axis limits")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 2, 6, 4)

        # Legend toggle shares the slider row: four sliders and a checkbox fit on one line, and
        # every row spent on controls is a row taken from the figure.
        self._show_legend = QCheckBox("Legend")
        self._show_legend.setChecked(True)
        self._show_legend.setToolTip(
            "Show or hide the plot legend (model curves only; raw points are never listed)."
        )
        self._show_legend.stateChanged.connect(lambda *_: self.apply_legend_visibility())

        grid = QHBoxLayout()
        grid.setSpacing(4)
        grid.addWidget(self._show_legend)
        self._axis_sliders: dict[str, QSlider] = {}
        self._axis_value_labels: dict[str, QLabel] = {}
        for axis, bound, text in (
            ("x", "min", "x min"),
            ("x", "max", "x max"),
            ("y", "min", "y min"),
            ("y", "max", "y max"),
        ):
            key = f"{axis}{bound}"
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, AXIS_SLIDER_STEPS)
            slider.setEnabled(False)
            slider.valueChanged.connect(self._on_axis_slider_changed)
            value_label = QLabel("—")
            value_label.setMinimumWidth(64)
            value_label.setStyleSheet(muted_label_style())
            self._axis_sliders[key] = slider
            self._axis_value_labels[key] = value_label
            grid.addWidget(QLabel(text))
            grid.addWidget(slider, stretch=1)
            grid.addWidget(value_label)

        self._btn_reset_axes = QPushButton("Reset")
        self._btn_reset_axes.setEnabled(False)
        self._btn_reset_axes.clicked.connect(self.reset_axes)
        grid.addWidget(self._btn_reset_axes)
        lay.addLayout(grid)
        return box

    # ---- figure picker --------------------------------------------------------
    def kind_combo(self) -> QComboBox:
        """The optional figure picker, for analyses that produce several plots."""
        return self._kind

    def set_kind_row_visible(self, visible: bool) -> None:
        """Show or hide the figure picker row."""
        self._kind_row.setVisible(bool(visible))

    def set_options_widget(self, widget: QWidget) -> None:
        """Place the window's plot-option controls on the pane's top row."""
        self._options_slot.addWidget(widget)

    def set_groups_visible(self, visible: bool) -> None:
        """Show or hide the group checklist (it only applies to grouped model plots)."""
        self._groups_box.setVisible(bool(visible))

    # ---- group checklist ------------------------------------------------------
    def set_levels(self, column: str, levels: Sequence[str]) -> None:
        """
        Rebuild the group checklist for *column*, preserving which levels were unchecked.

        Levels that disappear (a reload changed the frame) lose their state; new ones start checked.
        """
        previous = {name: box.isChecked() for name, box in self._group_boxes.items()}
        same_column = column == self._group_column
        self._group_column = column

        self._suspend_group_signal = True
        try:
            while self._groups_layout.count():
                item = self._groups_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._group_boxes = {}

            if not levels:
                empty = QLabel("—")
                empty.setStyleSheet(muted_label_style())
                self._groups_layout.addWidget(empty)
                self._groups_box.setTitle("Groups")
                return

            self._groups_box.setTitle(f"Groups · {column}")
            for level in levels:
                box = QCheckBox(str(level))
                box.setChecked(previous.get(str(level), True) if same_column else True)
                box.stateChanged.connect(self._on_group_toggled)
                self._groups_layout.addWidget(box)
                self._group_boxes[str(level)] = box
            self._groups_layout.addStretch(1)
        finally:
            self._suspend_group_signal = False

    def checked_levels(self) -> list[str]:
        """Levels currently checked, in the order they were added."""
        return [name for name, box in self._group_boxes.items() if box.isChecked()]

    def set_checked_levels(self, levels: Sequence[str] | None) -> None:
        """Check exactly *levels* (``None`` checks everything)."""
        self._suspend_group_signal = True
        try:
            wanted = None if levels is None else {str(v) for v in levels}
            for name, box in self._group_boxes.items():
                box.setChecked(True if wanted is None else name in wanted)
        finally:
            self._suspend_group_signal = False

    def _set_all_groups(self, state: bool) -> None:
        """Check or uncheck every level, then request a redraw once."""
        if not self._group_boxes:
            return
        self._suspend_group_signal = True
        try:
            for box in self._group_boxes.values():
                box.setChecked(state)
        finally:
            self._suspend_group_signal = False
        self.optionsChanged.emit()

    def _on_group_toggled(self, *_args: Any) -> None:
        """Request a redraw unless we are mid-rebuild."""
        if not self._suspend_group_signal:
            self.optionsChanged.emit()

    # ---- figure ---------------------------------------------------------------
    def clear(self) -> None:
        """Remove the current plot widget and release its figure."""
        while self._host_layout.count():
            item = self._host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except Exception as exc:
                log.debug("Could not close the previous figure: %s", exc)
        self._canvas = None
        self._fig = None
        self._axes = None
        self._disable_axis_sliders()

    def show_figure(self, fig: Any) -> None:
        """Embed *fig* as the current plot and reset the axis sliders to its limits."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self.clear()
        whiten_figure(fig)
        canvas = FigureCanvasQTAgg(fig)
        self._host_layout.addWidget(canvas)
        self._canvas = canvas
        self._fig = fig
        # Axis sliders only make sense for a single-axes figure; the mediation partial-path plot has
        # two, so leave them disabled there rather than rescaling an arbitrary one.
        self._axes = fig.axes[0] if len(fig.axes) == 1 else None
        fig.tight_layout()
        canvas.draw_idle()
        self._sync_axis_sliders()
        self.apply_legend_visibility()

    def show_error(self, message: str) -> None:
        """Replace the plot with an error message."""
        self.clear()
        label = QLabel(message)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"color: {COLOR_ERROR}; font-weight: normal;")
        self._host_layout.addWidget(label)

    def set_status(self, message: str) -> None:
        """Show a note above the canvas (e.g. an EMM grid that could not be evaluated)."""
        self._status.setText(message)
        self._status.setVisible(bool(message))

    def show_legend(self) -> bool:
        """Whether the legend toggle is on."""
        return self._show_legend.isChecked()

    def set_show_legend(self, value: bool) -> None:
        """Set the legend toggle without triggering a redraw of the data."""
        self._show_legend.setChecked(bool(value))

    def apply_legend_visibility(self) -> None:
        """Show or hide the current axes legends according to the toggle."""
        if self._fig is None or self._canvas is None:
            return
        for ax in self._fig.axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(self._show_legend.isChecked())
        self._canvas.draw_idle()

    # ---- axis sliders ---------------------------------------------------------
    def _disable_axis_sliders(self) -> None:
        """Grey out the axis-limit controls (no single-axes figure to rescale)."""
        for key, slider in self._axis_sliders.items():
            slider.blockSignals(True)
            slider.setEnabled(False)
            slider.blockSignals(False)
            self._axis_value_labels[key].setText("—")
        self._btn_reset_axes.setEnabled(False)
        self._axis_base = {}
        self._axis_span = {}

    def _sync_axis_sliders(self) -> None:
        """
        Capture the freshly drawn axes limits as the slider baseline and move the handles to match.

        Each slider spans the plotted range padded by :data:`AXIS_SLIDER_MARGIN` on both sides, so
        the handles start inset and the user can zoom out as well as in.
        """
        ax = self._axes
        if ax is None:
            self._disable_axis_sliders()
            return

        self._axis_base = {"x": tuple(ax.get_xlim()), "y": tuple(ax.get_ylim())}
        self._axis_span = {}
        for axis, (lo, hi) in self._axis_base.items():
            span = float(hi) - float(lo)
            pad = (abs(span) * AXIS_SLIDER_MARGIN) if span else 1.0
            self._axis_span[axis] = (float(lo) - pad, float(hi) + pad)

        for key, slider in self._axis_sliders.items():
            axis, bound = key[0], key[1:]
            value = self._axis_base[axis][0 if bound == "min" else 1]
            slider.blockSignals(True)
            slider.setEnabled(True)
            slider.setValue(self._axis_value_to_slider(axis, value))
            slider.blockSignals(False)
            self._axis_value_labels[key].setText(f"{value:.4g}")
        self._btn_reset_axes.setEnabled(True)

    def _axis_slider_to_value(self, axis: str, position: int) -> float:
        """Map a slider position to a data coordinate on *axis*."""
        lo, hi = self._axis_span[axis]
        return lo + (hi - lo) * (position / AXIS_SLIDER_STEPS)

    def _axis_value_to_slider(self, axis: str, value: float) -> int:
        """Map a data coordinate on *axis* back to a slider position."""
        lo, hi = self._axis_span[axis]
        if hi == lo:
            return 0
        frac = (float(value) - lo) / (hi - lo)
        return int(round(min(max(frac, 0.0), 1.0) * AXIS_SLIDER_STEPS))

    def _on_axis_slider_changed(self, *_args: Any) -> None:
        """Rescale the current figure's axes to the slider positions and redraw the canvas."""
        ax = self._axes
        if ax is None or not self._axis_span or self._canvas is None:
            return
        for axis, setter in (("x", ax.set_xlim), ("y", ax.set_ylim)):
            lo = self._axis_slider_to_value(axis, self._axis_sliders[f"{axis}min"].value())
            hi = self._axis_slider_to_value(axis, self._axis_sliders[f"{axis}max"].value())
            self._axis_value_labels[f"{axis}min"].setText(f"{lo:.4g}")
            self._axis_value_labels[f"{axis}max"].setText(f"{hi:.4g}")
            if hi <= lo:  # crossed handles would raise; keep a hair of range instead
                span = self._axis_span[axis]
                hi = lo + abs(span[1] - span[0]) / AXIS_SLIDER_STEPS or lo + 1e-9
            setter(lo, hi)
        self._canvas.draw_idle()

    def reset_axes(self) -> None:
        """Restore the autoscaled limits captured when the plot was drawn."""
        ax = self._axes
        if ax is None or not self._axis_base:
            return
        ax.set_xlim(*self._axis_base["x"])
        ax.set_ylim(*self._axis_base["y"])
        for key, slider in self._axis_sliders.items():
            axis, bound = key[0], key[1:]
            value = self._axis_base[axis][0 if bound == "min" else 1]
            slider.blockSignals(True)
            slider.setValue(self._axis_value_to_slider(axis, value))
            slider.blockSignals(False)
            self._axis_value_labels[key].setText(f"{value:.4g}")
        if self._canvas is not None:
            self._canvas.draw_idle()


__all__ = ["PlotPanel"]
