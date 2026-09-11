"""Napari dock showing a vessel flattened by curved planar reformation.

The upper canvas is the vessel drawn straight: rows are stations along the
centerline, spaced uniformly in millimetres, so a lesion's length and the calibre
on either side of it are readable off the vertical axis. Lumen and wall are drawn
over it. Clicking a row picks a station; the lower canvas then shows that station's
true perpendicular cross-section, and the 3D view marks where it is.

The centerline is exposed as an editable Points layer. Napari already lets a Points
layer be dragged in select mode, so retouching is: switch that layer to select,
move a control point, and the reformation re-renders through the moved curve. Mask
retouching happens with Napari's own brush on the volume, where the geometry is
unambiguous, and the panel re-renders when the mask changes.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup, using
from nvitk.gui.core.design import (
    COLOR_MUTED,
    SPACE_TIGHT,
    apply_theme,
    style_image_figure,
)
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.transform.cpr import resample_centerline
from nvitk.viz.vessel_cpr import (
    DEFAULT_JOIN_GAP_VOX,
    DEFAULT_N_RAY,
    DEFAULT_RAY_MM,
    VesselCpr,
    build_vessel_cpr,
    centerlines_for_labels,
    plane_corners,
    vessel_cross_section,
    vessel_name,
)

setup(globals())

DOCK_OBJECT_NAME = "nvitk_vessel_cpr_dock"

#: Overlay layers the panel owns. Removed on shutdown and rebuilt on re-render.
CPR_CENTERLINE = "Vessel CPR centerline"
CPR_CONTROL_POINTS = "Vessel CPR control points"
CPR_STATION = "Vessel CPR station"
CPR_PLANE = "Vessel CPR plane"
OVERLAY_LAYERS = (CPR_CENTERLINE, CPR_CONTROL_POINTS, CPR_STATION, CPR_PLANE)

#: Overlay colours. The toggles are tinted to match, so the legend and the
#: image cannot disagree about which band is which.
LUMEN_RGB = (1.0, 0.0, 0.0)
WALL_RGB = (1.0, 0.65, 0.0)
CENTERLINE_COLOR = "#6fa8dc"
STATION_COLOR = "#ffa400"
LUMEN_HEX = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in LUMEN_RGB)
WALL_HEX = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in WALL_RGB)

#: Below this main-window width the flat vessel is given its own floating window
#: rather than a sidebar slot it cannot use.
MIN_HOST_WIDTH = 1500
FLOATING_SIZE = (980, 460)

#: Control points shown for editing. The reformation samples far more finely; these
#: are handles on the curve, not the curve itself.
N_CONTROL_POINTS = 24

#: Re-rendering resamples the whole volume, so edits are coalesced rather than
#: run on every event of a drag or a brush stroke.
_RERENDER_MS = 250

#: Bounds on the station count offered in the panel. The floor is the shortest
#: run a frame can be built along; the ceiling keeps a typo from asking for a
#: reformation with more rows than the volume has voxels.
#: How long the controls sit still before a typed or dragged value is acted on.
_SETTLE_MS = 350

#: Floor on the station spacing, so a huge station count cannot ask for a step
#: of zero millimetres.
_MIN_STEP_MM = 0.01

MIN_STATIONS = 8
MAX_STATIONS = 20_000


def _is_left_mouse_button(event: Any) -> bool:
    """True for a left-click, across the button spellings Napari emits."""
    btn = getattr(event, "button", None)
    if btn in (0, 1, None):
        return True
    return str(btn).lower() in ("left", "lbutton", "mouse1")


def _remove_layers_named(viewer: Any, names) -> None:
    """Drop any of *names* that are currently in the viewer."""
    wanted = set(names)
    for layer in list(getattr(viewer, "layers", []) or []):
        if str(getattr(layer, "name", "")) in wanted:
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass


class VesselCprPanel(QWidget):
    """Flattened vessel beside its cross-section, with the controls that drive both.

    Laid out along the vessel: the flat image runs left-to-right with arc length on
    the x axis, the calibre profile sits directly under it on the same axis so a
    narrowing lines up with the place it happens, and the perpendicular
    cross-section stands to the right at the station under the cursor.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the controls, the flat-vessel canvas and the cross-section canvas."""
        super().__init__(parent)
        self._on_change: Any = None
        self._on_station: Any = None
        self._on_reset: Any = None
        self._on_stations: Any = None
        self._syncing = False
        self._vessel: VesselCpr | None = None
        self._edited = False
        self._failures: dict[int, str] = {}

        self._status = QLabel("Run the tool on a vessel mask to flatten it.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")

        self._vessel_combo = QComboBox()
        self._vessel_combo.setToolTip("Which vessel to flatten.")
        self._vessel_combo.setMinimumWidth(160)
        self._vessel_combo.currentIndexChanged.connect(lambda _i: self._emit_change())

        angle_tip = (
            "Rotate the cutting plane around the centerline. An eccentric narrowing "
            "can hide at one angle and be obvious at another."
        )
        self._angle = QSlider(Qt.Horizontal)
        self._angle.setRange(0, 179)
        self._angle.setMinimumWidth(110)
        self._angle.setToolTip(angle_tip)
        self._angle_spin = QSpinBox()
        self._angle_spin.setRange(0, 179)
        self._angle_spin.setSuffix("°")
        self._angle_spin.setToolTip(angle_tip)
        self._angle_spin.setKeyboardTracking(False)

        ray_tip = "How far either side of the centerline the flat view reaches."
        self._ray = QSlider(Qt.Horizontal)
        self._ray.setRange(2, 40)
        self._ray.setValue(int(DEFAULT_RAY_MM))
        self._ray.setMinimumWidth(110)
        self._ray.setToolTip(ray_tip)
        self._ray_spin = QDoubleSpinBox()
        self._ray_spin.setRange(2.0, 40.0)
        self._ray_spin.setDecimals(1)
        self._ray_spin.setSingleStep(0.5)
        self._ray_spin.setValue(float(DEFAULT_RAY_MM))
        self._ray_spin.setSuffix(" mm")
        self._ray_spin.setToolTip(ray_tip)
        self._ray_spin.setKeyboardTracking(False)

        self._stations_spin = QSpinBox()
        self._stations_spin.setRange(MIN_STATIONS, MAX_STATIONS)
        self._stations_spin.setValue(0)
        self._stations_spin.setEnabled(False)
        self._stations_spin.setKeyboardTracking(False)
        self._stations_spin.setToolTip(
            "How many stations the vessel is cut into. More stations sample the "
            "centerline more finely; the spacing in millimetres follows from the "
            "vessel's length."
        )

        # Sliders and boxes drive each other, and only the settled value starts a
        # resample: dragging a slider or typing a number would otherwise reformat
        # the whole volume once per intermediate value.
        self._angle.valueChanged.connect(
            lambda v: self._sync(self._angle_spin, int(v))
        )
        self._angle_spin.valueChanged.connect(
            lambda v: self._sync(self._angle, int(v))
        )
        self._ray.valueChanged.connect(
            lambda v: self._sync(self._ray_spin, float(v))
        )
        self._ray_spin.valueChanged.connect(
            lambda v: self._sync(self._ray, int(round(float(v))))
        )
        for widget in (self._angle, self._ray):
            widget.sliderReleased.connect(self._emit_change)
        for box in (self._angle_spin, self._ray_spin):
            box.editingFinished.connect(self._emit_change)
            box.valueChanged.connect(lambda _v: self._defer_change())
        self._stations_spin.valueChanged.connect(self._on_stations_changed)

        self._station = QSlider(Qt.Horizontal)
        self._station.setRange(0, 0)
        self._station.setToolTip("Station along the vessel. Click the flat image to jump.")
        self._station.valueChanged.connect(lambda v: self._emit_station(int(v)))
        self._station_label = QLabel("—")
        self._station_label.setMinimumWidth(150)
        self._station_label.setStyleSheet(f"color: {COLOR_MUTED};")

        self._show_lumen = QCheckBox("Lumen")
        self._show_lumen.setChecked(True)
        self._show_wall = QCheckBox("Wall")
        self._show_wall.setChecked(True)
        self._show_center = QCheckBox("Centerline")
        self._show_center.setChecked(True)
        self._show_profile = QCheckBox("Calibre")
        self._show_profile.setChecked(True)
        self._show_profile.setToolTip("Lumen diameter against distance along the vessel.")
        # Each toggle is tinted like the thing it draws, so the legend and the
        # image cannot disagree about which band is which.
        for box, hexed in (
            (self._show_lumen, LUMEN_HEX),
            (self._show_wall, WALL_HEX),
            (self._show_center, CENTERLINE_COLOR),
            (self._show_profile, LUMEN_HEX),
        ):
            box.setStyleSheet(f"color: {hexed};")
            box.toggled.connect(lambda _c: self.redraw())

        self._btn_render = QPushButton("Re-render")
        self._btn_render.setToolTip("Resample after editing the centerline or a mask.")
        self._btn_render.clicked.connect(lambda: self._emit_change())

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_SETTLE_MS)
        self._settle.timeout.connect(self._emit_change)

        self._btn_reset = QPushButton("Reset centerline")
        self._btn_reset.setToolTip(
            "Discard the control-point edits for this vessel and re-derive its "
            "centerline from the mask."
        )
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(lambda: self._emit_reset())

        root = QVBoxLayout(self)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._status)

        # Two short rows rather than one long bar: the dock can be dragged narrow,
        # and a single row collapses the toggles to unlabelled squares when it is.
        geometry = QHBoxLayout()
        geometry.setSpacing(SPACE_TIGHT)
        geometry.addWidget(self._caption("Vessel"))
        geometry.addWidget(self._vessel_combo, stretch=1)
        geometry.addSpacing(SPACE_TIGHT)
        geometry.addWidget(self._caption("Angle"))
        geometry.addWidget(self._angle, stretch=1)
        geometry.addWidget(self._angle_spin)
        geometry.addSpacing(SPACE_TIGHT)
        geometry.addWidget(self._caption("Ray"))
        geometry.addWidget(self._ray, stretch=1)
        geometry.addWidget(self._ray_spin)
        geometry.addSpacing(SPACE_TIGHT)
        geometry.addWidget(self._caption("Stations"))
        geometry.addWidget(self._stations_spin)
        root.addLayout(geometry)

        toggles = QHBoxLayout()
        toggles.setSpacing(SPACE_TIGHT)
        for box in (self._show_lumen, self._show_wall, self._show_center, self._show_profile):
            toggles.addWidget(box)
        toggles.addStretch(1)
        toggles.addWidget(self._btn_reset)
        toggles.addWidget(self._btn_render)
        root.addLayout(toggles)

        self._cpr_fig = self._cpr_ax = self._cpr_canvas = None
        self._profile_ax = None
        self._xs_fig = self._xs_ax = self._xs_canvas = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            # Flat vessel over its calibre profile, sharing the arc-length axis so a
            # narrowing on the profile sits directly under the place it happens.
            self._cpr_fig = Figure(figsize=(7.5, 3.4), dpi=96, layout="constrained")
            grid = self._cpr_fig.add_gridspec(2, 1, height_ratios=(3.0, 1.0))
            self._cpr_ax = self._cpr_fig.add_subplot(grid[0])
            self._profile_ax = self._cpr_fig.add_subplot(grid[1], sharex=self._cpr_ax)
            self._cpr_canvas = FigureCanvasQTAgg(self._cpr_fig)
            self._cpr_canvas.setMinimumHeight(220)
            self._cpr_canvas.setMinimumWidth(320)
            self._cpr_canvas.mpl_connect("button_press_event", self._on_canvas_click)
            self._cpr_canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

            self._xs_fig = Figure(figsize=(2.6, 2.6), dpi=96)
            self._xs_ax = self._xs_fig.add_subplot(111)
            self._xs_canvas = FigureCanvasQTAgg(self._xs_fig)
            self._xs_canvas.setMinimumWidth(180)
            self._xs_canvas.setMinimumHeight(180)

            for fig in (self._cpr_fig, self._xs_fig):
                style_image_figure(fig)

            split = QSplitter(Qt.Horizontal)
            split.addWidget(self._cpr_canvas)
            split.addWidget(self._xs_canvas)
            split.setStretchFactor(0, 4)
            split.setStretchFactor(1, 1)
            split.setChildrenCollapsible(False)
            root.addWidget(split, stretch=1)
        except Exception as exc:  # noqa: BLE001
            root.addWidget(QLabel(f"Matplotlib unavailable: {exc}"))

        station_row = QHBoxLayout()
        station_row.setSpacing(SPACE_TIGHT)
        station_row.addWidget(self._caption("Station"))
        station_row.addWidget(self._station, stretch=1)
        station_row.addWidget(self._station_label)
        root.addLayout(station_row)

    # ── layout helpers ───────────────────────────────────────────────────────

    def _caption(self, text: str) -> QLabel:
        """A muted field caption for the control bar."""
        caption = QLabel(text)
        caption.setStyleSheet(f"color: {COLOR_MUTED};")
        return caption

    # ── wiring ───────────────────────────────────────────────────────────────

    def set_callbacks(
        self,
        *,
        on_change: Any = None,
        on_station: Any = None,
        on_reset: Any = None,
        on_stations: Any = None,
    ) -> None:
        """Register what to call when the controls move."""
        self._on_change = on_change
        self._on_station = on_station
        self._on_reset = on_reset
        self._on_stations = on_stations

    def _emit_reset(self) -> None:
        """Ask the installer to throw this vessel's centerline edits away."""
        if self._on_reset is not None:
            self._on_reset()

    def set_edited(self, edited: bool) -> None:
        """Show whether the vessel on screen is running on a retouched centerline."""
        self._edited = bool(edited)
        self._btn_reset.setEnabled(bool(edited))

    def _sync(self, widget: QWidget, value: Any) -> None:
        """Mirror a value onto the partner widget without it echoing back."""
        if self._syncing:
            return
        self._syncing = True
        try:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        finally:
            self._syncing = False

    def _defer_change(self) -> None:
        """Resample once the controls settle, rather than on every step of a drag."""
        if self._angle.isSliderDown() or self._ray.isSliderDown():
            return
        self._settle.start()

    def _emit_change(self) -> None:
        """Ask the installer to resample with the current settings."""
        self._settle.stop()
        if self._on_change is not None:
            self._on_change()

    def _on_stations_changed(self, value: int) -> None:
        """Re-cut the vessel into *value* stations."""
        if self._syncing or not self._stations_spin.isEnabled():
            return
        if self._on_stations is not None:
            self._on_stations(int(value))

    def _emit_station(self, station: int) -> None:
        """Tell the installer the current station moved."""
        if self._on_station is not None:
            self._on_station(int(station))

    def _station_at(self, event: Any) -> int | None:
        """The station index under a mouse *event* on either arc-length axis."""
        if self._vessel is None or event.xdata is None:
            return None
        if event.inaxes not in (self._cpr_ax, self._profile_ax):
            return None
        with using("cpu"):
            arc = to_numpy(self._vessel.cpr.arc_length_mm)
            if arc.size == 0:
                return None
            return int(np.clip(np.searchsorted(arc, float(event.xdata)), 0, arc.size - 1))

    def _on_canvas_click(self, event: Any) -> None:
        """Clicking a column of the flat image selects that station."""
        station = self._station_at(event)
        if station is not None:
            self._station.setValue(station)

    def _on_canvas_motion(self, event: Any) -> None:
        """Read out the calibre under the cursor without committing to a station."""
        station = self._station_at(event)
        if station is None or self._vessel is None:
            return
        with using("cpu"):
            arc = to_numpy(self._vessel.cpr.arc_length_mm)
            width = to_numpy(self._vessel.diameter_mm())
        at = float(arc[station]) if station < arc.size else float("nan")
        here = float(width[station]) if station < width.size else float("nan")
        self._station_label.setText(f"{at:.1f} mm · lumen {here:.1f} mm")

    # ── state read by the installer ──────────────────────────────────────────

    def angle_deg(self) -> float:
        """Current cutting angle."""
        return float(self._angle_spin.value())

    def ray_mm(self) -> float:
        """Current ray half-length in millimetres."""
        return float(self._ray_spin.value())

    def station(self) -> int:
        """Current station index."""
        return int(self._station.value())

    def selected_label(self) -> int | None:
        """Label of the vessel being shown, or ``None`` when there is none."""
        data = self._vessel_combo.currentData()
        return None if data is None else int(data)

    def set_vessels(self, entries: list[tuple[int, str]]) -> None:
        """Populate the vessel picker, keeping the current choice where possible."""
        previous = self.selected_label()
        self._vessel_combo.blockSignals(True)
        self._vessel_combo.clear()
        for label, name in entries:
            self._vessel_combo.addItem(f"{name}  (label {label})", int(label))
        index = self._vessel_combo.findData(previous)
        self._vessel_combo.setCurrentIndex(max(index, 0))
        self._vessel_combo.blockSignals(False)

    def set_failure_reasons(self, reasons: dict[int, str]) -> None:
        """Record why some labels produced no centerline, for the caller to report."""
        self._failures = {int(k): str(v) for k, v in (reasons or {}).items()}

    def failure_reasons(self) -> dict[int, str]:
        """Labels that produced no centerline, mapped to why."""
        return dict(self._failures)

    # ── rendering ────────────────────────────────────────────────────────────

    def show_vessel(self, vessel: VesselCpr | None, *, message: str = "") -> None:
        """Display *vessel*, or clear to *message* when there is none."""
        self._vessel = vessel
        if vessel is None:
            if not message and self._failures:
                message = "No centerline: " + "; ".join(
                    f"label {lab} — {why}" for lab, why in self._failures.items()
                )
            self._status.setText(message or "No vessel to show.")
            self._station_label.setText("—")
            self._stations_spin.setEnabled(False)
            self._station.blockSignals(True)
            self._station.setRange(0, 0)
            self._station.blockSignals(False)
            self._clear_axes()
            return

        self._station.blockSignals(True)
        self._station.setRange(0, max(vessel.n_stations - 1, 0))
        if self._station.value() > vessel.n_stations - 1:
            self._station.setValue(max(vessel.n_stations - 1, 0))
        self._station.blockSignals(False)

        self._syncing = True
        try:
            self._stations_spin.setEnabled(True)
            self._stations_spin.setValue(
                int(min(max(vessel.n_stations, MIN_STATIONS), MAX_STATIONS))
            )
        finally:
            self._syncing = False

        diameters = to_numpy(vessel.diameter_mm())
        narrowest = float(diameters.min()) if diameters.size else float("nan")
        step = vessel.length_mm / max(vessel.n_stations - 1, 1)
        text = (
            f"{vessel.name} — {vessel.length_mm:.1f} mm long, "
            f"{vessel.n_stations} stations ({step:.2f} mm apart), "
            f"narrowest {narrowest:.1f} mm"
        )
        if self._edited:
            text += "  ·  centerline retouched"
        if vessel.cpr.fold_warning:
            text += f"  ·  {vessel.cpr.fold_warning}"
        if self._failures:
            text += "  ·  no centerline for label(s) " + ", ".join(
                str(lab) for lab in sorted(self._failures)
            )
        self._status.setText(text)
        self.redraw()

    def redraw(self) -> None:
        """Repaint the flat vessel and its calibre profile for the current vessel."""
        if self._vessel is None:
            return
        self._draw_cpr(self._vessel)

    def _clear_axes(self) -> None:
        """Blank every canvas."""
        for ax in (self._cpr_ax, self._profile_ax, self._xs_ax):
            if ax is not None:
                ax.clear()
                ax.set_axis_off()
        for fig, canvas in ((self._cpr_fig, self._cpr_canvas), (self._xs_fig, self._xs_canvas)):
            if canvas is not None:
                style_image_figure(fig)
                canvas.draw_idle()

    def _draw_cpr(self, vessel: VesselCpr) -> None:
        """Draw the flattened vessel running left-to-right, over its calibre profile."""
        if self._cpr_ax is None or self._cpr_canvas is None:
            return
        ax = self._cpr_ax
        ax.clear()
        ax.set_axis_on()

        # Matplotlib needs host arrays; this is the display boundary. The
        # reformation is stored station-major, so it is transposed to put arc
        # length on x and the ray across the vessel on y.
        with using("cpu"):
            image = to_numpy(vessel.cpr.image).T
            arc = to_numpy(vessel.cpr.arc_length_mm)
            ray = to_numpy(vessel.cpr.ray_mm)
            finite = image[np.isfinite(image)]
            # Percentile limits: a single bright voxel would otherwise flatten the
            # whole vessel to mid-grey.
            vmin, vmax = (
                (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0)))
                if finite.size
                else (0.0, 1.0)
            )
        if not vmax > vmin:
            vmin, vmax = None, None
        extent = [float(arc[0]), float(arc[-1]), float(ray[0]), float(ray[-1])]
        ax.imshow(
            image,
            cmap="gray",
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        if self._show_wall.isChecked() and vessel.wall is not None:
            self._overlay(ax, to_numpy(vessel.wall).T, extent, WALL_RGB, 0.35)
        if self._show_lumen.isChecked():
            self._overlay(ax, to_numpy(vessel.lumen).T, extent, LUMEN_RGB, 0.35)
        if self._show_center.isChecked():
            ax.axhline(0.0, color=CENTERLINE_COLOR, lw=0.8, alpha=0.9)

        station = self.station()
        if 0 <= station < arc.size:
            ax.axvline(float(arc[station]), color=STATION_COLOR, lw=1.2)

        ax.set_ylabel("across (mm)", fontsize=8)
        ax.tick_params(labelsize=7, labelbottom=False)
        self._draw_profile(vessel, arc, station)
        style_image_figure(self._cpr_fig)
        self._cpr_canvas.draw_idle()

    def _draw_profile(self, vessel: VesselCpr, arc: Any, station: int) -> None:
        """Draw lumen diameter against distance along the vessel under the flat image."""
        ax = self._profile_ax
        if ax is None:
            return
        ax.clear()
        if not self._show_profile.isChecked():
            ax.set_axis_off()
            # The flat image keeps the ticks when the profile is hidden.
            if self._cpr_ax is not None:
                self._cpr_ax.tick_params(labelbottom=True)
                self._cpr_ax.set_xlabel("along the vessel (mm)", fontsize=8)
            return
        ax.set_axis_on()
        with using("cpu"):
            width = to_numpy(vessel.diameter_mm())
            n = int(min(arc.size, width.size))
        if n:
            ax.plot(arc[:n], width[:n], color=LUMEN_HEX, lw=1.0)
            ax.fill_between(arc[:n], 0.0, width[:n], color=LUMEN_HEX, alpha=0.18)
            ax.set_ylim(0.0, float(width[:n].max()) * 1.15 or 1.0)
        if 0 <= station < arc.size:
            ax.axvline(float(arc[station]), color=STATION_COLOR, lw=1.2)
        ax.set_xlabel("along the vessel (mm)", fontsize=8)
        ax.set_ylabel("Ø mm", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)

    @staticmethod
    def _overlay(ax: Any, mask: Any, extent: list[float], rgb: tuple, alpha: float) -> None:
        """Paint a translucent mask over the flat image."""
        with using("cpu"):
            arr = to_numpy(mask)
            if not arr.any():
                return
            rgba = np.zeros((*arr.shape, 4), dtype=float)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
            rgba[..., 3] = (arr > 0).astype(float) * float(alpha)
        ax.imshow(rgba, aspect="auto", origin="lower", extent=extent, interpolation="nearest")

    def show_cross_section(self, image: Any, mask: Any = None, *, title: str = "") -> None:
        """Draw the perpendicular cross-section at the current station."""
        self._station_label.setText(title or "—")
        if self._xs_ax is None or self._xs_canvas is None:
            return
        ax = self._xs_ax
        ax.clear()
        ax.set_axis_off()
        with using("cpu"):
            arr = to_numpy(image)
            overlay = None
            if mask is not None:
                m = to_numpy(mask)
                if m.any():
                    overlay = np.zeros((*m.shape, 4), dtype=float)
                    overlay[..., 0] = LUMEN_RGB[0]
                    overlay[..., 3] = (m > 0).astype(float) * 0.35
        if arr.size:
            ax.imshow(arr, cmap="gray", origin="lower", interpolation="nearest")
            if overlay is not None:
                ax.imshow(overlay, origin="lower", interpolation="nearest")
        if title:
            ax.set_title(title, fontsize=8)
        style_image_figure(self._xs_fig)
        self._xs_fig.tight_layout(pad=0.2)
        self._xs_canvas.draw_idle()


def attach_vessel_cpr_dock(viewer: Any, panel: VesselCprPanel) -> Any:
    """Dock *panel* on Napari's left edge, floating it when the window is too narrow.

    The flat vessel wants width, and Napari's left column is a sidebar. Rather than
    squeeze the reformation into it, the dock is floated as its own window whenever
    the main window cannot spare :data:`MIN_DOCKED_WIDTH` for it.
    """
    apply_theme(panel)
    dock = attach_left_inspection_dock(
        viewer,
        panel,
        object_name=DOCK_OBJECT_NAME,
        title="Vessel CPR",
        tabify_with=["nvitk_vessel_cross_section_dock", "nvitk_hemodynamics_plot_dock"],
        minimum_width=420,
    )
    _float_if_narrow(viewer, dock)
    return dock


def _float_if_narrow(viewer: Any, dock: Any) -> None:
    """Float *dock* as its own window when the Napari window is too narrow to host it."""
    if dock is None or not hasattr(dock, "setFloating"):
        return
    try:
        win = viewer.window._qt_window
    except Exception:
        return
    if int(win.width()) >= MIN_HOST_WIDTH:
        return
    try:
        dock.setFloating(True)
        dock.resize(FLOATING_SIZE[0], FLOATING_SIZE[1])
        origin = win.mapToGlobal(win.rect().topLeft())
        dock.move(origin.x() + 60, origin.y() + 80)
        dock.show()
        dock.raise_()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Installer: panel + 3D overlays + editing
# ──────────────────────────────────────────────────────────────────────────────
def _control_point_indices(n_stations: int, n_wanted: int = N_CONTROL_POINTS) -> Any:
    """Evenly spaced station indices to expose as draggable handles."""
    with using("cpu"):
        n = max(int(n_stations), 2)
        return np.unique(
            np.linspace(0, n - 1, min(int(n_wanted), n)).round().astype(int)
        )


def _grid_ratio(layer_data: Any, mask: Any) -> Any:
    """Per-axis factor taking *mask*-grid voxel coordinates back to *layer_data*'s."""
    with using("cpu"):
        base = tuple(int(v) for v in getattr(layer_data, "shape", ()) or ())[:3]
        fine = tuple(int(v) for v in getattr(mask, "shape", ()) or ())[:3]
        if len(base) != 3 or len(fine) != 3 or base == fine:
            return np.ones((3,), dtype=float)
        return np.array(
            [max(b - 1, 1) / max(f - 1, 1) for b, f in zip(base, fine)], dtype=float
        )


def install_vessel_cpr(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    lumen_layer: Any,
    lumen_mask: Any,
    image: Any = None,
    wall_mask: Any = None,
    labels=None,
    centerline_mask: Any = None,
    spacing=None,
    step_mm: float = 0.5,
    join_gap_vox: int = DEFAULT_JOIN_GAP_VOX,
) -> VesselCprPanel:
    """Flatten the requested vessels and wire the panel, overlays and edits.

    The lumen is the required geometry: it defines the grid everything else is
    resampled onto, and it is what the centerline is traced through. Without an
    image the mask itself is reformatted, which still shows the vessel's course and
    calibre. State lives in ``app_state["vessel_cpr"]`` so a second run tears the
    first one down cleanly rather than stacking overlays and callbacks.
    """
    from nvitk.gui.core.spatial import layer_spacing, layer_spatial_kwargs

    shutdown_vessel_cpr(app_state)

    mask = to_numpy(lumen_mask)
    grey = mask.astype("float32") if image is None else to_numpy(image)
    sp = tuple(spacing) if spacing is not None else tuple(
        (layer_spacing(lumen_layer) or (1.0, 1.0, 1.0))[:3]
    )

    panel = VesselCprPanel()
    dock = attach_vessel_cpr_dock(viewer, panel)

    state: dict[str, Any] = {
        "viewer": viewer,
        "panel": panel,
        "dock": dock,
        "image": grey,
        "mask": mask,
        "wall": None if wall_mask is None else to_numpy(wall_mask),
        "spacing": sp,
        "step_mm": float(step_mm),
        "join_gap_vox": int(join_gap_vox),
        "labels": labels,
        "centerline_mask": None if centerline_mask is None else to_numpy(centerline_mask),
        "spatial": layer_spatial_kwargs(lumen_layer),
        # Overlays are added with the lumen layer's own scale/affine, so points
        # computed on an upsampled grid have to come back to the layer's voxel
        # coordinates first. A zoom aligns first and last voxel centres, so the
        # ratio is over the spans, not the sizes.
        "overlay_scale": _grid_ratio(getattr(lumen_layer, "data", None), mask),
        "lumen_layer": lumen_layer,
        "samples": {},
        "vessel": None,
        "handles": None,
        "handles_stale": True,
        "suspend_edit": False,
        "edited": {},
    }

    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(_RERENDER_MS)
    state["timer"] = timer

    def _derive_centerlines() -> None:
        """(Re)derive centerlines for the requested labels."""
        state["handles_stale"] = True
        reasons: dict[int, str] = {}
        state["samples"] = centerlines_for_labels(
            state["mask"],
            spacing=state["spacing"],
            labels=state["labels"],
            centerline_mask=state["centerline_mask"],
            step_mm=state["step_mm"],
            reasons=reasons,
            join_gap_vox=state["join_gap_vox"],
        )
        panel.set_failure_reasons(reasons)
        panel.set_vessels([(lab, vessel_name(lab)) for lab in state["samples"]])

    def _keep_selection(fn) -> None:
        """Run *fn*, then put the viewer's active layer back where it was.

        Adding a layer makes it active in Napari, so an overlay refresh would
        otherwise steal the selection from whatever the user was working on.
        """
        try:
            keep = viewer.layers.selection.active
        except Exception:
            keep = None
        try:
            fn()
        finally:
            if keep is not None:
                try:
                    if keep in list(viewer.layers):
                        viewer.layers.selection.active = keep
                except Exception:
                    pass

    def _render() -> None:
        """Resample the selected vessel and refresh the panel and overlays."""
        label = panel.selected_label()
        samples = state["samples"].get(label) if label is not None else None
        if samples is None:
            # Say which vessel failed and why, rather than a blanket "no centerline".
            why = panel.failure_reasons().get(label) if label is not None else None
            message = (
                f"No centerline for label {label}: {why}"
                if why
                else "No centerline could be derived from this mask."
            )
            panel.show_vessel(None, message=message)
            _clear_overlays()
            return
        # An edited centerline replaces the derived one for that vessel.
        panel.set_edited(int(label) in state["edited"])
        samples = state["edited"].get(label, samples)
        vessel = build_vessel_cpr(
            state["image"],
            state["mask"],
            label=int(label),
            samples=samples,
            wall_mask=state["wall"],
            angle_deg=panel.angle_deg(),
            ray_mm=panel.ray_mm(),
            n_ray=DEFAULT_N_RAY,
        )
        state["vessel"] = vessel
        panel.show_vessel(vessel)
        _keep_selection(lambda: _update_overlays(vessel))
        _show_station(panel.station())

    def _show_station(station: int) -> None:
        """Draw the cross-section at *station* and move the 3D marker to it."""
        vessel = state["vessel"]
        if vessel is None:
            return
        image_xs, mask_xs = vessel_cross_section(
            state["image"], vessel, station, ray_mm=panel.ray_mm(), lumen_mask=state["mask"]
        )
        arc = to_numpy(vessel.cpr.arc_length_mm)
        at = float(arc[station]) if 0 <= station < arc.size else 0.0
        width = to_numpy(vessel.diameter_mm())
        here = float(width[station]) if station < width.size else float("nan")
        panel.show_cross_section(
            image_xs, mask_xs, title=f"{at:.1f} mm along — lumen {here:.1f} mm"
        )
        _keep_selection(lambda: _update_station_overlays(vessel, station))
        panel.redraw()

    # ── overlays ─────────────────────────────────────────────────────────────

    def _layer_coords(points_vox: Any) -> Any:
        """Fine-grid voxel coordinates as the lumen layer's, for a Napari overlay."""
        with using("cpu"):
            return (to_numpy(points_vox) * to_numpy(state["overlay_scale"])).astype(
                "float32", copy=False
            )

    def _grid_coords(points_layer_vox: Any) -> Any:
        """The inverse: points read back off an overlay, onto the working grid."""
        with using("cpu"):
            return to_numpy(points_layer_vox).astype(float, copy=False) / to_numpy(
                state["overlay_scale"]
            )

    def _clear_overlays() -> None:
        """Remove every overlay layer the panel owns."""
        state["handles"] = None
        _remove_layers_named(viewer, OVERLAY_LAYERS)

    def _existing(name: str) -> Any:
        """The overlay layer called *name* if it is still in the viewer."""
        for lyr in getattr(viewer, "layers", []) or []:
            if str(getattr(lyr, "name", "")) == name:
                return lyr
        return None

    def _update_overlays(vessel: VesselCpr) -> None:
        """Draw the centerline and its draggable control points in 3D.

        The layers are updated in place rather than dropped and re-added. Re-adding
        them costs the user their work twice over: Napari selects a newly added
        layer, so the selection jumps to whichever overlay went in last, and a
        fresh Points layer comes up in pan/zoom — so the select tool the drag
        needs is gone the moment the drag is acted on.
        """
        # Napari layers hold host arrays.
        pts = _layer_coords(vessel.samples.points_vox)
        spatial = state["spatial"]

        line = _existing(CPR_CENTERLINE)
        try:
            if line is None:
                line = viewer.add_shapes(
                    [pts],
                    shape_type="path",
                    name=CPR_CENTERLINE,
                    edge_color=CENTERLINE_COLOR,
                    edge_width=0.4,
                    **spatial,
                )
                line.editable = False
            else:
                line.data = [pts]
        except Exception:
            pass

        handles = _existing(CPR_CONTROL_POINTS)
        idx = _control_point_indices(vessel.n_stations)
        # Handles follow a centerline that was re-derived, but not one the user is
        # in the middle of editing: snapping them onto the resampled curve after
        # every drag would tug each point away from where it was just put.
        refresh = (
            bool(state.get("handles_stale"))
            or state.get("control_label") != vessel.label
            or handles is None
            or int(getattr(handles.data, "shape", (0,))[0]) != int(idx.size)
        )
        state["suspend_edit"] = True
        try:
            if handles is None:
                handles = viewer.add_points(
                    pts[idx],
                    name=CPR_CONTROL_POINTS,
                    size=1.6,
                    face_color=STATION_COLOR,
                    border_width=0,
                    **spatial,
                )
                # Napari Points are draggable in select mode; that is the editing
                # gesture, so no custom mouse handling is needed here.
                handles.events.data.connect(_on_control_points_moved)
            elif refresh:
                handles.data = pts[idx]
        except Exception:
            handles = None
        finally:
            state["suspend_edit"] = False
        state["handles_stale"] = False
        state["control_indices"] = idx
        state["control_label"] = vessel.label
        state["handles"] = handles

    def _update_station_overlays(vessel: VesselCpr, station: int) -> None:
        """Move the station marker and the cross-section plane in 3D."""
        spatial = state["spatial"]
        pts = _layer_coords(vessel.samples.points_vox)
        idx = int(max(0, min(int(station), pts.shape[0] - 1)))

        marker = _existing(CPR_STATION)
        try:
            if marker is None:
                marker = viewer.add_points(
                    pts[idx][None, :],
                    name=CPR_STATION,
                    size=2.4,
                    face_color=STATION_COLOR,
                    border_width=0,
                    **spatial,
                )
                marker.editable = False
            else:
                marker.data = pts[idx][None, :]
        except Exception:
            pass

        corners = _layer_coords(plane_corners(vessel, idx, ray_mm=panel.ray_mm()))
        square = _existing(CPR_PLANE)
        try:
            if square is None:
                square = viewer.add_shapes(
                    [corners],
                    shape_type="polygon",
                    name=CPR_PLANE,
                    edge_color=STATION_COLOR,
                    face_color="transparent",
                    edge_width=0.3,
                    **spatial,
                )
                square.editable = False
            else:
                square.data = [corners]
        except Exception:
            pass

    # ── editing ──────────────────────────────────────────────────────────────

    def _on_control_points_moved(_event: Any = None) -> None:
        """Re-fit the centerline through the moved handles, then re-render."""
        if state.get("suspend_edit"):
            return
        handles = state.get("handles")
        label = state.get("control_label")
        if handles is None or label is None:
            return
        moved = _grid_coords(handles.data)
        if moved.shape[0] < 2:
            return
        try:
            # Interpolate the handles, do not smooth them. The default smoothing is
            # tuned for a raw skeleton — hundreds of staircased voxels that need
            # it — and applied to two dozen points the user placed deliberately it
            # suppresses the edit: a 3-voxel drag comes back as 0.7. The handles
            # are already the intended curve, so pass them through.
            state["edited"][int(label)] = resample_centerline(
                moved, state["spacing"], step_mm=state["step_mm"], smooth=0.0
            )
        except Exception:
            return
        timer.start()

    def _on_mask_changed(_event: Any = None) -> None:
        """Pick up a brush stroke on the lumen layer and re-render."""
        layer = state.get("lumen_layer")
        if layer is None:
            return
        fresh = to_numpy(layer.data)
        if tuple(fresh.shape) != tuple(state["mask"].shape):
            from nvitk.viz.vessel_cpr import upsample_volume

            fresh = to_numpy(upsample_volume(fresh, state["mask"].shape, order=0))
        state["mask"] = fresh
        # Geometry may have changed under the centerline, so re-derive it too.
        state["edited"].clear()
        _derive_centerlines()
        timer.start()

    def _set_stations(count: int) -> None:
        """Re-cut the current vessel into *count* stations.

        The station count is the spacing seen from the other side: the vessel's
        length is fixed, so asking for more stations is asking for a finer step.
        Edited centerlines are re-resampled through their own points rather than
        discarded — a retouch should survive a change of resolution.
        """
        label = panel.selected_label()
        samples = None
        if label is not None:
            samples = state["edited"].get(int(label)) or state["samples"].get(int(label))
        if samples is None:
            return
        length = float(samples.length_mm)
        if length <= 0.0:
            return
        state["step_mm"] = max(length / max(int(count) - 1, 1), _MIN_STEP_MM)

        for lab, edited in list(state["edited"].items()):
            try:
                state["edited"][lab] = resample_centerline(
                    edited.points_vox,
                    state["spacing"],
                    step_mm=state["step_mm"],
                    smooth=0.0,
                )
            except Exception:
                state["edited"].pop(lab, None)
        _derive_centerlines()
        _render()

    def _reset_centerline() -> None:
        """Throw away this vessel's control-point edits and re-derive it."""
        label = panel.selected_label()
        if label is None:
            return
        state["edited"].pop(int(label), None)
        panel.set_edited(False)
        _derive_centerlines()
        _render()

    timer.timeout.connect(_render)
    panel.set_callbacks(
        on_change=_render,
        on_station=_show_station,
        on_reset=_reset_centerline,
        on_stations=_set_stations,
    )

    if lumen_layer is not None:
        try:
            lumen_layer.events.data.connect(_on_mask_changed)
            state["mask_callback"] = _on_mask_changed
        except Exception:
            pass

    _derive_centerlines()
    _render()

    app_state["vessel_cpr"] = state
    return panel


def shutdown_vessel_cpr(app_state: dict[str, Any]) -> None:
    """Tear down a previous run: overlays, callbacks, timer and dock."""
    state = app_state.pop("vessel_cpr", None)
    if not isinstance(state, dict):
        return
    viewer = state.get("viewer")

    timer = state.get("timer")
    if timer is not None:
        try:
            timer.stop()
        except Exception:
            pass

    handles = state.get("handles")
    if handles is not None:
        try:
            handles.events.data.disconnect()
        except Exception:
            pass

    layer = state.get("lumen_layer")
    callback = state.get("mask_callback")
    if layer is not None and callback is not None:
        try:
            layer.events.data.disconnect(callback)
        except Exception:
            pass

    if viewer is not None:
        _remove_layers_named(viewer, OVERLAY_LAYERS)

    dock = state.get("dock")
    if dock is not None:
        try:
            dock.close()
        except Exception:
            pass


__all__ = [
    "CPR_CENTERLINE",
    "CPR_CONTROL_POINTS",
    "CPR_PLANE",
    "CPR_STATION",
    "DOCK_OBJECT_NAME",
    "OVERLAY_LAYERS",
    "VesselCprPanel",
    "attach_vessel_cpr_dock",
    "install_vessel_cpr",
    "shutdown_vessel_cpr",
]
