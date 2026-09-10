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
    QSlider,
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
        self._vessel: VesselCpr | None = None
        self._failures: dict[int, str] = {}

        self._status = QLabel("Run the tool on a vessel mask to flatten it.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")

        self._vessel_combo = QComboBox()
        self._vessel_combo.setToolTip("Which vessel to flatten.")
        self._vessel_combo.setMinimumWidth(160)
        self._vessel_combo.currentIndexChanged.connect(lambda _i: self._emit_change())

        self._angle = QSlider(Qt.Horizontal)
        self._angle.setRange(0, 179)
        self._angle.setMinimumWidth(110)
        self._angle.setToolTip(
            "Rotate the cutting plane around the centerline. An eccentric narrowing "
            "can hide at one angle and be obvious at another."
        )
        self._angle_label = QLabel("0°")
        self._angle_label.setMinimumWidth(38)
        self._angle.valueChanged.connect(self._on_angle_moved)

        self._ray = QSlider(Qt.Horizontal)
        self._ray.setRange(2, 40)
        self._ray.setValue(int(DEFAULT_RAY_MM))
        self._ray.setMinimumWidth(110)
        self._ray.setToolTip("How far either side of the centerline the flat view reaches.")
        self._ray_label = QLabel(f"{int(DEFAULT_RAY_MM)} mm")
        self._ray_label.setMinimumWidth(48)
        self._ray.valueChanged.connect(self._on_ray_moved)

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
        geometry.addWidget(self._angle_label)
        geometry.addSpacing(SPACE_TIGHT)
        geometry.addWidget(self._caption("Ray"))
        geometry.addWidget(self._ray, stretch=1)
        geometry.addWidget(self._ray_label)
        root.addLayout(geometry)

        toggles = QHBoxLayout()
        toggles.setSpacing(SPACE_TIGHT)
        for box in (self._show_lumen, self._show_wall, self._show_center, self._show_profile):
            toggles.addWidget(box)
        toggles.addStretch(1)
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
            self._cpr_fig = Figure(figsize=(7.5, 3.4), dpi=96)
            grid = self._cpr_fig.add_gridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.08)
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

    def set_callbacks(self, *, on_change: Any = None, on_station: Any = None) -> None:
        """Register what to call when the controls move."""
        self._on_change = on_change
        self._on_station = on_station

    def _emit_change(self) -> None:
        """Ask the installer to resample with the current settings."""
        if self._on_change is not None:
            self._on_change()

    def _emit_station(self, station: int) -> None:
        """Tell the installer the current station moved."""
        if self._on_station is not None:
            self._on_station(int(station))

    def _on_angle_moved(self, value: int) -> None:
        """Update the readout, then resample at the new angle."""
        self._angle_label.setText(f"{int(value)}°")
        self._emit_change()

    def _on_ray_moved(self, value: int) -> None:
        """Update the readout, then resample at the new ray length."""
        self._ray_label.setText(f"{int(value)} mm")
        self._emit_change()

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
        return float(self._angle.value())

    def ray_mm(self) -> float:
        """Current ray half-length in millimetres."""
        return float(self._ray.value())

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

        diameters = to_numpy(vessel.diameter_mm())
        narrowest = float(diameters.min()) if diameters.size else float("nan")
        text = (
            f"{vessel.name} — {vessel.length_mm:.1f} mm long, "
            f"{vessel.n_stations} stations, narrowest {narrowest:.1f} mm"
        )
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
        self._cpr_fig.tight_layout(pad=0.3)
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
        "labels": labels,
        "centerline_mask": None if centerline_mask is None else to_numpy(centerline_mask),
        "spatial": layer_spatial_kwargs(lumen_layer),
        "lumen_layer": lumen_layer,
        "samples": {},
        "vessel": None,
        "edited": {},
    }

    timer = QTimer()
    timer.setSingleShot(True)
    timer.setInterval(_RERENDER_MS)
    state["timer"] = timer

    def _derive_centerlines() -> None:
        """(Re)derive centerlines for the requested labels."""
        reasons: dict[int, str] = {}
        state["samples"] = centerlines_for_labels(
            state["mask"],
            spacing=state["spacing"],
            labels=state["labels"],
            centerline_mask=state["centerline_mask"],
            step_mm=state["step_mm"],
            reasons=reasons,
        )
        panel.set_failure_reasons(reasons)
        panel.set_vessels([(lab, vessel_name(lab)) for lab in state["samples"]])

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
        _update_overlays(vessel)
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
        _update_station_overlays(vessel, station)
        panel.redraw()

    # ── overlays ─────────────────────────────────────────────────────────────

    def _clear_overlays() -> None:
        """Remove every overlay layer the panel owns."""
        _remove_layers_named(viewer, OVERLAY_LAYERS)

    def _update_overlays(vessel: VesselCpr) -> None:
        """Draw the centerline and its draggable control points in 3D."""
        # Napari layers hold host arrays.
        pts = to_numpy(vessel.samples.points_vox).astype("float32", copy=False)
        spatial = state["spatial"]
        _remove_layers_named(viewer, (CPR_CENTERLINE, CPR_CONTROL_POINTS))
        try:
            line = viewer.add_shapes(
                [pts],
                shape_type="path",
                name=CPR_CENTERLINE,
                edge_color=CENTERLINE_COLOR,
                edge_width=0.4,
                **spatial,
            )
            line.editable = False
        except Exception:
            pass
        try:
            idx = _control_point_indices(vessel.n_stations)
            handles = viewer.add_points(
                pts[idx],
                name=CPR_CONTROL_POINTS,
                size=1.6,
                face_color=STATION_COLOR,
                border_width=0,
                **spatial,
            )
            state["control_indices"] = idx
            state["control_label"] = vessel.label
            # Napari Points are draggable in select mode; that is the editing
            # gesture, so no custom mouse handling is needed here.
            handles.events.data.connect(_on_control_points_moved)
            state["handles"] = handles
        except Exception:
            pass

    def _update_station_overlays(vessel: VesselCpr, station: int) -> None:
        """Move the station marker and the cross-section plane in 3D."""
        spatial = state["spatial"]
        _remove_layers_named(viewer, (CPR_STATION, CPR_PLANE))
        pts = to_numpy(vessel.samples.points_vox).astype("float32", copy=False)
        idx = int(max(0, min(int(station), pts.shape[0] - 1)))
        try:
            marker = viewer.add_points(
                pts[idx][None, :],
                name=CPR_STATION,
                size=2.4,
                face_color="#ffa400",
                border_width=0,
                **spatial,
            )
            marker.editable = False
        except Exception:
            pass
        try:
            square = viewer.add_shapes(
                [plane_corners(vessel, idx, ray_mm=panel.ray_mm())],
                shape_type="polygon",
                name=CPR_PLANE,
                edge_color="#ffa400",
                face_color="transparent",
                edge_width=0.3,
                **spatial,
            )
            square.editable = False
        except Exception:
            pass

    # ── editing ──────────────────────────────────────────────────────────────

    def _on_control_points_moved(_event: Any = None) -> None:
        """Re-fit the centerline through the moved handles, then re-render."""
        handles = state.get("handles")
        label = state.get("control_label")
        if handles is None or label is None:
            return
        moved = to_numpy(handles.data).astype(float, copy=False)
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
        state["mask"] = to_numpy(layer.data)
        # Geometry may have changed under the centerline, so re-derive it too.
        state["edited"].clear()
        _derive_centerlines()
        timer.start()

    timer.timeout.connect(_render)
    panel.set_callbacks(on_change=_render, on_station=_show_station)

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
