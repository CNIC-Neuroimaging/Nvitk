"""Napari dock showing a vessel flattened by curved planar reformation.

The upper canvas is the vessel drawn straight: rows are stations along the
centerline, spaced uniformly in millimetres, so a lesion's length and the calibre
on either side of it are readable off the vertical axis. Lumen and wall are drawn
over it. Clicking a row picks a station; the lower canvas then shows that station's
true perpendicular cross-section, and the 3D view marks where it is.

The centerline is exposed as an editable Points layer. Napari already lets a Points
layer be dragged in select mode, so retouching is: switch that layer to select,
move a control point, and the reformation re-renders through the moved curve.

The lumen mask is retouched right on the flat image or on the cross-section
(*Paint* / *Erase*): every point under the brush there is a point in the volume,
so a stroke paints (or erases) a ball of the brush's radius, in millimetres,
around each one — into the lumen layer itself, as one undo step, keeping the
other labels. Napari's own brush on the volume works too; either way the panel
re-renders when the mask changes, through the centerline it has.

An intensity image with time (3D+t) plays: the frames are sampled straight from
the image layer along the reformation (no volume is resampled per frame), and
the play button, the frame slider and Napari's own time slider follow each other.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.gui.core.design import (
    COLOR_MUTED,
    SPACE_TIGHT,
    apply_theme,
    style_image_figure,
)
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.transform.cpr import cut_direction, resample_centerline
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

#: A drag shorter than this is a click that wandered, not a correction.
_MIN_DRAG_MM = 0.25

#: The angle ring is drawn this many lumen radii out, so it clears the wall.
_RING_OVER_LUMEN = 2.4
#: How close to the ring a press must be, as a fraction of its radius.
_RING_GRAB_FRACTION = 0.22
#: Cross-section zoom bounds and the factor one Ctrl+wheel notch applies.
_MIN_XS_ZOOM = 1.0
_MAX_XS_ZOOM = 10.0
_XS_ZOOM_STEP = 1.25
#: Floor on the half-width shown, so a deep zoom cannot collapse the axes.
_MIN_XS_HALF_MM = 0.25

#: How far along the vessel a correction reaches, as a standard deviation in
#: millimetres. A centerline is smooth, so pulling one station and leaving its
#: neighbours put would produce a kink the reformation would then cut across.
_EDIT_REACH_MM = 4.0

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
        self._on_projection_edit: Any = None
        self._syncing = False
        #: ``(station, offset_mm)`` where a centerline drag began.
        self._drag: tuple[int, float] | None = None
        #: Layer whose window the intensities are drawn with, and its subscription.
        self._contrast_layer: Any = None
        self._contrast_sub: tuple[Any, Any] | None = None
        #: Last cross-section drawn, so it can be repainted without re-sampling.
        self._xs_last: tuple[Any, Any, str, float, float] | None = None
        #: Half-width of the cross-section view in mm; 0 means "fit the data".
        self._xs_zoom = 1.0
        #: True while the angle ring is being dragged.
        self._angle_drag = False
        self._vessel: VesselCpr | None = None
        self._edited = False
        self._failures: dict[int, str] = {}
        #: Retouching: what a stroke writes, where it is held ("cpr" / "xs"), and
        #: the dabs drawn on the canvas meanwhile.
        self._on_retouch: Any = None
        self._on_undo: Any = None
        self._stroke: dict[str, Any] | None = None
        self._dabs: list[Any] = []
        self._brush_outline: Any = None
        #: Frames of a 3D+t image: how many, and what to call when one is shown.
        self._on_frame: Any = None
        self._n_frames = 0

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
        #: Artists for the station marker, so stepping it does not repaint the
        #: whole flat image.
        self._station_line = None
        self._profile_station_line = None
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

        self._overlay_style = QComboBox()
        self._overlay_style.addItem("contour", "contour")
        self._overlay_style.addItem("filled", "filled")
        self._overlay_style.setToolTip(
            "“Filled” tints the lumen and wall. “Contour” outlines them instead, "
            "which leaves the image underneath readable — a calcified plaque or a "
            "thrombus is hidden by the tint that is meant to mark it."
        )
        self._overlay_style.currentIndexChanged.connect(lambda _i: self.redraw())

        self._edit_projection = QCheckBox("Edit on image")
        self._edit_projection.setToolTip(
            "Drag across the flat image to pull the centerline sideways there. The "
            "vertical axis is millimetres from the centerline, so the drag distance "
            "is the correction. Off, a click picks a station instead."
        )
        self._edit_projection.setStyleSheet(f"color: {STATION_COLOR};")

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

        # Retouching the lumen on the flat image and the cross-section.
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QToolButton] = {}
        for key, text, tip in (
            ("inspect", "Inspect", "Clicks pick a station (and Edit on image pulls the centerline)."),
            ("paint", "Paint", "Paint the vessel's label: drag on the flat image or the cross-section. Each "
                               "point is a point in the volume; a ball of the brush radius is painted around "
                               "it (other labels are kept). Right-drag erases."),
            ("erase", "Erase", "Erase the vessel's label the same way (other labels are left alone)."),
        ):
            button = QToolButton()
            button.setText(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            self._mode_group.addButton(button)
            self._mode_buttons[key] = button
        self._mode_buttons["inspect"].setChecked(True)
        self._mode_group.buttonToggled.connect(lambda _b, _on: self._on_mode_changed())
        self._brush_mm = QDoubleSpinBox()
        self._brush_mm.setRange(0.1, 30.0)
        self._brush_mm.setDecimals(2)
        self._brush_mm.setSingleStep(0.25)
        self._brush_mm.setValue(1.0)
        self._brush_mm.setSuffix(" mm")
        self._brush_mm.setToolTip("Brush radius, in millimetres (a ball in the volume).")
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setToolTip("Take back the last edit of the lumen layer (Ctrl+Z on the canvas).")
        self._btn_undo.clicked.connect(lambda: self._on_undo() if self._on_undo is not None else None)

        # Frames of a 3D+t intensity image.
        self._btn_play = QToolButton()
        self._btn_play.setText("▶")
        self._btn_play.setToolTip("Play / pause the image's frames.")
        self._btn_play.clicked.connect(self._toggle_play)
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setToolTip("Frame of the 3D+t image shown.")
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.5, 60.0)
        self._fps.setDecimals(1)
        self._fps.setValue(8.0)
        self._fps.setSuffix(" fps")
        self._fps.valueChanged.connect(lambda v: self._play_timer.setInterval(int(1000.0 / max(float(v), 0.1))))
        self._frame_label = QLabel("")
        self._frame_label.setStyleSheet(f"color: {COLOR_MUTED};")
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000.0 / 8.0))
        self._play_timer.timeout.connect(self._play_tick)

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
        toggles.addWidget(self._overlay_style)
        toggles.addSpacing(SPACE_TIGHT)
        toggles.addWidget(self._edit_projection)
        toggles.addStretch(1)
        toggles.addWidget(self._btn_reset)
        toggles.addWidget(self._btn_render)
        root.addLayout(toggles)

        retouch = QHBoxLayout()
        retouch.setSpacing(SPACE_TIGHT)
        retouch.addWidget(self._caption("Mouse"))
        for key in ("inspect", "paint", "erase"):
            retouch.addWidget(self._mode_buttons[key])
        retouch.addSpacing(SPACE_TIGHT)
        retouch.addWidget(self._caption("Brush"))
        retouch.addWidget(self._brush_mm)
        retouch.addWidget(self._btn_undo)
        self._retouch_hint = self._caption("")
        retouch.addWidget(self._retouch_hint, stretch=1)
        root.addLayout(retouch)

        self._time_row = QWidget()
        time_row = QHBoxLayout(self._time_row)
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(SPACE_TIGHT)
        time_row.addWidget(self._caption("Frame"))
        time_row.addWidget(self._btn_play)
        time_row.addWidget(self._frame_slider, stretch=1)
        time_row.addWidget(self._frame_label)
        time_row.addWidget(self._fps)
        root.addWidget(self._time_row)
        self._time_row.hide()

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
            self._cpr_canvas.mpl_connect("button_press_event", self._on_canvas_press)
            self._cpr_canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self._cpr_canvas.mpl_connect("button_release_event", self._on_canvas_release)

            self._xs_fig = Figure(figsize=(2.6, 2.6), dpi=96)
            self._xs_ax = self._xs_fig.add_subplot(111)
            self._xs_canvas = FigureCanvasQTAgg(self._xs_fig)
            self._xs_canvas.setMinimumWidth(180)
            self._xs_canvas.setMinimumHeight(180)
            self._xs_canvas.mpl_connect("button_press_event", self._on_xs_press)
            self._xs_canvas.mpl_connect("motion_notify_event", self._on_xs_motion)
            self._xs_canvas.mpl_connect("button_release_event", self._on_xs_release)
            self._xs_canvas.mpl_connect("scroll_event", self._on_xs_scroll)

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

    def set_contrast_source(self, layer: Any) -> None:
        """Draw with *layer*'s own window, and follow it when it moves."""
        previous = getattr(self, "_contrast_sub", None)
        if previous is not None:
            old_layer, callback = previous
            try:
                old_layer.events.contrast_limits.disconnect(callback)
            except Exception:
                pass
        self._contrast_sub = None
        self._contrast_layer = layer
        if layer is None:
            return

        def _changed(_event: Any = None) -> None:
            """Repaint both canvases at the layer's new window."""
            self.redraw()
            self._redraw_cross_section()

        try:
            layer.events.contrast_limits.connect(_changed)
            self._contrast_sub = (layer, _changed)
        except Exception:
            pass

    def contrast_limits(self) -> tuple[float, float] | None:
        """The window to draw intensities with, or ``None`` to stretch per image."""
        layer = getattr(self, "_contrast_layer", None)
        if layer is None:
            return None
        try:
            lo, hi = (float(v) for v in layer.contrast_limits)
        except (TypeError, ValueError):
            return None
        return (lo, hi) if hi > lo else None

    def set_wall_supplied(self, supplied: bool) -> None:
        """Show the wall band by default only when a real wall mask was given."""
        self._show_wall.blockSignals(True)
        self._show_wall.setChecked(bool(supplied))
        self._show_wall.blockSignals(False)
        self._show_wall.setToolTip(
            "Wall segmentation." if supplied
            else "No wall mask was supplied; this band is dilated from the lumen, "
                 "so it says where a wall of that thickness would be, not where "
                 "one is."
        )

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
        on_projection_edit: Any = None,
        on_retouch: Any = None,
        on_undo: Any = None,
        on_frame: Any = None,
    ) -> None:
        """Register what to call when the controls move.

        *on_retouch* is called as ``(kind, point_mm, erase, radius_mm)`` — kind
        ``"press"`` / ``"move"`` / ``"release"``, *point_mm* the stroke's point in
        the volume (millimetres on the reformation's grid, ``None`` on release);
        *on_frame* with a frame index of a 3D+t image.
        """
        self._on_change = on_change
        self._on_station = on_station
        self._on_reset = on_reset
        self._on_stations = on_stations
        self._on_projection_edit = on_projection_edit
        self._on_retouch = on_retouch
        self._on_undo = on_undo
        self._on_frame = on_frame

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

    def _on_canvas_press(self, event: Any) -> None:
        """Start a centerline drag, or pick a station when not editing."""
        if self._start_stroke(event, "cpr"):
            return
        station = self._station_at(event)
        if station is None:
            return
        if self._edit_projection.isChecked() and event.inaxes is self._cpr_ax:
            # The vertical axis is millimetres from the centerline, so where the
            # press lands *is* the starting offset — no separate reference needed.
            self._drag = (station, float(event.ydata))
            return
        self._station.setValue(station)

    def _on_canvas_release(self, event: Any) -> None:
        """Finish a centerline drag and hand the correction to the installer."""
        if self._end_stroke():
            return
        drag = self._drag
        self._drag = None
        if drag is None or event.inaxes is not self._cpr_ax or event.ydata is None:
            return
        station, start = drag
        offset = float(event.ydata) - float(start)
        if abs(offset) < _MIN_DRAG_MM:
            # A click that happened to be inside the image, not a correction.
            self._station.setValue(station)
            return
        if self._on_projection_edit is not None:
            self._on_projection_edit(int(station), offset)

    def _on_canvas_motion(self, event: Any) -> None:
        """Read out the calibre under the cursor without committing to a station."""
        if self._move_stroke(event, "cpr"):
            return
        station = self._station_at(event)
        if station is None or self._vessel is None:
            return
        if self._drag is not None and event.ydata is not None:
            self._station_label.setText(
                f"pull {float(event.ydata) - self._drag[1]:+.2f} mm at station {self._drag[0]}"
            )
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

    def move_station_marker(self, station: int) -> None:
        """Move the station line on the flat image without redrawing it.

        Stepping the station changes one vertical line on a picture whose pixels,
        contours and calibre trace are all unchanged. Repainting the lot for that
        is what made dragging the station slider crawl on a large volume.
        """
        if self._vessel is None or self._cpr_canvas is None:
            return
        with using("cpu"):
            arc = to_numpy(self._vessel.cpr.arc_length_mm)
            if not arc.size:
                return
            at = float(arc[int(np.clip(station, 0, arc.size - 1))])
        moved = False
        for line in (getattr(self, "_station_line", None),
                     getattr(self, "_profile_station_line", None)):
            if line is not None:
                try:
                    line.set_xdata([at, at])
                    moved = True
                except Exception:  # noqa: BLE001 — a stale artist just forces a redraw
                    moved = False
                    break
        if not moved:
            self.redraw()
            return
        self._cpr_canvas.draw_idle()

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
        self._xs_last = None
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
        # The clear took the stroke marks and the brush outline with it.
        self._dabs = [d for d in self._dabs if d.axes is not None and d.axes is not ax]
        if self._brush_outline is not None and self._brush_outline.axes in (ax, None):
            self._brush_outline = None

        # Matplotlib needs host arrays; this is the display boundary. The
        # reformation is stored station-major, so it is transposed to put arc
        # length on x and the ray across the vessel on y.
        with using("cpu"):
            image = to_numpy(vessel.cpr.image).T
            arc = to_numpy(vessel.cpr.arc_length_mm)
            ray = to_numpy(vessel.cpr.ray_mm)
            limits = self.contrast_limits()
            if limits is not None:
                vmin, vmax = limits
            else:
                finite = image[np.isfinite(image)]
                # No layer to follow: percentile limits, because a single bright
                # voxel would otherwise flatten the whole vessel to mid-grey.
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

        contour = str(self._overlay_style.currentData() or "filled") == "contour"
        if self._show_wall.isChecked() and vessel.wall is not None:
            self._draw_mask(ax, to_numpy(vessel.wall).T, extent, WALL_RGB, arc, ray, contour)
        if self._show_lumen.isChecked():
            self._draw_mask(ax, to_numpy(vessel.lumen).T, extent, LUMEN_RGB, arc, ray, contour)
        if self._show_center.isChecked():
            ax.axhline(0.0, color=CENTERLINE_COLOR, lw=0.8, alpha=0.9)

        station = self.station()
        if 0 <= station < arc.size:
            self._station_line = ax.axvline(
                float(arc[station]), color=STATION_COLOR, lw=1.2
            )

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
            self._profile_station_line = ax.axvline(
                float(arc[station]), color=STATION_COLOR, lw=1.2
            )
        ax.set_xlabel("along the vessel (mm)", fontsize=8)
        ax.set_ylabel("Ø mm", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)

    @staticmethod
    def _draw_mask(
        ax: Any,
        mask: Any,
        extent: list[float],
        rgb: tuple,
        arc: Any,
        ray: Any,
        contour: bool,
    ) -> None:
        """Mark a mask on the flat image, filled or as an outline."""
        if not contour:
            VesselCprPanel._overlay(ax, mask, extent, rgb, 0.35)
            return
        with using("cpu"):
            arr = to_numpy(mask)
            if not arr.any():
                return
            # Contour wants the axes' own coordinates, not the pixel grid, so the
            # outline lands on the same millimetres the image is drawn in.
            x = to_numpy(arc)[: arr.shape[1]]
            y = to_numpy(ray)[: arr.shape[0]]
        if x.size < 2 or y.size < 2:
            return
        colour = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)
        ax.contour(
            x, y, (arr > 0).astype(float)[: y.size, : x.size],
            levels=[0.5], colors=[colour], linewidths=1.0,
        )

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

    def show_cross_section(
        self,
        image: Any,
        mask: Any = None,
        *,
        title: str = "",
        ray_mm: float = DEFAULT_RAY_MM,
        lumen_mm: float = 0.0,
    ) -> None:
        """Draw the perpendicular cross-section, with the angle ring over it."""
        self._station_label.setText(title or "—")
        self._xs_last = (image, mask, title, float(ray_mm), float(lumen_mm))
        self._redraw_cross_section()

    def _redraw_cross_section(self) -> None:
        """Repaint the stored cross-section, at the current window and zoom."""
        if self._xs_ax is None or self._xs_canvas is None or self._xs_last is None:
            return
        image, mask, title, ray, lumen = self._xs_last
        ax = self._xs_ax
        ax.clear()
        ax.set_axis_off()
        self._dabs = [d for d in self._dabs if d.axes is not None and d.axes is not ax]
        if self._brush_outline is not None and self._brush_outline.axes in (ax, None):
            self._brush_outline = None
        with using("cpu"):
            arr = to_numpy(image)
            overlay = None
            if mask is not None:
                m = to_numpy(mask)
                if m.any():
                    overlay = np.zeros((*m.shape, 4), dtype=float)
                    overlay[..., 0] = LUMEN_RGB[0]
                    overlay[..., 3] = (m > 0).astype(float) * 0.35
        # Drawn in millimetres, which is what lets the ring and the lumen be
        # compared by eye and what the drag reads its angle out of.
        extent = [-ray, ray, -ray, ray]
        if arr.size:
            limits = self.contrast_limits()
            ax.imshow(
                arr, cmap="gray", origin="lower", extent=extent,
                interpolation="nearest",
                vmin=None if limits is None else limits[0],
                vmax=None if limits is None else limits[1],
            )
            if overlay is not None:
                ax.imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
        self._draw_angle_ring(ax, ray, lumen)
        half = max(ray / max(self._xs_zoom, 1e-6), _MIN_XS_HALF_MM)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        if title:
            ax.set_title(title, fontsize=8)
        style_image_figure(self._xs_fig)
        self._xs_fig.tight_layout(pad=0.2)
        self._xs_canvas.draw_idle()

    def _ring_radius(self, ray_mm: float, lumen_mm: float) -> float:
        """Radius of the angle ring: clear of the wall, inside the field of view.

        *lumen_mm* is a diameter, so it is halved first — scaling the diameter put
        the ring at nearly five lumen radii, far enough out that the angle could
        not be judged against the vessel it belongs to.
        """
        wanted = max(float(lumen_mm), 0.0) / 2.0 * _RING_OVER_LUMEN
        return min(max(wanted, 0.2 * float(ray_mm)), 0.88 * float(ray_mm))

    def _draw_angle_ring(self, ax: Any, ray_mm: float, lumen_mm: float) -> None:
        """Mark the centerline, and show the cut angle as a draggable ring."""
        from matplotlib.patches import Circle

        radius = self._ring_radius(ray_mm, lumen_mm)
        theta = math.radians(self.angle_deg())
        ax.add_patch(
            Circle(
                (0.0, 0.0), radius, fill=False,
                edgecolor=STATION_COLOR, lw=1.0, alpha=0.75, linestyle=(0, (4, 3)),
            )
        )
        # The reformation cuts along this direction, so the two arms are where the
        # flat image's left and right edges come from.
        dx, dy = math.cos(theta), math.sin(theta)
        for sign in (1.0, -1.0):
            ax.plot(
                [0.0, sign * dx * radius], [0.0, sign * dy * radius],
                color=STATION_COLOR, lw=1.2,
            )
        ax.plot([sign * dx * radius for sign in (1.0, -1.0)],
                [sign * dy * radius for sign in (1.0, -1.0)],
                linestyle="none", marker="o", markersize=4,
                color=STATION_COLOR, markeredgecolor="none")
        # The centerline itself: the one point the whole reformation hangs off.
        ax.plot([0.0], [0.0], marker="+", markersize=7, color=CENTERLINE_COLOR, mew=1.2)

    # ── cross-section interaction ────────────────────────────────────────────

    def _xs_radius_at(self, event: Any) -> float | None:
        """Distance of a cross-section event from the centerline, in millimetres."""
        if event.inaxes is not self._xs_ax or event.xdata is None or event.ydata is None:
            return None
        return math.hypot(float(event.xdata), float(event.ydata))

    def _on_xs_press(self, event: Any) -> None:
        """Grab the angle ring when the press lands near it."""
        if self._xs_last is None:
            return
        if self._start_stroke(event, "xs"):
            return
        radius = self._xs_radius_at(event)
        if radius is None:
            return
        ring = self._ring_radius(self._xs_last[3], self._xs_last[4])
        if abs(radius - ring) <= max(ring * _RING_GRAB_FRACTION, _MIN_XS_HALF_MM):
            self._angle_drag = True
            self._set_angle_from(event)

    def _on_xs_motion(self, event: Any) -> None:
        """Turn the cutting plane while the ring is held."""
        if self._move_stroke(event, "xs"):
            return
        if self._angle_drag:
            self._set_angle_from(event)

    def _on_xs_release(self, _event: Any = None) -> None:
        """Commit the new angle once the ring is let go."""
        if self._end_stroke():
            return
        if not self._angle_drag:
            return
        self._angle_drag = False
        self._emit_change()

    def _set_angle_from(self, event: Any) -> None:
        """Point the cut at the cursor, and redraw the ring without resampling."""
        if event.inaxes is not self._xs_ax or event.xdata is None or event.ydata is None:
            return
        # The cut is a line, not an arrow: 0 and 180 degrees are the same plane.
        degrees = math.degrees(math.atan2(float(event.ydata), float(event.xdata))) % 180.0
        self._sync(self._angle, int(round(degrees)))
        self._sync(self._angle_spin, int(round(degrees)))
        self._redraw_cross_section()

    def _on_xs_scroll(self, event: Any) -> None:
        """Ctrl+wheel zooms the cross-section about the centerline."""
        if event.inaxes is not self._xs_ax:
            return
        modifiers = getattr(event, "modifiers", None) or ()
        if "ctrl" not in modifiers and "control" not in modifiers:
            return
        step = _XS_ZOOM_STEP if getattr(event, "button", "") == "up" else 1.0 / _XS_ZOOM_STEP
        self._xs_zoom = min(max(self._xs_zoom * step, _MIN_XS_ZOOM), _MAX_XS_ZOOM)
        self._redraw_cross_section()

    def xs_zoom(self) -> float:
        """Current cross-section magnification."""
        return float(self._xs_zoom)

    # ── retouching the lumen ─────────────────────────────────────────────────

    def retouch_mode(self) -> str:
        """``"inspect"``, ``"paint"`` or ``"erase"``."""
        for key, button in self._mode_buttons.items():
            if button.isChecked():
                return key
        return "inspect"

    def set_retouch_mode(self, mode: str) -> None:
        if mode in self._mode_buttons:
            self._mode_buttons[mode].setChecked(True)

    def brush_mm(self) -> float:
        return float(self._brush_mm.value())

    def _on_mode_changed(self) -> None:
        mode = self.retouch_mode()
        drawing = mode != "inspect"
        for canvas in (self._cpr_canvas, self._xs_canvas):
            if canvas is not None:
                canvas.setCursor(Qt.CrossCursor) if drawing else canvas.unsetCursor()
        self._retouch_hint.setText(
            "Drag on the flat image or the cross-section · right-drag erases" if mode == "paint"
            else "Drag to erase the vessel's label" if mode == "erase" else "")
        if not drawing:
            self._hide_brush_outline()

    def _point_mm(self, event: Any, where: str) -> Any | None:
        """The point in the volume (mm, on the reformation's grid) under *event*."""
        vessel = self._vessel
        if vessel is None or event.xdata is None or event.ydata is None:
            return None
        with using("cpu"):
            samples = vessel.samples
            pts = to_numpy(samples.points_mm).astype(float)
            if where == "cpr":
                if event.inaxes is not self._cpr_ax:
                    return None
                arc = to_numpy(vessel.cpr.arc_length_mm).astype(float)
                direction = to_numpy(cut_direction(samples, vessel.cpr.angle_deg)).astype(float)
                x = float(np.clip(float(event.xdata), arc[0], arc[-1]))
                # Between two stations: the centre and the cut direction interpolated.
                i = int(np.clip(np.searchsorted(arc, x) - 1, 0, max(arc.size - 2, 0)))
                j = min(i + 1, arc.size - 1)
                f = 0.0 if j == i or arc[j] == arc[i] else (x - arc[i]) / (arc[j] - arc[i])
                centre = pts[i] * (1.0 - f) + pts[j] * f
                cut = direction[i] * (1.0 - f) + direction[j] * f
                cut /= max(float(np.linalg.norm(cut)), 1e-9)
                return centre + float(event.ydata) * cut
            if event.inaxes is not self._xs_ax:
                return None
            station = int(np.clip(self.station(), 0, pts.shape[0] - 1))
            u = to_numpy(samples.u).astype(float)[station]
            v = to_numpy(samples.v).astype(float)[station]
            return pts[station] + float(event.xdata) * u + float(event.ydata) * v

    def _start_stroke(self, event: Any, where: str) -> bool:
        """A press in Paint / Erase: start a stroke (True), else leave it (False)."""
        mode = self.retouch_mode()
        if mode == "inspect" or self._on_retouch is None:
            return False
        button = getattr(event, "button", 1)
        if int(getattr(button, "value", button) or 1) not in (1, 3):
            return False
        point = self._point_mm(event, where)
        if point is None:
            return False
        erase = mode == "erase" or int(getattr(button, "value", button) or 1) == 3
        self._stroke = {"where": where, "erase": erase}
        self._on_retouch("press", point, erase, self.brush_mm())
        self._dab(event, where, erase)
        return True

    def _move_stroke(self, event: Any, where: str) -> bool:
        if self._stroke is None:
            if self.retouch_mode() != "inspect":
                self._move_brush_outline(event, where)
            return False
        if self._stroke["where"] == where:
            point = self._point_mm(event, where)
            if point is not None and self._on_retouch is not None:
                self._on_retouch("move", point, self._stroke["erase"], self.brush_mm())
                self._dab(event, where, self._stroke["erase"])
        return True

    def _end_stroke(self) -> bool:
        if self._stroke is None:
            return False
        stroke, self._stroke = self._stroke, None
        if self._on_retouch is not None:
            self._on_retouch("release", None, stroke["erase"], self.brush_mm())
        return True

    def _dab(self, event: Any, where: str, erase: bool) -> None:
        """Mark where the brush went, until the re-render shows the new mask."""
        from matplotlib.patches import Ellipse

        ax = self._cpr_ax if where == "cpr" else self._xs_ax
        canvas = self._cpr_canvas if where == "cpr" else self._xs_canvas
        if ax is None or canvas is None:
            return
        r = self.brush_mm()
        colour = "#5bc0ff" if erase else LUMEN_HEX
        patch = Ellipse((float(event.xdata), float(event.ydata)), 2 * r, 2 * r, facecolor=colour,
                        edgecolor="none", alpha=0.35)
        ax.add_patch(patch)
        self._dabs.append(patch)
        canvas.draw_idle()

    def clear_dabs(self) -> None:
        for patch in self._dabs:
            try:
                patch.remove()
            except Exception:  # noqa: BLE001 — gone with an axes clear
                pass
        self._dabs = []

    def _move_brush_outline(self, event: Any, where: str) -> None:
        from matplotlib.patches import Ellipse

        ax = self._cpr_ax if where == "cpr" else self._xs_ax
        canvas = self._cpr_canvas if where == "cpr" else self._xs_canvas
        if ax is None or canvas is None or event.inaxes is not ax or event.xdata is None:
            self._hide_brush_outline()
            return
        r = self.brush_mm()
        outline = self._brush_outline
        if outline is None or outline.axes is not ax:
            self._hide_brush_outline()
            outline = Ellipse((0, 0), 2 * r, 2 * r, fill=False, edgecolor="white", lw=0.8, alpha=0.8)
            ax.add_patch(outline)
            self._brush_outline = outline
        outline.set_center((float(event.xdata), float(event.ydata)))
        outline.set_width(2 * r)
        outline.set_height(2 * r)
        canvas.draw_idle()

    def _hide_brush_outline(self) -> None:
        outline, self._brush_outline = self._brush_outline, None
        if outline is not None:
            try:
                canvas = outline.figure.canvas if outline.figure is not None else None
                outline.remove()
                if canvas is not None:
                    canvas.draw_idle()
            except Exception:  # noqa: BLE001
                pass

    # ── frames of a 3D+t image ───────────────────────────────────────────────

    def set_frames(self, count: int, frame: int = 0) -> None:
        """Show the frame controls for *count* frames (0 hides them)."""
        self._n_frames = int(count)
        self._time_row.setVisible(self._n_frames > 1)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, max(self._n_frames - 1, 0))
        self._frame_slider.setValue(min(max(int(frame), 0), max(self._n_frames - 1, 0)))
        self._frame_slider.blockSignals(False)
        self._update_frame_label()
        if self._n_frames <= 1:
            self._stop_play()

    def frame(self) -> int:
        return int(self._frame_slider.value())

    def set_frame(self, frame: int, *, emit: bool = False) -> None:
        """Move the frame slider (without announcing it unless *emit*)."""
        if not self._n_frames:
            return
        self._frame_slider.blockSignals(not emit)
        self._frame_slider.setValue(int(frame) % max(self._n_frames, 1))
        self._frame_slider.blockSignals(False)
        self._update_frame_label()

    def _update_frame_label(self) -> None:
        self._frame_label.setText(f"{self.frame() + 1} / {self._n_frames}" if self._n_frames else "")

    def _on_frame_slider(self, value: int) -> None:
        self._update_frame_label()
        if self._on_frame is not None:
            self._on_frame(int(value))

    def playing(self) -> bool:
        return self._play_timer.isActive()

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._stop_play()
        elif self._n_frames > 1:
            self._play_timer.setInterval(int(1000.0 / max(float(self._fps.value()), 0.1)))
            self._play_timer.start()
            self._btn_play.setText("⏸")

    def _stop_play(self) -> None:
        self._play_timer.stop()
        self._btn_play.setText("▶")

    def _play_tick(self) -> None:
        if self._n_frames > 1:
            self.set_frame((self.frame() + 1) % self._n_frames, emit=True)

    def show_frame_image(self, vessel: VesselCpr) -> None:
        """Repaint the flat vessel for another frame (same geometry, new intensities)."""
        self._vessel = vessel
        self.clear_dabs()
        self.redraw()


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


# ── frames of a 3D+t intensity image ─────────────────────────────────────────

#: Frame images kept (flat images and cross-sections, all frames of the current view).
_FRAME_CACHE_SIZE = 256


def _setup_frames(state: dict[str, Any], image_layer: Any, lumen_layer: Any) -> None:
    """For a time-first 3D+t *image_layer*: how many frames, which to start on,
    and the viewer's time dim it follows."""
    state["n_frames"] = 0
    state["frame_cache"] = {}
    if image_layer is None or lumen_layer is None:
        return
    data = getattr(image_layer, "data", None)
    if data is None or getattr(data, "ndim", 0) != 4:
        return
    from nvitk.gui.core.spatial import _time_axis_index

    t_ax = _time_axis_index(image_layer)
    if t_ax != 0:
        return  # nvitk opens 3D+t time-first; another layout is shown as one frame
    viewer = state["viewer"]
    state["image_layer"] = image_layer
    state["n_frames"] = int(data.shape[0])
    dim = int(viewer.dims.ndim) - int(image_layer.ndim)
    state["viewer_time_dim"] = dim if dim >= 0 else None
    try:
        state["frame0"] = int(viewer.dims.current_step[dim]) if dim >= 0 else 0
    except Exception:  # noqa: BLE001
        state["frame0"] = 0


def _image_coords(state: dict[str, Any], grid_coords: Any) -> Any:
    """``(3, ...)`` voxel coordinates on the reformation's grid as the image layer's
    own spatial voxel coordinates (through the world: any grid, any placement)."""
    lumen, image = state["lumen_layer"], state["image_layer"]
    # Napari's transforms, on host coordinates.
    with using("cpu"):
        pts = to_numpy(grid_coords).astype(float)
        shape = pts.shape[1:]
        flat = pts.reshape(3, -1).T * to_numpy(state["overlay_scale"]).astype(float)[None, :]
        world = to_numpy(lumen._data_to_world(flat)).astype(float)
        full = np.zeros((flat.shape[0], int(image.ndim)), dtype=float)
        full[:, -3:] = world[:, -3:]
        data = to_numpy(image._data_to_world.inverse(full)).astype(float)
        return data[:, 1:].T.reshape((3, *shape))


def _sample_frame(state: dict[str, Any], coords: Any, frame: int) -> Any:
    """The image's *frame* sampled at its voxel *coords* (linear, 0 outside)."""
    from scipy.ndimage import map_coordinates

    image = state["image_layer"]
    # The frames are Napari's (host) arrays and the samples a few thousand: sampled
    # on the host — sending a whole frame to the GPU would cost more than this.
    with using("cpu"):
        volume = to_numpy(image.data[int(frame)])
        values = map_coordinates(volume, to_numpy(coords), order=1, mode="constant", cval=0.0)
    return as_backend_array(values.astype("float32", copy=False))


def _cache_put(state: dict[str, Any], key: tuple, value: Any) -> Any:
    cache = state["frame_cache"]
    if len(cache) >= _FRAME_CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _with_frame_image(state: dict[str, Any], vessel: VesselCpr, frame: int) -> VesselCpr:
    """*vessel* with its flat image taken from *frame* of the 3D+t image."""
    from nvitk.transform.cpr import cpr_coords

    cpr = vessel.cpr
    n_ray = int(to_numpy(cpr.ray_mm).size)
    ray = float(abs(to_numpy(cpr.ray_mm)[-1])) if n_ray else 0.0
    key = ("cpr", id(vessel.samples), float(cpr.angle_deg), ray, n_ray, int(frame))
    image = state["frame_cache"].get(key)
    if image is None:
        coords_key = ("cpr-coords", id(vessel.samples), float(cpr.angle_deg), ray, n_ray)
        coords = state["frame_cache"].get(coords_key)
        if coords is None:
            grid = cpr_coords(vessel.samples, angle_deg=cpr.angle_deg, ray_mm=ray, n_ray=n_ray)
            coords = _cache_put(state, coords_key, _image_coords(state, grid))
        image = _cache_put(state, key, _sample_frame(state, coords, frame))
    return dataclasses.replace(vessel, cpr=dataclasses.replace(cpr, image=image))


def _frame_cross_section(state: dict[str, Any], vessel: VesselCpr, station: int, ray_mm: float,
                         frame: int, res: int = 129) -> Any:
    """The perpendicular cross-section at *station*, from *frame* of the 3D+t image."""
    from nvitk.transform.oblique import oblique_plane_coords

    key = ("xs", id(vessel.samples), int(station), float(ray_mm), int(res), int(frame))
    image = state["frame_cache"].get(key)
    if image is not None:
        return image
    samples = vessel.samples
    idx = int(max(0, min(int(station), samples.n_stations - 1)))
    scale = as_backend_array(samples.spacing).astype(float, copy=False)
    plane = oblique_plane_coords(
        as_backend_array(samples.points_vox).astype(float, copy=False)[idx],
        as_backend_array(samples.u).astype(float, copy=False)[idx] / scale,
        as_backend_array(samples.v).astype(float, copy=False)[idx] / scale,
        radius_vox=float(ray_mm),
        res=int(res),
    )
    coords = _image_coords(state, plane.coords)
    return _cache_put(state, key, _sample_frame(state, coords, frame))


def install_vessel_cpr(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    lumen_layer: Any,
    lumen_mask: Any,
    image: Any = None,
    image_layer: Any = None,
    wall_mask: Any = None,
    labels=None,
    centerline_mask: Any = None,
    spacing=None,
    step_mm: float = 0.5,
    join_gap_vox: int = DEFAULT_JOIN_GAP_VOX,
    schema_key: str | None = None,
    skeleton_method: str = "vmtk",
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
    # Follow the image layer's window, so the flat vessel and the cross-section
    # show the same greys the canvas does instead of a percentile stretch that
    # disagrees with it the moment the brightness is touched.
    panel.set_contrast_source(image_layer)
    # A wall the caller did not supply is derived from the lumen — a stand-in for
    # a segmentation, not a measurement of one — so it starts hidden.
    panel.set_wall_supplied(wall_mask is not None)
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
        "skeleton_method": str(skeleton_method),
        "schema_key": schema_key or None,
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

    _setup_frames(state, image_layer, lumen_layer)

    # Parented so the pending re-render cannot fire after the panel is gone.
    timer = QTimer(panel)
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
            skeleton_method=state.get("skeleton_method", "vmtk"),
        )
        panel.set_failure_reasons(reasons)
        panel.set_vessels(
            [(lab, vessel_name(lab, schema_key=state["schema_key"])) for lab in state["samples"]]
        )

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
        # The calibre profile depends on the centerline and the ray, not on the
        # cutting angle — so turning the angle reuses the one already measured
        # rather than resampling the whole vessel's cross-sections again.
        ray = panel.ray_mm()
        # Hold the samples object itself, not its id: an id alone would let a
        # freed centerline's address be reused by a new one and hand back the
        # wrong calibre.
        previous = state.get("diameter_cache")
        key = (int(label), samples, float(ray))
        vessel = build_vessel_cpr(
            state["image"],
            state["mask"],
            label=int(label),
            samples=samples,
            wall_mask=state["wall"],
            angle_deg=panel.angle_deg(),
            ray_mm=ray,
            n_ray=DEFAULT_N_RAY,
            schema_key=state["schema_key"],
            diameter=(
                previous[1]
                if previous is not None
                and previous[0][0] == key[0]
                and previous[0][1] is key[1]
                and previous[0][2] == key[2]
                else None
            ),
        )
        state["diameter_cache"] = (key, vessel.diameter)
        if state.get("n_frames", 0) > 1:
            # A 3D+t image: the frame on show, sampled straight from the layer.
            vessel = _with_frame_image(state, vessel, panel.frame())
        state["vessel"] = vessel
        panel.clear_dabs()
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
        if state.get("n_frames", 0) > 1:
            image_xs = _frame_cross_section(state, vessel, station, panel.ray_mm(), panel.frame())
        arc = to_numpy(vessel.cpr.arc_length_mm)
        at = float(arc[station]) if 0 <= station < arc.size else 0.0
        width = to_numpy(vessel.diameter_mm())
        here = float(width[station]) if station < width.size else float("nan")
        panel.show_cross_section(
            image_xs,
            mask_xs,
            title=f"{at:.1f} mm along — lumen {here:.1f} mm",
            ray_mm=panel.ray_mm(),
            lumen_mm=0.0 if here != here else here,
        )
        _keep_selection(lambda: _update_station_overlays(vessel, station))
        panel.move_station_marker(station)

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
        """Pick up an edit of the lumen layer (a stroke here, Napari's brush, an
        undo) and re-render.

        The centerlines are kept — derived or retouched — so the flat image does
        not jump under a retouch; a vessel that had none is derived now, and
        *Reset centerline* derives one again from the new mask. The calibre is
        measured again.
        """
        if state.get("stroke") is not None:
            return  # mid-stroke: the release re-renders once
        layer = state.get("lumen_layer")
        if layer is None:
            return
        fresh = to_numpy(layer.data)
        if tuple(fresh.shape) != tuple(state["mask"].shape):
            from nvitk.viz.vessel_cpr import upsample_volume

            fresh = to_numpy(upsample_volume(fresh, state["mask"].shape, order=0))
        state["mask"] = fresh
        state["diameter_cache"] = None
        missing = [lab for lab in (state["labels"] or []) if int(lab) not in state["samples"]]
        if missing or not state["samples"]:
            kept = dict(state["samples"])
            _derive_centerlines()
            state["samples"].update(kept)
        timer.start()

    # ── retouching the lumen from the panel ───────────────────────────────────

    def _retouch(kind: str, point_mm: Any, erase: bool, radius_mm: float) -> None:
        layer = state.get("lumen_layer")
        label = panel.selected_label()
        if layer is None or label is None:
            return
        if kind == "press":
            _end_stroke(commit=True)
            labels_layer = type(layer).__name__ == "Labels"
            previous = getattr(layer, "_block_history", None) if labels_layer else None
            if previous is not None:
                layer._block_history = True
            state["stroke"] = {"last": None, "erase": bool(erase), "label": int(label),
                               "labels_layer": labels_layer, "previous": previous, "seen": set()}
        stroke = state.get("stroke")
        if stroke is None:
            return
        if point_mm is not None:
            _stroke_to(stroke, point_mm, float(radius_mm))
        if kind == "release":
            _end_stroke(commit=True)

    def _stroke_to(stroke: dict[str, Any], point_mm: Any, radius_mm: float) -> None:
        """Dabs from the last point to *point_mm*, at most half a brush apart."""
        with using("cpu"):
            here = to_numpy(point_mm).astype(float)
            last = stroke["last"]
            if last is None:
                points = [here]
            else:
                n = max(int(math.ceil(float(np.linalg.norm(here - last)) / max(radius_mm * 0.5, 1e-3))), 1)
                points = [last + (here - last) * (k / n) for k in range(1, n + 1)]
        for point in points:
            _dab(point, radius_mm, stroke)
        stroke["last"] = here

    def _dab(point_mm: Any, radius_mm: float, stroke: dict[str, Any]) -> int:
        """Paint (erase) a ball of *radius_mm* around *point_mm* in the lumen layer."""
        from nvitk.gui.core.spatial import layer_spacing

        layer = state["lumen_layer"]
        data = layer.data
        # Host index arithmetic: the layer's voxels are Napari's (NumPy).
        with using("cpu"):
            grid_sp = to_numpy(state["spacing"]).astype(float)
            centre = to_numpy(point_mm).astype(float) / grid_sp * to_numpy(state["overlay_scale"]).astype(float)
            sp = to_numpy((layer_spacing(layer) or (1.0, 1.0, 1.0))[:3]).astype(float)
            reach = np.ceil(float(radius_mm) / sp).astype(int)
            axes = [np.arange(int(np.floor(c)) - r, int(np.ceil(c)) + r + 1) for c, r in zip(centre, reach)]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            dist2 = (((grid - centre[None, :]) * sp[None, :]) ** 2).sum(axis=1)
            grid = grid[dist2 <= float(radius_mm) ** 2]
            shape = to_numpy(list(data.shape[-3:]))
            grid = grid[np.all((grid >= 0) & (grid < shape), axis=1)]
            if not grid.size:
                return 0
            index = tuple(grid[:, d] for d in range(3))
            current = to_numpy(data[index])
            if stroke["erase"]:
                keep = current == stroke["label"]
                value = 0
            else:
                # The other labels are kept: only background and this vessel change.
                keep = (current == 0) | (current == stroke["label"])
                value = stroke["label"]
            index = tuple(ix[keep] for ix in index)
            n = int(index[0].size)
            if n:
                stroke["seen"].update(int(i) for i in np.ravel_multi_index(index, tuple(int(v) for v in shape)))
        if n:
            if stroke["labels_layer"]:
                layer.data_setitem(index, value, refresh=False)
            else:
                data[index] = value
        return n

    def _end_stroke(*, commit: bool) -> None:
        stroke, state["stroke"] = state.get("stroke"), None
        if stroke is None:
            return
        layer = state["lumen_layer"]
        if stroke["labels_layer"]:
            if stroke["previous"] is not None:
                layer._commit_staged_history()  # one undo step for the stroke (fires paint)
                layer._block_history = stroke["previous"]
        try:
            layer.refresh()
        except Exception:  # noqa: BLE001
            pass
        if not stroke["labels_layer"] or stroke["previous"] is None:
            _on_mask_changed()
        what = "erased" if stroke["erase"] else "painted"
        panel._retouch_hint.setText(f"{len(stroke['seen']):,} voxel(s) {what} (Undo takes the stroke back).")

    def _undo() -> None:
        layer = state.get("lumen_layer")
        if layer is not None and type(layer).__name__ == "Labels":
            layer.undo()  # fires paint: the mask is read again and re-rendered

    # ── frames of a 3D+t image ───────────────────────────────────────────────

    def _show_frame(frame: int) -> None:
        vessel = state.get("vessel")
        if vessel is None or state.get("n_frames", 0) <= 1:
            return
        vessel = _with_frame_image(state, vessel, int(frame))
        state["vessel"] = vessel
        panel.show_frame_image(vessel)
        _show_station(panel.station())
        # Napari's time slider follows the panel (and the panel it, below).
        dim = state.get("viewer_time_dim")
        if dim is not None and not state.get("following_viewer"):
            try:
                state["setting_viewer"] = True
                viewer.dims.set_current_step(dim, int(frame))
            except Exception:  # noqa: BLE001
                pass
            finally:
                state["setting_viewer"] = False

    def _on_viewer_dims(_event: Any = None) -> None:
        dim = state.get("viewer_time_dim")
        if dim is None or state.get("setting_viewer") or panel.playing():
            return
        try:
            frame = int(viewer.dims.current_step[dim])
        except Exception:  # noqa: BLE001
            return
        if frame != panel.frame() and 0 <= frame < state.get("n_frames", 0):
            state["following_viewer"] = True
            try:
                panel.set_frame(frame, emit=True)
            finally:
                state["following_viewer"] = False

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

    def _edit_from_projection(station: int, offset_mm: float) -> None:
        """Pull the centerline *offset_mm* sideways at *station*.

        The flat image's vertical axis is millimetres along the cut direction, so a
        drag on it is already a displacement in the plane the reformation was cut
        in — no unprojection needed. The pull falls off along the vessel over
        :data:`_EDIT_REACH_MM`, because moving one station alone leaves a kink the
        next reformation would cut straight across.
        """
        label = panel.selected_label()
        if label is None:
            return
        samples = state["edited"].get(int(label)) or state["samples"].get(int(label))
        if samples is None:
            return
        try:
            direction = cut_direction(samples, panel.angle_deg())
            arc = as_backend_array(samples.arc_length_mm).astype(float, copy=False)
            here = float(to_numpy(arc)[int(station)])
            reach = max(float(_EDIT_REACH_MM), float(state["step_mm"]))
            weight = np.exp(-((arc - here) ** 2) / (2.0 * reach * reach))
            moved_mm = (
                as_backend_array(samples.points_mm).astype(float, copy=False)
                + float(offset_mm) * weight[:, None] * direction
            )
            scale = as_backend_array(samples.spacing).astype(float, copy=False)
            state["edited"][int(label)] = resample_centerline(
                moved_mm / scale,
                state["spacing"],
                step_mm=state["step_mm"],
                smooth=0.0,
            )
        except Exception:
            return
        # Handles belong to the curve the user just changed, not the old one.
        state["handles_stale"] = True
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
        on_projection_edit=_edit_from_projection,
        on_retouch=_retouch,
        on_undo=_undo,
        on_frame=_show_frame,
    )

    if lumen_layer is not None:
        try:
            # ``data`` when the array is replaced, ``paint`` for brush strokes and
            # undo (Napari 0.7 announces a stroke with ``paint`` alone).
            lumen_layer.events.data.connect(_on_mask_changed)
            lumen_layer.events.paint.connect(_on_mask_changed)
            state["mask_callback"] = _on_mask_changed
        except Exception:
            pass
    if state.get("n_frames", 0) > 1:
        panel.set_frames(state["n_frames"], state.get("frame0", 0))
        try:
            viewer.dims.events.current_step.connect(_on_viewer_dims)
            state["dims_callback"] = _on_viewer_dims
        except Exception:  # noqa: BLE001
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
        for emitter in (layer.events.data, getattr(layer.events, "paint", None)):
            try:
                emitter.disconnect(callback)
            except Exception:
                pass
    dims_callback = state.get("dims_callback")
    if viewer is not None and dims_callback is not None:
        try:
            viewer.dims.events.current_step.disconnect(dims_callback)
        except Exception:
            pass
    panel = state.get("panel")
    if panel is not None:
        try:
            panel._stop_play()
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
