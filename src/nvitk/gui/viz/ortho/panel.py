"""The orthogonal-views panel: binds the active layer, keeps the three views and the 3D canvas in
step, and docks itself on the Napari window."""

from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np
from qtpy.QtCore import (
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.parallel import parallel_map
from nvitk.gui.core.design import (
    FONT_S,
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_TEXT,
    SPACE_TIGHT,
)
from nvitk.gui.core.spatial import layer_spacing
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
from nvitk.gui.labels.visibility import (
    is_label_like_layer,
    invalidate_label_ids,
    layer_label_ids,
)
from nvitk.gui.viz.ortho.constants import (
    DOCK_OBJECT_NAME,
    _PLANE_LAYER_SUFFIX,
    _BOX_LAYER_SUFFIX,
    BOX_AXIS_COLORS,
    _CANVAS_SYNC_MS,
    _STYLE_SYNC_MS,
    _REBUILD_SYNC_MS,
    _VIEW_CELLS,
    _PLAY_FPS,
    PROJECTIONS,
    _PROJECTION_SHORT,
    _PROJECTION_CACHE_SIZE,
    _SLICE_CACHE_TOTAL,
)
from nvitk.gui.viz.ortho.geometry import (
    AxisView,
    _axis_views,
    PlaneFrame,
    base_frame,
    rotate_frame,
    oblique_slice,
    line_direction_px,
)
from nvitk.gui.viz.ortho.sources import (
    _RESAMPLE_BUDGET_BYTES,
    RenderSource,
    _grid_key,
    layer_time_axis,
    layer_time_count,
    is_drawable_layer,
    _is_plain_2d,
    flat_axis,
    is_flat_layer,
    is_volume_layer,
    _layer_volume,
    _same_grid,
    resample_layer_to,
    project_volume,
    _flat_views,
    SliceCache,
)
from nvitk.gui.viz.ortho.render import (
    layer_contrast,
    layer_colormap,
    layer_gamma,
    colormap_table,
    image_rgba,
    label_rgba_lut,
    _label_rgba,
    label_color_table,
    slice_to_rgba,
    _oriented,
    layer_interpolation_smooth,
    layer_interpolation_order,
    PaintRun,
    composite_runs,
)
from nvitk.gui.viz.ortho.slice_view import (  # noqa: F401
    SliceView,
)
from nvitk.gui.viz.ortho.planes3d import (
    _plane_layer_name,
    displayed_axes,
    remove_ortho_planes,
    copy_plane_style,
    sync_ortho_planes,
    remove_ortho_boxes,
    sync_ortho_boxes,
    apply_clip,
    clear_clip,
)
from nvitk.gui.viz.ortho.layout import (
    _ortho_prefs,
    _save_ortho_pref,
    PaneGrid,
)

#: Columns of the 3D options' layers table.
_COL_SLICES, _COL_PLANE, _COL_CUT, _COL_MAIN = 1, 2, 3, 4
#: Set on a layer the slice views leave out (its "Slices" box unticked); still drawn in 3D.
SLICES_HIDDEN_ATTR = "_nvitk_slices_hidden"


def slices_hidden(layer: Any) -> bool:
    """Whether the slice views leave *layer* out (it stays visible, and drawn in 3D)."""
    return bool(getattr(layer, SLICES_HIDDEN_ATTR, False))


def set_slices_hidden(layer: Any, hidden: bool) -> None:
    try:
        setattr(layer, SLICES_HIDDEN_ATTR, bool(hidden))
    except Exception:  # noqa: BLE001
        pass


class OrthoViewerPanel(QWidget):
    """Axial / coronal / sagittal views of the active layer, plus 3D plane controls."""

    def __init__(self, viewer: Any, parent: QWidget | None = None, *, embedded: bool = False) -> None:
        """Build the 2x2 panes: three orthogonal views and the 3D controls.

        *embedded*: build the parts (views, 3D options, projection and time rows) but lay none of
        them out; a host such as the central viewport takes them through the accessors below and
        calls :meth:`install_shortcuts` on itself. The panel then stays a hidden controller.
        """
        super().__init__(parent)
        self._embedded = bool(embedded)
        #: ``fn(view_index)`` a host installs to maximise a view on a title double-click.
        self.maximize_hook: Any = None
        self._viewer = viewer
        self._layer: Any | None = None
        self._data: np.ndarray | None = None
        self._views: list[AxisView] = []
        self._position: list[int] = [0, 0, 0]
        self._spacing: tuple[float, ...] = (1.0, 1.0, 1.0)
        #: Every visible layer the panel draws, bottom-to-top in layer-list order.
        self._sources: list[RenderSource] = []
        #: ``(emitter, callback)`` for every per-layer subscription, so a rebuild
        #: can drop them all rather than stacking callbacks on dead layers.
        self._subs: list[tuple[Any, Any]] = []
        #: Volumes resampled onto the active grid, keyed by layer and grid.
        self._resample_cache: dict[tuple, np.ndarray] = {}
        #: Reordered-slice memory shared by every layer this panel draws, so the
        #: ceiling is on the panel rather than on each layer separately.
        self._slice_pool: dict[str, int] = {"used": 0, "cap": _SLICE_CACHE_TOTAL}
        #: Layers whose voxels changed, waiting for the next event-loop turn.
        self._dirty_layers: list[Any] = []
        self._dirty_all = False
        #: Layers currently carrying a see-inside cut this panel applied.
        self._clipped: list[Any] = []
        #: Guards the rebuild against the layer events its own overlays raise.
        self._suspend_rebuild = False
        #: RenderSources by layer id, and the grid they were built for.
        self._source_cache: dict[int, RenderSource] = {}
        self._source_ref: tuple = ()
        #: Last slice index each view rendered, so an unchanged view is not redrawn.
        self._rendered: dict[int, int] = {}
        #: One plane per view. Equal to the axis-aligned base until a crosshair
        #: end is dragged, which is what keeps the fast slicing path in use for
        #: the overwhelmingly common case.
        self._frames: list[PlaneFrame] = []
        self._base_frames: list[PlaneFrame] = []
        #: Current time point for 3D+t layers, and how many the drawn layers have.
        self._time = 0
        self._n_time = 1
        #: The axis a bound 2D image lies across (shown in the first view only), or
        #: ``None`` for a volume.
        self._flat_axis: int | None = None
        #: Per-layer choices from the layers table, keyed by ``id(layer)``:
        #: which layers get a slice image in 3D, and which ones the cut opens.
        self._plane_choice: dict[int, bool] = {}
        self._cut_choice: dict[int, bool] = {}
        #: Source names whose 3D slice planes are on the canvas, and the layers
        #: (by ``id``) whose voxels changed since their planes were filled.
        self._planed: set[str] = set()
        self._plane_dirty: set[int] = set()
        self._table_guard = False
        #: Whether layers on another grid are resampled onto the bound one (else
        #: they are left out), and the names left out on the last rebuild.
        stored = _ortho_prefs()
        self._resample_offgrid = bool(stored.get("resample", True))
        self._offgrid_skipped: list[str] = []
        #: What the main Napari canvas draws: layers hidden there only, layers on
        #: other grids, the whole canvas. Shared through the viewer.
        from nvitk.gui.viz.main_canvas import MainCanvasFilter

        self._main_canvas = MainCanvasFilter(viewer)
        try:
            viewer._nvitk_main_canvas = self._main_canvas
        except Exception:  # noqa: BLE001
            pass
        #: The person + R/L A/P H/F marker on the 3D canvas, created on first use.
        self._marker: Any | None = None

        # Pushing planes and clipping planes to the canvas re-uploads volumes, which
        # is far too heavy to do on every step of a scroll. Coalesce them.
        self._canvas_timer = QTimer(self)
        self._canvas_timer.setSingleShot(True)
        self._canvas_timer.setInterval(_CANVAS_SYNC_MS)
        self._canvas_timer.timeout.connect(self._sync_canvas)

        # Napari's sliders emit per mouse-move; without this a contrast drag
        # re-renders three slices per event.
        self._style_timer = QTimer(self)
        self._style_timer.setSingleShot(True)
        self._style_timer.setInterval(_STYLE_SYNC_MS)
        self._style_timer.timeout.connect(self._on_style_settled)
        # Labels shown / hidden in a label list: the colours change after the
        # layer's own colormap event has been seen (the filter is written last).
        try:
            from nvitk.gui.labels.selector import label_filter_hub

            label_filter_hub().changed.connect(self._on_label_filter_changed)
        except Exception:  # noqa: BLE001
            pass

        # Adding or moving layers arrives in bursts: a tool that drops four
        # overlays on the canvas fires four inserts. Rebuild once when they stop.
        # Zero-interval: not a wait, just a way to land *after* the write
        # that the event announced, and to fold a drag's many events into one.
        self._data_timer = QTimer(self)
        self._data_timer.setSingleShot(True)
        self._data_timer.setInterval(0)
        self._data_timer.timeout.connect(self._flush_dirty_data)

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(_REBUILD_SYNC_MS)
        self._rebuild_timer.timeout.connect(self._rebuild_sources)

        # A layer moved, mirrored or reoriented (its affine): re-lay the views. Not
        # Napari's own view settings (axis order, camera): those are the main window's.
        self._geometry_dirty: list[Any] = []
        self._geometry_timer = QTimer(self)
        self._geometry_timer.setSingleShot(True)
        self._geometry_timer.setInterval(0)
        self._geometry_timer.timeout.connect(self._flush_geometry)

        self._status = QLabel("Select a 3D or 3D+t image or labels layer.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")

        # Time row: only shown while a drawn layer is 3D+t. Kept in step with the
        # Napari time slider both ways, so the cine played there drives these
        # views too.
        self._time_label = QLabel("t 1 / 1")
        self._time_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: {FONT_S}px;")
        self._time_label.setMinimumWidth(110)
        self._time_slider = QSlider(Qt.Horizontal)
        self._time_slider.setRange(0, 0)
        self._time_slider.setToolTip(
            "Time point shown for 3D+t layers. Follows (and drives) the Napari time slider."
        )
        self._time_slider.valueChanged.connect(self._on_time_slider)
        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(SPACE_TIGHT)
        time_head = QLabel("TIME")
        self._time_head = time_head
        time_head.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: {FONT_S}px; font-weight: bold; letter-spacing: 1px;"
        )
        # Cine for the panel itself: steps the time (or energy) axis on a timer and
        # moves the Napari slider with it, so the canvas and the 3D planes follow.
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setToolTip("Play / pause the time (or energy) axis, looping.")
        self._play_btn.toggled.connect(self._on_play_toggled)
        self._play_fps = QSpinBox()
        self._play_fps.setRange(1, 60)
        self._play_fps.setValue(_PLAY_FPS)
        self._play_fps.setSuffix(" fps")
        self._play_fps.setToolTip("Playback rate (frames per second).")
        # Single-shot, re-armed after each frame is drawn: a frame slower than the
        # interval delays the next instead of queueing a backlog of them.
        self._play_timer = QTimer(self)
        self._play_timer.setSingleShot(True)
        self._play_timer.timeout.connect(self._play_tick)
        time_row.addWidget(time_head)
        time_row.addWidget(self._play_btn)
        time_row.addWidget(self._time_slider, stretch=1)
        time_row.addWidget(self._time_label)
        time_row.addWidget(self._play_fps)
        self._time_box = QWidget()
        self._time_box.setLayout(time_row)
        self._time_box.setVisible(False)

        # Projection row: the three views show a projection of the volume along their
        # own axis instead of a slice; the whole volume, or a slab on the crosshair.
        proj_row = QHBoxLayout()
        proj_row.setContentsMargins(0, 0, 0, 0)
        proj_row.setSpacing(SPACE_TIGHT)
        self._proj_btn = QPushButton("Projection")
        self._proj_btn.setCheckable(True)
        self._proj_btn.setToolTip(
            "Show a projection of the volume in the three views instead of slices: along "
            "each view's own axis, with the type picked next to it. Label layers are "
            "projected by their highest id. Drawing in the views is off meanwhile."
        )
        self._proj_kind = QComboBox()
        for key, text in PROJECTIONS.items():
            self._proj_kind.addItem(_PROJECTION_SHORT[key], key)
            self._proj_kind.setItemData(self._proj_kind.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
        self._proj_kind.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        stored_kind = str(stored.get("projection", "max"))
        self._proj_kind.setCurrentIndex(max(self._proj_kind.findData(stored_kind), 0))
        self._proj_kind.setToolTip("Projection type: MIP (maximum), MinIP (minimum), AIP (average).")
        self._proj_slab = QDoubleSpinBox()
        self._proj_slab.setRange(0.0, 1000.0)
        self._proj_slab.setDecimals(1)
        self._proj_slab.setSingleStep(1.0)
        self._proj_slab.setSuffix(" mm")
        self._proj_slab.setSpecialValueText("full")
        self._proj_slab.setMinimumWidth(self._proj_slab.fontMetrics().horizontalAdvance("000.0 mm") + 40)
        self._proj_slab.setValue(float(stored.get("projection_slab_mm", 0.0) or 0.0))
        self._proj_slab.setToolTip(
            "Slab thickness, centred on the crosshair (a thin-slab projection that follows "
            "the slice). 0 (full): the whole volume along the view's axis."
        )
        self._proj_btn.toggled.connect(self._on_projection_changed)
        self._proj_kind.currentIndexChanged.connect(lambda _i: self._on_projection_changed())
        self._proj_slab.valueChanged.connect(lambda _v: self._on_projection_changed())
        # The panel docks as narrow as 360 px: the button keeps its full text.
        self._proj_btn.setMinimumWidth(self._proj_btn.sizeHint().width())
        proj_row.addWidget(self._proj_btn)
        proj_row.addWidget(self._proj_kind)
        slab_label = QLabel("slab")
        slab_label.setStyleSheet(f"color: {COLOR_MUTED};")
        proj_row.addWidget(slab_label)
        proj_row.addWidget(self._proj_slab)
        proj_row.addStretch(1)
        self._proj_box = QWidget()
        self._proj_box.setLayout(proj_row)
        try:
            viewer.dims.events.current_step.connect(self._on_viewer_dims)
        except Exception:  # noqa: BLE001 (a viewer without dims events has no time to follow)
            pass

        self._slice_views: list[SliceView] = []
        for cell, (row, col) in enumerate(_VIEW_CELLS):
            view = SliceView(AxisView(axis=cell, title="—", rows=0, cols=1))
            view.sliceChanged.connect(self._on_slice_changed)
            view.pixelPicked.connect(
                lambda row, col, index=cell: self._on_pixel_picked(index, row, col)
            )
            view.handleDragged.connect(
                lambda line, angle, index=cell: self._on_handle_dragged(index, line, angle)
            )
            view.titleDoubleClicked.connect(
                lambda index=cell, row=row, col=col: self._toggle_view_maximized(index, row, col)
            )
            view.hovered.connect(lambda pixel, index=cell: self._on_hover(index, pixel))
            self._slice_views.append(view)
        # Drawing labels in the views: the Labeling tab's tools (or the active
        # Labels layer's mode) take a click before the crosshair does.
        from nvitk.gui.labels.ortho_edit import OrthoLabelEditor

        self._label_editor = OrthoLabelEditor(self)
        for cell, view in enumerate(self._slice_views):
            view.edit_handler = functools.partial(self._label_editor.handle, cell)
            view.box_handler = functools.partial(self._label_editor.box_edge, cell)
        # Transform mode: a drag in a view moves the layers in Transform mode (Ctrl turns them).
        from nvitk.gui.viz.ortho.shape_overlays import SurfaceCutter
        from nvitk.gui.viz.ortho.transform_drag import OrthoTransformDrag

        self._transform_drag = OrthoTransformDrag(self)
        for cell, view in enumerate(self._slice_views):
            view.transform_handler = functools.partial(self._transform_drag.handle, cell)
        # Points and meshes, drawn over the slices; recut when they change (folded per turn).
        self._surface_cutter = SurfaceCutter()
        self._overlay_version = 0
        self._overlay_keys: dict[int, tuple] = {}
        self._overlay_timer = QTimer(self)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(15)
        self._overlay_timer.timeout.connect(self._refresh_overlays)
        # Painted-label refreshes during a stroke, folded to one per event-loop turn.
        self._painted: list[Any] = []
        if not self._embedded:
            self.install_shortcuts(self)
        # While the Labeling tab's Pan key is held, a left-drag pans the views too.
        for view in self._slice_views:
            view.pan_check = self._pan_key_held
        # Axial | coronal over sagittal | the 3D card, with draggable dividers;
        # the sizes are remembered between sessions.
        axial, coronal, sagittal = self._slice_views
        #: Value of the bound layer under the cursor and at the crosshair centre.
        self._readout = QLabel("")
        self._readout.setStyleSheet(f"color: {COLOR_TEXT}; font-size: {FONT_S}px;")
        self._readout.setToolTip("Bound layer's value under the cursor [voxel] and at the crosshair centre.")
        self._hover: tuple[int, int, int] | None = None
        self._crosshair_visible = bool(stored.get("crosshair", True))
        for view in self._slice_views:
            view.set_crosshair_visible(self._crosshair_visible)
        #: What a click does while a labeling tool is on (or why it cannot draw).
        self._draw_hint = QLabel("")
        self._draw_hint.setWordWrap(True)
        self._draw_hint.setVisible(False)
        self._panes: PaneGrid | None = None
        if self._embedded:
            self._build_controls()
            return
        # The 3D card scrolls rather than set a floor: its table and toggles would
        # otherwise keep its pane (and so the whole column) from being narrowed.
        controls = QScrollArea()
        controls.setWidget(self._build_controls())
        controls.setWidgetResizable(True)
        controls.setFrameShape(QFrame.NoFrame)
        controls.setMinimumSize(60, 60)
        self._panes = PaneGrid([[axial, coronal], [sagittal, controls]])
        stored_panes = stored.get("panes")
        if isinstance(stored_panes, dict):
            self._panes.set_fractions(stored_panes.get("rows"), stored_panes.get("cols"))
        self._panes_timer = QTimer(self)
        self._panes_timer.setSingleShot(True)
        self._panes_timer.setInterval(400)
        self._panes_timer.timeout.connect(self._save_pane_sizes)
        self._panes.changed.connect(self._panes_timer.start)

        self._compose()

    # composition
    #
    # The panel is two things: the logic that binds a layer and keeps the views, the time axis
    # and the 3D canvas in step, and one arrangement of the widgets that logic drives. The parts
    # below are that arrangement's pieces; a host that lays them out differently (the central
    # viewport, with the 3D options in a pop-over) takes them from these accessors instead of
    # from the panel's own layout.

    def _compose(self) -> None:
        """The dock arrangement: status lines and the projection / time rows over the 2x2 grid."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._status)
        root.addWidget(self._draw_hint)
        root.addWidget(self._proj_box)
        root.addWidget(self._time_box)
        root.addWidget(self._panes, stretch=1)

    def install_shortcuts(self, host: QWidget) -> None:
        """Undo / redo / cancel of view drawing, active while focus is inside *host*."""
        from qtpy.QtGui import QKeySequence
        from qtpy.QtWidgets import QShortcut

        for keys, slot in (
            (QKeySequence.StandardKey.Undo, self._label_editor.undo),
            (QKeySequence.StandardKey.Redo, self._label_editor.redo),
            (QKeySequence("Ctrl+Shift+Z"), self._label_editor.redo),
            (QKeySequence("Escape"), self._label_editor.cancel),
        ):
            shortcut = QShortcut(QKeySequence(keys), host)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    def readout_label(self) -> QLabel:
        """Hover / crosshair-centre value readout of the bound layer."""
        return self._readout

    # crosshair and readout
    def crosshair_visible(self) -> bool:
        """Whether the views draw their crosshair lines."""
        return self._crosshair_visible

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the crosshair lines in the three views (remembered)."""
        self._crosshair_visible = bool(visible)
        for view in self._slice_views:
            view.set_crosshair_visible(self._crosshair_visible)
        _save_ortho_pref("crosshair", self._crosshair_visible)

    def voxel_at(self, view_index: int, row: int, col: int) -> tuple[int, int, int] | None:
        """The bound layer's voxel under a view pixel (``None`` off the volume or oblique)."""
        if self._data is None or view_index >= len(self._views) or self.is_oblique():
            return None
        if self._flat_axis is not None and view_index > 0:
            return None
        shape = tuple(int(n) for n in self._data.shape[:3])
        row_axis, r, col_axis, c = self._views[view_index].to_voxel(row, col, shape)
        voxel = [int(v) for v in self._position[:3]]
        voxel[row_axis], voxel[col_axis] = int(r), int(c)
        if any(v < 0 or v >= n for v, n in zip(voxel, shape)):
            return None
        return tuple(voxel)

    def set_position(self, voxel: Sequence[int]) -> None:
        """Move the crosshair to *voxel* (the bound layer's index, current time point)."""
        if self._data is None:
            return
        shape = tuple(int(n) for n in self._data.shape[:3])
        self._position = [int(np.clip(int(v), 0, n - 1)) for v, n in zip(voxel, shape)]
        self._redraw()
        self._canvas_timer.start()
        self._update_readout()

    def pick_3d(self, position: Sequence[float], view_direction: Sequence[float],
                dims_displayed: Sequence[int]) -> tuple[int, int, int] | None:
        """The bound layer's voxel a 3D click at world *position* points at, or ``None``.

        Along the camera ray through the volume: the first labelled voxel of a label layer, the
        first voxel over the iso threshold of an iso render, else the brightest (what a MIP or
        an attenuated render shows there).
        """
        layer, data = self._layer, self._data
        if layer is None or data is None or self._flat_axis is not None:
            return None
        try:
            near, far = layer.get_ray_intersections(position, view_direction, list(dims_displayed), world=True)
        except Exception:  # noqa: BLE001
            return None
        if near is None or far is None:
            return None
        near, far = np.asarray(near, dtype=float), np.asarray(far, dtype=float)
        if near.size > 3:
            axis = layer_time_axis(layer)
            keep = [d for d in range(near.size) if d != axis][-3:]
            near, far = near[keep], far[keep]
        shape = np.asarray(data.shape[:3])
        steps = int(np.ceil(np.linalg.norm(far - near))) * 2 + 2
        points = near[None, :] + (far - near)[None, :] * np.linspace(0.0, 1.0, steps)[:, None]
        index = np.clip(np.round(points).astype(int), 0, shape - 1)
        try:
            values = np.asarray(data[index[:, 0], index[:, 1], index[:, 2]])
        except Exception:  # noqa: BLE001 (a GPU array)
            from nvitk.core.array import to_numpy

            values = to_numpy(data)[index[:, 0], index[:, 1], index[:, 2]]
        values = values.astype(float, copy=False)
        if not np.isfinite(values).any():
            return None
        if is_label_like_layer(layer):
            hits = np.flatnonzero(values != 0)
        elif str(getattr(layer, "rendering", "")) == "iso":
            hits = np.flatnonzero(values >= float(getattr(layer, "iso_threshold", np.nanmax(values))))
        else:
            hits = np.array([int(np.nanargmax(values))])
        if hits.size == 0:
            return None
        return tuple(int(v) for v in index[int(hits[0])])

    def value_at(self, voxel: Sequence[int] | None) -> Any:
        """The bound layer's value at *voxel* (current time point), or ``None``."""
        if voxel is None or self._data is None:
            return None
        try:
            return np.asarray(self._data[tuple(int(v) for v in voxel)]).item()
        except Exception:  # noqa: BLE001 (a volume on the GPU, a ragged layer...)
            try:
                from nvitk.core.array import to_numpy

                return to_numpy(self._data[tuple(int(v) for v in voxel)]).item()
            except Exception:  # noqa: BLE001
                return None

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "—"
        layer = self._layer
        if is_label_like_layer(layer) if layer is not None else False:
            try:
                from nvitk.gui.labels.catalog import label_display

                return label_display(layer, int(value))
            except Exception:  # noqa: BLE001
                return str(int(value))
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def _on_hover(self, view_index: int, pixel: Any) -> None:
        self._hover = self.voxel_at(view_index, *pixel) if pixel is not None else None
        self._update_readout()

    def _update_readout(self) -> None:
        if self._data is None:
            self._readout.setText("")
            return
        parts = []
        if self._hover is not None:
            parts.append(f"hover {self._format_value(self.value_at(self._hover))} [{', '.join(map(str, self._hover))}]")
        centre = tuple(int(v) for v in self._position[:3])
        parts.append(f"centre {self._format_value(self.value_at(centre))} [{', '.join(map(str, centre))}]")
        self._readout.setText("   ·   ".join(parts))

    def is_embedded(self) -> bool:
        """Whether a host lays the parts out (the central viewport) rather than this panel."""
        return self._embedded

    def _toggle_view_maximized(self, index: int, row: int, col: int) -> None:
        """A view title was double-clicked: that view alone, and back."""
        if self.maximize_hook is not None:
            self.maximize_hook(index)
        elif self._panes is not None:
            self._panes.toggle_maximized(row, col)

    def slice_views(self) -> list[SliceView]:
        """The axial, coronal and sagittal views, in that order."""
        return list(self._slice_views)

    def controls3d(self) -> QWidget:
        """The 3D-view options (:class:`~nvitk.gui.viz.ortho.controls3d.Controls3D`)."""
        return self._controls3d

    def projection_bar(self) -> QWidget:
        """The projection row (on/off, MIP/MinIP/AIP, slab)."""
        return self._proj_box

    def time_bar(self) -> QWidget:
        """The time row, shown only while a drawn layer is 3D+t."""
        return self._time_box

    def status_label(self) -> QLabel:
        """The one-line status (bound layer, or why nothing is shown)."""
        return self._status

    def draw_hint_label(self) -> QLabel:
        """What a click does while a labeling tool is on, or why it cannot draw."""
        return self._draw_hint

    def pane_grid(self) -> PaneGrid | None:
        """The 2x2 grid of the dock arrangement (``None`` when embedded)."""
        return self._panes

    def _save_pane_sizes(self) -> None:
        """Remember how the panes are sized (not a view shown alone, which is a moment's)."""
        if self._panes is not None and self._panes.maximized() is None:
            _save_ortho_pref("panes", self._panes.fractions())

    def _build_controls(self) -> QWidget:
        """The fourth quadrant: what the 3D canvas should show (see :class:`Controls3D`)."""
        from nvitk.gui.viz.ortho.controls3d import Controls3D

        # Parented (and hidden) until the 3D pane's options pop-over takes it: a parentless,
        # Python-owned widget is deleted by PyQt's exit cleanup mid-walk of its object map, which
        # crashed the interpreter on exit.
        self._controls3d = Controls3D(self, parent=self)
        self._controls3d.hide()
        return self._controls3d


    # binding

    def refresh_from_layer(self, layer: Any | None) -> None:
        """Bind *layer* and redraw all three views, or show why it cannot be shown."""
        usable = layer is not None and getattr(layer, "data", None) is not None
        if usable and int(getattr(layer.data, "ndim", 0)) not in (3, 4) and not _is_plain_2d(layer):
            usable = False
        if not usable:
            if self._play_btn.isChecked():
                self._play_btn.setChecked(False)
            self._layer = None
            self._data = None
            self._flat_axis = None
            self._status.setText("Select a 2D, 3D or 3D+t image or labels layer.")
            for view in self._slice_views:
                view.set_slice(None, index=0, count=1)
            return

        # Re-binding the layer already shown (a refresh, a selection round-trip)
        # keeps the crosshair where the user put it.
        same_layer = layer is self._layer and self._data is not None
        # Binding a *different* layer that sits on the same grid changes nothing
        # anyone can see: the composite is drawn from every visible layer, not
        # from the bound one, and the bound layer only supplies the geometry,
        # which is, by definition, identical. Clicking between a CT and its
        # segmentation was re-slicing and re-compositing all three views for a
        # pixel-for-pixel identical result, ~60 ms of it on a whole-body study.
        same_grid = (
            not same_layer
            and self._layer is not None
            and self._data is not None
            and _grid_key(layer) == _grid_key(self._layer)
        )
        # On another grid the crosshair keeps its place in the world: the new layer's voxel
        # there (clipped to its volume), not the middle of the new volume, so clicking
        # between layers does not move the slices the views show.
        carried = None if (same_layer or same_grid) else self._crosshair_world()
        self._layer = layer
        self._data = _layer_volume(layer, self._time)
        if self._data is None:
            self._layer = None
            self._flat_axis = None
            self._status.setText("This layer has no image to show.")
            return
        # A 2D image goes in the first (axial) view; the other two stand empty.
        self._flat_axis = flat_axis(layer)
        self._views = (
            _flat_views(layer, self._flat_axis, self._viewer)
            if self._flat_axis is not None
            else _axis_views(layer, self._viewer)
        )
        # A projection needs a volume.
        self._proj_box.setEnabled(self._flat_axis is None)
        # Keep what stays valid. On the same grid the rendered slices are the
        # ones already on screen, so clearing them would force a redraw that
        # reproduces them exactly, and resetting the crosshair below would
        # leave the panel claiming a position it is not showing.
        if not same_grid:
            self._rendered = {}
        # Windowed once for the volume: per-slice percentiles both cost more and
        # make the same tissue change brightness as you scroll.
        # Per-layer colour state lives on each RenderSource now; this layer is
        # kept only as the geometry anchor everything else is drawn against.
        # The resample cache is deliberately *not* cleared here. Its key already
        # carries the reference grid, so an entry for another grid can never be
        # returned by mistake, and throwing it away meant that switching to a
        # layer on a different grid and back re-ran a whole-volume affine
        # transform each way, seconds at a time on a large study. The budget
        # below evicts instead.
        spacing = layer_spacing(layer)
        self._spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))[:3]
        if len(self._spacing) < 3:
            self._spacing = (1.0, 1.0, 1.0)
        moved_to = self._voxel_at_world(layer, carried) if carried is not None else None
        if moved_to is not None and len(moved_to) == self._data.ndim:
            self._position = moved_to
            # The exact point, while the crosshair stays on the voxel it landed on: hopping
            # through a coarser grid and back would otherwise round it a voxel away.
            import weakref

            self._world_anchor = (np.asarray(carried, dtype=float), tuple(moved_to), weakref.ref(layer))
        elif not (same_layer or same_grid) or len(self._position) != self._data.ndim:
            self._position = [int(s) // 2 for s in self._data.shape]
        else:
            self._position = [
                int(np.clip(p, 0, int(n) - 1)) for p, n in zip(self._position, self._data.shape)
            ]

        self._base_frames = [base_frame(view, self._spacing) for view in self._views]
        if not (same_layer or same_grid) or len(self._frames) != len(self._base_frames):
            self._frames = list(self._base_frames)

        self._clip_axis.blockSignals(True)
        self._clip_axis.clear()
        for view in self._views:
            self._clip_axis.addItem(view.title, view.axis)
        self._clip_axis.blockSignals(False)

        self._update_status()
        if same_grid:
            # Sources, crosshair and rendered slices all still stand.
            self._update_composite_label()
            self._label_editor.refresh()
            return
        self._rebuild_sources()
        self._redraw(force=True)

    def _crosshair_world(self) -> np.ndarray | None:
        """The crosshair centre in world coordinates (the bound layer's real placement:
        resampling off on the canvas does not change it), ``None`` without a 3D volume."""
        layer = self._layer
        if layer is None or self._data is None or self._flat_axis is not None:
            return None
        anchor = getattr(self, "_world_anchor", None)
        if anchor is not None and anchor[2]() is layer and anchor[1] == tuple(self._position):
            return np.array(anchor[0], dtype=float)
        found = self._position_world_layer()
        if found is None:
            return None
        try:
            from nvitk.gui.viz.main_canvas import real_data_to_world

            matrix = real_data_to_world(layer)
            _layer, point = found
            if matrix.shape != (len(point) + 1, len(point) + 1):
                return None
            world = matrix @ np.append(point, 1.0)
        except Exception:  # noqa: BLE001 (a layer mid-change: start from the middle)
            return None
        t_ax = layer_time_axis(layer)
        spatial = [d for d in range(int(layer.ndim)) if d != t_ax][-3:]
        return np.asarray(world[spatial], dtype=float) if len(spatial) == 3 else None

    def _voxel_at_world(self, layer: Any, world: np.ndarray) -> list[int] | None:
        """*layer*'s voxel (spatial axes, clipped to the volume) at the world point *world*."""
        if flat_axis(layer) is not None or self._data is None or self._data.ndim != 3:
            return None
        try:
            from nvitk.gui.viz.main_canvas import real_data_to_world

            matrix = real_data_to_world(layer)
            nd = int(layer.ndim)
            if matrix.shape != (nd + 1, nd + 1):
                return None
            t_ax = layer_time_axis(layer)
            spatial = [d for d in range(nd) if d != t_ax][-3:]
            full = np.zeros(nd + 1)
            full[spatial] = world
            full[-1] = 1.0
            if t_ax is not None:
                # The time row of the matrix: whatever frame, it does not move the space axes.
                full[t_ax] = (matrix @ np.append(np.eye(nd)[t_ax] * float(self._time), 1.0))[t_ax]
            data = np.linalg.solve(matrix, full)
        except Exception:  # noqa: BLE001
            return None
        if not np.all(np.isfinite(data)):
            return None
        return [int(np.clip(int(round(float(data[d]))), 0, int(n) - 1)) for d, n in zip(spatial, self._data.shape)]

    # render sources

    def _candidate_layers(self) -> list[Any]:
        """Layers eligible to be drawn, in the viewer's own order."""
        return [
            layer
            for layer in (getattr(self._viewer, "layers", []) or [])
            if is_drawable_layer(layer)
        ]

    def _reference_key(self) -> tuple:
        """Identity of the grid everything is resampled onto.

        The grid, not the layer that happens to define it. Keying on the bound
        layer's identity threw away every cached source (host copies, slice
        caches, resampled volumes) each time the user clicked a different layer,
        even when the new one sat on exactly the same shape and affine and every
        one of those caches was still valid.
        """
        from nvitk.gui.core.spatial import layer_affine

        affine = layer_affine(self._layer)
        return (
            tuple(self._data.shape) if self._data is not None else (),
            None if affine is None else affine.tobytes(),
        )

    def _aligned_data(self, layer: Any, data: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """``(data on the active grid, was it resampled)``.

        Cached per layer and grid: the aligner is a whole-volume affine transform,
        far too heavy to run per slice, and this way hiding and re-showing an
        off-grid mask costs nothing.
        """
        if self._layer is None:
            return data, False
        if _is_plain_2d(layer) or _is_plain_2d(self._layer):
            # A plain 2D image has no placement to align by: it overlays only another
            # plain 2D image of the same size, and is never resampled (nor reported).
            same = _is_plain_2d(layer) and _is_plain_2d(self._layer) and data.shape == self._data.shape
            return (data if same else None), False
        if _same_grid(layer, self._layer):
            return data, False
        if not self._resample_offgrid:
            self._offgrid_skipped.append(str(getattr(layer, "name", "?")))
            return None, False
        t_key = self._time if layer_time_axis(layer) is not None else None
        # Nearest stays nearest; anything smoother resamples linearly (a cubic
        # whole-volume resample would cost far more than it shows).
        order = min(layer_interpolation_order(layer), 1)
        key = (id(layer), self._reference_key(), t_key, order)
        hit = self._resample_cache.get(key)
        if hit is not None:
            return hit, True
        itemsize = np.dtype(data.dtype).itemsize if order == 0 else 4
        incoming = int(np.prod(self._data.shape)) * itemsize
        self._evict_resamples(incoming)
        if incoming > _RESAMPLE_BUDGET_BYTES:
            self._status.setText(
                f"“{getattr(layer, 'name', '?')}” is on another grid and there is no "
                "room left to resample it; it is not drawn."
            )
            return None, False
        out = resample_layer_to(layer, self._layer, data, order=order)
        if out is None:
            self._status.setText(
                f"“{getattr(layer, 'name', '?')}” is on another grid and has no affine "
                "to align it by; it is not drawn."
            )
            return None, False
        self._resample_cache[key] = out
        return out, True

    def _evict_resamples(self, incoming_bytes: int) -> None:
        """Make room for *incoming_bytes*, oldest entry first.

        Entries for the grid currently on screen are spared until nothing else
        is left: those are the ones about to be drawn, and dropping them would
        mean resampling the same volumes again on the very next paint.
        """
        reference = self._reference_key()
        used = sum(int(v.nbytes) for v in self._resample_cache.values())
        if used + incoming_bytes <= _RESAMPLE_BUDGET_BYTES:
            return
        for spare_current in (True, False):
            for key in list(self._resample_cache):
                if used + incoming_bytes <= _RESAMPLE_BUDGET_BYTES:
                    return
                if spare_current and key[1] == reference:
                    continue
                used -= int(self._resample_cache.pop(key).nbytes)

    def _source_for(self, layer: Any) -> RenderSource | None:
        """Build the draw-time description of *layer*, or ``None`` if it cannot be."""
        timed = layer_time_axis(layer) is not None
        raw = _layer_volume(layer, self._time)
        if raw is None:
            return None
        data, resampled = self._aligned_data(layer, raw)
        if data is None or self._data is None or data.shape != self._data.shape:
            return None
        # Short-circuit the label heuristic on the layer class: for an Image it
        # scans the whole volume, and that is per layer per rebind.
        is_label = type(layer).__name__ == "Labels" or is_label_like_layer(layer)
        source = RenderSource(
            layer=layer,
            data=data,
            cache=SliceCache(data, pool=self._slice_pool),
            is_label=is_label,
            opacity=float(getattr(layer, "opacity", 1.0) or 1.0),
            blending=str(getattr(layer, "blending", "translucent")),
            gamma=layer_gamma(layer),
            order=0 if is_label else 1,
            resampled=resampled,
            time=min(self._time, layer_time_count(layer) - 1) if timed else None,
        )
        self._refresh_style(source)
        return source

    def _refresh_style(self, source: RenderSource) -> None:
        """Re-read the colour state of one source, leaving its data alone."""
        layer = source.layer
        source.opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
        source.blending = str(getattr(layer, "blending", "translucent"))
        # Interpolation is display state too: the screen filter, and the spline
        # order an oblique plane is resliced with.
        source.smooth = layer_interpolation_smooth(layer)
        source.order = layer_interpolation_order(layer)
        if source.is_label:
            if source.label_ids is None:
                # Asked of the *layer*, not of ``source.data``: that cache
                # survives this source being rebuilt (hiding and re-showing a
                # layer used to rescan the whole volume), and a nearest-neighbour
                # resample can only ever contain a subset of the layer's ids, so
                # the LUT it builds still covers everything that can be drawn.
                source.label_ids = layer_label_ids(layer)
            source.lut = label_rgba_lut(layer, source.label_ids)
            source.label_table = label_color_table(source.label_ids, source.lut)
            source.table = None
            source.contrast = None
        else:
            source.gamma = layer_gamma(layer)
            source.table = colormap_table(layer_colormap(layer), source.gamma)
            source.contrast = layer_contrast(layer, source.data)

    def _rebuild_sources(self) -> None:
        """Rebuild the draw list from the viewer's visible layers."""
        if self._suspend_rebuild or self._layer is None or self._data is None:
            return
        # Reuse the description a layer already had. Rebuilds happen whenever the
        # layer list changes at all, and the expensive parts (the host copy, the
        # slice cache, any resampling) depend on the layer and the reference
        # grid, neither of which a reorder or an unrelated insert touches.
        reference = self._reference_key()
        keep = self._source_cache if self._source_ref == reference else {}
        cache: dict[int, RenderSource] = {}
        sources: list[RenderSource] = []
        self._offgrid_skipped = []
        for layer in self._candidate_layers():
            if not bool(getattr(layer, "visible", True)) or slices_hidden(layer):
                continue
            source = keep.get(id(layer))
            stale_time = (
                source is not None
                and source.time is not None
                and source.time != min(self._time, layer_time_count(layer) - 1)
            )
            if source is None or source.layer is not layer or stale_time:
                if stale_time and source.cache is not None:
                    source.cache.release()
                source = self._source_for(layer)
            if source is not None:
                self._refresh_style(source)
                cache[id(layer)] = source
                sources.append(source)
        # Hand back the reordered copies of anything no longer drawn, or the pool
        # stays full of layers the user has hidden or removed.
        for layer_id, old_source in self._source_cache.items():
            if cache.get(layer_id) is not old_source and old_source.cache is not None:
                old_source.cache.release()
        self._source_cache = cache
        self._source_ref = reference
        self._sources = sources
        self._rendered.clear()
        self._watch_layers()
        self._update_composite_label()
        self._update_time_range()
        self._refresh_layers_table()
        self._redraw(force=True)
        self._apply_clip()
        self._label_editor.refresh()
        self._main_canvas.schedule()

    def _update_composite_label(self) -> None:
        """Say what the composite is currently made of."""
        if not self._sources:
            self._composite_label.setText(
                "Nothing visible to draw — the active layer is hidden."
                if self._layer is not None and not bool(getattr(self._layer, "visible", True))
                else "No visible layers on this grid."
            )
            return
        names = [str(getattr(s.layer, "name", "?")) for s in self._sources]
        resampled = [str(getattr(s.layer, "name", "?")) for s in self._sources if s.resampled]
        reference = str(getattr(self._layer, "name", "?"))
        n = len(names)
        text = f"{n} layer{'s' if n != 1 else ''} drawn"
        if resampled:
            text += f"  ·  {len(resampled)} resampled onto “{reference}”"
        if self._offgrid_skipped:
            k = len(self._offgrid_skipped)
            text += f"  ·  {k} on another grid not drawn"
        self._composite_label.setText(text)
        # The names live in the tooltip: a list of them grew the card by a line
        # per layer and pushed the 3D controls out of the quadrant.
        lines = [f"Drawn on the grid of “{reference}”, bottom to top:"]
        lines += [f"  • {name}" + ("  (resampled)" if name in resampled else "") for name in names]
        if self._offgrid_skipped:
            lines.append("On another grid, not drawn (resampling is off):")
            lines += [f"  • {name}" for name in self._offgrid_skipped]
        self._composite_label.setToolTip("\n".join(lines))

    # event subscriptions

    def _connect(self, emitter: Any, callback: Any) -> None:
        """Subscribe *callback* to *emitter* and remember it for teardown."""
        try:
            emitter.connect(callback)
            self._subs.append((emitter, callback))
        except Exception:  # noqa: BLE001
            pass

    def _unwatch_layers(self) -> None:
        """Drop every per-layer subscription."""
        for emitter, callback in self._subs:
            try:
                emitter.disconnect(callback)
            except Exception:  # noqa: BLE001 (a removed layer's emitter may be gone)
                pass
        self._subs = []

    def _watch_layers(self) -> None:
        """Follow the display state of every layer that could be drawn.

        ``visible`` is watched on every *candidate*, not just the drawn ones: a
        hidden layer being un-hidden is precisely the event that has to reach the
        panel, and it cannot if only visible layers are subscribed.
        """
        self._unwatch_layers()
        drawn = {id(s.layer) for s in self._sources}
        for layer in self._candidate_layers():
            events = getattr(layer, "events", None)
            if events is None:
                continue
            self._connect(getattr(events, "visible", None), self._on_visible_changed)
            if id(layer) not in drawn:
                continue
            for name in ("opacity", "blending", "colormap", "gamma", "contrast_limits", "contour"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    self._connect(emitter, self._on_style_changed)
            for name in ("interpolation2d", "interpolation3d"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    self._connect(
                        emitter, functools.partial(self._on_interpolation_changed, layer=layer)
                    )
            # ``data`` (the array was replaced) and ``paint`` (a brush stroke),
            # but deliberately **not** ``set_data``. Despite the name, Napari
            # emits set_data from ``Layer._refresh_sync(data_displayed=True)``:
            # every slice refresh, so once per scroll of its own canvas. Acting
            # on it rebuilt every render source and redrew all three views on
            # each notch of the main viewer's scrollbar, which on a whole-body
            # CT made the entire GUI feel broken from the moment this panel was
            # opened. Those two events cover every real edit: a brush stroke
            # fires paint, an assignment fires data, a scroll fires neither.
            for name in ("data", "paint"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    # Bound to the layer: the handler has to know *whose* cached
                    # description to throw away, and the event does not say.
                    self._connect(
                        emitter,
                        functools.partial(self._on_layer_data_changed, layer=layer),
                    )
        # Placement: a drawn layer that moves is resampled again; the bound one
        # (drawn or not) sets the views' layout; an orientation change mirrors or
        # reorders it, and the views follow as the canvas does.
        watched = [s.layer for s in self._sources]
        if self._layer is not None and all(ly is not self._layer for ly in watched):
            watched.append(self._layer)
        for layer in watched:
            events = getattr(layer, "events", None)
            for name in ("affine", "scale", "translate"):
                emitter = getattr(events, name, None) if events is not None else None
                if emitter is not None:
                    self._connect(emitter, functools.partial(self._on_geometry_changed, layer=layer))
        # Points and meshes are drawn over the slices: anything that changes how they look.
        from nvitk.gui.viz.ortho.shape_overlays import OVERLAY_TYPES

        for layer in list(getattr(self._viewer, "layers", []) or []):
            if type(layer).__name__ not in OVERLAY_TYPES:
                continue
            events = getattr(layer, "events", None)
            for name in ("data", "visible", "opacity", "affine", "scale", "translate", "rotate", "shear",
                         "face_color", "border_color", "edge_color", "size", "shown", "border_width",
                         "vertex_values", "vertex_colors", "colormap", "contrast_limits", "gamma"):
                emitter = getattr(events, name, None) if events is not None else None
                if emitter is not None:
                    self._connect(emitter, self._on_overlay_changed)

    def _on_overlay_changed(self, _event: Any = None) -> None:
        """A point or mesh layer changed: redraw the overlays on the next turn."""
        self._overlay_version += 1
        self._overlay_timer.start()

    def plane_map(self, view_index: int) -> Any:
        """View *view_index*'s :class:`~nvitk.gui.viz.ortho.geometry.PlaneMap` (slice pixel <->
        world), or ``None`` without a volume (a 2D image, nothing bound)."""
        from nvitk.gui.viz.ortho.geometry import PlaneMap

        layer = self._layer
        if layer is None or self._data is None or self._flat_axis is not None:
            return None
        if view_index >= len(self._views) or view_index >= len(self._frames):
            return None
        try:
            from nvitk.gui.viz.main_canvas import real_data_to_world

            matrix = real_data_to_world(layer)
        except Exception:  # noqa: BLE001
            return None
        nd = int(layer.ndim)
        t_ax = layer_time_axis(layer)
        spatial = [d for d in range(nd) if d != t_ax][-3:]
        if matrix.shape != (nd + 1, nd + 1) or len(spatial) != 3:
            return None
        spatial_matrix = np.eye(4)
        spatial_matrix[:3, :3] = matrix[np.ix_(spatial, spatial)]
        spatial_matrix[:3, 3] = matrix[spatial, -1]
        if t_ax is not None:
            spatial_matrix[:3, 3] += matrix[spatial, t_ax] * float(self._time)
        view = self._views[view_index]
        anchor = view.to_pixel(self._position, self._data.shape)
        return PlaneMap(
            matrix=spatial_matrix,
            centre_vox=np.asarray(self._position[:3], dtype=float),
            anchor_px=(float(anchor[0]), float(anchor[1])),
            frame=self._frames[view_index],
            steps_mm=self._plane_steps(view),
            spacing=np.asarray(self._spacing, dtype=float),
        )

    def pin_reference(self, moving: Sequence[Any]) -> None:
        """Lay the views out on a layer that is not in *moving* (see
        :func:`~nvitk.gui.viz.layer_transform.pin_ortho_reference`)."""
        bound = self._layer
        if bound is None or not any(layer is bound for layer in moving):
            return
        others = [
            layer for layer in self._candidate_layers()
            if bool(getattr(layer, "visible", True)) and is_volume_layer(layer)
            and all(layer is not m for m in moving)
        ]
        if not others:
            return
        key = _grid_key(bound)
        same = [layer for layer in others if _grid_key(layer) == key]
        self.refresh_from_layer((same or others)[0])

    def _refresh_overlays(self) -> None:
        """Recut the points and meshes for every view whose plane (or the layers) changed."""
        from nvitk.gui.viz.ortho.shape_overlays import overlay_layers, overlay_signature, view_overlays

        views = self._slice_views
        layers = overlay_layers(self._viewer)
        if (not layers or self._data is None or self._flat_axis is not None
                or self._active_projection() is not None):
            self._overlay_keys.clear()
            if not layers:
                self._surface_cutter.clear()
            for widget in views:
                widget.set_overlays([])
            return
        for index, widget in enumerate(views):
            plane = self.plane_map(index)
            if plane is None:
                self._overlay_keys.pop(index, None)
                widget.set_overlays([])
                continue
            key = (plane.key(), self._overlay_version, overlay_signature(layers))
            if self._overlay_keys.get(index) == key:
                continue
            self._overlay_keys[index] = key
            widget.set_overlays(view_overlays(self._viewer, plane, self._surface_cutter))

    def _on_visible_changed(self, _event: Any = None) -> None:
        """A layer was shown or hidden: the draw list changed."""
        self.refresh_sources()

    def _on_geometry_changed(self, _event: Any = None, *, layer: Any = None) -> None:
        """*layer* was moved, mirrored or reoriented: re-lay it out on the next turn."""
        if layer is not None:
            self._geometry_dirty.append(layer)
        self._geometry_timer.start()

    def _flush_geometry(self) -> None:
        dirty, self._geometry_dirty = self._geometry_dirty, []
        if not dirty or self._layer is None:
            return
        if any(layer is self._layer for layer in dirty):
            # Every other layer was resampled against the old placement, and the
            # turned planes against the old layout: both start again.
            self._resample_cache.clear()
            self._source_cache.clear()
            self._plane_dirty.update(id(s.layer) for s in self._sources)
            self._frames = []
            self.refresh_from_layer(self._layer)
            return
        for layer in dirty:
            for key in [k for k in self._resample_cache if k[0] == id(layer)]:
                self._resample_cache.pop(key, None)
            self._source_cache.pop(id(layer), None)
            self._plane_dirty.add(id(layer))
        self._rebuild_sources()

    def _on_label_filter_changed(self, layer: Any = None) -> None:
        """A label list showed or hid labels of *layer*: recolour if it is drawn."""
        if any(source.layer is layer for source in self._sources):
            self._on_style_changed()

    def _on_style_changed(self, _event: Any = None) -> None:
        """A colour or opacity moved: refresh the tables, keep the data."""
        for source in self._sources:
            self._refresh_style(source)
        self._style_timer.start()

    def _on_interpolation_changed(self, _event: Any = None, *, layer: Any = None) -> None:
        """A layer's interpolation changed: new screen filter, reslice order, planes.

        A layer resampled from another grid is resampled again: its cached copy
        was made with the previous order.
        """
        source = next((s for s in self._sources if s.layer is layer), None)
        if source is not None and source.resampled:
            for key in [k for k in self._resample_cache if k[0] == id(layer)]:
                self._resample_cache.pop(key, None)
            self._source_cache.pop(id(layer), None)
            self._rebuild_sources()
        else:
            for src in self._sources:
                self._refresh_style(src)
            self._redraw(force=True)
        self._sync_plane_styles()

    def _on_layer_data_changed(self, _event: Any = None, *, layer: Any = None) -> None:
        """Queue *layer*'s caches for invalidation on the next event-loop turn.

        Deferred, not immediate, because ``paint`` is emitted *before* the
        voxels are written: Napari's ``Labels.data_setitem`` saves the undo atom
        (which fires the event) and only then assigns into the array. Acting on
        it straight away re-read the volume as it was before the stroke and
        cached that. One turn later the write has landed. A drag emits an event
        per mouse move, so this coalesces them too.
        """
        if layer is not None:
            self._dirty_layers.append(layer)
        else:
            self._dirty_all = True
        self._data_timer.start()

    def _flush_dirty_data(self) -> None:
        """Drop the cached descriptions of every layer whose voxels changed.

        Dropping them from the source cache is what makes the rebuild real.
        Without this, ``_rebuild_sources`` finds the layer by ``id`` and reuses
        the old ``RenderSource`` wholesale, so replacing ``layer.data`` left the
        panel drawing the previous volume, and an edit that introduced a new
        label id drew it with no colour of its own.
        """
        dirty, self._dirty_layers = self._dirty_layers, []
        all_dirty, self._dirty_all = self._dirty_all, False
        if not dirty and not all_dirty:
            return
        if all_dirty:
            self._resample_cache.clear()
            self._source_cache.clear()
            self._plane_dirty.update(id(s.layer) for s in self._sources)
        else:
            for layer in dirty:
                # Explicitly, not by waiting for the shared cache's own
                # listener: Napari does not order event handlers, so the
                # rebuild below could otherwise re-read the ids from before
                # this edit.
                invalidate_label_ids(layer)
                for key in [k for k in self._resample_cache if k[0] == id(layer)]:
                    self._resample_cache.pop(key, None)
                self._source_cache.pop(id(layer), None)
                self._plane_dirty.add(id(layer))
        self._rebuild_sources()
        # The 3D slices hold a copy of the voxels: refill the edited layers' now.
        if self._show_planes.isChecked():
            self._sync_planes()

    def _on_layers_renamed(self, _event: Any = None) -> None:
        """A rename re-keys the 3D cut layers, which are named after the source."""
        self._rebuild_sources()

    def refresh_sources(self) -> None:
        """Ask for a rebuild once the layer-list churn settles."""
        if self._suspend_rebuild:
            return
        self._rebuild_timer.start()

    #: Kept under its old name for callers that predate the multi-layer rework.
    refresh_overlay_choices = refresh_sources

    def bound_layer(self) -> Any | None:
        """The layer the panel is currently showing, if any."""
        return self._layer

    def refresh_painted(self, layer: Any, labels: Sequence[int] = ()) -> None:
        """Show a brush stroke in progress on *layer* without rebuilding its source.

        On the views' own grid the drawn array *is* the layer's (a view of it), so
        the new voxels are already there: only the reordered slice copies are stale,
        so they are dropped and not rebuilt until the stroke ends (the commit's
        ``paint`` event rebuilds the source as any edit does). A label id the
        colour table has not seen gets its colour now.
        """
        source = self._source_cache.get(id(layer))
        if source is None or source.layer is not layer or source.resampled:
            self._on_layer_data_changed(layer=layer)
            return
        if source.cache is not None and not getattr(source.cache, "_nvitk_live", False):
            source.cache.release()
            live = SliceCache(source.data, budget_bytes=0, pool=self._slice_pool)
            live._nvitk_live = True
            source.cache = live
        known = set(source.label_ids or [])
        fresh = {int(v) for v in labels if int(v) and int(v) not in known}
        if source.is_label and fresh:
            source.label_ids = sorted(known | fresh)
            self._refresh_style(source)
        self._redraw(force=True)

    def _pan_key_held(self) -> bool:
        labeling = getattr(self._viewer, "_nvitk_labeling_panel", None)
        try:
            return labeling is not None and bool(labeling.pan_held())
        except RuntimeError:  # the tab is gone
            return False

    def set_draw_hint(self, text: str, *, error: bool = False) -> None:
        """The line saying what a click in the views does while a labeling tool is on."""
        from nvitk.gui.core.design import COLOR_ERROR

        self._draw_hint.setText(text)
        self._draw_hint.setStyleSheet(f"color: {COLOR_ERROR if error else COLOR_ACCENT};")
        self._draw_hint.setVisible(bool(text))

    def is_oblique(self) -> bool:
        """True when any plane has been turned off the volume's axes."""
        return not all(
            frame.is_close_to(base)
            for frame, base in zip(self._frames, self._base_frames)
        )

    def _plane_steps(self, view: AxisView) -> tuple[float, float]:
        """Pixel size of a view, in millimetres down and across."""
        return (float(self._spacing[view.rows]), float(self._spacing[view.cols]))

    def _resample(self, view: AxisView, index: int, data: np.ndarray, order: int) -> np.ndarray:
        """The 2D plane for *view*, obliquely if its frame has been turned."""
        frame = self._frames[index]
        shape = (int(data.shape[view.rows]), int(data.shape[view.cols]))
        return oblique_slice(
            data,
            center_vox=self._position,
            anchor_px=view.to_pixel(self._position, data.shape),
            frame=frame,
            shape=shape,
            steps_mm=self._plane_steps(view),
            spacing=self._spacing,
            order=order,
        )

    def _render_view(self, view: AxisView, view_index: int = 0) -> list[PaintRun]:
        """*view* as paint runs: every visible layer, in layer-list order, grouped by
        interpolation (see :func:`composite_runs`)."""
        index = self._position[view.axis]
        oblique = self.is_oblique()
        projection = self._active_projection()
        drawn = [source for source in self._sources if source.opacity > 0.0]

        def _rgba(source: RenderSource) -> np.ndarray:
            """One layer's slice for this view, as RGBA (thread-safe: pure NumPy)."""
            if projection is not None:
                plane = _oriented(self._projected(source, view.axis, index, projection), view.axis, view)
                if source.is_label:
                    return _label_rgba(plane, source.layer, source.lut, source.label_table)
                table = (
                    source.table if source.table is not None
                    else colormap_table(layer_colormap(source.layer), source.gamma)
                )
                return image_rgba(plane, source.contrast, table)
            if oblique:
                plane = self._resample(view, view_index, source.data, source.order)
                return (
                    _label_rgba(plane, source.layer, source.lut)
                    if source.is_label
                    else image_rgba(
                        plane,
                        source.contrast,
                        source.table
                        if source.table is not None
                        else colormap_table(layer_colormap(source.layer), source.gamma),
                    )
                )
            return slice_to_rgba(
                source.layer, source.data, view.axis, index, view,
                contrast=source.contrast, table=source.table, lut=source.lut,
                cache=source.cache, label_table=source.label_table,
            )

        # Layers render concurrently on the worker pool (serially when this view
        # is itself already one of several rendering in parallel).
        planes = parallel_map(_rgba, drawn)
        blendings = [source.blending for source in drawn]
        opacities = [source.opacity for source in drawn]
        if not planes:
            shape = (
                int(self._data.shape[view.rows]),
                int(self._data.shape[view.cols]),
            )
            return [PaintRun(np.zeros((*shape, 3), dtype=np.uint8), False)]
        return composite_runs(planes, blendings, opacities, [source.smooth for source in drawn])

    def _redraw(self, *, force: bool = False) -> None:
        """Refresh the views, re-rendering only those whose slice moved.

        A crosshair move changes at most two of the three slices; redrawing all of
        them on every scroll step is wasted work on a large volume.
        """
        if self._data is None or self._layer is None:
            return
        oblique = self.is_oblique()
        self._btn_reset_orient.setEnabled(oblique)
        self._update_readout()
        try:
            # The Labeling box and a traced vessel follow the crosshair's planes.
            self._label_editor.overlays()
        except Exception:  # noqa: BLE001 (an overlay must never stop the views)
            pass
        flat = self._flat_axis is not None
        projection = self._active_projection()
        suffix = "  ·  oblique" if oblique else ""
        if projection is not None:
            suffix = f"  ·  {self._projection_label(projection)}"
        todo: list[int] = []
        for position, (widget, view) in enumerate(zip(self._slice_views, self._views)):
            widget._view = view
            if flat and position > 0:
                # A 2D image has a single plane, drawn in the first view.
                widget._title.setText("—  (2D image)")
                widget.set_lines([])
                if force or self._rendered.get(view.axis) != -1:
                    self._rendered[view.axis] = -1
                    widget.set_slice(None, index=0, count=1)
                continue
            widget._title.setText(view.title + suffix)
            index = self._position[view.axis]
            widget.set_lines([] if flat else self._crosshair_lines(position))
            # A whole-volume projection is the same picture at every index.
            key = -2 if projection is not None and self._proj_slab.value() <= 0 else index
            # A turned plane moves with the crosshair in every direction, so the
            # "same index, skip the redraw" shortcut no longer holds.
            if force or oblique or self._rendered.get(view.axis) != key:
                self._rendered[view.axis] = key
                todo.append(position)
            elif not flat:
                # Same slice, new crosshair: repaint the lines, keep the image.
                widget.set_crosshair(view.to_pixel(self._position, self._data.shape))
        if force:
            self._overlay_version += 1
        if not todo:
            self._refresh_overlays()
            return
        # Render off the GUI thread's critical path: the moved views concurrently
        # (each oblique view instead parallelises its own resampling, which is
        # where an oblique view spends its time). Qt widgets are only touched
        # back here, on the GUI thread.
        if oblique:
            images = [self._render_view(self._views[i], i) for i in todo]
        else:
            images = parallel_map(lambda i: self._render_view(self._views[i], i), todo)
        for position, rgb in zip(todo, images):
            view = self._views[position]
            aspect = self._spacing[view.rows] / max(self._spacing[view.cols], 1e-6)
            self._slice_views[position].set_slice(
                rgb,
                index=self._position[view.axis],
                count=int(self._data.shape[view.axis]),
                aspect=aspect,
                crosshair=None if flat else view.to_pixel(self._position, self._data.shape),
            )
        self._refresh_overlays()

    # projection

    def _active_projection(self) -> str | None:
        """The projection the views show (``"max"`` / ``"min"`` / ``"mean"``), or ``None``
        for slices: always ``None`` on a 2D image."""
        if self._flat_axis is not None or not self._proj_btn.isChecked():
            return None
        return str(self._proj_kind.currentData() or "max")

    def _projection_label(self, kind: str) -> str:
        """View-title wording: ``MIP``, or ``MIP 10 mm`` for a slab."""
        slab = float(self._proj_slab.value())
        short = _PROJECTION_SHORT.get(kind, kind)
        return f"{short} {slab:g} mm" if slab > 0 else short

    def _projected(self, source: RenderSource, axis: int, index: int, kind: str) -> np.ndarray:
        """*source* projected along *axis*: the whole axis, or the slab on *index*.

        Cached on the source, so a whole-volume projection is computed once per layer and
        axis, and scrolling a slab recomputes only that slab.
        """
        n = int(source.data.shape[axis])
        lo = hi = None
        slab_mm = float(self._proj_slab.value())
        if slab_mm > 0:
            count = max(1, int(round(slab_mm / max(float(self._spacing[axis]), 1e-6))))
            lo = int(np.clip(int(index) - count // 2, 0, max(n - 1, 0)))
            hi = int(min(lo + count, n))
        key = (int(axis), kind, lo, hi)
        cache = source.projections
        hit = cache.get(key)
        if hit is None:
            hit = project_volume(source.data, axis, kind, lo, hi, label=source.is_label)
            while len(cache) >= _PROJECTION_CACHE_SIZE:
                cache.pop(next(iter(cache)))
            cache[key] = hit
        return hit

    def _on_projection_changed(self, *_args: Any) -> None:
        """Projection on / off, or its type or slab changed: redraw the three views."""
        _save_ortho_pref("projection", str(self._proj_kind.currentData() or "max"))
        _save_ortho_pref("projection_slab_mm", float(self._proj_slab.value()))
        if self._active_projection() is not None and self.is_oblique():
            # A projection runs along the volume's own axes.
            self._frames = list(self._base_frames)
        self._rendered.clear()
        self._redraw(force=True)
        self._label_editor.refresh()

    # interaction

    def _on_slice_changed(self, axis: int, index: int) -> None:
        """Move the crosshair along *axis* and refresh the other two views."""
        if self._data is None:
            return
        self._position[int(axis)] = int(index)
        self._label_editor.slice_moved()
        self._redraw()
        self._canvas_timer.start()

    def _on_pixel_picked(self, view_index: int, row: int, col: int) -> None:
        """Place the crosshair from a click inside the view at *view_index*."""
        if self._data is None or view_index >= len(self._views):
            return
        if self._flat_axis is not None and view_index > 0:
            return  # an empty view of a 2D image
        view = self._views[view_index]
        row_axis, row_value, col_axis, col_value = view.to_voxel(row, col, self._data.shape)
        self._position[row_axis] = row_value
        self._position[col_axis] = col_value
        self._redraw()
        self._canvas_timer.start()

    def _crosshair_lines(self, view_index: int) -> list[tuple[float, float, str]]:
        """Where the other two planes cut across this view, with their colours.

        Each line is coloured like the view it belongs to (the same red / green /
        blue the 3D outlines use), so dragging one says which plane is about to
        turn without having to watch all three.
        """
        if not self._frames or view_index >= len(self._frames):
            return []
        frame = self._frames[view_index]
        steps = self._plane_steps(self._views[view_index])
        out: list[tuple[float, float, str]] = []
        for other in self._other_views(view_index):
            direction = line_direction_px(frame, self._frames[other].normal, steps)
            if direction is None:
                continue
            colour = BOX_AXIS_COLORS[self._views[other].axis % len(BOX_AXIS_COLORS)]
            out.append((direction[0], direction[1], colour))
        return out

    def _other_views(self, view_index: int) -> list[int]:
        """The two view positions whose planes are drawn as lines on *view_index*."""
        return [i for i in range(len(self._frames)) if i != int(view_index)]

    def _on_handle_dragged(self, view_index: int, line: int, screen_angle: float) -> None:
        """Turn the other two planes about this view's normal, following the drag.

        Only the other two move: the view being dragged in has to stay still, or
        the line would run away from the cursor that is steering it, which is
        also what makes this read as reorienting the volume rather than spinning
        the picture.
        """
        if self._data is None or view_index >= len(self._frames):
            return
        if self._flat_axis is not None or self._active_projection() is not None:
            return  # a 2D image has no planes to turn; a projection runs along the axes
        others = self._other_views(view_index)
        if line >= len(others):
            return
        frame = self._frames[view_index]
        steps = self._plane_steps(self._views[view_index])
        other_normal = self._frames[others[line]].normal
        trace = np.cross(np.asarray(frame.normal, float), np.asarray(other_normal, float))
        if float(np.linalg.norm(trace)) <= 1e-9:
            return
        # The cursor angle is measured in *pixels*, and a pixel is not square on
        # anisotropic spacing, so it is converted back through the two step sizes
        # before being turned into a direction in the plane. Solving the angle on
        # screen instead would under- or over-rotate by the aspect ratio.
        target = (
            float(np.sin(screen_angle)) * steps[0] * np.asarray(frame.row, float)
            + float(np.cos(screen_angle)) * steps[1] * np.asarray(frame.col, float)
        )
        if float(np.linalg.norm(target)) <= 1e-9:
            return
        # Turning the other planes about this one's normal turns their trace by
        # exactly the same angle, so the signed angle from trace to target *is*
        # the rotation to apply.
        angle = float(
            np.arctan2(
                float(np.dot(np.cross(trace, target), frame.normal)),
                float(np.dot(trace, target)),
            )
        )
        # A line has no head or tail: steer to the nearer of its two ends.
        angle = (angle + np.pi / 2.0) % np.pi - np.pi / 2.0
        if abs(angle) < 1e-9:
            return
        for other in others:
            self._frames[other] = rotate_frame(self._frames[other], frame.normal, angle)
        self._redraw(force=True)
        self._canvas_timer.start()
        self._label_editor.refresh()

    def _reset_orientation(self) -> None:
        """Put every plane back on the volume's own axes."""
        if not self._base_frames:
            return
        self._frames = list(self._base_frames)
        self._redraw(force=True)
        self._canvas_timer.start()
        self._label_editor.refresh()

    def _centre_crosshair(self) -> None:
        """Put the crosshair back at the middle of the volume."""
        if self._data is None:
            return
        self._position = [int(s) // 2 for s in self._data.shape]
        self._redraw()
        self._canvas_timer.start()

    def _on_resample_toggled(self, enabled: bool) -> None:
        """Draw off-grid layers resampled, or leave them out."""
        self._resample_offgrid = bool(enabled)
        _save_ortho_pref("resample", self._resample_offgrid)
        if not enabled:
            # Nothing will read them until resampling is back on.
            self._resample_cache.clear()
        # Sources built either way are stale: drop them all and rebuild.
        for source in self._source_cache.values():
            if source.cache is not None:
                source.cache.release()
        self._source_cache = {}
        self._source_ref = ()
        self._rebuild_sources()

    def _marker_reference_layer(self) -> Any | None:
        """The layer the orientation marker reads anatomy from: the bound one."""
        if self._layer is not None:
            return self._layer
        layers = getattr(self._viewer, "layers", None)
        return layers.selection.active if layers else None

    def _on_marker_toggled(self, enabled: bool) -> None:
        """Show the person + R/L A/P H/F marker on the 3D canvas, or hide it."""
        _save_ortho_pref("orientation_marker", bool(enabled))
        if self._marker is None and enabled:
            from nvitk.gui.viz.orientation_overlay import orientation_marker_for

            self._marker = orientation_marker_for(self._viewer, self._marker_reference_layer)
            if self._marker is None:
                self._status.setText("The orientation figure needs the Napari 3D canvas (vispy + pyvista).")
                self._marker_box.blockSignals(True)
                self._marker_box.setChecked(False)
                self._marker_box.blockSignals(False)
                return
        if self._marker is None:
            return
        self._marker.set_enabled(bool(enabled))
        if enabled and int(getattr(self._viewer.dims, "ndisplay", 2)) != 3:
            self._show_3d()

    def restore_marker_preference(self) -> None:
        """Turn the orientation figure back on if it was on last session."""
        if bool(_ortho_prefs().get("orientation_marker", False)) and not self._marker_box.isChecked():
            self._marker_box.setChecked(True)

    def _show_3d(self) -> None:
        """Switch the Napari canvas to its 3D view."""
        try:
            self._viewer.dims.ndisplay = 3
        except Exception:
            pass

    def _wants_slice_image(self) -> bool:
        """Whether the slice image is drawn alongside the always-on outline."""
        return bool(self._show_slice_image.isChecked())

    def _on_show_planes(self, enabled: bool) -> None:
        """Add or remove the 3D cut layers on the canvas."""
        if self._layer is None or _is_plain_2d(self._layer):
            return
        if not enabled:
            self._remove_all_planes()
            remove_ortho_boxes(self._viewer, self._cut_source_name())
            self._show_slice_image.setEnabled(False)
            return
        self._show_slice_image.setEnabled(True)
        try:
            self._draw_cuts()
            self._show_3d()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not add 3D cuts: {exc}")
            self._show_planes.blockSignals(True)
            self._show_planes.setChecked(False)
            self._show_planes.blockSignals(False)
            self._show_slice_image.setEnabled(False)

    def _on_plane_image_toggled(self) -> None:
        """Add or drop the slice images; the outlines stay either way."""
        if self._layer is None or not self._show_planes.isChecked():
            return
        if not self._wants_slice_image():
            self._remove_all_planes()
        try:
            self._draw_cuts()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not redraw the 3D cuts: {exc}")

    def _remove_all_planes(self) -> None:
        """Drop every slice-image plane this panel put on the canvas."""
        self._suspend_rebuild = True
        try:
            for name in set(self._planed) | {self._cut_source_name()}:
                remove_ortho_planes(self._viewer, name)
        finally:
            self._suspend_rebuild = False
        self._planed.clear()

    def _cut_source_name(self) -> str:
        """Name the 3D cut layers are keyed on: the bound layer's."""
        return str(getattr(self._layer, "name", "volume"))

    # per-layer choices

    def _wants_plane(self, source: RenderSource) -> bool:
        """Whether *source*'s slice image goes on the 3D planes.

        Defaults to the bound layer only: the historical behaviour. A layer
        resampled from another grid cannot: plane positions are voxel indices
        of the bound grid, which are not its own.
        """
        if source.resampled:
            return False
        key = id(source.layer)
        if key not in self._plane_choice:
            # Decided once, when the layer is first drawn. Following "whichever
            # layer is bound now" flipped the 3D slices to a segmentation the moment
            # it was clicked, on the same grid nothing redraws, so the table said
            # one thing and the canvas another until the crosshair next moved.
            self._plane_choice[key] = source.layer is self._layer
        return self._plane_choice[key]

    def _main_item(self, layer: Any) -> QTableWidgetItem:
        """The "Main" cell: drawn on the main canvas, or hidden there only."""
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        shown = not self._main_canvas.is_manually_hidden(layer)
        item.setCheckState(Qt.Checked if shown else Qt.Unchecked)
        return item

    def _on_main_canvas_changed(self) -> None:
        """Keep the "Main" column and the two canvas boxes in step with the filter."""
        for box, value in (
            (self._main_resample_box, self._main_canvas.resample()),
            (self._main_canvas_box, self._main_canvas.canvas_shown()),
        ):
            if box.isChecked() != value:
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)
        layers = getattr(self, "_table_layers", [])
        self._table_guard = True
        try:
            for row, layer in enumerate(layers):
                if row < self._layers_table.rowCount():
                    self._layers_table.setItem(row, _COL_MAIN, self._main_item(layer))
        finally:
            self._table_guard = False

    def restore_canvas_preference(self) -> None:
        """Hide the main canvas if it was hidden last session (the dock exists by now)."""
        if not self._main_canvas.canvas_shown():
            self._main_canvas.set_canvas_shown(False, remember=False)

    def _wants_cut(self, layer: Any) -> bool:
        """Whether the see-inside cut opens *layer* (default: every drawn layer)."""
        return self._cut_choice.get(id(layer), True)

    def _table_rows(self) -> list[tuple[Any, Any]]:
        """``(layer, its render source or None)`` per row: the drawn layers, those taken out of
        the slice views, and the mesh / points layers, in the layer list's order."""
        from nvitk.gui.viz.ortho.shape_overlays import OVERLAY_TYPES

        drawn = {id(source.layer): source for source in self._sources}
        rows = []
        for layer in list(getattr(self._viewer, "layers", []) or []):
            if id(layer) in drawn:
                rows.append((layer, drawn[id(layer)]))
            elif type(layer).__name__ in OVERLAY_TYPES:
                rows.append((layer, None))
            elif slices_hidden(layer) and is_drawable_layer(layer) and bool(getattr(layer, "visible", True)):
                rows.append((layer, None))
        return rows

    def _refresh_layers_table(self) -> None:
        """One row per layer the views can draw, with its Slices / 3D slice / Cut / Main boxes."""
        self._table_guard = True
        try:
            table = self._layers_table
            rows = self._table_rows()
            table.setRowCount(len(rows))
            self._table_layers = [layer for layer, _source in rows]

            def box(checked: bool, enabled: bool = True, tip: str = "") -> QTableWidgetItem:
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsUserCheckable | (Qt.ItemIsEnabled if enabled else Qt.NoItemFlags))
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                if tip:
                    item.setToolTip(tip)
                return item

            def blank(tip: str) -> QTableWidgetItem:
                item = QTableWidgetItem("")
                item.setFlags(Qt.NoItemFlags)
                item.setToolTip(tip)
                return item

            for row, (layer, source) in enumerate(rows):
                name = QTableWidgetItem(str(getattr(layer, "name", "?")))
                name.setFlags(Qt.ItemIsEnabled)
                table.setItem(row, 0, name)
                table.setItem(row, _COL_SLICES, box(not slices_hidden(layer)))
                if source is not None:
                    table.setItem(row, _COL_PLANE, box(
                        self._wants_plane(source), not source.resampled,
                        "Resampled from another grid: no 3D slice image." if source.resampled else ""))
                    table.setItem(row, _COL_CUT, box(self._wants_cut(layer)))
                else:
                    tip = ("Not in the slice views: no slice image or cut." if slices_hidden(layer)
                           and type(layer).__name__ not in ("Points", "Surface")
                           else "Meshes and points have no slice image or cut.")
                    table.setItem(row, _COL_PLANE, blank(tip))
                    table.setItem(row, _COL_CUT, blank(tip))
                table.setItem(row, _COL_MAIN, self._main_item(layer))
        finally:
            self._table_guard = False

    def _on_table_item_changed(self, item: Any) -> None:
        """A Slices / 3D slice / Cut / Main box was toggled in the layers table."""
        if self._table_guard or item is None:
            return
        row, col = item.row(), item.column()
        layers = getattr(self, "_table_layers", [])
        if row >= len(layers) or col not in (_COL_SLICES, _COL_PLANE, _COL_CUT, _COL_MAIN):
            return
        layer = layers[row]
        checked = item.checkState() == Qt.Checked
        if col == _COL_MAIN:
            self._main_canvas.set_manually_hidden(layer, not checked)
        elif col == _COL_SLICES:
            set_slices_hidden(layer, not checked)
            from nvitk.gui.viz.ortho.shape_overlays import OVERLAY_TYPES

            if type(layer).__name__ in OVERLAY_TYPES:
                self._on_overlay_changed()
            else:
                self._rebuild_sources()
        elif col == _COL_PLANE:
            self._plane_choice[id(layer)] = checked
            if self._show_planes.isChecked():
                try:
                    self._draw_cuts()
                except Exception as exc:  # noqa: BLE001
                    self._status.setText(f"Could not redraw the 3D cuts: {exc}")
        else:
            self._cut_choice[id(layer)] = checked
            self._apply_clip()

    def _draw_cuts(self) -> None:
        """Push the three cuts to the canvas: outlines always, images on request.

        Guarded: adding an overlay layer fires ``layers.events.inserted``, which
        rebuilds the draw list, which redraws, which lands back here.
        """
        self._suspend_rebuild = True
        try:
            # Both helpers walk axes (0, 1, 2); self._frames and self._views are
            # in *view* order, which _axis_views sorts by plane name. Re-key them
            # by array axis or every cut gets another view's frame.
            frames = views = None
            if self._frames and self._views:
                by_axis = {int(v.axis): i for i, v in enumerate(self._views)}
                order = [by_axis.get(a) for a in (0, 1, 2)]
                views = [self._views[i] if i is not None else None for i in order]
                if self.is_oblique():
                    frames = [self._frames[i] if i is not None else None for i in order]
            sync_ortho_boxes(
                self._viewer, self._layer, tuple(self._position),
                frames=frames, spacing=self._spacing, views=views,
            )
            wanted: set[str] = set()
            refresh, self._plane_dirty = self._plane_dirty, set()
            if self._wants_slice_image():
                stacked = 0
                for source in self._sources:
                    if not self._wants_plane(source):
                        continue
                    name = str(getattr(source.layer, "name", "volume"))
                    sync_ortho_planes(
                        self._viewer, source.layer, tuple(self._position),
                        frames=frames, spacing=self._spacing, time_index=self._time,
                        # Bottom-to-top: each layer above wraps the slab below it.
                        thickness=1.0 + 0.6 * stacked,
                        refresh_data=id(source.layer) in refresh,
                    )
                    stacked += 1
                    wanted.add(name)
            for name in self._planed - wanted:
                remove_ortho_planes(self._viewer, name)
            self._planed = wanted
        finally:
            self._suspend_rebuild = False

    def _sync_plane_styles(self) -> None:
        """Re-copy window, colours and gamma onto the planes of every planed layer."""
        if not self._planed:
            return
        by_name = {str(getattr(l, "name", "")): l for l in self._viewer.layers}
        for source in self._sources:
            name = str(getattr(source.layer, "name", ""))
            if name not in self._planed:
                continue
            for axis in (0, 1, 2):
                plane = by_name.get(_plane_layer_name(name, axis))
                if plane is not None:
                    copy_plane_style(source.layer, plane)

    def _on_style_settled(self) -> None:
        """A burst of colour/opacity changes ended: redraw, and restyle the 3D planes."""
        self._redraw(force=True)
        self._sync_plane_styles()

    def _sync_canvas(self) -> None:
        """Push the crosshair to the 3D planes, the 3D crosshair and the clip, once per settle."""
        self._sync_planes()
        self._sync_cross3d()
        self._apply_clip()

    def _sync_cross3d(self) -> None:
        """Draw (or drop) the crosshair lines on the 3D canvas."""
        from nvitk.gui.viz.ortho.planes3d import remove_ortho_crosshair, sync_ortho_crosshair

        box = getattr(self, "_cross3d_box", None)
        if self._layer is None or _is_plain_2d(self._layer):
            return
        name = str(getattr(self._layer, "name", "volume"))
        self._suspend_rebuild = True
        try:
            if box is not None and box.isChecked():
                sync_ortho_crosshair(self._viewer, self._layer, tuple(self._position))
            else:
                remove_ortho_crosshair(self._viewer, name)
        except Exception:  # noqa: BLE001 (a 2D or odd layer: no 3D crosshair)
            pass
        finally:
            self._suspend_rebuild = False

    def _on_cross3d_toggled(self, enabled: bool) -> None:
        _save_ortho_pref("crosshair_3d", bool(enabled))
        self._sync_cross3d()
        click = getattr(self, "_cross3d_click", None)
        if not enabled and click is not None and click.isChecked():
            click.setChecked(False)  # nothing to grab without the lines

    def _on_cross3d_click_toggled(self, enabled: bool) -> None:
        _save_ortho_pref("crosshair_3d_click", bool(enabled))
        box = getattr(self, "_cross3d_box", None)
        if enabled and box is not None and not box.isChecked():
            box.setChecked(True)  # the crosshair has to show to be clicked about

    def cross3d_interactive(self) -> bool:
        """True while a plain click on the 3D canvas moves the crosshair."""
        click = getattr(self, "_cross3d_click", None)
        return click is not None and click.isChecked()

    def _position_world_layer(self) -> tuple[Any, np.ndarray] | None:
        """``(bound layer, crosshair point as a full data vector)`` (time axis at the shown frame)."""
        layer = self._layer
        if layer is None or self._data is None or self._flat_axis is not None:
            return None
        nd = int(layer.ndim)
        t_ax = layer_time_axis(layer)
        spatial = [d for d in range(nd) if d != t_ax][-3:]
        point = np.zeros(nd, dtype=float)
        for k, d in enumerate(spatial):
            point[d] = float(self._position[k])
        if t_ax is not None:
            point[t_ax] = float(self._time)
        return layer, point

    def cross3d_grab(self, position: Sequence[float], view_direction: Sequence[float]) -> np.ndarray | None:
        """The crosshair point (data vector) when the cursor is over it on the 3D canvas."""
        from nvitk.gui.viz.ray3d import closest_on_ray, ray_in_data, screen_px

        found = self._position_world_layer()
        if found is None or view_direction is None:
            return None
        layer, point = found
        P, D = ray_in_data(layer, position, view_direction)
        on_ray, _t = closest_on_ray(P, D, point)
        return point if screen_px(self._viewer, layer, on_ray, point) <= 10.0 else None

    def cross3d_drag_to(self, position: Sequence[float], view_direction: Sequence[float], anchor: np.ndarray) -> None:
        """Slide the crosshair, grabbed at *anchor*, to the cursor (in the plane facing the camera)."""
        from nvitk.gui.viz.ray3d import drag_point

        found = self._position_world_layer()
        if found is None:
            return
        layer, _point = found
        hit = drag_point(layer, position, view_direction, anchor)
        if hit is None:
            return
        t_ax = layer_time_axis(layer)
        spatial = [d for d in range(int(layer.ndim)) if d != t_ax][-3:]
        voxel = [int(round(float(hit[d]))) for d in spatial]
        if voxel != [int(v) for v in self._position[:3]]:
            self.set_position(voxel)

    def pick_3d_click(self, position: Sequence[float], view_direction: Sequence[float],
                      dims_displayed: Sequence[int]) -> tuple[int, int, int] | None:
        """Where a click on the 3D canvas puts the crosshair: on the nearest shown slice the ray
        meets, else what the rendering shows there (:meth:`pick_3d`)."""
        from nvitk.gui.viz.ray3d import ray_in_data

        found = self._position_world_layer()
        planes = getattr(self, "_show_planes", None)
        if found is not None and planes is not None and planes.isChecked():
            layer, _point = found
            t_ax = layer_time_axis(layer)
            spatial = [d for d in range(int(layer.ndim)) if d != t_ax][-3:]
            P, D = ray_in_data(layer, position, view_direction)
            shape = [int(n) for n in self._data.shape[:3]]
            best = None
            for k, d in enumerate(spatial):
                if abs(float(D[d])) < 1e-9:
                    continue
                t = (float(self._position[k]) - float(P[d])) / float(D[d])
                hit = P + t * D
                voxel = [int(round(float(hit[e]))) for e in spatial]
                voxel[k] = int(self._position[k])
                if all(-0.5 <= float(hit[e]) <= n - 0.5 for e, n in zip(spatial, shape)) and (best is None or t < best[0]):
                    best = (t, voxel)
            if best is not None:
                return tuple(int(min(max(v, 0), n - 1)) for v, n in zip(best[1], shape))
        return self.pick_3d(position, view_direction, dims_displayed)

    def _sync_planes(self) -> None:
        """Move the 3D cuts to the crosshair, if they are showing."""
        if self._layer is None or not self._show_planes.isChecked() or _is_plain_2d(self._layer):
            return
        try:
            self._draw_cuts()
        except Exception:
            pass

    def _apply_clip(self) -> None:
        """Apply (or clear) the see-inside cut on the layers ticked for it.

        Every drawn layer is cut by default, overlays included: cutting only the
        base leaves a segmentation floating in front of the opened volume,
        covering the very interior the cut was made to expose. The layers table
        takes any of them out: keep a mask whole while the CT around it opens.
        """
        if self._layer is None or _is_plain_2d(self._layer):
            return  # a plain 2D image has nothing to cut in 3D
        side = str(self._clip_side.currentData() or "off")
        axis_data = self._clip_axis.currentData()
        axis = int(axis_data) if axis_data is not None else 0
        displayed = displayed_axes(self._viewer)
        drawn = [s.layer for s in self._sources] or [self._layer]
        targets = [layer for layer in drawn if self._wants_cut(layer)]
        # A layer that dropped out of the composite (or was unticked) keeps
        # whatever cut it was given otherwise, and nothing would take it off.
        for stale in self._clipped:
            if stale not in targets:
                clear_clip(stale)
        self._clipped = list(targets)
        for target in targets:
            try:
                t_axis, t_index, flipped = self._cut_in(target, axis)
                t_side = side
                if flipped and side in ("below", "above"):
                    t_side = "above" if side == "below" else "below"
                apply_clip(target, t_axis, t_index, t_side, displayed)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Could not clip {getattr(target, 'name', '?')}: {exc}")

    def _cut_in(self, layer: Any, axis: int) -> tuple[int, float, bool]:
        """The cut through the crosshair along the bound grid's *axis*, in *layer*'s
        own grid: ``(its axis, index along it, whether that axis runs the other way)``.

        On the bound grid that is the crosshair index itself. A layer on another
        grid (a PET under a CT) has its own voxel spacing and origin, so the bound
        grid's index would cut it somewhere else: the crosshair goes through world
        space into its voxels, and its axis closest to the bound one is cut.
        """
        bound = self._layer
        if layer is bound or bound is None or _same_grid(layer, bound):
            return int(axis), float(self._position[axis]), False
        b_dims = [d for d in range(int(bound.ndim)) if d != layer_time_axis(bound)][-3:]
        t_dims = [d for d in range(int(layer.ndim)) if d != layer_time_axis(layer)][-3:]
        point = [0.0] * int(bound.ndim)
        for k, d in enumerate(b_dims):
            point[d] = float(self._position[k])
        step = list(point)
        step[b_dims[int(axis)]] += 1.0
        here = np.asarray(layer.world_to_data(bound.data_to_world(point)), dtype=float)
        there = np.asarray(layer.world_to_data(bound.data_to_world(step)), dtype=float)
        direction = np.array([there[d] - here[d] for d in t_dims])
        k = int(np.argmax(np.abs(direction)))
        return k, float(here[t_dims[k]]), bool(direction[k] < 0)

    # time (3D+t)

    def _time_layer(self) -> Any | None:
        """The 3D+t layer whose time axis the panel follows (bound first)."""
        if self._layer is not None and layer_time_axis(self._layer) is not None:
            return self._layer
        for source in self._sources:
            if layer_time_axis(source.layer) is not None:
                return source.layer
        return None

    def _time_dim(self) -> int | None:
        """World (dims) index of the followed time axis, or ``None``."""
        layer = self._time_layer()
        if layer is None:
            return None
        try:
            offset = int(self._viewer.dims.ndim) - int(layer.data.ndim)
            return offset + int(layer_time_axis(layer))
        except Exception:  # noqa: BLE001
            return None

    def _update_time_range(self) -> None:
        """Show the time row when something drawn is 3D+t, sized to the longest."""
        counts = [layer_time_count(s.layer) for s in self._sources]
        if self._layer is not None:
            counts.append(layer_time_count(self._layer))
        n = max(counts or [1])
        self._n_time = int(n)
        self._time_box.setVisible(self._n_time > 1)
        if self._n_time <= 1 and self._play_btn.isChecked():
            self._play_btn.setChecked(False)
        self._time_slider.blockSignals(True)
        self._time_slider.setRange(0, max(self._n_time - 1, 0))
        self._time = int(np.clip(self._time, 0, max(self._n_time - 1, 0)))
        self._time_slider.setValue(self._time)
        self._time_slider.blockSignals(False)
        self._update_time_label()

    def _update_time_label(self) -> None:
        """``t 3 / 15  ·  0.120 s``: the frame and, when known, its time."""
        text = f"t {self._time + 1} / {self._n_time}"
        layer = self._time_layer()
        if layer is not None:
            from nvitk.gui.core.spatial import nvitk_metadata_from_layer

            md = nvitk_metadata_from_layer(layer)
            times = md.get("frame_times_s")
            phases = md.get("cardiac_phases_percent")
            energies = md.get("spectral_energies_kev")
            self._time_head.setText("ENERGY" if isinstance(energies, (list, tuple)) else "TIME")
            if isinstance(energies, (list, tuple)) and self._time < len(energies):
                text = f"E {self._time + 1} / {self._n_time}  ·  {float(energies[self._time]):g} keV"
            elif isinstance(phases, (list, tuple)) and self._time < len(phases):
                text += f"  ·  {float(phases[self._time]):g}% R-R"
            elif isinstance(times, (list, tuple)) and self._time < len(times):
                text += f"  ·  {float(times[self._time]):.3g} s"
            else:
                t_res = md.get("t_res", md.get("temporal_resolution"))
                if t_res:
                    text += f"  ·  {self._time * float(t_res):.3g} s"
        self._time_label.setText(text)

    def _update_status(self) -> None:
        """Bound layer, its grid, and the time point when it is 3D+t."""
        if self._layer is None or self._data is None:
            return
        name = getattr(self._layer, "name", "layer")
        if self._flat_axis is not None:
            plane = [int(v) for i, v in enumerate(self._data.shape) if i != self._flat_axis]
            extra = ""
            if layer_time_axis(self._layer) is not None:
                extra = f"  ·  2D+t, {layer_time_count(self._layer)} time points"
            self._status.setText(f"{name} - 2D image, {' x '.join(map(str, plane))} pixels{extra}")
            return
        shape = " x ".join(str(int(v)) for v in self._data.shape)
        extra = ""
        if layer_time_axis(self._layer) is not None:
            extra = f"  ·  3D+t, {layer_time_count(self._layer)} time points"
        self._status.setText(f"{name} - {shape} voxels{extra}")

    def _on_play_toggled(self, playing: bool) -> None:
        """Start or stop the cine."""
        self._play_btn.setText("⏸ Pause" if playing else "▶ Play")
        if playing and self._n_time > 1:
            self._play_timer.start(int(1000 / max(int(self._play_fps.value()), 1)))
        else:
            self._play_timer.stop()

    def _play_tick(self) -> None:
        """Advance one frame (looping) and re-arm the timer."""
        if not self._play_btn.isChecked() or self._n_time <= 1:
            return
        self._set_time((self._time + 1) % self._n_time, sync_viewer=True)
        self._play_timer.start(int(1000 / max(int(self._play_fps.value()), 1)))

    def is_playing(self) -> bool:
        """True while the cine is running."""
        return bool(self._play_btn.isChecked())

    def _on_time_slider(self, value: int) -> None:
        """The panel's own time slider moved."""
        self._set_time(int(value), sync_viewer=True)

    def _on_viewer_dims(self, _event: Any = None) -> None:
        """Follow the Napari time slider (cheap: compares one index per event)."""
        if self._n_time <= 1 or getattr(self, "_time_guard", False):
            return
        dim = self._time_dim()
        if dim is None:
            return
        try:
            t = int(self._viewer.dims.current_step[dim])
        except Exception:  # noqa: BLE001
            return
        if t != self._time:
            self._set_time(t, sync_viewer=False)

    def _set_time(self, t: int, *, sync_viewer: bool) -> None:
        """Show time point *t* in every 3D+t layer the panel draws."""
        t = int(np.clip(int(t), 0, max(self._n_time - 1, 0)))
        if t == self._time:
            return
        self._time = t
        self._time_slider.blockSignals(True)
        self._time_slider.setValue(t)
        self._time_slider.blockSignals(False)
        self._update_time_label()
        if self._layer is not None and layer_time_axis(self._layer) is not None:
            data = _layer_volume(self._layer, t)
            if data is not None:
                self._data = data
        # 3D+t sources are rebuilt at the new time point; 3D ones are reused.
        self._rebuild_sources()
        self._canvas_timer.start()
        if sync_viewer:
            dim = self._time_dim()
            if dim is not None:
                self._time_guard = True
                try:
                    self._viewer.dims.set_current_step(dim, t)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._time_guard = False

    def position(self) -> tuple[int, int, int]:
        """Current crosshair, as voxel indices in array-axis order."""
        return tuple(int(v) for v in self._position)


def open_ortho_views(viewer: Any, layer: Any | None = None) -> OrthoViewerPanel:
    """Open the orthogonal-views dock, or bring the existing one back into view.

    Kept on the viewer so the quick-access button and the Visualization tool both
    reach the same panel instead of stacking up docks.
    """
    panel = getattr(viewer, "_nvitk_ortho_panel", None)
    viewport = getattr(viewer, "_nvitk_viewport", None)
    if panel is not None and viewport is not None:
        # The views live in the central viewport: make sure some are on screen.
        viewport.ensure_slices_visible()
    if panel is None:
        panel = OrthoViewerPanel(viewer)
        attach_ortho_dock(viewer, panel)
        try:
            viewer._nvitk_ortho_panel = panel
        except Exception:
            pass
        # Next turn: the canvas has to exist before the marker can be put on it.
        QTimer.singleShot(0, panel.restore_marker_preference)
        QTimer.singleShot(0, panel.restore_canvas_preference)
    dock = getattr(panel, "_nvitk_dock", None)
    if dock is not None:
        dock.show()
        dock.raise_()
    if layer is None and getattr(viewer, "layers", None):
        layer = viewer.layers.selection.active
    panel.refresh_from_layer(layer)
    return panel


def attach_ortho_dock(viewer: Any, panel: OrthoViewerPanel) -> Any:
    """Dock *panel* on the Napari window and keep it on the active layer."""
    from nvitk.gui.core.design import apply_theme

    apply_theme(panel)
    # Not ``viewer.window.add_dock_widget``: that returns a Napari QtViewerDockWidget
    # whose _on_visibility_changed rebuilds its own title bar on *every* show,
    # which destroys the nvitk one. The shared helper builds a plain QDockWidget,
    # installs the pop-out/fullscreen controls itself, and can be tabbed with
    # Napari's layer-controls dock by name.
    dock = attach_left_inspection_dock(
        viewer,
        panel,
        object_name=DOCK_OBJECT_NAME,
        title="Orthogonal views",
        tabify_with="layer controls",
        minimum_width=360,
        frame=False,
    )
    panel._nvitk_dock = dock
    if dock is not None:
        # Closing the views with the main canvas hidden would leave nothing to look
        # at: the canvas comes back. (The toggle action follows closing, not a tab
        # switch that merely puts another dock in front.)
        def _closed(open_: bool) -> None:
            if not open_ and not panel._main_canvas.canvas_shown():
                panel._main_canvas.set_canvas_shown(True)

        dock.toggleViewAction().toggled.connect(_closed)
    connect_ortho_panel(viewer, panel)
    return dock


def _qt_deleted(obj: Any) -> bool:
    try:
        from qtpy import sip  # type: ignore[attr-defined]

        return bool(sip.isdeleted(obj))
    except Exception:  # noqa: BLE001
        return False


def connect_ortho_panel(viewer: Any, panel: OrthoViewerPanel) -> None:
    """Keep *panel* on the active layer and in step with the layer list (any host)."""

    def _refresh(_event: Any = None) -> None:
        """Rebind the panel to whichever layer is active.

        The panel's own plane layers are skipped: they are a *view* of the bound
        volume, so following the selection onto one would rebind the panel to its
        own output and reset the crosshair.
        """
        active = viewer.layers.selection.active if viewer.layers else None
        name = str(getattr(active, "name", "")) if active is not None else ""
        if _PLANE_LAYER_SUFFIX in name or _BOX_LAYER_SUFFIX in name:
            return
        # An empty selection is not a reason to unbind. Napari clears it
        # transiently while a layer is being added, and the panel's own 3D cuts
        # are layers, so rebinding here threw away the crosshair and the plane
        # rotation every time those were drawn.
        if active is None and panel.bound_layer() is not None:
            return
        # Nor is selecting something that is neither a volume nor a 2D image:
        # shapes, points, a mesh, a colour capture: the volume already on screen is
        # what the user still wants to look at. A 2D image (a plain 2D layer, a
        # localizer, a single tracker slice) is shown in the first view.
        bound = panel.bound_layer()
        if (
            active is not None
            and not is_volume_layer(active)
            and not is_flat_layer(active)
            and bound is not None
            and bound in viewer.layers
        ):
            return
        # A layer in Transform mode is about to be moved by hand: the views stay on (or move
        # to) another grid, or the first drag would re-lay them under the cursor.
        from nvitk.gui.viz.layer_transform import layer_mode, transform_targets

        if active is not None and layer_mode(active) == "transform":
            if bound is not None and bound in viewer.layers and bound is not active:
                return
            panel.pin_reference(transform_targets(viewer) or [active])
            if panel.bound_layer() is not active:
                return
        # Napari emits `active` twice for a single click, with the same layer
        # both times. Re-binding is not free (it re-materialises the volume,
        # rebuilds every render source and forces a redraw), so the second one
        # is pure waste. Data and layer-list changes arrive on their own events.
        if active is not None and active is panel.bound_layer():
            return
        panel.refresh_from_layer(active)

    def _layers_changed(_event: Any = None) -> None:
        """Keep the overlay picker in step with the viewer's layer list."""
        panel.refresh_overlay_choices()

    def _pick_in_3d(viewer_: Any, event: Any) -> Any:
        """A click (no drag) on the 3D canvas moves the crosshair to the point clicked: with
        Ctrl always, without it while "Click in 3D moves the crosshair" is on, when a drag on
        the crosshair's centre slides it too."""
        if viewer_.dims.ndisplay != 3 or int(getattr(event, "button", 1) or 1) != 1:
            return
        if _qt_deleted(panel) or panel.bound_layer() is None:
            return
        mods = set(getattr(event, "modifiers", ()) or ())
        interactive = panel.cross3d_interactive() and not mods
        if "Control" not in mods and not interactive:
            return
        if getattr(viewer_, "_nvitk_drag_owner", None):
            return  # the press grabbed something else (an edge of the ROI box)
        active = viewer_.layers.selection.active
        mode = str(getattr(getattr(active, "mode", "pan_zoom"), "value", getattr(active, "mode", "pan_zoom")))
        if interactive and mode not in ("pan_zoom", "transform", "select"):
            return  # a layer tool (painting in 3D...) has the click
        start = np.asarray(event.position, dtype=float)
        direction, displayed = event.view_direction, list(event.dims_displayed)
        grabbed = None
        if interactive:
            try:
                grabbed = panel.cross3d_grab(start, direction)
            except Exception:  # noqa: BLE001
                grabbed = None
        held = None
        if grabbed is not None:
            held = bool(viewer_.camera.mouse_pan)
            viewer_.camera.mouse_pan = False
            viewer_._nvitk_drag_owner = "crosshair"
        try:
            yield
            moved = False
            while event.type == "mouse_move":
                moved = moved or float(np.abs(np.asarray(event.position, dtype=float) - start).max()) > 1e-6
                if grabbed is not None and moved:
                    panel.cross3d_drag_to(event.position, event.view_direction, grabbed)
                yield
        finally:
            if held is not None:
                viewer_.camera.mouse_pan = held
                viewer_._nvitk_drag_owner = None
        if moved or direction is None:
            return  # a drag turned the camera (or slid the crosshair)
        voxel = panel.pick_3d_click(start, direction, displayed) if interactive else panel.pick_3d(start, direction, displayed)
        if voxel is not None:
            panel.set_position(voxel)

    # No ``destroyed`` hook to take it out again: a Python slot on ``destroyed`` runs during
    # PyQt's exit cleanup and crashed the interpreter on exit. The callback checks the panel.
    viewer.mouse_drag_callbacks.append(_pick_in_3d)
    panel._pick_callback = _pick_in_3d
    # Transform mode drags layers on the 3D canvas too (first: it claims the drag).
    from nvitk.gui.viz.layer_transform import install_transform_3d

    install_transform_3d(viewer)

    viewer.layers.selection.events.active.connect(_refresh)

    def _on_mode(event: Any = None) -> None:
        """The active layer switched to Transform mode: move the views off its grid now."""
        layer = getattr(event, "source", None)
        if _qt_deleted(panel) or layer is None or layer is not viewer.layers.selection.active:
            return
        from nvitk.gui.viz.layer_transform import layer_mode, transform_targets

        if layer_mode(layer) == "transform":
            panel.pin_reference(transform_targets(viewer) or [layer])

    def _watch_mode(layer: Any) -> None:
        emitter = getattr(getattr(layer, "events", None), "mode", None)
        if emitter is not None and not getattr(layer, "_nvitk_mode_watched", False):
            try:
                emitter.connect(_on_mode)
                layer._nvitk_mode_watched = True
            except Exception:  # noqa: BLE001
                pass

    for existing in list(viewer.layers):
        _watch_mode(existing)
    viewer.layers.events.inserted.connect(lambda event: _watch_mode(getattr(event, "value", None)))
    for name in ("inserted", "removed", "reordered", "moved"):
        emitter = getattr(viewer.layers.events, name, None)
        if emitter is not None:
            emitter.connect(_layers_changed)
    # A rename re-keys the 3D cut layers, which are named after their source.
    renamed = getattr(viewer.layers.events, "renamed", None)
    if renamed is not None:
        renamed.connect(_layers_changed)
    _refresh()
