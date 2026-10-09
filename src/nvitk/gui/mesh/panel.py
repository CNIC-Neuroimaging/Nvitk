"""The Meshlab panel (dock): MeshLab / ParaView basics for surfaces and point clouds, over time too.

Top to bottom: the active layer (size, area, volume) with open / save; the
operation pickers, the selected operation's parameters and Run; how the layer is
drawn (Display: colour by a field, colormap, solid colour, opacity, wireframe,
point size…); moving it by hand (Move: live, then Apply or Reset); a frame
slider for mesh time series; and a plot for profiles and curves over time.

Operations that need a point — a cross-section, a cut, an edit — do not read
the cursor: Run arms them and they wait for a click on the surface (in 3D, the
click is cast into the scene and lands on the front-most face; dragging still
rotates). A marker shows where the click landed; edits can be undone. The
selection tools paint with a brush or drag a box instead, until Done.

Display and Move act on every selected surface, points and series layer at
once (the active one leads); Undo puts back the last edit, whichever layers it
touched.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_ACCENT_DEEP,
    COLOR_BORDER,
    COLOR_BUTTON_OFF,
    COLOR_DISABLED,
    COLOR_MUTED,
    COLOR_ON_ACCENT,
    SPACE,
    SPACE_TIGHT,
    Card,
    clear_layout,
    style_figure,
)
from nvitk.gui.mesh.layers import (
    BUILTIN_FIELDS,
    COLORMAPS,
    add_mesh_layer,
    add_mesh_series_layer,
    add_point_cloud_layer,
    apply_display,
    display_of,
    field_names,
    fields_of,
    layer_kind,
    layer_to_mesh,
    layer_to_point_cloud,
    replace_mesh_layer,
    series_controller,
    set_display,
    set_parts,
    viewer_time_dim,
)
from nvitk.gui.mesh.operations import (
    ACTIVE_LAYER,
    CATEGORIES,
    MESHLAB_PREFIX,
    SURFACE_OUTPUTS,
    OpContext,
    accepts,
    categories,
    input_hint,
    label_info,
    operation,
    operations_for,
    present_label_ids,
    set_points,
)

_LAYER_NONE = "(none)"
#: Above this many faces the summary skips the topology checks (edge sorting).
_SUMMARY_TOPOLOGY_LIMIT = 2_000_000
#: Edits kept for Undo, per layer.
_UNDO_DEPTH = 20
#: Name of the marker layer showing where a pick landed.
_MARKER_NAME = "picked point"
#: Categories whose outputs are by-products (sections, centerlines, tables): the
#: input layer stays selected so the next measurement can follow.
_KEEP_SELECTION = ("Measure", "Vessels & tubes", "Compare", "MeshLab · Measures & distances")
#: Layer kinds Display and Move work on.
_SHAPE_KINDS = ("mesh", "series", "points")

_MESH_FILTER = (
    "Meshes and point clouds (*.stl *.obj *.off *.ply *.vtk *.vtp *.vtu *.gii *.xyz *.pts *.csv *.txt *.pcd *.pvd);;"
    "All files (*)"
)

#: Cards strip their children's borders (``QWidget { border: none }``); buttons in
#: them need their own, or they read as plain text.
_BUTTON_STYLE = (
    f"QPushButton {{ border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 4px 8px; }}"
    f"QPushButton:disabled {{ color: {COLOR_DISABLED}; }}"
    f"QPushButton:checked {{ background-color: {COLOR_ACCENT_DEEP}; color: {COLOR_ON_ACCENT}; }}"
)
_RUN_STYLE = (
    f"QPushButton {{ background-color: {COLOR_ACCENT_DEEP}; color: {COLOR_ON_ACCENT}; border-radius: 4px;"
    " padding: 6px; font-weight: bold; }"
    f"QPushButton:disabled {{ background-color: {COLOR_BUTTON_OFF}; color: {COLOR_DISABLED}; }}"
)
_INPUT_STYLE = f"border: 1px solid {COLOR_BORDER}; border-radius: 3px;"


def _notify(message: str, *, error: bool = False) -> None:
    from nvitk.gui.tools.runner import notify

    notify(message, error=error)


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {COLOR_MUTED}; font-weight: normal;")
    return label


def _button(text: str, tip: str = "", *, checkable: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(_BUTTON_STYLE)
    b.setCheckable(checkable)
    if tip:
        b.setToolTip(tip)
    return b


def _swatch(rgba: Any, size: int = 12) -> QIcon:
    pix = QPixmap(size, size)
    c = np.clip(np.asarray(rgba, dtype=float).ravel(), 0, 1)
    pix.fill(QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2]), 1.0))
    return QIcon(pix)


def ray_hit(origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray) -> np.ndarray | None:
    """Front-most intersection of a line with triangles (Möller–Trumbore, vectorised).

    The line is infinite: of all its hits, the one furthest *against* *direction*
    (nearest the camera, for Napari's view direction) is returned.
    """
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    s = np.asarray(origin, dtype=float) - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    if not hit.any():
        return None
    return np.asarray(origin, dtype=float) + float(t[hit].min()) * d


class LabelChecklist(QWidget):
    """The labels of an image / labels layer, each with its colour, name and a checkbox."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(4)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter labels…")
        self._filter.setStyleSheet(_INPUT_STYLE)
        self._filter.textChanged.connect(self._apply_filter)
        all_b = _button("All")
        none_b = _button("None")
        all_b.clicked.connect(lambda: self._set_all(True))
        none_b.clicked.connect(lambda: self._set_all(False))
        row.addWidget(self._filter, 1)
        row.addWidget(all_b)
        row.addWidget(none_b)
        lay.addLayout(row)
        self._list = QListWidget()
        self._list.setMinimumHeight(90)
        self._list.setMaximumHeight(170)
        self._list.setStyleSheet(_INPUT_STYLE)
        lay.addWidget(self._list)
        self._count = _muted()
        lay.addWidget(self._count)
        self._list.itemChanged.connect(lambda _i: self._update_count())

    def set_layer(self, layer: Any, checked: list[int] | None = None) -> None:
        """List *layer*'s labels; *checked* (default: all) start ticked."""
        from nvitk.gui.labels.visibility import is_label_like_layer

        self._list.blockSignals(True)
        self._list.clear()
        ids: list[int] = []
        if layer is not None and layer_kind(layer) == "image":
            try:
                if type(layer).__name__ == "Labels" or is_label_like_layer(layer):
                    ids = present_label_ids(layer)
            except Exception:  # noqa: BLE001
                ids = []
        info = label_info(layer, ids) if ids else {}
        # Ticks remembered from another layer only count where they still exist.
        wanted = set(checked or ()) & set(ids) or set(ids)
        for lid in ids:
            name, rgba = info[lid]
            item = QListWidgetItem(_swatch(rgba), f"{name}  ({lid})")
            item.setData(Qt.UserRole, lid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if lid in wanted else Qt.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._apply_filter(self._filter.text())
        self._update_count()

    def label_name(self, lid: int) -> str:
        """The name shown for label *lid* (without the id)."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if int(item.data(Qt.UserRole)) == int(lid):
                return item.text().rsplit("  (", 1)[0]
        return f"label {lid}"

    def value(self) -> list[int]:
        return [int(self._list.item(i).data(Qt.UserRole)) for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]

    def set_value(self, ids: Any) -> None:
        wanted = {int(i) for i in (ids or [])}
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setCheckState(Qt.Checked if int(item.data(Qt.UserRole)) in wanted else Qt.Unchecked)

    def _set_all(self, on: bool) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _apply_filter(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _update_count(self) -> None:
        n = self._list.count()
        if not n:
            self._count.setText("No labels: the whole mask (non-zero voxels) is meshed.")
        else:
            self._count.setText(f"{len(self.value())} of {n} label(s) selected")


class MeshPanel(QWidget):
    """Mesh and point-cloud tools for the active layer."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._values: dict[str, dict[str, Any]] = {}
        self._widgets: dict[str, QWidget] = {}
        self._layer: Any | None = None
        self._form_op: Any | None = None
        self._form_defaults: dict[str, Any] = {}
        self._picking: Any | None = None
        self._pick_layer: Any | None = None
        self._marker_ref: Any = lambda: None
        #: Undo steps, oldest first; each lists ``(weakref to layer, its previous content)``.
        self._undo_steps: list[list[tuple[Any, Any]]] = []
        #: The live move: ``{"layers": [(layer, affine before)], "centre": pivot}``.
        self._move_state: dict[str, Any] | None = None
        self._display_guard = False
        #: Brush strokes reuse the picked layer's geometry until Done.
        self._stroke: dict[str, Any] | None = None
        self._saved_pan: bool | None = None
        self._rubber: Any = None

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT)
        root.setSpacing(SPACE)
        root.setAlignment(Qt.AlignTop)

        root.addWidget(self._build_info_card())
        root.addWidget(self._build_operation_card())
        self._display_card = self._build_display_card()
        root.addWidget(self._display_card)
        self._move_card = self._build_move_card()
        root.addWidget(self._move_card)
        self._time_card = self._build_time_card()
        root.addWidget(self._time_card)
        self._plot_card = Card("Plot")
        self._plot_canvas = None
        self._plot_card.setVisible(False)
        root.addWidget(self._plot_card)
        root.addStretch(1)

        self._category.currentTextChanged.connect(self._on_category)
        self._operation.currentIndexChanged.connect(lambda _i: self._on_operation())
        viewer.layers.selection.events.active.connect(self._on_active)
        viewer.layers.selection.events.changed.connect(lambda _e=None: self._on_selection_changed())
        viewer.layers.events.inserted.connect(lambda _e=None: self._refresh_layer_combos())
        viewer.layers.events.removed.connect(lambda _e=None: self._refresh_layer_combos())
        self._on_category(self._category.currentText())
        self._on_active()

    # ── cards ────────────────────────────────────────────────────────────────

    def _build_info_card(self) -> Card:
        info = Card("Active layer")
        self._summary = QLabel("—")
        self._summary.setWordWrap(True)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info.add(self._summary)
        files = QHBoxLayout()
        files.setSpacing(SPACE_TIGHT)
        self._btn_open = _button("Open…", "Open meshes (STL, OBJ, OFF, PLY, VTK/VTP, GIfTI) or point clouds "
                                 "(XYZ, CSV, PCD, PLY). A ParaView .pvd opens as a time series.")
        self._btn_open.clicked.connect(self._open_files)
        self._btn_series = _button("Open series…", "Several mesh files, one per time frame, as one time series.")
        self._btn_series.clicked.connect(self._open_series)
        self._btn_save = _button("Save…", "Save the active surface / points (a series: one file per frame + .pvd).")
        self._btn_save.clicked.connect(self._save)
        for b in (self._btn_open, self._btn_series, self._btn_save):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            files.addWidget(b)
        info.add_layout(files)
        return info

    def _build_operation_card(self) -> Card:
        ops = Card("Operation")
        pickers = QFormLayout()
        pickers.setContentsMargins(0, 0, 0, 0)
        pickers.setVerticalSpacing(SPACE_TIGHT)
        self._category = QComboBox()
        self._category.addItems(categories())
        if self._category.count() > len(CATEGORIES):
            # MeshLab's filters (PyMeshLab) follow the native tools.
            self._category.insertSeparator(len(CATEGORIES))
        self._category.setMaxVisibleItems(30)
        self._operation = QComboBox()
        self._operation.setMaxVisibleItems(30)
        for combo in (self._category, self._operation):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        pickers.addRow("Category", self._category)
        pickers.addRow("Operation", self._operation)
        ops.add_layout(pickers)
        self._description = _muted()
        ops.add(self._description)
        self._docs = QLabel()
        self._docs.setOpenExternalLinks(True)
        self._docs.setTextFormat(Qt.RichText)
        self._docs.setVisible(False)
        ops.add(self._docs)
        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setVerticalSpacing(SPACE_TIGHT)
        self._form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        ops.add(self._form_host)
        self._replace = QCheckBox("Replace the active layer (else add a new one)")
        ops.add(self._replace)
        run_row = QHBoxLayout()
        run_row.setSpacing(SPACE_TIGHT)
        self._run = QPushButton("Run")
        self._run.setStyleSheet(_RUN_STYLE)
        self._run.clicked.connect(self._on_run_clicked)
        self._undo_btn = _button("Undo edit", "Put back the surface (or points) as it was before the last edit.")
        self._undo_btn.clicked.connect(self.undo)
        self._undo_btn.setVisible(False)
        run_row.addWidget(self._run, 1)
        run_row.addWidget(self._undo_btn)
        ops.add_layout(run_row)
        self._status = _muted()
        ops.add(self._status)
        return ops

    def _build_display_card(self) -> Card:
        card = Card("Display")
        self._display_note = _muted()
        self._display_note.setVisible(False)
        card.add(self._display_note)
        grid = QFormLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(SPACE_TIGHT)
        grid.setRowWrapPolicy(QFormLayout.WrapLongRows)
        colour_row = QHBoxLayout()
        colour_row.setSpacing(4)
        self._colour_by = QComboBox()
        self._colour_by.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._colour_by.setToolTip("A solid colour, or a per-vertex field: curvature, distances, diameters, "
                                   "labels, image values, the coordinates…")
        self._colour_btn = _button("■", "Pick the solid colour.")
        self._colour_btn.setFixedWidth(34)
        colour_row.addWidget(self._colour_by, 1)
        colour_row.addWidget(self._colour_btn)
        grid.addRow("Colour by", colour_row)
        self._cmap = QComboBox()
        self._cmap.addItems(list(COLORMAPS))
        grid.addRow("Colormap", self._cmap)
        range_row = QHBoxLayout()
        range_row.setSpacing(4)
        self._auto = QCheckBox("auto")
        self._auto.setToolTip("2nd–98th percentile of the field (all frames, for a series).")
        self._lo = QDoubleSpinBox()
        self._hi = QDoubleSpinBox()
        for box in (self._lo, self._hi):
            box.setRange(-1e9, 1e9)
            box.setDecimals(4)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        range_row.addWidget(self._auto)
        range_row.addWidget(self._lo, 1)
        range_row.addWidget(self._hi, 1)
        grid.addRow("Range", range_row)
        self._opacity = QSlider(Qt.Horizontal)
        self._opacity.setRange(5, 100)
        grid.addRow("Opacity", self._opacity)
        self._shading = QComboBox()
        self._shading.addItems(["smooth", "flat", "none"])
        self._shading_label = QLabel("Shading")
        grid.addRow(self._shading_label, self._shading)
        # What a surface draws, each on its own: faces, edges, vertices.
        surf_box = QVBoxLayout()
        surf_box.setContentsMargins(0, 0, 0, 0)
        surf_box.setSpacing(2)
        surf_row = QHBoxLayout()
        self._faces = QCheckBox("Faces")
        self._faces.setToolTip("Draw the surface's faces. Off: only what else is ticked (edges, vertices).")
        self._wire = QCheckBox("Wireframe")
        self._wire.setToolTip("Draw the edges: dark over the faces, in the surface's colour without them.")
        self._vertices = QCheckBox("Points")
        self._vertices.setToolTip("Draw the vertices as dots, coloured like the surface.")
        self._normals = QCheckBox("Normals")
        for box in (self._faces, self._wire, self._vertices, self._normals):
            surf_row.addWidget(box)
        surf_row.addStretch(1)
        surf_box.addLayout(surf_row)
        point_row = QHBoxLayout()
        self._vertex_size = QDoubleSpinBox()
        self._vertex_size.setRange(1.0, 30.0)
        self._vertex_size.setDecimals(1)
        self._vertex_size.setSuffix(" px")
        self._vertex_size.setToolTip("Size of the vertex dots on screen.")
        self._vertex_size_label = QLabel("Point size")
        point_row.addWidget(self._vertex_size_label)
        point_row.addWidget(self._vertex_size, 1)
        surf_box.addLayout(point_row)
        self._surf_opts = QWidget()
        self._surf_opts.setLayout(surf_box)
        grid.addRow("Show", self._surf_opts)
        pts_row = QHBoxLayout()
        pts_row.setSpacing(4)
        self._size = QDoubleSpinBox()
        self._size.setRange(0.01, 1000.0)
        self._size.setDecimals(2)
        self._size.setSuffix(" mm")
        self._symbol = QComboBox()
        self._symbol.addItems(["disc", "ring", "square", "cross", "x", "diamond", "star", "triangle_up"])
        self._spherical = QCheckBox("3D shading")
        pts_row.addWidget(self._size, 1)
        pts_row.addWidget(self._symbol, 1)
        pts_row.addWidget(self._spherical)
        self._pts_opts = QWidget()
        self._pts_opts.setLayout(pts_row)
        self._pts_label = QLabel("Points")
        grid.addRow(self._pts_label, self._pts_opts)
        card.add_layout(grid)
        self._field_info = _muted()
        card.add(self._field_info)

        self._colour_by.currentIndexChanged.connect(lambda _i: self._on_display_changed())
        self._cmap.currentTextChanged.connect(lambda _t: self._on_display_changed())
        self._auto.toggled.connect(lambda _c: self._on_display_changed())
        self._lo.valueChanged.connect(lambda _v: self._on_display_changed(manual_range=True))
        self._hi.valueChanged.connect(lambda _v: self._on_display_changed(manual_range=True))
        self._colour_btn.clicked.connect(self._pick_colour)
        self._opacity.valueChanged.connect(lambda v: self._set_layer_attr("opacity", v / 100.0))
        self._shading.currentTextChanged.connect(lambda t: self._set_layer_attr("shading", t, kind="Surface"))
        self._faces.toggled.connect(lambda on: self._on_parts(faces=bool(on)))
        self._wire.toggled.connect(lambda on: self._on_parts(wireframe=bool(on)))
        self._vertices.toggled.connect(lambda on: self._on_parts(points=bool(on)))
        self._vertex_size.valueChanged.connect(lambda v: self._on_parts(point_size=float(v)))
        self._normals.toggled.connect(self._on_normals)
        self._size.valueChanged.connect(lambda v: self._set_layer_attr("size", float(v), kind="Points"))
        self._symbol.currentTextChanged.connect(lambda t: self._set_layer_attr("symbol", t, kind="Points"))
        self._spherical.toggled.connect(
            lambda c: self._set_layer_attr("shading", "spherical" if c else "none", kind="Points"))
        card.setVisible(False)
        return card

    def _build_move_card(self) -> Card:
        card = Card("Move by hand")
        hint = _muted("Live preview; Apply bakes it into the vertices, Reset puts it back. "
                      "Rotation and scale are about the centre of everything selected.")
        card.add(hint)
        self._move_note = _muted()
        self._move_note.setVisible(False)
        card.add(self._move_note)
        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        self._move_boxes: dict[str, QDoubleSpinBox] = {}
        for row, (label, keys, rng, step, suffix) in enumerate((
            ("Move", ("tx", "ty", "tz"), 1e5, 0.5, " mm"),
            ("Turn", ("rx", "ry", "rz"), 360.0, 2.0, "°"),
        )):
            grid.addWidget(QLabel(label), row, 0)
            for col, key in enumerate(keys, 1):
                box = QDoubleSpinBox()
                box.setRange(-rng, rng)
                box.setSingleStep(step)
                box.setDecimals(2)
                box.setSuffix(suffix)
                box.setToolTip({"tx": "x (R-L)", "ty": "y (A-P)", "tz": "z (S-I)",
                                "rx": "about x", "ry": "about y", "rz": "about z"}[key])
                box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                box.valueChanged.connect(lambda _v: self._preview_move())
                grid.addWidget(box, row, col)
                self._move_boxes[key] = box
        grid.addWidget(QLabel("Scale"), 2, 0)
        scale = QDoubleSpinBox()
        scale.setRange(1.0, 1000.0)
        scale.setValue(100.0)
        scale.setSingleStep(1.0)
        scale.setSuffix(" %")
        scale.valueChanged.connect(lambda _v: self._preview_move())
        grid.addWidget(scale, 2, 1)
        self._move_boxes["s"] = scale
        card.add_layout(grid)
        row = QHBoxLayout()
        self._move_apply = _button("Apply", "Bake the move into the vertices / points.")
        self._move_reset = _button("Reset", "Back to where it was.")
        self._move_apply.clicked.connect(self.apply_move)
        self._move_reset.clicked.connect(self.reset_move)
        row.addWidget(self._move_apply)
        row.addWidget(self._move_reset)
        row.addStretch(1)
        card.add_layout(row)
        card.setVisible(False)
        return card

    def _build_time_card(self) -> Card:
        card = Card("Time series")
        row = QHBoxLayout()
        row.setSpacing(SPACE_TIGHT)
        self._play = _button("▶", checkable=True)
        self._play.setFixedWidth(36)
        self._play.toggled.connect(self._on_play)
        self._frame = QSlider(Qt.Horizontal)
        self._frame.valueChanged.connect(self._on_frame_slider)
        self._frame_label = QLabel("frame 1 / 1")
        self._fps = QSpinBox()
        self._fps.setRange(1, 60)
        self._fps.setValue(8)
        self._fps.setSuffix(" fps")
        row.addWidget(self._play)
        row.addWidget(self._frame, 1)
        row.addWidget(self._frame_label)
        row.addWidget(self._fps)
        card.add_layout(row)
        self._time_hint = _muted("Follows the viewer's time slider while a 3D+t image is open; otherwise this "
                                 "slider plays it.")
        card.add(self._time_hint)
        card.setVisible(False)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        return card

    # ── pickers and form ─────────────────────────────────────────────────────

    def _current_op(self) -> Any:
        return operation(str(self._operation.currentData() or ""))

    def _on_category(self, category: str) -> None:
        self._operation.blockSignals(True)
        self._operation.clear()
        for op in operations_for(category):
            self._operation.addItem(op.label, op.id)
        self._operation.blockSignals(False)
        self._on_operation()

    def show_operation(self, op_id: str) -> None:
        """Bring the Meshlab dock to the front with *op_id* selected (the search bar's entry point)."""
        from qtpy.QtWidgets import QDockWidget

        self.select(op_id)
        widget: Any = self
        while widget is not None and not isinstance(widget, QDockWidget):
            widget = widget.parentWidget()
        if widget is not None:
            widget.show()
            widget.raise_()
        op = self._current_op()
        if op is not None:
            hint = "Run, then click on the surface." if op.pick else "Set the parameters, then Run."
            self._status.setText(f"{op.label}: {hint}")

    def select(self, op_id: str) -> None:
        """Point the pickers at *op_id*."""
        op = operation(op_id)
        if op is None:
            raise KeyError(op_id)
        self._category.setCurrentText(op.category)
        idx = self._operation.findData(op_id)
        if idx >= 0:
            self._operation.setCurrentIndex(idx)

    def _remember(self) -> None:
        """Keep what the user changed in the form (defaults are recomputed each time)."""
        op = self._form_op
        if op is not None:
            self._values[op.id] = {k: v for k, v in self.params().items() if v != self._form_defaults.get(k)}

    def _meshlab_defaults(self, op: Any) -> dict[str, Any]:
        """MeshLab's defaults for *op* on the active layer (sizes and counts follow the mesh)."""
        layer = self._viewer.layers.selection.active
        if not op.id.startswith(MESHLAB_PREFIX) or not op.inputs or layer is None or not accepts(op, layer):
            return {}
        try:
            from nvitk.meshlab.pymeshlab_filters import meshlab_defaults

            obj = layer_to_mesh(layer) if type(layer).__name__ == "Surface" else layer_to_point_cloud(layer)
            return meshlab_defaults(obj, op.id[len(MESHLAB_PREFIX):])
        except Exception:  # noqa: BLE001 — the catalog's defaults are a fine fallback
            return {}

    def _on_operation(self) -> None:
        self._remember()
        self._stop_pick()
        op = self._current_op()
        self._form_op = op
        clear_layout(self._form)
        self._widgets = {}
        if op is None:
            return
        pick = ""
        if op.pick == "surface" and "click" not in op.description.lower():
            pick = "\nRun, then click on the surface in the viewer (dragging still rotates)."
        self._description.setText(f"{op.description}\nInput: {input_hint(op)}.{pick}")
        if op.url:
            self._docs.setText(f'<a href="{op.url}" style="color: {COLOR_ACCENT};">'
                               "MeshLab documentation of this filter ↗</a>")
        self._docs.setVisible(bool(op.url))
        stored = self._values.get(op.id, {})
        defaults = self._meshlab_defaults(op)
        self._groups_row: tuple[QWidget, QWidget] | None = None
        for spec in op.params:
            widget = self._make_widget(spec, defaults.get(spec.name, spec.default))
            if spec.kind in ("bool", "labels"):
                if spec.kind == "labels":
                    self._form.addRow(QLabel(spec.label))
                self._form.addRow(widget)
            else:
                label = QLabel(spec.label)
                label.setWordWrap(True)
                field = self._groups_field(widget) if spec.name == "label_groups" else widget
                self._form.addRow(label, field)
                if spec.name == "label_groups":
                    self._groups_row = (label, field)
            self._widgets[spec.name] = widget
        # Read back through the widgets (rounding included), then restore the user's own values.
        self._form_defaults = self.params()
        for name, value in stored.items():
            if name in self._widgets:
                try:
                    self.set_param(name, value)
                except Exception:  # noqa: BLE001 — a value that no longer fits is dropped
                    pass
        output = self._widgets.get("surface_output")
        if isinstance(output, QComboBox) and self._groups_row is not None:
            output.currentTextChanged.connect(lambda _t: self._sync_groups_row())
            self._sync_groups_row()
        self._replace.setVisible(op.replaceable)
        self._run.setText(self._run_label(op))
        self._update_run_state()

    def _groups_field(self, edit: QWidget) -> QWidget:
        """The groups text plus "Add ticked labels as a group"."""
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        if isinstance(edit, QLineEdit):
            edit.setPlaceholderText("Left: 1,2 ; Right: 3-5")
        lay.addWidget(edit)
        add = _button("Add ticked labels as a group",
                      "Tick the labels of one group above, press this (and name it), then tick the next group.")
        add.clicked.connect(lambda: self.add_group())
        lay.addWidget(add)
        return box

    def _sync_groups_row(self) -> None:
        output = self._widgets.get("surface_output")
        if self._groups_row is None or not isinstance(output, QComboBox):
            return
        show = output.currentText() == SURFACE_OUTPUTS[3]
        for w in self._groups_row:
            w.setVisible(show)

    def add_group(self, name: str | None = None) -> str:
        """Append the ticked labels as one named group (then untick them for the next one)."""
        checklist = next((w for w in self._widgets.values() if isinstance(w, LabelChecklist)), None)
        edit = self._widgets.get("label_groups")
        if checklist is None or not isinstance(edit, QLineEdit):
            return ""
        ids = checklist.value()
        if not ids:
            self._status.setText("Tick the labels of the group first.")
            return edit.text()
        existing = [c for c in edit.text().split(";") if c.strip()]
        if name is None:
            default = checklist.label_name(ids[0]) if len(ids) == 1 else f"group {len(existing) + 1}"
            from qtpy.QtWidgets import QInputDialog

            name, ok = QInputDialog.getText(self, "Group name", "Name of the group:", text=default)
            if not ok:
                return edit.text()
        name = str(name).replace(":", " ").replace(";", " ").strip() or f"group {len(existing) + 1}"
        existing.append(f"{name}: {','.join(str(i) for i in ids)}")
        edit.setText(" ; ".join(c.strip() for c in existing))
        checklist.set_value([])
        self._status.setText(f"Group '{name}' = labels {', '.join(str(i) for i in ids)}. Tick the next group, "
                             "or Run.")
        return edit.text()

    def _make_widget(self, spec: Any, value: Any) -> QWidget:
        kind = spec.kind
        if kind == "labels":
            w = LabelChecklist()
            w.set_layer(self._viewer.layers.selection.active, value)
            return w
        if kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.min if spec.min is not None else -2**31 + 1),
                       int(spec.max if spec.max is not None else 2**31 - 1))
            w.setValue(int(value if value is not None else 0))
        elif kind == "float":
            w = QDoubleSpinBox()
            default = abs(float(spec.default or 0.0))
            w.setDecimals(6 if 0 < default < 0.01 else 3)
            w.setRange(float(spec.min if spec.min is not None else -1e12), float(spec.max if spec.max is not None else 1e12))
            w.setSingleStep(0.1 if default < 10 else 1.0)
            w.setValue(float(value if value is not None else 0.0))
        elif kind == "bool":
            w = QCheckBox(spec.label)
            w.setChecked(bool(value))
        elif kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in spec.choices])
            w.setCurrentIndex(max(w.findText(str(value)), 0))
        elif kind == "layer":
            w = QComboBox()
            # MeshLab filters on two meshes: either one may be the active layer.
            w.setProperty("nvitk_active_option", spec.default == ACTIVE_LAYER)
            self._fill_layer_combo(w, str(value or ""))
        else:
            w = QLineEdit(str(value if value is not None else ""))
            w.setStyleSheet(_INPUT_STYLE + " padding: 2px 4px;")
            if "(% or mm)" in spec.label:
                w.setToolTip("A length: “1%” is 1% of the bounding-box diagonal; a plain number is in mm.")
        if kind != "bool":
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            w.setMinimumWidth(60)
        if isinstance(w, QComboBox):
            w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(8)
        return w

    def _fill_layer_combo(self, combo: QComboBox, current: str) -> None:
        active = self._viewer.layers.selection.active
        names = [ly.name for ly in self._viewer.layers if ly is not active and ly.name != _MARKER_NAME]
        combo.blockSignals(True)
        combo.clear()
        lead = [ACTIVE_LAYER, _LAYER_NONE] if combo.property("nvitk_active_option") else [_LAYER_NONE]
        combo.addItems([*lead, *names])
        idx = combo.findText(current)
        combo.setCurrentIndex(max(idx, 0))
        combo.blockSignals(False)

    def _refresh_layer_combos(self) -> None:
        op = self._current_op()
        if op is None:
            return
        for spec in op.params:
            w = self._widgets.get(spec.name)
            if spec.kind == "layer" and isinstance(w, QComboBox):
                self._fill_layer_combo(w, w.currentText())

    def params(self) -> dict[str, Any]:
        """The current form's values."""
        out: dict[str, Any] = {}
        for name, w in self._widgets.items():
            if isinstance(w, LabelChecklist):
                out[name] = w.value()
            elif isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[name] = w.value()
            elif isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, QLineEdit):
                out[name] = w.text()
        return out

    def set_param(self, name: str, value: Any) -> None:
        """Set one parameter of the current operation (scripting, tests)."""
        w = self._widgets[name]
        if isinstance(w, LabelChecklist):
            w.set_value(value)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(value)
        elif isinstance(w, QComboBox):
            if w.findText(str(value)) < 0:
                self._refresh_layer_combos()
            w.setCurrentText(str(value))
        elif isinstance(w, QLineEdit):
            w.setText(str(value))

    # ── active layer ─────────────────────────────────────────────────────────

    def _on_active(self, _event: Any = None) -> None:
        layer = self._viewer.layers.selection.active
        if layer is not None and layer.name == _MARKER_NAME:
            return
        changed = layer is not self._layer
        self._layer = layer
        n_selected = len(self._viewer.layers.selection)
        if layer is None and n_selected > 1:
            self._summary.setText(f"<b>{n_selected} layers selected</b><br>Display and Move apply to all of them; "
                                  "operations run on one layer — select it alone.")
        else:
            self._summary.setText(self._describe(layer))
        self._refresh_layer_combos()
        if changed:
            for w in self._widgets.values():
                if isinstance(w, LabelChecklist):
                    w.set_layer(layer, w.value())
            op = self._current_op()
            if op is not None and op.id.startswith(MESHLAB_PREFIX) and op.inputs and self._picking is None:
                self._on_operation()  # MeshLab's defaults depend on the mesh
        self._sync_time_card()
        self._sync_targets()
        self._sync_undo_button()
        self._update_run_state()

    def _lead_layer(self) -> Any | None:
        """The active layer, or with several selected (Napari then has no active
        one) the one clicked last."""
        selection = self._viewer.layers.selection
        if selection.active is not None:
            return selection.active
        current = getattr(selection, "_current", None)
        return current if current is not None and current in selection else None

    def _targets(self) -> list[Any]:
        """The selected surfaces, points and series, the leading layer first: what
        Display and Move act on."""
        selection = self._viewer.layers.selection
        out = [ly for ly in self._viewer.layers
               if ly in selection and ly.name != _MARKER_NAME and layer_kind(ly) in _SHAPE_KINDS]
        lead = self._lead_layer()
        if lead is not None and lead in out:
            out.remove(lead)
            out.insert(0, lead)
        return out

    def _on_selection_changed(self) -> None:
        selection = self._viewer.layers.selection
        if selection.active is None and len(selection) > 1:
            self._summary.setText(f"<b>{len(selection)} layers selected</b><br>Display and Move apply to all of "
                                  "them; an operation runs on each, as the labels of one multilabel mesh "
                                  "(one table, one undo).")
        self._sync_targets()
        self._update_run_state()

    def _run_targets(self, op: Any) -> list[Any]:
        """The selected layers *op* runs on, the leading one first.

        Several selected surfaces (or clouds, or images) are run one by one, each
        like a label of one multilabel mesh. A layer an operation's own layer
        parameter names (a reference to compare against) is an input, not a target.
        """
        if op is None or not op.inputs:
            return []
        selection = self._viewer.layers.selection
        lead = self._lead_layer()
        picked = [ly for ly in self._viewer.layers if ly in selection and ly.name != _MARKER_NAME and accepts(op, ly)]
        if not picked and lead is not None and accepts(op, lead):
            picked = [lead]
        referenced = {str(v) for k, v in self.params().items()
                      if any(spec.name == k and spec.kind == "layer" for spec in op.params)}
        targets = [ly for ly in picked if ly.name not in referenced] or picked
        if lead is not None and lead in targets:
            targets.remove(lead)
            targets.insert(0, lead)
        return targets

    def _sync_targets(self) -> None:
        """Refresh Display and Move for the current multi-selection."""
        targets = self._targets()
        if self._move_state is not None and [ly for ly, _b in self._move_state["layers"]] != targets:
            self.reset_move()
        self._sync_display_card()
        self._move_card.setVisible(bool(targets))
        many = len(targets) > 1
        for note in (self._display_note, self._move_note):
            note.setText(f"Applies to the {len(targets)} selected layers ({', '.join(t.name for t in targets[:4])}"
                         f"{'…' if len(targets) > 4 else ''})." if many else "")
            note.setVisible(many)

    def _update_run_state(self) -> None:
        if self._picking is not None:
            return
        op = self._current_op()
        targets = self._run_targets(op)
        many = len(targets) > 1
        ok = op is not None and (not op.inputs or bool(targets)) and not (many and op.pick)
        self._run.setEnabled(ok)
        self._run.setText(self._run_label(op) if not (ok and many) else f"Run on {len(targets)} layers")
        if op is not None and many and op.pick:
            self._status.setText("Picking works on one layer: select it alone (several are selected).")
        elif op is not None and not ok:
            self._status.setText(f"Select {input_hint(op)} to run this.")
        elif ok and self._status.text().startswith(("Select ", "Picking works")):
            self._status.setText("")

    def _describe(self, layer: Any) -> str:
        """One-paragraph summary of the active layer."""
        kind = layer_kind(layer)
        if not kind:
            return "Select a surface, points, image or labels layer."
        name = f"<b>{layer.name}</b>"
        try:
            if kind in ("mesh", "series"):
                mesh = layer_to_mesh(layer)
                from nvitk.meshlab.measure import surface_area, volume
                from nvitk.meshlab.topology import is_watertight

                parts = [f"{mesh.n_vertices:,} vertices", f"{mesh.n_faces:,} faces",
                         f"area {surface_area(mesh):,.4g} mm²"]
                if mesh.n_faces <= _SUMMARY_TOPOLOGY_LIMIT:
                    closed = is_watertight(mesh)
                    parts.append("closed" if closed else "open")
                    if closed:
                        parts.append(f"volume {volume(mesh):,.4g} mm³")
                prefix = "Mesh"
                sel = fields_of(layer).get("selected")
                if sel is not None and np.any(np.asarray(sel) > 0.5):
                    parts.append(f"{int(np.sum(np.asarray(sel) > 0.5)):,} selected")
                ctrl = series_controller(layer)
                if ctrl is not None:
                    prefix = f"Mesh series · {len(ctrl.series)} frames · frame {ctrl.frame + 1}"
                return f"{prefix} {name}<br>" + " · ".join(parts)
            if kind == "points":
                cloud = layer_to_point_cloud(layer)
                extent = np.ptp(cloud.points, axis=0) if cloud.n_points else np.zeros(3)
                return (f"Points {name}<br>{cloud.n_points:,} points · extent "
                        f"{' × '.join(f'{v:.4g}' for v in extent)} mm")
            shape = " × ".join(str(s) for s in np.shape(layer.data))
            n_lab = len(present_label_ids(layer))
            labs = f" · {n_lab} label(s)" if n_lab else ""
            return f"{type(layer).__name__} {name}<br>{shape} voxels{labs} — Create → Surfaces from labels."
        except Exception as exc:  # noqa: BLE001 — a summary must never break the panel
            return f"{name}: {exc}"

    # ── running ──────────────────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        if self._picking is not None:
            painting = self._picking.pick in ("brush", "box")
            layer = self._pick_layer
            self._stop_pick()
            if painting and layer is not None and layer in self._viewer.layers:
                from nvitk.gui.mesh.selection import selection_of

                self._status.setText(f"{int(selection_of(layer).sum()):,} selected on {layer.name}.")
                self._summary.setText(self._describe(layer))
            else:
                self._status.setText("Picking cancelled.")
            return
        op = self._current_op()
        if op is not None and op.pick:
            self.start_pick()
            return
        self.run_current()

    def run_current(self, *, point: np.ndarray | None = None, view_direction: np.ndarray | None = None,
                    layer: Any | None = None) -> Any:
        """Run the selected operation on *layer* (default: the active one), at *point* for picking ones."""
        from nvitk.gui.core.log_panel import gui_log

        op = self._current_op()
        if layer is None and op is not None and point is None:
            targets = self._run_targets(op)
            if len(targets) > 1 and not op.pick:
                return self._run_many(op, targets)
            if targets:
                layer = targets[0]
        layer = layer if layer is not None else self._viewer.layers.selection.active
        if op is None or (layer is None and op.inputs):
            return None
        if not accepts(op, layer):
            _notify(f"{op.label} needs {input_hint(op)}.", error=True)
            return None
        params = self._run_params(op)
        ctx = OpContext(self._viewer, layer, params, replace=self._replace.isChecked() and op.replaceable,
                        point=point, view_direction=view_direction, remember=self._push_undo)
        self._status.setText("Running…")
        self.repaint()
        try:
            result = op.run(ctx)
        except Exception as exc:  # noqa: BLE001 — report, never crash the panel
            from nvitk.gui.tools.runner import log_tool_failure

            log_tool_failure(exc)
            self._status.setText(f"Failed: {exc}")
            _notify(f"{op.label} failed: {exc}", error=True)
            return None
        self._status.setText(result.message)
        _notify(result.message)
        if result.table:
            try:
                from nvitk.gui.viz.results_window import show_results

                gui_log(show_results(self._viewer, result.title or op.label, result.table,
                                     row_header=result.row_header, on_select=result.on_row))
            except Exception:  # noqa: BLE001
                gui_log(str(result.table))
        if result.plot:
            self._show_plot(result.plot, result.title or op.label)
        # Auxiliary outputs (sections, centerlines, markers) should not take the
        # selection away from the layer being worked on.
        if (op.pick or op.category in _KEEP_SELECTION) and layer in self._viewer.layers:
            self._viewer.layers.selection.active = layer
        self._on_active()
        # A new layer passing through the selection may have swapped the status for a hint.
        self._status.setText(result.message)
        return result

    def _run_params(self, op: Any) -> dict[str, Any]:
        """The form's parameters for *op* (for a MeshLab filter, only those changed)."""
        params = self.params()
        if op.id.startswith(MESHLAB_PREFIX):
            # Only what the user changed: MeshLab computes the rest for each mesh.
            kinds = {spec.name: spec.kind for spec in op.params}
            params = {k: v for k, v in params.items() if kinds.get(k) == "layer" or v != self._form_defaults.get(k)}
        return params

    def _run_many(self, op: Any, targets: list[Any]) -> Any:
        """Run *op* on each of *targets*, as the labels of one multilabel mesh.

        One combined result (a table with a row — or a block of rows — per layer,
        every layer's curves on one plot, row clicks highlighting in the right
        layer), one undo step for every layer it changed, and the selection left
        as it was.
        """
        from nvitk.gui.core.log_panel import gui_log
        from nvitk.gui.mesh.operations import combine_results
        from nvitk.gui.tools.runner import log_tool_failure

        params = self._run_params(op)
        changed: list[tuple[Any, Any]] = []
        results: list[tuple[str, Any]] = []
        failures: list[str] = []
        self._status.setText(f"Running on {len(targets)} layers…")
        self.repaint()
        for k, layer in enumerate(targets, 1):
            if layer not in self._viewer.layers:
                continue
            self._status.setText(f"Running on {layer.name} ({k}/{len(targets)})…")
            QApplication.processEvents()
            ctx = OpContext(self._viewer, layer, dict(params), replace=self._replace.isChecked() and op.replaceable,
                            remember=lambda ly, prev: changed.append((weakref.ref(ly), prev)))
            try:
                result = op.run(ctx)
            except Exception as exc:  # noqa: BLE001 — one layer failing must not stop the rest
                log_tool_failure(exc)
                failures.append(f"{layer.name}: {exc}")
                continue
            results.append((layer.name, result))
            gui_log(f"{op.label} · {layer.name}: {result.message}")
        if changed:
            # One Undo puts every layer this run changed back.
            self._push_undo_step(changed)
        if not results:
            message = f"{op.label} failed on every layer: " + "; ".join(failures)
            self._status.setText(message)
            _notify(message, error=True)
            return None
        result = combine_results(op.label, results)
        summary = f"{op.label}: ran on {len(results)} of {len(targets)} layers."
        if failures:
            summary += " Failed: " + "; ".join(failures)
        _notify(summary, error=bool(failures))
        if result.table:
            try:
                from nvitk.gui.viz.results_window import show_results

                gui_log(show_results(self._viewer, result.title or op.label, result.table,
                                     row_header=result.row_header, on_select=result.on_row))
            except Exception:  # noqa: BLE001
                gui_log(str(result.table))
        if result.plot:
            self._show_plot(result.plot, result.title or op.label)
        # The layers worked on stay selected, as they were, whatever the run added.
        selection = self._viewer.layers.selection
        alive = [ly for ly in targets if ly in self._viewer.layers]
        if alive:
            selection.clear()
            selection.update(alive)
        self._sync_targets()
        self._update_run_state()
        self._status.setText(summary)
        return result

    # ── picking a point ──────────────────────────────────────────────────────

    def start_pick(self) -> None:
        """Arm the selected operation: it runs on the next click in the viewer (the
        selection tools: on every brush stroke or box, until Done)."""
        op = self._current_op()
        layer = self._viewer.layers.selection.active
        if op is None or layer is None or not accepts(op, layer):
            return
        self._picking = op
        self._pick_layer = layer
        self._stroke = None
        if self._pick_callback not in self._viewer.mouse_drag_callbacks:
            self._viewer.mouse_drag_callbacks.append(self._pick_callback)
        if op.pick in ("brush", "box"):
            # Drags select now; the brush hands drags that miss the surface back to the camera.
            self._saved_pan = bool(self._viewer.camera.mouse_pan)
            self._viewer.camera.mouse_pan = False
            self._run.setText("Done selecting")
            if op.pick == "brush":
                self._status.setText("Drag on the surface to paint the selection (red); drag off it to turn the "
                                     "camera; the wheel zooms. Done when finished.")
            else:
                self._status.setText("Drag a rectangle over the view. Turning the camera is paused (the wheel "
                                     "still zooms); Done when finished.")
            return
        self._run.setText("Cancel picking")
        where = "on the surface" if layer_kind(layer) != "points" else "on the points"
        self._status.setText(f"Click {where} in the viewer ({op.label}). Dragging still rotates.")

    def _stop_pick(self) -> None:
        self._picking = None
        self._stroke = None
        self._hide_rubber()
        try:
            self._viewer.mouse_drag_callbacks.remove(self._pick_callback)
        except ValueError:
            pass
        if self._saved_pan is not None:
            self._viewer.camera.mouse_pan = self._saved_pan
            self._saved_pan = None
        op = self._current_op()
        self._run.setText(self._run_label(op))
        self._update_run_state()

    @staticmethod
    def _run_label(op: Any) -> str:
        if op is None or not op.pick:
            return "Run"
        return "Run — then select in the viewer" if op.pick in ("brush", "box") else "Run — then click on the surface"

    def _pick_callback(self, viewer: Any, event: Any) -> Any:
        """Napari mouse callback: a click (no drag) picks; a drag stays a camera move.
        The selection tools take the drag instead: a brush stroke or a box."""
        if self._picking is None or getattr(event, "button", 1) != 1 or event.type != "mouse_press":
            return
        if self._picking.pick == "brush":
            yield from self._brush_drag(event)
            return
        if self._picking.pick == "box":
            yield from self._box_drag(event)
            return
        dragged = False
        yield
        while event.type == "mouse_move":
            dragged = True
            yield
        if dragged:
            return
        point, view_dir = self.resolve_pick(event.position, getattr(event, "view_direction", None))
        if point is None:
            self._status.setText("Missed the surface — click on it (or Cancel).")
            return
        self.pick_at(point, view_dir)

    # ── selecting with the brush and the box ─────────────────────────────────

    def _stroke_cache(self, *, fresh_selection: bool = False) -> dict[str, Any]:
        """The picked layer's geometry, kept for the length of the tool, and its
        selection — read again from the layer at the start of every stroke or box
        (*fresh_selection*): Clear, Invert, an undo or any other operation may have
        changed it while the tool stayed armed."""
        from scipy.spatial import cKDTree

        from nvitk.gui.mesh.selection import selection_of

        layer = self._pick_layer
        n = len(layer.vertices if type(layer).__name__ == "Surface" else layer.data) if layer is not None else 0
        if self._stroke is None or self._stroke["layer"] is not layer or len(self._stroke["points"]) != n:
            if type(layer).__name__ == "Surface":
                mesh = layer_to_mesh(layer)
                pts, tris = mesh.vertices, mesh.triangles
            else:
                pts, tris = layer_to_point_cloud(layer).points, None
            self._stroke = {"layer": layer, "points": pts, "triangles": tris, "tree": cKDTree(pts),
                            "selected": selection_of(layer)}
        elif fresh_selection:
            self._stroke["selected"] = selection_of(layer)
        return self._stroke

    def _brush_point(self, event: Any) -> np.ndarray | None:
        """Where the brush is: the front-most surface point under the mouse (3D), or the
        mouse on the slice when a vertex is within reach (2D)."""
        st = self._stroke_cache()
        pos = np.asarray(event.position, dtype=float)[-3:]
        if int(self._viewer.dims.ndisplay) == 3:
            vd = getattr(event, "view_direction", None)
            direction = np.asarray(vd if vd is not None else (0.0, 0.0, -1.0), dtype=float)[-3:]
            if st["triangles"] is not None:
                return ray_hit(pos, direction, st["triangles"])
            point, _d = self.resolve_pick(pos, direction)
            return point
        radius = float(self.params().get("sel_radius") or 1.0)
        dist, _i = st["tree"].query(pos)
        return pos if dist <= radius else None

    def _paint(self, centre: np.ndarray) -> int:
        from nvitk.gui.mesh.selection import set_selection

        st = self._stroke_cache()
        radius = float(self.params().get("sel_radius") or 1.0)
        idx = st["tree"].query_ball_point(np.asarray(centre, dtype=float), radius)
        if idx:
            st["selected"][np.asarray(idx, dtype=int)] = self.params().get("sel_mode", "add") != "remove"
            set_selection(st["layer"], st["selected"])
        return int(st["selected"].sum())

    def _brush_drag(self, event: Any) -> Any:
        self._stroke_cache(fresh_selection=True)
        hit = self._brush_point(event)
        if hit is None:
            # Off the surface: this drag turns the camera, as without the tool.
            self._viewer.camera.mouse_pan = True
            yield
            while event.type == "mouse_move":
                yield
            self._viewer.camera.mouse_pan = False
            return
        n = self._paint(hit)
        yield
        while event.type == "mouse_move":
            hit = self._brush_point(event)
            if hit is not None:
                n = self._paint(hit)
            yield
        self._status.setText(f"{n:,} selected on {self._pick_layer.name}. Paint more, or Done.")

    def _canvas_widget(self) -> Any:
        try:
            canvas = self._viewer.window._qt_viewer.canvas
        except Exception:  # noqa: BLE001
            return None
        return getattr(canvas, "native", None)

    def _show_rubber(self, p0: Any, p1: Any) -> None:
        from qtpy.QtCore import QPoint, QRect
        from qtpy.QtWidgets import QRubberBand

        host = self._canvas_widget()
        if host is None:
            return
        if self._rubber is None or self._rubber.parent() is not host:
            self._rubber = QRubberBand(QRubberBand.Rectangle, host)
        rect = QRect(QPoint(int(p0[0]), int(p0[1])), QPoint(int(p1[0]), int(p1[1]))).normalized()
        self._rubber.setGeometry(rect)
        self._rubber.show()

    def _hide_rubber(self) -> None:
        if self._rubber is not None:
            try:
                self._rubber.hide()
            except RuntimeError:
                self._rubber = None

    def _box_drag(self, event: Any) -> Any:
        from nvitk.gui.mesh.selection import combine, in_box, set_selection

        p0 = np.asarray(event.pos, dtype=float)[:2]
        self._show_rubber(p0, p0)
        yield
        while event.type == "mouse_move":
            self._show_rubber(p0, event.pos)
            yield
        self._hide_rubber()
        p1 = np.asarray(event.pos, dtype=float)[:2]
        if np.abs(p1 - p0).max() < 3:
            return  # a click, not a box
        layer = self._pick_layer
        params = self.params()
        self._stroke_cache(fresh_selection=True)
        picked = in_box(self._viewer, layer, p0, p1, facing_only=bool(params.get("sel_facing", True)),
                        view_direction=getattr(event, "view_direction", None))
        st = self._stroke_cache()
        st["selected"] = combine(st["selected"], picked, str(params.get("sel_mode") or "add"))
        n = set_selection(layer, st["selected"])
        self._status.setText(f"{n:,} selected on {layer.name} ({int(picked.sum()):,} in the box). "
                             "Drag another box, or Done.")

    def resolve_pick(self, position: Any, view_direction: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The world point a click at *position* lands on (front-most hit in 3D)."""
        layer = self._pick_layer
        pos = np.asarray(position, dtype=float)[-3:]
        if int(self._viewer.dims.ndisplay) == 3 and view_direction is not None:
            direction = np.asarray(view_direction, dtype=float)[-3:]
            if type(layer).__name__ == "Surface":
                mesh = layer_to_mesh(layer)
                return ray_hit(pos, direction, mesh.triangles), direction
            if type(layer).__name__ == "Points":
                pts = layer_to_point_cloud(layer).points
                d = direction / (np.linalg.norm(direction) or 1.0)
                rel = pts - pos
                perp = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
                size = float(np.mean(np.asarray(layer.size))) if np.size(layer.size) else 1.0
                near = perp <= max(size, 1e-6)
                if not near.any():
                    return None, d
                idx = np.flatnonzero(near)
                return pts[idx[np.argmin(rel[idx] @ d)]], d
            return pos, direction
        # 2D: the clicked point on the slice; the view direction is the slice normal.
        normal = np.zeros(3)
        try:
            normal[int(self._viewer.dims.not_displayed[-1]) - (int(self._viewer.dims.ndim) - 3)] = 1.0
        except Exception:  # noqa: BLE001
            normal[2] = 1.0
        op = self._picking or self._current_op()
        if type(layer).__name__ == "Surface" and op is not None and op.id not in ("cross_section", "vessel_section"):
            mesh = layer_to_mesh(layer)
            pos = mesh.vertices[int(np.argmin(np.linalg.norm(mesh.vertices - pos, axis=1)))]
        return pos, normal

    def pick_at(self, point: np.ndarray, view_direction: np.ndarray | None = None) -> None:
        """Finish a pick at *point* (world): mark it and run the armed operation."""
        layer = self._pick_layer
        self._show_marker(point, layer)
        self._stop_pick()
        if layer is not None and layer in self._viewer.layers:
            self._viewer.layers.selection.active = layer
        QTimer.singleShot(0, lambda: self.run_current(point=np.asarray(point, dtype=float),
                                                      view_direction=view_direction, layer=layer))

    def _show_marker(self, point: np.ndarray, layer: Any) -> None:
        marker = self._marker_ref()
        pt = np.asarray(point, dtype=np.float32)[None, :]
        try:
            span = float(np.ptp(layer_to_mesh(layer).vertices, axis=0).max()) if type(layer).__name__ == "Surface" else 10.0
        except Exception:  # noqa: BLE001
            span = 10.0
        if marker is None or marker not in self._viewer.layers:
            active = self._viewer.layers.selection.active
            marker = self._viewer.add_points(pt, name=_MARKER_NAME, size=max(span / 60.0, 0.5),
                                             face_color="#ff3b3b", border_width=0, out_of_slice_display=True)
            try:
                marker.shading = "spherical"
            except Exception:  # noqa: BLE001
                pass
            self._marker_ref = weakref.ref(marker)
            if active is not None and active in self._viewer.layers:
                self._viewer.layers.selection.active = active
        else:
            marker.data = pt

    # ── undo ─────────────────────────────────────────────────────────────────

    def _push_undo(self, layer: Any, previous: Any) -> None:
        """Remember *layer*'s content before an edit (one undo step)."""
        self._push_undo_step([(weakref.ref(layer), previous)])

    def _push_undo_step(self, step: list[tuple[Any, Any]]) -> None:
        self._undo_steps.append(step)
        del self._undo_steps[:-_UNDO_DEPTH]
        self._sync_undo_button()

    def _sync_undo_button(self) -> None:
        live = [s for s in self._undo_steps if any(r() is not None and r() in self._viewer.layers for r, _p in s)]
        self._undo_steps = live
        self._undo_btn.setVisible(bool(live))
        if live:
            names = [r().name for r, _p in live[-1] if r() is not None]
            self._undo_btn.setToolTip(f"Put back {', '.join(names)} as before the last edit "
                                      f"({len(live)} step(s) to undo).")

    def undo(self) -> None:
        """Put back the layers the last edit changed (a move of several layers is one step)."""
        self._sync_undo_button()
        if not self._undo_steps:
            return
        restored = []
        for ref, previous in self._undo_steps.pop():
            layer = ref()
            if layer is None or layer not in self._viewer.layers:
                continue
            if type(layer).__name__ == "Surface":
                replace_mesh_layer(layer, previous)
            else:
                set_points(layer, previous)
            restored.append(layer.name)
        self._status.setText(f"Undone on {', '.join(restored)} ({len(self._undo_steps)} more step(s) to undo).")
        self._on_active()

    # ── display ──────────────────────────────────────────────────────────────

    def _sync_display_card(self) -> None:
        targets = self._targets()
        self._display_card.setVisible(bool(targets))
        if not targets:
            return
        layer = targets[0]
        surfaces = [t for t in targets if type(t).__name__ == "Surface"]
        points = [t for t in targets if type(t).__name__ == "Points"]
        self._display_guard = True
        try:
            disp = display_of(layer)
            self._colour_by.clear()
            self._colour_by.addItem("solid colour", "")
            names: list[str] = []
            for t in targets:
                names += [n for n in field_names(t) if n not in names]
            for name in names:
                self._colour_by.addItem(name, name)
            idx = self._colour_by.findData(disp.field) if disp.mode == "field" else 0
            self._colour_by.setCurrentIndex(max(idx, 0))
            self._cmap.setCurrentText(disp.colormap)
            self._auto.setChecked(disp.auto_range)
            self._lo.setValue(float(disp.limits[0]))
            self._hi.setValue(float(disp.limits[1]))
            self._colour_btn.setStyleSheet(_BUTTON_STYLE + "QPushButton { color: %s; }" % QColor.fromRgbF(
                *[float(c) for c in disp.color[:3]]).name())
            self._opacity.setValue(int(round(float(getattr(layer, "opacity", 1.0)) * 100)))
            for w in (self._shading, self._shading_label, self._surf_opts):
                w.setVisible(bool(surfaces))
            for w in (self._pts_opts, self._pts_label):
                w.setVisible(bool(points))
            if surfaces:
                surf = surfaces[0]
                self._shading.setCurrentText(str(getattr(surf, "shading", "smooth")))
                parts = display_of(surf)
                self._faces.setChecked(bool(parts.faces))
                self._wire.setChecked(bool(parts.wireframe))
                self._vertices.setChecked(bool(parts.points))
                self._vertex_size.setValue(float(parts.point_size))
                for w in (self._vertex_size, self._vertex_size_label):
                    w.setVisible(bool(parts.points))
                self._normals.setChecked(bool(getattr(getattr(getattr(surf, "normals", None), "face", None),
                                                      "visible", False)))
            if points:
                pts = points[0]
                size = np.asarray(pts.size)
                self._size.setValue(float(size.mean()) if size.size else 1.0)
                self._symbol.setCurrentText(str(getattr(pts, "symbol", ["disc"])[0]).split(".")[-1].lower()
                                            if np.size(getattr(pts, "symbol", [])) else "disc")
                self._spherical.setChecked(str(getattr(pts, "shading", "none")).endswith("spherical"))
            self._update_field_info()
            self._sync_range_enabled()
        finally:
            self._display_guard = False

    def _sync_range_enabled(self) -> None:
        field_mode = bool(self._colour_by.currentData())
        self._cmap.setEnabled(field_mode)
        self._auto.setEnabled(field_mode)
        manual = field_mode and not self._auto.isChecked()
        self._lo.setEnabled(manual)
        self._hi.setEnabled(manual)
        self._colour_btn.setEnabled(not field_mode)

    def _update_field_info(self) -> None:
        targets = self._targets()
        name = self._colour_by.currentData()
        if not name or not targets:
            self._field_info.setText("")
            return
        having = [t for t in targets if name in field_names(t)]
        vals = fields_of(targets[0]).get(name) if targets[0] in having else None
        where = f" — on {len(having)} of the {len(targets)} layers" if len(targets) > 1 else ""
        if vals is None:
            self._field_info.setText(f"{name}: computed from the coordinates{where}." if name in BUILTIN_FIELDS
                                     else f"{name}{where}.")
            return
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            self._field_info.setText(f"{name}: {v.min():.4g} … {v.max():.4g} (median {np.median(v):.4g}){where}")

    def _on_display_changed(self, *, manual_range: bool = False) -> None:
        targets = self._targets()
        if self._display_guard or not targets:
            return
        field = str(self._colour_by.currentData() or "")
        changes: dict[str, Any] = {"mode": "field" if field else "solid", "field": field,
                                   "colormap": self._cmap.currentText(), "auto_range": self._auto.isChecked()}
        if manual_range:
            changes["auto_range"] = False
            self._display_guard = True
            self._auto.setChecked(False)
            self._display_guard = False
        if not changes["auto_range"]:
            changes["limits"] = (float(self._lo.value()), float(self._hi.value()))
        shown = None
        for t in targets:
            if field and field not in field_names(t):
                continue  # this layer has no such field: it keeps its colouring
            try:
                disp = set_display(t, **changes)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Display ({t.name}): {exc}")
                continue
            shown = disp if shown is None else shown
        if shown is not None:
            self._display_guard = True
            self._lo.setValue(float(shown.limits[0]))
            self._hi.setValue(float(shown.limits[1]))
            self._display_guard = False
        self._sync_range_enabled()
        self._update_field_info()

    def _pick_colour(self) -> None:
        targets = self._targets()
        if not targets:
            return
        c = display_of(targets[0]).color
        chosen = QColorDialog.getColor(QColor.fromRgbF(*[float(v) for v in c[:3]]), self, "Colour")
        if chosen.isValid():
            self.set_solid_colour((chosen.redF(), chosen.greenF(), chosen.blueF(), 1.0))

    def set_solid_colour(self, rgba: tuple[float, ...]) -> None:
        """Colour every target layer in *rgba*."""
        for t in self._targets():
            set_display(t, mode="solid", color=tuple(float(v) for v in rgba))
        self._sync_display_card()

    def _set_layer_attr(self, name: str, value: Any, *, kind: str = "") -> None:
        """Set a Napari attribute on every target layer (of *kind*, when given)."""
        if self._display_guard:
            return
        for t in self._targets():
            if kind and type(t).__name__ != kind:
                continue
            try:
                setattr(t, name, value)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Display ({t.name}): {exc}")

    def _on_parts(self, **changes: Any) -> None:
        """Faces / wireframe / points (point size) on every selected surface."""
        if "points" in changes:
            for w in (self._vertex_size, self._vertex_size_label):
                w.setVisible(bool(changes["points"]))
        if self._display_guard:
            return
        for t in self._targets():
            if type(t).__name__ != "Surface":
                continue
            try:
                set_parts(t, **changes)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Display ({t.name}): {exc}")

    def _on_normals(self, on: bool) -> None:
        if self._display_guard:
            return
        for t in self._targets():
            try:
                t.normals.face.visible = bool(on)
            except Exception:  # noqa: BLE001
                pass

    # ── moving by hand ───────────────────────────────────────────────────────

    def _reset_move_boxes(self) -> None:
        for key, box in getattr(self, "_move_boxes", {}).items():
            box.blockSignals(True)
            box.setValue(100.0 if key == "s" else 0.0)
            box.blockSignals(False)

    def _move_matrix(self) -> np.ndarray:
        from nvitk.meshlab.transform import compose_affine

        b = self._move_boxes
        centre = self._move_state["centre"] if self._move_state is not None else np.zeros(3)
        return compose_affine(
            translate=(b["tx"].value(), b["ty"].value(), b["tz"].value()),
            rotate_deg=(b["rx"].value(), b["ry"].value(), b["rz"].value()),
            scale=b["s"].value() / 100.0,
            centre=centre,
        )

    def _move_full(self, nd: int) -> np.ndarray:
        """The move as an (nd+1)² affine acting on the last three world dims."""
        move = self._move_matrix()
        mat = np.eye(nd + 1)
        mat[nd - 3:nd, nd - 3:nd] = move[:3, :3]
        mat[nd - 3:nd, nd] = move[:3, 3]
        return mat

    def _preview_move(self) -> None:
        from nvitk.gui.mesh.selection import layer_points

        targets = self._targets()
        if not targets:
            return
        if self._move_state is None or [ly for ly, _b in self._move_state["layers"]] != targets:
            if self._move_state is not None:
                self._restore_affines()
            pts = [layer_points(t) for t in targets]
            pts = [q for q in pts if len(q)]
            centre = np.vstack(pts).mean(axis=0) if pts else np.zeros(3)
            self._move_state = {"layers": [(t, np.asarray(t.affine.affine_matrix, dtype=float)) for t in targets],
                                "centre": centre}
        for layer, base in self._move_state["layers"]:
            layer.affine = self._move_full(base.shape[0] - 1) @ base

    def _restore_affines(self) -> None:
        for layer, base in (self._move_state or {}).get("layers", []):
            if layer in self._viewer.layers:
                layer.affine = base

    def apply_move(self) -> None:
        """Bake the previewed move into the vertices / points of every moved layer."""
        from nvitk.meshlab.transform import apply_affine

        if self._move_state is None:
            return
        step: list[tuple[Any, Any]] = []
        moved = []
        for layer, base in self._move_state["layers"]:
            if layer not in self._viewer.layers:
                continue
            ctrl = series_controller(layer)
            if ctrl is not None:
                # The frames are in the layer's data space: conjugate the world move by its affine.
                data_move = np.linalg.inv(base) @ self._move_full(base.shape[0] - 1) @ base
                mat = np.eye(4)
                mat[:3, :3] = data_move[-4:-1, -4:-1]
                mat[:3, 3] = data_move[-4:-1, -1]
                ctrl.series.frames = [apply_affine(f, mat) if f.n_vertices else f for f in ctrl.series]
                layer.affine = base
                ctrl.refresh()
            elif type(layer).__name__ == "Surface":
                step.append((weakref.ref(layer), layer_to_mesh_unmoved(layer, base)))
                replace_mesh_layer(layer, layer_to_mesh(layer))
            else:
                step.append((weakref.ref(layer), layer_to_cloud_unmoved(layer, base)))
                set_points(layer, layer_to_point_cloud(layer))
            moved.append(layer.name)
        if step:
            self._push_undo_step(step)
        self._move_state = None
        self._reset_move_boxes()
        self._status.setText(f"Move applied to {', '.join(moved)} (Undo edit puts it back).")
        self._on_active()

    def reset_move(self) -> None:
        """Drop the previewed move."""
        self._restore_affines()
        self._move_state = None
        self._reset_move_boxes()

    # ── time ─────────────────────────────────────────────────────────────────

    def _sync_time_card(self) -> None:
        ctrl = series_controller(self._layer)
        self._time_card.setVisible(ctrl is not None)
        if ctrl is None:
            if self._play.isChecked():
                self._play.setChecked(False)
            return
        ctrl.on_frame = self._on_series_frame
        self._frame.blockSignals(True)
        self._frame.setRange(0, len(ctrl.series) - 1)
        self._frame.setValue(max(ctrl.frame, 0))
        self._frame.blockSignals(False)
        self._on_series_frame(max(ctrl.frame, 0))
        self._time_hint.setVisible(viewer_time_dim(self._viewer) is not None)

    def _on_series_frame(self, frame: int) -> None:
        ctrl = series_controller(self._layer)
        if ctrl is None:
            return
        t = float(ctrl.series.times[frame]) if len(ctrl.series.times) > frame else frame
        self._frame_label.setText(f"frame {frame + 1} / {len(ctrl.series)} · t = {t:.3g}")
        if self._frame.value() != frame:
            self._frame.blockSignals(True)
            self._frame.setValue(frame)
            self._frame.blockSignals(False)

    def _on_frame_slider(self, frame: int) -> None:
        from nvitk.gui.mesh.layers import go_to_frame

        go_to_frame(self._viewer, self._layer, int(frame))

    def _on_play(self, playing: bool) -> None:
        self._play.setText("❚❚" if playing else "▶")
        if playing:
            self._play_timer.start(int(1000 / max(1, self._fps.value())))
        else:
            self._play_timer.stop()

    def _play_tick(self) -> None:
        ctrl = series_controller(self._layer)
        if ctrl is None:
            self._play.setChecked(False)
            return
        self._play_timer.setInterval(int(1000 / max(1, self._fps.value())))
        self._frame.setValue((self._frame.value() + 1) % len(ctrl.series))

    # ── plot ─────────────────────────────────────────────────────────────────

    def _show_plot(self, plot: dict[str, Any], title: str) -> None:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception:  # noqa: BLE001
            return
        if self._plot_canvas is None:
            fig = Figure(figsize=(4, 2.6), tight_layout=True)
            self._plot_canvas = FigureCanvasQTAgg(fig)
            self._plot_canvas.setMinimumHeight(220)
            self._plot_card.add(self._plot_canvas)
        fig = self._plot_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if "lines" in plot:
            for line in plot["lines"]:
                ax.plot(np.asarray(line["x"], float), np.asarray(line["y"], float), lw=1.4, label=line.get("label", ""))
            ax.set_ylabel(plot.get("ylabel", ""), fontsize=8)
            if len(plot["lines"]) > 1:
                ax.legend(fontsize=7, frameon=False)
        else:
            x = np.asarray(plot.get("x", []), dtype=float)
            first = True
            for name, values in plot.get("series", {}).items():
                target = ax if first else ax.twinx()
                colour = "#d98200" if first else "#2f7fd1"
                target.plot(x, np.asarray(values, dtype=float), marker="o", ms=3, color=colour, label=name)
                target.set_ylabel(name, color=colour, fontsize=8)
                target.tick_params(labelsize=7)
                first = False
        ax.set_xlabel(plot.get("xlabel", ""), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        style_figure(fig)
        self._plot_canvas.draw_idle()
        self._plot_card.setVisible(True)

    # ── files ────────────────────────────────────────────────────────────────

    def open_paths(self, paths: list[str], *, as_series: bool = False) -> list[Any]:
        """Add meshes / point clouds from *paths* (one series when *as_series*)."""
        from pathlib import Path

        from nvitk.meshlab.io import read_mesh_series, read_pvd, read_surface
        from nvitk.types import Mesh

        layers = []
        if as_series:
            series = read_mesh_series(paths)
            layer = add_mesh_series_layer(self._viewer, series, name=series.name)
            apply_display(layer)
            return [layer]
        for p in paths:
            path = Path(p)
            if path.suffix.lower() == ".pvd":
                series = read_pvd(path)
                layer = add_mesh_series_layer(self._viewer, series, name=series.name)
                apply_display(layer)
                layers.append(layer)
                continue
            obj = read_surface(path)
            if isinstance(obj, Mesh):
                layers.append(add_mesh_layer(self._viewer, obj, name=obj.name))
            else:
                layers.append(add_point_cloud_layer(self._viewer, obj, name=obj.name))
        return layers

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open meshes / point clouds", "", _MESH_FILTER)
        if paths:
            try:
                self.open_paths(paths)
            except Exception as exc:  # noqa: BLE001
                _notify(f"Could not open: {exc}", error=True)

    def _open_series(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open a mesh time series (one file per frame)", "", _MESH_FILTER)
        if paths:
            try:
                self.open_paths(paths, as_series=True)
            except Exception as exc:  # noqa: BLE001
                _notify(f"Could not open the series: {exc}", error=True)

    def save_layer(self, layer: Any, path: str) -> list[Any]:
        """Write *layer* to *path* (a directory for a series)."""
        from nvitk.meshlab.io import write_mesh_series, write_surface

        ctrl = series_controller(layer)
        if ctrl is not None:
            return write_mesh_series(ctrl.series, path)
        if type(layer).__name__ == "Surface":
            return [write_surface(path, layer_to_mesh(layer))]
        return [write_surface(path, layer_to_point_cloud(layer))]

    def _save(self) -> None:
        layer = self._viewer.layers.selection.active
        kind = layer_kind(layer)
        if kind not in ("mesh", "series", "points"):
            _notify("Select a surface, mesh series or points layer to save.", error=True)
            return
        try:
            if kind == "series":
                folder = QFileDialog.getExistingDirectory(self, "Folder for the series (one file per frame + .pvd)")
                if not folder:
                    return
                written = self.save_layer(layer, folder)
            else:
                filt = ("STL (*.stl);;PLY (*.ply);;OBJ (*.obj);;VTK PolyData (*.vtp);;OFF (*.off);;GIfTI (*.gii)"
                        if kind == "mesh" else "XYZ (*.xyz);;CSV (*.csv);;PLY (*.ply);;VTK PolyData (*.vtp)")
                default = f"{layer.name}.stl" if kind == "mesh" else f"{layer.name}.xyz"
                path, _ = QFileDialog.getSaveFileName(self, "Save", default, filt)
                if not path:
                    return
                written = self.save_layer(layer, path)
        except Exception as exc:  # noqa: BLE001
            _notify(f"Save failed: {exc}", error=True)
            return
        _notify(f"Saved {layer.name} → {written[0] if len(written) == 1 else f'{len(written)} files'}")


def layer_to_mesh_unmoved(layer: Any, base_affine: np.ndarray) -> Any:
    """The surface as it was before a live move (for Undo)."""
    current = layer.affine.affine_matrix
    layer.affine = base_affine
    try:
        return layer_to_mesh(layer)
    finally:
        layer.affine = current


def layer_to_cloud_unmoved(layer: Any, base_affine: np.ndarray) -> Any:
    """The points as they were before a live move (for Undo)."""
    current = layer.affine.affine_matrix
    layer.affine = base_affine
    try:
        return layer_to_point_cloud(layer)
    finally:
        layer.affine = current


def build_mesh_panel(viewer: Any) -> MeshPanel:
    """The Meshlab dock's widget."""
    panel = MeshPanel(viewer)
    viewer._nvitk_mesh_panel = panel
    return panel


__all__ = ["LabelChecklist", "MeshPanel", "build_mesh_panel", "ray_hit"]
