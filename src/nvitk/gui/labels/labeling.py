"""The **Labeling** tab: manual segmentation and re-segmentation of a label layer.

One panel around the label layer being drawn, from Napari's own tools to the ones
it lacks:

* **Layer** — the Labels layer edited (a new one fitted to an image, or an Image
  mask converted for undo), and the reference image the intensity tools read;
* **Label** — the active label: id, name (named by hand, or picked from the
  layer's vocabulary), colour, a list of every label with its size, go-to;
* **Draw** — Napari's paint / erase / fill / pick / polygon modes and the magic
  wand, brush size, preserve-other-labels, 2D or 3D brush, Undo / Redo — on the
  canvas and in the orthogonal views (:mod:`nvitk.gui.labels.ortho_edit`);
* **Editable area** — paint only where the reference intensity is in a range
  and/or inside a mask (brush, bucket and every tool here);
* **Refine** — grow, shrink, smooth, fill holes, keep the largest piece, drop
  small islands, threshold fill — on the slice on screen or the whole volume;
* **Slices** — interpolate between drawn slices, copy a slice to the next one;
* **Manage** — merge, split into pieces, renumber, delete.

Every edit goes through Napari's history: Ctrl+Z (or Undo here) takes back one
tool application, like a brush stroke. The operations themselves are in
:mod:`nvitk.gui.labels.editing`.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Sequence

import numpy as np
from qtpy.QtCore import QStringListModel, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_ACCENT_DEEP,
    COLOR_BORDER,
    COLOR_BORDER_STRONG,
    COLOR_DISABLED,
    COLOR_ERROR,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_ON_ACCENT,
    SPACE,
    SPACE_TIGHT,
    Card,
)
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.gui.labels import editing as E
from nvitk.gui.labels.catalog import (
    custom_label_names,
    get_schema,
    label_name,
    layer_schema_key,
    set_label_name,
)
from nvitk.gui.labels.region_tools import (
    EXTRA_TOOLS,
    REGION_TOOLS,
    TUNED,
    RegionStroke,
    SmartBrushStroke,
    VesselTrace,
)
from nvitk.gui.labels.roi_box import RoiBox, crop_layers, roi_box, spatial_dims, spatial_shape
from nvitk.gui.labels.selector import label_filter_hub, set_layer_visible_ids
from nvitk.gui.labels.visibility import (
    copy_layer_metadata_for_output,
    ensure_labels_layer,
    get_label_color,
    invalidate_label_ids,
    is_label_like_layer,
    label_source_data,
    layer_label_ids,
    set_label_color,
    stored_visible_ids,
)

#: Dock object name, the key the saved layout knows it by.
LABELING_DOCK_NAME = "nvitk:labeling"
_NONE = "(none)"
#: The vessel tracer's preview on the canvas.
_TRACE_PREVIEW = "✎ vessel path (Labeling)"
#: Napari Labels modes behind the Draw buttons; the wand is ours (pan/zoom underneath).
_TOOLS: tuple[tuple[str, str, str], ...] = (
    ("pan_zoom", "Pan", "Move and zoom the view (Space held does it from any tool)."),
    ("paint", "Paint", "Paint the active label (Napari: 2). [ and ] change the brush size."),
    ("erase", "Erase", "Erase to background (Napari: 1)."),
    ("fill", "Fill", "Bucket fill: the region clicked becomes the active label (Napari: 3)."),
    ("pick", "Pick", "Click a label on the canvas to make it the active one (Napari: 4)."),
    ("polygon", "Polygon", "Click the corners of a polygon — on the canvas (2D) or in an orthogonal view — "
                           "then double-click, or click the first corner, to fill it with the active label."),
    ("wand", "Wand", "Magic wand: click — or drag, to add a region at every step — to grow the active label "
                     "through intensities like the clicked voxel's (options below); Shift takes it out; Alt+drag "
                     "tunes the tolerance; in 3D, click on what the image shows."),
)
#: The tab's tools for vessels and region flooding (second row of the Draw card).
_MORE_TOOLS: tuple[tuple[str, str, str], ...] = (
    ("flood", "Flood", "Adaptive flood: click or drag; the region grows through intensities within k standard "
                       "deviations of its own mean, both learned from what it takes in. Alt+drag tunes k."),
    ("vessel", "Vessel", "Vessel flood: click or drag on a vessel; it grows through the vesselness (tubes at "
                         "the radii below), not into blobs or the background. Alt+drag tunes how far it goes."),
    ("tracer", "Tracer", "Vessel tracer: click points along a vessel — on the canvas, in 3D, in any orthogonal "
                         "view or slice; the path follows the lumen. Double-click or Fill tube labels a tube "
                         "around it."),
    ("smart", "Smart brush", "A brush that paints only the voxels like the one under its centre (the brush's two "
                             "intensity populations split by Otsu, or a tolerance). Ctrl+drag pans."),
)
#: What each extra tool's hint line says.
_TOOL_HINTS = {
    "wand": "Click, or drag to add along the way · Shift removes · Alt+drag tunes the tolerance · "
            "Ctrl+drag pans · right-drag erases.",
    "flood": "Click, or drag to add along the way · Shift removes · Alt+drag tunes k · Ctrl+drag pans.",
    "vessel": "Click on a vessel, or drag along it · Shift removes · Alt+drag tunes the vesselness fraction · "
              "Ctrl+drag pans.",
    "tracer": "Click points along the vessel (any view, any slice, or in 3D); double-click or Fill tube.",
    "smart": "Drag to paint the voxels like the one under the brush's centre · Ctrl+drag pans.",
}


#: Cards strip their children's borders (``QWidget { border: none }``): buttons in
#: them carry their own, or they read as plain text and a tool's checked state
#: does not show.
_BUTTON_STYLE = (
    f"QPushButton, QToolButton {{ border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 4px 8px; }}"
    f"QPushButton:hover, QToolButton:hover {{ border-color: {COLOR_BORDER_STRONG}; }}"
    f"QPushButton:disabled, QToolButton:disabled {{ color: {COLOR_DISABLED}; }}"
    f"QPushButton:checked, QToolButton:checked {{ background-color: {COLOR_ACCENT_DEEP};"
    f" border-color: {COLOR_ACCENT}; color: {COLOR_ON_ACCENT}; }}"
)
_INPUT_STYLE = f"border: 1px solid {COLOR_BORDER}; border-radius: 3px;"
#: Rows the label list grows to before it scrolls.
_LIST_ROWS = (3, 8)


def _swatch_style(color: QColor) -> str:
    """A round, filled colour dot for the active label's swatch button."""
    return (
        f"QToolButton {{ background-color: rgba({color.red()},{color.green()},{color.blue()},{color.alpha()});"
        f" border: 1px solid {COLOR_BORDER_STRONG}; border-radius: 10px; }}"
    )


def _swatch_icon(color: QColor, size: int = 14) -> QIcon:
    """A round colour dot."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    from qtpy.QtGui import QPainter

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(color)
    painter.setPen(QColor(0, 0, 0, 90))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pix)


def _rgba_qcolor(rgba: Any) -> QColor:
    vals = [min(max(float(v), 0.0), 1.0) for v in list(np.ravel(rgba))[:4]] + [1.0]
    return QColor.fromRgbF(vals[0], vals[1], vals[2], vals[3])


def _row(*widgets: Any, stretch_last: bool = False) -> QHBoxLayout:
    """Widgets side by side (a str becomes a label)."""
    row = QHBoxLayout()
    row.setSpacing(SPACE_TIGHT)
    for i, widget in enumerate(widgets):
        if isinstance(widget, str):
            widget = QLabel(widget)
        stretch = 1 if (stretch_last and i == len(widgets) - 1) else 0
        row.addWidget(widget, stretch)
    return row


class _LayerControlsBox(QWidget):
    """Napari's own layer controls for one layer, folded under a header.

    Built with Napari's ``create_qt_layer_controls`` — the very widget its layer
    controls dock shows (opacity, blending, colormap, contrast, gamma,
    interpolation; for labels the modes, brush, contour, colour mode…) — and
    rebuilt when the layer changes; the previous one is closed (disconnected from
    its layer) first.
    """

    def __init__(self, title: str, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._title = title
        self._layer: Callable[[], Any] = lambda: None
        self._controls: Any | None = None
        self._header = QToolButton()
        self._header.setCheckable(True)
        self._header.setChecked(False)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.ArrowType.RightArrow)
        self._header.setStyleSheet("QToolButton { border: none; font-weight: bold; padding: 2px 0px; }")
        self._header.toggled.connect(self._on_toggled)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 0, 0, 0)
        self._empty = QLabel("No layer.")
        self._empty.setStyleSheet(f"color: {COLOR_MUTED};")
        self._body_layout.addWidget(self._empty)
        self._body.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._header)
        layout.addWidget(self._body)
        try:
            viewer.dims.events.ndisplay.connect(self._on_ndisplay)
        except Exception:  # noqa: BLE001
            pass
        self._update_header()

    def _update_header(self) -> None:
        layer = self._layer()
        self._header.setText(f"{self._title}: {layer.name}" if layer is not None else f"{self._title}: —")

    def _on_toggled(self, on: bool) -> None:
        self._header.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self._body.setVisible(on)

    def _on_ndisplay(self, _event: Any = None) -> None:
        if self._controls is not None:
            try:
                self._controls.ndisplay = int(self._viewer.dims.ndisplay)
            except Exception:  # noqa: BLE001
                pass

    def set_layer(self, layer: Any | None) -> None:
        """Show *layer*'s controls (``None``: none)."""
        if self._layer() is layer and (layer is None or self._controls is not None):
            return
        self._close_controls()
        self._layer = weakref.ref(layer) if layer is not None else (lambda: None)
        self._update_header()
        if layer is None:
            self._empty.setVisible(True)
            return
        try:
            from napari._qt.layer_controls.qt_layer_controls_container import create_qt_layer_controls

            controls = create_qt_layer_controls(layer)
            controls.ndisplay = int(self._viewer.dims.ndisplay)
        except Exception as exc:  # noqa: BLE001 — a layer type without controls
            self._empty.setText(f"No controls for “{layer.name}”: {exc}")
            self._empty.setVisible(True)
            return
        self._empty.setVisible(False)
        self._controls = controls
        self._body_layout.addWidget(controls)

    def _close_controls(self) -> None:
        controls, self._controls = self._controls, None
        if controls is None:
            return
        try:
            controls.close()  # Napari's: disconnects it from its layer's events
        except Exception:  # noqa: BLE001
            pass
        controls.hide()  # before unparenting: a queued layout show must not pop it up as a window
        controls.setParent(None)
        controls.deleteLater()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self._close_controls()
        super().closeEvent(event)


def _button(text: str, tip: str, slot: Callable[[], Any]) -> QPushButton:
    button = QPushButton(text)
    button.setStyleSheet(_BUTTON_STYLE)
    button.setToolTip(tip)
    button.clicked.connect(lambda _checked=False: slot())
    return button


class LabelingPanel(QWidget):
    """The Labeling tab (see the module docstring)."""

    #: The layer edited or the magic wand changed: views that draw with these
    #: tools (the orthogonal views) re-read them.
    tool_changed = Signal()

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._layer_ref: Callable[[], Any] = lambda: None
        self._connections: list[tuple[Any, Any]] = []
        self._syncing = False
        #: The tab's own Draw tool, when one is on: "wand" or "polygon" (Napari's
        #: layer is then in pan/zoom, the clicks are the tab's).
        self._extra: str | None = None
        #: The polygon being drawn on the canvas: its layer, plane, corners, preview.
        self._poly: dict[str, Any] | None = None
        self._counts: dict[int, int] = {}
        #: The tool a right-drag put aside for the eraser, until the button is released.
        self._erase_restore: str | None = None
        #: The vessel being traced, and its preview layer.
        self._trace: VesselTrace | None = None
        self._trace_ref: Callable[[], Any] = lambda: None
        self._clicked = False
        #: The box (Box card): drawn on the canvas or in the orthogonal views.
        self._box_armed = False
        self._box_restore: str | None = None
        self._syncing_box = False
        self._from_view_buttons: list[QPushButton] = []
        # Strokes written live (outside the history) are shown once per event-loop turn.
        self._live_pending: tuple[Any, tuple[int, ...]] | None = None
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(0)
        self._live_timer.timeout.connect(self._live_flush)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_TIGHT, SPACE, SPACE_TIGHT, SPACE)
        root.setSpacing(SPACE)
        root.addWidget(self._build_layer_card())
        root.addWidget(self._build_controls_section())
        root.addWidget(self._build_label_card())
        root.addWidget(self._build_draw_card())
        root.addWidget(self._build_area_card())
        root.addWidget(self._build_box_card())
        root.addWidget(self._build_refine_card())
        root.addWidget(self._build_slices_card())
        root.addWidget(self._build_manage_card())
        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        root.addStretch(1)

        # Counting voxels scans the volume: once a burst of strokes is over.
        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(400)
        self._count_timer.timeout.connect(self._refresh_label_list)
        self._combo_timer = QTimer(self)
        self._combo_timer.setSingleShot(True)
        self._combo_timer.setInterval(0)
        self._combo_timer.timeout.connect(self._refresh_layer_combos)

        layers = viewer.layers
        for name in ("inserted", "removed", "reordered"):
            getattr(layers.events, name).connect(lambda _e=None: self._combo_timer.start())
        layers.events.removed.connect(self._on_layer_removed)
        layers.selection.events.active.connect(self._on_active_changed)
        viewer.dims.events.ndisplay.connect(self._update_camera_lock)
        # First in the list: a right-drag must switch to the eraser before the
        # layer's own callbacks see the press.
        viewer.mouse_drag_callbacks.insert(0, self._right_erase)
        viewer.mouse_drag_callbacks.append(self._canvas_callback)
        viewer.mouse_double_click_callbacks.append(self._canvas_double_click)
        from nvitk.gui.core.napari_fixes import register_labels_3d_fallback

        register_labels_3d_fallback(self._brush_3d_fallback)
        hub = label_filter_hub()
        hub.names_changed.connect(self._on_names_changed)
        hub.labels_changed.connect(self._on_labels_changed)
        self._refresh_layer_combos()
        self._on_active_changed()

    # ── building ──────────────────────────────────────────────────────────────

    def _build_layer_card(self) -> Card:
        card = Card("Layer")
        self._layer_combo = QComboBox()
        self._layer_combo.setToolTip("The label layer being drawn.")
        self._layer_combo.activated.connect(self._on_layer_combo)
        new = _button("New…", "A new, empty Labels layer on the reference image's grid.", self.new_layer)
        card.add_layout(_row("Labels", self._layer_combo, new))
        self._layer_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._ref_combo = QComboBox()
        self._ref_combo.setToolTip(
            "The image the intensity tools read (editable range, wand, threshold): one "
            "on the same grid as the labels."
        )
        self._ref_combo.activated.connect(lambda _i: self._on_reference_changed())
        self._ref_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card.add_layout(_row("Reference", self._ref_combo, stretch_last=True))
        # Napari's own controls of the two layers: everything its layer-controls
        # dock offers, here beside the tools (folded until wanted). Outside the
        # card (see _build_controls_section): a card flattens its children's style.
        self._labels_controls = _LayerControlsBox("Labels controls", self._viewer)
        self._labels_controls.setToolTip("Napari's controls of the label layer: opacity, blending, colour "
                                         "mode, contour, show selected, the editing modes…")
        self._image_controls = _LayerControlsBox("Image controls", self._viewer)
        self._image_controls.setToolTip("Napari's controls of the reference image: opacity, blending, "
                                        "contrast limits, auto-contrast, gamma, colormap, interpolation…")
        self._convert_row = QWidget()
        conv = QHBoxLayout(self._convert_row)
        conv.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("An image mask: no undo, no brush.")
        hint.setStyleSheet(f"color: {COLOR_MUTED};")
        conv.addWidget(hint, 1)
        conv.addWidget(_button("Convert to Labels", "Replace it by an equivalent Labels layer.", self.convert_target))
        card.add(self._convert_row)
        self._convert_row.hide()
        return card

    def _build_controls_section(self) -> QWidget:
        """Napari's layer controls of the label layer and of the reference image."""
        from nvitk.gui.core.design import section_heading

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(2)
        layout.addWidget(section_heading("Layer display (Napari controls)"))
        layout.addWidget(self._labels_controls)
        layout.addWidget(self._image_controls)
        return box

    def _build_label_card(self) -> Card:
        card = Card("Label")
        self._swatch = QToolButton()
        self._swatch.setFixedSize(20, 20)
        self._swatch.setToolTip("The active label's colour — click to change it.")
        self._swatch.clicked.connect(self.edit_colour)
        self._id_spin = QSpinBox()
        self._id_spin.setRange(0, 2_147_483_647)
        self._id_spin.setToolTip("The active label's id (0 is the background: painting with it erases).")
        self._id_spin.valueChanged.connect(self._on_id_changed)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Name")
        self._name_edit.setToolTip(
            "Name the active label. Typing shows the layer vocabulary's names; picking "
            "one makes its id the active label."
        )
        self._name_edit.returnPressed.connect(self._on_name_entered)
        self._completer = QCompleter([], self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._completer.activated.connect(self._on_vocabulary_name)
        self._name_edit.setCompleter(self._completer)
        new_id = _button("New id", "The next unused id becomes the active label.", self.new_label)
        row = _row(self._swatch, self._id_spin, self._name_edit, new_id)
        row.setStretch(2, 1)
        card.add_layout(row)

        self._list = QListWidget()
        self._list.setStyleSheet(_INPUT_STYLE)
        self._list.setToolTip("Click: make it the active label · double-click: rename · right-click: more")
        self._list.itemClicked.connect(lambda item: self.select_label(int(item.data(Qt.ItemDataRole.UserRole))))
        self._list.itemDoubleClicked.connect(lambda item: self._rename_from_list(int(item.data(Qt.ItemDataRole.UserRole))))
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._list_menu)
        card.add(self._list)
        self._stats = QLabel("")
        self._stats.setStyleSheet(f"color: {COLOR_MUTED};")
        go = _button("Go to", "Show the active label: its centre's slice, the camera on it.", self.go_to)
        card.add_layout(_row(self._stats, go))
        return card

    def _build_draw_card(self) -> Card:
        card = Card("Draw")
        card.setToolTip("These tools draw on the main canvas and in the orthogonal views alike. "
                        "A right-drag erases with the brush, whatever the tool.")
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        grid = QGridLayout()
        grid.setSpacing(SPACE_TIGHT)
        self._tool_buttons: dict[str, QToolButton] = {}
        for i, (mode, text, tip) in enumerate(_TOOLS):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setStyleSheet(_BUTTON_STYLE)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._tool_group.addButton(button)
            self._tool_buttons[mode] = button
            button.clicked.connect(lambda _c=False, m=mode: self.set_tool(m))
            grid.addWidget(button, i // 4, i % 4)
        card.add_layout(grid)
        more = QLabel("Vessels & flooding")
        more.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px; font-weight: bold;")
        card.add(more)
        grid_more = QGridLayout()
        grid_more.setSpacing(SPACE_TIGHT)
        for i, (mode, text, tip) in enumerate(_MORE_TOOLS):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setStyleSheet(_BUTTON_STYLE)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._tool_group.addButton(button)
            self._tool_buttons[mode] = button
            button.clicked.connect(lambda _c=False, m=mode: self.set_tool(m))
            grid_more.addWidget(button, 0, i)
        card.add_layout(grid_more)
        hint = QLabel("Right-drag erases, whatever the tool.")
        hint.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        card.add(hint)

        self._brush = QSlider(Qt.Orientation.Horizontal)
        self._brush.setRange(1, 100)
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 100)
        self._brush.valueChanged.connect(self._brush_spin.setValue)
        self._brush_spin.valueChanged.connect(self._brush.setValue)
        self._brush_spin.valueChanged.connect(lambda v: self._set_layer("brush_size", int(v)))
        card.add_layout(_row("Brush", self._brush, self._brush_spin))

        self._preserve = QCheckBox("Preserve other labels")
        self._preserve.setToolTip("Brush, bucket and every tool here only write over background and the active label.")
        self._preserve.toggled.connect(lambda v: self._set_layer("preserve_labels", bool(v)))
        self._brush3d = QCheckBox("3D brush / fill")
        self._brush3d.setToolTip("Paint and fill through slices (a ball) rather than in the slice on screen.")
        self._brush3d.toggled.connect(lambda v: self._set_layer("n_edit_dimensions", 3 if v else 2))
        self._contiguous = QCheckBox("Fill contiguous only")
        self._contiguous.setToolTip("The bucket fills only the connected region clicked, not every voxel of that label.")
        self._contiguous.toggled.connect(lambda v: self._set_layer("contiguous", bool(v)))
        self._contour = QCheckBox("Outlines")
        self._contour.setToolTip("Draw the labels as outlines, to see the image underneath.")
        self._contour.toggled.connect(lambda v: self._set_layer("contour", 1 if v else 0))
        grid2 = QGridLayout()
        grid2.addWidget(self._preserve, 0, 0)
        grid2.addWidget(self._brush3d, 0, 1)
        grid2.addWidget(self._contiguous, 1, 0)
        grid2.addWidget(self._contour, 1, 1)
        card.add_layout(grid2)

        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 100)
        self._opacity.valueChanged.connect(lambda v: self._set_layer("opacity", v / 100.0))
        undo = _button("Undo", "Take back the last edit (Ctrl+Z on the canvas).", lambda: self._history("undo"))
        redo = _button("Redo", "Do it again (Ctrl+Shift+Z).", lambda: self._history("redo"))
        card.add_layout(_row("Opacity", self._opacity, undo, redo))

        self._tool_hint = QLabel("")
        self._tool_hint.setWordWrap(True)
        self._tool_hint.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        card.add(self._tool_hint)
        self._tool_hint.hide()
        card.add(self._build_wand_box())
        card.add(self._build_more_boxes())
        self._poly_box = QWidget()
        poly = QHBoxLayout(self._poly_box)
        poly.setContentsMargins(0, 0, 0, 0)
        self._poly_hint = QLabel("Click the corners; double-click or click the first one to fill.")
        self._poly_hint.setWordWrap(True)
        self._poly_hint.setStyleSheet(f"color: {COLOR_MUTED};")
        poly.addWidget(self._poly_hint, 1)
        poly.addWidget(_button("Cancel", "Drop the polygon being drawn.", self.cancel_polygon))
        card.add(self._poly_box)
        self._poly_box.hide()
        return card

    def _build_wand_box(self) -> QWidget:
        """The region tools' options: how a region grows (shared by the wand, the
        flood and the vessel flood), then the wand's own tolerance."""
        self._wand_box = QWidget()  # the shared growing options
        grid = QGridLayout(self._wand_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE_TIGHT)
        self._wand_tol = QDoubleSpinBox()
        self._wand_tol.setRange(0.0, 1e9)
        self._wand_tol.setDecimals(2)
        self._wand_tol.setValue(10.0)
        self._wand_tol.setToolTip("How far from the seed's intensity the region may grow (Alt+drag tunes it).")
        self._wand_tol_mode = QComboBox()
        self._wand_tol_mode.addItem("% of the window", "percent")
        self._wand_tol_mode.addItem("absolute", "absolute")
        self._wand_tol_mode.setToolTip("% of the reference's display window — or of the box's intensity range "
                                       "(Box card) — which adapts to any image; or image units (HU on a CT).")
        self._wand_smooth = QDoubleSpinBox()
        self._wand_smooth.setRange(0.0, 10.0)
        self._wand_smooth.setDecimals(1)
        self._wand_smooth.setSingleStep(0.5)
        self._wand_smooth.setValue(1.0)
        self._wand_smooth.setSuffix(" vox")
        self._wand_smooth.setToolTip("Smooth the image first (Gaussian σ): noise no longer splits the "
                                     "region into single voxels.")
        self._wand_seed = QSpinBox()
        self._wand_seed.setRange(0, 10)
        self._wand_seed.setValue(1)
        self._wand_seed.setSuffix(" vox")
        self._wand_seed.setToolTip("The seed value is the mean around the click, this many voxels each way.")
        self._wand_conn = QComboBox()
        self._wand_conn.addItem("faces", False)
        self._wand_conn.addItem("all neighbours", True)
        self._wand_conn.setToolTip("Grow through face neighbours only (tighter), or diagonals too.")
        self._wand_maxd = QDoubleSpinBox()
        self._wand_maxd.setRange(0.0, 1000.0)
        self._wand_maxd.setDecimals(1)
        self._wand_maxd.setSuffix(" mm")
        self._wand_maxd.setSpecialValueText("no limit")
        self._wand_maxd.setToolTip("Keep the region within this distance of the click (stops leaks).")
        self._wand_close = QSpinBox()
        self._wand_close.setRange(0, 10)
        self._wand_close.setSuffix(" vox")
        self._wand_close.setSpecialValueText("off")
        self._wand_close.setToolTip("Close gaps up to this size in the region.")
        self._wand_holes = QCheckBox("Fill holes")
        self._wand_holes.setChecked(True)
        self._wand_3d = QCheckBox("3D")
        self._wand_3d.setToolTip("Grow through the whole volume (inside the box, with Keep edits inside the "
                                 "box), not only the slice clicked. Always on for clicks on the 3D canvas.")
        self._wand_3d.toggled.connect(self._remember_grow_3d)
        #: The 3D switch per region tool (vessels are 3D things).
        self._grow3d = {"wand": False, "flood": False, "vessel": True}
        self._wand_tol_row = [QLabel("Tolerance ±"), self._wand_tol, self._wand_tol_mode]
        rows = (
            (self._wand_tol_row[0], self._wand_tol, self._wand_tol_mode),
            (QLabel("Smoothing"), self._wand_smooth, None),
            (QLabel("Seed average"), self._wand_seed, None),
            (QLabel("Connectivity"), self._wand_conn, None),
            (QLabel("Max distance"), self._wand_maxd, None),
            (QLabel("Close gaps"), self._wand_close, None),
        )
        for r, (label, a, b) in enumerate(rows):
            grid.addWidget(label, r, 0)
            if b is None:
                grid.addWidget(a, r, 1, 1, 2)
            else:
                grid.addWidget(a, r, 1)
                grid.addWidget(b, r, 2)
        grid.addWidget(self._wand_holes, len(rows), 0, 1, 2)
        grid.addWidget(self._wand_3d, len(rows), 2)
        self._wand_box.hide()
        return self._wand_box

    def _build_more_boxes(self) -> QWidget:
        """Options of the flood, the vessel tools and the smart brush."""
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE_TIGHT)

        # Adaptive flood: k and the rounds of re-estimation.
        self._flood_box = QWidget()
        g = QGridLayout(self._flood_box)
        g.setContentsMargins(0, 0, 0, 0)
        self._flood_k = QDoubleSpinBox()
        self._flood_k.setRange(0.1, 10.0)
        self._flood_k.setDecimals(2)
        self._flood_k.setSingleStep(0.25)
        self._flood_k.setValue(2.5)
        self._flood_k.setSuffix(" σ")
        self._flood_k.setToolTip("How many standard deviations of the region's own intensity it reaches "
                                 "(Alt+drag tunes it).")
        self._flood_rounds = QSpinBox()
        self._flood_rounds.setRange(0, 20)
        self._flood_rounds.setValue(4)
        self._flood_rounds.setToolTip("How many times the mean and spread are learned again from the region.")
        g.addWidget(QLabel("Spread k"), 0, 0)
        g.addWidget(self._flood_k, 0, 1)
        g.addWidget(QLabel("Rounds"), 1, 0)
        g.addWidget(self._flood_rounds, 1, 1)
        lay.addWidget(self._flood_box)

        # Vessels: radii and polarity (vessel flood and tracer).
        self._vessel_common = QWidget()
        g = QGridLayout(self._vessel_common)
        g.setContentsMargins(0, 0, 0, 0)
        self._vessel_r0 = QDoubleSpinBox()
        self._vessel_r1 = QDoubleSpinBox()
        for spin, value in ((self._vessel_r0, 0.5), (self._vessel_r1, 3.0)):
            spin.setRange(0.1, 50.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" mm")
            spin.setValue(value)
        self._vessel_r0.setToolTip("Radius of the thinnest vessel looked for.")
        self._vessel_r1.setToolTip("Radius of the widest vessel looked for.")
        self._vessel_bright = QComboBox()
        self._vessel_bright.addItem("bright (contrast, TOF)", True)
        self._vessel_bright.addItem("dark (black blood)", False)
        g.addWidget(QLabel("Radius"), 0, 0)
        g.addLayout(_row(self._vessel_r0, "–", self._vessel_r1), 0, 1)
        g.addWidget(QLabel("Vessels"), 1, 0)
        g.addWidget(self._vessel_bright, 1, 1)
        lay.addWidget(self._vessel_common)

        # Vessel flood: how far through the vesselness, and an intensity bound.
        self._vessel_box = QWidget()
        g = QGridLayout(self._vessel_box)
        g.setContentsMargins(0, 0, 0, 0)
        self._vessel_frac = QDoubleSpinBox()
        self._vessel_frac.setRange(0.01, 1.0)
        self._vessel_frac.setDecimals(3)
        self._vessel_frac.setSingleStep(0.05)
        self._vessel_frac.setValue(0.15)
        self._vessel_frac.setToolTip("Grow through voxels whose vesselness is at least this fraction of the "
                                     "seed's (lower goes further; Alt+drag tunes it).")
        self._vessel_tol = QDoubleSpinBox()
        self._vessel_tol.setRange(0.0, 100.0)
        self._vessel_tol.setDecimals(1)
        self._vessel_tol.setValue(40.0)
        self._vessel_tol.setSuffix(" %")
        self._vessel_tol.setSpecialValueText("off")
        self._vessel_tol.setToolTip("And no darker (brighter, for dark vessels) than the seed by more than this "
                                    "% of the window.")
        g.addWidget(QLabel("Vesselness ≥"), 0, 0)
        g.addWidget(self._vessel_frac, 0, 1)
        g.addWidget(QLabel("Intensity within"), 1, 0)
        g.addWidget(self._vessel_tol, 1, 1)
        lay.addWidget(self._vessel_box)

        # Tracer: the tube's radius and its buttons.
        self._tracer_box = QWidget()
        g = QGridLayout(self._tracer_box)
        g.setContentsMargins(0, 0, 0, 0)
        self._tracer_radius = QDoubleSpinBox()
        self._tracer_radius.setRange(0.0, 50.0)
        self._tracer_radius.setDecimals(1)
        self._tracer_radius.setSingleStep(0.5)
        self._tracer_radius.setSuffix(" mm")
        self._tracer_radius.setSpecialValueText("measured")
        self._tracer_radius.setToolTip("The tube's radius; 'measured': the lumen's, point by point along the path.")
        self._tracer_info = QLabel("No points yet.")
        self._tracer_info.setStyleSheet(f"color: {COLOR_MUTED};")
        g.addWidget(QLabel("Tube radius"), 0, 0)
        g.addWidget(self._tracer_radius, 0, 1)
        g.addLayout(_row(_button("Fill tube", "Label a tube around the path (one undo step).", self.tracer_fill),
                         _button("Undo point", "Take the last point back.", self.tracer_undo),
                         _button("Cancel", "Drop the points.", self.tracer_cancel)), 1, 0, 1, 2)
        g.addWidget(self._tracer_info, 2, 0, 1, 2)
        lay.addWidget(self._tracer_box)

        # Smart brush: Otsu, or a tolerance.
        self._smart_box = QWidget()
        g = QGridLayout(self._smart_box)
        g.setContentsMargins(0, 0, 0, 0)
        self._smart_tol = QDoubleSpinBox()
        self._smart_tol.setRange(0.0, 100.0)
        self._smart_tol.setDecimals(1)
        self._smart_tol.setValue(0.0)
        self._smart_tol.setSuffix(" %")
        self._smart_tol.setSpecialValueText("auto (Otsu)")
        self._smart_tol.setToolTip("Paint voxels within this % of the window of the centre's intensity; "
                                   "auto: split the brush's voxels in two by Otsu, keep the centre's side.")
        g.addWidget(QLabel("Tolerance ±"), 0, 0)
        g.addWidget(self._smart_tol, 0, 1)
        lay.addWidget(self._smart_box)

        for box in (self._flood_box, self._vessel_common, self._vessel_box, self._tracer_box, self._smart_box):
            box.hide()
        return host

    def _remember_grow_3d(self, on: bool) -> None:
        if self._extra in self._grow3d:
            self._grow3d[self._extra] = bool(on)

    def intensity_window(self) -> tuple[float, float]:
        """The intensity range relative settings are read against: the box's
        (Box card, when asked) or the reference's display window."""
        ref = self.reference()
        if ref is None:
            return 0.0, 1.0
        box_range = self._box_intensity_range(ref)
        if box_range is not None:
            return box_range
        lo, hi = (float(v) for v in getattr(ref, "contrast_limits", (0.0, 1.0)))
        return lo, hi

    def wand_options(self, layer: Any | None = None) -> dict[str, Any]:
        """The wand's settings, as :func:`~nvitk.gui.labels.editing.wand_region` takes them."""
        from nvitk.gui.core.spatial import layer_spacing

        tol = float(self._wand_tol.value())
        if self._wand_tol_mode.currentData() == "percent":
            lo, hi = self.intensity_window()
            tol = tol / 100.0 * abs(hi - lo)
        layer = self.target() if layer is None else layer
        spacing = layer_spacing(layer) if layer is not None else None
        return {
            "tolerance": tol,
            "smooth_sigma": float(self._wand_smooth.value()),
            "seed_radius": int(self._wand_seed.value()),
            "full_connectivity": bool(self._wand_conn.currentData()),
            "max_distance_mm": float(self._wand_maxd.value()),
            "spacing": spacing,
            "fill_holes": bool(self._wand_holes.isChecked()),
            "close_radius": int(self._wand_close.value()),
        }

    def region_options(self, layer: Any, tool: str, free: Sequence[int]) -> dict[str, Any]:
        """Everything a region tool (:class:`~nvitk.gui.labels.region_tools.RegionStroke`)
        reads: the shared growing options, the tool's own, the setting Alt+drag tunes
        (``tune``), and the voxel spacing of the region's axes *free*."""
        from nvitk.gui.core.spatial import layer_spacing

        spacing_all = layer_spacing(layer) if layer is not None else None
        spacing = ([float(spacing_all[d]) for d in free]
                   if spacing_all is not None and len(spacing_all) >= max(free, default=-1) + 1 else None)
        lo, hi = self.intensity_window()
        window = abs(hi - lo) or 1.0
        wand = self.wand_options(layer)
        tune = {"wand": wand["tolerance"], "flood": float(self._flood_k.value()),
                "vessel": float(self._vessel_frac.value())}.get(tool, 0.0)
        vessel_tol = float(self._vessel_tol.value())
        return {
            "spacing": spacing,
            "smooth_sigma": wand["smooth_sigma"],
            "full_connectivity": wand["full_connectivity"],
            "max_distance_mm": wand["max_distance_mm"],
            "fill_holes": wand["fill_holes"],
            "close_radius": wand["close_radius"],
            "seed_radius": wand["seed_radius"],
            "tune": tune,
            "iterations": int(self._flood_rounds.value()),
            "radii_mm": (float(self._vessel_r0.value()), max(float(self._vessel_r1.value()),
                                                             float(self._vessel_r0.value()))),
            "bright": bool(self._vessel_bright.currentData()),
            "tolerance": vessel_tol / 100.0 * window if vessel_tol > 0 else None,
            "tube_radius_mm": float(self._tracer_radius.value()),
        }

    def tune_step(self, tool: str) -> float:
        """How far Alt+drag moves a tool's tuned setting per screen pixel."""
        lo, hi = self.intensity_window()
        return {"wand": 0.005 * (abs(hi - lo) or 1.0), "flood": 0.02, "vessel": -0.002}.get(tool, 0.0)

    def clamp_tune(self, tool: str, value: float) -> float:
        lo, hi = {"wand": (0.0, 1e12), "flood": (0.1, 10.0), "vessel": (0.01, 1.0)}.get(tool, (-1e12, 1e12))
        return float(min(max(value, lo), hi))

    def store_tune(self, tool: str, value: float) -> None:
        """Keep a setting tuned with Alt+drag in its box."""
        if tool == "wand":
            if self._wand_tol_mode.currentData() == "percent":
                lo, hi = self.intensity_window()
                value = value / (abs(hi - lo) or 1.0) * 100.0
            self._wand_tol.setValue(float(value))
        elif tool == "flood":
            self._flood_k.setValue(float(value))
        elif tool == "vessel":
            self._vessel_frac.setValue(float(value))

    def smart_tolerance(self) -> float | None:
        """The smart brush's tolerance in image units (``None``: Otsu)."""
        pct = float(self._smart_tol.value())
        if pct <= 0:
            return None
        lo, hi = self.intensity_window()
        return pct / 100.0 * (abs(hi - lo) or 1.0)

    def _build_area_card(self) -> Card:
        card = Card("Editable area")
        self._range_on = QCheckBox("Only where the reference is between")
        self._range_on.setToolTip(
            "Brush, bucket and tools only paint voxels whose reference intensity lies in "
            "the range (erasing is never restricted)."
        )
        self._range_lo = QDoubleSpinBox()
        self._range_hi = QDoubleSpinBox()
        for spin in (self._range_lo, self._range_hi):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(2)
            spin.valueChanged.connect(lambda _v: self._apply_guard())
        self._range_hi.setValue(1000.0)
        from_view = _button("From view", "The reference's current display window (the box's intensity "
                            "range, with Intensity range from the box).", self._range_from_view)
        self._from_view_buttons = [from_view]
        self._range_on.toggled.connect(lambda _v: self._apply_guard())
        card.add(self._range_on)
        card.add_layout(_row(self._range_lo, "–", self._range_hi, from_view))
        self._inside_on = QCheckBox("Only inside")
        self._inside_on.setToolTip("Paint only where this mask is non-zero.")
        self._inside_combo = QComboBox()
        self._inside_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._inside_on.toggled.connect(lambda _v: self._on_inside_changed())
        self._inside_combo.activated.connect(lambda _i: self._on_inside_changed())
        card.add_layout(_row(self._inside_on, self._inside_combo, stretch_last=True))
        # Which of the mask's labels make the area (all, until some are unticked).
        self._inside_box = QWidget()
        box = QVBoxLayout(self._inside_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        self._inside_ids = QListWidget()
        self._inside_ids.setMaximumHeight(130)
        self._inside_ids.setToolTip("Tick the labels of the mask to paint inside; the others are off limits.")
        self._inside_ids.itemChanged.connect(self._on_inside_ticked)
        self._inside_count = QLabel("")
        self._inside_count.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        box.addWidget(self._inside_ids)
        box.addLayout(_row(self._inside_count,
                           _button("All", "Every label of the mask.", lambda: self._tick_inside(True)),
                           _button("None", "No label (nothing is editable).", lambda: self._tick_inside(False))))
        card.add(self._inside_box)
        self._inside_box.hide()
        #: Labels unticked per mask layer (new labels start ticked).
        self._inside_off: dict[str, set[int]] = {}
        self._inside_cache: tuple[Any, Any] | None = None
        return card

    def _build_box_card(self) -> Card:
        """The box: drawn on the canvas or in the orthogonal views, or typed; crop to
        it, read intensity ranges in it, keep edits inside it."""
        card = Card("Box")
        card.setToolTip("A box on the volume: crop to it, take the tools' intensity ranges from it, keep "
                        "edits inside it.")
        self._box_draw = QToolButton()
        self._box_draw.setText("Draw box")
        self._box_draw.setCheckable(True)
        self._box_draw.setStyleSheet(_BUTTON_STYLE)
        self._box_draw.setToolTip("Drag a rectangle on the canvas (2D) or in an orthogonal view: it sets the box "
                                  "on those two axes; a drag in another view sets the third.")
        self._box_draw.toggled.connect(self._arm_box)
        self._box_draw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card.add_layout(_row(self._box_draw,
                             _button("Around label", "The box around the active label, 5 voxels more each way.",
                                     self._box_fit_label),
                             _button("Clear", "No box.", self._box_clear)))
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACE_TIGHT)
        self._box_axis_labels: list[QLabel] = []
        self._box_dashes: list[QLabel] = []
        self._box_spins: list[tuple[QSpinBox, QSpinBox]] = []
        for k in range(3):
            label = QLabel(f"Axis {k}")
            dash = QLabel("–")
            lo, hi = QSpinBox(), QSpinBox()
            for spin in (lo, hi):
                spin.setRange(0, 100000)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda _v: self._box_from_spins())
            lo.setToolTip("First voxel in the box.")
            hi.setToolTip("Last voxel in the box.")
            grid.addWidget(label, k, 0)
            grid.addWidget(lo, k, 1)
            grid.addWidget(dash, k, 2)
            grid.addWidget(hi, k, 3)
            self._box_axis_labels.append(label)
            self._box_dashes.append(dash)
            self._box_spins.append((lo, hi))
        card.add_layout(grid)
        self._box_depth_n = QSpinBox()
        self._box_depth_n.setRange(0, 10000)
        self._box_depth_n.setValue(10)
        self._box_depth_n.setPrefix("± ")
        self._box_depth_n.setToolTip("Slices each side of the one on screen.")
        card.add_layout(_row("Depth", self._box_depth_n,
                             _button("Set", "The box's depth: the slice on screen ± this many.", self._box_depth),
                             _button("Full", "The box's depth: the whole volume.", self._box_full_depth)))
        self._box_range = QCheckBox("Intensity range from the box")
        self._box_range.setToolTip("The wand's and the smart brush's % of the window, the vessel tools' "
                                   "intensity bound and the From… buttons read the 1–99 % intensity range "
                                   "inside the box, not the display window.")
        self._box_range.toggled.connect(self._on_box_range)
        self._box_keep = QCheckBox("Keep edits inside the box")
        self._box_keep.setToolTip("Every tool and the brush write only inside the box; the region tools grow "
                                  "only there (faster on a big volume).")
        self._box_keep.toggled.connect(lambda _v: self._apply_guard())
        card.add(self._box_range)
        card.add(self._box_keep)
        self._crop_ref = QCheckBox("reference")
        self._crop_labels = QCheckBox("labels")
        self._crop_all = QCheckBox("all on its grid")
        self._crop_ref.setChecked(True)
        self._crop_labels.setChecked(True)
        self._crop_all.setToolTip("Every image and label layer on the box's grid.")
        card.add_layout(_row("Crop", self._crop_ref, self._crop_labels, self._crop_all,
                             _button("Crop", "New layers cut to the box, placed where they came from (their "
                                     "origin, and the affine they are saved with, move with the box).",
                                     self.crop_box)))
        self._box_info = QLabel("No box.")
        self._box_info.setStyleSheet(f"color: {COLOR_MUTED};")
        card.add(self._box_info)
        self._box_intensity_cache: tuple[Any, Any] | None = None
        self.box().changed.connect(self._on_box_changed)
        self._on_box_changed()
        return card

    def _build_refine_card(self) -> Card:
        card = Card("Refine the active label")
        self._scope_slice = QRadioButton("Slice on screen")
        self._scope_volume = QRadioButton("Whole volume")
        self._scope_slice.setChecked(True)
        self._scope_volume.setToolTip("The whole volume (for a 3D+t layer, the frame on screen).")
        card.add_layout(_row(self._scope_slice, self._scope_volume))
        self._radius = QSpinBox()
        self._radius.setRange(1, 50)
        self._radius.setValue(1)
        self._radius.setToolTip("Voxels grown, shrunk or smoothed over.")
        self._min_size = QSpinBox()
        self._min_size.setRange(1, 10_000_000)
        self._min_size.setValue(50)
        self._min_size.setToolTip("Pieces smaller than this many voxels are dropped.")
        grid = QGridLayout()
        grid.setSpacing(SPACE_TIGHT)
        actions = (
            ("Grow", "Dilate the label by the radius.", lambda: self._refine("grow")),
            ("Shrink", "Erode the label by the radius.", lambda: self._refine("shrink")),
            ("Smooth", "Round it off: remove spurs, fill dents of about the radius.", lambda: self._refine("smooth")),
            ("Fill holes", "Fill the holes the label encloses.", lambda: self._refine("holes")),
            ("Keep largest", "Keep only its largest connected piece.", lambda: self._refine("largest")),
            ("Remove islands", "Drop the pieces smaller than the size.", lambda: self._refine("islands")),
        )
        for i, (text, tip, slot) in enumerate(actions):
            grid.addWidget(_button(text, tip, slot), i // 2, i % 2)
        card.add_layout(grid)
        card.add_layout(_row("Radius", self._radius, "Min size", self._min_size))

        self._thr_lo = QDoubleSpinBox()
        self._thr_hi = QDoubleSpinBox()
        for spin in (self._thr_lo, self._thr_hi):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(2)
        self._thr_hi.setValue(1000.0)
        thr = _button("Threshold fill", "Give the active label to every voxel whose reference "
                      "intensity is in the range (editable area and preserve apply).", self.threshold)
        thr_view = _button("From view", "The reference's current display window (the box's intensity "
                           "range, with Intensity range from the box).", self._thr_from_view)
        self._from_view_buttons.append(thr_view)
        card.add_layout(_row(self._thr_lo, "–", self._thr_hi))
        card.add_layout(_row(thr_view, thr))
        return card

    def _build_slices_card(self) -> Card:
        card = Card("Between slices")
        self._interp_axis = QComboBox()
        self._interp_axis.addItem("Across the view", "view")
        self._interp_axis.addItem("Auto (sparsest axis)", "auto")
        self._interp_method = QComboBox()
        self._interp_method.addItem("Shape", "shape")
        self._interp_method.addItem("Nearest", "nearest")
        self._interp_method.setToolTip("Shape: morph one outline into the next. Nearest: copy the closest drawn slice.")
        interp = _button("Interpolate", "Fill the slices between the ones drawn for the active label.", self.interpolate)
        card.add_layout(_row(self._interp_axis, self._interp_method))
        card.add(interp)
        self._copy_move = QCheckBox("Then show the slice copied to")
        self._copy_move.setToolTip("After copying, show the slice copied to (draw, correct, copy on).")
        self._copy_move.setChecked(True)
        prev_b = _button("◀ Previous", "Copy the active label's outline on this slice to the previous one.",
                         lambda: self.copy_slice(-1))
        next_b = _button("Next ▶", "Copy the active label's outline on this slice to the next one.",
                         lambda: self.copy_slice(+1))
        card.add_layout(_row("Copy slice to", prev_b, next_b))
        card.add(self._copy_move)
        return card

    def _build_manage_card(self) -> Card:
        card = Card("Manage labels")
        self._merge_combo = QComboBox()
        self._merge_combo.setToolTip("The label merged into the active one.")
        self._merge_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        merge = _button("Merge in", "The chosen label's voxels become the active label.", self.merge)
        card.add_layout(_row(self._merge_combo, merge))
        grid = QGridLayout()
        grid.setSpacing(SPACE_TIGHT)
        grid.addWidget(_button("Split pieces", "Each connected piece of the active label gets its "
                               "own id (the largest keeps it).", self.split), 0, 0)
        grid.addWidget(_button("Renumber 1…N", "Renumber the labels consecutively, in order.", self.renumber), 0, 1)
        grid.addWidget(_button("Delete label", "Clear the active label (Ctrl+Z restores it).", self.delete_active), 1, 0)
        card.add_layout(grid)
        return card

    # ── layers ────────────────────────────────────────────────────────────────

    def target(self) -> Any | None:
        """The label layer being edited."""
        layer = self._layer_ref()
        if layer is not None and layer in self._viewer.layers:
            return layer
        return None

    def reference(self) -> Any | None:
        """The reference image (same grid as the target), if any."""
        name = self._ref_combo.currentData()
        if not name or name not in self._viewer.layers:
            return None
        layer = self._viewer.layers[name]
        target = self.target()
        if target is not None and tuple(layer.data.shape) != tuple(target.data.shape):
            return None
        return layer

    def _visible_image_for(self, layer: Any) -> Any | None:
        """The image a 3D click on *layer* is read from: the reference, else the
        top visible image on the same grid."""
        ref = self.reference() if self.target() is layer else None
        if ref is not None:
            return ref
        for other in reversed(self._image_layers(tuple(layer.data.shape))):
            if other.visible:
                return other
        return None

    def _brush_3d_fallback(self, layer: Any, event: Any) -> Any:
        """Where a brush stroke on the 3D canvas lands when its ray meets no label
        (Napari's rule): the voxel the image shows under the cursor."""
        if layer not in self._viewer.layers:
            return None
        image = self._visible_image_for(layer)
        if image is None:
            return None
        return E.pick_visible_voxel(image, event.position, getattr(event, "view_direction", None))

    def _label_layers(self) -> list[Any]:
        out = []
        for layer in self._viewer.layers:
            kind = type(layer).__name__
            if kind == "Labels" or (kind == "Image" and getattr(layer, "_nvitk_label_like", None) is True):
                out.append(layer)
        return out

    def _image_layers(self, shape: tuple[int, ...] | None) -> list[Any]:
        out = []
        for layer in self._viewer.layers:
            if type(layer).__name__ != "Image" or getattr(layer, "_nvitk_label_like", None) is True:
                continue
            if getattr(layer, "rgb", False) or getattr(layer, "multiscale", False):
                continue
            if shape is not None and tuple(layer.data.shape) != tuple(shape):
                continue
            out.append(layer)
        return out

    def _refresh_layer_combos(self) -> None:
        target = self.target()
        self._syncing = True
        try:
            self._layer_combo.clear()
            for layer in self._label_layers():
                suffix = "" if type(layer).__name__ == "Labels" else "  (image mask)"
                self._layer_combo.addItem(layer.name + suffix, layer.name)
            if target is not None:
                self._layer_combo.setCurrentIndex(max(self._layer_combo.findData(target.name), 0))
            shape = tuple(target.data.shape) if target is not None else None
            current = self._ref_combo.currentData()
            self._ref_combo.clear()
            self._ref_combo.addItem(_NONE, "")
            for layer in self._image_layers(shape):
                self._ref_combo.addItem(layer.name, layer.name)
            idx = self._ref_combo.findData(current) if current else -1
            if idx < 0 and self._ref_combo.count() > 1:
                idx = 1
            self._ref_combo.setCurrentIndex(max(idx, 0))
            inside = self._inside_combo.currentData()
            self._inside_combo.clear()
            for layer in self._label_layers():
                if layer is not target and shape is not None and tuple(layer.data.shape) == shape:
                    self._inside_combo.addItem(layer.name, layer.name)
            idx = self._inside_combo.findData(inside) if inside else -1
            self._inside_combo.setCurrentIndex(max(idx, 0))
        finally:
            self._syncing = False
        if target is None:
            labels = self._label_layers()
            if labels:
                self.bind(labels[-1])
        self._labels_controls.set_layer(self.target())
        self._image_controls.set_layer(self.reference())
        self._refresh_inside_ids()
        self._on_box_changed()  # the box's axes follow the layer (and apply the guard)

    def _on_layer_combo(self, _index: int) -> None:
        name = self._layer_combo.currentData()
        if name and name in self._viewer.layers:
            self.bind(self._viewer.layers[name])

    def _on_active_changed(self, _event: Any = None) -> None:
        layer = self._viewer.layers.selection.active
        if layer is None:
            return
        if type(layer).__name__ == "Labels" or (type(layer).__name__ == "Image" and is_label_like_layer(layer)):
            if layer is not self.target():
                self.bind(layer)
        elif type(layer).__name__ == "Image":
            target = self.target()
            if target is not None and tuple(layer.data.shape) == tuple(target.data.shape):
                idx = self._ref_combo.findData(layer.name)
                if idx >= 0:
                    self._ref_combo.setCurrentIndex(idx)
                    self._on_reference_changed()

    def _on_layer_removed(self, event: Any) -> None:
        if getattr(event, "value", None) is self._layer_ref():
            self.bind(None)

    def _on_reference_changed(self) -> None:
        self._image_controls.set_layer(self.reference())
        # Ranges still at their defaults start from the new reference's window.
        if (self._range_lo.value(), self._range_hi.value()) == (0.0, 1000.0):
            self._range_from_view()
        if (self._thr_lo.value(), self._thr_hi.value()) == (0.0, 1000.0):
            self._thr_from_view()
        self._apply_guard()

    def bind(self, layer: Any | None) -> None:
        """Edit *layer* (``None``: nothing)."""
        old = self._layer_ref()
        if old is layer:
            return
        for emitter, slot in self._connections:
            try:
                emitter.disconnect(slot)
            except Exception:  # noqa: BLE001
                pass
        self._connections = []
        if old is not None:
            guard = E.paint_guard(old)
            if guard is not None:
                guard.intensity = guard.inside = None
        self._layer_ref = weakref.ref(layer) if layer is not None else (lambda: None)
        self._set_wand(False)
        if layer is not None:
            events = layer.events
            for name, slot in (
                ("mode", self._sync_from_layer),
                ("selected_label", self._sync_from_layer),
                ("brush_size", self._sync_from_layer),
                ("preserve_labels", self._sync_from_layer),
                ("n_edit_dimensions", self._sync_from_layer),
                ("contiguous", self._sync_from_layer),
                ("contour", self._sync_from_layer),
                ("opacity", self._sync_from_layer),
                ("paint", self._on_paint),
                ("data", self._on_data),
                ("name", lambda _e=None: self._combo_timer.start()),
            ):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    emitter.connect(slot)
                    self._connections.append((emitter, slot))
            E.announce_history_loads(layer)
        self._refresh_layer_combos()
        self._sync_from_layer()
        self._refresh_vocabulary()
        self._refresh_label_list()
        self.tool_changed.emit()

    # ── syncing with the layer ───────────────────────────────────────────────

    def _set_layer(self, attr: str, value: Any) -> None:
        layer = self.target()
        if self._syncing or layer is None or not hasattr(layer, attr):
            return
        try:
            setattr(layer, attr, value)
        except Exception as exc:  # noqa: BLE001
            self._say(f"Could not set {attr}: {exc}", error=True)

    def _sync_from_layer(self, _event: Any = None) -> None:
        layer = self.target()
        labels = layer is not None and E.is_labels(layer)
        self._convert_row.setVisible(layer is not None and not labels)
        self._syncing = True
        try:
            if labels and self._erase_restore is None:  # not mid right-drag: the tool is only lent
                mode = str(getattr(layer.mode, "value", layer.mode))
                if mode == "polygon":
                    # Napari's own polygon (its controls, its key): the tab's instead —
                    # it places the corners right whatever the view, and works in the
                    # orthogonal views too.
                    QTimer.singleShot(0, lambda: self.set_tool("polygon"))
                elif self._extra is None or mode not in ("pan_zoom",):
                    if self._extra is not None and mode != "pan_zoom":
                        self._set_extra(None)
                    button = self._tool_buttons.get(mode)
                    if button is not None:
                        button.setChecked(True)
                self._id_spin.setValue(int(layer.selected_label))
                self._brush_spin.setValue(int(round(float(layer.brush_size))))
                self._preserve.setChecked(bool(layer.preserve_labels))
                self._brush3d.setChecked(int(layer.n_edit_dimensions) >= 3)
                self._contiguous.setChecked(bool(layer.contiguous))
                self._contour.setChecked(int(layer.contour) > 0)
            if layer is not None:
                self._opacity.setValue(int(round(float(layer.opacity) * 100)))
        finally:
            self._syncing = False
        for button in self._tool_buttons.values():
            button.setEnabled(labels)
        self._update_active_label_view()

    def _update_active_label_view(self) -> None:
        layer = self.target()
        lid = self.label_id()
        if layer is None:
            self._swatch.setStyleSheet(_swatch_style(QColor(0, 0, 0, 0)))
            self._stats.setText("No label layer — make one with New…")
            return
        try:
            colour = _rgba_qcolor(get_label_color(layer, lid)) if lid else QColor(0, 0, 0, 0)
        except Exception:  # noqa: BLE001
            colour = QColor(200, 200, 200)
        self._swatch.setStyleSheet(_swatch_style(colour))
        if not self._name_edit.hasFocus():
            self._name_edit.setText((label_name(layer, lid) or "") if lid else "")
        count = self._counts.get(lid, 0)
        if lid == 0:
            self._stats.setText("Label 0 is the background: painting with it erases.")
        else:
            vol = count * E.voxel_volume_mm3(layer)
            self._stats.setText(f"{count:,} voxels · {vol:,.1f} mm³ ({vol / 1000.0:,.2f} mL)")
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setSelected(int(item.data(Qt.ItemDataRole.UserRole)) == lid)

    def _on_paint(self, _event: Any = None) -> None:
        self._keep_active_visible()
        self._count_timer.start()

    def _on_data(self, _event: Any = None) -> None:
        self._count_timer.start()

    def _on_names_changed(self, layer: Any) -> None:
        if layer is self.target():
            self._rebuild_list()
        if layer is self._inside_layer():
            self._refresh_inside_ids()

    def _on_labels_changed(self, layer: Any) -> None:
        if layer is self.target():
            self._count_timer.start()
        if layer is self._inside_layer():
            self._inside_cache = None
            self._refresh_inside_ids()
            self._apply_guard()

    def _refresh_vocabulary(self) -> None:
        layer = self.target()
        key = layer_schema_key(layer) if layer is not None else None
        schema = get_schema(key) if key else None
        names = sorted({str(n) for n in (schema.id_to_name.values() if schema else [])})
        self._completer.setModel(QStringListModel(names, self._completer))

    def _refresh_label_list(self) -> None:
        """Count every label's voxels again (a scan of the volume), then list them."""
        layer = self.target()
        try:
            self._counts = E.label_counts(layer) if layer is not None else {}
        except Exception:  # noqa: BLE001 — a layer mid-removal
            self._counts = {}
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        """List the labels from the last count (names and colours read anew)."""
        layer = self.target()
        self._list.clear()
        self._merge_combo.clear()
        if layer is None:
            self._fit_list()
            self._update_active_label_view()
            return
        lid_now = self.label_id()
        for lid, count in sorted(self._counts.items()):
            name = label_name(layer, lid)
            text = f"{name} ({lid})" if name else f"Label {lid}"
            item = QListWidgetItem(f"{text} — {count:,} vox")
            item.setData(Qt.ItemDataRole.UserRole, lid)
            try:
                item.setIcon(_swatch_icon(_rgba_qcolor(get_label_color(layer, lid))))
            except Exception:  # noqa: BLE001
                pass
            self._list.addItem(item)
            if lid != lid_now:
                self._merge_combo.addItem(text, lid)
        self._fit_list()
        self._update_active_label_view()

    def _fit_list(self) -> None:
        """The list as tall as its rows, between a few and a screenful-ish."""
        lo, hi = _LIST_ROWS
        row = self._list.sizeHintForRow(0) if self._list.count() else 20
        rows = min(max(self._list.count(), lo), hi)
        self._list.setFixedHeight(int(rows * max(row, 18) + 2 * self._list.frameWidth() + 4))

    # ── the active label ─────────────────────────────────────────────────────

    def label_id(self) -> int:
        return int(self._id_spin.value())

    def select_label(self, lid: int) -> None:
        """Make *lid* the label drawn with."""
        layer = self.target()
        self._id_spin.setValue(int(lid))
        if layer is not None and E.is_labels(layer):
            layer.selected_label = int(lid)
        self._keep_active_visible()
        self._rebuild_list()

    def _on_id_changed(self, value: int) -> None:
        if self._syncing:
            self._update_active_label_view()
            return
        layer = self.target()
        if layer is not None and E.is_labels(layer) and int(layer.selected_label) != int(value):
            layer.selected_label = int(value)
        self._rebuild_list()

    def new_label(self) -> None:
        layer = self.target()
        if layer is None:
            self._say("No label layer.", error=True)
            return
        self.select_label(E.next_free_label(layer))
        self._say(f"Drawing with the new label {self.label_id()}.")

    def _on_name_entered(self) -> None:
        layer = self.target()
        lid = self.label_id()
        if layer is None or lid == 0:
            return
        self._rename(layer, lid, self._name_edit.text())
        self._name_edit.clearFocus()

    def _on_vocabulary_name(self, text: str) -> None:
        """A vocabulary name was picked: its id becomes the active label."""
        layer = self.target()
        key = layer_schema_key(layer) if layer is not None else None
        schema = get_schema(key) if key else None
        if schema is None:
            return
        for lid, name in schema.id_to_name.items():
            if str(name) == text:
                self.select_label(int(lid))
                self._say(f"Drawing with {text} ({lid}).")
                return

    def _rename(self, layer: Any, lid: int, text: str) -> None:
        own = custom_label_names(layer).get(lid)
        default = label_name(layer, lid) if not own else None
        text = text.strip()
        if default and text == default:
            return
        if set_label_name(layer, lid, text):
            label_filter_hub().names_changed.emit(layer)

    def _rename_from_list(self, lid: int) -> None:
        self.select_label(lid)
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def edit_colour(self) -> None:
        layer = self.target()
        lid = self.label_id()
        if layer is None or lid == 0:
            return
        if not E.is_labels(layer):
            self._say("Convert the mask to a Labels layer to change its colours.", error=True)
            return
        current = _rgba_qcolor(get_label_color(layer, lid))
        chosen = QColorDialog.getColor(current, self, f"Label {lid} colour")
        if not chosen.isValid():
            return
        visible = stored_visible_ids(layer)
        set_label_color(layer, lid, [chosen.redF(), chosen.greenF(), chosen.blueF(), chosen.alphaF()],
                        selected_ids=None if visible is None else list(visible))
        label_filter_hub().changed.emit(layer)
        self._rebuild_list()

    def _keep_active_visible(self) -> None:
        """A label filter on the layer must not hide the label being drawn."""
        layer = self.target()
        lid = self.label_id()
        if layer is None or lid == 0:
            return
        visible = stored_visible_ids(layer)
        if visible is None or lid in visible:
            return
        present = sorted(set(layer_label_ids(layer)) | {lid})
        set_layer_visible_ids(layer, sorted(set(visible) | {lid}), self._viewer, present=present)

    def go_to(self) -> None:
        layer = self.target()
        if layer is None:
            return
        if not E.go_to_label(self._viewer, layer, self.label_id()):
            self._say(f"Label {self.label_id()} is not in “{layer.name}”.", error=True)

    def edit(self, layer: Any, lid: int) -> None:
        """Bring this tab up on *layer*, drawing with *lid* (from the layer list's menu)."""
        self.bind(layer)
        self.select_label(lid)
        manager = getattr(self._viewer, "_nvitk_panel_manager", None)
        dock = getattr(self._viewer, "_nvitk_labeling_dock", None)
        if manager is not None and dock is not None:
            manager.show_dock(dock)

    def _list_menu(self, pos: Any) -> None:
        item = self._list.itemAt(pos)
        layer = self.target()
        if item is None or layer is None:
            return
        lid = int(item.data(Qt.ItemDataRole.UserRole))
        ids = layer_label_ids(layer)
        menu = QMenu(self)
        menu.addAction("Draw with it", lambda: self.select_label(lid))
        menu.addAction("Rename…", lambda: self._rename_from_list(lid))
        menu.addAction("Show only this", lambda: set_layer_visible_ids(layer, [lid], self._viewer, present=ids))
        menu.addAction("Show all", lambda: set_layer_visible_ids(layer, ids, self._viewer, present=ids))
        menu.addAction("Go to", lambda: E.go_to_label(self._viewer, layer, lid))
        if lid != self.label_id() and self.label_id() != 0:
            menu.addAction(f"Merge into label {self.label_id()}", lambda: self._merge(lid, self.label_id()))
        menu.addSeparator()
        menu.addAction(f"Delete label {lid}", lambda: self._delete(lid))
        menu.exec(self._list.viewport().mapToGlobal(pos))

    # ── drawing ──────────────────────────────────────────────────────────────

    def set_tool(self, mode: str) -> None:
        """Pick a Draw tool: a Napari Labels mode, or one of the tab's own (the wand,
        the flood and vessel tools, the smart brush, the polygon)."""
        layer = self.target()
        if layer is None or not E.is_labels(layer):
            return
        # The mode first: Napari resets the layer's camera panning with it, and the
        # tab's tools then set it for themselves.
        layer.mode = "pan_zoom" if mode in EXTRA_TOOLS else mode
        self._set_extra(mode if mode in EXTRA_TOOLS else None)
        self._tool_buttons[mode].setChecked(True)
        try:
            self._viewer.layers.selection.active = layer
        except Exception:  # noqa: BLE001
            pass

    def _set_wand(self, on: bool) -> None:
        self._set_extra("wand" if on else None)

    def _set_extra(self, tool: str | None) -> None:
        changed = self._extra != tool
        if self._extra == "polygon" and tool != "polygon":
            self.cancel_polygon()
        if self._extra == "tracer" and tool != "tracer":
            self.tracer_cancel()
        self._extra = tool
        region = tool in REGION_TOOLS
        self._wand_box.setVisible(region)
        for widget in self._wand_tol_row:
            widget.setVisible(tool == "wand")
        if region:
            self._wand_3d.blockSignals(True)
            self._wand_3d.setChecked(self._grow3d.get(tool, False))
            self._wand_3d.blockSignals(False)
        self._flood_box.setVisible(tool == "flood")
        self._vessel_common.setVisible(tool in ("vessel", "tracer"))
        self._vessel_box.setVisible(tool == "vessel")
        self._tracer_box.setVisible(tool == "tracer")
        self._smart_box.setVisible(tool == "smart")
        self._poly_box.setVisible(tool == "polygon")
        self._tool_hint.setText(_TOOL_HINTS.get(tool or "", ""))
        self._tool_hint.setVisible(tool in _TOOL_HINTS)
        self._update_camera_lock()
        if changed:
            self.tool_changed.emit()

    def extra_tool(self) -> str | None:
        """The tab's own Draw tool in use (``None``: one of Napari's)."""
        return self._extra

    def wand_active(self) -> bool:
        """True while the magic wand is the Draw tool."""
        return self._extra == "wand"

    def polygon_active(self) -> bool:
        """True while the polygon is the Draw tool."""
        return self._extra == "polygon"

    def wand_in_3d(self) -> bool:
        """True when the region tools grow through the volume rather than in the plane clicked."""
        return bool(self._wand_3d.isChecked())

    def _history(self, which: str) -> None:
        layer = self.target()
        if layer is not None and E.is_labels(layer):
            getattr(layer, which)()
            self._count_timer.start()

    # ── the canvas ───────────────────────────────────────────────────────────

    def _update_camera_lock(self, _event: Any = None) -> None:
        """A drag draws (the region tools in 2D, the smart brush, drawing the box)
        instead of moving the camera; Ctrl+drag pans then.

        Set on the label layer, not the camera: Napari hands the camera the
        active layer's ``mouse_pan`` whenever the layer or its mode changes."""
        lock = (self._box_armed
                or self._extra == "smart"
                or (self._extra in REGION_TOOLS and not E.is_3d_view(self._viewer)))
        layer = self.target()
        if layer is not None and E.is_labels(layer):
            mode = str(getattr(layer.mode, "value", layer.mode))
            if mode == "pan_zoom":
                layer.mouse_pan = not lock
        elif self._box_armed:
            self._viewer.camera.mouse_pan = False

    def _pan_drag(self, event: Any) -> Any:
        """Ctrl+drag while a drawing drag holds the camera: pan by hand (2D)."""
        camera = self._viewer.camera
        last = [float(v) for v in event.pos[:2]]
        yield
        while event.type == "mouse_move":
            x, y = (float(v) for v in event.pos[:2])
            zoom = float(camera.zoom) or 1.0
            centre = list(camera.center)
            centre[-1] -= (x - last[0]) / zoom
            centre[-2] -= (y - last[1]) / zoom
            camera.center = tuple(centre)
            last = [x, y]
            yield

    def _click_only(self, event: Any) -> Any:
        """Follow a press to its release; ``self._clicked`` tells whether it moved."""
        start = tuple(event.position)
        self._clicked = True
        yield
        while event.type == "mouse_move":
            if any(abs(a - b) > 1e-6 for a, b in zip(event.position, start)):
                self._clicked = False
            yield

    def _event_seed(self, layer: Any, event: Any) -> tuple[int, ...] | None:
        """The voxel a click points at: under the cursor in 2D; in 3D, the one the
        image shows along the ray (the reference's rendering decides)."""
        if E.is_3d_view(self._viewer):
            image = self._visible_image_for(layer)
            if image is None:
                return None
            return E.pick_visible_voxel(image, event.position, getattr(event, "view_direction", None))
        return E.event_voxel(layer, event)

    def _canvas_callback(self, viewer: Any, event: Any) -> Any:
        """The tab's tools on the main canvas.

        Region tools (wand, flood, vessel): a drag adds a region at every step
        (2D; Shift takes out), a click in 3D grows from what the image shows,
        Alt+drag tunes the tool's setting live. Tracer and polygon: clicks. The
        smart brush: drags. Ctrl+drag pans wherever a drag draws. Drawing the
        box takes the drag before any of them.
        """
        if event.button != 1:
            return
        if self._box_armed:
            yield from self._box_drag(event)
            return
        extra = self._extra
        if extra is None:
            return
        mods = set(getattr(event, "modifiers", ()) or ())
        three_d = E.is_3d_view(viewer)
        if "Control" in mods and not three_d and (extra in REGION_TOOLS or extra == "smart"):
            yield from self._pan_drag(event)
            return
        if extra in REGION_TOOLS and "Alt" in mods:
            yield from self._tune_drag(event)
            return
        if extra in REGION_TOOLS and not three_d:
            yield from self._region_drag(event, remove="Shift" in mods)
            return
        if extra == "smart":
            yield from self._smart_drag(event)
            return
        yield from self._click_only(event)
        if not self._clicked:
            return
        if extra in REGION_TOOLS:
            self.region_click_event(event, remove="Shift" in mods)
        elif extra == "tracer":
            self._tracer_click_event(event)
        elif extra == "polygon":
            self._polygon_click(event)

    def _canvas_double_click(self, viewer: Any, event: Any) -> None:
        """A double-click closes the polygon being drawn, or fills the traced tube."""
        if self._extra == "polygon" and self._poly is not None:
            self.finish_polygon()
        elif self._extra == "tracer" and self._trace is not None and len(self._trace.points) >= 2:
            self.tracer_fill()

    # ── region tools ─────────────────────────────────────────────────────────

    def region_index(self, layer: Any, voxel: Sequence[int], free: Sequence[int]) -> tuple[Any, ...]:
        """Where a region tool works from *voxel*: the axes *free* whole (the plane
        clicked, or the volume — inside the box when edits are kept in it), the
        others fixed at the voxel."""
        free = {int(d) for d in free}
        index = tuple(slice(None) if d in free else int(voxel[d]) for d in range(layer.data.ndim))
        return self._limit_to_box(layer, index)

    def _region_free(self, layer: Any, *, three_d: bool) -> list[int]:
        """The canvas's region axes: the plane on screen, or every spatial axis."""
        if three_d or E.is_3d_view(self._viewer):
            t_ax = E.time_axis(layer)
            return [d for d in range(layer.data.ndim) if d != t_ax]
        return [int(d) for d in E.displayed_axes(self._viewer, layer)]

    def begin_region(self, layer: Any, voxel: Sequence[int], index: Sequence[Any], *,
                     tool: str | None = None, remove: bool = False) -> RegionStroke | None:
        """Start a stroke of the region tool (default: the one in use) at *voxel*,
        inside ``layer.data[index]`` — for the canvas and the orthogonal views."""
        tool = tool or self._extra
        try:
            E.announce_history_loads(layer)
            stroke = RegionStroke(self, layer, index, tool, remove=remove)
            if stroke.local(voxel) is None:
                raise ValueError("the click is outside the box (Keep edits inside the box is on)."
                                 if self._box_keep.isChecked() else "the click is outside the volume.")
            stroke.add(voxel)
        except Exception as exc:  # noqa: BLE001 — shown, not raised into Qt
            self._say(f"{tool.capitalize() if tool else 'Region'}: {exc}", error=True)
            return None
        return stroke

    def finish_region(self, stroke: RegionStroke | None, *, tuned: bool = False) -> None:
        """Write a region stroke (one undo step) and report it."""
        if stroke is None:
            return
        try:
            changed = stroke.commit()
        except Exception as exc:  # noqa: BLE001
            stroke.cancel()
            self._say(f"{stroke.tool.capitalize()}: {exc}", error=True)
            return
        if tuned:
            self.store_tune(stroke.tool, stroke.value)
        verb = "removed" if stroke.remove else "added"
        name = {"wand": "Wand", "flood": "Flood", "vessel": "Vessel flood"}[stroke.tool]
        extra = f", {TUNED[stroke.tool]} {stroke.value:.4g}" if tuned else ""
        self._say(f"{name}: {verb} {changed:,} voxel(s) (region {stroke.size:,}{extra}).")

    def _region_drag(self, event: Any, *, remove: bool) -> Any:
        """A drag on the 2D canvas: a region from every voxel reached."""
        layer = self.target()
        seed = E.event_voxel(layer, event) if layer is not None else None
        stroke = None
        if seed is not None:
            index = self.region_index(layer, seed, self._region_free(layer, three_d=self.wand_in_3d()))
            stroke = self.begin_region(layer, seed, index, remove=remove)
        yield
        while event.type == "mouse_move":
            if stroke is not None:
                voxel = E.event_voxel(layer, event)
                if voxel is not None:
                    try:
                        stroke.add(voxel)
                    except Exception as exc:  # noqa: BLE001
                        self._say(f"{stroke.tool.capitalize()}: {exc}", error=True)
            yield
        self.finish_region(stroke)

    def _tune_drag(self, event: Any) -> Any:
        """Alt+drag: one seed, the tool's setting following the mouse (right: more)."""
        layer = self.target()
        seed = self._event_seed(layer, event) if layer is not None else None
        stroke = None
        if seed is not None:
            three_d = E.is_3d_view(self._viewer) or self.wand_in_3d()
            index = self.region_index(layer, seed, self._region_free(layer, three_d=three_d))
            stroke = self.begin_region(layer, seed, index)
        camera = self._viewer.camera
        held = bool(camera.mouse_pan)
        camera.mouse_pan = False  # in 3D too: the drag tunes, it does not turn the view
        x0 = float(event.pos[0])
        last = stroke.value if stroke is not None else 0.0
        yield
        while event.type == "mouse_move":
            if stroke is not None:
                value = self.clamp_tune(stroke.tool, stroke.options["tune"]
                                        + (float(event.pos[0]) - x0) * self.tune_step(stroke.tool))
                if abs(value - last) > 1e-9:
                    try:
                        stroke.tune(value)
                        last = value
                        self._say(f"{TUNED[stroke.tool].capitalize()}: {value:.4g} — region {stroke.size:,} voxel(s)")
                    except Exception as exc:  # noqa: BLE001
                        self._say(str(exc), error=True)
            yield
        camera.mouse_pan = held
        self.finish_region(stroke, tuned=True)

    def region_click_event(self, event: Any, *, remove: bool = False) -> None:
        """A click with a region tool: one region (in 3D, from the voxel the image shows)."""
        layer = self.target()
        if layer is None:
            return
        if self.reference() is None:
            self._say("This tool reads a reference image: pick one under Layer.", error=True)
            return
        seed = self._event_seed(layer, event)
        if seed is None:
            self._say("The click missed the volume.", error=True)
            return
        index = self.region_index(layer, seed, self._region_free(layer, three_d=self.wand_in_3d()))
        self.finish_region(self.begin_region(layer, seed, index, remove=remove))

    def live_refresh(self, layer: Any, labels: Sequence[int] = ()) -> None:
        """Show voxels written outside the history (a stroke in progress) on the
        canvas and in the orthogonal views — once per event-loop turn."""
        self._live_pending = (weakref.ref(layer), tuple(int(v) for v in labels))
        self._live_timer.start()

    def _live_flush(self) -> None:
        pending, self._live_pending = self._live_pending, None
        layer = pending[0]() if pending else None
        if layer is None:
            return
        try:
            layer.refresh()
        except Exception:  # noqa: BLE001
            pass
        ortho = getattr(self._viewer, "_nvitk_ortho_panel", None)
        if ortho is not None:
            try:
                ortho.refresh_painted(layer, pending[1])
            except Exception:  # noqa: BLE001
                pass

    def after_write(self, layer: Any) -> None:
        """What follows an edit written by a stroke: the label lists, the counts."""
        invalidate_label_ids(layer)
        label_filter_hub().labels_changed.emit(layer)
        self._keep_active_visible()
        self._count_timer.start()

    # ── the smart brush ──────────────────────────────────────────────────────

    def _smart_dims(self, layer: Any) -> list[int]:
        if E.is_3d_view(self._viewer) or int(getattr(layer, "n_edit_dimensions", 2)) >= 3:
            t_ax = E.time_axis(layer)
            return [d for d in range(layer.data.ndim) if d != t_ax]
        return [int(d) for d in E.displayed_axes(self._viewer, layer)]

    def begin_smart(self, layer: Any, dims: Sequence[int]) -> SmartBrushStroke | None:
        try:
            E.announce_history_loads(layer)
            return SmartBrushStroke(self, layer, dims)
        except Exception as exc:  # noqa: BLE001
            self._say(f"Smart brush: {exc}", error=True)
            return None

    def finish_smart(self, stroke: SmartBrushStroke | None) -> None:
        if stroke is not None:
            n = stroke.finish()
            self._say(f"Smart brush: {n:,} voxel(s) painted with label {stroke.label}.")

    def _smart_drag(self, event: Any) -> Any:
        layer = self.target()
        seed = self._event_seed(layer, event) if layer is not None else None
        if seed is None:
            if E.is_3d_view(self._viewer):
                # Off the volume in 3D: this drag turns the camera, as without the tool.
                camera = self._viewer.camera
                held = bool(camera.mouse_pan)
                camera.mouse_pan = True
                yield
                while event.type == "mouse_move":
                    yield
                camera.mouse_pan = held
            return
        stroke = self.begin_smart(layer, self._smart_dims(layer))
        if stroke is not None:
            stroke.extend(seed)
        yield
        while event.type == "mouse_move":
            if stroke is not None:
                voxel = self._event_seed(layer, event)
                if voxel is not None:
                    stroke.extend(voxel)
            yield
        self.finish_smart(stroke)

    def _right_erase(self, viewer: Any, event: Any) -> Any:
        """A right-drag on the canvas erases with the brush, whatever the Draw tool;
        the tool comes back when the button is released. Napari's own eraser does
        the work (one stroke, one Ctrl+Z), so the brush size and 3D brush apply."""
        if getattr(event, "button", 1) != 2 or self._erase_restore is not None:
            return
        layer = self.target()
        if layer is None or not E.is_labels(layer) or viewer.layers.selection.active is not layer:
            return
        mode = str(getattr(layer.mode, "value", layer.mode))
        tool = self._extra or mode
        if self._extra is None and mode not in ("paint", "fill", "erase"):
            return  # pan / pick / transform: the right button keeps doing what it does
        self._erase_restore = tool
        if mode != "erase":
            layer.mode = "erase"
        yield
        while event.type == "mouse_move":
            yield
        # After Napari's eraser has closed its stroke (it runs after this callback).
        QTimer.singleShot(0, self._end_right_erase)

    def _end_right_erase(self) -> None:
        tool, self._erase_restore = self._erase_restore, None
        if tool is None:
            return
        layer = self.target()
        if layer is None or not E.is_labels(layer):
            return
        if tool in ("wand", "polygon") or tool in self._tool_buttons:
            self.set_tool(tool)
        self._count_timer.start()

    # ── the polygon on the canvas ────────────────────────────────────────────

    def _polygon_click(self, event: Any) -> None:
        layer = self.target()
        if layer is None:
            return
        if E.is_3d_view(self._viewer):
            self._say("Draw polygons in a 2D view, or in the orthogonal views.", error=True)
            return
        try:
            index = E.region(self._viewer, layer, "slice")
            position = [float(v) for v in layer.world_to_data(event.position)][-layer.data.ndim:]
        except Exception as exc:  # noqa: BLE001
            self._say(f"Polygon: {exc}", error=True)
            return
        dims = [d for d, ix in enumerate(index) if isinstance(ix, slice)]
        point = (position[dims[0]], position[dims[1]])
        poly = self._poly
        if poly is not None and (poly["layer"]() is not layer or poly["index"] != index):
            self.cancel_polygon()  # another slice or layer: start again there
            poly = None
        if poly is None:
            poly = self._poly = {"layer": weakref.ref(layer), "index": index, "dims": dims,
                                 "points": [], "preview": lambda: None}
        points = poly["points"]
        if points and abs(point[0] - points[-1][0]) < 0.5 and abs(point[1] - points[-1][1]) < 0.5:
            return  # the second click of a double-click
        if len(points) >= 3 and abs(point[0] - points[0][0]) <= 1.0 and abs(point[1] - points[0][1]) <= 1.0:
            self.finish_polygon()  # back on the first corner: close it
            return
        points.append(point)
        self._show_polygon_preview()
        self._poly_hint.setText(f"{len(points)} corner(s) — double-click or click the first one to fill.")

    def _show_polygon_preview(self) -> None:
        """The corners so far as a path on the canvas (a temporary Shapes layer)."""
        poly = self._poly
        layer = poly["layer"]() if poly else None
        if layer is None:
            return
        base = list(poly["index"])
        world = []
        for r, c in poly["points"] + poly["points"][:1]:
            point = [float(v) if not isinstance(v, slice) else 0.0 for v in base]
            point[poly["dims"][0]], point[poly["dims"][1]] = r, c
            world.append([float(v) for v in layer.data_to_world(point)])
        preview = poly["preview"]()
        try:
            if preview is None or preview not in self._viewer.layers:
                preview = self._viewer.add_shapes(
                    [world], shape_type="path", edge_color="#ffd34d", edge_width=1.0, name="✎ polygon (Labeling)",
                )
                preview._nvitk_internal = True
                poly["preview"] = weakref.ref(preview)
                self._viewer.layers.selection.active = layer
            else:
                preview.data = [world]
        except Exception:  # noqa: BLE001 — the preview is a convenience
            pass

    def finish_polygon(self) -> None:
        """Fill the polygon drawn on the canvas with the active label."""
        poly, self._poly = self._poly, None
        self._remove_polygon_preview(poly)
        if poly is None or len(poly["points"]) < 3:
            self._poly_hint.setText("Click the corners; double-click or click the first one to fill.")
            return
        rows = [p[0] for p in poly["points"]]
        cols = [p[1] for p in poly["points"]]
        self.fill_polygon(poly["index"], rows, cols)
        self._poly_hint.setText("Click the corners; double-click or click the first one to fill.")

    def cancel_polygon(self) -> None:
        """Drop the polygon being drawn."""
        poly, self._poly = self._poly, None
        self._remove_polygon_preview(poly)
        if hasattr(self, "_poly_hint"):
            self._poly_hint.setText("Click the corners; double-click or click the first one to fill.")

    def _remove_polygon_preview(self, poly: dict[str, Any] | None) -> None:
        preview = poly["preview"]() if poly else None
        if preview is not None and preview in self._viewer.layers:
            try:
                self._viewer.layers.remove(preview)
            except Exception:  # noqa: BLE001
                pass

    def fill_polygon(self, index: tuple[Any, ...], rows: list[float], cols: list[float]) -> None:
        """Give the active label to a polygon in the plane ``data[index]`` (its corners
        *rows* / *cols* along the plane's two axes, in order) — from the canvas or an
        orthogonal view. Preserve-labels and the editable area apply; one undo step."""

        def run() -> str:
            layer = self._editable_target()
            lid = self.label_id()
            if lid == 0:
                raise ValueError("Pick the label to draw (not the background).")
            work = layer.data[tuple(index)]
            mask = E.polygon_mask(work.shape, rows, cols) & self._allowed(layer, tuple(index), work)
            out = as_backend_array(work).copy()
            out[mask] = lid
            return f"Polygon: {self._write(layer, tuple(index), out):,} voxel(s) labelled {lid}."

        self._run("Polygon", run)

    def wand_at(self, event: Any, *, remove: bool = False) -> None:
        """The magic wand from a click on the canvas (2D or 3D)."""
        if self.reference() is None:
            self._say("The wand reads a reference image: pick one under Layer.", error=True)
            return
        extra, self._extra = self._extra, "wand"
        try:
            self.region_click_event(event, remove=remove)
        finally:
            self._extra = extra

    def wand_from(self, seed: tuple[int, ...], index: tuple[Any, ...], *, remove: bool = False,
                  tool: str = "wand") -> None:
        """A region tool (the magic wand by default) from the voxel *seed*, within
        ``data[index]`` (a plane or a volume) — one click on any view of the layer."""
        layer = self.target()
        if layer is None:
            return
        if self.reference() is None:
            self._say("The wand reads a reference image: pick one under Layer.", error=True)
            return
        self.finish_region(self.begin_region(layer, seed, self._limit_to_box(layer, index), tool=tool,
                                             remove=remove))

    # ── the vessel tracer ────────────────────────────────────────────────────

    def tracer_add(self, layer: Any, voxel: Sequence[int]) -> None:
        """Another point along the vessel (from the canvas or an orthogonal view)."""
        if self.reference() is None:
            self._say("The vessel tracer reads a reference image: pick one under Layer.", error=True)
            return
        try:
            trace = self._trace
            if trace is not None and trace.layer is not layer:
                self.tracer_cancel()
                trace = None
            if trace is None:
                index = self.region_index(layer, voxel, self._region_free(layer, three_d=True))
                trace = self._trace = VesselTrace(self, layer, index)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                trace.add(voxel)
            finally:
                QApplication.restoreOverrideCursor()
        except Exception as exc:  # noqa: BLE001
            self._say(f"Tracer: {exc}", error=True)
            return
        self._show_trace()

    def _tracer_click_event(self, event: Any) -> None:
        layer = self.target()
        if layer is None:
            return
        voxel = self._event_seed(layer, event)
        if voxel is None:
            self._say("The click missed the volume.", error=True)
            return
        self.tracer_add(layer, voxel)

    def tracer_undo(self) -> None:
        if self._trace is not None:
            self._trace.undo()
            if not self._trace.points:
                self.tracer_cancel()
                return
            self._show_trace()

    def tracer_cancel(self) -> None:
        self._trace = None
        self._show_trace()

    def tracer_fill(self) -> None:
        trace = self._trace
        if trace is None:
            self._say("Tracer: click points along the vessel first.", error=True)
            return

        def run() -> str:
            changed, radius = trace.fill()
            self._trace = None
            self._show_trace()
            how = "measured" if float(self._tracer_radius.value()) <= 0 else "fixed"
            return (f"Tracer: tube of {changed:,} voxel(s) labelled {self.label_id()} "
                    f"(radius {radius:.2f} mm, {how}).")

        self._run("Tracer", run)

    def _show_trace(self) -> None:
        """The points and the path so far: a Points layer on the canvas, lines in
        the orthogonal views."""
        trace = self._trace
        layer = trace.layer if trace is not None else None
        preview = self._trace_preview()
        ortho = getattr(self._viewer, "_nvitk_ortho_panel", None)
        if layer is None:
            if preview is not None:
                try:
                    self._viewer.layers.remove(preview)
                except Exception:  # noqa: BLE001
                    pass
            if ortho is not None:
                ortho._label_editor.show_trace(None, [], [])
            self._tracer_info.setText("No points yet.")
            return
        path = trace.path()
        clicked = set(trace.points)
        voxels = list(dict.fromkeys(list(path) + list(trace.points)))
        world = [[float(v) for v in layer.data_to_world(list(p))] for p in voxels]
        sizes = [3.0 if p in clicked else 1.0 for p in voxels]
        try:
            if preview is None:
                active = self._viewer.layers.selection.active
                preview = self._viewer.add_points(world, size=sizes, face_color="#ffd34d", border_width=0,
                                                  name=_TRACE_PREVIEW, out_of_slice_display=True)
                preview._nvitk_internal = True
                self._trace_ref = weakref.ref(preview)
                if active is not None and active in self._viewer.layers:
                    self._viewer.layers.selection.active = active
            else:
                preview.data = world
                preview.size = sizes
        except Exception:  # noqa: BLE001 — the preview is a convenience
            pass
        if ortho is not None:
            ortho._label_editor.show_trace(layer, trace.points, path)
        self._tracer_info.setText(f"{len(trace.points)} point(s), path of {len(path)} voxel(s) — double-click "
                                  "or Fill tube.")

    def _trace_preview(self) -> Any | None:
        preview = self._trace_ref()
        return preview if preview is not None and preview in self._viewer.layers else None

    # ── editable area ────────────────────────────────────────────────────────

    def _apply_guard(self) -> None:
        if self._syncing:
            return
        layer = self.target()
        if layer is None or not E.is_labels(layer):
            return
        guard = E.paint_guard(layer, create=True)
        ref = self.reference()
        if self._range_on.isChecked() and ref is not None:
            guard.intensity = ref.data
            guard.low, guard.high = float(self._range_lo.value()), float(self._range_hi.value())
        else:
            guard.intensity = None
        mask = None
        if self._inside_on.isChecked():
            name = self._inside_combo.currentData()
            if name and name in self._viewer.layers:
                other = self._viewer.layers[name]
                if tuple(other.data.shape) == tuple(layer.data.shape):
                    mask = self._inside_mask(other)
        guard.inside = mask
        box = self.box()
        guard.box = box.index(layer) if (self._box_keep.isChecked() and box.applies(layer)) else None
        if self._range_on.isChecked() and ref is None:
            self._say("The intensity range needs a reference image on the labels' grid.", error=True)

    # ── editable area: which labels of the mask ──────────────────────────────

    def _inside_layer(self) -> Any | None:
        name = self._inside_combo.currentData()
        return self._viewer.layers[name] if name and name in self._viewer.layers else None

    def _on_inside_changed(self) -> None:
        self._refresh_inside_ids()
        self._apply_guard()

    def _refresh_inside_ids(self) -> None:
        """List the mask's labels, ticked unless unticked before."""
        other = self._inside_layer() if self._inside_on.isChecked() else None
        self._inside_box.setVisible(other is not None)
        self._inside_ids.blockSignals(True)
        self._inside_ids.clear()
        if other is not None:
            off = self._inside_off.get(other.name, set())
            try:
                ids = [int(i) for i in layer_label_ids(other, max_labels=10_000) if int(i) != 0]
            except Exception:  # noqa: BLE001
                ids = []
            for lid in ids:
                name = label_name(other, lid)
                item = QListWidgetItem(f"{name} ({lid})" if name else f"Label {lid}")
                item.setData(Qt.ItemDataRole.UserRole, lid)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked if lid in off else Qt.CheckState.Checked)
                try:
                    item.setIcon(_swatch_icon(_rgba_qcolor(get_label_color(other, lid))))
                except Exception:  # noqa: BLE001
                    pass
                self._inside_ids.addItem(item)
        self._inside_ids.blockSignals(False)
        self._update_inside_count()

    def _inside_ticked(self) -> list[int]:
        out = []
        for i in range(self._inside_ids.count()):
            item = self._inside_ids.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return out

    def _on_inside_ticked(self, _item: Any = None) -> None:
        other = self._inside_layer()
        if other is not None:
            ticked = set(self._inside_ticked())
            self._inside_off[other.name] = {
                int(self._inside_ids.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self._inside_ids.count())} - ticked
        self._update_inside_count()
        self._apply_guard()

    def _tick_inside(self, on: bool) -> None:
        self._inside_ids.blockSignals(True)
        for i in range(self._inside_ids.count()):
            self._inside_ids.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._inside_ids.blockSignals(False)
        self._on_inside_ticked()

    def _update_inside_count(self) -> None:
        n = self._inside_ids.count()
        k = len(self._inside_ticked())
        self._inside_count.setText(f"{k} of {n} label(s)" if n else "No labels in the mask.")

    def _inside_mask(self, other: Any) -> Any:
        """Where painting is allowed: the mask's ticked labels (any non-zero when all are)."""
        data = label_source_data(other)
        n = self._inside_ids.count()
        ticked = self._inside_ticked()
        if n == 0 or len(ticked) == n:
            return data
        key = (id(other), id(data), tuple(sorted(ticked)))
        if self._inside_cache is not None and self._inside_cache[0] == key:
            return self._inside_cache[1]
        if not ticked:
            allowed = np.zeros(data.shape, dtype=bool)
        else:
            # On the backend; the guard checks Napari's (host) history items with it.
            from nvitk.core.backend import get_array_module

            ids = as_backend_array([int(x) for x in sorted(ticked)])
            allowed = to_numpy(get_array_module(ids).isin(as_backend_array(data), ids))
        self._inside_cache = (key, allowed)
        return allowed

    def _range_from_view(self) -> None:
        if self.reference() is None:
            return
        lo, hi = self.intensity_window()
        self._range_lo.setValue(lo)
        self._range_hi.setValue(hi)
        self._apply_guard()

    def _thr_from_view(self) -> None:
        if self.reference() is not None:
            lo, hi = self.intensity_window()
            self._thr_lo.setValue(lo)
            self._thr_hi.setValue(hi)

    # ── the box ──────────────────────────────────────────────────────────────

    def box(self) -> RoiBox:
        """The viewer's box (:mod:`nvitk.gui.labels.roi_box`)."""
        return roi_box(self._viewer)

    def box_armed(self) -> bool:
        """True while the next drag (canvas or orthogonal view) draws the box."""
        return self._box_armed

    def _box_grid_layer(self) -> Any | None:
        """The layer a box is drawn on: the label layer edited, else the reference."""
        return self.target() or self.reference()

    def _arm_box(self, on: bool) -> None:
        if bool(on) == self._box_armed:
            return
        layer = self.target()
        if on:
            if self._box_grid_layer() is None:
                self._box_draw.setChecked(False)
                self._say("Pick a label layer (or a reference image) to draw the box on.", error=True)
                return
            # The drag draws: Napari's brush must not paint meanwhile.
            self._box_restore = (self._extra or str(getattr(layer.mode, "value", layer.mode))) \
                if layer is not None and E.is_labels(layer) else None
            if self._box_restore is not None:
                self.set_tool("pan_zoom")
            self._box_armed = True
            self._say("Drag a rectangle on the canvas (2D) or in an orthogonal view.")
        else:
            self._box_armed = False
            tool, self._box_restore = self._box_restore, None
            if tool is not None and tool in self._tool_buttons:
                self.set_tool(tool)
        self._update_camera_lock()
        self.tool_changed.emit()

    def box_drawn(self) -> None:
        """A box drag ended: the tool in use before comes back."""
        if self._box_draw.isChecked():
            self._box_draw.setChecked(False)

    def _box_drag(self, event: Any) -> Any:
        """Drawing the box on the canvas: a rectangle on the two axes on screen."""
        layer = self._box_grid_layer()
        if layer is None or E.is_3d_view(self._viewer):
            if layer is not None:
                self._say("Draw the box in a 2D view, or in the orthogonal views.", error=True)
            yield
            while event.type == "mouse_move":
                yield
            return
        a, b = [int(d) for d in E.displayed_axes(self._viewer, layer)][-2:]
        box = self.box()
        before = (box.shape, list(box.lo), list(box.hi), box.layer)

        def at(position: Any) -> list[float]:
            return [float(v) for v in layer.world_to_data(position)][-layer.data.ndim:]

        p0 = at(event.position)
        moved = False
        yield
        while event.type == "mouse_move":
            p1 = at(event.position)
            ranges = {d: (int(round(min(p0[d], p1[d]))), int(round(max(p0[d], p1[d]))) + 1) for d in (a, b)}
            box.set_axes(layer, ranges, emit=False)
            moved = True
            yield
        if not moved:
            if before[0] is None:
                box.clear()
            elif before[3] is not None:
                box.set(before[3], before[1], before[2], emit=False)
            return
        box.changed.emit()
        self.box_drawn()

    def _box_clear(self) -> None:
        self.box().clear()

    def _box_fit_label(self) -> None:
        layer = self.target()
        lid = self.label_id()
        if layer is None or lid == 0:
            self._say("Pick a label first.", error=True)
            return
        from nvitk.segmentation.mask_ops import region_bounds

        dims = spatial_dims(layer)
        index = tuple(slice(None) if d in dims else E.current_point(self._viewer, layer)[d]
                      for d in range(layer.data.ndim))
        bounds = region_bounds(as_backend_array(layer.data[index]) == lid, pad=5)
        if bounds is None:
            self._say(f"Label {lid} has no voxels here.", error=True)
            return
        self.box().set(layer, bounds[0], bounds[1])

    def _box_depth(self) -> None:
        layer = self._box_grid_layer()
        if layer is None:
            return
        normal = E.normal_axis(self._viewer, layer)
        if normal is None:
            self._say("The depth is the axis across a 2D view's slices: switch the canvas to 2D.", error=True)
            return
        here = E.current_point(self._viewer, layer)[normal]
        n = int(self._box_depth_n.value())
        self.box().set_axes(layer, {normal: (here - n, here + n + 1)})

    def _box_full_depth(self) -> None:
        layer = self._box_grid_layer()
        if layer is None:
            return
        normal = E.normal_axis(self._viewer, layer)
        if normal is None:
            self._say("The depth is the axis across a 2D view's slices: switch the canvas to 2D.", error=True)
            return
        self.box().set_axes(layer, {normal: (0, int(layer.data.shape[normal]))})

    def _box_from_spins(self) -> None:
        if self._syncing_box:
            return
        layer = self._box_grid_layer()
        box = self.box()
        if layer is None:
            return
        dims = spatial_dims(layer)
        lo = [self._box_spins[k][0].value() for k in range(len(dims))]
        hi = [self._box_spins[k][1].value() + 1 for k in range(len(dims))]
        if not box.active and all(a == 0 for a in lo) and all(h == n for h, n in zip(hi, spatial_shape(layer))):
            return  # the spins only show the whole volume: no box yet
        box.set(layer, lo, hi)

    def _on_box_changed(self) -> None:
        box = self.box()
        layer = box.layer or self._box_grid_layer()
        self._syncing_box = True
        try:
            dims = spatial_dims(layer) if layer is not None else []
            labels = [str(v) for v in (getattr(layer, "axis_labels", None) or [])] if layer is not None else []
            for k, (label, (lo, hi)) in enumerate(zip(self._box_axis_labels, self._box_spins)):
                shown = k < len(dims)
                for w in (label, self._box_dashes[k], lo, hi):
                    w.setVisible(shown)
                if not shown:
                    continue
                d = dims[k]
                n = int(layer.data.shape[d])
                # Named axes (X, Y, Z…) by their name; Napari's defaults (-3, -2…) by number.
                name = labels[d] if d < len(labels) and labels[d].isalpha() and len(labels[d]) <= 2 else f"Axis {d}"
                label.setText(f"{name} ({n})")
                lo.setRange(0, n - 1)
                hi.setRange(0, n - 1)
                if box.applies(layer):
                    lo.setValue(box.lo[k])
                    hi.setValue(box.hi[k] - 1)
                else:
                    lo.setValue(0)
                    hi.setValue(n - 1)
        finally:
            self._syncing_box = False
        self._box_info.setText(box.describe(layer) if box.active else "No box — Draw box, Around label, or type it.")
        self._box_intensity_cache = None
        self._apply_guard()
        ortho = getattr(self._viewer, "_nvitk_ortho_panel", None)
        if ortho is not None:
            try:
                ortho._label_editor.show_box()
            except Exception:  # noqa: BLE001
                pass

    def _on_box_range(self, on: bool) -> None:
        for button in self._from_view_buttons:
            button.setText("From box" if on else "From view")
        self._box_intensity_cache = None

    def _box_intensity_range(self, image: Any) -> tuple[float, float] | None:
        """The box's 1–99 % intensity range of *image*, when asked for (Box card)."""
        box = self.box()
        if not self._box_range.isChecked() or not box.applies(image):
            return None
        key = (tuple(box.lo), tuple(box.hi), id(image), id(image.data))
        if self._box_intensity_cache is not None and self._box_intensity_cache[0] == key:
            return self._box_intensity_cache[1]
        value = box.intensity_range(image)
        self._box_intensity_cache = (key, value)
        return value

    def _limit_to_box(self, layer: Any, index: Sequence[Any]) -> tuple[Any, ...]:
        """An edit's region, cut down to the box when edits are kept inside it."""
        if self._box_keep.isChecked():
            return self.box().limit(layer, index)
        return tuple(index)

    def crop_box(self) -> None:
        box = self.box()
        if not box.active:
            self._say("Draw the box first.", error=True)
            return
        layers: list[Any] = []
        if self._crop_all.isChecked():
            layers = [ly for ly in self._viewer.layers if type(ly).__name__ in ("Image", "Labels") and box.applies(ly)
                      and not getattr(ly, "_nvitk_internal", False)]
        else:
            if self._crop_ref.isChecked() and self.reference() is not None:
                layers.append(self.reference())
            if self._crop_labels.isChecked() and self.target() is not None:
                layers.append(self.target())
        layers = [ly for ly in layers if box.applies(ly)]
        if not layers:
            self._say("Nothing to crop: tick what to cut (on the box's grid).", error=True)
            return

        def run() -> str:
            made = crop_layers(self._viewer, layers, box)
            return f"Cropped to the box ({box.describe()}): " + ", ".join(ly.name for ly in made) + "."

        self._run("Crop", run)

    def _allowed(self, layer: Any, index: tuple[Any, ...], work: Any) -> Any:
        """Where the active label may be written in ``data[index]``."""
        allowed = E.writable(work, self.label_id(), preserve=self._preserve.isChecked())
        guard = E.paint_guard(layer)
        if guard is not None and guard.active:
            extra = guard.allowed(index)
            if extra is not None:
                allowed = allowed & extra
        return allowed

    # ── tools ────────────────────────────────────────────────────────────────

    def _scope(self) -> str:
        return "volume" if self._scope_volume.isChecked() else "slice"

    def _editable_target(self) -> Any:
        layer = self.target()
        if layer is None:
            raise ValueError("No label layer: pick one under Layer, or make one with New….")
        if not E.is_labels(layer):
            raise ValueError("This is an image mask: Convert to Labels first (it gives you undo).")
        return layer

    def _write(self, layer: Any, index: tuple[Any, ...], new: Any) -> int:
        """Write a tool's result; the guard leaves it alone (the tool already applied the area)."""
        guard = E.paint_guard(layer)
        if guard is not None:
            guard.suspended = True
        try:
            changed = E.write_region(layer, index, new)
        finally:
            if guard is not None:
                guard.suspended = False
        label_filter_hub().labels_changed.emit(layer)
        self._keep_active_visible()
        self._count_timer.start()
        return changed

    def _refine(self, what: str) -> None:
        def run() -> str:
            layer = self._editable_target()
            lid = self.label_id()
            if lid == 0:
                raise ValueError("Pick the label to refine (not the background).")
            index = E.region(self._viewer, layer, self._scope())
            work = layer.data[index]
            r = int(self._radius.value())
            if what == "grow":
                out = E.grow(work, lid, r, self._allowed(layer, index, work))
            elif what == "shrink":
                out = E.shrink(work, lid, r)
            elif what == "smooth":
                out = E.smooth(work, lid, r, self._allowed(layer, index, work))
            elif what == "holes":
                out = E.fill_holes(work, lid, self._allowed(layer, index, work))
            elif what == "largest":
                out = E.keep_largest(work, lid)
            else:
                out = E.remove_islands(work, lid, int(self._min_size.value()))
            changed = self._write(layer, index, out)
            where = "in the slice" if self._scope() == "slice" else "in the volume"
            return f"{what.capitalize()}: {changed:,} voxel(s) changed {where}."

        self._run(what, run)

    def threshold(self) -> None:
        def run() -> str:
            layer = self._editable_target()
            ref = self.reference()
            if ref is None:
                raise ValueError("Threshold fill reads a reference image: pick one under Layer.")
            lid = self.label_id()
            if lid == 0:
                raise ValueError("Pick the label to fill (not the background).")
            index = E.region(self._viewer, layer, self._scope())
            work = layer.data[index]
            out = E.threshold_fill(work, ref.data[index], self._thr_lo.value(), self._thr_hi.value(), lid,
                                   self._allowed(layer, index, work))
            return f"Threshold fill: {self._write(layer, index, out):,} voxel(s) labelled."

        self._run("Threshold", run)

    def interpolate(self) -> None:
        def run() -> str:
            layer = self._editable_target()
            lid = self.label_id()
            if lid == 0:
                raise ValueError("Pick the label to interpolate (not the background).")
            index = E.region(self._viewer, layer, "volume")
            axis = None
            if self._interp_axis.currentData() == "view":
                axis = E.normal_axis(self._viewer, layer)
                if axis is None:
                    raise ValueError("“Across the view” needs a 2D view of a 3D volume (or pick Auto).")
                # The axis within the volume region (a time axis before it drops out).
                axis -= sum(1 for d in range(axis) if not isinstance(index[d], slice))
            from nvitk.gui.core.spatial import layer_spacing

            spacing = layer_spacing(layer)
            out = E.interpolate_slices(layer.data[index], lid, axis=axis,
                                       method=self._interp_method.currentData(),
                                       spacing=spacing, preserve=self._preserve.isChecked())
            return f"Interpolation: {self._write(layer, index, out):,} voxel(s) filled between drawn slices."

        self._run("Interpolate", run)

    def copy_slice(self, step: int) -> None:
        def run() -> str:
            layer = self._editable_target()
            lid = self.label_id()
            if lid == 0:
                raise ValueError("Pick the label to copy (not the background).")
            guard = E.paint_guard(layer)
            if guard is not None:
                guard.suspended = True
            try:
                changed, target = E.copy_slice(self._viewer, layer, lid, step, preserve=self._preserve.isChecked())
            finally:
                if guard is not None:
                    guard.suspended = False
            if target is None:
                return "No slice beyond this one."
            label_filter_hub().labels_changed.emit(layer)
            self._count_timer.start()
            if self._copy_move.isChecked():
                E.step_to_slice(self._viewer, layer, E.normal_axis(self._viewer, layer), target)
            return f"Copied to slice {target}: {changed:,} voxel(s) changed."

        self._run("Copy slice", run)

    def merge(self) -> None:
        source = self._merge_combo.currentData()
        if source is None:
            return
        self._merge(int(source), self.label_id())

    def _merge(self, source: int, target_id: int) -> None:
        def run() -> str:
            layer = self._editable_target()
            if target_id == 0 or source == target_id:
                raise ValueError("Merge one label into a different, non-background one.")
            n = E.replace_label(layer, source, target_id)
            label_filter_hub().labels_changed.emit(layer)
            label_filter_hub().names_changed.emit(layer)
            self._count_timer.start()
            return f"Merged label {source} into {target_id} ({n:,} voxels)."

        self._run("Merge", run)

    def split(self) -> None:
        def run() -> str:
            layer = self._editable_target()
            lid = self.label_id()
            if lid == 0:
                raise ValueError("Pick the label to split (not the background).")
            index = E.region(self._viewer, layer, "volume")
            out, new_ids = E.split_components(layer.data[index], lid, E.next_free_label(layer))
            self._write(layer, index, out)
            if not new_ids:
                return f"Label {lid} is in one piece."
            return f"Split label {lid}: the other pieces are labels {', '.join(map(str, new_ids))}."

        self._run("Split", run)

    def renumber(self) -> None:
        def run() -> str:
            layer = self._editable_target()
            mapping = E.relabel_consecutive(layer)
            label_filter_hub().labels_changed.emit(layer)
            label_filter_hub().names_changed.emit(layer)
            self._count_timer.start()
            if not mapping:
                return "The labels are already numbered 1…N."
            if self.label_id() in mapping:
                self.select_label(mapping[self.label_id()])
            return "Renumbered: " + ", ".join(f"{a}→{b}" for a, b in sorted(mapping.items()))

        self._run("Renumber", run)

    def delete_active(self) -> None:
        self._delete(self.label_id())

    def _delete(self, lid: int) -> None:
        layer = self.target()
        if layer is None or lid == 0:
            return
        delegate = getattr(self._viewer, "_nvitk_label_chip_delegate", None)
        if delegate is not None:
            delegate.delete(layer, lid)
        else:
            if not E.is_labels(layer):
                answer = QMessageBox.question(self, "Delete label", f"Clear label {lid}? An image mask has no undo.")
                if answer != QMessageBox.StandardButton.Yes:
                    return
            E.delete_label(layer, lid)
            label_filter_hub().labels_changed.emit(layer)
        self._count_timer.start()
        self._say(f"Deleted label {lid}" + (" — Ctrl+Z to undo." if E.is_labels(layer) else "."))

    # ── new / convert ────────────────────────────────────────────────────────

    def new_layer(self) -> None:
        """A new, empty Labels layer on the reference image's grid (or the active image's)."""
        from nvitk.gui.core.spatial import layer_spatial_kwargs

        base = self.reference()
        if base is None:
            active = self._viewer.layers.selection.active
            base = active if active is not None and type(active).__name__ in ("Image", "Labels") else None
        if base is None:
            images = self._image_layers(None)
            base = images[-1] if images else None
        if base is None:
            self._say("Open an image first: the new labels take its grid.", error=True)
            return
        meta = copy_layer_metadata_for_output(dict(getattr(base, "metadata", None) or {}))
        meta.pop("nvitk_label_names", None)
        layer = self._viewer.add_labels(
            np.zeros(tuple(base.data.shape), dtype=np.int32),
            name=f"{base.name}_labels",
            metadata=meta,
            **layer_spatial_kwargs(base),
        )
        self.bind(layer)
        ref_idx = self._ref_combo.findData(base.name)
        if ref_idx >= 0:
            self._ref_combo.setCurrentIndex(ref_idx)
            self._on_reference_changed()
        self.select_label(1)
        self.set_tool("paint")
        self._say(f"New labels “{layer.name}” on {base.name}'s grid — painting label 1.")

    def convert_target(self) -> None:
        layer = self.target()
        if layer is None or E.is_labels(layer):
            return
        try:
            new = ensure_labels_layer(self._viewer, layer)
        except Exception as exc:  # noqa: BLE001
            self._say(f"Could not convert: {exc}", error=True)
            return
        self.bind(new)
        self._say(f"“{new.name}” is a Labels layer now: brush, tools and undo work on it.")

    # ── feedback ─────────────────────────────────────────────────────────────

    def _run(self, name: str, fn: Callable[[], str]) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            message = fn()
        except Exception as exc:  # noqa: BLE001 — shown, not raised into Qt
            self._say(f"{name}: {exc}", error=True)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._say(message)

    def _say(self, text: str, *, error: bool = False) -> None:
        self._status.setStyleSheet(f"color: {COLOR_ERROR if error else COLOR_OK};")
        self._status.setText(text)
        try:
            self._viewer.status = text
        except Exception:  # noqa: BLE001
            pass


def build_labeling_panel(viewer: Any) -> LabelingPanel:
    """The Labeling tab's panel, registered on *viewer* for the layer list's menu."""
    panel = LabelingPanel(viewer)
    viewer._nvitk_labeling_panel = panel
    return panel


__all__ = ["LABELING_DOCK_NAME", "LabelingPanel", "build_labeling_panel"]
