"""The "3D view" options of the orthogonal views, as a widget of their own.

What the 3D canvas shows: whether layers on other grids are resampled, how the main canvas draws,
the orientation figure, the per-layer table (3D slice / Cut / Main), the three slices in 3D, the
see-inside cut, and the centre / reset / 3D buttons.

The widgets stay *attributes of the panel* (``panel._resample_box`` and so on), because the
panel's logic reads and updates them; this class only owns where they are laid out. That is what
lets the options move out of the 2x2 grid (into a pop-over over the 3D pane) without the panel
noticing.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import COLOR_MUTED, FONT_S, SPACE_TIGHT, Card, section_heading


class Controls3D(QWidget):
    """The 3D-view options card, bound to *panel* (an :class:`OrthoViewerPanel`)."""

    def __init__(self, panel: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        # No overlay picker: the panel draws every visible layer, so the layer
        # list *is* the control surface. This only reports what that came to.
        panel._composite_label = QLabel("—")
        panel._composite_label.setWordWrap(True)
        panel._composite_label.setStyleSheet(f"color: {COLOR_MUTED};")
        panel._composite_label.setToolTip(
            "Every visible layer on this grid is drawn, in the layer list's own "
            "order and with each layer's colormap, opacity and blending. Hide a "
            "layer in the list to take it out."
        )

        card = Card("3D view")
        card.add(panel._composite_label)

        panel._resample_box = QCheckBox("Resample layers on other grids")
        panel._resample_box.setToolTip(
            "On: a visible layer on another grid (another series, a registration "
            "output) is resampled onto the active layer's grid and drawn. Off: only "
            "layers already on that grid are drawn — no resampling cost, no "
            "interpolated pixels."
        )
        panel._resample_box.setChecked(panel._resample_offgrid)
        panel._resample_box.toggled.connect(panel._on_resample_toggled)
        card.add(panel._resample_box)

        # The main canvas has its own say over resampling: Napari draws every layer
        # in world space through its affine; turned, flipped and placed as acquired,
        # layers on other grids resampled into alignment. Off, each image and label
        # layer is drawn as stored, on its own voxel grid. Which layers are drawn
        # there at all is the table's "Main" column.
        panel._main_resample_box = QCheckBox("Main canvas: resample layers into world space")
        panel._main_resample_box.setToolTip(
            "On: the main Napari canvas draws every layer in world (scanner) space through "
            "its affine — an oblique or flipped acquisition turned into place, layers on "
            "other grids resampled into alignment. Off: every image and label layer is drawn "
            "as stored, on its own voxel grid (voxel size kept; no rotation, flip, "
            "obliquity or offset), the way these views lay a volume out — layers on the "
            "same grid still overlay, layers on other grids are not aligned. Meshes, points "
            "and shapes stay in world space. Tools, saving and exports always use the real "
            "geometry."
        )
        panel._main_resample_box.setChecked(panel._main_canvas.resample())
        panel._main_resample_box.toggled.connect(panel._main_canvas.set_resample)
        card.add(panel._main_resample_box)

        panel._main_canvas_box = QCheckBox("Show the main canvas")
        panel._main_canvas_box.setToolTip(
            "Off: the main Napari canvas is hidden and stops rendering; these views "
            "and the other docks take its space. Also under View ▸ Show main canvas."
        )
        panel._main_canvas_box.setChecked(panel._main_canvas.canvas_shown())
        panel._main_canvas_box.toggled.connect(panel._main_canvas.set_canvas_shown)
        card.add(panel._main_canvas_box)
        panel._main_canvas.changed.connect(panel._on_main_canvas_changed)

        panel._marker_box = QCheckBox("Orientation figure on the 3D canvas")
        panel._marker_box.setToolTip(
            "A person and R/L, A/P, H/F arrows in the corner of the 3D view, turning "
            "with the camera — the marker of the QC reports' 3D renders."
        )
        panel._marker_box.toggled.connect(panel._on_marker_toggled)
        card.add(panel._marker_box)

        # One row per drawn layer: whether it gets a slice image in 3D, and
        # whether the see-inside cut opens it. A mask can stay whole while the
        # CT around it is cut away, or the other way round.
        panel._layers_table = QTableWidget(0, 4)
        panel._layers_table.setHorizontalHeaderLabels(["Layer", "3D slice", "Cut", "Main"])
        panel._layers_table.verticalHeader().setVisible(False)
        panel._layers_table.setSelectionMode(QAbstractItemView.NoSelection)
        panel._layers_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = panel._layers_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        panel._layers_table.setMinimumHeight(70)
        panel._layers_table.setToolTip(
            "3D slice: draw this layer's slice image on the 3D planes (needs "
            "'…and the slice image' below). Cut: let 'See inside' open this layer. "
            "Main: draw this layer on the main Napari canvas — unticked, it is hidden "
            "there only and still drawn in these views and listed as visible."
        )
        panel._layers_table.itemChanged.connect(panel._on_table_item_changed)
        from nvitk.gui.core.widgets import resizable  # drag the handle under it to resize
        card.add(resizable(panel._layers_table, height=150, pref_key="ortho.layers_table"))

        panel._show_planes = QCheckBox("Show the three slices in 3D")
        panel._show_planes.setToolTip(
            "Push the three cuts onto the Napari canvas, so the 3D view shows the "
            "same slices these panels do."
        )
        panel._show_planes.toggled.connect(panel._on_show_planes)
        card.add(panel._show_planes)

        panel._cross3d_box = QCheckBox("Crosshair lines in 3D")
        panel._cross3d_box.setToolTip(
            "Three lines through the crosshair point, one along each axis (red, green, blue like "
            "the slice outlines): where the 2D views' crosshair is, on the 3D canvas."
        )
        from nvitk.gui.viz.ortho.layout import _ortho_prefs

        panel._cross3d_box.setChecked(bool(_ortho_prefs().get("crosshair_3d", False)))
        panel._cross3d_box.toggled.connect(panel._on_cross3d_toggled)
        card.add(panel._cross3d_box)

        panel._show_slice_image = QCheckBox("…and the slice image, not just its outline")
        panel._show_slice_image.setToolTip(
            "The coloured outline — red, green and blue for the three axes — is "
            "always drawn: it says where each cut is without hiding what is behind "
            "it. Tick this to draw the slice itself as well, which costs a texture "
            "upload per cut."
        )
        panel._show_slice_image.setEnabled(False)
        panel._show_slice_image.toggled.connect(lambda _c: panel._on_plane_image_toggled())
        card.add(panel._show_slice_image)

        card.add(section_heading("See inside"))
        hint = QLabel(
            "Cut the volume at one view's slice and drop everything on one side of it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLOR_MUTED}; font-size: {FONT_S}px;")
        card.add(hint)

        clip_row = QHBoxLayout()
        clip_row.setSpacing(SPACE_TIGHT)
        panel._clip_axis = QComboBox()
        panel._clip_side = QComboBox()
        panel._clip_side.addItem("off", "off")
        panel._clip_side.addItem("keep below", "below")
        panel._clip_side.addItem("keep above", "above")
        panel._clip_axis.currentIndexChanged.connect(lambda _i: panel._apply_clip())
        panel._clip_side.currentIndexChanged.connect(lambda _i: panel._apply_clip())
        clip_row.addWidget(panel._clip_axis, stretch=1)
        clip_row.addWidget(panel._clip_side, stretch=1)
        card.add_layout(clip_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACE_TIGHT)
        panel._btn_centre = QPushButton("Centre crosshair")
        panel._btn_centre.clicked.connect(panel._centre_crosshair)
        panel._btn_reset_orient = QPushButton("Reset orientation")
        panel._btn_reset_orient.setToolTip(
            "Put the three planes back on the volume's own axes, undoing any "
            "rotation made by dragging the crosshair ends."
        )
        panel._btn_reset_orient.setEnabled(False)
        panel._btn_reset_orient.clicked.connect(panel._reset_orientation)
        panel._btn_3d = QPushButton("3D canvas")
        panel._btn_3d.setToolTip("Switch the Napari canvas to its 3D view.")
        panel._btn_3d.clicked.connect(panel._show_3d)
        btn_row.addWidget(panel._btn_centre)
        btn_row.addWidget(panel._btn_reset_orient)
        btn_row.addWidget(panel._btn_3d)
        card.add_layout(btn_row)
        card.body().addStretch(1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(card)
        #: The card itself, for a host that wants to restyle or re-title it.
        self.card = card


__all__ = ["Controls3D"]
