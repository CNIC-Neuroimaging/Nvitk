"""Live spatial / affine properties for the active Napari layer.

Most of what this panel reports is *per-axis* — size, label, anatomical code,
spacing, extent — so it is laid out as one row per array axis rather than as a
stack of parallel tuples the reader has to line up by eye. The transforms are
shown as actual matrices, with the rotation block and the translation column
distinguished, and the raw text report stays one click away via **Copy**.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    AXIS_COLORS,
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_MUTED,
    COLOR_TEXT,
    SPACE,
    SPACE_TIGHT,
    Card,
    cell,
    chip,
    clear_layout,
    column_heading,
    fmt_number,
    kv_row,
    matrix_grid,
    measure,
)
from nvitk.gui.core.spatial import (
    AxisProperties,
    LayerSpatialProperties,
    format_layer_spatial_info,
    layer_spatial_properties,
)

_DIRECTION_WORD: dict[str, str] = {
    "R": "Right",
    "L": "Left",
    "A": "Anterior",
    "P": "Posterior",
    "S": "Superior",
    "I": "Inferior",
}


def _axis_grid(axes: list[AxisProperties]) -> QWidget:
    """One row per array axis: label, anatomical code, size, spacing and extent."""
    holder = QWidget()
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(5)

    right = Qt.AlignRight
    for col, (text, align) in enumerate(
        [
            ("AXIS", Qt.AlignLeft),
            ("DIRECTION", Qt.AlignLeft),
            ("SIZE", right),
            ("SPACING", right),
            ("EXTENT", right),
        ]
    ):
        grid.addWidget(column_heading(text, align), 0, col)

    for row, ax in enumerate(axes, start=1):
        name = f"{ax.index}" + (f" · {ax.label}" if ax.label else "")
        grid.addWidget(cell(name, color=COLOR_MUTED, mono=True), row, 0)

        code = (ax.code or "").upper()
        if code:
            direction = QWidget()
            dir_row = QHBoxLayout(direction)
            dir_row.setContentsMargins(0, 0, 0, 0)
            dir_row.setSpacing(SPACE_TIGHT)
            dir_row.addWidget(chip(code, AXIS_COLORS.get(code, COLOR_ACCENT)))
            dir_row.addWidget(cell(_DIRECTION_WORD.get(code, ""), color=COLOR_MUTED))
            dir_row.addStretch(1)
            grid.addWidget(direction, row, 1)
        else:
            grid.addWidget(cell("—", color=COLOR_MUTED), row, 1)

        grid.addWidget(cell(f"{ax.size}", mono=True, align=right), row, 2)
        grid.addWidget(
            measure(fmt_number(ax.spacing)) if ax.spacing is not None
            else cell("—", color=COLOR_MUTED, mono=True, align=right),
            row,
            3,
        )
        grid.addWidget(
            measure(fmt_number(ax.extent, 2), color=COLOR_MUTED) if ax.extent is not None
            else cell("—", color=COLOR_MUTED, mono=True, align=right),
            row,
            4,
        )
    grid.setColumnStretch(1, 1)
    return holder


def _summary_line(props: LayerSpatialProperties) -> str:
    """One-line size summary: a voxel grid for raster layers, an element count otherwise."""
    if not props.shape:
        return "No data on this layer."
    if not props.is_raster:
        count = int(props.shape[0])
        noun = "element" if count == 1 else "elements"
        return f"{count:,} {noun}  ({' × '.join(str(s) for s in props.shape)})"
    grid = " × ".join(str(s) for s in props.shape)
    return f"{grid} voxels  ({int(np.prod(props.shape)):,} total)"


class ImagePropertiesPanel(QWidget):
    """Show spacing, FOV, origin, orientation, and affine for the selected layer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the header strip, the scrollable card body, and the toolbar."""
        super().__init__(parent)
        self._status = QLabel("Select a layer to view spatial properties.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")

        self._title = QLabel("")
        self._title.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 14px; font-weight: bold;")
        self._title.setWordWrap(True)

        self._badges = QWidget()
        self._badge_row = QHBoxLayout(self._badges)
        self._badge_row.setContentsMargins(0, 0, 0, 0)
        self._badge_row.setSpacing(SPACE_TIGHT)
        self._badge_row.addStretch(1)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        header_layout.addWidget(self._title)
        header_layout.addWidget(self._badges)
        header_layout.addWidget(self._status)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(SPACE)
        self._body_layout.setAlignment(Qt.AlignTop)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._body)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(f"QScrollArea {{ background-color: {COLOR_BG}; border: none; }}")
        self._body.setStyleSheet(f"background-color: {COLOR_BG};")

        self._btn_refresh = QPushButton("Refresh")
        self._btn_copy = QPushButton("Copy")
        self._btn_copy.setToolTip("Copy the full plain-text report to the clipboard")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_copy)
        btn_row.addStretch(1)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(header)
        root.addLayout(btn_row)
        root.addWidget(self._scroll, stretch=1)
        self.setLayout(root)

        self._btn_refresh.clicked.connect(self._refresh_last_layer)
        self._btn_copy.clicked.connect(self._copy_report)
        self._last_layer: Any | None = None

    # ── internals ────────────────────────────────────────────────────────────

    def _refresh_last_layer(self) -> None:
        """Re-render spatial properties for whichever layer was last shown."""
        self.refresh_from_layer(self._last_layer)

    def _copy_report(self) -> None:
        """Put the plain-text spatial report for the current layer on the clipboard."""
        if self._last_layer is None:
            return
        try:
            text = format_layer_spatial_info(self._last_layer)
        except Exception:
            return
        from qtpy.QtWidgets import QApplication

        QApplication.clipboard().setText(text)
        self._status.setText("Copied the full report to the clipboard.")

    def _clear_body(self) -> None:
        """Remove every card from the scrollable body."""
        clear_layout(self._body_layout)

    def _set_badges(self, chips: list[tuple[str, str]]) -> None:
        """Replace the header badges with *chips* of ``(text, color)``."""
        clear_layout(self._badge_row)
        for text, color in chips:
            self._badge_row.addWidget(chip(text, color))
        self._badge_row.addStretch(1)

    def _build_cards(self, props: LayerSpatialProperties) -> None:
        """Populate the body with one card per family of properties in *props*."""
        if props.axes:
            axes_card = Card("Axes")
            axes_card.add(_axis_grid(props.axes))
            self._body_layout.addWidget(axes_card)

        placement = Card("Placement")
        origin = (
            ", ".join(fmt_number(v, 3) for v in props.origin) if props.origin is not None else "—"
        )
        placement.add(kv_row("Origin", origin))
        if props.scale is not None:
            placement.add(
                kv_row(
                    "Napari scale",
                    ", ".join(fmt_number(v, 4) for v in props.scale),
                    color=COLOR_MUTED,
                )
            )
        fov = props.fov
        if fov is not None:
            placement.add(
                kv_row("Field of view", " × ".join(f"{fmt_number(v, 2)}" for v in fov) + " mm")
            )
        self._body_layout.addWidget(placement)

        if props.direction is not None:
            direction_card = Card("Direction cosines")
            direction_card.add(matrix_grid(props.direction))
            self._body_layout.addWidget(direction_card)

        if props.affine is not None:
            domain = "voxel" if props.is_raster else "data"
            affine_card = Card(f"Affine  ({domain} → world)")
            affine_card.add(matrix_grid(props.affine, digits=6, mark_last_column=True))
            note = cell("Amber column: translation (mm)", color=COLOR_MUTED)
            note.setStyleSheet(f"color: {COLOR_MUTED}; border: none; font-size: 10px;")
            affine_card.add(note)
            self._body_layout.addWidget(affine_card)

        if props.affine_source is not None:
            src_card = Card("File affine  (differs from display)")
            src_card.add(matrix_grid(props.affine_source, digits=6, mark_last_column=True))
            self._body_layout.addWidget(src_card)

        source_card = Card("Source")
        source = props.source or "—"
        source_label = cell(source, color=COLOR_TEXT if props.source else COLOR_MUTED, mono=True)
        source_label.setWordWrap(True)
        source_label.setToolTip(source)
        source_card.add(source_label)
        self._body_layout.addWidget(source_card)

    # ── public API ───────────────────────────────────────────────────────────

    def refresh_from_layer(self, layer: Any | None) -> None:
        """Display *layer*'s spatial properties (spacing, FOV, origin, affine), or a placeholder
        message if *layer* is ``None`` or its properties can't be read."""
        self._last_layer = layer
        self._clear_body()
        if layer is None:
            self._title.setText("")
            self._set_badges([])
            self._status.setText("No layer selected.")
            return
        name = getattr(layer, "name", "layer")
        try:
            props = layer_spatial_properties(layer)
        except Exception as exc:
            self._title.setText(str(name))
            self._set_badges([])
            self._status.setText(f"Could not read properties for “{name}”: {exc}")
            return

        self._title.setText(props.name)
        chips: list[tuple[str, str]] = [(props.layer_type, COLOR_ACCENT)]
        if props.orientation:
            chips.append((props.orientation, AXIS_COLORS.get(props.orientation[0], COLOR_ACCENT)))
        if props.dtype:
            chips.append((props.dtype, COLOR_MUTED))
        self._set_badges(chips)
        self._status.setText(_summary_line(props))
        self._build_cards(props)
