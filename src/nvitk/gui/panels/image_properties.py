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
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.spatial import (
    AxisProperties,
    LayerSpatialProperties,
    format_layer_spatial_info,
    layer_spatial_properties,
)

# Matches the dark chrome of the sibling dock panels (see ``dicom_tags``).
_BG = "#2b2b2b"
_BG_RAISED = "#323232"
_FG = "#e8e8e8"
_FG_MUTED = "#9a9a9a"
_BORDER = "#454545"
_ACCENT = "#6fa8dc"
_TRANSLATION = "#e5a25b"

# Anatomical direction → the accent used for its axis chip, so orientation reads
# at a glance instead of one letter at a time.
_AXIS_COLORS: dict[str, str] = {
    "R": "#e06c6c",
    "L": "#e06c6c",
    "A": "#7bb47b",
    "P": "#7bb47b",
    "S": "#6fa8dc",
    "I": "#6fa8dc",
}

_DIRECTION_WORD: dict[str, str] = {
    "R": "Right",
    "L": "Left",
    "A": "Anterior",
    "P": "Posterior",
    "S": "Superior",
    "I": "Inferior",
}


def _mono_font(size: int = 10) -> QFont:
    """Monospace font at *size* points, for numbers that should stay in columns."""
    font = QFont("Monospace")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(size)
    return font


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format a float compactly, dropping trailing zeros; ``—`` when unknown."""
    if value is None:
        return "—"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _clear_layout(layout: Any) -> None:
    """Empty *layout*, unparenting each widget so it stops painting immediately.

    ``deleteLater`` alone defers destruction to the next event-loop pass, during
    which the old widgets still draw over the newly built ones.
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


class _Card(QFrame):
    """Titled container grouping one family of properties."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Build a bordered card with *title* as its heading."""
        super().__init__(parent)
        self.setObjectName("card")
        # Scoped to #card: a bare ``QFrame`` rule would cascade the border onto
        # every child container and box each row.
        self.setStyleSheet(
            f"QFrame#card {{ background-color: {_BG_RAISED};"
            f" border: 1px solid {_BORDER}; border-radius: 5px; }}"
            " QWidget { background: transparent; border: none; }"
        )
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(10, 8, 10, 10)
        self._root.setSpacing(6)
        heading = QLabel(title.upper())
        heading.setStyleSheet(
            f"color: {_FG_MUTED}; font-size: 10px; font-weight: bold;"
            " letter-spacing: 1px; border: none;"
        )
        self._root.addWidget(heading)

    def add(self, widget: QWidget) -> None:
        """Append *widget* to the card body."""
        self._root.addWidget(widget)


def _chip(text: str, color: str) -> QLabel:
    """Small rounded badge label in *color*."""
    chip = QLabel(text)
    chip.setAlignment(Qt.AlignCenter)
    chip.setStyleSheet(
        f"color: {color}; border: 1px solid {color}; border-radius: 3px;"
        " padding: 1px 6px; font-weight: bold; font-size: 10px;"
    )
    chip.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return chip


def _measure(value: str, unit: str = "mm", *, color: str = _FG, align: Any = Qt.AlignRight) -> QLabel:
    """A number with its unit set in muted type, so the figures stay dominant."""
    label = QLabel(f"{value}<span style='color:{_FG_MUTED};font-size:10px;'> {unit}</span>")
    label.setTextFormat(Qt.RichText)
    label.setStyleSheet(f"color: {color}; border: none;")
    label.setFont(_mono_font())
    label.setAlignment(align | Qt.AlignVCenter)
    return label


def _cell(text: str, *, color: str = _FG, mono: bool = False, align: Any = Qt.AlignLeft) -> QLabel:
    """One grid cell as a styled, selectable label."""
    label = QLabel(text)
    label.setStyleSheet(f"color: {color}; border: none;")
    if mono:
        label.setFont(_mono_font())
    label.setAlignment(align | Qt.AlignVCenter)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _header_cell(text: str, align: Any = Qt.AlignLeft) -> QLabel:
    """Column heading for a property grid."""
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {_FG_MUTED}; border: none; border-bottom: 1px solid {_BORDER};"
        " padding-bottom: 3px; font-size: 10px; font-weight: bold;"
    )
    label.setAlignment(align | Qt.AlignVCenter)
    return label


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
        grid.addWidget(_header_cell(text, align), 0, col)

    for row, ax in enumerate(axes, start=1):
        name = f"{ax.index}" + (f" · {ax.label}" if ax.label else "")
        grid.addWidget(_cell(name, color=_FG_MUTED, mono=True), row, 0)

        code = (ax.code or "").upper()
        if code:
            direction = QWidget()
            dir_row = QHBoxLayout(direction)
            dir_row.setContentsMargins(0, 0, 0, 0)
            dir_row.setSpacing(6)
            dir_row.addWidget(_chip(code, _AXIS_COLORS.get(code, _ACCENT)))
            dir_row.addWidget(_cell(_DIRECTION_WORD.get(code, ""), color=_FG_MUTED))
            dir_row.addStretch(1)
            grid.addWidget(direction, row, 1)
        else:
            grid.addWidget(_cell("—", color=_FG_MUTED), row, 1)

        grid.addWidget(_cell(f"{ax.size}", mono=True, align=right), row, 2)
        grid.addWidget(
            _measure(_fmt(ax.spacing)) if ax.spacing is not None
            else _cell("—", color=_FG_MUTED, mono=True, align=right),
            row,
            3,
        )
        grid.addWidget(
            _measure(_fmt(ax.extent, 2), color=_FG_MUTED) if ax.extent is not None
            else _cell("—", color=_FG_MUTED, mono=True, align=right),
            row,
            4,
        )
    grid.setColumnStretch(1, 1)
    return holder


def _matrix_grid(matrix: np.ndarray, *, digits: int = 4, mark_translation: bool = False) -> QWidget:
    """Render *matrix* as an aligned numeric grid.

    With *mark_translation*, the last column of a 4x4 is tinted so the offset
    reads apart from the rotation/scale block it sits next to.
    """
    holder = QWidget()
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(3)
    arr = np.asarray(matrix, dtype=float)
    rows, cols = arr.shape[0], arr.shape[1]
    for r in range(rows):
        for c in range(cols):
            is_translation = mark_translation and c == cols - 1 and r < rows - 1
            grid.addWidget(
                _cell(
                    _fmt(arr[r, c], digits),
                    color=_TRANSLATION if is_translation else _FG,
                    mono=True,
                    align=Qt.AlignRight,
                ),
                r,
                c,
            )
    for c in range(cols):
        grid.setColumnStretch(c, 1)
    return holder


def _kv_row(label: str, value: str, *, mono: bool = True, color: str = _FG) -> QWidget:
    """A single ``label: value`` line."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    key = _cell(label, color=_FG_MUTED)
    key.setMinimumWidth(96)
    row.addWidget(key)
    row.addWidget(_cell(value, color=color, mono=mono), stretch=1)
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
        self._status.setStyleSheet(f"color: {_FG_MUTED};")

        self._title = QLabel("")
        self._title.setStyleSheet(f"color: {_FG}; font-size: 14px; font-weight: bold;")
        self._title.setWordWrap(True)

        self._badges = QWidget()
        self._badge_row = QHBoxLayout(self._badges)
        self._badge_row.setContentsMargins(0, 0, 0, 0)
        self._badge_row.setSpacing(6)
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
        self._body_layout.setSpacing(8)
        self._body_layout.setAlignment(Qt.AlignTop)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._body)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(f"QScrollArea {{ background-color: {_BG}; border: none; }}")
        self._body.setStyleSheet(f"background-color: {_BG};")

        self._btn_refresh = QPushButton("Refresh")
        self._btn_copy = QPushButton("Copy")
        self._btn_copy.setToolTip("Copy the full plain-text report to the clipboard")
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_copy)
        btn_row.addStretch(1)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
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
        _clear_layout(self._body_layout)

    def _set_badges(self, chips: list[tuple[str, str]]) -> None:
        """Replace the header badges with *chips* of ``(text, color)``."""
        _clear_layout(self._badge_row)
        for text, color in chips:
            self._badge_row.addWidget(_chip(text, color))
        self._badge_row.addStretch(1)

    def _build_cards(self, props: LayerSpatialProperties) -> None:
        """Populate the body with one card per family of properties in *props*."""
        if props.axes:
            axes_card = _Card("Axes")
            axes_card.add(_axis_grid(props.axes))
            self._body_layout.addWidget(axes_card)

        placement = _Card("Placement")
        origin = (
            ", ".join(_fmt(v, 3) for v in props.origin) if props.origin is not None else "—"
        )
        placement.add(_kv_row("Origin", origin))
        if props.scale is not None:
            placement.add(
                _kv_row(
                    "Napari scale",
                    ", ".join(_fmt(v, 4) for v in props.scale),
                    color=_FG_MUTED,
                )
            )
        fov = props.fov
        if fov is not None:
            placement.add(
                _kv_row("Field of view", " × ".join(f"{_fmt(v, 2)}" for v in fov) + " mm")
            )
        self._body_layout.addWidget(placement)

        if props.direction is not None:
            direction_card = _Card("Direction cosines")
            direction_card.add(_matrix_grid(props.direction))
            self._body_layout.addWidget(direction_card)

        if props.affine is not None:
            domain = "voxel" if props.is_raster else "data"
            affine_card = _Card(f"Affine  ({domain} → world)")
            affine_card.add(_matrix_grid(props.affine, digits=6, mark_translation=True))
            note = _cell("Amber column: translation (mm)", color=_FG_MUTED)
            note.setStyleSheet(f"color: {_FG_MUTED}; border: none; font-size: 10px;")
            affine_card.add(note)
            self._body_layout.addWidget(affine_card)

        if props.affine_source is not None:
            src_card = _Card("File affine  (differs from display)")
            src_card.add(_matrix_grid(props.affine_source, digits=6, mark_translation=True))
            self._body_layout.addWidget(src_card)

        source_card = _Card("Source")
        source = props.source or "—"
        source_label = _cell(source, color=_FG if props.source else _FG_MUTED, mono=True)
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
        chips: list[tuple[str, str]] = [(props.layer_type, _ACCENT)]
        if props.orientation:
            chips.append((props.orientation, _AXIS_COLORS.get(props.orientation[0], _ACCENT)))
        if props.dtype:
            chips.append((props.dtype, _FG_MUTED))
        self._set_badges(chips)
        self._status.setText(_summary_line(props))
        self._build_cards(props)
