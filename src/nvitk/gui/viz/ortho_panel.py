"""Three orthogonal 2D slice views of a volume, plus 3D planes on the Napari canvas.

The layout a radiologist expects, and the one 3D Slicer opens with: axial,
coronal and sagittal through one shared crosshair, each scrollable on its own,
each showing where the other two are cutting. Scroll a view, click in it, or drag
its slider — the crosshair moves and the other two follow.

The fourth quadrant drives the *main* Napari canvas rather than adding a second
one: with ``Show 3D planes`` on, the three slices are pushed onto the canvas as
``depiction="plane"`` layers at the crosshair, so the 3D view shows the same
three cuts in space. ``Clip volume`` then takes one axis and throws away
everything on one side of its slice, which is how you see *inside* a volume
instead of only at its surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy
from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_TEXT,
    COLOR_WELL,
    SPACE,
    SPACE_TIGHT,
    Card,
    section_heading,
)
from nvitk.gui.core.orientation import layer_orientation_codes
from nvitk.gui.core.spatial import layer_spacing
from nvitk.gui.labels.visibility import (
    get_label_color,
    is_label_like_layer,
    label_source_data,
    unique_layer_labels,
)

#: Axis codes → the plane you are looking at when you slice along that axis.
_PLANE_NAMES: dict[str, str] = {
    "S": "Axial", "I": "Axial",
    "A": "Coronal", "P": "Coronal",
    "R": "Sagittal", "L": "Sagittal",
}

#: Name for the layer nvitk adds to the canvas for each 3D slice plane.
_PLANE_LAYER_SUFFIX = "_ortho_plane"
_BOX_LAYER_SUFFIX = "_ortho_box"

#: One colour per array axis for the slice outlines, in axis order. Red / green /
#: blue so an outline says which of the three cuts it is without a legend.
BOX_AXIS_COLORS: tuple[str, str, str] = ("#ff4d4d", "#4dd964", "#4d9dff")
_BOX_EDGE_WIDTH = 0.6

#: How strongly an overlay is drawn over the base slice.
_OVERLAY_OPACITY = 0.55

#: Delay before pushing a crosshair move to the 3D canvas. Plane and clipping
#: updates re-upload volumes, so they are coalesced rather than run per step.
_CANVAS_SYNC_MS = 90

#: Zoom bounds and the factor one Ctrl+wheel notch applies. The floor is "fit to
#: the view", which is what the panels do without a zoom at all.
_MIN_ZOOM = 1.0
_MAX_ZOOM = 12.0
_ZOOM_STEP = 1.25

#: Largest volume for which a per-axis contiguous copy is worth its memory.
_SLICE_CACHE_BUDGET = 512 * 1024 * 1024


#: Anatomical opposite of each axis code.
_OPPOSITE: dict[str, str] = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}

#: What belongs at the top and at the right of each plane, in neurological
#: convention: the viewer's right is the patient's right, and superior is up.
_PLANE_UP_RIGHT: dict[str, tuple[str, str]] = {
    "Axial": ("A", "R"),
    "Coronal": ("S", "R"),
    "Sagittal": ("S", "A"),
}


@dataclass(frozen=True)
class AxisView:
    """One orthogonal view: which array axis it steps along, and how it is drawn."""

    axis: int
    title: str
    #: Array axis drawn down the image, and whether its direction is reversed.
    rows: int
    flip_rows: bool = False
    #: Array axis drawn across the image, and whether its direction is reversed.
    cols: int = 0
    flip_cols: bool = False

    def to_pixel(self, position: tuple[int, ...] | list[int], shape: tuple[int, ...]) -> tuple[int, int]:
        """Voxel *position* as a (row, column) pixel in this view's rendered slice."""
        row = int(position[self.rows])
        col = int(position[self.cols])
        if self.flip_rows:
            row = int(shape[self.rows]) - 1 - row
        if self.flip_cols:
            col = int(shape[self.cols]) - 1 - col
        return row, col

    def to_voxel(self, row: int, col: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """A (row, column) pixel back to ``(row axis, index, column axis, index)``."""
        r, c = int(row), int(col)
        if self.flip_rows:
            r = int(shape[self.rows]) - 1 - r
        if self.flip_cols:
            c = int(shape[self.cols]) - 1 - c
        return self.rows, r, self.cols, c


def _orient_in_plane(axis: int, codes: str, plane: str) -> tuple[int, bool, int, bool]:
    """Choose which in-plane axis is drawn down and which across, and their signs.

    A raw NumPy slice is drawn in array-axis order, which for a NIfTI volume is
    rarely the way anyone reads that plane: an axial slice of an RAS volume comes
    out with the patient running down the image and anterior to the right. This
    picks the row and column axes from the anatomical codes and flips whichever
    run the wrong way, so up is superior (or anterior, on an axial) and the
    viewer's right is the patient's right.
    """
    rest = [i for i in range(3) if i != axis]
    up, right = _PLANE_UP_RIGHT.get(plane, ("", ""))
    codes_u = codes.upper()

    def code_of(i: int) -> str:
        return codes_u[i] if i < len(codes_u) else ""

    if up and right:
        vertical = [i for i in rest if code_of(i) in (up, _OPPOSITE[up])]
        horizontal = [i for i in rest if code_of(i) in (right, _OPPOSITE[right])]
        if len(vertical) == 1 and len(horizontal) == 1 and vertical[0] != horizontal[0]:
            row_axis, col_axis = vertical[0], horizontal[0]
            # Rows increase downward, so an axis pointing at "up" must be reversed.
            return (
                row_axis,
                code_of(row_axis) == up,
                col_axis,
                code_of(col_axis) == _OPPOSITE[right],
            )
    # No usable codes: fall back to array order, unflipped.
    return rest[0], False, rest[1], False


#: Reading order of the three planes, matching how a viewer like 3D Slicer lays
#: them out. Views whose plane cannot be named keep their array-axis order.
_PLANE_ORDER: dict[str, int] = {"Axial": 0, "Coronal": 1, "Sagittal": 2}


def _axis_views(layer: Any) -> list[AxisView]:
    """The three orthogonal views for *layer*, named from its anatomical axis codes.

    Slicing *along* an axis shows the plane perpendicular to it: stepping the
    superior axis gives axial slices, the anterior axis coronal, the left-right
    axis sagittal.
    """
    codes = layer_orientation_codes(layer) or ""
    views: list[AxisView] = []
    for axis in range(3):
        code = codes[axis].upper() if axis < len(codes) else ""
        plane = _PLANE_NAMES.get(code, f"Axis {axis}")
        label = f"{plane}" + (f"  ·  {code}" if code else "")
        row_axis, flip_rows, col_axis, flip_cols = _orient_in_plane(axis, codes, plane)
        views.append(
            AxisView(
                axis=axis,
                title=label,
                rows=row_axis,
                flip_rows=flip_rows,
                cols=col_axis,
                flip_cols=flip_cols,
            )
        )
    views.sort(key=lambda v: (_PLANE_ORDER.get(v.title.split(" ")[0], 3), v.axis))
    return views


def _slice_of(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    """The 2D slice of *data* at *index* along *axis*, clamped into range."""
    n = int(data.shape[axis])
    idx = int(np.clip(index, 0, max(n - 1, 0)))
    return np.take(data, idx, axis=axis)


class SliceCache:
    """Serves 2D slices of a volume, keeping the slow axes contiguous.

    A C-ordered volume gives axis 0 for free, but a slice along the last axis is a
    strided gather over the whole array — around 40x slower on a 400x512x512 CT,
    which is exactly why scrolling axial felt heavier than the others. Reordering
    that axis into its own contiguous copy makes every axis equally cheap.

    The copy is built lazily, on the first scroll of that axis, and only when it
    fits :attr:`budget_bytes` — a large volume keeps the strided read rather than
    silently doubling the session's memory.
    """

    def __init__(self, data: np.ndarray, *, budget_bytes: int = _SLICE_CACHE_BUDGET) -> None:
        """Wrap *data*, reordering nothing until an axis is actually asked for."""
        self._data = data
        self._budget = int(budget_bytes)
        self._reordered: dict[int, np.ndarray | None] = {}

    def _fast_axis(self, axis: int) -> np.ndarray | None:
        """A contiguous copy with *axis* first, or ``None`` if it is not worth making."""
        if axis in self._reordered:
            return self._reordered[axis]
        arr = self._data
        # Axis 0 of a C-ordered array is already contiguous; nothing to gain.
        already_fast = axis == 0 and bool(arr.flags.c_contiguous)
        if already_fast or arr.nbytes > self._budget:
            self._reordered[axis] = None
            return None
        try:
            self._reordered[axis] = np.ascontiguousarray(np.moveaxis(arr, axis, 0))
        except (MemoryError, ValueError):
            self._reordered[axis] = None
        return self._reordered[axis]

    def slice(self, axis: int, index: int) -> np.ndarray:
        """The 2D slice at *index* along *axis*."""
        fast = self._fast_axis(int(axis))
        if fast is None:
            return _slice_of(self._data, int(axis), int(index))
        n = int(fast.shape[0])
        return fast[int(np.clip(index, 0, max(n - 1, 0)))]


def volume_contrast(data: np.ndarray, *, sample: int = 400_000) -> tuple[float, float]:
    """Robust display range for a volume, from a subsample of its finite voxels.

    Computed once for the volume rather than per slice: per-slice percentiles are
    both slower and *wrong* — the window would shift as you scroll, so the same
    tissue changes brightness from one slice to the next.
    """
    arr = np.asarray(data)
    flat = arr.reshape(-1)
    if flat.size > sample:
        flat = flat[:: max(int(flat.size // sample), 1)]
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, (1.0, 99.0)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def layer_contrast(layer: Any, data: np.ndarray) -> tuple[float, float] | None:
    """The window to draw *layer* with — its own, when it has one.

    Following the layer means the panels show what the Napari canvas shows: adjust
    brightness on the layer controls and these views track it, instead of staying
    on a percentile window computed once and disagreeing with the canvas from then
    on. ``None`` for label layers, which are drawn through their colour table.
    """
    if is_label_like_layer(layer):
        return None
    limits = getattr(layer, "contrast_limits", None)
    try:
        lo, hi = float(limits[0]), float(limits[1])
    except (TypeError, ValueError, IndexError):
        return volume_contrast(data)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return (lo, hi)
    return volume_contrast(data)


def _grayscale_rgb(plane: np.ndarray, contrast: tuple[float, float] | None = None) -> np.ndarray:
    """Window a 2D intensity slice to RGB uint8 using *contrast* (or its own range)."""
    arr = np.asarray(plane, dtype=np.float32)
    lo, hi = contrast if contrast is not None else volume_contrast(arr)
    if hi <= lo:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    norm = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def label_lut(layer: Any, label_ids: list[int]) -> dict[int, np.ndarray]:
    """RGB uint8 per label id, read once from the layer's own colours."""
    lut: dict[int, np.ndarray] = {}
    for lid in label_ids:
        rgba = get_label_color(layer, int(lid))
        lut[int(lid)] = (np.clip(np.asarray(rgba[:3], dtype=float), 0, 1) * 255).astype(np.uint8)
    return lut


def _label_rgb(plane: np.ndarray, layer: Any, lut: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Colour a 2D label slice, mapping ids through *lut* in one pass."""
    arr = np.rint(np.asarray(plane, dtype=np.float64)).astype(np.int64, copy=False)
    present = [int(v) for v in np.unique(arr) if int(v) != 0]
    if not present:
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    colors = lut if lut is not None else label_lut(layer, present)
    # One table indexed by id beats one boolean mask per id.
    table = np.zeros((max(present) + 1, 3), dtype=np.uint8)
    for lid in present:
        table[lid] = colors.get(lid, np.array([255, 255, 255], dtype=np.uint8))
    return table[np.clip(arr, 0, table.shape[0] - 1)]


def _oriented(plane: np.ndarray, axis: int, view: AxisView | None) -> np.ndarray:
    """Transpose / flip a raw slice into *view*'s anatomical orientation."""
    if view is None:
        return plane
    rest = [i for i in range(3) if i != axis]
    # ``np.take`` leaves the surviving axes in array order; put the view's row
    # axis first when it is the other one.
    if view.rows != rest[0]:
        plane = plane.T
    if view.flip_rows:
        plane = plane[::-1, :]
    if view.flip_cols:
        plane = plane[:, ::-1]
    return plane


def blend_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    opacity: float = 0.5,
) -> np.ndarray:
    """Composite *overlay* over *base*, leaving *base* showing where overlay is black.

    A label overlay is black exactly where it has no label, so treating black as
    transparent is what makes "raw image plus segmentation" read correctly.
    """
    if overlay.shape != base.shape:
        return base
    mask = overlay.any(axis=2)
    if not mask.any():
        return base
    out = base.astype(np.float32)
    a = float(np.clip(opacity, 0.0, 1.0))
    out[mask] = out[mask] * (1.0 - a) + overlay[mask].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def slice_to_rgb(
    layer: Any,
    data: np.ndarray,
    axis: int,
    index: int,
    view: AxisView | None = None,
    *,
    contrast: tuple[float, float] | None = None,
    lut: dict[int, np.ndarray] | None = None,
    cache: SliceCache | None = None,
) -> np.ndarray:
    """Render one orthogonal slice of *layer* as an RGB uint8 image.

    With a *view*, the slice is transposed and flipped into that view's anatomical
    orientation; without one it is drawn in raw array-axis order.
    """
    raw = cache.slice(axis, index) if cache is not None else _slice_of(data, axis, index)
    plane = _oriented(raw, axis, view)
    if is_label_like_layer(layer):
        return _label_rgb(plane, layer, lut)
    return _grayscale_rgb(plane, contrast)


class SliceView(QWidget):
    """One scrollable orthogonal slice, with crosshairs marking the other two."""

    #: (axis, new index) when the user scrolls, drags or clicks this view.
    sliceChanged = Signal(int, int)
    #: (row, column) pixel in the *rendered* slice when the user clicks in it.
    #: The panel maps it back through the view's orientation.
    pixelPicked = Signal(int, int)

    def __init__(self, view: AxisView, parent: QWidget | None = None) -> None:
        """Build the titled image canvas and its slice slider."""
        super().__init__(parent)
        self._view = view
        self._rgb: np.ndarray | None = None
        self._aspect = 1.0
        self._cross: tuple[int, int] | None = None
        self._count = 1
        #: Magnification over the fit-to-canvas size, and the normalised point of
        #: the slice held at the canvas centre.
        self._zoom = 1.0
        self._pan = [0.5, 0.5]
        self._drag_from: tuple[float, float] | None = None

        self._title = QLabel(view.title)
        self._title.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        self._index_label = QLabel("—")
        self._index_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 10px;")
        self._index_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._title)
        head.addStretch(1)
        head.addWidget(self._index_label)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setMinimumSize(140, 140)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet(
            f"background-color: #000000; border: 1px solid {COLOR_BORDER}; border-radius: 4px;"
        )
        self._canvas.installEventFilter(self)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(
            lambda value: self.sliceChanged.emit(self._view.axis, int(value))
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_TIGHT)
        root.addLayout(head)
        root.addWidget(self._canvas, stretch=1)
        root.addWidget(self._slider)

    # ── rendering ────────────────────────────────────────────────────────────

    def set_slice(
        self,
        rgb: np.ndarray | None,
        *,
        index: int,
        count: int,
        aspect: float = 1.0,
        crosshair: tuple[int, int] | None = None,
    ) -> None:
        """Show *rgb* as this view's slice, at *index* of *count*."""
        self._rgb = rgb
        self._aspect = float(aspect) if np.isfinite(aspect) and aspect > 0 else 1.0
        self._cross = crosshair
        self._count = max(int(count), 1)
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(self._count - 1, 0))
        self._slider.setValue(int(np.clip(index, 0, max(self._count - 1, 0))))
        self._slider.blockSignals(False)
        self._index_label.setText(f"{int(index) + 1} / {self._count}")
        self._repaint()

    def set_crosshair(self, crosshair: tuple[int, int] | None) -> None:
        """Move the crosshair without re-uploading the slice image."""
        self._cross = crosshair
        self._repaint()

    def _geometry(self) -> tuple[int, int, float, float] | None:
        """``(width, height, off_x, off_y)`` of the drawn slice inside the canvas.

        One place, used by the painter and by the click mapping alike: the two
        computing the transform separately is how a zoomed view ends up putting
        the crosshair somewhere other than where it was clicked.
        """
        if self._rgb is None or self._rgb.size == 0:
            return None
        h, w = self._rgb.shape[:2]
        target = self._canvas.size()
        # Physical aspect: a 3 mm slice spacing must not be drawn as if isotropic.
        fit_w = max(int(target.width()), 1)
        fit_h = max(int(fit_w * (h * self._aspect) / max(w, 1)), 1)
        if fit_h > target.height():
            fit_h = max(int(target.height()), 1)
            fit_w = max(int(fit_h * max(w, 1) / max(h * self._aspect, 1e-6)), 1)
        scaled_w = max(int(fit_w * self._zoom), 1)
        scaled_h = max(int(fit_h * self._zoom), 1)
        # The panned-to point sits at the canvas centre.
        off_x = target.width() / 2.0 - self._pan[0] * scaled_w
        off_y = target.height() / 2.0 - self._pan[1] * scaled_h
        if scaled_w <= target.width():
            off_x = (target.width() - scaled_w) / 2.0
        else:
            off_x = min(0.0, max(off_x, target.width() - scaled_w))
        if scaled_h <= target.height():
            off_y = (target.height() - scaled_h) / 2.0
        else:
            off_y = min(0.0, max(off_y, target.height() - scaled_h))
        return scaled_w, scaled_h, off_x, off_y

    def set_zoom(self, zoom: float, *, about: tuple[float, float] | None = None) -> None:
        """Set the magnification, keeping *about* (normalised) under the cursor."""
        new_zoom = float(np.clip(zoom, _MIN_ZOOM, _MAX_ZOOM))
        if about is not None and new_zoom > 1.0:
            self._pan = [float(np.clip(v, 0.0, 1.0)) for v in about]
        if new_zoom <= 1.0:
            self._pan = [0.5, 0.5]
        self._zoom = new_zoom
        self._repaint()

    def zoom(self) -> float:
        """Current magnification."""
        return float(self._zoom)

    def _repaint(self) -> None:
        """Scale the current slice into the canvas and draw the crosshairs on it."""
        from qtpy.QtGui import QColor, QPainter, QPen

        geometry = self._geometry()
        if geometry is None:
            self._canvas.setPixmap(QPixmap())
            self._canvas.setText("No slice")
            return
        scaled_w, scaled_h, off_x, off_y = geometry
        rgb = np.ascontiguousarray(self._rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        slice_map = QPixmap.fromImage(image).scaled(
            scaled_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )

        # Painted into a canvas-sized pixmap rather than handed over directly, so a
        # zoomed slice is cropped by the view instead of resizing the widget.
        canvas = QPixmap(self._canvas.size())
        canvas.fill(QColor("#000000"))
        painter = QPainter(canvas)
        painter.drawPixmap(int(round(off_x)), int(round(off_y)), slice_map)
        if self._cross is not None:
            pen = QPen(QColor(COLOR_ACCENT))
            pen.setWidth(1)
            painter.setPen(pen)
            row, col = self._cross
            y = int(round(off_y + (row + 0.5) / max(h, 1) * scaled_h))
            x = int(round(off_x + (col + 0.5) / max(w, 1) * scaled_w))
            painter.drawLine(0, y, canvas.width(), y)
            painter.drawLine(x, 0, x, canvas.height())
        if self._zoom > 1.0:
            pen = QPen(QColor(COLOR_MUTED))
            painter.setPen(pen)
            painter.drawText(6, canvas.height() - 6, f"{self._zoom:.1f}x")
        painter.end()
        self._canvas.setPixmap(canvas)

    def resizeEvent(self, event: Any) -> None:
        """Re-scale the slice when the view is resized."""
        super().resizeEvent(event)
        self._repaint()

    # ── interaction ──────────────────────────────────────────────────────────

    def eventFilter(self, obj: Any, event: Any) -> bool:
        """Scroll to step slices; click to move the crosshair."""
        from qtpy.QtCore import QEvent

        if obj is not self._canvas:
            return False
        if event.type() == QEvent.Wheel:
            up = event.angleDelta().y() > 0
            modifiers = event.modifiers()
            # Ctrl+wheel zooms, plain wheel scrolls slices. Scrolling is the far
            # commoner gesture here, so it keeps the unmodified wheel.
            if modifiers & Qt.ControlModifier:
                self.set_zoom(
                    self._zoom * (_ZOOM_STEP if up else 1.0 / _ZOOM_STEP),
                    about=self._normalised_at(event),
                )
                return True
            step = 1 if up else -1
            self._slider.setValue(int(np.clip(self._slider.value() + step, 0, self._count - 1)))
            return True
        if event.type() == QEvent.MouseButtonDblClick:
            self.set_zoom(1.0)
            return True
        if event.type() == QEvent.MouseButtonPress:
            if self._is_pan(event):
                self._drag_from = self._cursor_xy(event)
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseMove:
            if self._drag_from is not None and self._is_pan(event):
                self._pan_by(event)
                return True
            if not event.buttons():
                return False
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseButtonRelease:
            self._drag_from = None
            return False
        return False

    @staticmethod
    def _cursor_xy(event: Any) -> tuple[float, float]:
        """Cursor position on the canvas, across the Qt spellings."""
        pos = event.position() if hasattr(event, "position") else event.pos()
        return float(pos.x()), float(pos.y())

    @staticmethod
    def _is_pan(event: Any) -> bool:
        """True for the pan gesture: middle-drag, or shift with the left button."""
        buttons = event.buttons()
        if buttons & Qt.MiddleButton:
            return True
        return bool(buttons & Qt.LeftButton) and bool(event.modifiers() & Qt.ShiftModifier)

    def _normalised_at(self, event: Any) -> tuple[float, float]:
        """Where the cursor is in the slice, as fractions of its drawn size."""
        geometry = self._geometry()
        if geometry is None:
            return (0.5, 0.5)
        scaled_w, scaled_h, off_x, off_y = geometry
        x, y = self._cursor_xy(event)
        return (
            float(np.clip((x - off_x) / max(scaled_w, 1), 0.0, 1.0)),
            float(np.clip((y - off_y) / max(scaled_h, 1), 0.0, 1.0)),
        )

    def _pan_by(self, event: Any) -> None:
        """Drag the zoomed slice under the canvas."""
        geometry = self._geometry()
        if geometry is None or self._drag_from is None:
            return
        scaled_w, scaled_h, _ox, _oy = geometry
        x, y = self._cursor_xy(event)
        self._pan[0] = float(np.clip(self._pan[0] - (x - self._drag_from[0]) / max(scaled_w, 1), 0.0, 1.0))
        self._pan[1] = float(np.clip(self._pan[1] - (y - self._drag_from[1]) / max(scaled_h, 1), 0.0, 1.0))
        self._drag_from = (x, y)
        self._repaint()

    def _emit_click(self, event: Any) -> None:
        """Translate a click on the canvas into a crosshair position."""
        geometry = self._geometry()
        if geometry is None:
            return
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        x, y = self._cursor_xy(event)
        px, py = x - off_x, y - off_y
        if not (0 <= px < scaled_w and 0 <= py < scaled_h):
            return
        col = int(np.clip(px / scaled_w * w, 0, w - 1))
        row = int(np.clip(py / scaled_h * h, 0, h - 1))
        self.pixelPicked.emit(row, col)


# ──────────────────────────────────────────────────────────────────────────────
# 3D planes and clipping on the Napari canvas
# ──────────────────────────────────────────────────────────────────────────────
def _plane_layer_name(source_name: str, axis: int) -> str:
    """Name of the canvas layer carrying *axis*'s 3D slice plane."""
    return f"{source_name}{_PLANE_LAYER_SUFFIX}_{axis}"


def displayed_axes(viewer: Any, ndim: int = 3) -> list[int]:
    """Array axes in the order Napari is currently displaying them.

    ``viewer.dims.order`` is rarely the identity here — nvitk sets it so a volume
    opens axially — and a plane's geometry is expressed in *that* order, not in
    array-axis order.
    """
    try:
        order = [int(a) for a in viewer.dims.displayed]
    except Exception:
        order = []
    if len(order) != ndim or sorted(order) != list(range(ndim)):
        return list(range(ndim))
    return order


def _plane_geometry(
    axis: int,
    position: tuple[int, int, int],
    shape: tuple[int, ...],
    displayed: list[int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(position, normal)`` for a plane cutting *axis*, in displayed-dims order.

    Napari documents both as "defined in sliced data coordinates (currently
    displayed dims)". Building them in array-axis order instead puts the plane
    perpendicular to the wrong axis whenever ``dims.order`` is a permutation —
    which is why the axial and sagittal planes came out swapped while the coronal
    one (the axis a swap leaves alone) looked correct.
    """
    slot = displayed.index(int(axis))
    normal = [0.0, 0.0, 0.0]
    normal[slot] = 1.0
    point = [
        float(position[data_axis]) if data_axis == int(axis) else float(shape[data_axis]) / 2.0
        for data_axis in displayed
    ]
    return tuple(point), tuple(normal)


def remove_ortho_planes(viewer: Any, source_name: str) -> int:
    """Drop every 3D plane layer nvitk added for *source_name*; returns how many."""
    removed = 0
    for layer in list(getattr(viewer, "layers", []) or []):
        name = str(getattr(layer, "name", ""))
        if name.startswith(f"{source_name}{_PLANE_LAYER_SUFFIX}"):
            try:
                viewer.layers.remove(layer)
                removed += 1
            except Exception:
                pass
    return removed


def sync_ortho_planes(
    viewer: Any,
    layer: Any,
    position: tuple[int, int, int],
    *,
    axes: tuple[int, ...] = (0, 1, 2),
) -> list[Any]:
    """Show *layer* as three ``depiction="plane"`` slices at *position* on the canvas.

    Napari renders a plane-depicted Image as a single slab through the volume, so
    three of them at the crosshair give the 3D view the same three cuts the 2D
    views show. Existing plane layers are moved rather than recreated — rebuilding
    them on every scroll re-uploads the volume to the GPU each time.
    """
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    source_name = str(getattr(layer, "name", "volume"))
    by_name = {str(getattr(l, "name", "")): l for l in viewer.layers}
    wanted = [_plane_layer_name(source_name, a) for a in axes]
    # Moving existing planes needs only their shape, not the volume: re-deriving
    # the array on every scroll step is what made scrolling crawl.
    if all(name in by_name for name in wanted):
        shape = tuple(int(v) for v in by_name[wanted[0]].data.shape[-3:])
        displayed = displayed_axes(viewer, len(shape))
        out = []
        for axis, name in zip(axes, wanted):
            point, normal = _plane_geometry(axis, position, shape, displayed)
            existing = by_name[name]
            existing.plane = {"position": point, "normal": normal, "thickness": 1.0}
            out.append(existing)
        return out

    data = to_numpy(label_source_data(layer) if is_label_like_layer(layer) else layer.data)
    if data.ndim != 3:
        raise ValueError("3D planes need a 3D layer.")

    spatial = layer_spatial_kwargs(layer)
    # Adding a layer makes it active, which would re-bind anything watching the
    # active layer — including the panel that asked for these planes.
    try:
        previously_active = viewer.layers.selection.active
    except Exception:
        previously_active = None

    displayed = displayed_axes(viewer, data.ndim)
    out: list[Any] = []
    for axis in axes:
        name = _plane_layer_name(source_name, axis)
        existing = by_name.get(name)
        point, normal = _plane_geometry(axis, position, data.shape, displayed)
        if existing is None:
            existing = viewer.add_image(
                data,
                name=name,
                depiction="plane",
                rendering="mip",
                blending="additive",
                opacity=0.9,
                **spatial,
            )
        existing.plane = {"position": point, "normal": normal, "thickness": 1.0}
        out.append(existing)

    if previously_active is not None:
        try:
            viewer.layers.selection.active = previously_active
        except Exception:
            pass
    return out


def _box_layer_name(source_name: str, axis: int) -> str:
    """Name of the bounding-box outline layer for *axis*."""
    return f"{source_name}{_BOX_LAYER_SUFFIX}_{axis}"


def remove_ortho_boxes(viewer: Any, source_name: str) -> int:
    """Drop any slice-outline layers belonging to *source_name*; returns how many."""
    wanted = {_box_layer_name(source_name, a) for a in (0, 1, 2)}
    removed = 0
    for layer in list(getattr(viewer, "layers", []) or []):
        if str(getattr(layer, "name", "")) in wanted:
            try:
                viewer.layers.remove(layer)
                removed += 1
            except Exception:
                pass
    return removed


def slice_box_corners(axis: int, index: int, shape: tuple[int, ...]) -> np.ndarray:
    """The four corners of the slice at *index* along *axis*, in voxel coordinates."""
    others = [a for a in range(3) if a != int(axis)]
    lo_a, hi_a = -0.5, float(shape[others[0]]) - 0.5
    lo_b, hi_b = -0.5, float(shape[others[1]]) - 0.5
    corners = np.zeros((4, 3), dtype=float)
    corners[:, int(axis)] = float(index)
    for k, (a, b) in enumerate(((lo_a, lo_b), (lo_a, hi_b), (hi_a, hi_b), (hi_a, lo_b))):
        corners[k, others[0]] = a
        corners[k, others[1]] = b
    return corners


def sync_ortho_boxes(
    viewer: Any,
    layer: Any,
    position: tuple[int, int, int],
    *,
    axes: tuple[int, ...] = (0, 1, 2),
) -> list[Any]:
    """Outline each slice on the canvas instead of drawing the slice itself.

    A plane-depicted Image re-uploads the whole volume to the GPU, and three of
    them hide most of what is behind them. An outline says where the cut is
    without obscuring the anatomy or costing a texture, which is what is wanted
    when the 3D view is there to show the vessels rather than the slices.
    """
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    source_name = str(getattr(layer, "name", "volume"))
    # Read the shape off the array, never through ``asarray``: the layer's data may
    # live on the GPU, and materialising a whole volume on the host to learn how
    # big it is costs a full copy per crosshair move.
    data = getattr(layer, "data", None)
    shape = getattr(data, "shape", None)
    if shape is None:
        # Multiscale layers hold a list of arrays; the first is the full grid.
        try:
            shape = data[0].shape
        except (TypeError, IndexError, KeyError, AttributeError):
            raise ValueError("Slice outlines need a layer with a 3D array.") from None
    shape = tuple(int(v) for v in shape)[-3:]
    if len(shape) != 3:
        raise ValueError("Slice outlines need a 3D layer.")
    by_name = {str(getattr(l, "name", "")): l for l in viewer.layers}
    spatial = layer_spatial_kwargs(layer)

    try:
        previously_active = viewer.layers.selection.active
    except Exception:
        previously_active = None

    out: list[Any] = []
    for axis in axes:
        name = _box_layer_name(source_name, axis)
        corners = slice_box_corners(axis, int(position[axis]), shape)
        existing = by_name.get(name)
        colour = BOX_AXIS_COLORS[int(axis) % len(BOX_AXIS_COLORS)]
        if existing is None:
            existing = viewer.add_shapes(
                [corners],
                shape_type="polygon",
                name=name,
                edge_color=colour,
                face_color="transparent",
                edge_width=_BOX_EDGE_WIDTH,
                **spatial,
            )
            existing.editable = False
        else:
            existing.data = [corners]
        out.append(existing)

    if previously_active is not None:
        try:
            viewer.layers.selection.active = previously_active
        except Exception:
            pass
    return out


def _scene_point(layer: Any, displayed: list[int], data_point: list[float]) -> np.ndarray:
    """A data-space point in vispy *scene* coordinates, the way Napari computes them.

    Deliberately mirrors ``VispyBaseLayer._on_matrix_change``: the layer transform
    is sliced to the displayed dims, then the axes are reversed for vispy. Deriving
    it any other way means guessing at a convention, and the two differ exactly
    when ``dims.order`` is a permutation — which is nvitk's normal case.
    """
    transform = layer._transforms.simplified.set_slice(list(displayed))
    ordered = [float(data_point[d]) for d in displayed]
    world = np.asarray(transform(ordered), dtype=float)
    # Napari reverses the axes when handing the transform to vispy.
    return world[::-1]


def clip_geometry(
    layer: Any,
    axis: int,
    index: int,
    side: str,
    displayed: list[int] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(position, normal)`` for a cut at *index* along *axis*, in Napari's own frame.

    Napari converts a clipping plane for vispy with a plain component reversal
    (``as_array()[..., ::-1]``), but the node's transform is built from the layer
    sliced to the *displayed* dims and then reversed — so vispy's scene axes are
    ``reversed(displayed)``, not ``reversed(range(3))``. A plane built in raw axis
    order therefore cuts the wrong axis whenever ``dims.order`` is a permutation.

    Both vectors are computed in scene coordinates through Napari's own transform
    and then pre-reversed, so Napari's reversal restores them exactly.
    """
    shape = tuple(int(v) for v in np.asarray(layer.data).shape[-3:])
    order = list(displayed) if displayed else list(range(3))

    cut = [float(v) / 2.0 for v in shape]
    cut[int(axis)] = float(index)
    # A step towards the half that "above" keeps, so the normal comes out of the
    # geometry rather than out of an assumed sign.
    kept = list(cut)
    kept[int(axis)] += 1.0 if str(side) == "above" else -1.0

    scene_cut = _scene_point(layer, order, cut)
    scene_kept = _scene_point(layer, order, kept)
    direction = scene_kept - scene_cut
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        # A degenerate axis: fall back to a unit normal on the matching scene axis.
        direction = np.zeros(3)
        direction[len(order) - 1 - order.index(int(axis))] = 1.0
        length = 1.0
    normal = direction / length

    # The shader keeps ``dot(loc - position, normal) >= 0``: the side the normal
    # points towards. Un-reverse both, so Napari's own reversal lands them right.
    return (
        tuple(float(v) for v in scene_cut[::-1]),
        tuple(float(v) for v in normal[::-1]),
    )


def apply_clip(
    layer: Any,
    axis: int,
    index: int,
    side: str,
    displayed: list[int] | None = None,
) -> None:
    """Cut *layer* at *index* along *axis* so one side of the slice is not drawn.

    ``side="below"`` keeps the half with lower indices, ``"above"`` the higher, and
    anything else clears the cut. This is the "see inside" control: the volume is
    opened at the plane the 2D view is showing, so the interior is visible instead
    of only the outer surface.
    """
    from napari.layers.utils.plane import ClippingPlane

    if str(side) not in ("below", "above"):
        layer.experimental_clipping_planes = []
        return

    position, normal = clip_geometry(layer, axis, index, side, displayed)
    layer.experimental_clipping_planes = [
        ClippingPlane(position=position, normal=normal, enabled=True)
    ]


def clear_clip(layer: Any) -> None:
    """Remove any clipping planes nvitk put on *layer*."""
    try:
        layer.experimental_clipping_planes = []
    except Exception:
        pass


class OrthoViewerPanel(QWidget):
    """Axial / coronal / sagittal views of the active layer, plus 3D plane controls."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        """Build the 2x2 grid: three orthogonal views and the 3D controls."""
        super().__init__(parent)
        self._viewer = viewer
        self._layer: Any | None = None
        self._data: np.ndarray | None = None
        self._views: list[AxisView] = []
        self._position: list[int] = [0, 0, 0]
        self._spacing: tuple[float, ...] = (1.0, 1.0, 1.0)
        self._contrast: tuple[float, float] | None = None
        self._lut: dict[int, np.ndarray] | None = None
        self._cache: SliceCache | None = None
        self._overlay: Any | None = None
        self._overlay_data: np.ndarray | None = None
        self._overlay_lut: dict[int, np.ndarray] | None = None
        self._overlay_contrast: tuple[float, float] | None = None
        self._overlay_cache: SliceCache | None = None
        #: ``(layer, callback)`` for the contrast subscriptions, so a rebind can
        #: drop the previous one instead of stacking callbacks on old layers.
        self._contrast_sub: tuple[Any, Any] | None = None
        self._overlay_contrast_sub: tuple[Any, Any] | None = None
        #: Last slice index each view rendered, so an unchanged view is not redrawn.
        self._rendered: dict[int, int] = {}

        # Pushing planes and clipping planes to the canvas re-uploads volumes, which
        # is far too heavy to do on every step of a scroll. Coalesce them.
        self._canvas_timer = QTimer(self)
        self._canvas_timer.setSingleShot(True)
        self._canvas_timer.setInterval(_CANVAS_SYNC_MS)
        self._canvas_timer.timeout.connect(self._sync_canvas)

        self._status = QLabel("Select a 3D image or labels layer.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")

        self._slice_views: list[SliceView] = []
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(SPACE)
        for cell, (row, col) in enumerate([(0, 0), (0, 1), (1, 0)]):
            view = SliceView(AxisView(axis=cell, title="—", rows=0, cols=1))
            view.sliceChanged.connect(self._on_slice_changed)
            view.pixelPicked.connect(
                lambda row, col, index=cell: self._on_pixel_picked(index, row, col)
            )
            grid.addWidget(view, row, col)
            self._slice_views.append(view)
        grid.addWidget(self._build_controls(), 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._status)
        root.addLayout(grid, stretch=1)

    def _build_controls(self) -> QWidget:
        """The fourth quadrant: what the 3D canvas should show."""
        card_overlay = QHBoxLayout()
        card_overlay.setSpacing(SPACE_TIGHT)
        overlay_label = QLabel("Overlay")
        overlay_label.setStyleSheet(f"color: {COLOR_MUTED};")
        self._overlay_combo = QComboBox()
        self._overlay_combo.setToolTip(
            "Draw a second layer on top of this one — a segmentation over its raw "
            "image, say. Labels keep their own colours and leave the base showing "
            "wherever they are empty."
        )
        self._overlay_combo.currentIndexChanged.connect(lambda _i: self._on_overlay_changed())
        card_overlay.addWidget(overlay_label)
        card_overlay.addWidget(self._overlay_combo, stretch=1)

        card = Card("3D view")
        card.add_layout(card_overlay)

        self._show_planes = QCheckBox("Show the three slices in 3D")
        self._show_planes.setToolTip(
            "Push the three cuts onto the Napari canvas, so the 3D view shows the "
            "same slices these panels do."
        )
        self._show_planes.toggled.connect(self._on_show_planes)
        card.add(self._show_planes)

        style_row = QHBoxLayout()
        style_row.setSpacing(SPACE_TIGHT)
        style_label = QLabel("As")
        style_label.setStyleSheet(f"color: {COLOR_MUTED};")
        self._plane_style = QComboBox()
        self._plane_style.addItem("slice image", "image")
        self._plane_style.addItem("outline only", "box")
        self._plane_style.setToolTip(
            "“Slice image” draws the cut itself. “Outline only” draws just its "
            "border — red, green and blue for the three axes — which leaves the "
            "anatomy behind it visible and costs no texture upload."
        )
        self._plane_style.currentIndexChanged.connect(lambda _i: self._on_plane_style())
        style_row.addWidget(style_label)
        style_row.addWidget(self._plane_style, stretch=1)
        card.add_layout(style_row)

        card.add(section_heading("See inside"))
        hint = QLabel(
            "Cut the volume at one view's slice and drop everything on one side of it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        card.add(hint)

        clip_row = QHBoxLayout()
        clip_row.setSpacing(SPACE_TIGHT)
        self._clip_axis = QComboBox()
        self._clip_side = QComboBox()
        self._clip_side.addItem("off", "off")
        self._clip_side.addItem("keep below", "below")
        self._clip_side.addItem("keep above", "above")
        self._clip_axis.currentIndexChanged.connect(lambda _i: self._apply_clip())
        self._clip_side.currentIndexChanged.connect(lambda _i: self._apply_clip())
        clip_row.addWidget(self._clip_axis, stretch=1)
        clip_row.addWidget(self._clip_side, stretch=1)
        card.add_layout(clip_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACE_TIGHT)
        self._btn_centre = QPushButton("Centre crosshair")
        self._btn_centre.clicked.connect(self._centre_crosshair)
        self._btn_3d = QPushButton("3D canvas")
        self._btn_3d.setToolTip("Switch the Napari canvas to its 3D view.")
        self._btn_3d.clicked.connect(self._show_3d)
        btn_row.addWidget(self._btn_centre)
        btn_row.addWidget(self._btn_3d)
        card.add_layout(btn_row)
        card.body().addStretch(1)
        return card

    # ── binding ──────────────────────────────────────────────────────────────

    def refresh_from_layer(self, layer: Any | None) -> None:
        """Bind *layer* and redraw all three views, or show why it cannot be shown."""
        usable = layer is not None and getattr(layer, "data", None) is not None
        if usable and int(getattr(layer.data, "ndim", 0)) != 3:
            usable = False
        if not usable:
            self._layer = None
            self._data = None
            self._status.setText("Select a 3D image or labels layer.")
            for view in self._slice_views:
                view.set_slice(None, index=0, count=1)
            return

        # Re-binding the layer already shown (a refresh, a selection round-trip)
        # keeps the crosshair where the user put it.
        same_layer = layer is self._layer and self._data is not None
        self._layer = layer
        self._data = to_numpy(
            label_source_data(layer) if is_label_like_layer(layer) else layer.data
        )
        self._views = _axis_views(layer)
        self._rendered = {}
        # Windowed once for the volume: per-slice percentiles both cost more and
        # make the same tissue change brightness as you scroll.
        self._cache = SliceCache(self._data)
        if is_label_like_layer(layer):
            self._contrast = None
            self._lut = label_lut(layer, unique_layer_labels(self._data))
        else:
            self._contrast = layer_contrast(layer, self._data)
            self._lut = None
        self._watch_contrast(layer)
        spacing = layer_spacing(layer)
        self._spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))[:3]
        if len(self._spacing) < 3:
            self._spacing = (1.0, 1.0, 1.0)
        if not same_layer or len(self._position) != self._data.ndim:
            self._position = [int(s) // 2 for s in self._data.shape]
        else:
            self._position = [
                int(np.clip(p, 0, int(n) - 1)) for p, n in zip(self._position, self._data.shape)
            ]

        self._clip_axis.blockSignals(True)
        self._clip_axis.clear()
        for view in self._views:
            self._clip_axis.addItem(view.title, view.axis)
        self._clip_axis.blockSignals(False)

        self._refresh_overlay_choices()

        name = getattr(layer, "name", "layer")
        shape = " x ".join(str(int(s)) for s in self._data.shape)
        self._status.setText(f"{name} - {shape} voxels")
        self._redraw(force=True)

    def _watch_contrast(self, layer: Any, *, overlay: bool = False) -> None:
        """Redraw when *layer*'s window moves, and drop the previous subscription.

        Without the unsubscribe the panel would keep redrawing for every layer it
        has ever been pointed at, and each rebind would add another callback to
        the same layer.
        """
        key = "_overlay_contrast_sub" if overlay else "_contrast_sub"
        previous = getattr(self, key, None)
        if previous is not None:
            old_layer, callback = previous
            try:
                old_layer.events.contrast_limits.disconnect(callback)
            except Exception:
                pass
            setattr(self, key, None)
        if layer is None or is_label_like_layer(layer):
            return

        def _changed(_event: Any = None, _overlay: bool = overlay) -> None:
            """Pick the layer's new window up and repaint."""
            if _overlay:
                if self._overlay is None or self._overlay_data is None:
                    return
                self._overlay_contrast = layer_contrast(self._overlay, self._overlay_data)
            else:
                if self._layer is None or self._data is None:
                    return
                self._contrast = layer_contrast(self._layer, self._data)
            self._redraw(force=True)

        try:
            layer.events.contrast_limits.connect(_changed)
            setattr(self, key, (layer, _changed))
        except Exception:
            pass

    def _render_view(self, view: AxisView) -> np.ndarray:
        """The RGB image for *view* at the current crosshair, overlay included."""
        index = self._position[view.axis]
        rgb = slice_to_rgb(
            self._layer, self._data, view.axis, index, view,
            contrast=self._contrast, lut=self._lut, cache=self._cache,
        )
        if self._overlay_data is not None and self._overlay_data.shape == self._data.shape:
            over = slice_to_rgb(
                self._overlay, self._overlay_data, view.axis, index, view,
                contrast=self._overlay_contrast, lut=self._overlay_lut,
                cache=self._overlay_cache,
            )
            rgb = blend_overlay(rgb, over, _OVERLAY_OPACITY)
        return rgb

    def _redraw(self, *, force: bool = False) -> None:
        """Refresh the views, re-rendering only those whose slice actually moved.

        A crosshair move changes at most two of the three slices; redrawing all of
        them on every scroll step is wasted work on a large volume.
        """
        if self._data is None or self._layer is None:
            return
        for widget, view in zip(self._slice_views, self._views):
            widget._view = view
            widget._title.setText(view.title)
            index = self._position[view.axis]
            crosshair = view.to_pixel(self._position, self._data.shape)
            if force or self._rendered.get(view.axis) != index:
                self._rendered[view.axis] = index
                aspect = self._spacing[view.rows] / max(self._spacing[view.cols], 1e-6)
                widget.set_slice(
                    self._render_view(view),
                    index=index,
                    count=int(self._data.shape[view.axis]),
                    aspect=aspect,
                    crosshair=crosshair,
                )
            else:
                # Same slice, new crosshair: repaint the lines, keep the image.
                widget.set_crosshair(crosshair)

    # ── interaction ──────────────────────────────────────────────────────────

    def _on_slice_changed(self, axis: int, index: int) -> None:
        """Move the crosshair along *axis* and refresh the other two views."""
        if self._data is None:
            return
        self._position[int(axis)] = int(index)
        self._redraw()
        self._canvas_timer.start()

    def _on_pixel_picked(self, view_index: int, row: int, col: int) -> None:
        """Place the crosshair from a click inside the view at *view_index*."""
        if self._data is None or view_index >= len(self._views):
            return
        view = self._views[view_index]
        row_axis, row_value, col_axis, col_value = view.to_voxel(row, col, self._data.shape)
        self._position[row_axis] = row_value
        self._position[col_axis] = col_value
        self._redraw()
        self._canvas_timer.start()

    def _centre_crosshair(self) -> None:
        """Put the crosshair back at the middle of the volume."""
        if self._data is None:
            return
        self._position = [int(s) // 2 for s in self._data.shape]
        self._redraw()
        self._canvas_timer.start()

    def _show_3d(self) -> None:
        """Switch the Napari canvas to its 3D view."""
        try:
            self._viewer.dims.ndisplay = 3
        except Exception:
            pass

    def _plane_mode(self) -> str:
        """``"image"`` or ``"box"`` — how the cuts are drawn on the 3D canvas."""
        return str(self._plane_style.currentData() or "image")

    def _on_show_planes(self, enabled: bool) -> None:
        """Add or remove the 3D cut layers on the canvas."""
        if self._layer is None:
            return
        name = str(getattr(self._layer, "name", "volume"))
        if not enabled:
            remove_ortho_planes(self._viewer, name)
            remove_ortho_boxes(self._viewer, name)
            return
        try:
            self._draw_cuts()
            self._show_3d()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not add 3D cuts: {exc}")
            self._show_planes.blockSignals(True)
            self._show_planes.setChecked(False)
            self._show_planes.blockSignals(False)

    def _on_plane_style(self) -> None:
        """Swap between drawing the slices and outlining them."""
        if self._layer is None or not self._show_planes.isChecked():
            return
        # Drop the other representation, or both would be on the canvas at once.
        name = str(getattr(self._layer, "name", "volume"))
        if self._plane_mode() == "box":
            remove_ortho_planes(self._viewer, name)
        else:
            remove_ortho_boxes(self._viewer, name)
        try:
            self._draw_cuts()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not redraw the 3D cuts: {exc}")

    def _draw_cuts(self) -> None:
        """Push the three cuts to the canvas in the selected style."""
        if self._plane_mode() == "box":
            sync_ortho_boxes(self._viewer, self._layer, tuple(self._position))
        else:
            sync_ortho_planes(self._viewer, self._layer, tuple(self._position))

    def _sync_canvas(self) -> None:
        """Push the crosshair to the 3D planes and the clip, once per settle."""
        self._sync_planes()
        self._apply_clip()

    def refresh_overlay_choices(self) -> None:
        """Re-list the overlay candidates, keeping the current pick if it survives.

        Called whenever the viewer's layer list changes: a mask produced *after*
        the panel was opened would otherwise never appear, which left the picker
        stuck on "none".
        """
        self._refresh_overlay_choices()

    def _overlay_candidate(self, other: Any) -> bool:
        """True if *other* can be drawn over the bound layer."""
        if other is self._layer or self._data is None:
            return False
        if _PLANE_LAYER_SUFFIX in str(getattr(other, "name", "")):
            return False
        data = getattr(other, "data", None)
        shape = getattr(data, "shape", None)
        if shape is None:
            # Multiscale layers hold a list of arrays; the first is the full grid.
            try:
                shape = getattr(data[0], "shape", None)
            except (TypeError, IndexError, KeyError):
                shape = None
        if shape is None:
            return False
        # Same voxel grid, ignoring any leading singleton/time axes.
        return tuple(int(v) for v in shape)[-3:] == tuple(int(v) for v in self._data.shape)[-3:]

    def _refresh_overlay_choices(self) -> None:
        """List the layers that could sit on top of the bound one."""
        previous = str(self._overlay_combo.currentData() or "")
        self._overlay_combo.blockSignals(True)
        self._overlay_combo.clear()
        self._overlay_combo.addItem("none", "")
        for other in getattr(self._viewer, "layers", []) or []:
            if self._overlay_candidate(other):
                name = str(getattr(other, "name", ""))
                self._overlay_combo.addItem(name, name)
        index = self._overlay_combo.findData(previous)
        self._overlay_combo.setCurrentIndex(max(index, 0))
        self._overlay_combo.blockSignals(False)
        self._bind_overlay(str(self._overlay_combo.currentData() or ""))

    def _bind_overlay(self, name: str) -> None:
        """Cache the overlay layer's data, colours and window."""
        self._overlay = None
        self._overlay_data = None
        self._overlay_lut = None
        self._overlay_contrast = None
        self._overlay_cache = None
        if not name:
            return
        layer = next(
            (l for l in self._viewer.layers if str(getattr(l, "name", "")) == name), None
        )
        if layer is None:
            return
        self._overlay = layer
        self._overlay_data = to_numpy(
            label_source_data(layer) if is_label_like_layer(layer) else layer.data
        )
        self._overlay_cache = SliceCache(self._overlay_data)
        if is_label_like_layer(layer):
            self._overlay_lut = label_lut(layer, unique_layer_labels(self._overlay_data))
        else:
            self._overlay_contrast = layer_contrast(layer, self._overlay_data)
        self._watch_contrast(layer, overlay=True)

    def _on_overlay_changed(self) -> None:
        """Rebind the overlay and redraw every view with it."""
        # The layer being dropped keeps whatever cut it was given otherwise, and
        # nothing in the panel would ever take it off again.
        previous = self._overlay
        self._bind_overlay(str(self._overlay_combo.currentData() or ""))
        if previous is not None and previous is not self._overlay:
            clear_clip(previous)
        self._redraw(force=True)
        self._apply_clip()

    def _sync_planes(self) -> None:
        """Move the 3D cuts to the crosshair, if they are showing."""
        if self._layer is None or not self._show_planes.isChecked():
            return
        try:
            self._draw_cuts()
        except Exception:
            pass

    def _apply_clip(self) -> None:
        """Apply (or clear) the see-inside cut on the bound layer and its overlay.

        The overlay is cut with it. Cutting only the base leaves a segmentation
        floating in front of the opened volume, covering the very interior the cut
        was made to expose — and reading as if the mask extended past the tissue.
        """
        if self._layer is None:
            return
        side = str(self._clip_side.currentData() or "off")
        axis_data = self._clip_axis.currentData()
        axis = int(axis_data) if axis_data is not None else 0
        displayed = displayed_axes(self._viewer)
        targets = [self._layer]
        if self._overlay is not None and self._overlay is not self._layer:
            targets.append(self._overlay)
        for target in targets:
            try:
                apply_clip(target, axis, self._position[axis], side, displayed)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Could not clip {getattr(target, 'name', '?')}: {exc}")

    def position(self) -> tuple[int, int, int]:
        """Current crosshair, as voxel indices in array-axis order."""
        return tuple(int(v) for v in self._position)


def open_ortho_views(viewer: Any, layer: Any | None = None) -> OrthoViewerPanel:
    """Open the orthogonal-views dock, or bring the existing one back into view.

    Kept on the viewer so the quick-access button and the Visualization tool both
    reach the same panel instead of stacking up docks.
    """
    panel = getattr(viewer, "_nvitk_ortho_panel", None)
    if panel is None:
        panel = OrthoViewerPanel(viewer)
        attach_ortho_dock(viewer, panel)
        try:
            viewer._nvitk_ortho_panel = panel
        except Exception:
            pass
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
    dock = viewer.window.add_dock_widget(panel, area="left", name="Orthogonal views")
    panel._nvitk_dock = dock

    def _refresh(_event: Any = None) -> None:
        """Rebind the panel to whichever layer is active.

        The panel's own plane layers are skipped: they are a *view* of the bound
        volume, so following the selection onto one would rebind the panel to its
        own output and reset the crosshair.
        """
        active = viewer.layers.selection.active if viewer.layers else None
        if active is not None and _PLANE_LAYER_SUFFIX in str(getattr(active, "name", "")):
            return
        panel.refresh_from_layer(active)

    def _layers_changed(_event: Any = None) -> None:
        """Keep the overlay picker in step with the viewer's layer list."""
        panel.refresh_overlay_choices()

    viewer.layers.selection.events.active.connect(_refresh)
    viewer.layers.events.inserted.connect(_layers_changed)
    viewer.layers.events.removed.connect(_layers_changed)
    _refresh()
    return dock


__all__ = [
    "AxisView",
    "OrthoViewerPanel",
    "SliceView",
    "apply_clip",
    "attach_ortho_dock",
    "clear_clip",
    "blend_overlay",
    "clip_geometry",
    "SliceCache",
    "label_lut",
    "open_ortho_views",
    "displayed_axes",
    "remove_ortho_planes",
    "slice_to_rgb",
    "sync_ortho_planes",
    "volume_contrast",
]
