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
from typing import Any, Sequence

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
from nvitk.gui.viz.left_dock import attach_left_inspection_dock
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
DOCK_OBJECT_NAME = "nvitk_ortho_dock"

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

#: Coalescing window for colour/opacity bursts.
_STYLE_SYNC_MS = 30

#: Coalescing window for layer-list churn.
_REBUILD_SYNC_MS = 60

#: Zoom bounds and the factor one Ctrl+wheel notch applies. The floor is "fit to
#: the view", which is what the panels do without a zoom at all.
_MIN_ZOOM = 1.0
_MAX_ZOOM = 12.0
_ZOOM_STEP = 1.25

#: How close to a crosshair line a press must be to grab it, and how far from the
#: centre it must be for that grab to mean "rotate" rather than "move".
_HANDLE_TOL_PX = 10.0
_HANDLE_MIN_FRACTION = 0.55

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


def _unit(vector: np.ndarray) -> np.ndarray:
    """*vector* scaled to unit length, or unchanged when it is degenerate."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


@dataclass(frozen=True)
class PlaneFrame:
    """One view's plane: unit row/column directions and its normal, in millimetres.

    Millimetres, not voxels: on anisotropic spacing a rotation applied to raw
    array-axis vectors shears the plane rather than turning it, and the angle the
    user dragged is not the angle they get.
    """

    row: np.ndarray
    col: np.ndarray
    normal: np.ndarray

    def is_close_to(self, other: PlaneFrame, tol: float = 1e-9) -> bool:
        """True when this frame is the same plane and orientation as *other*."""
        return bool(
            np.allclose(self.row, other.row, atol=tol)
            and np.allclose(self.col, other.col, atol=tol)
            and np.allclose(self.normal, other.normal, atol=tol)
        )


def base_frame(view: AxisView, spacing: Sequence[float]) -> PlaneFrame:
    """The axis-aligned frame *view* starts from, in millimetre space."""
    sp = np.asarray(spacing, dtype=float)[:3]
    row = np.zeros(3)
    row[view.rows] = -1.0 if view.flip_rows else 1.0
    col = np.zeros(3)
    col[view.cols] = -1.0 if view.flip_cols else 1.0
    row_mm = _unit(row * sp)
    col_mm = _unit(col * sp)
    return PlaneFrame(row_mm, col_mm, _unit(np.cross(row_mm, col_mm)))


def rotation_about(axis_mm: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix turning by *angle_rad* about a unit axis."""
    k = _unit(np.asarray(axis_mm, dtype=float))
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * kx + (1.0 - np.cos(angle_rad)) * (kx @ kx)


def rotate_frame(frame: PlaneFrame, axis_mm: np.ndarray, angle_rad: float) -> PlaneFrame:
    """*frame* turned about *axis_mm*, re-normalised against drift."""
    rot = rotation_about(axis_mm, float(angle_rad))
    return PlaneFrame(
        _unit(rot @ frame.row), _unit(rot @ frame.col), _unit(rot @ frame.normal)
    )


def oblique_slice(
    data: np.ndarray,
    *,
    center_vox: Sequence[float],
    anchor_px: tuple[float, float],
    frame: PlaneFrame,
    shape: tuple[int, int],
    steps_mm: tuple[float, float],
    spacing: Sequence[float],
    order: int = 1,
) -> np.ndarray:
    """Sample the plane *frame* through *center_vox* onto a ``shape`` pixel grid.

    *anchor_px* is the pixel that lands on *center_vox*. Anchoring on the
    crosshair rather than a corner is what makes an unrotated frame reproduce the
    axis-aligned slice exactly, and keeps the crosshair still on screen while the
    plane turns under it.
    """
    from scipy.ndimage import map_coordinates

    height, width = int(shape[0]), int(shape[1])
    sp = np.asarray(spacing, dtype=float)[:3]
    centre = np.asarray(center_vox, dtype=float)[:3]
    # A step of one pixel is steps_mm along the frame direction, converted back
    # into voxel indices by the spacing.
    step_row = float(steps_mm[0]) * np.asarray(frame.row, dtype=float) / sp
    step_col = float(steps_mm[1]) * np.asarray(frame.col, dtype=float) / sp
    rows = (np.arange(height, dtype=float) - float(anchor_px[0]))[:, None, None]
    cols = (np.arange(width, dtype=float) - float(anchor_px[1]))[None, :, None]
    coords = centre[None, None, :] + rows * step_row[None, None, :] + cols * step_col[None, None, :]
    return map_coordinates(
        data, np.moveaxis(coords, 2, 0), order=int(order), mode="constant", cval=0.0
    )


def line_direction_px(
    view_frame: PlaneFrame,
    other_normal: np.ndarray,
    steps_mm: tuple[float, float],
) -> tuple[float, float] | None:
    """``(d_row, d_col)`` of another plane's trace across this view, in pixels.

    Two planes meet along ``cross(n_a, n_b)``; drawn on this view that is the line
    marking where the other view is cutting. ``None`` when the planes are parallel
    and there is no trace to draw.
    """
    direction = np.cross(np.asarray(view_frame.normal, float), np.asarray(other_normal, float))
    if float(np.linalg.norm(direction)) <= 1e-9:
        return None
    d_row = float(np.dot(direction, view_frame.row)) / max(float(steps_mm[0]), 1e-9)
    d_col = float(np.dot(direction, view_frame.col)) / max(float(steps_mm[1]), 1e-9)
    if abs(d_row) < 1e-12 and abs(d_col) < 1e-12:
        return None
    norm = float(np.hypot(d_row, d_col))
    return (d_row / norm, d_col / norm)


#: Ceiling on cached resampled volumes, mirroring the slice-cache budget. One
#: off-grid layer costs a full copy on the active grid.
_RESAMPLE_BUDGET_BYTES = 512 * 1024 * 1024


@dataclass
class RenderSource:
    """One layer as the panel draws it: data on the active grid, plus its style.

    Mutable on purpose. Opacity, contrast and the colour table change on every
    tick of a slider and must be refreshed without touching ``data`` or ``cache``,
    which are the expensive parts.
    """

    layer: Any
    data: np.ndarray
    cache: SliceCache | None
    is_label: bool
    lut: dict[int, np.ndarray] | None = None
    table: np.ndarray | None = None
    contrast: tuple[float, float] | None = None
    opacity: float = 1.0
    blending: str = "translucent"
    gamma: float = 1.0
    order: int = 1
    resampled: bool = False


#: The only layer types the panel draws. Shapes, Points, Vectors, Surfaces and
#: Tracks are annotations, not imagery: a tool's centerline paths, station markers
#: and cut outlines would otherwise be composited over the very anatomy they are
#: annotating, and hide it.
DRAWN_LAYER_TYPES: tuple[str, ...] = ("Image", "Labels")


def _layer_shape(layer: Any) -> tuple[int, ...] | None:
    """A layer's array shape, read without materialising the array."""
    data = getattr(layer, "data", None)
    shape = getattr(data, "shape", None)
    if shape is None:
        # Multiscale layers hold a list of arrays; the first is the full grid.
        try:
            shape = data[0].shape
        except (TypeError, IndexError, KeyError, AttributeError):
            return None
    try:
        return tuple(int(v) for v in shape)
    except (TypeError, ValueError):
        return None


def is_drawable_layer(layer: Any) -> bool:
    """Whether the panel should draw *layer*, decided without touching its data.

    Deliberately cheap: this runs for every layer in the viewer every time the
    layer list changes, and the tools add and move overlay layers constantly.
    Materialising a volume here — which is what reading the data to find out
    costs — made every tool that adds a layer pay for every layer on screen.
    """
    if type(layer).__name__ not in DRAWN_LAYER_TYPES:
        return False
    name = str(getattr(layer, "name", ""))
    if _PLANE_LAYER_SUFFIX in name or _BOX_LAYER_SUFFIX in name:
        return False
    shape = _layer_shape(layer)
    return shape is not None and len(shape) >= 3


def _layer_volume(layer: Any) -> np.ndarray | None:
    """*layer*'s 3D host array, or ``None`` when it has none to draw.

    One place for the label-source choice, the multiscale unwrap and the 4D
    reduction, all of which were previously repeated at each call site.
    """
    data = getattr(layer, "data", None)
    if data is None:
        return None
    if not hasattr(data, "shape"):
        # Multiscale layers hold a list of arrays; the first is the full grid.
        try:
            data = data[0]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        arr = to_numpy(label_source_data(layer) if is_label_like_layer(layer) else layer.data)
    except Exception:  # noqa: BLE001
        return None
    arr = np.asarray(arr)
    if arr.ndim > 3:
        # A 4D layer contributes its first volume, the way the rest of the panel
        # treats one.
        arr = arr[(0,) * (arr.ndim - 3)]
    return arr if arr.ndim == 3 else None


def _same_grid(layer: Any, reference: Any) -> bool:
    """Whether *layer* already sits on *reference*'s voxel grid.

    Read off the array object and the affine without materialising either: this
    runs for every layer on every rebind, and the expensive resampler behind it
    re-checks properly anyway.
    """
    from nvitk.gui.core.spatial import layer_affine

    def _shape(obj: Any) -> tuple[int, ...] | None:
        """Trailing three dimensions of a layer's data, without copying it."""
        data = getattr(obj, "data", None)
        shape = getattr(data, "shape", None)
        if shape is None:
            try:
                shape = data[0].shape
            except (TypeError, IndexError, KeyError, AttributeError):
                return None
        return tuple(int(v) for v in shape)[-3:]

    if _shape(layer) != _shape(reference):
        return False
    a, b = layer_affine(layer), layer_affine(reference)
    if a is None or b is None:
        return a is b or (a is None and b is None)
    return bool(np.allclose(a, b, atol=1e-3))


def resample_layer_to(
    layer: Any, reference: Any, data: np.ndarray, *, order: int
) -> np.ndarray | None:
    """*data* put on *reference*'s grid, or ``None`` when it cannot be.

    ``None`` rather than an exception for the case the aligner refuses — a layer
    with no affine and a different shape — because a panel that cannot draw one
    layer should say so and draw the rest.
    """
    from nvitk.core.backend import using
    from nvitk.gui.core.spatial import align_mask_to_reference_layer

    try:
        with using("cpu"):
            _ref, aligned, _resampled = align_mask_to_reference_layer(
                layer, reference, data, order=int(order)
            )
        return np.asarray(to_numpy(aligned.data))
    except Exception:  # noqa: BLE001 — reported by the caller, never raised into paint
        return None


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


# ── RGBA pipeline ─────────────────────────────────────────────────────────────
#: Entries in a tabulated colormap. Matches the size of the texture Napari's own
#: shader samples, so the two agree to within a quantisation step.
_COLORMAP_STEPS = 256


def layer_colormap(layer: Any) -> Any | None:
    """*layer*'s colormap, or ``None`` for a label layer or one without.

    Label colormaps map *integers* and raise on float input, so they are
    deliberately excluded here — labels go through :func:`label_rgba_lut`.
    """
    if is_label_like_layer(layer):
        return None
    return getattr(layer, "colormap", None)


def layer_gamma(layer: Any) -> float:
    """*layer*'s display gamma, defaulting to 1."""
    try:
        return float(getattr(layer, "gamma", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def colormap_table(colormap: Any, gamma: float = 1.0, size: int = _COLORMAP_STEPS) -> np.ndarray:
    """*colormap* sampled into a ``(size, 4)`` uint8 table, with *gamma* folded in.

    Tabulated once rather than mapped per pixel: ``Colormap.map`` over a 250k-voxel
    slice costs milliseconds per redraw, and the GPU does exactly this — samples a
    256-texel texture — so the result matches what the canvas shows.
    """
    ramp = np.linspace(0.0, 1.0, int(size), dtype=np.float32) ** max(float(gamma), 1e-6)
    try:
        rgba = np.asarray(colormap.map(ramp), dtype=np.float32)
    except Exception:  # noqa: BLE001 — an unusable colormap falls back to grey
        rgba = np.repeat(ramp[:, None], 4, axis=1)
        rgba[:, 3] = 1.0
    if rgba.ndim != 2 or rgba.shape[1] < 3:
        rgba = np.repeat(ramp[:, None], 4, axis=1)
        rgba[:, 3] = 1.0
    if rgba.shape[1] == 3:
        rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1), dtype=np.float32)], axis=1)
    return (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)


def image_rgba(
    plane: np.ndarray, contrast: tuple[float, float] | None, table: np.ndarray
) -> np.ndarray:
    """Window a 2D intensity slice and colour it through *table*; ``(H, W, 4)`` uint8."""
    arr = np.asarray(plane, dtype=np.float32)
    lo, hi = contrast if contrast is not None else volume_contrast(arr)
    if hi <= lo:
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    norm = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    index = (norm * (table.shape[0] - 1)).astype(np.uint16)
    return table[index]


def label_rgba_lut(layer: Any, label_ids: list[int]) -> dict[int, np.ndarray]:
    """RGBA uint8 per label id, read from the layer's own colours.

    Alpha is kept, unlike :func:`label_lut`: a label nvitk has hidden is stored
    as fully transparent, and dropping that would paint it solid black instead of
    leaving what is underneath showing.
    """
    lut: dict[int, np.ndarray] = {}
    for lid in label_ids:
        rgba = get_label_color(layer, int(lid))
        lut[int(lid)] = (np.clip(np.asarray(rgba, dtype=float), 0, 1) * 255).astype(np.uint8)
    return lut


def _label_rgba(
    plane: np.ndarray, layer: Any, lut: dict[int, np.ndarray] | None = None
) -> np.ndarray:
    """Colour a 2D label slice to ``(H, W, 4)`` uint8; background transparent."""
    arr = np.rint(np.asarray(plane, dtype=np.float64)).astype(np.int64, copy=False)
    present = [int(v) for v in np.unique(arr) if int(v) != 0]
    if not present:
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    colors = lut if lut is not None else label_rgba_lut(layer, present)
    table = np.zeros((max(present) + 1, 4), dtype=np.uint8)
    for lid in present:
        table[lid] = colors.get(lid, np.array([255, 255, 255, 255], dtype=np.uint8))
    return table[np.clip(arr, 0, table.shape[0] - 1)]


def slice_to_rgba(
    layer: Any,
    data: np.ndarray,
    axis: int,
    index: int,
    view: AxisView | None = None,
    *,
    contrast: tuple[float, float] | None = None,
    table: np.ndarray | None = None,
    lut: dict[int, np.ndarray] | None = None,
    cache: SliceCache | None = None,
) -> np.ndarray:
    """One orthogonal slice as ``(H, W, 4)`` uint8, ready to composite."""
    raw = cache.slice(axis, index) if cache is not None else _slice_of(data, axis, index)
    plane = _oriented(raw, axis, view)
    if is_label_like_layer(layer):
        return _label_rgba(plane, layer, lut)
    if table is None:
        table = colormap_table(layer_colormap(layer), layer_gamma(layer))
    return image_rgba(plane, contrast, table)


def composite(
    planes: Sequence[np.ndarray],
    blendings: Sequence[str],
    opacities: Sequence[float],
) -> np.ndarray:
    """Blend RGBA *planes* bottom-to-top into one RGB uint8 image.

    Walks in layer-list order, the way Napari draws, and implements its blending
    modes (``napari.layers.base._base_constants``). This reproduces the 2D canvas;
    the depth-test difference between ``translucent`` and ``translucent_no_depth``
    only shows in 3D.
    """
    if not len(planes):
        return np.zeros((1, 1, 3), dtype=np.uint8)
    shape = np.asarray(planes[0]).shape[:2]
    out = np.zeros((*shape, 3), dtype=np.float32)
    for plane, blending, opacity in zip(planes, blendings, opacities):
        src = np.asarray(plane, dtype=np.float32)
        if src.shape[:2] != shape:
            continue
        rgb = src[..., :3]
        alpha = (src[..., 3:4] / 255.0) * float(np.clip(opacity, 0.0, 1.0))
        mode = str(blending or "translucent")
        if mode == "opaque":
            out = np.where(alpha > 0, rgb, out)
        elif mode == "additive":
            out = out + rgb * alpha
        elif mode == "minimum":
            out = np.minimum(out, np.where(alpha > 0, rgb, out))
        elif mode == "multiplicative":
            out = out * (rgb / 255.0)
        else:  # translucent, translucent_no_depth
            out = out * (1.0 - alpha) + rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


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
    #: (line index, screen angle in radians) while a crosshair end is dragged.
    handleDragged = Signal(int, float)

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
        #: ``(d_row, d_col, colour)`` per crosshair line. Empty means the plain
        #: horizontal/vertical cross, which is what an unrotated view shows.
        self._lines: list[tuple[float, float, str]] = []
        #: Index of the line whose end is being dragged, if any.
        self._rotating: int | None = None

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

    def set_lines(self, lines: list[tuple[float, float, str]]) -> None:
        """Set the crosshair line directions, in pixel space, with their colours."""
        self._lines = list(lines or [])
        self._repaint()

    def _line_geometry(self) -> list[tuple[float, float, float, float, str]] | None:
        """Each crosshair line as ``(cx, cy, d_x, d_y, colour)`` in canvas pixels."""
        geometry = self._geometry()
        if geometry is None or self._cross is None or self._rgb is None:
            return None
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        row, col = self._cross
        cx = off_x + (col + 0.5) / max(w, 1) * scaled_w
        cy = off_y + (row + 0.5) / max(h, 1) * scaled_h
        out = []
        entries = self._lines or [(1.0, 0.0, COLOR_ACCENT), (0.0, 1.0, COLOR_ACCENT)]
        for d_row, d_col, colour in entries:
            # Pixel directions scale with the drawn size, so a rotated line keeps
            # its angle on screen whatever the zoom or the aspect correction.
            dx = float(d_col) * scaled_w / max(w, 1)
            dy = float(d_row) * scaled_h / max(h, 1)
            norm = float(np.hypot(dx, dy))
            if norm <= 1e-9:
                continue
            out.append((cx, cy, dx / norm, dy / norm, colour))
        return out

    def _handle_at(self, event: Any) -> int | None:
        """Index of the crosshair line whose end is under the cursor, if any."""
        lines = self._line_geometry()
        if not lines:
            return None
        x, y = self._cursor_xy(event)
        reach = max(self._canvas.width(), self._canvas.height()) / 2.0
        best, best_distance = None, _HANDLE_TOL_PX
        for index, (cx, cy, dx, dy, _colour) in enumerate(lines):
            vx, vy = x - cx, y - cy
            along = vx * dx + vy * dy
            # Only the ends grab: near the middle the same drag has to keep
            # meaning "move the crosshair", which is the commoner gesture.
            if abs(along) < _HANDLE_MIN_FRACTION * reach:
                continue
            across = abs(vx * dy - vy * dx)
            if across < best_distance:
                best, best_distance = index, across
        return best

    def _screen_angle(self, event: Any) -> float:
        """Angle of the cursor about the crosshair, in the slice's pixel frame."""
        geometry = self._geometry()
        if geometry is None or self._cross is None or self._rgb is None:
            return 0.0
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        row, col = self._cross
        x, y = self._cursor_xy(event)
        # Back into the slice's own pixel units, so the angle does not depend on
        # the zoom or on the aspect correction the canvas applies.
        d_col = (x - off_x) / max(scaled_w, 1) * w - (col + 0.5)
        d_row = (y - off_y) / max(scaled_h, 1) * h - (row + 0.5)
        return float(np.arctan2(d_row, d_col))

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
        for cx, cy, dx, dy, colour in self._line_geometry() or []:
            pen = QPen(QColor(colour))
            pen.setWidth(1)
            painter.setPen(pen)
            reach = float(canvas.width() + canvas.height())
            painter.drawLine(
                int(round(cx - dx * reach)), int(round(cy - dy * reach)),
                int(round(cx + dx * reach)), int(round(cy + dy * reach)),
            )
            # Mark the ends that can be grabbed, so the gesture is discoverable.
            handle = _HANDLE_MIN_FRACTION * max(canvas.width(), canvas.height()) / 2.0
            for sign in (-1.0, 1.0):
                painter.drawEllipse(
                    int(round(cx + sign * dx * handle)) - 3,
                    int(round(cy + sign * dy * handle)) - 3,
                    6, 6,
                )
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
            handle = self._handle_at(event)
            if handle is not None:
                self._rotating = handle
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseMove:
            if self._drag_from is not None and self._is_pan(event):
                self._pan_by(event)
                return True
            if not event.buttons():
                return False
            if self._rotating is not None:
                self.handleDragged.emit(int(self._rotating), self._screen_angle(event))
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseButtonRelease:
            self._drag_from = None
            self._rotating = None
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
    frame: PlaneFrame | None = None,
    spacing: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """``(point, normal)`` for one cut, in Napari's displayed-dims order.

    Two frames meet here and must not be confused. ``Plane.position`` and
    ``Plane.normal`` are data (voxel-index) coordinates *permuted into displayed
    order* — Napari's own convention. ``PlaneFrame`` holds unit vectors in
    spacing-scaled millimetre space; a direction converts to voxels by dividing
    by the spacing, so the plane normal is the cross product of the two converted
    in-plane directions, permuted **after** the cross product (an odd permutation
    flips a cross product's sign).

    ``frame=None`` reproduces the axis-aligned behaviour exactly.
    """
    if frame is None:
        slot = displayed.index(int(axis))
        normal = [0.0, 0.0, 0.0]
        normal[slot] = 1.0
        point = [
            float(position[data_axis])
            if data_axis == int(axis)
            else float(shape[data_axis]) / 2.0
            for data_axis in displayed
        ]
        return point, normal

    sp = np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0), dtype=float)[:3]
    row_vox = np.asarray(frame.row, dtype=float) / sp
    col_vox = np.asarray(frame.col, dtype=float) / sp
    normal_vox = np.cross(row_vox, col_vox)
    length = float(np.linalg.norm(normal_vox))
    if length <= 1e-12:
        normal_vox = np.zeros(3)
        normal_vox[int(axis)] = 1.0
    else:
        normal_vox = normal_vox / length
    # A turned plane passes through the crosshair, not the volume's middle:
    # rotating about the centre would slide the cut away from what is on screen.
    point = [float(position[data_axis]) for data_axis in displayed]
    normal = [float(normal_vox[data_axis]) for data_axis in displayed]
    return point, normal

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
    frames: Sequence[PlaneFrame] | None = None,
    spacing: Sequence[float] | None = None,
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
        for i, (axis, name) in enumerate(zip(axes, wanted)):
            point, normal = _plane_geometry(
                axis, position, shape, displayed,
                frames[i] if frames is not None and i < len(frames) else None,
                spacing,
            )
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
    for i, axis in enumerate(axes):
        name = _plane_layer_name(source_name, axis)
        existing = by_name.get(name)
        point, normal = _plane_geometry(
            axis, position, data.shape, displayed,
            frames[i] if frames is not None and i < len(frames) else None,
            spacing,
        )
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


def oblique_box_corners(
    center_vox: Sequence[float],
    frame: PlaneFrame,
    half_extent_mm: tuple[float, float],
    spacing: Sequence[float],
) -> np.ndarray:
    """Corners of a turned slice outline, in voxel coordinates.

    The same recipe the CPR plane marker uses: step out along the frame's two
    in-plane directions, converted from millimetres to voxels by the spacing.
    """
    sp = np.asarray(spacing, dtype=float)[:3]
    centre = np.asarray(center_vox, dtype=float)[:3]
    row = np.asarray(frame.row, dtype=float) / sp
    col = np.asarray(frame.col, dtype=float) / sp
    r, c = float(half_extent_mm[0]), float(half_extent_mm[1])
    return np.stack([
        centre - r * row - c * col,
        centre - r * row + c * col,
        centre + r * row + c * col,
        centre + r * row - c * col,
    ]).astype(float)


def sync_ortho_boxes(
    viewer: Any,
    layer: Any,
    position: tuple[int, int, int],
    *,
    axes: tuple[int, ...] = (0, 1, 2),
    frames: Sequence[PlaneFrame] | None = None,
    spacing: Sequence[float] | None = None,
    views: Sequence[AxisView] | None = None,
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
    sp = tuple(float(v) for v in (spacing or (1.0, 1.0, 1.0)))[:3]
    for i, axis in enumerate(axes):
        name = _box_layer_name(source_name, axis)
        frame = frames[i] if frames is not None and i < len(frames) else None
        if frame is None:
            corners = slice_box_corners(axis, int(position[axis]), shape)
        else:
            view = views[i] if views is not None and i < len(views) else None
            rows = int(shape[view.rows]) if view is not None else int(shape[axis])
            cols = int(shape[view.cols]) if view is not None else int(shape[axis])
            # Half the field of view the 2D panel shows, so the outline marks the
            # same extent the user is looking at.
            half = (
                rows * sp[view.rows if view is not None else axis] / 2.0,
                cols * sp[view.cols if view is not None else axis] / 2.0,
            )
            corners = oblique_box_corners(position, frame, half, sp)
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
        #: Every visible layer the panel draws, bottom-to-top in layer-list order.
        self._sources: list[RenderSource] = []
        #: ``(emitter, callback)`` for every per-layer subscription, so a rebuild
        #: can drop them all rather than stacking callbacks on dead layers.
        self._subs: list[tuple[Any, Any]] = []
        #: Volumes resampled onto the active grid, keyed by layer and grid.
        self._resample_cache: dict[tuple, np.ndarray] = {}
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
        self._style_timer.timeout.connect(lambda: self._redraw(force=True))

        # Adding or moving layers arrives in bursts — a tool that drops four
        # overlays on the canvas fires four inserts. Rebuild once when they stop.
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(_REBUILD_SYNC_MS)
        self._rebuild_timer.timeout.connect(self._rebuild_sources)

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
            view.handleDragged.connect(
                lambda line, angle, index=cell: self._on_handle_dragged(index, line, angle)
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
        # No overlay picker: the panel draws every visible layer, so the layer
        # list *is* the control surface. This only reports what that came to.
        self._composite_label = QLabel("—")
        self._composite_label.setWordWrap(True)
        self._composite_label.setStyleSheet(f"color: {COLOR_MUTED};")
        self._composite_label.setToolTip(
            "Every visible layer on this grid is drawn, in the layer list's own "
            "order and with each layer's colormap, opacity and blending. Hide a "
            "layer in the list to take it out."
        )

        card = Card("3D view")
        card.add(self._composite_label)

        self._show_planes = QCheckBox("Show the three slices in 3D")
        self._show_planes.setToolTip(
            "Push the three cuts onto the Napari canvas, so the 3D view shows the "
            "same slices these panels do."
        )
        self._show_planes.toggled.connect(self._on_show_planes)
        card.add(self._show_planes)

        self._show_slice_image = QCheckBox("…and the slice image, not just its outline")
        self._show_slice_image.setToolTip(
            "The coloured outline — red, green and blue for the three axes — is "
            "always drawn: it says where each cut is without hiding what is behind "
            "it. Tick this to draw the slice itself as well, which costs a texture "
            "upload per cut."
        )
        self._show_slice_image.setEnabled(False)
        self._show_slice_image.toggled.connect(lambda _c: self._on_plane_image_toggled())
        card.add(self._show_slice_image)

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
        self._btn_reset_orient = QPushButton("Reset orientation")
        self._btn_reset_orient.setToolTip(
            "Put the three planes back on the volume's own axes, undoing any "
            "rotation made by dragging the crosshair ends."
        )
        self._btn_reset_orient.setEnabled(False)
        self._btn_reset_orient.clicked.connect(self._reset_orientation)
        self._btn_3d = QPushButton("3D canvas")
        self._btn_3d.setToolTip("Switch the Napari canvas to its 3D view.")
        self._btn_3d.clicked.connect(self._show_3d)
        btn_row.addWidget(self._btn_centre)
        btn_row.addWidget(self._btn_reset_orient)
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
        # Per-layer colour state lives on each RenderSource now; this layer is
        # kept only as the geometry anchor everything else is drawn against.
        if not same_layer:
            self._resample_cache.clear()
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

        self._base_frames = [base_frame(view, self._spacing) for view in self._views]
        if not same_layer or len(self._frames) != len(self._base_frames):
            self._frames = list(self._base_frames)

        self._clip_axis.blockSignals(True)
        self._clip_axis.clear()
        for view in self._views:
            self._clip_axis.addItem(view.title, view.axis)
        self._clip_axis.blockSignals(False)

        self._rebuild_sources()

        name = getattr(layer, "name", "layer")
        shape = " x ".join(str(int(s)) for s in self._data.shape)
        self._status.setText(f"{name} - {shape} voxels")
        self._redraw(force=True)

    # ── render sources ───────────────────────────────────────────────────────

    def _candidate_layers(self) -> list[Any]:
        """Layers eligible to be drawn, in the viewer's own order."""
        return [
            layer
            for layer in (getattr(self._viewer, "layers", []) or [])
            if is_drawable_layer(layer)
        ]

    def _reference_key(self) -> tuple:
        """Identity of the grid everything is resampled onto."""
        from nvitk.gui.core.spatial import layer_affine

        affine = layer_affine(self._layer)
        return (
            id(self._layer),
            tuple(self._data.shape) if self._data is not None else (),
            None if affine is None else affine.tobytes(),
        )

    def _aligned_data(self, layer: Any, data: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """``(data on the active grid, was it resampled)``.

        Cached per layer and grid: the aligner is a whole-volume affine transform,
        far too heavy to run per slice, and this way hiding and re-showing an
        off-grid mask costs nothing.
        """
        if self._layer is None or _same_grid(layer, self._layer):
            return data, False
        key = (id(layer), self._reference_key())
        hit = self._resample_cache.get(key)
        if hit is not None:
            return hit, True
        budget = sum(int(v.nbytes) for v in self._resample_cache.values())
        if budget + int(self._data.nbytes) > _RESAMPLE_BUDGET_BYTES:
            self._status.setText(
                f"“{getattr(layer, 'name', '?')}” is on another grid and there is no "
                "room left to resample it; it is not drawn."
            )
            return None, False
        order = 0 if is_label_like_layer(layer) else 1
        out = resample_layer_to(layer, self._layer, data, order=order)
        if out is None:
            self._status.setText(
                f"“{getattr(layer, 'name', '?')}” is on another grid and has no affine "
                "to align it by; it is not drawn."
            )
            return None, False
        self._resample_cache[key] = out
        return out, True

    def _source_for(self, layer: Any) -> RenderSource | None:
        """Build the draw-time description of *layer*, or ``None`` if it cannot be."""
        raw = _layer_volume(layer)
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
            cache=SliceCache(data),
            is_label=is_label,
            opacity=float(getattr(layer, "opacity", 1.0) or 1.0),
            blending=str(getattr(layer, "blending", "translucent")),
            gamma=layer_gamma(layer),
            order=0 if is_label else 1,
            resampled=resampled,
        )
        self._refresh_style(source)
        return source

    def _refresh_style(self, source: RenderSource) -> None:
        """Re-read the colour state of one source, leaving its data alone."""
        layer = source.layer
        source.opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
        source.blending = str(getattr(layer, "blending", "translucent"))
        if source.is_label:
            source.lut = label_rgba_lut(layer, unique_layer_labels(source.data))
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
        # layer list changes at all, and the expensive parts — the host copy, the
        # slice cache, any resampling — depend on the layer and the reference
        # grid, neither of which a reorder or an unrelated insert touches.
        reference = self._reference_key()
        keep = self._source_cache if self._source_ref == reference else {}
        cache: dict[int, RenderSource] = {}
        sources: list[RenderSource] = []
        for layer in self._candidate_layers():
            if not bool(getattr(layer, "visible", True)):
                continue
            source = keep.get(id(layer))
            if source is None or source.layer is not layer:
                source = self._source_for(layer)
            if source is not None:
                self._refresh_style(source)
                cache[id(layer)] = source
                sources.append(source)
        self._source_cache = cache
        self._source_ref = reference
        self._sources = sources
        self._rendered.clear()
        self._watch_layers()
        self._update_composite_label()
        self._redraw(force=True)
        self._apply_clip()

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
        extra = sum(1 for s in self._sources if s.resampled)
        text = f"{len(names)} layer(s): " + " + ".join(names)
        if extra:
            text += f"  ·  {extra} resampled onto “{getattr(self._layer, 'name', '?')}”"
        self._composite_label.setText(text)

    # ── event subscriptions ──────────────────────────────────────────────────

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
            except Exception:  # noqa: BLE001 — a removed layer's emitter may be gone
                pass
        self._subs = []

    def _watch_layers(self) -> None:
        """Follow the display state of every layer that could be drawn.

        ``visible`` is watched on every *candidate*, not just the drawn ones — a
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
            for name in ("opacity", "blending", "colormap", "gamma", "contrast_limits"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    self._connect(emitter, self._on_style_changed)
            for name in ("data", "set_data"):
                emitter = getattr(events, name, None)
                if emitter is not None:
                    self._connect(emitter, self._on_layer_data_changed)

    def _on_visible_changed(self, _event: Any = None) -> None:
        """A layer was shown or hidden: the draw list changed."""
        self.refresh_sources()

    def _on_style_changed(self, _event: Any = None) -> None:
        """A colour or opacity moved: refresh the tables, keep the data."""
        for source in self._sources:
            self._refresh_style(source)
        self._style_timer.start()

    def _on_layer_data_changed(self, _event: Any = None) -> None:
        """A layer's voxels changed: its caches are stale."""
        self._resample_cache.clear()
        self._rebuild_sources()

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

    def _render_view(self, view: AxisView, view_index: int = 0) -> np.ndarray:
        """The RGB image for *view*: every visible layer, in layer-list order."""
        index = self._position[view.axis]
        oblique = self.is_oblique()
        planes: list[np.ndarray] = []
        blendings: list[str] = []
        opacities: list[float] = []
        for source in self._sources:
            if source.opacity <= 0.0:
                continue
            if oblique:
                plane = self._resample(view, view_index, source.data, source.order)
                rgba = (
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
            else:
                rgba = slice_to_rgba(
                    source.layer, source.data, view.axis, index, view,
                    contrast=source.contrast, table=source.table, lut=source.lut,
                    cache=source.cache,
                )
            planes.append(rgba)
            blendings.append(source.blending)
            opacities.append(source.opacity)
        if not planes:
            shape = (
                int(self._data.shape[view.rows]),
                int(self._data.shape[view.cols]),
            )
            return np.zeros((*shape, 3), dtype=np.uint8)
        return composite(planes, blendings, opacities)

    def _redraw(self, *, force: bool = False) -> None:
        """Refresh the views, re-rendering only those whose slice actually moved.

        A crosshair move changes at most two of the three slices; redrawing all of
        them on every scroll step is wasted work on a large volume.
        """
        if self._data is None or self._layer is None:
            return
        oblique = self.is_oblique()
        self._btn_reset_orient.setEnabled(oblique)
        for position, (widget, view) in enumerate(zip(self._slice_views, self._views)):
            widget._view = view
            widget._title.setText(view.title + ("  ·  oblique" if oblique else ""))
            index = self._position[view.axis]
            crosshair = view.to_pixel(self._position, self._data.shape)
            widget.set_lines(self._crosshair_lines(position))
            # A turned plane moves with the crosshair in every direction, so the
            # "same index, skip the redraw" shortcut no longer holds.
            if force or oblique or self._rendered.get(view.axis) != index:
                self._rendered[view.axis] = index
                aspect = self._spacing[view.rows] / max(self._spacing[view.cols], 1e-6)
                widget.set_slice(
                    self._render_view(view, position),
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

    def _crosshair_lines(self, view_index: int) -> list[tuple[float, float, str]]:
        """Where the other two planes cut across this view, with their colours.

        Each line is coloured like the view it belongs to — the same red / green /
        blue the 3D outlines use — so dragging one says which plane is about to
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
        the line would run away from the cursor that is steering it — which is
        also what makes this read as reorienting the volume rather than spinning
        the picture.
        """
        if self._data is None or view_index >= len(self._frames):
            return
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
        # anisotropic spacing — so it is converted back through the two step sizes
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

    def _reset_orientation(self) -> None:
        """Put every plane back on the volume's own axes."""
        if not self._base_frames:
            return
        self._frames = list(self._base_frames)
        self._redraw(force=True)
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

    def _wants_slice_image(self) -> bool:
        """Whether the slice image is drawn alongside the always-on outline."""
        return bool(self._show_slice_image.isChecked())

    def _on_show_planes(self, enabled: bool) -> None:
        """Add or remove the 3D cut layers on the canvas."""
        if self._layer is None:
            return
        name = self._cut_source_name()
        if not enabled:
            remove_ortho_planes(self._viewer, name)
            remove_ortho_boxes(self._viewer, name)
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
            remove_ortho_planes(self._viewer, self._cut_source_name())
        try:
            self._draw_cuts()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not redraw the 3D cuts: {exc}")

    def _cut_source_name(self) -> str:
        """Name the 3D cut layers are keyed on — the bound layer's."""
        return str(getattr(self._layer, "name", "volume"))

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
            if self._wants_slice_image():
                sync_ortho_planes(
                    self._viewer, self._layer, tuple(self._position),
                    frames=frames, spacing=self._spacing,
                )
        finally:
            self._suspend_rebuild = False

    def _sync_canvas(self) -> None:
        """Push the crosshair to the 3D planes and the clip, once per settle."""
        self._sync_planes()
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
        targets = [s.layer for s in self._sources] or [self._layer]
        # A layer that dropped out of the composite keeps whatever cut it was
        # given otherwise, and nothing here would ever take it off again.
        for stale in self._clipped:
            if stale not in targets:
                clear_clip(stale)
        self._clipped = list(targets)
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
    )
    panel._nvitk_dock = dock

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
        # are layers — so rebinding here threw away the crosshair and the plane
        # rotation every time those were drawn.
        if active is None and panel.bound_layer() is not None:
            return
        panel.refresh_from_layer(active)

    def _layers_changed(_event: Any = None) -> None:
        """Keep the overlay picker in step with the viewer's layer list."""
        panel.refresh_overlay_choices()

    viewer.layers.selection.events.active.connect(_refresh)
    for name in ("inserted", "removed", "reordered", "moved"):
        emitter = getattr(viewer.layers.events, name, None)
        if emitter is not None:
            emitter.connect(_layers_changed)
    # A rename re-keys the 3D cut layers, which are named after their source.
    renamed = getattr(viewer.layers.events, "renamed", None)
    if renamed is not None:
        renamed.connect(_layers_changed)
    _refresh()
    return dock


__all__ = [
    "colormap_table",
    "composite",
    "image_rgba",
    "label_rgba_lut",
    "layer_colormap",
    "layer_gamma",
    "slice_to_rgba",
    "AxisView",
    "OrthoViewerPanel",
    "SliceView",
    "apply_clip",
    "attach_ortho_dock",
    "clear_clip",
    "blend_overlay",
    "oblique_box_corners",
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
