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

import functools
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy
from nvitk.core.parallel import parallel_map
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
    invalidate_label_ids,
    layer_label_ids,
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
#: Per *standard* notch (120 eighths of a degree). Applied in proportion to the
#: wheel's actual delta: high-resolution wheels and touchpads send many small
#: events per notch, and treating each as a full notch is what made a 25 % step
#: race through the whole zoom range in one flick.
_ZOOM_STEP = 1.1
_WHEEL_NOTCH = 120.0

#: Default cine rate of the time/energy play button.
_PLAY_FPS = 8

#: How close to a crosshair line a press must be to grab it, and how far from the
#: centre it must be for that grab to mean "rotate" rather than "move".
_HANDLE_TOL_PX = 10.0
_HANDLE_MIN_FRACTION = 0.55

#: Largest volume for which a per-axis contiguous copy is worth its memory.
#:
#: Measured, rather than guessed. On a 419 MB volume a strided read of the last
#: axis takes 0.94 ms against 0.04 ms from a reordered copy — a real speedup, but
#: one that costs 458 ms to build and only repays after ~500 slices on that axis,
#: while 0.94 ms was already far inside a frame. The old 512 MB ceiling let every
#: layer reorder itself: five layers of a 241 MB study built 2.4 GB of copies on
#: panel open. Small volumes still take the fast path, where it is nearly free.
_SLICE_CACHE_BUDGET = 64 * 1024 * 1024

#: Total reordered bytes allowed across every layer the panel draws. The
#: per-array limit above says nothing about how many arrays there are.
_SLICE_CACHE_TOTAL = 256 * 1024 * 1024


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
    # Slab-parallel on the shared worker pool, bit-identical to SciPy's.
    from nvitk.transform.threaded import map_coordinates

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


def _resample_budget() -> int:
    """Ceiling on cached resampled volumes: 15 % of RAM, never under 512 MB.

    One off-grid layer costs a full copy on the active grid. The old fixed
    512 MB refused to draw any off-grid layer at all next to a 512x512x416
    float64 CT (870 MB on its own) — on a workstation with tens of GB free.
    """
    from nvitk.core.parallel import physical_memory_bytes

    return max(512 * 1024 * 1024, int(0.15 * physical_memory_bytes()))


_RESAMPLE_BUDGET_BYTES = _resample_budget()


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
    #: Time point ``data`` was taken at, for a 3D+t layer (``None`` for a 3D one).
    time: int | None = None
    #: Drawn smoothed when magnified — the layer's napari 2D interpolation.
    smooth: bool = False
    #: Distinct label ids in ``data``, computed once. Finding them is a full
    #: ``np.unique`` over the volume (~24 ms on a 50 MB mask) and they cannot
    #: change without ``data`` changing, which rebuilds the source anyway — so
    #: recomputing it on every style event was the single largest cost of
    #: clicking a layer in the list.
    label_ids: list[int] | None = None
    #: ``(max_id + 1, 4)`` uint8 lookup built once per style refresh, so a slice
    #: render is a single fancy-index instead of a per-slice unique + rebuild.
    label_table: np.ndarray | None = None


#: The only layer types the panel draws. Shapes, Points, Vectors, Surfaces and
#: Tracks are annotations, not imagery: a tool's centerline paths, station markers
#: and cut outlines would otherwise be composited over the very anatomy they are
#: annotating, and hide it.
DRAWN_LAYER_TYPES: tuple[str, ...] = ("Image", "Labels")


def _grid_key(layer: Any) -> tuple:
    """The sampling grid a layer sits on: shape plus world placement.

    Two layers with the same key produce the same slices at the same crosshair,
    so swapping which one is *bound* cannot change a single pixel.
    """
    from nvitk.gui.core.spatial import layer_affine

    shape = _spatial_shape(layer)
    affine = layer_affine(layer)
    raw_scale = getattr(layer, "scale", None)
    # Explicitly against None: a layer's scale is a NumPy array, and ``or`` on
    # one raises rather than falling back.
    scale = () if raw_scale is None else tuple(float(s) for s in raw_scale)
    return (
        shape,
        None if affine is None else affine.tobytes(),
        scale,
    )


def layer_time_axis(layer: Any) -> int | None:
    """Array axis holding time on a 3D+t layer (``0`` when shown time-first), else ``None``."""
    shape = _layer_shape(layer)
    if shape is None or len(shape) != 4:
        return None
    from nvitk.gui.core.spatial import _time_axis_index

    return int(_time_axis_index(layer))


def layer_time_count(layer: Any) -> int:
    """Number of time points of *layer* (1 for a plain volume)."""
    axis = layer_time_axis(layer)
    shape = _layer_shape(layer)
    return int(shape[axis]) if axis is not None and shape is not None else 1


def _spatial_shape(layer: Any) -> tuple[int, int, int] | None:
    """The three spatial dimensions of *layer*'s array, time axis removed."""
    shape = _layer_shape(layer)
    if shape is None or len(shape) < 3:
        return None
    axis = layer_time_axis(layer)
    if axis is not None:
        return tuple(int(v) for i, v in enumerate(shape) if i != axis)  # type: ignore[return-value]
    return tuple(int(v) for v in shape[-3:])  # type: ignore[return-value]


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
    if bool(getattr(layer, "rgb", False)):
        # A colour capture (dose sheet, tracker graph): its last axis is RGB, not depth.
        return False
    name = str(getattr(layer, "name", ""))
    if _PLANE_LAYER_SUFFIX in name or _BOX_LAYER_SUFFIX in name:
        return False
    shape = _layer_shape(layer)
    return shape is not None and len(shape) >= 3


def is_volume_layer(layer: Any) -> bool:
    """A layer the panel can bind to as its grid: three spatial axes, none of them
    a single voxel. A positioned 2D slice is a one-slice 3D layer, and a 2D+t
    series a one-slice 3D+t one — drawable over a volume, but no grid of their own.
    """
    if not is_drawable_layer(layer):
        return False
    shape = _layer_shape(layer) or ()
    return len(shape) >= 3 and all(int(n) > 1 for n in shape[-3:])


def _layer_volume(layer: Any, time_index: int = 0) -> np.ndarray | None:
    """*layer*'s 3D host array, or ``None`` when it has none to draw.

    One place for the label-source choice, the multiscale unwrap and the 4D
    reduction, all of which were previously repeated at each call site. A 3D+t
    layer contributes the volume at *time_index* (clamped), as a view.
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
    if arr.ndim == 4:
        axis = layer_time_axis(layer)
        axis = arr.ndim - 1 if axis is None else axis
        t = int(np.clip(int(time_index), 0, int(arr.shape[axis]) - 1))
        arr = arr[_axis_index(arr.ndim, axis, t)]  # a view (np.take would copy all frames)
    elif arr.ndim > 4:
        arr = arr[(0,) * (arr.ndim - 3)]
    return arr if arr.ndim == 3 else None


def _same_grid(layer: Any, reference: Any) -> bool:
    """Whether *layer* already sits on *reference*'s voxel grid.

    Read off the array object and the affine without materialising either: this
    runs for every layer on every rebind, and the expensive resampler behind it
    re-checks properly anyway.
    """
    from nvitk.gui.core.spatial import layer_affine

    if _spatial_shape(layer) != _spatial_shape(reference):
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
    from nvitk.gui.core.spatial import layer_affine

    # Both grids known: compose the voxel-to-voxel map and resample slab-parallel
    # on the worker pool (~7x on 8 threads) — the same composition resample_to
    # uses, without building Image objects for a display copy.
    src_aff, ref_aff = layer_affine(layer), layer_affine(reference)
    ref_shape = _spatial_shape(reference)
    if src_aff is not None and ref_aff is not None and ref_shape is not None:
        try:
            from nvitk.transform.threaded import affine_transform

            vox = np.linalg.inv(ref_aff) @ src_aff          # source voxel → reference voxel
            inv = np.linalg.inv(vox)                        # reference voxel → source voxel
            host = to_numpy(data)
            # A display copy: intensities in float32 (half of a float64 CT's
            # footprint, far below what a screen can show); labels keep their ids.
            out_dtype = host.dtype if int(order) == 0 else np.float32
            return affine_transform(
                host, inv[:3, :3], offset=inv[:3, 3], output_shape=ref_shape,
                order=int(order), mode="constant", cval=0.0, prefilter=int(order) > 1,
                output=out_dtype,
            )
        except Exception:  # noqa: BLE001 — fall through to the generic aligner
            pass

    from nvitk.core.backend import using
    from nvitk.gui.core.spatial import align_mask_to_reference_layer

    try:
        with using("cpu"):
            _ref, aligned, _resampled = align_mask_to_reference_layer(
                layer, reference, data, order=int(order)
            )
        return to_numpy(aligned.data)
    except Exception:  # noqa: BLE001 — reported by the caller, never raised into paint
        return None


def _axis_index(ndim: int, axis: int, index: int) -> tuple:
    """Basic-indexing key selecting *index* along *axis* — a view, never a copy."""
    key: list[Any] = [slice(None)] * int(ndim)
    key[int(axis)] = int(index)
    return tuple(key)


def _slice_of(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    """The 2D slice of *data* at *index* along *axis*, clamped into range.

    Basic indexing, deliberately not ``np.take``: ``take`` first makes the whole
    input C-contiguous, and a NIfTI volume comes off disk in Fortran order — so
    every slice of a 512x512x416 CT copied the full 870 MB volume (~1.4 s per
    slice, per layer, per view). An indexed view reads only the slice.
    """
    n = int(data.shape[axis])
    idx = int(np.clip(index, 0, max(n - 1, 0)))
    return data[_axis_index(data.ndim, axis, idx)]


#: Guards every panel's shared reordered-slice budget across render threads.
_POOL_LOCK = threading.Lock()


class SliceCache:
    """Serves 2D slices of a volume, keeping the slow axes contiguous.

    A C-ordered volume gives axis 0 for free, but a slice along the last axis is a
    strided gather over the whole array — around 40x slower on a 400x512x512 CT,
    which is exactly why scrolling axial felt heavier than the others. Reordering
    that axis into its own contiguous copy makes every axis equally cheap.

    The copy is built lazily, on the first scroll of that axis, and only for the
    *innermost* axis (the smallest stride — last for C order, first for the
    Fortran order NIfTI arrays arrive in): that is the one whose slice is a
    gather with a stride between every element. The other axes yield
    contiguous runs and measure no faster reordered — so reordering them was
    pure memory.

    Panels share a *pool* so that the ceiling is on the session rather than on
    each array: the per-array limit alone said nothing about how many arrays
    there would be, and one study with five layers open built gigabytes.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        budget_bytes: int = _SLICE_CACHE_BUDGET,
        pool: dict[str, int] | None = None,
    ) -> None:
        """Wrap *data*, reordering nothing until an axis is actually asked for."""
        self._data = data
        self._budget = int(budget_bytes)
        self._pool = pool
        self._reordered: dict[int, np.ndarray | None] = {}
        # Views render on worker threads: two of them asking for the same axis
        # must not both build (and both charge the pool for) the same copy.
        self._lock = threading.Lock()

    def _worth_reordering(self, axis: int, arr: np.ndarray) -> bool:
        """Whether a contiguous copy of *axis* would earn its memory."""
        # Only the innermost (smallest-stride) axis is genuinely slow to slice:
        # its slice gathers one element per stride. That is the *last* axis of
        # a C-ordered array but the *first* of a Fortran-ordered one, which is
        # how nibabel hands NIfTI volumes over — so ask the strides, not the
        # axis number.
        strides = [abs(int(st)) for st in arr.strides]
        if not strides or axis != int(np.argmin(strides)):
            return False
        if arr.nbytes > self._budget:
            return False
        if self._pool is not None:
            if self._pool["used"] + arr.nbytes > self._pool["cap"]:
                return False
        return True

    def _fast_axis(self, axis: int) -> np.ndarray | None:
        """A contiguous copy with *axis* first, or ``None`` if it is not worth making."""
        if axis in self._reordered:
            return self._reordered[axis]
        with self._lock:
            if axis in self._reordered:
                return self._reordered[axis]
            arr = self._data
            with _POOL_LOCK:
                worth = self._worth_reordering(axis, arr)
                if worth and self._pool is not None:
                    self._pool["used"] += int(arr.nbytes)  # reserve before building
            if not worth:
                self._reordered[axis] = None
                return None
            try:
                self._reordered[axis] = np.ascontiguousarray(np.moveaxis(arr, axis, 0))
            except (MemoryError, ValueError):
                self._reordered[axis] = None
                if self._pool is not None:
                    with _POOL_LOCK:
                        self._pool["used"] = max(self._pool["used"] - int(arr.nbytes), 0)
            return self._reordered[axis]

    def release(self) -> None:
        """Give the reordered copies back to the pool."""
        with self._lock:
            for arr in self._reordered.values():
                if arr is not None and self._pool is not None:
                    with _POOL_LOCK:
                        self._pool["used"] = max(self._pool["used"] - int(arr.nbytes), 0)
            self._reordered.clear()

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
    # Windowed in place. Each of ``nan_to_num``, the subtraction, the division
    # and the clip used to allocate its own copy of the slice; on a 512x512
    # float64 CT that is five megabytes of churn per view per redraw.
    scale = float(table.shape[0] - 1) / (hi - lo)
    norm = np.subtract(arr, lo, dtype=np.float32)
    np.nan_to_num(norm, copy=False, nan=0.0, posinf=hi - lo, neginf=0.0)
    np.multiply(norm, scale, out=norm)
    np.clip(norm, 0.0, float(table.shape[0] - 1), out=norm)
    return table[norm.astype(np.uint16, copy=False)]


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
    plane: np.ndarray,
    layer: Any,
    lut: dict[int, np.ndarray] | None = None,
    table: np.ndarray | None = None,
) -> np.ndarray:
    """Colour a 2D label slice to ``(H, W, 4)`` uint8; background transparent.

    A Labels layer drawn as outlines (its ``contour``) is outlined here too —
    Napari's own contour, so the views match the canvas, and whatever the canvas
    is doing: in a 3D canvas Napari cannot draw contours, the views still can.
    """
    raw = np.asarray(plane)
    # An integer label slice is already what the lookup wants. Rounding it
    # through float64 and back cost two full-size temporaries per view per
    # redraw for a value that could not have changed.
    if np.issubdtype(raw.dtype, np.integer):
        arr = raw
    else:
        arr = np.rint(np.asarray(raw, dtype=np.float64)).astype(np.int64, copy=False)
    contour = int(getattr(layer, "contour", 0) or 0)
    if contour > 0 and arr.ndim == 2 and arr.size:
        from napari.layers.labels._labels_utils import get_contours

        arr = get_contours(arr, contour, 0)
    if table is None:
        present = [int(v) for v in np.unique(arr) if int(v) != 0]
        if not present:
            return np.zeros((*arr.shape, 4), dtype=np.uint8)
        table = label_color_table(present, lut if lut is not None else label_rgba_lut(layer, present))
    if table.shape[0] <= 1:
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    idx = np.clip(arr, 0, table.shape[0] - 1)
    return table[idx]


def label_color_table(label_ids: Sequence[int], lut: dict[int, np.ndarray]) -> np.ndarray:
    """A ``(max_id + 1, 4)`` uint8 lookup table for *label_ids*.

    Built once per layer rather than per slice: it depends only on the ids and
    their colours, neither of which changes while scrolling.
    """
    ids = [int(v) for v in label_ids if int(v) != 0]
    if not ids:
        return np.zeros((1, 4), dtype=np.uint8)
    table = np.zeros((max(ids) + 1, 4), dtype=np.uint8)
    white = np.array([255, 255, 255, 255], dtype=np.uint8)
    for lid in ids:
        table[lid] = lut.get(lid, white)
    return table


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
    label_table: np.ndarray | None = None,
) -> np.ndarray:
    """One orthogonal slice as ``(H, W, 4)`` uint8, ready to composite."""
    raw = cache.slice(axis, index) if cache is not None else _slice_of(data, axis, index)
    plane = _oriented(raw, axis, view)
    if is_label_like_layer(layer):
        return _label_rgba(plane, layer, lut, label_table)
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
    # Written in place throughout. Every arithmetic step here used to allocate a
    # fresh (H, W, 3) float array, so a two-layer composite of a 512x512 slice
    # churned tens of megabytes per view per redraw for a result of 768 KB.
    scratch = np.empty((*shape, 3), dtype=np.float32)
    alpha = np.empty((*shape, 1), dtype=np.float32)
    for plane, blending, opacity in zip(planes, blendings, opacities):
        src = np.asarray(plane)
        if src.shape[:2] != shape:
            continue
        rgb = src[..., :3]
        np.multiply(src[..., 3:4], float(np.clip(opacity, 0.0, 1.0)) / 255.0, out=alpha,
                    dtype=np.float32, casting="unsafe")
        mode = str(blending or "translucent")
        if mode == "opaque":
            np.copyto(out, rgb, where=alpha > 0, casting="unsafe")
        elif mode == "additive":
            np.multiply(rgb, alpha, out=scratch, dtype=np.float32, casting="unsafe")
            np.add(out, scratch, out=out)
        elif mode == "minimum":
            np.copyto(scratch, out)
            np.copyto(scratch, rgb, where=alpha > 0, casting="unsafe")
            np.minimum(out, scratch, out=out)
        elif mode == "multiplicative":
            np.multiply(rgb, 1.0 / 255.0, out=scratch, dtype=np.float32, casting="unsafe")
            np.multiply(out, scratch, out=out)
        else:  # translucent, translucent_no_depth
            np.multiply(rgb, alpha, out=scratch, dtype=np.float32, casting="unsafe")
            np.subtract(1.0, alpha, out=alpha)
            np.multiply(out, alpha, out=out)
            np.add(out, scratch, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8, copy=False)


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


# ── interpolation ─────────────────────────────────────────────────────────────


def layer_interpolation_smooth(layer: Any) -> bool:
    """Whether *layer* is drawn smoothed when magnified (its napari 2D interpolation).

    Labels never are — napari draws them nearest-neighbour, whatever the image
    layers under them do — and an Image layer follows ``interpolation2d``
    (napari's own default is ``nearest``).
    """
    if type(layer).__name__ == "Labels" or is_label_like_layer(layer):
        return False
    return str(getattr(layer, "interpolation2d", "nearest") or "nearest").lower() != "nearest"


def layer_interpolation_order(layer: Any) -> int:
    """Spline order matching *layer*'s interpolation: 0 nearest, 1 linear, 3 the cubic family."""
    if type(layer).__name__ == "Labels" or is_label_like_layer(layer):
        return 0
    mode = str(getattr(layer, "interpolation2d", "nearest") or "nearest").lower()
    if mode == "nearest":
        return 0
    if mode in ("linear", "bilinear"):
        return 1
    return 3


@dataclass(frozen=True)
class PaintRun:
    """A run of consecutive layers drawn with one interpolation, ready to paint.

    *mode* says how it lands on what is under it: ``"base"`` is opaque RGB (the
    bottom run), ``"over"`` premultiplied RGBA painted source-over, ``"plus"`` RGB
    added on top (additive layers).
    """

    image: np.ndarray
    smooth: bool
    mode: str = "base"


def composite_over_rgba(
    planes: Sequence[np.ndarray],
    blendings: Sequence[str],
    opacities: Sequence[float],
) -> np.ndarray:
    """Translucent/opaque *planes* over transparency → premultiplied RGBA uint8."""
    shape = np.asarray(planes[0]).shape[:2]
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    acc = np.zeros((*shape, 1), dtype=np.float32)
    for plane, blending, opacity in zip(planes, blendings, opacities):
        src = np.asarray(plane)
        if src.shape[:2] != shape:
            continue
        a = src[..., 3:4].astype(np.float32) * (float(np.clip(opacity, 0.0, 1.0)) / 255.0)
        colour = src[..., :3].astype(np.float32)
        if str(blending) == "opaque":
            hit = a > 0
            np.copyto(rgb, colour, where=hit)
            np.copyto(acc, 1.0, where=hit)
            continue
        rgb = colour * a + rgb * (1.0 - a)
        acc = a + acc * (1.0 - a)
    out = np.empty((*shape, 4), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3:] = np.clip(acc * 255.0, 0, 255).astype(np.uint8)
    return out


def composite_runs(
    planes: Sequence[np.ndarray],
    blendings: Sequence[str],
    opacities: Sequence[float],
    smooth: Sequence[bool],
) -> list[PaintRun]:
    """Split the layer stack into runs that share an interpolation, each composited.

    The bottom run is composited exactly as :func:`composite` does; each later run
    becomes premultiplied RGBA (translucent/opaque layers) or an additive RGB, so
    Qt can scale every run with its own filter and stack them in order. A later
    run mixing other blend modes (minimum, multiplicative) cannot be expressed
    that way, so the whole stack then falls back to one run, smoothed if any layer
    in it is.
    """
    if not len(planes):
        return [PaintRun(np.zeros((1, 1, 3), dtype=np.uint8), False)]
    groups: list[list[int]] = []
    for i, flag in enumerate(smooth):
        if groups and bool(smooth[groups[-1][-1]]) == bool(flag):
            groups[-1].append(i)
        else:
            groups.append([i])

    def _pick(indices: list[int]) -> tuple[list, list, list]:
        return ([planes[i] for i in indices], [blendings[i] for i in indices],
                [opacities[i] for i in indices])

    runs = [PaintRun(composite(*_pick(groups[0])), bool(smooth[groups[0][0]]), "base")]
    for group in groups[1:]:
        p, b, o = _pick(group)
        modes = {str(m or "translucent") for m in b}
        if modes <= {"translucent", "translucent_no_depth", "opaque"}:
            runs.append(PaintRun(composite_over_rgba(p, b, o), bool(smooth[group[0]]), "over"))
        elif modes == {"additive"}:
            runs.append(PaintRun(composite(p, b, o), bool(smooth[group[0]]), "plus"))
        else:
            return [PaintRun(composite(planes, blendings, opacities), any(smooth), "base")]
    return runs


def flatten_runs(runs: Sequence[PaintRun]) -> np.ndarray:
    """The runs merged at native resolution: what the view shows, before scaling."""
    out = np.asarray(runs[0].image, dtype=np.float32)[..., :3].copy()
    for run in runs[1:]:
        img = np.asarray(run.image, dtype=np.float32)
        if run.mode == "over":
            a = img[..., 3:4] / 255.0
            out = img[..., :3] + out * (1.0 - a)
        elif run.mode == "plus":
            out = out + img[..., :3]
    return np.clip(out, 0, 255).astype(np.uint8)


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
        #: The slice as interpolation runs (see :class:`PaintRun`); ``_rgb`` is
        #: their flattened native-resolution composite.
        self._runs: list[PaintRun] = []
        #: Wheel travel not yet turned into a slice step (eighths of a degree).
        self._wheel_accum = 0.0
        self._aspect = 1.0
        self._cross: tuple[int, int] | None = None
        self._count = 1
        #: Magnification over the fit-to-canvas size, and the normalised point of
        #: the slice held at the canvas centre.
        self._zoom = 1.0
        self._pan = [0.5, 0.5]
        self._drag_from: tuple[float, float] | None = None
        #: Where a pan gesture was pressed: one released without moving is a
        #: Shift+click, which the label editor may want (the wand removes with it).
        self._pan_origin: tuple[float, float] | None = None
        #: ``(d_row, d_col, colour)`` per crosshair line. Empty means the plain
        #: horizontal/vertical cross, which is what an unrotated view shows.
        self._lines: list[tuple[float, float, str]] = []
        #: Index of the line whose end is being dragged, if any.
        self._rotating: int | None = None
        #: Label editing (:class:`~nvitk.gui.labels.ortho_edit.OrthoLabelEditor`):
        #: ``handler(kind, pixel, event) -> bool`` sees presses, drags and releases
        #: first; True means it drew, and the crosshair stays put.
        self.edit_handler: Any | None = None
        self._editing = False
        #: Brush outline ``(radius down, radius across)`` in slice pixels, drawn at
        #: the cursor while a brush tool is on; and the cursor's canvas position.
        self._brush: tuple[float, float] | None = None
        self._hover: tuple[float, float] | None = None
        #: A polygon being drawn in this view: its corners as (row, column) slice pixels.
        self._polyline: list[tuple[float, float]] = []
        #: The Labeling box in this plane: (row0, col0, row1, col1) slice-pixel edges,
        #: and whether the slice is inside the box's depth (drawn faint when not).
        self._box: tuple[float, float, float, float] | None = None
        self._box_inside = True
        #: A traced vessel's path and clicked points, as (row, column) slice pixels.
        self._trace: list[tuple[float, float]] = []
        self._trace_marks: list[tuple[float, float]] = []
        #: The composited slice scaled into the canvas, kept between repaints that
        #: only move overlays (the brush outline follows every mouse move).
        self._base: QPixmap | None = None
        self._base_key: tuple | None = None
        #: Bumped by every new slice image: the cache key (an ``id`` could be reused).
        self._serial = 0

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
        """Show *rgb* as this view's slice, at *index* of *count*.

        *rgb* is an RGB array, or a list of :class:`PaintRun` — the layer stack split
        by interpolation, each run scaled with its own filter when painted.
        """
        if isinstance(rgb, (list, tuple)):
            self._runs = list(rgb)
            self._rgb = (
                np.asarray(self._runs[0].image)[..., :3] if len(self._runs) == 1
                else flatten_runs(self._runs)
            ) if self._runs else None
        else:
            self._rgb = rgb
            self._runs = [PaintRun(np.asarray(rgb), True, "base")] if rgb is not None else []
        self._serial += 1
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

    def set_brush(self, radii: tuple[float, float] | None) -> None:
        """Draw the brush outline (radius down, radius across, in slice pixels) at the
        cursor, or stop drawing it (``None``)."""
        radii = None if radii is None else (float(radii[0]), float(radii[1]))
        if radii == self._brush:
            return
        self._brush = radii
        self._canvas.setMouseTracking(radii is not None)
        if radii is None:
            self._hover = None
        self._repaint()

    def set_polyline(self, points: list[tuple[float, float]] | None) -> None:
        """Draw a polygon's corners so far (slice pixels), or nothing."""
        self._polyline = list(points or [])
        self._repaint()

    def set_box(self, rect: tuple[float, float, float, float] | None, *, inside: bool = True) -> None:
        """Draw the Labeling box's rectangle (slice-pixel edges ``(r0, c0, r1, c1)``),
        faint when this slice is outside its depth; ``None``: no box."""
        rect = None if rect is None else tuple(float(v) for v in rect)
        if rect == self._box and bool(inside) == self._box_inside:
            return
        self._box, self._box_inside = rect, bool(inside)
        self._repaint()

    def set_trace(self, path: list[tuple[float, float]] | None, marks: list[tuple[float, float]] | None = None) -> None:
        """Draw a traced vessel: its path and its clicked points (slice pixels)."""
        self._trace = list(path or [])
        self._trace_marks = list(marks or [])
        self._repaint()

    def set_draw_cursor(self, drawing: bool) -> None:
        """A cross cursor while a drawing tool is on (a click edits, not moves)."""
        if drawing:
            self._canvas.setCursor(Qt.CrossCursor)
        else:
            self._canvas.unsetCursor()

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
        if self._zoom <= 1.0:
            # Fitted: centred.
            return (scaled_w, scaled_h,
                    (target.width() - scaled_w) / 2.0, (target.height() - scaled_h) / 2.0)
        # Zoomed: the panned-to point sits at the canvas centre — nothing else. It
        # is always a point of the slice (pan is 0…1), so the slice can never leave
        # the view; centring or pinning it to the edges instead is what made a zoom
        # drift away from the cursor until the slice outgrew the canvas.
        off_x = target.width() / 2.0 - self._pan[0] * scaled_w
        off_y = target.height() / 2.0 - self._pan[1] * scaled_h
        return scaled_w, scaled_h, off_x, off_y

    def set_zoom(
        self,
        zoom: float,
        *,
        about: tuple[float, float] | None = None,
        anchor: tuple[float, float] | None = None,
    ) -> None:
        """Set the magnification.

        With *anchor* (a canvas position, the cursor's), the slice point under it
        stays under it — the zoom every image viewer does, so the wheel closes in
        on what the mouse is over. *about* (a normalised slice point) is instead
        put at the canvas centre.
        """
        new_zoom = float(np.clip(zoom, _MIN_ZOOM, _MAX_ZOOM))
        geometry = self._geometry() if anchor is not None else None
        self._zoom = new_zoom
        if new_zoom <= 1.0:
            self._pan = [0.5, 0.5]
        elif geometry is not None:
            scaled_w, scaled_h, off_x, off_y = geometry
            x, y = anchor
            # The slice point under the cursor, as a fraction of the drawn slice…
            u = (x - off_x) / max(scaled_w, 1)
            v = (y - off_y) / max(scaled_h, 1)
            # …and the pan that leaves it there at the new size (_geometry places
            # the panned-to point at the centre, then keeps the slice on the canvas).
            new_geometry = self._geometry()
            if new_geometry is not None:
                new_w, new_h = new_geometry[0], new_geometry[1]
                centre_x = self._canvas.width() / 2.0
                centre_y = self._canvas.height() / 2.0
                self._pan = [
                    float(np.clip(u - (x - centre_x) / max(new_w, 1), 0.0, 1.0)),
                    float(np.clip(v - (y - centre_y) / max(new_h, 1), 0.0, 1.0)),
                ]
        elif about is not None:
            self._pan = [float(np.clip(c, 0.0, 1.0)) for c in about]
        self._repaint()

    def zoom(self) -> float:
        """Current magnification."""
        return float(self._zoom)

    def _repaint(self) -> None:
        """Scale the current slice into the canvas and draw the crosshairs on it."""
        from qtpy.QtCore import QPointF, QRectF
        from qtpy.QtGui import QColor, QPainter, QPen

        geometry = self._geometry()
        if geometry is None:
            self._canvas.setPixmap(QPixmap())
            self._canvas.setText("No slice")
            return
        scaled_w, scaled_h, off_x, off_y = geometry

        # Painted into a canvas-sized pixmap rather than handed over directly, so a
        # zoomed slice is cropped by the view instead of resizing the widget. The
        # scaled slice is kept: a repaint that only moves an overlay (the brush
        # outline, at every mouse move) starts from it instead of rescaling.
        key = (self._serial, self._canvas.width(), self._canvas.height(),
               scaled_w, scaled_h, round(off_x, 2), round(off_y, 2))
        if self._base is None or self._base_key != key:
            base = QPixmap(self._canvas.size())
            base.fill(QColor("#000000"))
            painter = QPainter(base)
            # Each interpolation run is scaled with the filter its layers use in
            # napari — nearest stays blocky, linear/cubic smooth — then stacked.
            for run in self._runs or [PaintRun(np.asarray(self._rgb), True, "base")]:
                data = np.ascontiguousarray(run.image, dtype=np.uint8)
                h, w = data.shape[:2]
                if data.ndim == 3 and data.shape[2] == 4:
                    image = QImage(data.data, w, h, 4 * w, QImage.Format_RGBA8888_Premultiplied).copy()
                else:
                    rgb3 = np.ascontiguousarray(data[..., :3])
                    image = QImage(rgb3.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                filt = Qt.SmoothTransformation if run.smooth else Qt.FastTransformation
                run_map = QPixmap.fromImage(image).scaled(scaled_w, scaled_h, Qt.IgnoreAspectRatio, filt)
                painter.setCompositionMode(
                    QPainter.CompositionMode_Plus if run.mode == "plus"
                    else QPainter.CompositionMode_SourceOver
                )
                painter.drawPixmap(int(round(off_x)), int(round(off_y)), run_map)
            painter.end()
            self._base, self._base_key = base, key
        canvas = QPixmap(self._base)
        painter = QPainter(canvas)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
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
        if self._polyline:
            h, w = self._rgb.shape[:2]
            pts = [QPointF(off_x + (c + 0.5) / max(w, 1) * scaled_w, off_y + (r + 0.5) / max(h, 1) * scaled_h)
                   for r, c in self._polyline]
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#ffd34d"), 1.5))
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(a, b)
            for pt in pts:
                painter.drawRect(int(pt.x()) - 2, int(pt.y()) - 2, 4, 4)
        if self._box is not None:
            h, w = self._rgb.shape[:2]
            r0, c0, r1, c1 = self._box
            x0, x1 = off_x + c0 / max(w, 1) * scaled_w, off_x + c1 / max(w, 1) * scaled_w
            y0, y1 = off_y + r0 / max(h, 1) * scaled_h, off_y + r1 / max(h, 1) * scaled_h
            colour = QColor("#4dd2ff")
            if not self._box_inside:
                colour.setAlpha(110)
            pen = QPen(colour, 1.5)
            pen.setStyle(Qt.PenStyle.SolidLine if self._box_inside else Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(QPointF(min(x0, x1), min(y0, y1)), QPointF(max(x0, x1), max(y0, y1))))
        if self._trace or self._trace_marks:
            h, w = self._rgb.shape[:2]

            def point(rc: tuple[float, float]) -> Any:
                return QPointF(off_x + (rc[1] + 0.5) / max(w, 1) * scaled_w, off_y + (rc[0] + 0.5) / max(h, 1) * scaled_h)

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#ffd34d"), 1.2))
            pts = [point(rc) for rc in self._trace]
            for a, b in zip(pts, pts[1:]):
                painter.drawLine(a, b)
            painter.setBrush(QColor("#ffd34d"))
            for rc in self._trace_marks:
                painter.drawEllipse(point(rc), 3.0, 3.0)
            painter.setBrush(Qt.NoBrush)
        if self._brush is not None and self._hover is not None:
            # The brush's footprint, round in millimetres like the slice is.
            h, w = self._rgb.shape[:2]
            rx = self._brush[1] * scaled_w / max(w, 1)
            ry = self._brush[0] * scaled_h / max(h, 1)
            hx, hy = self._hover
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(hx, hy), max(rx, 1.0), max(ry, 1.0))
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
            delta = float(event.angleDelta().y())
            modifiers = event.modifiers()
            # Ctrl+wheel zooms, plain wheel scrolls slices. Scrolling is the far
            # commoner gesture here, so it keeps the unmodified wheel.
            if modifiers & Qt.ControlModifier:
                # In proportion to the travel: a standard notch is one 10 % step,
                # a touchpad's stream of small deltas adds up to the same thing.
                self.set_zoom(
                    self._zoom * (_ZOOM_STEP ** (delta / _WHEEL_NOTCH)),
                    anchor=self._cursor_xy(event),
                )
                return True
            # One slice per notch, however the travel arrives.
            self._wheel_accum += delta
            steps = int(self._wheel_accum / _WHEEL_NOTCH)
            if steps:
                self._wheel_accum -= steps * _WHEEL_NOTCH
                self._slider.setValue(
                    int(np.clip(self._slider.value() + steps, 0, self._count - 1))
                )
            return True
        if event.type() == QEvent.MouseButtonDblClick:
            # The label editor's first (a polygon closes on a double-click), else a zoom reset.
            if not self._edit(event, "double"):
                self.set_zoom(1.0)
            return True
        if event.type() == QEvent.MouseButtonPress:
            if self._is_pan(event):
                self._drag_from = self._cursor_xy(event)
                self._pan_origin = self._drag_from
                return True
            if self._edit(event, "press"):
                self._editing = True
                return True
            handle = self._handle_at(event)
            if handle is not None:
                self._rotating = handle
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseMove:
            if self._brush is not None:
                self._hover = self._cursor_xy(event)
            if self._drag_from is not None and self._is_pan(event):
                self._pan_by(event)
                return True
            if self._editing:
                self._edit(event, "move")
                if self._brush is not None:
                    self._repaint()
                return True
            if not event.buttons():
                if self._brush is not None:
                    self._repaint()
                return False
            if self._rotating is not None:
                self.handleDragged.emit(int(self._rotating), self._screen_angle(event))
                return True
            self._emit_click(event)
            return True
        if event.type() == QEvent.MouseButtonRelease:
            origin, self._pan_origin = self._pan_origin, None
            self._drag_from = None
            self._rotating = None
            if origin is not None and event.button() == Qt.LeftButton:
                x, y = self._cursor_xy(event)
                if abs(x - origin[0]) + abs(y - origin[1]) < 4.0:
                    # A Shift+click, not a pan: the label editor may take it.
                    self._edit(event, "shift-click")
                return True
            if self._editing:
                self._editing = False
                self._edit(event, "release")
                return True
            return False
        if event.type() == QEvent.Leave and self._hover is not None:
            self._hover = None
            self._repaint()
        return False

    def _edit(self, event: Any, kind: str) -> bool:
        """Offer a mouse event to the label editor; True when it took it."""
        handler = self.edit_handler
        if handler is None:
            return False
        if kind in ("shift-click", "double") and event.button() != Qt.LeftButton:
            return False
        # The right button too: a right-drag erases with the brush, whatever the tool.
        if kind == "press" and event.button() not in (Qt.LeftButton, Qt.RightButton):
            return False
        try:
            return bool(handler(kind, self._pixel_at(event), event))
        except Exception:  # noqa: BLE001 — an editor fault must not wedge the view
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

    def _pixel_at(self, event: Any) -> tuple[int, int] | None:
        """The (row, column) slice pixel under the cursor, ``None`` off the slice."""
        geometry = self._geometry()
        if geometry is None:
            return None
        scaled_w, scaled_h, off_x, off_y = geometry
        h, w = self._rgb.shape[:2]
        x, y = self._cursor_xy(event)
        px, py = x - off_x, y - off_y
        if not (0 <= px < scaled_w and 0 <= py < scaled_h):
            return None
        col = int(np.clip(px / scaled_w * w, 0, w - 1))
        row = int(np.clip(py / scaled_h * h, 0, h - 1))
        return row, col

    def _emit_click(self, event: Any) -> None:
        """Translate a click on the canvas into a crosshair position."""
        pixel = self._pixel_at(event)
        if pixel is not None:
            self.pixelPicked.emit(*pixel)


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
        # With a 3D+t layer open the world has a leading time dim; the spatial
        # dims (the trailing three, where every 3D layer aligns) are what count.
        offset = int(viewer.dims.ndim) - int(ndim)
        order = [a - offset for a in order]
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


#: Display properties a slice plane copies from the layer it shows, so the cut in
#: 3D reads with the same window, colours and brightness as the 2D views.
_IMAGE_STYLE_ATTRS: tuple[str, ...] = (
    "colormap", "contrast_limits", "gamma", "interpolation2d", "interpolation3d",
)
_LABEL_STYLE_ATTRS: tuple[str, ...] = ("colormap",)


def copy_plane_style(source: Any, plane: Any) -> None:
    """Give a slice *plane* the display window and colours of the *source* layer.

    Planes used to be added with Napari's defaults — the full data range and a
    grey ramp — so a CT windowed for soft tissue appeared washed out in 3D, and
    additive blending brightened it further wherever it crossed the volume.
    Only properties that differ are written: each assignment is an event.
    """
    labels = type(source).__name__ == "Labels" or is_label_like_layer(source)
    for attr in (_LABEL_STYLE_ATTRS if labels else _IMAGE_STYLE_ATTRS):
        if not hasattr(source, attr) or not hasattr(plane, attr):
            continue
        try:
            value = getattr(source, attr)
            current = getattr(plane, attr)
            if attr == "contrast_limits":
                if np.allclose(np.asarray(current, float), np.asarray(value, float)):
                    continue
                # Widen the range first: limits outside it are clipped otherwise.
                lo, hi = (float(v) for v in value)
                rng = getattr(plane, "contrast_limits_range", None)
                if rng is not None and (lo < float(rng[0]) or hi > float(rng[1])):
                    plane.contrast_limits_range = (min(lo, float(rng[0])), max(hi, float(rng[1])))
            elif attr == "colormap":
                if getattr(current, "name", current) == getattr(value, "name", value) and not labels:
                    continue
            elif current == value:
                continue
            setattr(plane, attr, value)
        except Exception:  # noqa: BLE001 — a style that will not copy is not worth a failed sync
            continue


def sync_ortho_planes(
    viewer: Any,
    layer: Any,
    position: tuple[int, int, int],
    *,
    axes: tuple[int, ...] = (0, 1, 2),
    frames: Sequence[PlaneFrame] | None = None,
    spacing: Sequence[float] | None = None,
    time_index: int = 0,
    thickness: float = 1.0,
    refresh_data: bool = False,
) -> list[Any]:
    """Show *layer* as three ``depiction="plane"`` slices at *position* on the canvas.

    Napari renders a plane-depicted Image as a single slab through the volume, so
    three of them at the crosshair give the 3D view the same three cuts the 2D
    views show. Existing plane layers are moved rather than recreated — rebuilding
    them on every scroll re-uploads the volume to the GPU each time.

    The planes carry *layer*'s window, colormap and gamma (labels: its colours),
    and a 3D+t layer is shown at *time_index* — its planes are re-fed only when
    the time point changes, or with *refresh_data* (the layer was edited).

    *thickness* is the slab each plane renders: layers drawn over another give
    theirs a little more, so their slice wraps the one below and is drawn in front
    of it instead of fighting it for the same depth.
    """
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    source_name = str(getattr(layer, "name", "volume"))
    by_name = {str(getattr(l, "name", "")): l for l in viewer.layers}
    wanted = [_plane_layer_name(source_name, a) for a in axes]
    is_4d = layer_time_axis(layer) is not None
    t_key = int(time_index) if is_4d else 0
    # Moving existing planes needs only their shape, not the volume: re-deriving
    # the array on every scroll step is what made scrolling crawl.
    if all(name in by_name for name in wanted):
        planes = [by_name[name] for name in wanted]
        stale = refresh_data or any(
            (getattr(pl, "metadata", None) or {}).get("nvitk_ortho_t", 0) != t_key for pl in planes
        )
        data = _layer_volume(layer, t_key) if stale else None
        shape = tuple(int(v) for v in planes[0].data.shape[-3:])
        displayed = displayed_axes(viewer, len(shape))
        out = []
        for i, (axis, existing) in enumerate(zip(axes, planes)):
            point, normal = _plane_geometry(
                axis, position, shape, displayed,
                frames[i] if frames is not None and i < len(frames) else None,
                spacing,
            )
            if data is not None:
                labels = type(layer).__name__ == "Labels" or is_label_like_layer(layer)
                existing.data = np.asarray(data).astype(np.int32, copy=False) if labels else data
                existing.metadata["nvitk_ortho_t"] = t_key
            copy_plane_style(layer, existing)
            existing.plane = {"position": point, "normal": normal, "thickness": float(thickness)}
            out.append(existing)
        return out

    data = _layer_volume(layer, t_key)
    if data is None or data.ndim != 3:
        raise ValueError("3D planes need a 3D (or 3D+t) layer.")

    spatial = layer_spatial_kwargs(layer, ndim=3)
    labels = type(layer).__name__ == "Labels" or is_label_like_layer(layer)
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
            if labels:
                existing = viewer.add_labels(
                    np.asarray(data).astype(np.int32, copy=False),
                    name=name, depiction="plane", opacity=1.0, **spatial,
                )
                existing._nvitk_label_like = True
            else:
                # Translucent at full opacity draws the slice exactly as the 2D
                # view does; additive (the old setting) added the volume's own
                # brightness on top wherever the plane crossed it.
                existing = viewer.add_image(
                    data, name=name, depiction="plane", rendering="mip",
                    blending="translucent", opacity=1.0, **spatial,
                )
            existing.metadata["nvitk_ortho_t"] = t_key
        copy_plane_style(layer, existing)
        existing.plane = {"position": point, "normal": normal, "thickness": float(thickness)}
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


def _same_corners(shapes_layer: Any, corners: Any) -> bool:
    """Whether *shapes_layer* already outlines exactly *corners*."""
    try:
        current = shapes_layer.data
        if len(current) != 1:
            return False
        old = np.asarray(current[0], dtype=float)
        new = np.asarray(corners, dtype=float)
        return old.shape == new.shape and bool(np.array_equal(old, new))
    except Exception:  # noqa: BLE001 — an unreadable layer is simply rewritten
        return False


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
    shape = _spatial_shape(layer)
    if shape is None or len(shape) != 3:
        raise ValueError("Slice outlines need a 3D (or 3D+t) layer.")
    by_name = {str(getattr(l, "name", "")): l for l in viewer.layers}
    # The outlines are 3D shapes: a 3D+t layer lends them its spatial placement.
    spatial = layer_spatial_kwargs(layer, ndim=3)

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
            # Only write when the outline actually moved. Assigning ``data`` on a
            # Shapes layer runs Napari's whole event cascade — re-validating the
            # model, recomputing the world extent — for ~11 ms, and dragging one
            # crosshair leaves the other two planes exactly where they were.
            if not _same_corners(existing, corners):
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
    # A 3D+t layer's spatial dims are its trailing three; *displayed* and
    # *data_point* are spatial (0-2), so shift them onto the layer's own dims.
    lead = max(int(getattr(layer, "ndim", 3)) - 3, 0)
    transform = layer._transforms.simplified.set_slice([lead + int(d) for d in displayed])
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
    shape = _spatial_shape(layer) or tuple(int(v) for v in layer.data.shape[-3:])
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


#: Preference key holding the 3D card's toggles (``resample``, ``orientation_marker``).
_ORTHO_PREFS_KEY = "ortho_view"


def _ortho_prefs() -> dict[str, Any]:
    """The 3D card's stored toggles (empty when nothing is stored)."""
    try:
        from nvitk.gui.core.prefs import load_prefs

        stored = load_prefs().get(_ORTHO_PREFS_KEY)
    except Exception:  # noqa: BLE001 — a preference must never break the panel
        return {}
    return dict(stored) if isinstance(stored, dict) else {}


def _save_ortho_pref(key: str, value: Any) -> None:
    """Remember one 3D-card toggle for the next session."""
    try:
        from nvitk.gui.core.prefs import save_prefs

        save_prefs({_ORTHO_PREFS_KEY: {**_ortho_prefs(), key: value}})
    except Exception:  # noqa: BLE001
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

        # Adding or moving layers arrives in bursts — a tool that drops four
        # overlays on the canvas fires four inserts. Rebuild once when they stop.
        # Zero-interval: the point is not to wait but to land *after* the write
        # that the event announced, and to fold a drag's many events into one.
        self._data_timer = QTimer(self)
        self._data_timer.setSingleShot(True)
        self._data_timer.setInterval(0)
        self._data_timer.timeout.connect(self._flush_dirty_data)

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(_REBUILD_SYNC_MS)
        self._rebuild_timer.timeout.connect(self._rebuild_sources)

        self._status = QLabel("Select a 3D or 3D+t image or labels layer.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")

        # Time row: only shown while a drawn layer is 3D+t. Kept in step with the
        # Napari time slider both ways, so the cine played there drives these
        # views too.
        self._time_label = QLabel("t 1 / 1")
        self._time_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 10px;")
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
            f"color: {COLOR_MUTED}; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
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
        try:
            viewer.dims.events.current_step.connect(self._on_viewer_dims)
        except Exception:  # noqa: BLE001 — a viewer without dims events has no time to follow
            pass

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
        # Drawing labels in the views: the Labeling tab's tools (or the active
        # Labels layer's mode) take a click before the crosshair does.
        from nvitk.gui.labels.ortho_edit import OrthoLabelEditor

        self._label_editor = OrthoLabelEditor(self)
        for cell, view in enumerate(self._slice_views):
            view.edit_handler = functools.partial(self._label_editor.handle, cell)
        # Painted-label refreshes during a stroke, folded to one per event-loop turn.
        self._painted: list[Any] = []
        from qtpy.QtGui import QKeySequence
        from qtpy.QtWidgets import QShortcut

        for keys, slot in (
            (QKeySequence.StandardKey.Undo, self._label_editor.undo),
            (QKeySequence.StandardKey.Redo, self._label_editor.redo),
            (QKeySequence("Ctrl+Shift+Z"), self._label_editor.redo),
            (QKeySequence("Escape"), self._label_editor.cancel),
        ):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        grid.addWidget(self._build_controls(), 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        #: What a click does while a labeling tool is on (or why it cannot draw).
        self._draw_hint = QLabel("")
        self._draw_hint.setWordWrap(True)
        self._draw_hint.setVisible(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._status)
        root.addWidget(self._draw_hint)
        root.addWidget(self._time_box)
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

        self._resample_box = QCheckBox("Resample layers on other grids")
        self._resample_box.setToolTip(
            "On: a visible layer on another grid (another series, a registration "
            "output) is resampled onto the active layer's grid and drawn. Off: only "
            "layers already on that grid are drawn — no resampling cost, no "
            "interpolated pixels."
        )
        self._resample_box.setChecked(self._resample_offgrid)
        self._resample_box.toggled.connect(self._on_resample_toggled)
        card.add(self._resample_box)

        self._marker_box = QCheckBox("Orientation figure on the 3D canvas")
        self._marker_box.setToolTip(
            "A person and R/L, A/P, H/F arrows in the corner of the 3D view, turning "
            "with the camera — the marker of the QC reports' 3D renders."
        )
        self._marker_box.toggled.connect(self._on_marker_toggled)
        card.add(self._marker_box)

        # One row per drawn layer: whether it gets a slice image in 3D, and
        # whether the see-inside cut opens it. A mask can stay whole while the
        # CT around it is cut away — or the other way round.
        self._layers_table = QTableWidget(0, 3)
        self._layers_table.setHorizontalHeaderLabels(["Layer", "3D slice", "Cut"])
        self._layers_table.verticalHeader().setVisible(False)
        self._layers_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._layers_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self._layers_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._layers_table.setMinimumHeight(70)
        self._layers_table.setMaximumHeight(150)
        self._layers_table.setToolTip(
            "3D slice: draw this layer's slice image on the 3D planes (needs "
            "'…and the slice image' below). Cut: let 'See inside' open this layer."
        )
        self._layers_table.itemChanged.connect(self._on_table_item_changed)
        card.add(self._layers_table)

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
        if usable and int(getattr(layer.data, "ndim", 0)) not in (3, 4):
            usable = False
        if not usable:
            if self._play_btn.isChecked():
                self._play_btn.setChecked(False)
            self._layer = None
            self._data = None
            self._status.setText("Select a 3D or 3D+t image or labels layer.")
            for view in self._slice_views:
                view.set_slice(None, index=0, count=1)
            return

        # Re-binding the layer already shown (a refresh, a selection round-trip)
        # keeps the crosshair where the user put it.
        same_layer = layer is self._layer and self._data is not None
        # Binding a *different* layer that sits on the same grid changes nothing
        # anyone can see: the composite is drawn from every visible layer, not
        # from the bound one, and the bound layer only supplies the geometry —
        # which is, by definition, identical. Clicking between a CT and its
        # segmentation was re-slicing and re-compositing all three views for a
        # pixel-for-pixel identical result, ~60 ms of it on a whole-body study.
        same_grid = (
            not same_layer
            and self._layer is not None
            and self._data is not None
            and _grid_key(layer) == _grid_key(self._layer)
        )
        self._layer = layer
        self._data = _layer_volume(layer, self._time)
        if self._data is None:
            self._layer = None
            self._status.setText("This layer has no 3D volume to show.")
            return
        self._views = _axis_views(layer)
        # Keep what stays valid. On the same grid the rendered slices are the
        # ones already on screen, so clearing them would force a redraw that
        # reproduces them exactly — and resetting the crosshair below would
        # leave the panel claiming a position it is not showing.
        if not same_grid:
            self._rendered = {}
        # Windowed once for the volume: per-slice percentiles both cost more and
        # make the same tissue change brightness as you scroll.
        # Per-layer colour state lives on each RenderSource now; this layer is
        # kept only as the geometry anchor everything else is drawn against.
        # The resample cache is deliberately *not* cleared here. Its key already
        # carries the reference grid, so an entry for another grid can never be
        # returned by mistake — and throwing it away meant that switching to a
        # layer on a different grid and back re-ran a whole-volume affine
        # transform each way, seconds at a time on a large study. The budget
        # below evicts instead.
        spacing = layer_spacing(layer)
        self._spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))[:3]
        if len(self._spacing) < 3:
            self._spacing = (1.0, 1.0, 1.0)
        if not (same_layer or same_grid) or len(self._position) != self._data.ndim:
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

    # ── render sources ───────────────────────────────────────────────────────

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
        layer's identity threw away every cached source — host copies, slice
        caches, resampled volumes — each time the user clicked a different layer,
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
        if self._layer is None or _same_grid(layer, self._layer):
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
        # layer list changes at all, and the expensive parts — the host copy, the
        # slice cache, any resampling — depend on the layer and the reference
        # grid, neither of which a reorder or an unrelated insert touches.
        reference = self._reference_key()
        keep = self._source_cache if self._source_ref == reference else {}
        cache: dict[int, RenderSource] = {}
        sources: list[RenderSource] = []
        self._offgrid_skipped = []
        for layer in self._candidate_layers():
            if not bool(getattr(layer, "visible", True)):
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
            # emits set_data from ``Layer._refresh_sync(data_displayed=True)`` —
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

    def _on_visible_changed(self, _event: Any = None) -> None:
        """A layer was shown or hidden: the draw list changed."""
        self.refresh_sources()

    def _on_style_changed(self, _event: Any = None) -> None:
        """A colour or opacity moved: refresh the tables, keep the data."""
        for source in self._sources:
            self._refresh_style(source)
        self._style_timer.start()

    def _on_interpolation_changed(self, _event: Any = None, *, layer: Any = None) -> None:
        """A layer's interpolation changed: new screen filter, reslice order, planes.

        A layer resampled from another grid is resampled again — its cached copy
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
        drawn = [source for source in self._sources if source.opacity > 0.0]

        def _rgba(source: RenderSource) -> np.ndarray:
            """One layer's slice for this view, as RGBA (thread-safe: pure NumPy)."""
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
        """Refresh the views, re-rendering only those whose slice actually moved.

        A crosshair move changes at most two of the three slices; redrawing all of
        them on every scroll step is wasted work on a large volume.
        """
        if self._data is None or self._layer is None:
            return
        oblique = self.is_oblique()
        self._btn_reset_orient.setEnabled(oblique)
        try:
            # The Labeling box and a traced vessel follow the crosshair's planes.
            self._label_editor.overlays()
        except Exception:  # noqa: BLE001 — an overlay must never stop the views
            pass
        todo: list[int] = []
        for position, (widget, view) in enumerate(zip(self._slice_views, self._views)):
            widget._view = view
            widget._title.setText(view.title + ("  ·  oblique" if oblique else ""))
            index = self._position[view.axis]
            widget.set_lines(self._crosshair_lines(position))
            # A turned plane moves with the crosshair in every direction, so the
            # "same index, skip the redraw" shortcut no longer holds.
            if force or oblique or self._rendered.get(view.axis) != index:
                self._rendered[view.axis] = index
                todo.append(position)
            else:
                # Same slice, new crosshair: repaint the lines, keep the image.
                widget.set_crosshair(view.to_pixel(self._position, self._data.shape))
        if not todo:
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
                crosshair=view.to_pixel(self._position, self._data.shape),
            )

    # ── interaction ──────────────────────────────────────────────────────────

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
        if self._layer is None:
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
        """Name the 3D cut layers are keyed on — the bound layer's."""
        return str(getattr(self._layer, "name", "volume"))

    # ── per-layer choices ────────────────────────────────────────────────────

    def _wants_plane(self, source: RenderSource) -> bool:
        """Whether *source*'s slice image goes on the 3D planes.

        Defaults to the bound layer only — the historical behaviour. A layer
        resampled from another grid cannot: plane positions are voxel indices
        of the bound grid, which are not its own.
        """
        if source.resampled:
            return False
        key = id(source.layer)
        if key not in self._plane_choice:
            # Decided once, when the layer is first drawn. Following "whichever
            # layer is bound now" flipped the 3D slices to a segmentation the moment
            # it was clicked — on the same grid nothing redraws, so the table said
            # one thing and the canvas another until the crosshair next moved.
            self._plane_choice[key] = source.layer is self._layer
        return self._plane_choice[key]

    def _wants_cut(self, layer: Any) -> bool:
        """Whether the see-inside cut opens *layer* (default: every drawn layer)."""
        return self._cut_choice.get(id(layer), True)

    def _refresh_layers_table(self) -> None:
        """One row per drawn layer, with its 3D-slice and cut checkboxes."""
        self._table_guard = True
        try:
            table = self._layers_table
            table.setRowCount(len(self._sources))
            self._table_layers = [s.layer for s in self._sources]
            for row, source in enumerate(self._sources):
                name = QTableWidgetItem(str(getattr(source.layer, "name", "?")))
                name.setFlags(Qt.ItemIsEnabled)
                table.setItem(row, 0, name)
                plane = QTableWidgetItem()
                flags = Qt.ItemIsUserCheckable | (Qt.ItemIsEnabled if not source.resampled else Qt.NoItemFlags)
                plane.setFlags(flags)
                plane.setCheckState(Qt.Checked if self._wants_plane(source) else Qt.Unchecked)
                if source.resampled:
                    plane.setToolTip("Resampled from another grid: no 3D slice image.")
                table.setItem(row, 1, plane)
                cut = QTableWidgetItem()
                cut.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                cut.setCheckState(Qt.Checked if self._wants_cut(source.layer) else Qt.Unchecked)
                table.setItem(row, 2, cut)
        finally:
            self._table_guard = False

    def _on_table_item_changed(self, item: Any) -> None:
        """A 3D-slice or cut checkbox was toggled in the layers table."""
        if self._table_guard or item is None:
            return
        row, col = item.row(), item.column()
        layers = getattr(self, "_table_layers", [])
        if row >= len(layers) or col not in (1, 2):
            return
        layer = layers[row]
        checked = item.checkState() == Qt.Checked
        if col == 1:
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
        """Apply (or clear) the see-inside cut on the layers ticked for it.

        Every drawn layer is cut by default, overlays included: cutting only the
        base leaves a segmentation floating in front of the opened volume,
        covering the very interior the cut was made to expose. The layers table
        takes any of them out — keep a mask whole while the CT around it opens.
        """
        if self._layer is None:
            return
        side = str(self._clip_side.currentData() or "off")
        axis_data = self._clip_axis.currentData()
        axis = int(axis_data) if axis_data is not None else 0
        displayed = displayed_axes(self._viewer)
        drawn = [s.layer for s in self._sources] or [self._layer]
        targets = [layer for layer in drawn if self._wants_cut(layer)]
        # A layer that dropped out of the composite — or was unticked — keeps
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

    # ── time (3D+t) ──────────────────────────────────────────────────────────

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
        """``t 3 / 15  ·  0.120 s`` — the frame and, when known, its time."""
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
    if panel is None:
        panel = OrthoViewerPanel(viewer)
        attach_ortho_dock(viewer, panel)
        try:
            viewer._nvitk_ortho_panel = panel
        except Exception:
            pass
        # Next turn: the canvas has to exist before the marker can be put on it.
        QTimer.singleShot(0, panel.restore_marker_preference)
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
        # Nor is selecting something that is not a volume — a localizer, a single
        # tracker slice, a screen capture. Binding one makes it the grid every
        # other layer is resampled onto (a one-voxel slab), or blanks the panel;
        # the volume already on screen is what the user still wants to look at.
        bound = panel.bound_layer()
        if (
            active is not None
            and not is_volume_layer(active)
            and bound is not None
            and bound in viewer.layers
        ):
            return
        # Napari emits `active` twice for a single click, with the same layer
        # both times. Re-binding is not free — it re-materialises the volume,
        # rebuilds every render source and forces a redraw — so the second one
        # is pure waste. Data and layer-list changes arrive on their own events.
        if active is not None and active is panel.bound_layer():
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
    "copy_plane_style",
    "layer_time_axis",
    "layer_time_count",
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
