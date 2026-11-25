"""What the views draw: render sources per layer, grid matching and resampling, time axes,
slice caches, and intensity projections."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.parallel import parallel_map
from nvitk.gui.labels.visibility import (
    is_label_like_layer,
    label_source_data,
)
from nvitk.gui.viz.ortho.constants import (
    _PLANE_LAYER_SUFFIX,
    _BOX_LAYER_SUFFIX,
    _PROJECTION_CHUNK_VOXELS,
    _SLICE_CACHE_BUDGET,
)
from nvitk.gui.viz.ortho.geometry import (
    AxisView,
    _axis_views,
)

def _resample_budget() -> int:
    """Ceiling on cached resampled volumes: 15 % of RAM, never under 512 MB.

    One off-grid layer costs a full copy on the active grid. The old fixed
    512 MB refused to draw any off-grid layer at all next to a 512x512x416
    float64 CT (870 MB on its own), on a workstation with tens of GB free.
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
    #: Drawn smoothed when magnified: the layer's napari 2D interpolation.
    smooth: bool = False
    #: Distinct label ids in ``data``, computed once. Finding them is a full
    #: ``np.unique`` over the volume (~24 ms on a 50 MB mask) and they cannot
    #: change without ``data`` changing, which rebuilds the source anyway, so
    #: recomputing it on every style event was the single largest cost of
    #: clicking a layer in the list.
    label_ids: list[int] | None = None
    #: ``(max_id + 1, 4)`` uint8 lookup built once per style refresh, so a slice
    #: render is a single fancy-index instead of a per-slice unique + rebuild.
    label_table: np.ndarray | None = None
    #: Projections of ``data`` by ``(axis, kind, slab start, slab end)``. Kept here so
    #: they live exactly as long as the data they were made from: an edit or a new time
    #: point builds a new source, and with it an empty cache.
    projections: dict[tuple, np.ndarray] = field(default_factory=dict)


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
    Materialising a volume here (which is what reading the data to find out
    costs) made every tool that adds a layer pay for every layer on screen.
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
    # Two axes: a plain 2D image, drawn as a one-slice stack (see _is_plain_2d).
    return shape is not None and len(shape) >= 2


def _is_plain_2d(layer: Any) -> bool:
    """A plain 2D image or labels layer, with no third axis at all: a photo, a plain TIFF,
    a 2D tool output. A 2D slice with a real position is a one-voxel-thick 3D layer
    instead (see :func:`flat_axis`)."""
    if type(layer).__name__ not in DRAWN_LAYER_TYPES or bool(getattr(layer, "rgb", False)):
        return False
    shape = _layer_shape(layer)
    return shape is not None and len(shape) == 2


def flat_axis(layer: Any) -> int | None:
    """The axis a 2D image lies across in the panel's 3D layout; ``None`` for a volume.

    ``0`` for a plain 2D layer (drawn as a one-slice stack); for a positioned slice (one
    spatial axis one voxel thick, a 2D+t series included) that axis.
    """
    if _is_plain_2d(layer):
        return 0
    if not is_drawable_layer(layer):
        return None
    spatial = _spatial_shape(layer)
    if spatial is None:
        return None
    thin = [i for i, n in enumerate(spatial) if int(n) == 1]
    return thin[0] if len(thin) == 1 else None


def is_flat_layer(layer: Any) -> bool:
    """Whether *layer* is a 2D image the panel shows in its first view (:func:`flat_axis`)."""
    return flat_axis(layer) is not None


def is_volume_layer(layer: Any) -> bool:
    """A layer the panel can bind to as its grid: three spatial axes, none of them
    a single voxel. A positioned 2D slice is a one-slice 3D layer, and a 2D+t
    series a one-slice 3D+t one: drawable over a volume, but no grid of their own.
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
    if arr.ndim == 2 and _is_plain_2d(layer):
        return arr[None]  # a one-slice stack: a view, axis 0 across the image
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

    ``None`` rather than an exception for the case the aligner refuses (a layer
    with no affine and a different shape), because a panel that cannot draw one
    layer should say so and draw the rest.
    """
    from nvitk.gui.core.spatial import layer_affine

    # Both grids known: compose the voxel-to-voxel map and resample slab-parallel
    # on the worker pool (~7x on 8 threads); the same composition resample_to
    # uses, without building Image objects for a display copy.
    src_aff, ref_aff = layer_affine(layer), layer_affine(reference)
    ref_shape = _spatial_shape(reference)
    if src_aff is not None and ref_aff is not None and ref_shape is not None:
        try:
            from nvitk.transform.threaded import affine_transform

            vox = np.linalg.inv(ref_aff) @ src_aff          # source voxel -> reference voxel
            inv = np.linalg.inv(vox)                        # reference voxel -> source voxel
            host = to_numpy(data)
            # A display copy: intensities in float32 (half of a float64 CT's
            # footprint, far below what a screen can show); labels keep their ids.
            out_dtype = host.dtype if int(order) == 0 else np.float32
            return affine_transform(
                host, inv[:3, :3], offset=inv[:3, 3], output_shape=ref_shape,
                order=int(order), mode="constant", cval=0.0, prefilter=int(order) > 1,
                output=out_dtype,
            )
        except Exception:  # noqa: BLE001 (fall through to the generic aligner)
            pass

    from nvitk.core.backend import using
    from nvitk.gui.core.spatial import align_mask_to_reference_layer

    try:
        with using("cpu"):
            _ref, aligned, _resampled = align_mask_to_reference_layer(
                layer, reference, data, order=int(order)
            )
        return to_numpy(aligned.data)
    except Exception:  # noqa: BLE001 (reported by the caller, never raised into paint)
        return None


def _axis_index(ndim: int, axis: int, index: int) -> tuple:
    """Basic-indexing key selecting *index* along *axis*: a view, never a copy."""
    key: list[Any] = [slice(None)] * int(ndim)
    key[int(axis)] = int(index)
    return tuple(key)


def _slice_of(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    """The 2D slice of *data* at *index* along *axis*, clamped into range.

    Basic indexing, deliberately not ``np.take``: ``take`` first makes the whole
    input C-contiguous, and a NIfTI volume comes off disk in Fortran order, so
    every slice of a 512x512x416 CT copied the full 870 MB volume (~1.4 s per
    slice, per layer, per view). An indexed view reads only the slice.
    """
    n = int(data.shape[axis])
    idx = int(np.clip(index, 0, max(n - 1, 0)))
    return data[_axis_index(data.ndim, axis, idx)]


def project_volume(
    data: np.ndarray,
    axis: int,
    kind: str,
    lo: int | None = None,
    hi: int | None = None,
    *,
    label: bool = False,
) -> np.ndarray:
    """*data* projected along *axis* over the slab ``[lo, hi)`` (the whole axis when ``None``).

    *kind*: ``"max"`` (MIP), ``"min"`` (MinIP) or ``"mean"`` (average). A label volume
    (*label*) always takes the highest id along the ray, so a projected mask shows every
    structure in it. Large slabs are reduced in parallel chunks across the plane.
    """
    axis = int(axis)
    n = int(data.shape[axis])
    lo = 0 if lo is None else int(np.clip(lo, 0, max(n - 1, 0)))
    hi = n if hi is None else int(np.clip(hi, lo + 1, n))
    slab = data[tuple(slice(lo, hi) if d == axis else slice(None) for d in range(data.ndim))]
    mode = "max" if label else str(kind)

    def _reduce(block: np.ndarray) -> np.ndarray:
        """One block's projection along *axis*."""
        if mode == "max":
            return block.max(axis=axis)
        if mode == "min":
            return block.min(axis=axis)
        return block.mean(axis=axis, dtype=np.float32)

    if slab.size <= _PROJECTION_CHUNK_VOXELS:
        return _reduce(slab)
    # Split across the plane's first axis: the reductions release the GIL, so the
    # chunks run concurrently.
    split = next(d for d in range(slab.ndim) if d != axis)
    edges = np.linspace(0, int(slab.shape[split]), 9).astype(int)
    blocks = [
        slab[tuple(slice(a, b) if d == split else slice(None) for d in range(slab.ndim))]
        for a, b in zip(edges[:-1], edges[1:]) if b > a
    ]
    parts = parallel_map(_reduce, blocks)
    return np.concatenate(parts, axis=split - (1 if split > axis else 0))


def _flat_views(layer: Any, axis: int, viewer: Any = None) -> list[AxisView]:
    """The views for a 2D image: its own plane first (drawn in the first, axial pane),
    then the two views that stand empty."""
    if _is_plain_2d(layer):
        # Drawn as stored, the way Napari shows a 2D layer: rows down, columns across.
        return [
            AxisView(axis=0, title="2D image", rows=1, cols=2),
            AxisView(axis=1, title="—", rows=0, cols=2),
            AxisView(axis=2, title="—", rows=0, cols=1),
        ]
    views = _axis_views(layer, viewer)
    head = next(v for v in views if v.axis == axis)
    rest = [v for v in views if v.axis != axis]
    return [replace(head, title=f"2D image  ·  {head.title}"), *rest]


#: Guards every panel's shared reordered-slice budget across render threads.
_POOL_LOCK = threading.Lock()


class SliceCache:
    """Serves 2D slices of a volume, keeping the slow axes contiguous.

    A C-ordered volume gives axis 0 for free, but a slice along the last axis is a
    strided gather over the whole array: around 40x slower on a 400x512x512 CT,
    which is exactly why scrolling axial felt heavier than the others. Reordering
    that axis into its own contiguous copy makes every axis equally cheap.

    The copy is built lazily, on the first scroll of that axis, and only for the
    *innermost* axis (the smallest stride: last for C order, first for the
    Fortran order NIfTI arrays arrive in): that is the one whose slice is a
    gather with a stride between every element. The other axes yield
    contiguous runs and measure no faster reordered, so reordering them was
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
        """Wrap *data*, reordering nothing until an axis is asked for."""
        self._data = data
        self._budget = int(budget_bytes)
        self._pool = pool
        self._reordered: dict[int, np.ndarray | None] = {}
        # Views render on worker threads: two of them asking for the same axis
        # must not both build (and both charge the pool for) the same copy.
        self._lock = threading.Lock()

    def _worth_reordering(self, axis: int, arr: np.ndarray) -> bool:
        """Whether a contiguous copy of *axis* would earn its memory."""
        # Only the innermost (smallest-stride) axis is slow to slice:
        # its slice gathers one element per stride. That is the *last* axis of
        # a C-ordered array but the *first* of a Fortran-ordered one, which is
        # how nibabel hands NIfTI volumes over, so ask the strides, not the
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
