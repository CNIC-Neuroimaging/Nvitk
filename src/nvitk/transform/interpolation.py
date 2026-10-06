"""
Axis-selective resampling and slice interpolation for images and label masks.

Description
-----------
The everyday interpolation jobs that :func:`~nvitk.transform.isotropy.isotropy`
and :func:`~nvitk.transform.resampling.resample_to` do not cover:

- :func:`resample_axes` — up/down-sample any subset of axes (spatial or time) by a
  factor, to a target spacing, or to a target size, with an anti-aliasing filter
  on the axes that shrink. Masks resample by nearest neighbour or *shape-based*
  (signed-distance) interpolation, which upsamples a staircase mask into a smooth
  one instead of blowing up its voxels.
- :func:`block_reduce_axes` — integer downsampling by block mean / max / min /
  median / sum, or a majority vote for label maps.
- :func:`interpolate_mask_slices` — fill the slices between sparsely annotated
  ones (draw every 5th slice, interpolate the rest): shape-based interpolation of
  each label between consecutive annotated slices, along one or several axes.
- :func:`fill_missing_slices` — rebuild missing (empty / NaN) or listed slices of
  an image from their neighbours along an axis.

Geometry
--------
Resampling keeps the field of view: voxel *edges* stay aligned, so the new voxel
``j`` along an axis shrunk by ``s = n_old / n_new`` sits at old index
``(j + 0.5) * s - 0.5`` (SciPy's ``grid_mode``). :func:`resample_index_map` gives
that map as a homogeneous matrix; the output's affine is the input's composed
with it, its spacing ``s`` times the old one, and ``t_res`` too when time is
resampled. Axes are array axes; :func:`parse_axes` also accepts the letters of
``image.axes`` (``"Z"``, ``"X,Y"``, ``"T"``).

Arrays are processed on the host (``scipy.ndimage``; resampling is slab-parallel
through :mod:`nvitk.transform.threaded`).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as _hnp
from scipy import ndimage as _hndi

from nvitk.core.array import to_numpy
from nvitk.types import Image

#: Spline orders by name, for pickers.
INTERPOLATION_ORDERS: dict[str, int] = {
    "nearest": 0,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "quartic": 4,
    "quintic": 5,
}

#: How a label map is resampled: nearest neighbour, or shape-based (per-label
#: signed distance, interpolated linearly and thresholded — smooth when upsampling).
LABEL_METHODS: tuple[str, ...] = ("nearest", "shape")

#: Block reductions of :func:`block_reduce_axes` (``mode`` = majority vote, for labels).
REDUCE_METHODS: tuple[str, ...] = ("mean", "max", "min", "median", "sum", "mode")

#: How :func:`interpolate_mask_slices` fills a gap: shape-based (signed-distance
#: blend of the two annotated slices) or a copy of the nearest annotated slice.
SLICE_METHODS: tuple[str, ...] = ("shape", "nearest")

#: How the target of :func:`resample_axes` is given.
RESAMPLE_MODES: tuple[str, ...] = ("factor", "spacing", "size")

_TIME_LETTERS = ("T", "C")


# ──────────────────────────────────────────────────────────────────────────────
# Axes and geometry
# ──────────────────────────────────────────────────────────────────────────────


def _plane(data: _hnp.ndarray, axis: int, index: int) -> _hnp.ndarray:
    """Slice *index* along *axis* as a view.

    Not ``np.take``: that first makes the whole input C-contiguous, so on a
    Fortran-ordered NIfTI array every slice would copy the entire volume.
    """
    key = [slice(None)] * data.ndim
    key[axis] = int(index)
    return data[tuple(key)]


def _axes_string(image: Image) -> str:
    """``image.axes`` upper-cased, or ``""`` when it does not label every axis."""
    axes = str(getattr(image, "axes", "") or "").upper()
    return axes if len(axes) == image.ndim else ""


def parse_axes(spec: Any, ndim: int, axes: str | None = None) -> tuple[int, ...]:
    """Array axes from ``"0,2"`` / ``"Z"`` / ``"x,y"`` / ``"T"`` / ``"all"`` / a sequence.

    Letters are looked up in *axes* (the image's axis string). ``"all"`` (or an
    empty spec) is every axis; ``"spatial"`` every axis except a time axis.
    Negative indices count from the end.
    """
    letters = (axes or "").upper()
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "all")):
        return tuple(range(ndim))
    if isinstance(spec, str) and spec.strip().lower() == "spatial":
        return tuple(i for i in range(ndim) if not (letters and letters[i] in _TIME_LETTERS))
    if isinstance(spec, str):
        tokens = [t.strip() for t in spec.replace(";", ",").replace(" ", ",").split(",") if t.strip()]
    elif isinstance(spec, (int, _hnp.integer)):
        tokens = [int(spec)]
    else:
        tokens = list(spec)
    out: list[int] = []
    for tok in tokens:
        if isinstance(tok, str) and not tok.lstrip("-").isdigit():
            ch = tok.upper()
            if len(ch) != 1 or ch not in letters:
                raise ValueError(
                    f"Unknown axis {tok!r}: use indices 0..{ndim - 1}"
                    + (f" or one of {', '.join(letters)}" if letters else "")
                    + "."
                )
            idx = letters.index(ch)
        else:
            idx = int(tok)
            if idx < 0:
                idx += ndim
        if not 0 <= idx < ndim:
            raise ValueError(f"Axis {tok!r} is out of range for a {ndim}-D image.")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("Select at least one axis.")
    return tuple(sorted(out))


def _spatial_axes(image: Image) -> list[int]:
    """Array axes the 4x4 affine's columns describe, in column order."""
    letters = _axes_string(image)
    if letters:
        spatial = [i for i, ch in enumerate(letters) if ch not in _TIME_LETTERS]
        if len(spatial) >= 3:
            return spatial[:3]
    return list(range(min(3, image.ndim)))


def _time_axis(image: Image) -> int | None:
    """The array axis labelled ``T`` (or ``C``), if any."""
    letters = _axes_string(image)
    for ch in _TIME_LETTERS:
        if ch in letters:
            return letters.index(ch)
    return None


def axis_spacing(image: Image) -> list[float]:
    """Physical step per array axis: affine column norms / spacing, ``t_res`` for time."""
    out = [1.0] * image.ndim
    spatial = _spatial_axes(image)
    affine = image.affine
    if affine is not None and _hnp.asarray(affine).shape == (4, 4):
        aff = _hnp.asarray(to_numpy(affine), dtype=float)
        for col, ax in enumerate(spatial):
            out[ax] = float(_hnp.linalg.norm(aff[:3, col])) or 1.0
    else:
        sp = image.spacing
        if sp is not None:
            for col, ax in enumerate(spatial):
                if col < len(sp):
                    out[ax] = float(sp[col]) or 1.0
    t_ax = _time_axis(image)
    if t_ax is not None:
        t_res = image.temporal_resolution
        if t_res:
            out[t_ax] = float(t_res)
    return out


def resample_index_map(old_shape: Sequence[int], new_shape: Sequence[int]) -> _hnp.ndarray:
    """Homogeneous ``(n+1, n+1)`` map from new voxel index to old voxel index.

    Field-of-view preserving (voxel edges aligned): along an axis resized from
    ``n`` to ``m`` voxels, ``old = new * s + (s - 1) / 2`` with ``s = n / m``.
    Compose it after a data→world affine to place the resampled array.
    """
    n = len(old_shape)
    if len(new_shape) != n:
        raise ValueError("old_shape and new_shape need the same number of axes.")
    mat = _hnp.eye(n + 1)
    for ax, (a, b) in enumerate(zip(old_shape, new_shape)):
        s = float(a) / float(b)
        mat[ax, ax] = s
        mat[ax, n] = (s - 1.0) / 2.0
    return mat


def _regridded(image: Image, data: Any, old_shape: Sequence[int]) -> Image:
    """*data* as a new :class:`Image` on the grid :func:`resample_index_map` describes."""
    new_shape = tuple(int(v) for v in data.shape)
    index_map = resample_index_map(old_shape, new_shape)
    out = image.with_data(data)
    meta = out.metadata
    meta["shape"] = new_shape
    spatial = _spatial_axes(image)
    affine = image.affine
    if affine is not None and _hnp.asarray(to_numpy(affine)).shape == (4, 4):
        sub = _hnp.eye(4)
        n = image.ndim
        for col, ax in enumerate(spatial):
            sub[col, col] = index_map[ax, ax]
            sub[col, 3] = index_map[ax, n]
        meta["affine"] = _hnp.asarray(to_numpy(affine), dtype=float) @ sub
        meta.pop("affine_source", None)
    old_spacing = axis_spacing(image)
    scale = [index_map[ax, ax] for ax in range(image.ndim)]
    new_spatial = tuple(old_spacing[ax] * scale[ax] for ax in spatial)
    if len(new_spatial) == 3:
        meta["spacing"] = new_spatial
        meta["x_res"], meta["y_res"], meta["z_res"] = new_spatial
    t_ax = _time_axis(image)
    if t_ax is not None and image.temporal_resolution:
        t_res = float(image.temporal_resolution) * scale[t_ax]
        meta["t_res"] = t_res
        if "temporal_resolution" in meta:
            meta["temporal_resolution"] = t_res
        meta.pop("frame_times_s", None)
    return out


def target_shape(
    image: Image,
    *,
    axes: Sequence[int],
    mode: str = "factor",
    values: Sequence[float] | float = 2.0,
) -> tuple[int, ...]:
    """Output shape for :func:`resample_axes`.

    *mode* ``"factor"`` multiplies the selected axes' sizes (``2`` doubles them,
    ``0.5`` halves them); ``"spacing"`` asks for a physical step per axis (mm, or
    seconds on a time axis); ``"size"`` gives the voxel counts directly. *values*
    is one number for every selected axis or one per selected axis.
    """
    mode = str(mode).lower()
    if mode not in RESAMPLE_MODES:
        raise ValueError(f"mode must be one of {RESAMPLE_MODES}; got {mode!r}.")
    if isinstance(values, (int, float, _hnp.floating, _hnp.integer)):
        vals = [float(values)] * len(axes)
    else:
        vals = [float(v) for v in values]
        if len(vals) == 1:
            vals = vals * len(axes)
    if len(vals) != len(axes):
        raise ValueError(f"Give one value, or one per selected axis ({len(axes)}); got {len(vals)}.")
    shape = list(int(v) for v in image.shape)
    spacing = axis_spacing(image)
    for ax, val in zip(axes, vals):
        if val <= 0:
            raise ValueError("Resampling values must be positive.")
        if mode == "factor":
            new = shape[ax] * val
        elif mode == "spacing":
            new = shape[ax] * spacing[ax] / val
        else:
            new = val
        shape[ax] = max(1, int(round(new)))
    return tuple(shape)


# ──────────────────────────────────────────────────────────────────────────────
# Resampling
# ──────────────────────────────────────────────────────────────────────────────


def _resample_array(
    data: _hnp.ndarray,
    new_shape: Sequence[int],
    *,
    order: int,
    mode: str = "nearest",
    antialias: bool = True,
    output_dtype: Any = None,
) -> _hnp.ndarray:
    """Field-of-view-preserving spline resampling of *data* to *new_shape*."""
    from nvitk.transform.threaded import affine_transform

    index_map = resample_index_map(data.shape, new_shape)
    scale = _hnp.diag(index_map)[:-1]
    shift = index_map[:-1, -1]
    src = data
    if antialias and order > 0 and _hnp.any(scale > 1.0 + 1e-9):
        # Gaussian prefilter on the shrinking axes only (skimage's sigma rule).
        sigma = [max(0.0, (s - 1.0) / 2.0) for s in scale]
        src = _hndi.gaussian_filter(data.astype(_hnp.float32, copy=False), sigma=sigma, mode="nearest")
    if src.dtype == bool:
        src = src.astype(_hnp.uint8)
    out = affine_transform(
        src,
        scale,
        offset=shift,
        output_shape=tuple(int(v) for v in new_shape),
        order=int(order),
        mode=mode,
        output=output_dtype if output_dtype is not None else None,
    )
    return out


def _bbox(mask: _hnp.ndarray, margin: int) -> tuple[slice, ...] | None:
    """Bounding box of *mask* grown by *margin* voxels, or ``None`` when empty."""
    coords = _hnp.nonzero(mask)
    if not coords or coords[0].size == 0:
        return None
    return tuple(
        slice(max(0, int(c.min()) - margin), min(n, int(c.max()) + margin + 1))
        for c, n in zip(coords, mask.shape)
    )


def signed_distance(mask: _hnp.ndarray, spacing: Sequence[float] | None = None) -> _hnp.ndarray:
    """Signed Euclidean distance, positive inside *mask* (physical units with *spacing*).

    Inside voxels get their distance to the nearest outside voxel and vice versa,
    each shifted by half a voxel so the zero level sits on the boundary.
    """
    mask = _hnp.asarray(mask, dtype=bool)
    sampling = None if spacing is None else tuple(float(s) for s in spacing)
    if not mask.any():
        return _hnp.full(mask.shape, -_hnp.inf, dtype=_hnp.float32)
    if mask.all():
        return _hnp.full(mask.shape, _hnp.inf, dtype=_hnp.float32)
    inside = _hndi.distance_transform_edt(mask, sampling=sampling)
    outside = _hndi.distance_transform_edt(~mask, sampling=sampling)
    half = 0.5 * (min(sampling) if sampling else 1.0)
    return (_hnp.where(mask, inside - half, -(outside - half))).astype(_hnp.float32)


def _resample_labels_shape(
    labels: _hnp.ndarray,
    new_shape: Sequence[int],
    spacing: Sequence[float],
    label_ids: Iterable[int] | None = None,
) -> _hnp.ndarray:
    """Shape-based resampling of a label map: each label's signed distance,
    resampled linearly; every new voxel takes the label it is deepest inside."""
    from nvitk.transform.threaded import affine_transform

    old_shape = labels.shape
    index_map = resample_index_map(old_shape, new_shape)
    scale = _hnp.diag(index_map)[:-1]
    shift = index_map[:-1, -1]
    out = _hnp.zeros(tuple(int(v) for v in new_shape), dtype=labels.dtype)
    best = _hnp.zeros(out.shape, dtype=_hnp.float32)
    ids = sorted(int(v) for v in (label_ids if label_ids is not None else _hnp.unique(labels)) if int(v) != 0)
    margin = int(_hnp.ceil(max(2.0, 2.0 * float(scale.max()))))
    for lid in ids:
        mask = labels == lid
        box = _bbox(mask, margin)
        if box is None:
            continue
        sdf = signed_distance(mask[box], spacing)
        sdf = _hnp.nan_to_num(sdf, posinf=1e6, neginf=-1e6)
        # New voxels whose old coordinate falls inside the box.
        lo_new, hi_new, local_shift = [], [], []
        for ax, sl in enumerate(box):
            s, t = float(scale[ax]), float(shift[ax])
            lo = int(_hnp.ceil((sl.start - t) / s - 1e-9))
            hi = int(_hnp.floor((sl.stop - 1 - t) / s + 1e-9))
            lo, hi = max(lo, 0), min(hi, int(new_shape[ax]) - 1)
            lo_new.append(lo)
            hi_new.append(hi)
            local_shift.append(lo * s + t - sl.start)
        if any(h < l for l, h in zip(lo_new, hi_new)):
            continue
        sub_shape = tuple(h - l + 1 for l, h in zip(lo_new, hi_new))
        sub = affine_transform(
            sdf, scale, offset=_hnp.asarray(local_shift), output_shape=sub_shape,
            order=1, mode="nearest",
        )
        region = tuple(slice(l, h + 1) for l, h in zip(lo_new, hi_new))
        win = sub > best[region]
        best[region] = _hnp.where(win, sub, best[region])
        out[region] = _hnp.where(win, lid, out[region])
    return out


def resample_axes(
    image: Image,
    *,
    axes: Any = "all",
    mode: str = "factor",
    values: Sequence[float] | float = 2.0,
    shape: Sequence[int] | None = None,
    order: int | str = 1,
    antialias: bool = True,
    labels: bool | None = None,
    label_method: str = "nearest",
    boundary: str = "nearest",
) -> Image:
    """Up- or down-sample the selected *axes* of *image*.

    Parameters
    ----------
    axes
        Axes to resample (see :func:`parse_axes`): ``"2"``, ``"X,Y"``, ``"T"``…
    mode, values
        The target, as in :func:`target_shape` (factor, spacing or size).
    shape
        Full output shape, overriding *mode*/*values*.
    order
        Spline order 0–5 or a name from :data:`INTERPOLATION_ORDERS`.
    antialias
        Gaussian prefilter on axes that shrink (images only).
    labels
        Treat *image* as a label map. ``None``: integer/bool data with an order of
        0 is resampled as labels.
    label_method
        ``"nearest"`` or ``"shape"`` (signed-distance interpolation, smooth when
        upsampling; spatial axes only — a time axis is nearest-neighbour).
    boundary
        SciPy boundary mode for intensity interpolation.

    Returns
    -------
    Image
        Resampled image with its affine, spacing (and ``t_res``) updated; the
        field of view is preserved.
    """
    data = to_numpy(image.data)
    if isinstance(order, str):
        if order not in INTERPOLATION_ORDERS:
            raise ValueError(f"order must be 0-5 or one of {tuple(INTERPOLATION_ORDERS)}.")
        order = INTERPOLATION_ORDERS[order]
    order = int(order)
    sel = parse_axes(axes, data.ndim, _axes_string(image))
    if shape is not None:
        new_shape = tuple(int(v) for v in shape)
        if len(new_shape) != data.ndim:
            raise ValueError("shape needs one entry per axis.")
    else:
        new_shape = target_shape(image, axes=sel, mode=mode, values=values)
    if new_shape == tuple(data.shape):
        return image.copy()
    if labels is None:
        labels = data.dtype == bool or (_hnp.issubdtype(data.dtype, _hnp.integer) and order == 0)
    if labels:
        method = str(label_method).lower()
        if method not in LABEL_METHODS:
            raise ValueError(f"label_method must be one of {LABEL_METHODS}.")
        t_ax = _time_axis(image)
        if method == "shape" and t_ax is None:
            out = _resample_labels_shape(data, new_shape, axis_spacing(image))
        elif method == "shape":
            # Per time frame, then nearest along time.
            frame_shape = list(new_shape)
            n_t_new = frame_shape.pop(t_ax)
            spacing = [s for i, s in enumerate(axis_spacing(image)) if i != t_ax]
            frames = [
                _resample_labels_shape(_plane(data, t_ax, k), frame_shape, spacing)
                for k in range(data.shape[t_ax])
            ]
            stacked = _hnp.stack(frames, axis=t_ax)
            out = _resample_array(stacked, new_shape, order=0, antialias=False, output_dtype=data.dtype) \
                if n_t_new != data.shape[t_ax] else stacked
        else:
            out = _resample_array(data, new_shape, order=0, antialias=False, output_dtype=data.dtype)
        return _regridded(image, out, data.shape)
    out_dtype = data.dtype if order == 0 else (_hnp.float64 if data.dtype == _hnp.float64 else _hnp.float32)
    out = _resample_array(
        data, new_shape, order=order, mode=boundary, antialias=antialias, output_dtype=out_dtype,
    )
    return _regridded(image, out, data.shape)


def block_reduce_axes(
    image: Image,
    *,
    axes: Any = "all",
    factors: Sequence[int] | int = 2,
    method: str = "mean",
) -> Image:
    """Downsample *axes* by integer *factors*, reducing each block with *method*.

    Trailing voxels that do not fill a whole block are dropped (the field of view
    shrinks by less than one output voxel). ``"mode"`` is a majority vote — the
    reduction for label maps; ties go to the larger id.
    """
    data = to_numpy(image.data)
    method = str(method).lower()
    if method not in REDUCE_METHODS:
        raise ValueError(f"method must be one of {REDUCE_METHODS}.")
    sel = parse_axes(axes, data.ndim, _axes_string(image))
    if isinstance(factors, (int, _hnp.integer)):
        facs = [int(factors)] * len(sel)
    else:
        facs = [int(f) for f in factors]
        if len(facs) == 1:
            facs = facs * len(sel)
    if len(facs) != len(sel):
        raise ValueError("Give one factor, or one per selected axis.")
    if any(f < 1 for f in facs):
        raise ValueError("Block factors must be >= 1.")
    block = [1] * data.ndim
    for ax, f in zip(sel, facs):
        block[ax] = f
    new_shape = [n // b for n, b in zip(data.shape, block)]
    if any(n == 0 for n in new_shape):
        raise ValueError("A block factor is larger than its axis.")
    trimmed = data[tuple(slice(0, n * b) for n, b in zip(new_shape, block))]
    # Interleave (n0, b0, n1, b1, ...) and reduce over the block axes.
    view = trimmed.reshape([v for pair in zip(new_shape, block) for v in pair])
    red_axes = tuple(range(1, 2 * data.ndim, 2))
    if method == "mode":
        flat = _hnp.moveaxis(view, red_axes, tuple(range(data.ndim, 2 * data.ndim)))
        flat = flat.reshape(*new_shape, -1)
        ids = _hnp.unique(flat)
        best = _hnp.zeros(new_shape, dtype=_hnp.int64)
        out = _hnp.zeros(new_shape, dtype=data.dtype)
        for lid in ids:
            count = (flat == lid).sum(axis=-1)
            win = count >= best
            best = _hnp.where(win, count, best)
            out = _hnp.where(win, lid, out).astype(data.dtype)
    else:
        func = {"mean": _hnp.mean, "max": _hnp.max, "min": _hnp.min, "median": _hnp.median, "sum": _hnp.sum}[method]
        out = func(view, axis=red_axes)
        if method in ("max", "min"):
            out = out.astype(data.dtype)
        elif method == "mean" and data.dtype != _hnp.float64:
            out = out.astype(_hnp.float32)
    # Block reduction aligns voxel edges like the resampler, over the kept extent.
    kept = [n * b for n, b in zip(new_shape, block)]
    res = _regridded(_crop_image(image, kept), _hnp.ascontiguousarray(out), kept)
    return res


def _crop_image(image: Image, shape: Sequence[int]) -> Image:
    """*image* cropped to its leading *shape* (metadata otherwise unchanged)."""
    if tuple(shape) == tuple(image.shape):
        return image
    data = to_numpy(image.data)[tuple(slice(0, int(n)) for n in shape)]
    return image.with_data(data)


# ──────────────────────────────────────────────────────────────────────────────
# Slice interpolation
# ──────────────────────────────────────────────────────────────────────────────


def annotated_slices(mask: _hnp.ndarray, axis: int) -> _hnp.ndarray:
    """Indices along *axis* of the slices where *mask* has any voxel set."""
    other = tuple(i for i in range(mask.ndim) if i != axis)
    return _hnp.nonzero(_hnp.any(mask, axis=other))[0]


def sparsest_axis(mask: _hnp.ndarray) -> int:
    """The axis along which *mask*'s annotation has the most empty slices inside its extent.

    Sparse annotation (every *n*-th slice drawn) leaves gaps along exactly one axis;
    a solid mask has none along any, and the last axis is returned.
    """
    best_axis, best_gaps = mask.ndim - 1, -1
    for axis in range(mask.ndim):
        idx = annotated_slices(mask, axis)
        if idx.size < 2:
            continue
        gaps = int(idx[-1] - idx[0] + 1 - idx.size)
        if gaps > best_gaps:
            best_axis, best_gaps = axis, gaps
    return best_axis


def _slice_spacing(spacing: Sequence[float] | None, axis: int, ndim: int) -> tuple[float, ...] | None:
    """In-plane spacing of a slice normal to *axis*."""
    if spacing is None:
        return None
    return tuple(float(spacing[i]) for i in range(ndim) if i != axis)


def _interpolate_label_along(
    mask: _hnp.ndarray,
    axis: int,
    *,
    method: str,
    max_gap: int,
    spacing: Sequence[float] | None,
) -> _hnp.ndarray:
    """Boolean fill of the gaps between annotated slices of one label along *axis*."""
    fill = _hnp.zeros(mask.shape, dtype=bool)
    idx = annotated_slices(mask, axis)
    if idx.size < 2:
        return fill
    in_plane = _slice_spacing(spacing, axis, mask.ndim)
    cache: dict[int, _hnp.ndarray] = {}

    def _sdf(k: int) -> _hnp.ndarray:
        if k not in cache:
            sdf = signed_distance(_plane(mask, axis, k), in_plane)
            cache[k] = _hnp.nan_to_num(sdf, posinf=1e6, neginf=-1e6)
        return cache[k]

    for k0, k1 in zip(idx[:-1], idx[1:]):
        gap = int(k1 - k0 - 1)
        if gap <= 0 or (max_gap and gap > max_gap):
            continue
        for k in range(int(k0) + 1, int(k1)):
            t = (k - k0) / float(k1 - k0)
            if method == "nearest":
                src = k0 if t <= 0.5 else k1
                plane = _plane(mask, axis, int(src))
            else:
                plane = ((1.0 - t) * _sdf(int(k0)) + t * _sdf(int(k1))) > 0
            index = [slice(None)] * mask.ndim
            index[axis] = k
            fill[tuple(index)] = plane
    return fill


def interpolate_mask_slices(
    mask: Image,
    *,
    axes: Any = "auto",
    method: str = "shape",
    label_ids: Iterable[int] | None = None,
    max_gap: int = 0,
    overwrite: bool = False,
) -> Image:
    """Fill the slices between annotated ones, per label, along one or more axes.

    Parameters
    ----------
    mask
        Binary or label map (3D, or 3D+t — interpolated within each frame).
    axes
        Axes to interpolate along (:func:`parse_axes`), or ``"auto"`` for each
        label's sparsest axis (:func:`sparsest_axis`). Several axes are filled
        independently and united — e.g. annotation drawn on axial *and* sagittal
        slices.
    method
        ``"shape"``: blend the signed distances of the two annotated slices
        bounding a gap (smoothly morphs one contour into the next); ``"nearest"``:
        copy the nearest annotated slice.
    label_ids
        Labels to interpolate (default: every non-zero id).
    max_gap
        Leave gaps longer than this many slices alone (``0`` = any length): two
        separate structures along the axis are not bridged.
    overwrite
        Let a label fill voxels that already belong to another label (by default
        only background is filled).

    Returns
    -------
    Image
        The mask with its gaps filled, same grid and dtype.
    """
    data = to_numpy(mask.data)
    method = str(method).lower()
    if method not in SLICE_METHODS:
        raise ValueError(f"method must be one of {SLICE_METHODS}.")
    t_ax = _time_axis(mask) if data.ndim == 4 else None
    if data.ndim == 4 and t_ax is None:
        t_ax = 3
    spacing = axis_spacing(mask)
    out = data.copy()
    is_bool = data.dtype == bool
    ids = (
        [int(v) for v in label_ids]
        if label_ids is not None
        else ([1] if is_bool else sorted(int(v) for v in _hnp.unique(data) if int(v) != 0))
    )
    auto = isinstance(axes, str) and axes.strip().lower() in ("auto", "")
    letters = _axes_string(mask)

    frames = range(data.shape[t_ax]) if t_ax is not None else [None]
    for frame in frames:
        vol = data if frame is None else _plane(data, t_ax, frame)
        target = out if frame is None else None
        result = vol.copy() if frame is not None else None
        sp = spacing if frame is None else [s for i, s in enumerate(spacing) if i != t_ax]
        if auto:
            sel_axes = None
        else:
            sel_axes = parse_axes(axes, data.ndim, letters)
            if t_ax is not None:
                if t_ax in sel_axes:
                    raise ValueError("Interpolate masks along spatial axes; time is handled per frame.")
                sel_axes = tuple(a - (1 if a > t_ax else 0) for a in sel_axes)
        for lid in ids:
            lab = vol if is_bool else (vol == lid)
            if not lab.any():
                continue
            label_axes = (sparsest_axis(lab),) if sel_axes is None else sel_axes
            fill = _hnp.zeros(lab.shape, dtype=bool)
            for ax in label_axes:
                fill |= _interpolate_label_along(lab, ax, method=method, max_gap=int(max_gap), spacing=sp)
            fill &= ~lab
            dest = target if frame is None else result
            if not overwrite:
                fill &= dest == 0
            dest[fill] = True if is_bool else lid
        if frame is not None:
            index = [slice(None)] * data.ndim
            index[t_ax] = frame
            out[tuple(index)] = result
    return mask.with_data(out)


def _parse_indices(spec: Any, n: int) -> list[int]:
    """``"3,7-9,12"`` → ``[3, 7, 8, 9, 12]`` (within ``0..n-1``)."""
    if spec is None:
        return []
    if not isinstance(spec, str):
        return sorted({int(v) for v in spec if 0 <= int(v) < n})
    out: set[int] = set()
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if "-" in tok.lstrip("-"):
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(i for i in out if 0 <= i < n)


def missing_slices(data: _hnp.ndarray, axis: int) -> list[int]:
    """Slices along *axis* that hold a NaN, or are all zero *between* non-empty slices.

    Empty slices at either end are background (a scan ending in air, a mask that
    stops), not dropouts, and are left alone.
    """
    other = tuple(i for i in range(data.ndim) if i != axis)
    bad = _hnp.zeros(data.shape[axis], dtype=bool)
    if _hnp.issubdtype(data.dtype, _hnp.floating):
        bad |= ~_hnp.all(_hnp.isfinite(data), axis=other)
    nonzero = _hnp.any(_hnp.nan_to_num(data) != 0, axis=other) & ~bad
    filled = _hnp.nonzero(nonzero)[0]
    if filled.size >= 2:
        interior = _hnp.zeros_like(bad)
        interior[filled[0]:filled[-1] + 1] = True
        bad |= interior & ~nonzero
    return [int(i) for i in _hnp.nonzero(bad)[0]]


def fill_missing_slices(
    image: Image,
    *,
    axis: Any = 2,
    slices: Any = None,
    order: int | str = 1,
) -> Image:
    """Rebuild missing slices of *image* from their neighbours along *axis*.

    *slices* lists the slices to rebuild (``"10,14-16"`` or indices); by default
    every slice that is all zero or holds a NaN. Each is interpolated along the
    axis between the nearest good slices on either side — linearly (``order`` 1)
    or by copying the nearer one (0); a run at the volume's edge copies the
    nearest good slice.
    """
    data = to_numpy(image.data)
    if isinstance(order, str):
        order = INTERPOLATION_ORDERS.get(order, 1)
    ax = parse_axes(axis, data.ndim, _axes_string(image))[0]
    n = data.shape[ax]
    bad = _parse_indices(slices, n) if slices not in (None, "", "auto") else missing_slices(data, ax)
    if not bad:
        return image.copy()
    good = [i for i in range(n) if i not in set(bad)]
    if not good:
        raise ValueError("Every slice along that axis is missing; nothing to interpolate from.")
    out_dtype = data.dtype if int(order) == 0 else (_hnp.float64 if data.dtype == _hnp.float64 else _hnp.float32)
    out = data.astype(out_dtype, copy=True)
    good_arr = _hnp.asarray(good)
    for k in bad:
        pos = int(_hnp.searchsorted(good_arr, k))
        lo = good_arr[pos - 1] if pos > 0 else None
        hi = good_arr[pos] if pos < good_arr.size else None
        if lo is None or hi is None or int(order) == 0:
            if lo is None:
                src = hi
            elif hi is None:
                src = lo
            else:
                src = lo if (k - lo) <= (hi - k) else hi
            plane = _plane(data, ax, int(src)).astype(out_dtype)
        else:
            t = (k - lo) / float(hi - lo)
            a = _plane(data, ax, int(lo)).astype(_hnp.float64)
            b = _plane(data, ax, int(hi)).astype(_hnp.float64)
            plane = ((1.0 - t) * a + t * b).astype(out_dtype)
        index = [slice(None)] * data.ndim
        index[ax] = k
        out[tuple(index)] = plane
    return image.with_data(out)


__all__ = [
    "INTERPOLATION_ORDERS",
    "LABEL_METHODS",
    "REDUCE_METHODS",
    "RESAMPLE_MODES",
    "SLICE_METHODS",
    "annotated_slices",
    "axis_spacing",
    "block_reduce_axes",
    "fill_missing_slices",
    "interpolate_mask_slices",
    "missing_slices",
    "parse_axes",
    "resample_axes",
    "resample_index_map",
    "signed_distance",
    "sparsest_axis",
    "target_shape",
]
