"""Binary mask logical operators, and keeping / removing an image's voxels by a mask.

Everything runs on the active backend (CuPy when the GPU is on): inputs are
coerced with :func:`~nvitk.core.array.as_backend_array`, results come back as
backend arrays (wrapped as an :class:`~nvitk.types.Image` when one went in).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())

#: ``fill`` keywords :func:`apply_mask_to_image` understands besides a number.
FILL_MIN = "min"
FILL_NAN = "nan"


# ──────────────────────────────────────────────────────────────────────────────
# Foregrounds
# ──────────────────────────────────────────────────────────────────────────────


def _data(obj: Image | Any) -> Any:
    """Backend array of an :class:`Image` or an array."""
    return as_backend_array(obj.data if isinstance(obj, Image) else obj)


def _as_bool(mask: Image | Any) -> Any:
    """Backend boolean foreground (``> 0``) from an :class:`Image` or array."""
    return _data(mask) > 0


def _foreground_mask(
    mask: Image | Any,
    *,
    label_ids: Sequence[int] | None = None,
) -> Any:
    """Boolean foreground from a mask / label map.

    Empty *label_ids* → any nonzero voxel. Otherwise only listed label ids.
    """
    arr = _data(mask)
    if label_ids:
        ids = as_backend_array([int(x) for x in label_ids])
        if arr.dtype.kind == "f":
            arr = np.rint(arr)
        return np.isin(arr.astype(np.int64, copy=False), ids)
    return arr != 0


def _wrap_like(original: Image | Any, data: Any) -> Image | Any:
    """Re-wrap *data* as an :class:`Image` when *original* was one; else return *data*."""
    if isinstance(original, Image):
        return original.with_data(data)
    return data


def _check_same_shape(a: Any, b: Any) -> None:
    """Raise ``ValueError`` unless two masks share the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"Mask shapes must match; got {a.shape} vs {b.shape}.")


# ──────────────────────────────────────────────────────────────────────────────
# Set operations
# ──────────────────────────────────────────────────────────────────────────────


def mask_union(mask_a: Image | Any, mask_b: Image | Any) -> Image | Any:
    """Voxels where either mask is foreground (OR)."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    return _wrap_like(mask_a, (a | b).astype(np.uint8))


def mask_intersection(mask_a: Image | Any, mask_b: Image | Any) -> Image | Any:
    """Voxels where both masks are foreground (AND)."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    return _wrap_like(mask_a, (a & b).astype(np.uint8))


def mask_subtract(
    mask_a: Image | Any,
    mask_b: Image | Any,
    *,
    keep_overlap: bool = False,
) -> Image | Any:
    """Foreground in *mask_a* not in *mask_b* (A \\ B), or overlap only if *keep_overlap*."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    out = (a & b) if keep_overlap else (a & ~b)
    return _wrap_like(mask_a, out.astype(np.uint8))


def mask_xor(mask_a: Image | Any, mask_b: Image | Any) -> Image | Any:
    """Symmetric difference (A ⊕ B)."""
    a, b = _as_bool(mask_a), _as_bool(mask_b)
    _check_same_shape(a, b)
    return _wrap_like(mask_a, (a ^ b).astype(np.uint8))


def mask_complement(
    mask: Image | Any,
    within: Image | Any | None = None,
) -> Image | Any:
    """Logical NOT of *mask*; optional *within* ROI limits the complement region."""
    a = _as_bool(mask)
    if within is None:
        return _wrap_like(mask, (~a).astype(np.uint8))
    w = _as_bool(within)
    _check_same_shape(a, w)
    return _wrap_like(mask, (w & ~a).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────────────
# Keeping / removing image regions
# ──────────────────────────────────────────────────────────────────────────────


def mask_region(
    mask: Image | Any,
    *,
    label_ids: Sequence[int] | None = None,
    margin_mm: float = 0.0,
    spacing: Sequence[float] | None = None,
) -> Any:
    """The region a mask marks: its *label_ids* (any nonzero when none), grown by a
    positive *margin_mm* or shrunk by a negative one.

    The margin is a Euclidean distance in millimetres with the voxel *spacing*
    (one per mask axis; voxels when ``None``), so it is round in the patient, not
    in the voxel grid.
    """
    region = _foreground_mask(mask, label_ids=label_ids)
    margin = float(margin_mm or 0.0)
    if margin == 0.0 or not bool(region.any()):
        return region
    sampling = (
        tuple(abs(float(s)) or 1.0 for s in list(spacing)[: region.ndim])
        if spacing is not None and len(spacing) >= region.ndim
        else None
    )
    if margin > 0:
        # Distance of every background voxel to the region: within the margin joins it.
        return ndi.distance_transform_edt(~region, sampling=sampling) <= margin
    if bool(region.all()):
        # No background to measure from: nothing borders the region.
        return region
    # Depth of every region voxel: shallower than the margin leaves.
    return ndi.distance_transform_edt(region, sampling=sampling) > -margin


def _fill_scalar(vol: Any, fill_value: float | str) -> float:
    """The number written into removed voxels for *fill_value* (``"min"``, ``"nan"`` or a number)."""
    if isinstance(fill_value, str):
        key = fill_value.strip().lower()
        if key == FILL_NAN:
            return float("nan")
        if key == FILL_MIN:
            finite = vol[np.isfinite(vol)] if vol.dtype.kind == "f" else vol.ravel()
            return float(finite.min()) if int(finite.size) else 0.0
        return float(key)
    return float(fill_value)


def _output_dtype(dtype: Any, fill: float) -> Any:
    """*dtype* when *fill* is a value it can hold, else a float type that can."""
    if fill != fill:  # NaN
        return np.result_type(dtype, np.float32)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        if float(fill).is_integer() and info.min <= fill <= info.max:
            return dtype
        return np.result_type(dtype, np.float32)
    if dtype.kind == "b":
        return dtype if fill in (0.0, 1.0) else np.float32
    return dtype


def apply_mask_to_image(
    image: Image | Any,
    mask: Image | Any,
    *,
    mode: str = "keep_inside",
    fill_value: float | str = 0.0,
    label_ids: Sequence[int] | None = None,
    margin_mm: float = 0.0,
    spacing: Sequence[float] | None = None,
    time_axis: int | None = None,
) -> Image | Any:
    """Keep an image's voxels inside a mask's labels, or remove them.

    Parameters
    ----------
    image
        Intensity volume; 3D, or 3D+t (a 3D *mask* then applies to every frame).
    mask
        Binary mask or label map on the same voxel grid as *image* (its spatial
        part, for a 3D+t image).
    mode
        ``keep_inside`` — keep the voxels in the region, fill the rest.
        ``keep_outside`` (``remove_inside``) — remove the region: fill it, keep the rest.
    fill_value
        What removed voxels become: a number, ``"min"`` (the image's minimum —
        air on a CT) or ``"nan"`` (makes the result floating point).
    label_ids
        Labels of *mask* that make the region; ``None`` / empty → every nonzero label.
    margin_mm
        Grow (> 0) or shrink (< 0) the region by this distance first
        (:func:`mask_region`), with the image's *spacing*.
    spacing
        Voxel size along the mask's axes, for *margin_mm* (voxels when ``None``).
    time_axis
        The time axis of a 3D+t *image* whose *mask* is 3D (default: the image's
        ``T`` axis, else the last).

    Returns
    -------
    Image or array
        The image with the removed voxels filled, its dtype kept when the fill
        value fits in it.
    """
    vol = _data(image)
    region = mask_region(mask, label_ids=label_ids, margin_mm=margin_mm, spacing=spacing)
    if region.shape != vol.shape:
        # A 3D mask over a 3D+t image: the same region in every frame — and only that:
        # a mask matching some other three axes is a misalignment, not a broadcast.
        expanded = False
        if vol.ndim == region.ndim + 1:
            if time_axis is None and isinstance(image, Image):
                axes = str(image.axes or "").upper()
                time_axis = axes.index("T") if "T" in axes and len(axes) == vol.ndim else None
            t_ax = vol.ndim - 1 if time_axis is None else int(time_axis)
            if tuple(s for i, s in enumerate(vol.shape) if i != t_ax) == tuple(region.shape):
                region = np.expand_dims(region, t_ax)
                expanded = True
        if not expanded:
            raise ValueError(
                f"Mask shape {tuple(region.shape)} must match image shape {tuple(vol.shape)}"
                + (" without its time axis." if vol.ndim == region.ndim + 1 else ".")
            )
    mode_key = str(mode or "keep_inside").strip().lower().replace("-", "_")
    if mode_key in ("keep_inside", "inside", "in", "keep"):
        keep = region
    elif mode_key in ("keep_outside", "outside", "out", "remove_inside", "remove"):
        keep = ~region
    else:
        raise ValueError(
            f"Unknown mask apply mode {mode!r}; use 'keep_inside' or 'keep_outside'."
        )
    fill = _fill_scalar(vol, fill_value)
    dtype = _output_dtype(vol.dtype, fill)
    # A Python scalar of the output's kind: it broadcasts without promoting the dtype.
    scalar = int(fill) if dtype.kind in "iub" else float(fill)
    out = np.where(keep, vol.astype(dtype, copy=False), scalar)
    return _wrap_like(image, out.astype(dtype, copy=False))


def region_bounds(region: Any, *, pad: int = 0) -> tuple[list[int], list[int]] | None:
    """``(lo, hi)`` voxel bounds of a region's foreground, grown by *pad* (``None`` when empty)."""
    region = as_backend_array(region)
    if not bool(region.any()):
        return None
    lo, hi = [], []
    for axis in range(region.ndim):
        other = tuple(a for a in range(region.ndim) if a != axis)
        hit = to_numpy(np.nonzero(np.any(region, axis=other) if other else region)[0])
        lo.append(max(int(hit.min()) - int(pad), 0))
        hi.append(min(int(hit.max()) + 1 + int(pad), int(region.shape[axis])))
    return lo, hi


__all__ = [
    "FILL_MIN",
    "FILL_NAN",
    "apply_mask_to_image",
    "mask_complement",
    "mask_intersection",
    "mask_region",
    "mask_subtract",
    "mask_union",
    "mask_xor",
    "region_bounds",
]
