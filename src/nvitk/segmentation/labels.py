"""
Label-map primitives on :class:`~nvitk.types.Image` label volumes.

All functions are backend-aware (NumPy or CuPy), accept either an :class:`Image`
or a raw array, and return a new :class:`Image` when the input was an
:class:`Image` (never mutate the caller's data).
"""

from __future__ import annotations

from typing import Any, Iterable

from nvitk.core import as_backend_array
from nvitk.core.array import to_numpy
from nvitk.core.backend import get_current_backend, setup
from nvitk.types import Image

setup(globals())


def _as_array(img: Image | Any) -> Any:
    if isinstance(img, Image):
        return as_backend_array(img.data)
    return as_backend_array(img)


def _wrap_like(original: Image | Any, data: Any) -> Image | Any:
    if isinstance(original, Image):
        return original.with_data(data)
    return data


def get_label(
    label_img: Image | Any,
    label_id: int,
    *,
    crop: bool = False,
    margin: int = 0,
    missing: str = "raise",
) -> Image | Any:
    """
    Extract a binary mask for *label_id* from *label_img*.

    Parameters
    ----------
    label_img
        Label volume (integer-valued).
    label_id
        Label value to extract.
    crop
        If True, crop the output to the label's tight bounding box (plus *margin*).
    margin
        Voxel padding around the bounding box when ``crop=True``.
    missing
        ``'raise'`` (default) raises :class:`ValueError` if the label is absent;
        ``'empty'`` returns an empty (all-zero) mask of matching shape.

    Returns
    -------
    Image | ndarray
        uint8 binary mask. Matches the input type (Image in, Image out).
    """
    arr = _as_array(label_img)
    mask = (arr == label_id)

    if not bool(mask.any()):
        if missing == "raise":
            raise ValueError(f"Label {label_id} not found in image.")
        if missing != "empty":
            raise ValueError(f"Unknown missing={missing!r}; expected 'raise' or 'empty'.")

    if crop:
        coords = np.where(mask)
        if len(coords[0]) == 0:
            out = mask.astype(np.uint8)
        else:
            low = [int(np.min(c)) for c in coords]
            high = [int(np.max(c)) + 1 for c in coords]
            shape = arr.shape
            for i in range(len(low)):
                low[i] = max(0, low[i] - margin)
                high[i] = min(shape[i], high[i] + margin)
            slicer = tuple(slice(low[i], high[i]) for i in range(len(low)))
            out = mask[slicer].astype(np.uint8)
    else:
        out = mask.astype(np.uint8)

    return _wrap_like(label_img, out)


def combine_labels(
    label_img: Image | Any,
    ids: Iterable[int],
    *,
    new_id: int = 1,
) -> Image | Any:
    """
    Return a uint8 mask where any voxel ∈ *ids* is set to *new_id*, else 0.
    """
    arr = _as_array(label_img)
    ids_list = [int(v) for v in ids]
    if len(ids_list) == 0:
        raise ValueError("ids must be a non-empty iterable of integers.")
    if new_id in ids_list:
        raise ValueError(f"new_id={new_id} must not be in ids={ids_list}.")

    out = np.zeros_like(arr, dtype=np.uint8)
    for lbl in ids_list:
        out = np.where(arr == lbl, np.uint8(new_id), out)
    return _wrap_like(label_img, out)


def remove_labels(
    label_img: Image | Any,
    ids: int | Iterable[int],
    *,
    fill: int = 0,
) -> Image | Any:
    """
    Return a new label image where labels in *ids* are replaced by *fill*.

    Unlike the BioImaging legacy implementation, this function never mutates the
    caller's data.
    """
    arr = _as_array(label_img)
    ids_list = [int(ids)] if isinstance(ids, int) else [int(v) for v in ids]
    if fill in ids_list:
        raise ValueError(f"fill={fill} must not be in ids={ids_list}.")

    out = arr.copy()
    for lbl in ids_list:
        out = np.where(out == lbl, np.uint8(fill), out)
    return _wrap_like(label_img, out.astype(np.uint8))


def append_labels(
    mask_source: Image | Any,
    mask_target: Image | Any,
    *,
    remap_collisions: bool = True,
) -> Image | Any:
    """
    Overlay *mask_source*'s non-zero labels onto *mask_target*.

    When *remap_collisions* is True (default) any source label that already
    exists in the target is remapped to a new free integer (starting from
    ``max(target, source) + 1``). Labels outside source keep their original IDs.
    """
    src = _as_array(mask_source)
    tgt = _as_array(mask_target)

    merged = tgt.copy()
    src_labels = [int(v) for v in np.unique(src) if int(v) > 0]
    tgt_labels = {int(v) for v in np.unique(tgt) if int(v) > 0}
    max_id = int(max(int(np.max(tgt)) if tgt.size else 0, int(np.max(src)) if src.size else 0)) + 1

    for lbl in src_labels:
        if lbl not in tgt_labels:
            target_id = lbl
        elif remap_collisions:
            target_id = max_id
            max_id += 1
        else:
            continue
        merged = np.where(src == lbl, np.uint8(target_id), merged)

    out = merged.astype(np.uint8)
    return _wrap_like(mask_target, out)


def biggest_cc(mask: Image | Any, *, structure: Any = None) -> Image | Any:
    """Return the largest connected component of a binary *mask* as uint8."""
    arr = _as_array(mask)
    arr = as_backend_array(arr)
    labeled, _num = ndi.label(arr, structure=structure)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    if int(sizes.sum()) == 0:
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        winner = int(as_backend_array(sizes).argmax())
        out = (labeled == winner).astype(np.uint8)
    return _wrap_like(mask, out)


def percentile_cc(
    mask: Image | Any,
    percentile: float = 0.9,
    *,
    structure: Any = None,
) -> Image | Any:
    """
    Keep connected components whose size is above the *percentile* of CC sizes.

    *percentile* is in ``[0, 1]``.
    """
    arr = _as_array(mask)
    labeled, _num = ndi.label(arr, structure=structure)
    sizes = np.bincount(labeled.ravel())
    sizes = sizes[1:]
    if sizes.size == 0:
        out = np.zeros_like(arr, dtype=np.uint8)
        return _wrap_like(mask, out)

    thr = np.percentile(sizes, float(percentile) * 100)
    keep = np.where(sizes >= thr)[0] + 1
    out = np.isin(labeled, keep).astype(np.uint8)
    return _wrap_like(mask, out)


def adjust_masks(
    mask1: Image | Any,
    mask2: Image | Any,
    *,
    axis: int = 2,
) -> tuple[Image | Any, Image | Any]:
    """
    Keep only slices along *axis* (default Z) where both masks have any label;
    zero out the rest in both.
    """
    a = _as_array(mask1)
    b = _as_array(mask2)

    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("adjust_masks expects 3D masks.")

    reduce_axes = tuple(i for i in range(3) if i != axis)
    # ``any`` runs on the active backend (NumPy/CuPy).
    slice_bool_a = a.any(axis=reduce_axes)
    slice_bool_b = b.any(axis=reduce_axes)

    # The per-slice iteration below is cheap (axis_len ints); copy to host so
    # Python-level control flow has plain NumPy booleans.
    idx_a = to_numpy(slice_bool_a)
    idx_b = to_numpy(slice_bool_b)
    common_mask = idx_a & idx_b

    if not common_mask.any():
        return (
            _wrap_like(mask1, np.zeros_like(a, dtype=a.dtype)),
            _wrap_like(mask2, np.zeros_like(b, dtype=b.dtype)),
        )

    out_a = a.copy()
    out_b = b.copy()
    axis_len = a.shape[axis]
    for z in range(axis_len):
        if not bool(common_mask[z]):
            slicer = [slice(None)] * 3
            slicer[axis] = z
            out_a[tuple(slicer)] = 0
            out_b[tuple(slicer)] = 0

    return _wrap_like(mask1, out_a), _wrap_like(mask2, out_b)


__all__ = [
    "get_label",
    "combine_labels",
    "remove_labels",
    "append_labels",
    "biggest_cc",
    "percentile_cc",
    "adjust_masks",
]
