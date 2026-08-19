"""
Label-map primitives on :class:`~nvitk.types.Image` label volumes.

All functions are backend-aware (NumPy or CuPy), accept either an :class:`Image`
or a raw array, and return a new :class:`Image` when the input was an
:class:`Image` (never mutate the caller's data).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from nvitk.core import as_backend_array
from nvitk.core.array import to_numpy
from nvitk.core.backend import get_current_backend, setup
from nvitk.core.logger import Logger
from nvitk.types import Image

setup(globals())

log = Logger()


def _as_array(img: Image | Any) -> Any:
    """Backend array view of an :class:`Image` or raw array."""
    if isinstance(img, Image):
        return as_backend_array(img.data)
    return as_backend_array(img)


def _wrap_like(original: Image | Any, data: Any) -> Image | Any:
    """Re-wrap *data* as an :class:`Image` when *original* was one; else return *data*."""
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


def biggest_cc(
    mask: Image | Any,
    *,
    structure: Any = None,
    n: int = 1,
) -> Image | Any:
    """Return the *n* largest connected component(s) of a binary *mask* as uint8.

    ``n=1`` (default) keeps only the single largest component.
    """
    from nvitk.morphology.components import keep_largest_components

    kept = keep_largest_components(mask, n=int(n), structure=structure)
    arr = _as_array(kept).astype(np.uint8, copy=False)
    return _wrap_like(mask, arr)


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


# ---------------------------------------------------------------------------
# Per-label application
# ---------------------------------------------------------------------------
def _label_dtype(source: Any, labels: Sequence[int]) -> Any:
    """
    Smallest integer dtype that can hold every id in *labels*.

    An integer input keeps its own dtype — a mask stored as ``int32`` should not silently narrow
    to ``uint8`` just because its ids happen to be small, since callers key layers and lookup
    tables off it. Same rule as :func:`~nvitk.morphology.centerline.skeletonize_labeled`.
    """
    dtype = getattr(source, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        return dtype
    top = max((abs(int(v)) for v in labels), default=0)
    if top <= 255:
        return np.uint8
    return np.uint16 if top <= 65535 else np.int32


def apply_per_label(
    label_img: Image | Any,
    op: Any,
    *,
    label_ids: Sequence[int] | None = None,
    dtype: Any = None,
    overlap: str = "first",
) -> Image | Any:
    """
    Run a **binary** operation independently on each label and recombine, preserving the ids.

    Why
    ---
    A binary operation applied to the union of several labels answers a different question from
    the same operation applied to each of them. Connected components over ``{1, 3}`` fused into
    one mask reports one component where two labels touch, and drops the smaller label entirely
    when they do not; a dilation grows each label into its neighbour and erases the boundary
    between them. Running per label and painting the ids back keeps the parcellation the caller
    started with, which is almost always what a multi-label selection means.

    Parameters
    ----------
    label_img : Image or array
        Integer label volume. Zero is background.
    op : callable
        ``op(binary) -> binary``, where *binary* is this function's own per-label mask, wrapped as
        an :class:`~nvitk.types.Image` when *label_img* was one so the operation keeps its spacing
        and affine. Anything non-zero in the result is taken as foreground.
    label_ids : sequence of int, optional
        Labels to operate on, in the order they are applied. ``None`` uses every non-zero label.
        **Labels not named here are copied through untouched** — an operation on a selection must
        not delete the parts of the volume it was not asked about.
    dtype : optional
        Output dtype. Defaults to the input's own integer dtype, else the smallest that holds
        every id.
    overlap : {"first", "last"}
        Which label wins where two results cover the same voxel — the inputs are disjoint but the
        outputs need not be (a dilation of two adjacent labels overlaps by construction).
        ``"first"`` keeps the earlier label in *label_ids* order, ``"last"`` the later one.
        Made explicit because the alternative is an ordering-dependent result that looks like a
        bug the first time two labels touch.

    Returns
    -------
    Image or array
        Same kind as *label_img*, never a view of it.

    Examples
    --------
    >>> from nvitk.morphology.components import keep_largest_components
    >>> cleaned = apply_per_label(          # doctest: +SKIP
    ...     labels, lambda m: keep_largest_components(m, n=1), label_ids=[1, 3]
    ... )
    """
    overlap = str(overlap).strip().lower()
    if overlap not in {"first", "last"}:
        raise ValueError(f"overlap must be 'first' or 'last', not {overlap!r}.")

    arr = _as_array(label_img)
    if label_ids is None:
        ids = [int(v) for v in np.unique(arr) if int(v) != 0]
    else:
        ids = [int(v) for v in label_ids if int(v) != 0]
    if not ids:
        raise ValueError("apply_per_label needs at least one non-zero label to operate on.")

    out_dtype = dtype if dtype is not None else _label_dtype(arr, ids)
    selected = np.isin(arr, as_backend_array(ids))

    # Unselected labels pass through: the caller asked about a subset, not for the rest to vanish.
    out = np.where(selected, 0, arr).astype(out_dtype, copy=True)
    # Tracks what the operation has already claimed, so ``overlap`` is decided here rather than by
    # whichever label happened to be written last.
    claimed = np.zeros(arr.shape, dtype=bool)

    for label in ids:
        region = arr == label
        if not bool(region.any()):
            log.debug("apply_per_label: label %d has no voxels — skipped.", label)
            continue
        result = op(_wrap_like(label_img, region))
        kept = _as_array(result) != 0
        if kept.shape != arr.shape:
            raise ValueError(
                f"The operation returned shape {tuple(kept.shape)} for label {label}, but the "
                f"label volume is {tuple(arr.shape)}. Per-label application needs a shape-"
                f"preserving operation — resampling and cropping have to run on the whole volume."
            )
        if overlap == "first":
            kept = kept & ~claimed
        # Boolean assignment rather than ``np.where``: it writes in place at the output dtype and
        # works identically on NumPy and CuPy, with no host round-trip for the fill value.
        out[kept] = label
        claimed |= kept

    return _wrap_like(label_img, out)


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
    "apply_per_label",
    "biggest_cc",
    "percentile_cc",
    "adjust_masks",
]
