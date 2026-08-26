"""
Topology-aware post-processing for multi-class vessel segmentations.

Description
-----------
A network trained with a voxelwise loss produces plausible-looking masks that are often
fragmented: a class appears as its true branch plus a handful of stray islands elsewhere in the
volume. Those islands cost almost nothing in Dice but wreck the connected-component and
centerline metrics that vessel benchmarks score, and they are trivially removable — a vessel
class is one connected structure by construction.

Two cheap, order-dependent operations do most of the work:

1. :func:`remove_small_islands` — drop components below an absolute or relative size.
2. :func:`keep_largest_per_class` — reduce each class to its *n* largest components.

Both operate per class on a multi-class label map and never move a voxel between classes; a
voxel is either kept or returned to background.

Array / axis conventions
------------------------
Shape-agnostic 3D label maps of integer dtype. Physical sizes are given in mm³ and converted
with an explicit ``spacing``; voxel counts are accepted too but must be named as such.

I/O and arrays: backend ``np`` after ``setup(globals())``; accepts and returns
:class:`~nvitk.types.Image` or bare arrays, preserving geometry.
"""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.morphology.components import keep_largest_components, remove_small_components
from nvitk.types import Image

setup(globals())

log = Logger()

#: Connectivity used throughout: vessels run diagonally through the voxel grid, so 6-connectivity
#: would split a single smoothly-curving vessel into several components.
DEFAULT_CONNECTIVITY: int = 3


def _unwrap(labelmap: Any) -> tuple[Any, Image | None]:
    """Split *labelmap* into a backend array and the ``Image`` to rewrap with (or ``None``)."""
    if isinstance(labelmap, Image):
        return as_backend_array(labelmap.data), labelmap
    return as_backend_array(labelmap), None


def _present_labels(data: Any, labels: Sequence[int] | None) -> list[int]:
    """Foreground label values to process."""
    if labels is not None:
        return [int(v) for v in labels]
    return [int(v) for v in to_numpy(np.unique(data)) if int(v) != 0]


def voxel_volume_mm3(spacing: Sequence[float] | None) -> float:
    """Volume of one voxel in mm³; 1.0 when *spacing* is unknown (i.e. counts are voxels)."""
    if spacing is None:
        return 1.0
    volume = 1.0
    for value in spacing:
        volume *= float(value)
    return volume


def remove_small_islands(
    labelmap: Any,
    *,
    labels: Sequence[int] | None = None,
    min_voxels: int | None = None,
    min_volume_mm3: float | None = None,
    spacing: Sequence[float] | None = None,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> Any:
    """Drop, per class, connected components smaller than a threshold.

    Parameters
    ----------
    min_voxels, min_volume_mm3
        Give one. ``min_volume_mm3`` is preferred when *spacing* is known — a fixed voxel count
        means different things on 0.3 mm and 0.6 mm data, and this cohort spans both.

    Returns
    -------
    Image or array
        Same type and geometry as the input.
    """
    if (min_voxels is None) == (min_volume_mm3 is None):
        raise ValueError("Give exactly one of min_voxels or min_volume_mm3.")

    data, source = _unwrap(labelmap)
    threshold = (
        int(min_voxels)
        if min_voxels is not None
        else max(1, int(round(float(min_volume_mm3) / voxel_volume_mm3(spacing))))
    )

    out = data.copy()
    removed = 0
    for value in _present_labels(data, labels):
        mask = data == value
        kept = remove_small_components(mask, min_size=threshold, connectivity=connectivity)
        kept = as_backend_array(kept)
        dropped = int(to_numpy(np.logical_and(mask, ~kept).sum()))
        if dropped:
            out[np.logical_and(mask, ~kept)] = 0
            removed += dropped
    if removed:
        log.info(
            "Removed %d voxel(s) in components below %d voxel(s) (%.2f mm3).",
            removed,
            threshold,
            threshold * voxel_volume_mm3(spacing),
        )
    return source.with_data(out) if source is not None else out


def keep_largest_per_class(
    labelmap: Any,
    *,
    labels: Sequence[int] | None = None,
    n: int = 1,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> Any:
    """Reduce each class to its *n* largest connected components.

    ``n=1`` encodes "each vessel class is one structure", which holds for most TopBrain classes.
    It is deliberately **not** the default in :func:`postprocess_labelmap`: a few classes are
    genuinely multi-component in a given field of view (a vessel leaving and re-entering the
    volume), and silently deleting the second piece would trade a fragmentation error for a
    missing-structure error, which scores worse.
    """
    data, source = _unwrap(labelmap)
    out = data.copy()
    for value in _present_labels(data, labels):
        mask = data == value
        kept = as_backend_array(keep_largest_components(mask, n=int(n), connectivity=connectivity))
        out[np.logical_and(mask, ~kept)] = 0
    return source.with_data(out) if source is not None else out


def postprocess_labelmap(
    labelmap: Any,
    *,
    labels: Sequence[int] | None = None,
    spacing: Sequence[float] | None = None,
    min_volume_mm3: float | None = 5.0,
    largest_only: bool = False,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> Any:
    """Run the standard clean-up: drop small islands, optionally keep only the largest component.

    Ordering matters — islands are removed first, so a class whose largest component is itself
    spurious noise is not preserved by :func:`keep_largest_per_class`.

    Parameters
    ----------
    min_volume_mm3
        ``None`` skips island removal. The 5 mm³ default is a few voxels at this cohort's
        resolution: small enough to keep the thinnest genuine branches, large enough to remove
        isolated speckle.
    largest_only
        Also reduce each class to a single component. Off by default — see
        :func:`keep_largest_per_class`.
    """
    result = labelmap
    if min_volume_mm3 is not None:
        result = remove_small_islands(
            result,
            labels=labels,
            min_volume_mm3=min_volume_mm3,
            spacing=spacing,
            connectivity=connectivity,
        )
    if largest_only:
        result = keep_largest_per_class(
            result, labels=labels, n=1, connectivity=connectivity
        )
    return result


__all__ = [
    "DEFAULT_CONNECTIVITY",
    "keep_largest_per_class",
    "postprocess_labelmap",
    "remove_small_islands",
    "voxel_volume_mm3",
]
