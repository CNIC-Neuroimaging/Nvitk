"""Connected-component labeling and area filtering (backend-aware via ``ndi``)."""

from __future__ import annotations

from typing import Any

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.types import Image

from ._common import _as_array, _coerce_to_current_backend, _wrap_like

setup(globals())


def _label_structure(ndim: int, connectivity: int) -> Any:
    conn = max(1, min(int(connectivity), ndim))
    return ndi.generate_binary_structure(ndim, conn)


def label_connected(
    mask: Image | Any,
    *,
    connectivity: int = 1,
) -> tuple[Any, int]:
    """Label connected components of a binary *mask*.

    Returns ``(labeled_array, num_features)`` on the active backend.
    """
    arr = _coerce_to_current_backend(_as_array(mask).astype(bool, copy=False))
    structure = _label_structure(int(arr.ndim), connectivity)
    labeled, num = ndi.label(arr, structure=structure)
    return labeled, int(num)


def remove_small_components(
    mask: Image | Any,
    *,
    min_size: int,
    connectivity: int = 1,
) -> Any:
    """Drop connected components with fewer than *min_size* voxels."""
    arr = _coerce_to_current_backend(_as_array(mask).astype(bool, copy=False))
    if int(np.count_nonzero(arr)) == 0:
        return _wrap_like(mask, arr)
    labeled, num = label_connected(arr, connectivity=connectivity)
    if num == 0:
        out = np.zeros_like(arr, dtype=bool)
        return _wrap_like(mask, out)
    counts = np.bincount(labeled.ravel())
    keep = np.array(
        [i for i in range(1, int(len(counts))) if int(counts[i]) >= int(min_size)],
        dtype=labeled.dtype,
    )
    if keep.size == 0:
        out = np.zeros_like(arr, dtype=bool)
    else:
        out = np.isin(labeled, keep)
    return _wrap_like(mask, as_backend_array(out.astype(bool, copy=False)))


def remove_small_components_by_fraction(
    mask: Image | Any,
    *,
    min_fraction: float,
    connectivity: int = 1,
) -> Any:
    """Area-opening: remove components smaller than *min_fraction* of foreground."""
    arr = _coerce_to_current_backend(_as_array(mask).astype(bool, copy=False))
    n_fg = int(np.count_nonzero(arr))
    if n_fg == 0:
        return _wrap_like(mask, arr)
    min_size = max(1, int(round(float(min_fraction) * n_fg)))
    return remove_small_components(arr, min_size=min_size, connectivity=connectivity)


def keep_component_closest_to_center(
    labeled: Any,
    *,
    center: tuple[float, float] | None = None,
) -> Any:
    """Binary mask of the labeled component whose centroid is nearest *center*."""
    lab = _coerce_to_current_backend(labeled)
    if int(lab.max()) == 0:
        return as_backend_array(np.zeros(lab.shape, dtype=bool))
    cy, cx = center if center is not None else ((lab.shape[0] - 1) / 2.0, (lab.shape[1] - 1) / 2.0)
    best_id = 0
    best_d = np.inf
    for comp_id in range(1, int(lab.max()) + 1):
        comp = lab == comp_id
        yi, xi = np.nonzero(as_backend_array(comp))
        if yi.size == 0:
            continue
        my, mx = float(np.mean(yi)), float(np.mean(xi))
        d = (my - cy) ** 2 + (mx - cx) ** 2
        if d < best_d:
            best_d = d
            best_id = comp_id
    return as_backend_array(lab == best_id)


__all__ = [
    "keep_component_closest_to_ºcenter",
    "label_connected",
    "remove_small_components",
    "remove_small_components_by_fraction",
]
