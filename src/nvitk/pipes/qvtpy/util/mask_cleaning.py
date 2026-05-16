"""Morphological cleaning for eICAB multilabel and CD binary masks."""

from __future__ import annotations

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology import open as morph_open
from nvitk.morphology.components import label_connected, remove_small_components_by_fraction

setup(globals())


# ---------------------------------------------------------------------------
# eICAB multilabel island removal (+ optional bridge opening)
# ---------------------------------------------------------------------------


def clean_multilabel_islands(
    labels: np.ndarray,
    *,
    min_fraction: float = 0.005,
    connectivity: int = 1,
    bridge_open_radius: int = 0,
) -> np.ndarray:
    """Remove per-label connected components smaller than *min_fraction* of label foreground.

    Parameters
    ----------
    bridge_open_radius
        If > 0, apply binary opening with a ball footprint of this radius (voxels)
        on each label hull before CC filtering (reduces thin communicating bridges).
    """
    arr = as_backend_array(labels).astype(np.int32, copy=False)
    out = np.zeros_like(arr)
    for lid in sorted(int(v) for v in np.unique(arr) if int(v) != 0):
        roi = arr == lid
        if bridge_open_radius > 0:
            opened = morph_open(roi.astype(np.uint8), footprint=int(bridge_open_radius), connectivity=1)
            roi = as_backend_array(opened).astype(bool, copy=False)
        n_fg = int(np.count_nonzero(roi))
        if n_fg == 0:
            continue
        min_size = max(1, int(round(float(min_fraction) * n_fg)))
        labeled, _ = label_connected(roi, connectivity=int(connectivity))
        labeled_np = as_backend_array(labeled)
        for comp_id in range(1, int(labeled_np.max()) + 1):
            comp = labeled_np == comp_id
            if int(np.count_nonzero(comp)) >= min_size:
                out[comp] = lid
    return as_backend_array(out).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Binary / venous-slab area opening
# ---------------------------------------------------------------------------


def clean_binary_mask(
    mask: np.ndarray,
    *,
    min_fraction: float = 0.005,
    connectivity: int = 1,
) -> np.ndarray:
    """Area-opening on a boolean 3D mask."""
    cleaned = remove_small_components_by_fraction(
        mask,
        min_fraction=min_fraction,
        connectivity=int(connectivity),
    )
    return as_backend_array(cleaned).astype(bool, copy=False)


def clean_venous_slab_mask(
    venous_mask: np.ndarray,
    *,
    min_fraction: float = 0.005,
    connectivity: int = 1,
) -> np.ndarray:
    """Second-pass area opening on venous-slab-restricted foreground."""
    return clean_binary_mask(
        venous_mask,
        min_fraction=min_fraction,
        connectivity=connectivity,
    )


def keep_largest_component_per_label(labels: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component for each positive label id."""
    arr = as_backend_array(labels).astype(np.int32, copy=False)
    out = np.zeros_like(arr)
    for lid in sorted(int(v) for v in np.unique(arr) if int(v) != 0):
        roi = arr == lid
        n_fg = int(np.count_nonzero(roi))
        if n_fg == 0:
            continue
        labeled, num = label_connected(roi, connectivity=1)
        labeled_np = as_backend_array(labeled)
        if int(num) <= 1:
            out[roi] = lid
            continue
        counts = np.bincount(labeled_np.ravel())
        if counts.size <= 1:
            out[roi] = lid
            continue
        largest_comp = int(1 + np.argmax(counts[1:]))
        out[labeled_np == largest_comp] = lid
    return as_backend_array(out.astype(np.int32, copy=False))


__all__ = [
    "clean_binary_mask",
    "clean_multilabel_islands",
    "clean_venous_slab_mask",
    "keep_largest_component_per_label",
]
