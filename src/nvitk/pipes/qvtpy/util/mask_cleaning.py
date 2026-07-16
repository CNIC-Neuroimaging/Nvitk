"""Morphological cleaning for eICAB multilabel and CD binary masks.

Used in stage 3 (eICAB island removal, venous slab opening) and stage 4
(:func:`keep_largest_component_per_label` after local thresholding).
"""

from __future__ import annotations

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup, get_current_backend
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


def clean_volume_seg_for_pitc(
    volume_seg: np.ndarray,
    centerlines: dict[int, np.ndarray] | None = None,
    *,
    seed_dilate: int = 2,
) -> np.ndarray:
    """Remove isolated label islands before PITC/PWV station sampling.

    1. Keep the largest 6-connected component per label (drops speckles).
    2. When *centerlines* are given, further keep only components that touch a
       dilated centerline seed for that label (drops remote islands). If the
       seed does not touch the mask, the largest-CC result is retained.
    """
    cleaned = keep_largest_component_per_label(volume_seg)
    if not centerlines:
        return cleaned

    from nvitk.morphology.components import keep_components_touching_seeds

    arr = as_backend_array(cleaned).astype(np.int32, copy=True)
    shape = arr.shape
    seed = np.zeros(shape, dtype=np.int32)
    for lid, pts in centerlines.items():
        lid = int(lid)
        pts_np = as_backend_array(pts).astype(np.float64).reshape(-1, 3)
        if pts_np.shape[0] == 0:
            continue
        for p in pts_np:
            ijk = (int(round(float(p[0]))), int(round(float(p[1]))), int(round(float(p[2]))))
            if (
                0 <= ijk[0] < shape[0]
                and 0 <= ijk[1] < shape[1]
                and 0 <= ijk[2] < shape[2]
            ):
                seed[ijk] = lid

    if int(seed_dilate) > 0 and np.any(seed):
        dilated = np.zeros_like(seed)
        for lid in sorted(int(v) for v in np.unique(seed) if int(v) != 0):
            dil = ndi.binary_dilation(
                seed == lid, iterations=int(seed_dilate),
                brute_force=get_current_backend() == "cupy"
            )
            dilated[dil] = lid
        seed = dilated

    out = arr.copy()
    for lid in sorted(int(v) for v in np.unique(arr) if int(v) != 0):
        if lid not in {int(k) for k in centerlines.keys()}:
            continue
        roi = arr == lid
        if not np.any(roi):
            continue
        seeds_l = seed == lid
        if not np.any(seeds_l):
            continue
        kept = keep_components_touching_seeds(roi, seeds_l, connectivity=1)
        kept_np = as_backend_array(kept).astype(bool, copy=False)
        if not np.any(kept_np):
            # Seed missed the mask (CL/seg mismatch); keep largest-CC result.
            continue
        out[roi] = 0
        out[kept_np] = lid
    return as_backend_array(out.astype(np.int32, copy=False))


def keep_seed_connected_per_label(
    labels: np.ndarray,
    seed_mask: np.ndarray,
    *,
    label_ids: frozenset[int] | set[int],
) -> np.ndarray:
    """Per-label cleanup: seed-connected CCs for *label_ids*, largest CC elsewhere."""
    from nvitk.morphology.components import keep_components_touching_seeds

    arr = as_backend_array(labels).astype(np.int32, copy=False)
    seeds = as_backend_array(seed_mask).astype(np.int32, copy=False)
    out = arr.copy()
    for lid in sorted(int(v) for v in np.unique(arr) if int(v) != 0):
        roi = out == lid
        if not np.any(roi):
            continue
        if int(lid) in label_ids:
            kept = keep_components_touching_seeds(roi, seeds == lid, connectivity=1)
            out[roi] = 0
            out[as_backend_array(kept).astype(bool)] = lid
        else:
            labeled, num = label_connected(roi, connectivity=1)
            labeled_np = as_backend_array(labeled)
            if int(num) <= 1:
                continue
            counts = np.bincount(labeled_np.ravel())
            largest_comp = int(1 + np.argmax(counts[1:]))
            out[roi] = 0
            out[labeled_np == largest_comp] = lid
    return as_backend_array(out.astype(np.int32, copy=False))


__all__ = [
    "clean_binary_mask",
    "clean_multilabel_islands",
    "clean_venous_slab_mask",
    "clean_volume_seg_for_pitc",
    "keep_largest_component_per_label",
    "keep_seed_connected_per_label",
]
