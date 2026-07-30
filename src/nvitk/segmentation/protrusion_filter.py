"""Curvature / protrusion filter for vessel masks (wart removal).

Removes small, high-curvature surface bumps that stick out from a morphological
core, while protecting seed / pre-expansion voxels so real distal branches are
kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.logger import Logger

log = Logger()

# Defaults tuned for ~0.5–1 mm 4D-flow voxels: small surface warts, not M2/P2 trunks.
PP_DISTAL_SMOOTH_SIGMA_DEFAULT: float = 1.0
PP_DISTAL_CURVATURE_PERCENTILE_DEFAULT: float = 80.0
PP_DISTAL_MAX_CC_VOXELS_DEFAULT: int = 48
PP_DISTAL_OPEN_RADIUS_DEFAULT: int = 1


def _footprint_ball(radius: int) -> np.ndarray:
    """Approximate ball structuring element of the given voxel radius."""
    r = max(0, int(radius))
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    try:
        from skimage.morphology import ball

        return ball(r)
    except Exception:  
        size = 2 * r + 1
        zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        return (zz * zz + yy * yy + xx * xx) <= (r * r)


def filter_mask_protrusions(
    mask: np.ndarray,
    *,
    protect: np.ndarray | None = None,
    smooth_sigma: float = PP_DISTAL_SMOOTH_SIGMA_DEFAULT,
    curvature_percentile: float = PP_DISTAL_CURVATURE_PERCENTILE_DEFAULT,
    max_cc_voxels: int = PP_DISTAL_MAX_CC_VOXELS_DEFAULT,
    open_radius: int = PP_DISTAL_OPEN_RADIUS_DEFAULT,
    surface_dilate: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove small wart-like protrusions from a binary vessel mask.

    Candidates are surface voxels that (a) lie outside a light morphological
    opening of the mask, and/or (b) have high mean-curvature proxy (Laplace of a
    smoothed signed-distance field). Connected components of those candidates
    that are small (``<= max_cc_voxels``) and do not touch *protect* are cleared.

    Parameters
    ----------
    mask
        Binary (or label-ROI) vessel mask.
    protect
        Voxels that must never be removed (e.g. pre-distal-expansion seeds).
    smooth_sigma
        Gaussian sigma (voxels) for SDF smoothing before Laplace.
    curvature_percentile
        Surface voxels at/above this Laplace percentile are curvature candidates.
    max_cc_voxels
        Only protrusion CCs at or below this size are removed.
    open_radius
        Morphological opening radius used to define the tubular core.
    surface_dilate
        Dilate candidate seeds by this many voxels within the mask before CC.
    """
    m = to_numpy(mask).astype(bool, copy=True)
    info: dict[str, Any] = {
        "n_before": int(m.sum()),
        "n_removed": 0,
        "n_cc_removed": 0,
        "max_cc_voxels": int(max_cc_voxels),
        "curvature_percentile": float(curvature_percentile),
        "open_radius": int(open_radius),
        "smooth_sigma": float(smooth_sigma),
    }
    if int(m.sum()) < 8:
        info["n_after"] = int(m.sum())
        return as_backend_array(m), info

    protect_b = np.zeros(m.shape, dtype=bool)
    if protect is not None:
        protect_b = to_numpy(protect).astype(bool, copy=False)
        if protect_b.shape != m.shape:
            raise ValueError(
                f"protect shape {protect_b.shape} must match mask shape {m.shape}"
            )
        protect_b &= m

    fp = _footprint_ball(open_radius)
    core = ndi.binary_opening(m, structure=fp) if int(open_radius) > 0 else m.copy()
    if not np.any(core):
        # Opening wiped the mask — do not remove anything.
        info["n_after"] = int(m.sum())
        info["skipped"] = "empty_core"
        return as_backend_array(m), info

    stick = m & ~core

    # Signed distance: negative inside, positive outside.
    phi = ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)
    sigma = max(0.0, float(smooth_sigma))
    phi_s = ndi.gaussian_filter(phi.astype(np.float64), sigma=sigma) if sigma > 0 else phi
    # Mean-curvature proxy of the zero level-set (Laplace of SDF).
    curv = ndi.laplace(phi_s)

    eroded = ndi.binary_erosion(m, structure=np.ones((3, 3, 3), dtype=bool))
    surface = m & ~eroded
    candidates = stick.copy()
    thr = None
    if np.any(surface):
        h_surf = curv[surface]
        thr = float(np.percentile(h_surf, float(curvature_percentile)))
        # Convex outward bumps of a solid (phi < 0 inside) tend to have positive Laplace.
        candidates |= surface & (curv >= thr)
    info["curvature_threshold"] = thr

    if int(surface_dilate) > 0 and np.any(candidates):
        dil_fp = _footprint_ball(int(surface_dilate))
        candidates = ndi.binary_dilation(candidates, structure=dil_fp) & m

    # Never treat protected voxels as removable candidates.
    candidates &= ~protect_b
    if not np.any(candidates):
        info["n_after"] = int(m.sum())
        info["skipped"] = "no_candidates"
        return as_backend_array(m), info

    labeled, n_lab = ndi.label(candidates, structure=ndi.generate_binary_structure(3, 3))
    max_sz = max(1, int(max_cc_voxels))
    removed = np.zeros(m.shape, dtype=bool)
    n_cc = 0
    for lab in range(1, int(n_lab) + 1):
        cc = labeled == lab
        n_vox = int(cc.sum())
        if n_vox <= 0 or n_vox > max_sz:
            continue
        if np.any(cc & protect_b):
            continue
        removed |= cc
        n_cc += 1

    if np.any(removed):
        m[removed] = False
    m |= protect_b

    info["n_removed"] = int(removed.sum())
    info["n_cc_removed"] = int(n_cc)
    info["n_after"] = int(m.sum())
    info["n_candidates"] = int(candidates.sum())
    return as_backend_array(m), info


def filter_label_protrusions_in_seg(
    seg: np.ndarray,
    label_ids: list[int] | tuple[int, ...] | set[int],
    *,
    protect_by_label: dict[int, np.ndarray] | None = None,
    smooth_sigma: float = PP_DISTAL_SMOOTH_SIGMA_DEFAULT,
    curvature_percentile: float = PP_DISTAL_CURVATURE_PERCENTILE_DEFAULT,
    max_cc_voxels: int = PP_DISTAL_MAX_CC_VOXELS_DEFAULT,
    open_radius: int = PP_DISTAL_OPEN_RADIUS_DEFAULT,
) -> dict[str, Any]:
    """In-place protrusion filter for selected label ids in a multilabel *seg*."""
    seg_np = as_backend_array(seg)
    summary: dict[str, Any] = {"labels": {}, "n_removed_total": 0}
    for lid in sorted(int(x) for x in label_ids):
        full = seg_np == int(lid)
        if not np.any(full):
            continue
        protect = None
        if protect_by_label is not None and int(lid) in protect_by_label:
            protect = protect_by_label[int(lid)]
        cleaned, meta = filter_mask_protrusions(
            full,
            protect=protect,
            smooth_sigma=smooth_sigma,
            curvature_percentile=curvature_percentile,
            max_cc_voxels=max_cc_voxels,
            open_radius=open_radius,
        )
        remove = full & ~as_backend_array(cleaned).astype(bool)
        n_rem = int(np.count_nonzero(remove))
        if n_rem > 0:
            seg_np[remove] = 0
        summary["labels"][str(int(lid))] = meta
        summary["n_removed_total"] += n_rem
        if n_rem > 0:
            log.info(
                f"pp-distal label={int(lid)}: removed {n_rem} wart voxel(s) "
                f"in {meta.get('n_cc_removed', 0)} CC(s) "
                f"({meta.get('n_before')} → {meta.get('n_after')})"
            )
    return summary


__all__ = [
    "PP_DISTAL_CURVATURE_PERCENTILE_DEFAULT",
    "PP_DISTAL_MAX_CC_VOXELS_DEFAULT",
    "PP_DISTAL_OPEN_RADIUS_DEFAULT",
    "PP_DISTAL_SMOOTH_SIGMA_DEFAULT",
    "filter_label_protrusions_in_seg",
    "filter_mask_protrusions",
]
