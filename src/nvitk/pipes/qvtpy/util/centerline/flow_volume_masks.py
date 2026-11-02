"""Global complex-difference vessel masks for qvtpy stage 3.

**Inputs**

- 3D ``ComplexDifference`` volume (float), typically from stage-0 ``phase2volume``.

**Outputs**

- Boolean foreground mask after sliding threshold + area opening.
- :func:`venous_search_region`: superior Y-slab mask restricting venous geometry heuristics.
- :func:`arterial_exclusion_mask`: eICAB arterial voxels to subtract from venous CD ROI.
- :func:`extend_venous_mask`: the dim sinus stretches the global threshold misses.
"""

from __future__ import annotations

import math

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d
from nvitk.morphology import dilate as morph_dilate
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.qvtpy.labels import QVTPY_ARTERIAL_LABEL_IDS

setup(globals())


# Venous search region (superior Y-slab)


def venous_search_region(shape: tuple[int, int, int]) -> np.ndarray:
    """Boolean slab: first ``round(ny/3)`` planes along axis 1 (shape ``(nx, ny, nz)``)."""
    _, ny, _ = shape
    third_y = max(1, int(round(ny / 3.0)))
    ven = np.zeros(shape, dtype=bool)
    ven[:, :third_y, :] = True
    return ven


def arterial_exclusion_mask(
    labels: np.ndarray,
    *,
    arterial_ids: frozenset[int] | set[int] | None = None,
    dilate_vox: int = 1,
) -> np.ndarray:
    """Boolean mask of arterial voxels (optionally dilated) to exclude from venous CD.

    eICAB arteries that enter the superior venous slab otherwise remain in the
    global CD vessel binary and can be skeletonized / labeled as sinuses.
    Prefer a whole-brain (WB) eICAB mask here: CW often omits distal territory.
    """
    arr = as_backend_array(labels).astype(np.int32, copy=False)
    ids = arterial_ids if arterial_ids is not None else QVTPY_ARTERIAL_LABEL_IDS
    art = np.zeros(arr.shape, dtype=bool)
    for lid in ids:
        art |= arr == int(lid)
    dilate = max(0, int(dilate_vox))
    if dilate > 0 and bool(np.any(art)):
        # 6-connected dilation via base tool (brute_force handled internally).
        art = as_backend_array(morph_dilate(art, iterations=dilate, connectivity=1)).astype(bool)
    return as_backend_array(art.astype(bool, copy=False))


# Public API: global vessel mask + area opening


def binary_vessel_segment_cd(
    cd: np.ndarray,
    *,
    up_thresh: float = 0.8,
    shift_hm_flag: bool = True,
    min_component_fraction: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Global vessel boolean mask: sliding threshold on CD, then area opening (face 3D)."""
    segment, opt_thresh = binary_mask_sliding_threshold_3d(
        cd,
        up_thresh=float(up_thresh),
        smf=10,
        shift_hm_flag=bool(shift_hm_flag),
        med_filt_flag=False,
    )
    segment = remove_small_components_by_fraction(
        segment,
        min_fraction=float(min_component_fraction),
        connectivity=1,
    )
    return as_backend_array((segment).astype(bool, copy=False)), float(opt_thresh)


#: The venous mask's bright sinuses keep their thickness: voxels within this many of them are only
#: added where they join a new, dim stretch (a thickened confluence hides its junctions from the
#: skeleton the sinuses are named on).
VENOUS_EXTEND_NEAR_VOX = 2
#: Dim stretches smaller than this (mm³) are not added (blobs at the confluence, not sinuses).
VENOUS_EXTEND_MIN_MM3 = 30.0


def extend_venous_mask(
    cd: np.ndarray, venous: np.ndarray, region: np.ndarray, *, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Add to *venous* (the global sliding threshold's sinus voxels) the dim sinus stretches it
    misses, such as a faint transverse sinus. The global threshold is set by the arteries' bright
    signal (0.1-0.3 x the blood signal); recomputed on the venous search *region* only (slab, brain,
    no arteries) it is ~1.5x lower, near the tissue's 99.9th percentile. Of the region's voxels
    above it, the pieces connected to *venous* are kept; of those, only what lies more than
    ``VENOUS_EXTEND_NEAR_VOX`` voxels from *venous* in pieces of at least ``VENOUS_EXTEND_MIN_MM3``,
    plus the voxels joining them to it. Returns the extended mask and a summary."""
    cd_b = as_backend_array(cd).astype(np.float64)
    venous_b = as_backend_array(venous).astype(bool)
    region_b = as_backend_array(region).astype(bool)
    _mask, thr = binary_mask_sliding_threshold_3d(np.where(region_b, cd_b, 0.0), med_filt_flag=False)
    cube = np.ones((3, 3, 3), dtype=bool)
    cc, _n = ndi.label(((cd_b > float(thr)) & region_b) | venous_b, structure=cube)
    keep = np.unique(cc[venous_b])
    grown = np.isin(cc, keep[keep > 0])
    near = ndi.binary_dilation(venous_b, structure=cube, iterations=VENOUS_EXTEND_NEAR_VOX)
    new = grown & ~near
    cc_new, _n = ndi.label(new, structure=cube)
    sizes = np.bincount(cc_new.ravel())
    sizes[0] = 0
    min_vox = math.ceil(VENOUS_EXTEND_MIN_MM3 / math.prod(float(v) for v in spacing))
    new = (sizes >= min_vox)[cc_new] & new
    link = grown & near & ~venous_b & ndi.binary_dilation(new, structure=cube, iterations=VENOUS_EXTEND_NEAR_VOX)
    out = venous_b | new | link
    return out, {"region_threshold": float(thr), "voxels_added": int(np.count_nonzero(out & ~venous_b))}


__all__ = [
    "VENOUS_EXTEND_MIN_MM3",
    "VENOUS_EXTEND_NEAR_VOX",
    "arterial_exclusion_mask",
    "binary_vessel_segment_cd",
    "extend_venous_mask",
    "venous_search_region",
]
