"""Global complex-difference vessel masks for qvtpy stage 3.

**Inputs**

- 3D ``ComplexDifference`` volume (float), typically from stage-0 ``phase2volume``.

**Outputs**

- Boolean foreground mask after sliding threshold + area opening.
- :func:`venous_search_region` — superior Y-slab mask restricting venous geometry heuristics.
- :func:`arterial_exclusion_mask` — eICAB arterial voxels to subtract from venous CD ROI.
"""

from __future__ import annotations

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup, get_current_backend
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.qvtpy.labels import QVTPY_ARTERIAL_LABEL_IDS

setup(globals())


# ---------------------------------------------------------------------------
# Venous search region (superior Y-slab)
# ---------------------------------------------------------------------------


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
        art = ndi.binary_dilation(art, iterations=dilate, brute_force=get_current_backend() == "cupy")
    return as_backend_array(art.astype(bool, copy=False))


# ---------------------------------------------------------------------------
# Sliding threshold on 3D complex-difference (MATLAB slidingThreshold.m)
# ---------------------------------------------------------------------------


from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d


# ---- Public API: global vessel mask + area opening ---------------------------


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


__all__ = [
    "arterial_exclusion_mask",
    "binary_vessel_segment_cd",
    "venous_search_region",
]
