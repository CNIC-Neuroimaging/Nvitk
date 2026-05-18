"""Global complex-difference vessel masks for qvtpy stage 3.

**Inputs**

- 3D ``ComplexDifference`` volume (float), typically from stage-0 ``phase2volume``.

**Outputs**

- Boolean foreground mask after sliding threshold + area opening.
- :func:`venous_search_region` — superior Y-slab mask restricting venous geometry heuristics.
"""

from __future__ import annotations

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology.components import remove_small_components_by_fraction

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
    return as_backend_array(to_numpy(segment).astype(bool, copy=False)), float(opt_thresh)


__all__ = [
    "binary_vessel_segment_cd",
    "venous_search_region",
]
