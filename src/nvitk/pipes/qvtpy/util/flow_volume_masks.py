"""Shared 4D-flow contrast masks (complex-difference angiogram) for qvtpy stages 3–4."""

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


def _binary_mask_sliding_threshold(
    cd: np.ndarray,
    *,
    step: float = 0.001,
    up_thresh: float = 0.8,
    smf: int = 10,
    shift_hm_flag: bool = True,
    med_filt_flag: bool = True,
) -> tuple[np.ndarray, float]:
    """Sliding-threshold binary mask on CD (NumPy float volume). Returns ``(mask, opt_thresh)``."""
    # ---- Preprocess + occupancy curve Sval(threshold) ------------------------
    cd = as_backend_array(cd).astype(np.float64, order="C")
    if med_filt_flag:
        cdcrop = ndi.median_filter(cd, size=3, mode="constant", cval=0.0)
    else:
        cdcrop = cd
    max_val = float(np.max(cdcrop))
    if max_val <= 0.0:
        return np.zeros(cd.shape, dtype=bool), 0.0

    x = np.arange(0.0, up_thresh + step * 0.5, step, dtype=np.float64)
    sval = np.empty(x.shape, dtype=np.float32)
    for i, n in enumerate(x):
        sval[i] = float(np.count_nonzero(cdcrop > (max_val * n)))

    # ---- Smooth Sval, curvature knee, optional FWHM shift --------------------
    smf = int(max(1, smf))
    kernel = np.ones(smf, dtype=np.float64) / float(smf)
    y = np.convolve(sval.astype(np.float64), kernel, mode="same")
    ymax = float(np.max(y))
    if ymax <= 0.0:
        return np.zeros(cd.shape, dtype=bool), 0.0
    y = y / ymax

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    num = dx * ddy
    denom = dx * dx + dy * dy
    curvature_sm = num / (np.sqrt(denom) ** 3)
    curvature_sm = np.nan_to_num(curvature_sm, nan=0.0, posinf=0.0, neginf=0.0)
    curvature_sm = np.maximum(curvature_sm, 0.0)
    curvature_sm = np.convolve(curvature_sm, kernel, mode="same")

    idx = int(np.argmax(curvature_sm))
    if shift_hm_flag:
        cmax = float(np.max(curvature_sm))
        if cmax <= 0.0:
            opt_frac = float(x[idx])
        else:
            above = curvature_sm >= (cmax * 0.5)
            positions = np.flatnonzero(above)
            if positions.size == 0:
                full_width = 0
            else:
                full_width = int(positions[-1] - positions[0])
            j = min(idx + full_width, x.size - 1)
            opt_frac = float(x[j])
    else:
        opt_frac = float(x[idx])

    # ---- Binarize at optimized absolute threshold ----------------------------
    opt_thresh = max_val * opt_frac
    segment = cdcrop > opt_thresh
    return segment.astype(bool, copy=False), float(opt_thresh)


# ---- Public API: global vessel mask + area opening ---------------------------


def binary_vessel_segment_cd(
    cd: np.ndarray,
    *,
    up_thresh: float = 0.8,
    shift_hm_flag: bool = True,
    min_component_fraction: float = 0.005,
) -> tuple[np.ndarray, float]:
    """Global vessel boolean mask: sliding threshold on CD, then area opening (face 3D)."""
    segment, opt_thresh = _binary_mask_sliding_threshold(
        cd,
        step=0.001,
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
