"""Sliding-threshold binary segmentation (MATLAB slidingThreshold.m)."""

from __future__ import annotations

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup

setup(globals())


def binary_mask_sliding_threshold_3d(
    cd: np.ndarray,
    *,
    step: float = 0.001,
    up_thresh: float = 0.8,
    smf: int = 10,
    shift_hm_flag: bool = True,
    med_filt_flag: bool = True,
) -> tuple[np.ndarray, float]:
    """Sliding-threshold binary mask on a 3D CD volume. Returns ``(mask, opt_thresh)``."""
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

    opt_thresh = max_val * opt_frac
    segment = cdcrop > opt_thresh
    return as_backend_array(to_numpy(segment).astype(bool, copy=False)), float(opt_thresh)


def binary_mask_sliding_threshold_2d(
    fused: np.ndarray,
    *,
    step: float = 0.001,
    up_thresh: float = 0.8,
    smf: int = 90,
    shift_hm_flag: bool = True,
) -> np.ndarray:
    """2D sliding-threshold binary mask on normalized fused contrast."""
    img = as_backend_array(fused).astype(np.float64)
    max_val = float(np.max(img))
    if max_val <= 0.0:
        return np.zeros(img.shape, dtype=bool)

    x = np.arange(0.0, up_thresh + step * 0.5, step, dtype=np.float64)
    sval = np.array([float(np.count_nonzero(img > (max_val * n))) for n in x], dtype=np.float64)
    smf = int(max(1, smf))
    kernel = np.ones(smf, dtype=np.float64) / float(smf)
    y = np.convolve(sval, kernel, mode="same")
    ymax = float(np.max(y))
    if ymax <= 0.0:
        return np.zeros(img.shape, dtype=bool)
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

    thresh = max_val * opt_frac
    return (img > thresh).astype(bool, copy=False)


__all__ = [
    "binary_mask_sliding_threshold_2d",
    "binary_mask_sliding_threshold_3d",
]
