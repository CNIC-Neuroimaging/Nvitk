"""Minimal Frangi → GMM hysteresis → watershed blood-flood (no nvitk).

Vendored/simplified from nvitk.segmentation.blood_flood for the Slicer
Mouse TOF CoW module. Requires: numpy, scipy, scikit-image, scikit-learn.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from scipy import ndimage as ndi

HYST_LOW_FACTOR_DEFAULT: float = 3.0
HYST_HIGH_FACTOR_DEFAULT: float = 0.5
FRANGI_SIGMAS_DEFAULT: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
MIN_TREE_CC_VOXELS_DEFAULT: int = 5
TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT: float = 55.0
_GMM_MAX_FIT_SAMPLES: int = 200_000
_BLAS_THREAD_CAP: int = 8


@contextmanager
def _blas_thread_limit(n_threads: int = _BLAS_THREAD_CAP) -> Iterator[None]:
    n = max(1, int(n_threads))
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=n):
            yield
    except Exception:
        yield


@dataclass
class BloodFloodResult:
    labels: np.ndarray
    tree: np.ndarray
    vesselness: np.ndarray
    vesselness_mode: str
    info: dict[str, Any] = field(default_factory=dict)


def intensity_vesselness(
    intensity: np.ndarray,
    *,
    sigmas: tuple[float, ...] | list[float] = FRANGI_SIGMAS_DEFAULT,
) -> tuple[np.ndarray, str]:
    vol = np.asarray(intensity, dtype=np.float64)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.percentile(vol, 1.0)), float(np.percentile(vol, 99.5))
    if hi <= lo + 1e-8:
        hi = float(vol.max()) if float(vol.max()) > lo else lo + 1.0
    vol_norm = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

    try:
        from skimage.filters import frangi

        with _blas_thread_limit():
            v = frangi(
                vol_norm,
                sigmas=tuple(float(s) for s in sigmas),
                black_ridges=False,
            )
        v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0)
        vmax = float(v.max())
        if vmax > 0.0:
            v = v / vmax
        return v, "frangi"
    except Exception:
        return vol_norm, "intensity_normalized"


def _thresholds_sigma(
    min_mean: float,
    min_var: float,
    max_mean: float,
    max_var: float,
    *,
    low_factor: float,
    high_factor: float,
) -> tuple[float, float]:
    lowt = float(min_mean) + float(low_factor) * float(np.sqrt(max(min_var, 0.0)))
    hight = float(max_mean) + float(high_factor) * float(np.sqrt(max(max_var, 0.0)))
    return lowt, hight


def apply_hysteresis_threshold_3d(
    image: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    low = float(min(low, high))
    mask_low = img > low
    mask_high = img > high
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels_low, n_lab = ndi.label(mask_low, structure=structure)
    if n_lab == 0:
        return np.zeros(img.shape, dtype=bool)
    sums = ndi.sum(mask_high, labels_low, index=np.arange(1, n_lab + 1))
    connected = np.zeros(n_lab + 1, dtype=bool)
    connected[1:] = np.asarray(sums) > 0
    return connected[labels_low]


def hysteresis_vessel_tree(
    vesselness: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    low_factor: float = HYST_LOW_FACTOR_DEFAULT,
    high_factor: float = HYST_HIGH_FACTOR_DEFAULT,
    min_cc_voxels: int = MIN_TREE_CC_VOXELS_DEFAULT,
    fit_positive_percentile: float = 50.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.mixture import GaussianMixture

    v = np.asarray(vesselness, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        samples_all = v[m]
    else:
        m = None
        samples_all = v.ravel()
    samples_all = samples_all[np.isfinite(samples_all)]
    meta: dict[str, Any] = {
        "low_factor": float(low_factor),
        "high_factor": float(high_factor),
        "n_samples_mask": int(samples_all.size),
    }
    pos = samples_all[samples_all > 0]
    if pos.size < 50:
        thr = float(np.percentile(pos, 90.0)) if pos.size else 0.0
        tree = v > thr
        if m is not None:
            tree &= m
        meta.update({"mode": "percentile_fallback", "threshold": thr, "n_samples": int(pos.size)})
        return tree.astype(bool), meta

    fit_floor = float(np.percentile(pos, float(fit_positive_percentile)))
    samples = pos[pos >= fit_floor]
    if samples.size < 50:
        samples = pos
    meta["n_samples"] = int(samples.size)
    meta["fit_floor"] = float(fit_floor)

    if samples.size > _GMM_MAX_FIT_SAMPLES:
        rng = np.random.default_rng(0)
        idx = rng.choice(samples.size, size=_GMM_MAX_FIT_SAMPLES, replace=False)
        samples = samples[idx]
        meta["n_samples_fit"] = int(samples.size)
        meta["gmm_subsampled"] = True
    else:
        meta["n_samples_fit"] = int(samples.size)
        meta["gmm_subsampled"] = False

    gmm = GaussianMixture(
        n_components=3, tol=1e-3, max_iter=100, n_init=1, random_state=0
    )
    with _blas_thread_limit():
        gmm.fit(np.asarray(samples).reshape(-1, 1))
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    order = np.argsort(means)
    means_s = means[order]
    vars_s = variances[order]
    median_mean, median_var = float(means_s[1]), float(vars_s[1])
    max_mean, max_var = float(means_s[2]), float(vars_s[2])
    lowt, hight = _thresholds_sigma(
        median_mean,
        median_var,
        max_mean,
        max_var,
        low_factor=float(low_factor),
        high_factor=float(high_factor),
    )
    if hight < lowt:
        lowt, hight = hight, lowt
    vmax = float(np.max(samples))
    if not np.any(v > hight):
        hight = min(hight, max(vmax * 0.99, float(np.quantile(samples, 0.995))))
        if hight < lowt:
            lowt = max(0.0, hight * 0.9)
        meta["hight_clamped"] = True
    tree = apply_hysteresis_threshold_3d(v, low=lowt, high=hight)
    if m is not None:
        tree &= m

    min_cc = max(1, int(min_cc_voxels))
    keep = np.zeros(1, dtype=bool)
    lab, n_lab = ndi.label(tree, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_lab > 0:
        counts = np.bincount(lab.ravel())
        keep = np.zeros(n_lab + 1, dtype=bool)
        keep[1:] = counts[1:] >= min_cc
        tree = keep[lab]

    meta.update(
        {
            "mode": "gmm_hysteresis",
            "means": [float(x) for x in means_s],
            "lowt": float(lowt),
            "hight": float(hight),
            "n_tree_voxels": int(np.count_nonzero(tree)),
            "n_cc_kept": int(np.count_nonzero(keep[1:])) if n_lab > 0 else 0,
        }
    )
    return tree.astype(bool), meta


def keep_tree_components_touching_markers(
    tree: np.ndarray,
    markers: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    vessels = np.asarray(tree, dtype=bool)
    marks = np.asarray(markers) != 0
    if not np.any(vessels):
        return vessels, {"n_before": 0, "n_after": 0, "n_cc_kept": 0}
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n_lab = ndi.label(vessels, structure=structure)
    if n_lab == 0:
        return vessels, {"n_before": 0, "n_after": 0, "n_cc_kept": 0}
    touch = ndi.sum(marks, lab, index=np.arange(1, n_lab + 1))
    keep = np.zeros(n_lab + 1, dtype=bool)
    keep[1:] = np.asarray(touch) > 0
    out = keep[lab] | marks
    return out, {
        "n_before": int(np.count_nonzero(vessels)),
        "n_after": int(np.count_nonzero(out)),
        "n_cc_total": int(n_lab),
        "n_cc_kept": int(np.count_nonzero(keep[1:])),
    }


def thicken_tree_in_intensity(
    tree: np.ndarray,
    intensity: np.ndarray,
    *,
    iterations: int = 1,
    gate_percentile: float = 70.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    cores = np.asarray(tree, dtype=bool)
    vol = np.asarray(intensity, dtype=np.float64)
    pos = vol[vol > 0]
    n_iter = max(0, int(iterations))
    if n_iter == 0 or not np.any(cores) or pos.size == 0:
        return cores, {
            "iterations": n_iter,
            "n_before": int(np.count_nonzero(cores)),
            "n_after": int(np.count_nonzero(cores)),
        }
    thr = float(np.percentile(pos, float(gate_percentile)))
    gate = vol >= thr
    structure = np.ones((3, 3, 3), dtype=bool)
    thick = cores.copy()
    for _ in range(n_iter):
        thick = ndi.binary_dilation(thick, structure=structure) & gate
        thick |= cores
    return thick.astype(bool), {
        "iterations": n_iter,
        "gate_percentile": float(gate_percentile),
        "gate_threshold": thr,
        "n_before": int(np.count_nonzero(cores)),
        "n_after": int(np.count_nonzero(thick)),
    }


def thin_tree_by_vesselness(
    tree: np.ndarray,
    vesselness: np.ndarray,
    *,
    keep_percentile: float = TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    protect: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vessels = np.asarray(tree, dtype=bool)
    v = np.asarray(vesselness, dtype=np.float64)
    n_before = int(np.count_nonzero(vessels))
    if n_before == 0:
        return vessels, {"n_before": 0, "n_after": 0, "threshold": 0.0}
    thr = float(np.percentile(v[vessels], float(keep_percentile)))
    keep = v >= thr
    if protect is not None:
        keep = keep | np.asarray(protect, dtype=bool)
    out = vessels & keep
    return out.astype(bool), {
        "percentile": float(keep_percentile),
        "threshold": thr,
        "n_before": n_before,
        "n_after": int(np.count_nonzero(out)),
    }


def watershed_labels_into_vessels(
    vessels_bin: np.ndarray,
    markers: np.ndarray,
    *,
    connectivity: int = 3,
    erode_markers: bool = True,
) -> np.ndarray:
    from skimage.segmentation import watershed

    vessels = np.asarray(vessels_bin, dtype=bool)
    marks = np.asarray(markers, dtype=np.int32).copy()
    marks[~vessels] = 0
    if not np.any(marks) or not np.any(vessels):
        return np.zeros(vessels.shape, dtype=np.int32)

    if erode_markers:
        binary = marks != 0
        eroded = ndi.binary_erosion(binary, structure=np.ones((3, 3, 3), dtype=bool))
        if np.any(eroded):
            marks = marks * eroded.astype(np.int32)
        if not np.any(marks):
            marks = np.asarray(markers, dtype=np.int32).copy()
            marks[~vessels] = 0

    dist = ndi.distance_transform_edt(vessels)
    labels = watershed(
        -np.asarray(dist),
        np.asarray(marks),
        mask=np.asarray(vessels),
        connectivity=int(connectivity),
    )
    return np.asarray(labels, dtype=np.int32)


def blood_flood(
    intensity: np.ndarray,
    markers: np.ndarray,
    *,
    barrier: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    frangi_sigmas: tuple[float, ...] | list[float] | None = None,
    hyst_low_factor: float = HYST_LOW_FACTOR_DEFAULT,
    hyst_high_factor: float = HYST_HIGH_FACTOR_DEFAULT,
    thicken_iter: int = 0,
    thicken_gate_percentile: float = 85.0,
    thin_vesselness_percentile: float | None = TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    protect: np.ndarray | None = None,
    erode_markers: bool = False,
    connectivity: int = 3,
) -> BloodFloodResult:
    vol = np.asarray(intensity, dtype=np.float64)
    marks = np.asarray(markers, dtype=np.int32).copy()
    if marks.shape != vol.shape:
        raise ValueError(
            f"markers shape {marks.shape} must match intensity shape {vol.shape}"
        )
    sigmas = tuple(frangi_sigmas) if frangi_sigmas is not None else FRANGI_SIGMAS_DEFAULT
    info: dict[str, Any] = {"method": "frangi_hysteresis_watershed", "mode": "expand"}

    vesselness, vmode = intensity_vesselness(vol, sigmas=sigmas)
    tree_hyst, tree_meta = hysteresis_vessel_tree(
        vesselness,
        mask,
        low_factor=float(hyst_low_factor),
        high_factor=float(hyst_high_factor),
    )
    info["tree"] = tree_meta
    tree, touch_meta = keep_tree_components_touching_markers(tree_hyst, marks)
    info["tree_marker_cc"] = touch_meta

    thick_n = max(0, int(thicken_iter))
    if thick_n > 0:
        tree, thick_meta = thicken_tree_in_intensity(
            tree, vol, iterations=thick_n, gate_percentile=float(thicken_gate_percentile)
        )
        info["tree_thicken"] = thick_meta

    hard = None
    if barrier is not None:
        hard = np.asarray(barrier, dtype=bool)
        tree = (tree & ~hard) | (marks != 0)

    protect_b = np.asarray(protect, dtype=bool) if protect is not None else None
    if thin_vesselness_percentile is not None and np.count_nonzero(tree) > 0:
        protect_thin = protect_b if protect_b is not None else (marks != 0)
        if protect_b is not None:
            protect_thin = protect_thin | (marks != 0)
        tree, thin_meta = thin_tree_by_vesselness(
            tree,
            vesselness,
            keep_percentile=float(thin_vesselness_percentile),
            protect=protect_thin,
        )
        if hard is not None:
            tree = (tree & ~hard) | (marks != 0)
        info["tree_thin"] = thin_meta

    tree = tree | (marks != 0)
    if hard is not None:
        tree = (tree & ~hard) | (marks != 0)

    labels = watershed_labels_into_vessels(
        tree, marks, connectivity=int(connectivity), erode_markers=bool(erode_markers)
    )
    return BloodFloodResult(
        labels=labels,
        tree=tree.astype(bool),
        vesselness=vesselness,
        vesselness_mode=vmode,
        info=info,
    )


def blood_flood_from_scratch(
    intensity: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    barrier: np.ndarray | None = None,
    frangi_sigmas: tuple[float, ...] | list[float] | None = None,
    hyst_low_factor: float = HYST_LOW_FACTOR_DEFAULT,
    hyst_high_factor: float = HYST_HIGH_FACTOR_DEFAULT,
    thicken_iter: int = 0,
    thicken_gate_percentile: float = 85.0,
    thin_vesselness_percentile: float | None = TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    min_cc_voxels: int = MIN_TREE_CC_VOXELS_DEFAULT,
    connectivity: int = 3,
) -> BloodFloodResult:
    vol = np.asarray(intensity, dtype=np.float64)
    sigmas = tuple(frangi_sigmas) if frangi_sigmas is not None else FRANGI_SIGMAS_DEFAULT
    info: dict[str, Any] = {"method": "frangi_hysteresis_cc", "mode": "from_scratch"}

    vesselness, vmode = intensity_vesselness(vol, sigmas=sigmas)
    tree, tree_meta = hysteresis_vessel_tree(
        vesselness,
        mask,
        low_factor=float(hyst_low_factor),
        high_factor=float(hyst_high_factor),
        min_cc_voxels=int(min_cc_voxels),
    )
    info["tree"] = tree_meta

    thick_n = max(0, int(thicken_iter))
    if thick_n > 0:
        tree, _ = thicken_tree_in_intensity(
            tree, vol, iterations=thick_n, gate_percentile=float(thicken_gate_percentile)
        )

    if barrier is not None:
        tree = tree & ~np.asarray(barrier, dtype=bool)

    if thin_vesselness_percentile is not None and np.count_nonzero(tree) > 0:
        tree, _ = thin_tree_by_vesselness(
            tree, vesselness, keep_percentile=float(thin_vesselness_percentile), protect=None
        )
        if barrier is not None:
            tree = tree & ~np.asarray(barrier, dtype=bool)

    if int(connectivity) >= 3:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
    else:
        structure = ndi.generate_binary_structure(3, int(connectivity))
    labels, n_lab = ndi.label(tree, structure=structure)
    info["n_components"] = int(n_lab)
    return BloodFloodResult(
        labels=np.asarray(labels, dtype=np.int32),
        tree=tree.astype(bool),
        vesselness=vesselness,
        vesselness_mode=vmode,
        info=info,
    )
