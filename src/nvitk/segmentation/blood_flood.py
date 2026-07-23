"""Blood-vessel flood fill: Frangi vesselness → GMM hysteresis tree → watershed.

Python-only analogue of eICAB whole-brain ``vessels_flood_filling`` (no vasculature /
VED binaries). Designed as a reusable segmentation primitive for qvtpy distal
expansion and for standalone / GUI tools.

Pipeline
--------
1. Tubular vesselness on an intensity volume (Frangi; fallback = normalized intensity)
2. GMM + hysteresis → binary vessel tree
3. Keep only connected components that touch marker seeds
4. Optional lumen thicken / vesselness thinning
5. Distance-transform watershed of markers into the tree

Hard barriers (e.g. dilated ICA/basilar) can be punched out of the tree so they
are never labeled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger

setup(globals())

log = Logger()

HYST_LOW_FACTOR_DEFAULT: float = 3.0
HYST_HIGH_FACTOR_DEFAULT: float = 0.5
FRANGI_SIGMAS_DEFAULT: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
MIN_TREE_CC_VOXELS_DEFAULT: int = 5
TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT: float = 55.0

# Backward-compatible aliases used by qvtpy distal expand.
_DISTAL_HYST_LOW_FACTOR_DEFAULT = HYST_LOW_FACTOR_DEFAULT
_DISTAL_HYST_HIGH_FACTOR_DEFAULT = HYST_HIGH_FACTOR_DEFAULT
_DISTAL_FRANGI_SIGMAS_DEFAULT = FRANGI_SIGMAS_DEFAULT
_DISTAL_MIN_TREE_CC_VOXELS = MIN_TREE_CC_VOXELS_DEFAULT


@dataclass
class BloodFloodResult:
    """Labeled flood output and diagnostics."""

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
    """Tubular vesselness from a bright-blood intensity volume (skimage Frangi).

    Fallback when Frangi is unavailable: percentile-normalized intensity in ``[0, 1]``.
    """
    vol = as_backend_array(intensity).astype(np.float64)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.percentile(vol, 1.0)), float(np.percentile(vol, 99.5))
    if hi <= lo + 1e-8:
        hi = float(vol.max()) if float(vol.max()) > lo else lo + 1.0
    vol_norm = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)

    try:
        from skimage.filters import frangi

        v = frangi(
            to_numpy(vol_norm),
            sigmas=tuple(float(s) for s in sigmas),
            black_ridges=False,
        )
        v = np.nan_to_num(as_backend_array(v).astype(np.float64), nan=0.0)
        vmax = float(v.max())
        if vmax > 0.0:
            v = v / vmax
        log.info(
            f"blood_flood vesselness: Frangi (sigmas={list(sigmas)}, max={vmax:.4g})"
        )
        return v, "frangi"
    except Exception as exc:  # noqa: BLE001
        log.warning(f"blood_flood vesselness: Frangi unavailable ({exc}); using intensity")
        return vol_norm, "intensity_normalized"


def cd_vesselness(
    cd: np.ndarray,
    *,
    sigmas: tuple[float, ...] | list[float] = FRANGI_SIGMAS_DEFAULT,
) -> tuple[np.ndarray, str]:
    """Alias of :func:`intensity_vesselness` (qvtpy / CD naming)."""
    return intensity_vesselness(cd, sigmas=sigmas)


def _thresholds_sigma(
    min_mean: float,
    min_var: float,
    max_mean: float,
    max_var: float,
    *,
    low_factor: float,
    high_factor: float,
) -> tuple[float, float]:
    """eICAB-style hysteresis bounds from GMM component mean/variance."""
    lowt = float(min_mean) + float(low_factor) * float(np.sqrt(max(min_var, 0.0)))
    hight = float(max_mean) + float(high_factor) * float(np.sqrt(max(max_var, 0.0)))
    return lowt, hight


def apply_hysteresis_threshold_3d(
    image: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Keep ``>low`` CCs that touch any ``>high`` voxel (26-connectivity)."""
    from scipy import ndimage as ndi

    img = as_backend_array(image).astype(np.float64)
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
    """GMM + hysteresis binary vessel tree (eICAB ``hysteresis_thresholding_brain``).

    GMM is fit on the upper half of positive vesselness inside ``mask`` so near-zero
    Frangi noise does not pull ``lowt`` toward zero (which floods parenchyma).
    """
    from scipy import ndimage as ndi
    from sklearn.mixture import GaussianMixture

    v = as_backend_array(vesselness).astype(np.float64)
    if mask is not None:
        m = as_backend_array(mask).astype(bool)
        assert m.shape == v.shape, "vesselness and mask shapes must match"
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
        meta.update(
            {
                "mode": "percentile_fallback",
                "threshold": thr,
                "n_samples": int(pos.size),
            }
        )
        return tree.astype(bool), meta

    fit_floor = float(np.percentile(pos, float(fit_positive_percentile)))
    samples = pos[pos >= fit_floor]
    if samples.size < 50:
        samples = pos
    meta["n_samples"] = int(samples.size)
    meta["fit_floor"] = float(fit_floor)

    gmm = GaussianMixture(
        n_components=3,
        tol=1e-3,
        max_iter=100,
        n_init=1,
        random_state=0,
    )
    gmm.fit(samples.reshape(-1, 1))
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
    log.info(
        "blood_flood tree: hysteresis "
        f"lowt={lowt:.4g} hight={hight:.4g} "
        f"voxels={meta['n_tree_voxels']}"
    )
    return tree.astype(bool), meta


def keep_tree_components_touching_markers(
    tree: np.ndarray,
    markers: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain only vessel-tree CCs that touch at least one marker voxel."""
    from scipy import ndimage as ndi

    vessels = as_backend_array(tree).astype(bool)
    marks = as_backend_array(markers) != 0
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
    meta = {
        "n_before": int(np.count_nonzero(vessels)),
        "n_after": int(np.count_nonzero(out)),
        "n_cc_total": int(n_lab),
        "n_cc_kept": int(np.count_nonzero(keep[1:])),
    }
    log.info(
        "blood_flood tree: marker-connected CCs "
        f"{meta['n_cc_kept']}/{meta['n_cc_total']} "
        f"({meta['n_before']} → {meta['n_after']} voxels)"
    )
    return out, meta


def thicken_tree_in_intensity(
    tree: np.ndarray,
    intensity: np.ndarray,
    *,
    iterations: int = 1,
    gate_percentile: float = 70.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """1-voxel lumen thicken inside a high-intensity gate (not a parenchyma flood)."""
    from scipy import ndimage as ndi

    cores = as_backend_array(tree).astype(bool)
    vol = as_backend_array(intensity).astype(np.float64)
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
    meta = {
        "iterations": n_iter,
        "gate_percentile": float(gate_percentile),
        "gate_threshold": thr,
        "n_before": int(np.count_nonzero(cores)),
        "n_after": int(np.count_nonzero(thick)),
    }
    if meta["n_after"] != meta["n_before"]:
        log.info(
            "blood_flood tree: lumen thicken "
            f"{meta['n_before']} → {meta['n_after']} "
            f"(iter={n_iter}, gate_p={gate_percentile:g})"
        )
    return thick.astype(bool), meta


def thicken_tree_in_cd(
    tree: np.ndarray,
    cd: np.ndarray,
    *,
    iterations: int = 1,
    gate_percentile: float = 70.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Alias of :func:`thicken_tree_in_intensity` (qvtpy / CD naming)."""
    return thicken_tree_in_intensity(
        tree, cd, iterations=iterations, gate_percentile=gate_percentile
    )


def thin_tree_by_vesselness(
    tree: np.ndarray,
    vesselness: np.ndarray,
    *,
    keep_percentile: float = TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    protect: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop weak Frangi shell voxels; optionally protect a boolean zone."""
    vessels = as_backend_array(tree).astype(bool)
    v = as_backend_array(vesselness).astype(np.float64)
    n_before = int(np.count_nonzero(vessels))
    if n_before == 0:
        return vessels, {"n_before": 0, "n_after": 0, "threshold": 0.0}
    thr = float(np.percentile(v[vessels], float(keep_percentile)))
    keep = v >= thr
    if protect is not None:
        keep = keep | as_backend_array(protect).astype(bool)
    out = vessels & keep
    meta = {
        "percentile": float(keep_percentile),
        "threshold": thr,
        "n_before": n_before,
        "n_after": int(np.count_nonzero(out)),
    }
    return out.astype(bool), meta


def watershed_labels_into_vessels(
    vessels_bin: np.ndarray,
    markers: np.ndarray,
    *,
    connectivity: int = 3,
    erode_markers: bool = True,
) -> np.ndarray:
    """Watershed markers into the binary vessel tree (eICAB ``watershed_segment``)."""
    from skimage.segmentation import watershed

    vessels = as_backend_array(vessels_bin).astype(bool)
    marks = as_backend_array(markers).astype(np.int32, copy=True)
    marks[~vessels] = 0
    if not np.any(marks) or not np.any(vessels):
        return np.zeros(vessels.shape, dtype=np.int32)

    if erode_markers:
        binary = marks != 0
        eroded = ndi.binary_erosion(binary, structure=np.ones((3, 3, 3), dtype=bool))
        if np.any(eroded):
            marks = marks * eroded.astype(np.int32)
        if not np.any(marks):
            marks = as_backend_array(markers).astype(np.int32, copy=True)
            marks[~vessels] = 0

    dist = ndi.distance_transform_edt(vessels)
    labels = watershed(
        -to_numpy(dist),
        to_numpy(marks),
        mask=to_numpy(vessels),
        connectivity=int(connectivity),
    )
    return as_backend_array(labels).astype(np.int32, copy=False)


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
    """Run the full Frangi → hysteresis → marker-CC → watershed flood.

    Parameters
    ----------
    intensity
        Bright-blood volume (e.g. CD / TOF).
    markers
        Integer seed labels (``0`` = background). Non-zero voxels seed the flood.
    barrier
        Optional hard wall: these voxels are removed from the vessel tree (markers
        are forced back so seeds always remain).
    mask
        Optional foreground for the GMM / hysteresis fit.
    protect
        Optional voxels preserved through vesselness thinning (e.g. ACA corridor).

    Returns
    -------
    BloodFloodResult
        ``labels`` is the watershed assignment inside the tree (``0`` unlabeled).
    """
    vol = as_backend_array(intensity).astype(np.float64)
    marks = as_backend_array(markers).astype(np.int32, copy=True)
    if marks.shape != vol.shape:
        raise ValueError(
            f"markers shape {marks.shape} must match intensity shape {vol.shape}"
        )
    sigmas = (
        tuple(frangi_sigmas)
        if frangi_sigmas is not None
        else FRANGI_SIGMAS_DEFAULT
    )
    info: dict[str, Any] = {
        "method": "frangi_hysteresis_watershed",
        "frangi_sigmas": list(sigmas),
        "hyst_low_factor": float(hyst_low_factor),
        "hyst_high_factor": float(hyst_high_factor),
        "thicken_iter": int(thicken_iter),
    }

    vesselness, vmode = intensity_vesselness(vol, sigmas=sigmas)
    info["vesselness_mode"] = vmode
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
            tree,
            vol,
            iterations=thick_n,
            gate_percentile=float(thicken_gate_percentile),
        )
        info["tree_thicken"] = thick_meta

    hard = None
    if barrier is not None:
        hard = as_backend_array(barrier).astype(bool)
        if hard.shape != vol.shape:
            raise ValueError(
                f"barrier shape {hard.shape} must match intensity shape {vol.shape}"
            )
        tree = (tree & ~hard) | (marks != 0)
        info["barrier_voxels"] = int(np.count_nonzero(hard))

    if protect is not None:
        protect_b = as_backend_array(protect).astype(bool)
        if protect_b.shape != vol.shape:
            raise ValueError(
                f"protect shape {protect_b.shape} must match intensity shape {vol.shape}"
            )
    else:
        protect_b = None

    if thin_vesselness_percentile is not None and np.count_nonzero(tree) > 0:
        protect_thin = protect_b
        if protect_thin is None:
            protect_thin = marks != 0
        else:
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
        tree,
        marks,
        connectivity=int(connectivity),
        erode_markers=bool(erode_markers),
    )
    info["n_tree_voxels"] = int(np.count_nonzero(tree))
    info["n_labeled"] = int(np.count_nonzero(labels))
    info["n_marker_voxels"] = int(np.count_nonzero(marks))

    return BloodFloodResult(
        labels=labels,
        tree=tree.astype(bool),
        vesselness=vesselness,
        vesselness_mode=vmode,
        info=info,
    )


__all__ = [
    "BloodFloodResult",
    "FRANGI_SIGMAS_DEFAULT",
    "HYST_HIGH_FACTOR_DEFAULT",
    "HYST_LOW_FACTOR_DEFAULT",
    "MIN_TREE_CC_VOXELS_DEFAULT",
    "TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT",
    "_DISTAL_FRANGI_SIGMAS_DEFAULT",
    "_DISTAL_HYST_HIGH_FACTOR_DEFAULT",
    "_DISTAL_HYST_LOW_FACTOR_DEFAULT",
    "apply_hysteresis_threshold_3d",
    "blood_flood",
    "cd_vesselness",
    "hysteresis_vessel_tree",
    "intensity_vesselness",
    "keep_tree_components_touching_markers",
    "thicken_tree_in_cd",
    "thicken_tree_in_intensity",
    "thin_tree_by_vesselness",
    "watershed_labels_into_vessels",
]
