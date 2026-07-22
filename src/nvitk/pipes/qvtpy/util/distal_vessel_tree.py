"""eICAB-inspired distal vessel tree (Python-only: Frangi + hysteresis + watershed).

Does **not** call vasculature / VED binaries. Structure mirrors eICAB WB
``vessels_flood_filling``: tubular vesselness → GMM hysteresis binary tree →
distance-transform watershed of arterial markers into that tree.
"""

from __future__ import annotations

from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger

setup(globals())

log = Logger()

_DISTAL_HYST_LOW_FACTOR_DEFAULT: float = 3.0
_DISTAL_HYST_HIGH_FACTOR_DEFAULT: float = 0.5
_DISTAL_FRANGI_SIGMAS_DEFAULT: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
_DISTAL_MIN_TREE_CC_VOXELS: int = 5


def cd_vesselness(
    cd: np.ndarray,
    *,
    sigmas: tuple[float, ...] | list[float] = _DISTAL_FRANGI_SIGMAS_DEFAULT,
) -> tuple[np.ndarray, str]:
    """Tubular vesselness from CD (skimage Frangi); fallback = normalized CD."""
    cd_np = as_backend_array(cd).astype(np.float64)
    cd_np = np.nan_to_num(cd_np, nan=0.0, posinf=0.0, neginf=0.0)
    # Scale to [0, 1] for Frangi stability across scanners.
    lo, hi = float(np.percentile(cd_np, 1.0)), float(np.percentile(cd_np, 99.5))
    if hi <= lo + 1e-8:
        hi = float(cd_np.max()) if float(cd_np.max()) > lo else lo + 1.0
    cd_norm = np.clip((cd_np - lo) / (hi - lo), 0.0, 1.0)

    try:
        from skimage.filters import frangi

        v = frangi(
            cd_norm,
            sigmas=tuple(float(s) for s in sigmas),
            black_ridges=False,
        )
        v = np.nan_to_num(as_backend_array(v).astype(np.float64), nan=0.0)
        vmax = float(v.max())
        if vmax > 0.0:
            v = v / vmax
        log.info(
            f"distal vesselness: Frangi (sigmas={list(sigmas)}, "
            f"max={vmax:.4g})"
        )
        return v, "frangi"
    except Exception as exc:  # noqa: BLE001
        log.warning(f"distal vesselness: Frangi unavailable ({exc}); using CD")
        return cd_norm, "cd_normalized"


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
    low_factor: float = _DISTAL_HYST_LOW_FACTOR_DEFAULT,
    high_factor: float = _DISTAL_HYST_HIGH_FACTOR_DEFAULT,
    min_cc_voxels: int = _DISTAL_MIN_TREE_CC_VOXELS,
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
        meta.update({"mode": "percentile_fallback", "threshold": thr, "n_samples": int(pos.size)})
        return tree.astype(bool), meta

    # Exclude near-zero Frangi floor from the GMM (brain-mask eICAB already does this
    # by masking CSF/background; our CD fg mask does not).
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
    # eICAB: median component for low, max for high.
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

    # Drop tiny speck CCs.
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
        "distal vessel tree: hysteresis "
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
    # Always keep marker voxels even if outside the Frangi tree.
    out = keep[lab] | marks
    meta = {
        "n_before": int(np.count_nonzero(vessels)),
        "n_after": int(np.count_nonzero(out)),
        "n_cc_total": int(n_lab),
        "n_cc_kept": int(np.count_nonzero(keep[1:])),
    }
    log.info(
        "distal vessel tree: marker-connected CCs "
        f"{meta['n_cc_kept']}/{meta['n_cc_total']} "
        f"({meta['n_before']} → {meta['n_after']} voxels)"
    )
    return out, meta


def thicken_tree_in_cd(
    tree: np.ndarray,
    cd: np.ndarray,
    *,
    iterations: int = 1,
    gate_percentile: float = 70.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """1-voxel lumen thicken inside a high CD gate (not a parenchyma flood)."""
    from scipy import ndimage as ndi

    cores = as_backend_array(tree).astype(bool)
    cd_np = as_backend_array(cd).astype(np.float64)
    cd_pos = cd_np[cd_np > 0]
    n_iter = max(0, int(iterations))
    if n_iter == 0 or not np.any(cores) or cd_pos.size == 0:
        return cores, {
            "iterations": n_iter,
            "n_before": int(np.count_nonzero(cores)),
            "n_after": int(np.count_nonzero(cores)),
        }
    thr = float(np.percentile(cd_pos, float(gate_percentile)))
    gate = cd_np >= thr
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
            "distal vessel tree: lumen thicken "
            f"{meta['n_before']} → {meta['n_after']} "
            f"(iter={n_iter}, gate_p={gate_percentile:g})"
        )
    return thick.astype(bool), meta


def watershed_labels_into_vessels(
    vessels_bin: np.ndarray,
    markers: np.ndarray,
    *,
    connectivity: int = 3,
    erode_markers: bool = True,
) -> np.ndarray:
    """Watershed markers into the binary vessel tree (eICAB ``watershed_segment``)."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    vessels = as_backend_array(vessels_bin).astype(bool)
    marks = as_backend_array(markers).astype(np.int32, copy=True)
    # Markers must lie inside the mask.
    marks[~vessels] = 0
    if not np.any(marks) or not np.any(vessels):
        return np.zeros(vessels.shape, dtype=np.int32)

    if erode_markers:
        # eICAB erodes binary markers before watershed for skimage stability.
        binary = marks != 0
        eroded = ndi.binary_erosion(binary, structure=np.ones((3, 3, 3), dtype=bool))
        if np.any(eroded):
            marks = marks * eroded.astype(np.int32)
        if not np.any(marks):
            # Erosion wiped everything; restore originals clipped to vessels.
            marks = as_backend_array(markers).astype(np.int32, copy=True)
            marks[~vessels] = 0

    dist = ndi.distance_transform_edt(vessels)
    labels = watershed(
        -dist,
        marks,
        mask=vessels,
        connectivity=int(connectivity),
    )
    return as_backend_array(labels).astype(np.int32, copy=False)


__all__ = [
    "apply_hysteresis_threshold_3d",
    "cd_vesselness",
    "hysteresis_vessel_tree",
    "keep_tree_components_touching_markers",
    "thicken_tree_in_cd",
    "watershed_labels_into_vessels",
    "_DISTAL_FRANGI_SIGMAS_DEFAULT",
    "_DISTAL_HYST_HIGH_FACTOR_DEFAULT",
    "_DISTAL_HYST_LOW_FACTOR_DEFAULT",
]
