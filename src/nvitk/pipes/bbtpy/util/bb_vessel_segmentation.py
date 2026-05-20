"""
Black-blood lumen segmentation from warped eICAB labels.

Per vessel: dilate the eICAB label on ``vwi_bb``, estimate a hypointense threshold from
intensities inside that dilated ROI only, and paste ``wvi < threshold`` voxels into
``seg_bb``. No region growing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d
from nvitk.morphology.binary import dilate
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.bbtpy.labels import bb_vessel_name

setup(globals())

log = Logger()

SEG_BB_NIFTI = "seg_bb.nii.gz"
SEGMENTATION_META_JSON = "segmentation_meta.json"
SEG_STRATEGY = "eicab_mask_hypointense_threshold"

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]


@dataclass
class BbSegResult:
    """Segmentation result before NIfTI write."""

    seg: np.ndarray
    stats: list[dict[str, Any]]


def _dilate_label_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    m = as_backend_array(mask).astype(bool, copy=False)
    if radius <= 0 or not np.any(m):
        return m
    return as_backend_array(
        dilate(m.astype(np.uint8), footprint=int(radius), connectivity=1)
    ).astype(bool, copy=False)


def _hypointense_threshold_in_roi(
    wvi: np.ndarray,
    roi_mask: np.ndarray,
    algorithm: ThrAlgorithm,
) -> tuple[np.ndarray, float | None, str | None]:
    """Threshold dark voxels inside *roi_mask* using ROI intensities only."""
    wvi_np = as_backend_array(wvi).astype(np.float64)
    roi = as_backend_array(roi_mask).astype(bool, copy=False)
    if not np.any(roi):
        return np.zeros_like(roi, dtype=bool), None, "empty ROI"

    samples = wvi_np[roi]
    if samples.size < 2:
        return np.zeros(roi.shape, dtype=bool), None, "insufficient ROI samples"

    if algorithm == "otsu":
        try:
            from skimage.filters import threshold_otsu
        except ImportError as exc:
            raise ImportError("otsu requires scikit-image") from exc
        try:
            t = float(threshold_otsu(samples))
        except ValueError as exc:
            return np.zeros(roi.shape, dtype=bool), None, f"otsu failed: {exc}"
        lumen = roi & (wvi_np < t)
        return as_backend_array(lumen).astype(bool), t, None

    xs, ys, zs = np.nonzero(roi)
    i0, i1 = int(xs.min()), int(xs.max())
    j0, j1 = int(ys.min()), int(ys.max())
    k0, k1 = int(zs.min()), int(zs.max())
    wvi_crop = wvi_np[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
    roi_crop = roi[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]

    vmax = float(np.max(wvi_crop[roi_crop]))
    if vmax <= 0.0:
        return np.zeros(roi.shape, dtype=bool), 0.0, None

    inv = vmax - wvi_crop
    shift_hm = algorithm == "lthr"
    mask_inv, opt_inv = binary_mask_sliding_threshold_3d(
        inv,
        shift_hm_flag=shift_hm,
        med_filt_flag=True,
    )
    lumen_crop = as_backend_array(mask_inv).astype(bool) & roi_crop
    opt_t = vmax - float(opt_inv)

    lumen = np.zeros(roi.shape, dtype=bool)
    lumen[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] = lumen_crop
    return as_backend_array(lumen).astype(bool), opt_t, None


def _paste_vessel_mask(
    seg: np.ndarray,
    vessel_mask: np.ndarray,
    label_id: int,
) -> int:
    """Write *vessel_mask* into empty voxels of *seg*."""
    m = as_backend_array(vessel_mask).astype(bool, copy=False)
    free = seg == 0
    write = m & free
    n = int(np.count_nonzero(write))
    if n > 0:
        seg[write] = int(label_id)
    return n


def build_seg_bb(
    wvi: np.ndarray,
    eicab_bb: np.ndarray,
    *,
    eicab_dilate: int = 4,
    thr_algorithm: ThrAlgorithm = "lsthr",
    min_component_frac: float = 0.005,
) -> BbSegResult:
    """Per-vessel dilated eICAB ROI + hypointense threshold → ``seg_bb``."""
    wvi_np = as_backend_array(wvi).astype(np.float64)
    eicab_np = as_backend_array(eicab_bb).astype(np.int32, copy=False)
    if tuple(wvi_np.shape[:3]) != tuple(eicab_np.shape[:3]):
        raise ValueError("eicab_bb shape must match wvi_bb")

    seg = np.zeros(eicab_np.shape, dtype=np.int32)
    stats: list[dict[str, Any]] = []
    dil_rad = max(0, int(eicab_dilate))
    label_ids = sorted(int(v) for v in np.unique(eicab_np) if int(v) > 0)
    log.step(
        f"eICAB-mask threshold seg: {len(label_ids)} label(s), "
        f"dilate={dil_rad}, thr={thr_algorithm}"
    )

    for lid in label_ids:
        core = as_backend_array(eicab_np == int(lid)).astype(bool, copy=False)
        if not np.any(core):
            stats.append({"label_id": lid, "warning": "empty eICAB label", "n_voxels": 0})
            continue

        roi = _dilate_label_mask(core, radius=dil_rad)
        lumen, opt_t, warn = _hypointense_threshold_in_roi(
            wvi_np, roi, thr_algorithm
        )
        if float(min_component_frac) > 0.0 and np.any(lumen):
            lumen = as_backend_array(
                remove_small_components_by_fraction(
                    lumen,
                    min_fraction=float(min_component_frac),
                    connectivity=1,
                )
            ).astype(bool, copy=False)

        n = _paste_vessel_mask(seg, lumen, lid)
        stats.append(
            {
                "label_id": lid,
                "thr_algorithm": thr_algorithm,
                "opt_thresh": opt_t,
                "eicab_dilate": dil_rad,
                "n_voxels_in_roi": int(np.count_nonzero(roi)),
                "n_voxels_segmented": n,
                "warning": warn,
            }
        )
        log.step(
            f"{bb_vessel_name(lid)} (id={lid}): {n} voxels "
            f"(opt_t={opt_t}, roi={int(np.count_nonzero(roi))})"
        )

    return BbSegResult(seg=seg, stats=stats)


def run_bb_segmentation(
    wvi: np.ndarray,
    eicab_bb: np.ndarray,
    out_dir: Path,
    *,
    eicab_dilate: int = 4,
    thr_algorithm: ThrAlgorithm = "lsthr",
    min_component_frac: float = 0.005,
    metadata: dict[str, Any] | None = None,
    skip_existing: bool = False,
) -> Path:
    """Write ``seg_bb.nii.gz`` and ``segmentation_meta.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / SEG_BB_NIFTI
    meta_path = out_dir / SEGMENTATION_META_JSON
    if skip_existing and seg_path.is_file() and meta_path.is_file():
        return seg_path

    log.step(f"BB segmentation | strategy={SEG_STRATEGY}")
    result = build_seg_bb(
        wvi,
        eicab_bb,
        eicab_dilate=eicab_dilate,
        thr_algorithm=thr_algorithm,
        min_component_frac=min_component_frac,
    )

    from nvitk.io.imageio import imsave

    imsave(seg_path, result.seg, metadata=dict(metadata or {}))
    log.step(f"wrote {seg_path.name}")
    meta: dict[str, Any] = {
        "strategy": SEG_STRATEGY,
        "eicab_dilate": eicab_dilate,
        "thr_algorithm": thr_algorithm,
        "min_component_frac": min_component_frac,
        "vessel_stats": result.stats,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return seg_path
