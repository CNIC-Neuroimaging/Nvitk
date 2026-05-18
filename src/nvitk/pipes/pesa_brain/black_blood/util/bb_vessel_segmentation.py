"""
Black-blood artery segmentation via centerline-guided region growing.

Black-blood (VWI) lumen is **hypointense** (dark). Region growing uses
``polarity='hypointense'`` on native ``vwi_bb`` intensities (no inversion).

Outputs (under stage2 dir)
--------------------------
- ``seg_bb.nii.gz`` — multilabel BB segmentation (same affine as ``vwi_bb``).
- ``segmentation_meta.json`` — strategy parameters and per-vessel stats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.morphology.binary import dilate
from nvitk.pipes.pesa_brain.black_blood.labels import bb_vessel_name
from nvitk.segmentation.region_growing import region_grow_into_label_volume

setup(globals())

log = Logger()

SEG_BB_NIFTI = "seg_bb.nii.gz"
SEGMENTATION_META_JSON = "segmentation_meta.json"
BB_RG_POLARITY = "hypointense"


@dataclass
class BbSegResult:
    """Centerline-growth result before NIfTI write."""

    seg: np.ndarray
    stats: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Barriers between vessels
# ---------------------------------------------------------------------------


def _dilate_bool_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    m = as_backend_array(mask).astype(bool, copy=False)
    if radius <= 0 or not np.any(m):
        return m
    return as_backend_array(
        dilate(m.astype(np.uint8), footprint=int(radius), connectivity=1)
    ).astype(bool, copy=False)


def _dilated_other_centerlines_barrier(
    centerlines_mask: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    *,
    radius: int,
) -> np.ndarray:
    i0, i1, j0, j1, k0, k1 = bbox
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    other = as_backend_array((clm != 0) & (clm != int(label_id))).astype(bool, copy=False)
    other = _dilate_bool_mask(other, radius=radius)
    return other[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]


def _dilated_other_segmentation_barrier(
    seg: np.ndarray,
    label_id: int,
    *,
    radius: int,
) -> np.ndarray:
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    other = np.zeros(seg_np.shape, dtype=bool)
    for other_id in np.unique(seg_np):
        oid = int(other_id)
        if oid == 0 or oid == int(label_id):
            continue
        other |= as_backend_array(seg_np == oid).astype(bool, copy=False)
    if not np.any(other):
        return other
    return _dilate_bool_mask(other, radius=radius)


# ---------------------------------------------------------------------------
# Centerline-guided region growing (hypointense / dark lumen)
# ---------------------------------------------------------------------------


def build_seg_bb_centerline_growth(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    rg_intensity_frac: float = 0.45,
    rg_abs_ceiling: float | None = None,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 2,
) -> BbSegResult:
    """Grow each labeled centerline into dark (hypointense) neighbouring voxels."""
    wvi_np = as_backend_array(wvi).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    seg = clm.copy()
    stats: list[dict[str, Any]] = []
    cl_rad = max(0, int(cl_barrier_radius))
    rg_rad = max(0, int(rg_barrier_radius))
    label_ids = sorted(int(v) for v in np.unique(clm) if int(v) > 0)
    log.step(
        f"hypointense region-growing {len(label_ids)} vessel label(s) "
        f"(frac={rg_intensity_frac})"
    )

    for lid in label_ids:
        other_cl = _dilated_other_centerlines_barrier(
            clm,
            lid,
            (0, seg.shape[0] - 1, 0, seg.shape[1] - 1, 0, seg.shape[2] - 1),
            radius=cl_rad,
        )
        other_seg = _dilated_other_segmentation_barrier(seg, lid, radius=rg_rad)
        forbidden = other_cl | other_seg
        n = region_grow_into_label_volume(
            seg,
            wvi_np,
            lid,
            intensity_frac=float(rg_intensity_frac),
            abs_floor=rg_abs_ceiling,
            forbidden=forbidden,
            polarity=BB_RG_POLARITY,
        )
        stats.append({"label_id": lid, "n_voxels_grown": n})
        log.step(
            f"{bb_vessel_name(lid)} (id={lid}): +{n} voxels "
            f"(total label voxels={int(np.count_nonzero(seg == lid))})"
        )

    return BbSegResult(seg=seg, stats=stats)


def run_bb_segmentation(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    out_dir: Path,
    *,
    rg_intensity_frac: float = 0.45,
    rg_abs_ceiling: float | None = None,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 2,
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

    log.step("centerline-growth on native vwi_bb (hypointense RG)")
    result = build_seg_bb_centerline_growth(
        wvi,
        centerlines_mask,
        rg_intensity_frac=rg_intensity_frac,
        rg_abs_ceiling=rg_abs_ceiling,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
    )

    from nvitk.io.imageio import imsave  

    md = dict(metadata or {})
    imsave(seg_path, result.seg, metadata=md)
    log.step(f"wrote {seg_path.name}")
    meta: dict[str, Any] = {
        "strategy": "centerline-growth",
        "rg_polarity": BB_RG_POLARITY,
        "rg_intensity_frac": rg_intensity_frac,
        "rg_abs_ceiling": rg_abs_ceiling,
        "cl_barrier_radius": cl_barrier_radius,
        "rg_barrier_radius": rg_barrier_radius,
        "vessel_stats": result.stats,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return seg_path
