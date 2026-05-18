"""
Black-blood artery segmentation (eICAB-guided centerline lumen).

Per vessel: dilated eICAB corridor intersected with a centerline distance tube,
seed-adaptive hypointense ceiling from local (tube) intensities, centerline-connected
components, then bbox region growing confined to the grow domain.
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
from nvitk.morphology.components import (
    keep_components_touching_seeds,
    remove_small_components_by_fraction,
)
from nvitk.pipes.pesa_brain.black_blood.labels import BB_ICA_IDS, bb_vessel_name
from nvitk.segmentation.region_growing import region_grow_into_label_volume

setup(globals())

log = Logger()

SEG_BB_NIFTI = "seg_bb.nii.gz"
SEGMENTATION_META_JSON = "segmentation_meta.json"
BB_RG_POLARITY = "hypointense"
SEG_STRATEGY = "eicab_guided_centerline_lumen"


@dataclass
class BbSegResult:
    """Segmentation result before NIfTI write."""

    seg: np.ndarray
    stats: list[dict[str, Any]]


@dataclass(frozen=True)
class _VesselSegTuning:
    """Per-vessel segmentation knobs (ICA defaults are stricter)."""

    eicab_prior_dilate: int
    centerline_max_dist: int
    lumen_intensity_frac: float
    lumen_percentile: float


# ---------------------------------------------------------------------------
# Bbox / barriers / ROI
# ---------------------------------------------------------------------------


def _bbox_with_symmetric_padding(
    roi: np.ndarray,
    shape: tuple[int, int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int, int, int] | None:
    m = as_backend_array(roi.astype(bool, copy=False))
    if not np.any(m):
        return None
    pad = max(0, int(padding))
    xs, ys, zs = np.nonzero(m)
    nx, ny, nz = shape
    return (
        max(0, int(xs.min()) - pad),
        min(nx - 1, int(xs.max()) + pad),
        max(0, int(ys.min()) - pad),
        min(ny - 1, int(ys.max()) + pad),
        max(0, int(zs.min()) - pad),
        min(nz - 1, int(zs.max()) + pad),
    )


def _dilate_bool_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    m = as_backend_array(mask).astype(bool, copy=False)
    if radius <= 0 or not np.any(m):
        return m
    return as_backend_array(
        dilate(m.astype(np.uint8), footprint=int(radius), connectivity=1)
    ).astype(bool, copy=False)


def _centerline_tube_mask(seed_slab: np.ndarray, max_dist: int) -> np.ndarray:
    """Voxels within *max_dist* (6-connected) of centerline seeds."""
    seeds = as_backend_array(seed_slab).astype(bool, copy=False)
    if not np.any(seeds):
        return np.zeros_like(seeds, dtype=bool)
    rad = int(max_dist)
    if rad <= 0:
        return seeds
    dist = ndi.distance_transform_edt(~seeds)
    return as_backend_array(dist <= float(rad)).astype(bool, copy=False)


def _tuning_for_label(
    label_id: int,
    *,
    eicab_prior_dilate: int,
    centerline_max_dist: int,
    ica_centerline_max_dist: int,
    lumen_intensity_frac: float,
    lumen_percentile: float,
    ica_lumen_intensity_frac: float,
    ica_lumen_percentile: float,
) -> _VesselSegTuning:
    """Return per-label tuning (stricter for ICA)."""
    lid = int(label_id)
    if lid in BB_ICA_IDS:
        return _VesselSegTuning(
            eicab_prior_dilate=min(int(eicab_prior_dilate), 2),
            centerline_max_dist=max(1, int(ica_centerline_max_dist)),
            lumen_intensity_frac=float(ica_lumen_intensity_frac),
            lumen_percentile=float(ica_lumen_percentile),
        )
    return _VesselSegTuning(
        eicab_prior_dilate=int(eicab_prior_dilate),
        centerline_max_dist=max(1, int(centerline_max_dist)),
        lumen_intensity_frac=float(lumen_intensity_frac),
        lumen_percentile=float(lumen_percentile),
    )


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


def _forbidden_in_bbox(
    seg: np.ndarray,
    centerlines_mask: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    *,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    grow_domain: np.ndarray | None = None,
) -> np.ndarray:
    """BBox forbidden mask for constrained RG (other vessels + optional outside grow domain)."""
    other_cl = _dilated_other_centerlines_barrier(
        centerlines_mask, label_id, bbox, radius=cl_barrier_radius
    )
    other_seg_full = _dilated_other_segmentation_barrier(
        seg, label_id, radius=rg_barrier_radius
    )
    i0, i1, j0, j1, k0, k1 = bbox
    other_seg = other_seg_full[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
    forbidden = other_cl | other_seg
    if grow_domain is not None:
        forbidden = forbidden | ~as_backend_array(grow_domain).astype(bool, copy=False)
    return forbidden


def _forbidden_for_free_rg(
    seg: np.ndarray,
    centerlines_mask: np.ndarray,
    label_id: int,
    *,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
) -> np.ndarray:
    """Full-volume forbidden mask for unconstrained RG (other centerlines + other seg)."""
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    other_cl = as_backend_array((clm != 0) & (clm != int(label_id))).astype(bool, copy=False)
    other_cl = _dilate_bool_mask(other_cl, radius=cl_barrier_radius)
    other_seg = _dilated_other_segmentation_barrier(seg, label_id, radius=rg_barrier_radius)
    return other_cl | other_seg


def _eicab_prior_slab(
    eicab_bb: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    *,
    dilate_radius: int,
) -> np.ndarray:
    """Dilated eICAB label mask inside *bbox* (vessel corridor prior)."""
    i0, i1, j0, j1, k0, k1 = bbox
    eicab_np = as_backend_array(eicab_bb).astype(np.int32, copy=False)
    prior = as_backend_array(eicab_np == int(label_id)).astype(bool, copy=False)
    prior = _dilate_bool_mask(prior, radius=dilate_radius)
    return prior[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]


def _seed_intensity_stats(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    label_id: int,
) -> tuple[float, float, float]:
    """Return ``(seed_mean, seed_p25, seed_p75)`` from centerline voxels."""
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    int_np = as_backend_array(wvi).astype(np.float64)
    seeds = np.argwhere(clm == int(label_id))
    if seeds.size == 0:
        return 0.0, 0.0, 0.0
    vals = int_np[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    return (
        float(np.mean(vals)),
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 75)),
    )


def _hypointense_ceiling(
    wvi_crop: np.ndarray,
    local_mask: np.ndarray,
    seed_mean: float,
    seed_p75: float,
    *,
    lumen_intensity_frac: float,
    lumen_percentile: float,
) -> float:
    """Intensity ceiling for hypointense lumen: ``I <= ceiling``.

    Uses intensities in *local_mask* (centerline tube) for the percentile cap so
    dark background in a wide eICAB corridor does not raise the threshold.
    """
    mean = float(seed_mean)
    frac = float(lumen_intensity_frac)
    by_seed = mean * frac if mean > 0.0 else 0.0
    by_p75 = float(seed_p75) * frac if seed_p75 > 0.0 else by_seed
    by_seed = max(by_seed, by_p75)

    local = as_backend_array(local_mask).astype(bool, copy=False)
    wvi_np = as_backend_array(wvi_crop).astype(np.float64)
    samples = wvi_np[local]
    if samples.size == 0:
        samples = wvi_np.ravel()
    pctl = float(np.percentile(samples, float(lumen_percentile)))

    if by_seed <= 0.0:
        return pctl
    return float(min(by_seed, pctl))


def _lumen_mask_in_bbox(
    wvi_crop: np.ndarray,
    seed_slab: np.ndarray,
    grow_domain: np.ndarray,
    *,
    ceiling: float,
    min_component_frac: float,
) -> np.ndarray:
    """Hypointense candidates in grow domain, centerline-connected, island-cleaned."""
    wvi_np = as_backend_array(wvi_crop).astype(np.float64)
    domain = as_backend_array(grow_domain).astype(bool, copy=False)
    seeds = as_backend_array(seed_slab).astype(bool, copy=False)

    candidates = domain & (wvi_np < float(ceiling))
    connected = keep_components_touching_seeds(candidates, seeds, connectivity=1)
    connected = as_backend_array(connected).astype(bool, copy=False)
    if float(min_component_frac) > 0.0 and np.any(connected):
        connected = as_backend_array(
            remove_small_components_by_fraction(
                connected,
                min_fraction=float(min_component_frac),
                connectivity=1,
            )
        ).astype(bool, copy=False)
    return connected


def _paste_label_slab(
    seg: np.ndarray,
    crop_mask: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    *,
    forbidden: np.ndarray | None = None,
) -> int:
    i0, i1, j0, j1, k0, k1 = bbox
    slab = seg[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
    m = as_backend_array(crop_mask.astype(bool, copy=False))
    free = slab == 0
    if forbidden is not None:
        free = free & ~as_backend_array(forbidden).astype(bool, copy=False)
    write = m & free
    n = int(np.count_nonzero(write))
    if n > 0:
        slab[write] = int(label_id)
    return n


def _region_grow_in_bbox(
    seg: np.ndarray,
    wvi: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    *,
    rg_intensity_frac: float,
    rg_abs_ceiling: float | None,
    forbidden: np.ndarray | None,
) -> int:
    """Hypointense RG inside *bbox* with spatial *forbidden* mask."""
    i0, i1, j0, j1, k0, k1 = bbox
    seg_crop = seg[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
    wvi_crop = as_backend_array(wvi).astype(np.float64)[
        i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1
    ]
    return region_grow_into_label_volume(
        seg_crop,
        wvi_crop,
        int(label_id),
        intensity_frac=float(rg_intensity_frac),
        abs_floor=rg_abs_ceiling,
        forbidden=forbidden,
        polarity=BB_RG_POLARITY,
    )


def _region_grow_free(
    seg: np.ndarray,
    wvi: np.ndarray,
    label_id: int,
    *,
    rg_intensity_frac: float,
    rg_abs_ceiling: float | None,
    centerlines_mask: np.ndarray,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
) -> int:
    """Hypointense RG on the full volume (not limited by bbox, tube, or eICAB prior)."""
    forbidden = _forbidden_for_free_rg(
        seg,
        centerlines_mask,
        label_id,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
    )
    return region_grow_into_label_volume(
        seg,
        as_backend_array(wvi).astype(np.float64),
        int(label_id),
        intensity_frac=float(rg_intensity_frac),
        abs_floor=rg_abs_ceiling,
        forbidden=forbidden,
        polarity=BB_RG_POLARITY,
    )


def _region_grow_vessel(
    seg: np.ndarray,
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    grow_domain: np.ndarray,
    *,
    rg_constraint: bool,
    rg_intensity_frac: float,
    rg_abs_ceiling: float | None,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
) -> int:
    """Hypointense RG with optional bbox / eICAB / tube constraints."""
    if rg_constraint:
        forbidden = _forbidden_in_bbox(
            seg,
            centerlines_mask,
            label_id,
            bbox,
            cl_barrier_radius=cl_barrier_radius,
            rg_barrier_radius=rg_barrier_radius,
            grow_domain=grow_domain,
        )
        return _region_grow_in_bbox(
            seg,
            wvi,
            label_id,
            bbox,
            rg_intensity_frac=rg_intensity_frac,
            rg_abs_ceiling=rg_abs_ceiling,
            forbidden=forbidden,
        )
    return _region_grow_free(
        seg,
        wvi,
        label_id,
        rg_intensity_frac=rg_intensity_frac,
        rg_abs_ceiling=rg_abs_ceiling,
        centerlines_mask=centerlines_mask,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
    )


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def build_seg_bb(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    eicab_bb: np.ndarray,
    *,
    bbox_padding: int = 8,
    eicab_prior_dilate: int = 3,
    centerline_max_dist: int = 12,
    ica_centerline_max_dist: int = 6,
    lumen_intensity_frac: float = 1.2,
    lumen_percentile: float = 55.0,
    ica_lumen_intensity_frac: float = 1.05,
    ica_lumen_percentile: float = 42.0,
    rg_intensity_frac: float = 1.0,
    rg_constraint: bool = True,
    min_component_frac: float = 0.005,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 2,
) -> BbSegResult:
    """eICAB-guided hypointense lumen segmentation on native black-blood."""
    wvi_np = as_backend_array(wvi).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    eicab_np = as_backend_array(eicab_bb).astype(np.int32, copy=False)
    if tuple(clm.shape[:3]) != tuple(wvi_np.shape[:3]):
        raise ValueError("centerlines_mask shape must match wvi_bb")
    if tuple(eicab_np.shape[:3]) != tuple(wvi_np.shape[:3]):
        raise ValueError("eicab_bb shape must match wvi_bb")

    seg = np.zeros(clm.shape, dtype=np.int32)
    stats: list[dict[str, Any]] = []
    pad = max(0, int(bbox_padding))
    cl_rad = max(0, int(cl_barrier_radius))
    rg_rad = max(0, int(rg_barrier_radius))
    shape = tuple(int(s) for s in clm.shape[:3])
    label_ids = sorted(int(v) for v in np.unique(clm) if int(v) > 0)
    rg_mode = "constrained" if rg_constraint else "free"
    log.step(f"eICAB-guided lumen seg: {len(label_ids)} label(s), rg={rg_mode}")

    for lid in label_ids:
        tuning = _tuning_for_label(
            lid,
            eicab_prior_dilate=eicab_prior_dilate,
            centerline_max_dist=centerline_max_dist,
            ica_centerline_max_dist=ica_centerline_max_dist,
            lumen_intensity_frac=lumen_intensity_frac,
            lumen_percentile=lumen_percentile,
            ica_lumen_intensity_frac=ica_lumen_intensity_frac,
            ica_lumen_percentile=ica_lumen_percentile,
        )

        roi = clm == lid
        bbox = _bbox_with_symmetric_padding(roi, shape, padding=pad)
        if bbox is None:
            stats.append({"label_id": lid, "warning": "empty centerline", "n_voxels": 0})
            continue

        i0, i1, j0, j1, k0, k1 = bbox
        wvi_crop = wvi_np[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        cl_slab = clm[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        seed_slab = cl_slab == int(lid)
        eicab_prior = _eicab_prior_slab(
            eicab_np, lid, bbox, dilate_radius=tuning.eicab_prior_dilate
        )
        tube = _centerline_tube_mask(seed_slab, tuning.centerline_max_dist)
        grow_domain = eicab_prior & tube

        if not np.any(grow_domain):
            stats.append(
                {
                    "label_id": lid,
                    "warning": "empty grow domain",
                    "bbox": bbox,
                    "n_voxels": 0,
                }
            )
            continue

        seed_mean, seed_p25, seed_p75 = _seed_intensity_stats(wvi_np, clm, lid)
        ceiling = _hypointense_ceiling(
            wvi_crop,
            tube,
            seed_mean,
            seed_p75,
            lumen_intensity_frac=tuning.lumen_intensity_frac,
            lumen_percentile=tuning.lumen_percentile,
        )

        lumen_mask = _lumen_mask_in_bbox(
            wvi_crop,
            seed_slab,
            grow_domain,
            ceiling=ceiling,
            min_component_frac=min_component_frac,
        )

        cl_barrier = _dilated_other_centerlines_barrier(clm, lid, bbox, radius=cl_rad)
        n_thr = _paste_label_slab(seg, lumen_mask, lid, bbox, forbidden=cl_barrier)

        n_rg = _region_grow_vessel(
            seg,
            wvi_np,
            clm,
            lid,
            bbox,
            grow_domain,
            rg_constraint=rg_constraint,
            rg_intensity_frac=float(rg_intensity_frac),
            rg_abs_ceiling=ceiling,
            cl_barrier_radius=cl_rad,
            rg_barrier_radius=rg_rad,
        )

        stats.append(
            {
                "label_id": lid,
                "bbox": bbox,
                "rg_constraint": bool(rg_constraint),
                "tuning": {
                    "eicab_prior_dilate": tuning.eicab_prior_dilate,
                    "centerline_max_dist": tuning.centerline_max_dist,
                    "lumen_intensity_frac": tuning.lumen_intensity_frac,
                    "lumen_percentile": tuning.lumen_percentile,
                },
                "seed_mean": seed_mean,
                "seed_p25": seed_p25,
                "seed_p75": seed_p75,
                "ceiling": ceiling,
                "n_voxels_after_threshold": n_thr,
                "n_voxels_grown": n_rg,
                "n_voxels": int(np.count_nonzero(seg == lid)),
            }
        )
        log.step(
            f"{bb_vessel_name(lid)} (id={lid}): thr={n_thr} +rg={n_rg} "
            f"(total={int(np.count_nonzero(seg == lid))}, ceiling={ceiling:.1f}, "
            f"tube_r={tuning.centerline_max_dist})"
        )

    return BbSegResult(seg=seg, stats=stats)


def run_bb_segmentation(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    eicab_bb: np.ndarray,
    out_dir: Path,
    *,
    bbox_padding: int = 8,
    eicab_prior_dilate: int = 3,
    centerline_max_dist: int = 12,
    ica_centerline_max_dist: int = 6,
    lumen_intensity_frac: float = 1.2,
    lumen_percentile: float = 55.0,
    ica_lumen_intensity_frac: float = 1.05,
    ica_lumen_percentile: float = 42.0,
    rg_intensity_frac: float = 1.0,
    rg_constraint: bool = True,
    min_component_frac: float = 0.005,
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

    log.step(f"BB segmentation | strategy={SEG_STRATEGY}")
    result = build_seg_bb(
        wvi,
        centerlines_mask,
        eicab_bb,
        bbox_padding=bbox_padding,
        eicab_prior_dilate=eicab_prior_dilate,
        centerline_max_dist=centerline_max_dist,
        ica_centerline_max_dist=ica_centerline_max_dist,
        lumen_intensity_frac=lumen_intensity_frac,
        lumen_percentile=lumen_percentile,
        ica_lumen_intensity_frac=ica_lumen_intensity_frac,
        ica_lumen_percentile=ica_lumen_percentile,
        rg_intensity_frac=rg_intensity_frac,
        rg_constraint=rg_constraint,
        min_component_frac=min_component_frac,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
    )

    from nvitk.io.imageio import imsave

    md = dict(metadata or {})
    imsave(seg_path, result.seg, metadata=md)
    log.step(f"wrote {seg_path.name}")
    meta: dict[str, Any] = {
        "strategy": SEG_STRATEGY,
        "rg_polarity": BB_RG_POLARITY,
        "bbox_padding": bbox_padding,
        "eicab_prior_dilate": eicab_prior_dilate,
        "centerline_max_dist": centerline_max_dist,
        "ica_centerline_max_dist": ica_centerline_max_dist,
        "lumen_intensity_frac": lumen_intensity_frac,
        "lumen_percentile": lumen_percentile,
        "ica_lumen_intensity_frac": ica_lumen_intensity_frac,
        "ica_lumen_percentile": ica_lumen_percentile,
        "rg_intensity_frac": rg_intensity_frac,
        "rg_constraint": rg_constraint,
        "min_component_frac": min_component_frac,
        "cl_barrier_radius": cl_barrier_radius,
        "rg_barrier_radius": rg_barrier_radius,
        "vessel_stats": result.stats,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return seg_path
