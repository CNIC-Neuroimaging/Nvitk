"""Black-blood artery segmentation: crop-resegment and centerline-growth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from nvitk.core.array import as_backend_array
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d
from nvitk.morphology.binary import dilate
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.segmentation.region_growing import region_grow_into_label_volume

ThrAlgorithm = Literal["otsu", "lsthr", "lthr"]
SegStrategy = Literal["crop-resegment", "centerline-growth"]

SEG_BB_NIFTI = "seg_bb.nii.gz"
SEGMENTATION_META_JSON = "segmentation_meta.json"


@dataclass
class BbSegResult:
    seg: np.ndarray
    stats: list[dict[str, Any]]


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
    m = np.asarray(mask, dtype=bool)
    if radius <= 0 or not np.any(m):
        return m
    return np.asarray(
        as_backend_array(dilate(m.astype(np.uint8), footprint=int(radius), connectivity=1)),
        dtype=bool,
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
    other = np.asarray((clm != 0) & (clm != int(label_id)), dtype=bool)
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
        other |= np.asarray(seg_np == oid, dtype=bool)
    if not np.any(other):
        return other
    return _dilate_bool_mask(other, radius=radius)


def _threshold_crop(
    wvi_crop: np.ndarray,
    algorithm: ThrAlgorithm,
    *,
    min_component_frac: float,
) -> tuple[np.ndarray, float | None, str | None]:
    wvi_crop = as_backend_array(wvi_crop).astype(np.float64)
    min_frac = float(min_component_frac)
    if algorithm == "otsu":
        pos = wvi_crop[wvi_crop > 0]
        if pos.size < 2:
            return np.zeros(wvi_crop.shape, dtype=bool), None, "otsu: insufficient foreground"
        try:
            from skimage.filters import threshold_otsu
        except ImportError as exc:
            raise ImportError("otsu requires scikit-image") from exc
        try:
            t = float(threshold_otsu(pos))
        except ValueError as exc:
            return np.zeros(wvi_crop.shape, dtype=bool), None, f"otsu failed: {exc}"
        mask = (wvi_crop > t).astype(bool, copy=False)
        if min_frac > 0:
            mask = remove_small_components_by_fraction(
                mask, min_fraction=min_frac, connectivity=1
            )
        return as_backend_array(mask).astype(bool), t, None

    shift_hm = algorithm == "lthr"
    mask, opt_thresh = binary_mask_sliding_threshold_3d(
        wvi_crop,
        shift_hm_flag=shift_hm,
        med_filt_flag=True,
    )
    if min_frac > 0:
        mask = remove_small_components_by_fraction(
            mask, min_fraction=min_frac, connectivity=1
        )
    return as_backend_array(mask).astype(bool), float(opt_thresh), None


def _paste_crop_mask(
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


def build_seg_bb_crop_resegment(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    thr_algorithm: ThrAlgorithm = "otsu",
    crop_padding_bbox: int = 3,
    cl_barrier_radius: int = 2,
    min_component_frac: float = 0.005,
) -> BbSegResult:
    """Per-vessel bbox crop, threshold, paste with centerline barriers."""
    wvi_np = as_backend_array(wvi).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in clm.shape[:3])
    seg = np.zeros(shape, dtype=np.int32)
    stats: list[dict[str, Any]] = []
    cl_rad = max(0, int(cl_barrier_radius))

    for lid in sorted(int(v) for v in np.unique(clm) if int(v) > 0):
        roi = clm == lid
        bbox = _bbox_with_symmetric_padding(roi, shape, padding=crop_padding_bbox)
        if bbox is None:
            stats.append(
                {
                    "label_id": lid,
                    "n_voxels": 0,
                    "warning": "empty centerline mask for label",
                }
            )
            continue
        i0, i1, j0, j1, k0, k1 = bbox
        wvi_crop = wvi_np[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        crop_mask, opt_t, warn = _threshold_crop(
            wvi_crop, thr_algorithm, min_component_frac=min_component_frac
        )
        cl_barrier = _dilated_other_centerlines_barrier(
            clm, lid, bbox, radius=cl_rad
        )
        n = _paste_crop_mask(seg, crop_mask, lid, bbox, forbidden=cl_barrier)
        stats.append(
            {
                "label_id": lid,
                "bbox": bbox,
                "opt_thresh": opt_t,
                "n_voxels": n,
                "warning": warn,
            }
        )

    return BbSegResult(seg=seg, stats=stats)


def build_seg_bb_centerline_growth(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    rg_intensity_frac: float = 0.45,
    rg_abs_floor: float | None = None,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 2,
) -> BbSegResult:
    """Initialize seg from centerlines; region-grow per label with barriers."""
    wvi_np = as_backend_array(wvi).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    seg = clm.copy()
    stats: list[dict[str, Any]] = []
    cl_rad = max(0, int(cl_barrier_radius))
    rg_rad = max(0, int(rg_barrier_radius))

    for lid in sorted(int(v) for v in np.unique(clm) if int(v) > 0):
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
            abs_floor=rg_abs_floor,
            forbidden=forbidden,
        )
        stats.append({"label_id": lid, "n_voxels_grown": n})

    return BbSegResult(seg=seg, stats=stats)


def run_bb_segmentation(
    wvi: np.ndarray,
    centerlines_mask: np.ndarray,
    out_dir: Path,
    *,
    strategy: SegStrategy,
    thr_algorithm: ThrAlgorithm = "otsu",
    crop_padding_bbox: int = 3,
    cl_barrier_radius: int = 2,
    min_component_frac: float = 0.005,
    rg_intensity_frac: float = 0.45,
    rg_abs_floor: float | None = None,
    rg_barrier_radius: int = 2,
    skip_existing: bool = False,
) -> Path:
    """Write ``seg_bb.nii.gz`` and ``segmentation_meta.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / SEG_BB_NIFTI
    meta_path = out_dir / SEGMENTATION_META_JSON
    if skip_existing and seg_path.is_file() and meta_path.is_file():
        return seg_path

    if strategy == "crop-resegment":
        result = build_seg_bb_crop_resegment(
            wvi,
            centerlines_mask,
            thr_algorithm=thr_algorithm,
            crop_padding_bbox=crop_padding_bbox,
            cl_barrier_radius=cl_barrier_radius,
            min_component_frac=min_component_frac,
        )
    elif strategy == "centerline-growth":
        result = build_seg_bb_centerline_growth(
            wvi,
            centerlines_mask,
            rg_intensity_frac=rg_intensity_frac,
            rg_abs_floor=rg_abs_floor,
            cl_barrier_radius=cl_barrier_radius,
            rg_barrier_radius=rg_barrier_radius,
        )
    else:
        raise ValueError(f"Unknown seg strategy: {strategy!r}")

    from nvitk.io.imageio import imsave  # local import keeps module import-light

    imsave(seg_path, result.seg)
    meta: dict[str, Any] = {
        "strategy": strategy,
        "thr_algorithm": thr_algorithm if strategy == "crop-resegment" else None,
        "vessel_stats": result.stats,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return seg_path
