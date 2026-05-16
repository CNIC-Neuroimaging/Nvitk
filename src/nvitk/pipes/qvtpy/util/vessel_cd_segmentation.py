"""Per-vessel local CD crop, threshold, and optional region growing for stage 4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.qvtpy.util.flow_volume_masks import _binary_mask_sliding_threshold

setup(globals())

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]
_CROP_MIN_COMPONENT_FRAC = 0.005


@dataclass
class VesselSegStats:
    """Per-label segmentation statistics."""

    label_id: int
    bbox: tuple[int, int, int, int, int, int]
    thr_algorithm: str
    opt_thresh: float | None
    n_voxels_after_threshold: int
    n_voxels_after_region_growing: int
    warning: str | None = None


@dataclass
class LocalSegResult:
    """``seg_4dflow`` volume and per-vessel metadata."""

    segmentation: np.ndarray
    vessel_stats: list[VesselSegStats] = field(default_factory=list)


def _bbox_with_padding(
    roi: np.ndarray,
    shape: tuple[int, int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Return ``(i0, i1, j0, j1, k0, k1)`` inclusive max indices, or None if empty."""
    m = as_backend_array(roi.astype(bool, copy=False))
    if not np.any(m):
        return None
    xs, ys, zs = np.nonzero(m)
    pad = max(0, int(padding))
    nx, ny, nz = shape
    i0 = max(0, int(xs.min()) - pad)
    i1 = min(nx - 1, int(xs.max()) + pad)
    j0 = max(0, int(ys.min()) - pad)
    j1 = min(ny - 1, int(ys.max()) + pad)
    k0 = max(0, int(zs.min()) - pad)
    k1 = min(nz - 1, int(zs.max()) + pad)
    return i0, i1, j0, j1, k0, k1


def _threshold_crop(
    cd_crop: np.ndarray,
    algorithm: ThrAlgorithm,
) -> tuple[np.ndarray, float | None, str | None]:
    """Binary mask on *cd_crop*. Returns ``(mask, opt_thresh, warning)``."""
    cd_crop = as_backend_array(cd_crop).astype(np.float64)
    warn: str | None = None
    if algorithm == "otsu":
        pos = cd_crop[cd_crop > 0]
        if pos.size < 2:
            return np.zeros(cd_crop.shape, dtype=bool), None, "otsu: insufficient foreground"
        try:
            from skimage.filters import threshold_otsu
        except ImportError as exc:
            raise ImportError("otsu requires scikit-image") from exc
        try:
            t = float(threshold_otsu(pos))
        except ValueError as exc:
            return np.zeros(cd_crop.shape, dtype=bool), None, f"otsu failed: {exc}"
        mask = (cd_crop > t).astype(bool, copy=False)
        mask = remove_small_components_by_fraction(
            mask,
            min_fraction=_CROP_MIN_COMPONENT_FRAC,
            connectivity=1,
        )
        return as_backend_array(mask).astype(bool), t, warn

    shift_hm = algorithm == "lthr"
    mask, opt_thresh = _binary_mask_sliding_threshold(
        cd_crop,
        shift_hm_flag=shift_hm,
        med_filt_flag=True,
    )
    mask = remove_small_components_by_fraction(
        mask,
        min_fraction=_CROP_MIN_COMPONENT_FRAC,
        connectivity=1,
    )
    return as_backend_array(mask).astype(bool), float(opt_thresh), warn


def _paste_crop_mask(
    seg: np.ndarray,
    crop_mask: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
) -> int:
    """Write *crop_mask* into *seg* at *bbox* where ``seg == 0``. Returns voxel count."""
    i0, i1, j0, j1, k0, k1 = bbox
    slab = seg[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
    m = as_backend_array(crop_mask.astype(bool, copy=False))
    free = slab == 0
    write = m & free
    n = int(np.count_nonzero(write))
    if n > 0:
        slab[write] = int(label_id)
    return n


def _region_grow_vessel(
    seg: np.ndarray,
    cd: np.ndarray,
    label_id: int,
    *,
    rg_intensity_frac: float,
    rg_abs_floor: float | None,
) -> int:
    """6-connected region growing for *label_id* into voxels with ``seg == 0``."""
    seg_np = as_backend_array(seg)
    cd_np = as_backend_array(cd).astype(np.float64)
    seeds = np.argwhere(seg_np == int(label_id))
    if seeds.size == 0:
        return 0

    seed_vals = cd_np[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    seed_mean = float(np.mean(seed_vals))
    floor = float(rg_abs_floor) if rg_abs_floor is not None else 0.0
    grow_thresh = max(seed_mean * float(rg_intensity_frac), floor)

    nx, ny, nz = seg_np.shape
    q: deque[tuple[int, int, int]] = deque(
        (int(i), int(j), int(k)) for i, j, k in seeds
    )
    seen = set(q)
    n_added = 0

    while q:
        i, j, k = q.popleft()
        for di, dj, dk in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue
            if (ni, nj, nk) in seen:
                continue
            seen.add((ni, nj, nk))
            if seg_np[ni, nj, nk] != 0:
                continue
            if cd_np[ni, nj, nk] < grow_thresh:
                continue
            seg_np[ni, nj, nk] = int(label_id)
            n_added += 1
            q.append((ni, nj, nk))

    return n_added


def build_seg_4dflow_local(
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    crop_padding_bbox: int = 0,
    thr_algorithm: ThrAlgorithm = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = 0.5,
) -> LocalSegResult:
    """Build multilabel ``seg_4dflow`` from CD and per-label centerline backbone."""
    cd = as_backend_array(cd).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in clm.shape[:3])
    seg = np.zeros(shape, dtype=np.int32)

    label_ids = sorted(int(v) for v in np.unique(clm) if int(v) > 0)
    stats: list[VesselSegStats] = []
    opt_thresh_by_label: dict[int, float | None] = {}

    for lid in label_ids:
        roi = clm == lid
        bbox = _bbox_with_padding(roi, shape, padding=crop_padding_bbox)
        if bbox is None:
            stats.append(
                VesselSegStats(
                    label_id=lid,
                    bbox=(0, 0, 0, 0, 0, 0),
                    thr_algorithm=thr_algorithm,
                    opt_thresh=None,
                    n_voxels_after_threshold=0,
                    n_voxels_after_region_growing=0,
                    warning="empty centerline mask for label",
                )
            )
            opt_thresh_by_label[lid] = None
            continue

        i0, i1, j0, j1, k0, k1 = bbox
        cd_crop = cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        crop_mask, opt_t, warn = _threshold_crop(cd_crop, thr_algorithm)
        opt_thresh_by_label[lid] = opt_t
        n_thr = _paste_crop_mask(seg, crop_mask, lid, bbox)

        stats.append(
            VesselSegStats(
                label_id=lid,
                bbox=bbox,
                thr_algorithm=thr_algorithm,
                opt_thresh=opt_t,
                n_voxels_after_threshold=n_thr,
                n_voxels_after_region_growing=n_thr,
                warning=warn,
            )
        )

    if region_growing:
        for st in stats:
            lid = st.label_id
            floor = opt_thresh_by_label.get(lid)
            _region_grow_vessel(
                seg,
                cd,
                lid,
                rg_intensity_frac=rg_intensity_frac,
                rg_abs_floor=floor,
            )
            st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))

    return LocalSegResult(
        segmentation=as_backend_array(seg.astype(np.int32, copy=False)),
        vessel_stats=stats,
    )


def vessel_stats_to_dict(st: VesselSegStats) -> dict[str, Any]:
    i0, i1, j0, j1, k0, k1 = st.bbox
    return {
        "label_id": st.label_id,
        "bbox": {
            "i0": i0,
            "i1": i1,
            "j0": j0,
            "j1": j1,
            "k0": k0,
            "k1": k1,
        },
        "thr_algorithm": st.thr_algorithm,
        "opt_thresh": st.opt_thresh,
        "n_voxels_after_threshold": st.n_voxels_after_threshold,
        "n_voxels_after_region_growing": st.n_voxels_after_region_growing,
        "warning": st.warning,
    }


__all__ = [
    "LocalSegResult",
    "ThrAlgorithm",
    "VesselSegStats",
    "build_seg_4dflow_local",
    "vessel_stats_to_dict",
]
