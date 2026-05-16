"""Per-vessel local CD crop, threshold, and optional region growing for stage 4.

Array indices ``(i, j, k)`` are treated as **(X, Y, Z)** for asymmetric bbox padding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology import dilate
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ACA_IDS,
    QVTPY_ICA_BASILAR_IDS,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_RMCA,
)
from nvitk.pipes.qvtpy.util.flow_volume_masks import _binary_mask_sliding_threshold
from nvitk.pipes.qvtpy.util.mask_cleaning import clean_multilabel_islands

setup(globals())

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]
_CROP_MIN_COMPONENT_FRAC = 0.005
VESSEL_EXTRA_PADDING: int = 10


@dataclass(frozen=True)
class BboxFacePadding:
    """Per-face bbox expansion in voxels (toward lower / higher index on each axis)."""

    pad_i_min: int
    pad_i_max: int
    pad_j_min: int
    pad_j_max: int
    pad_k_min: int
    pad_k_max: int

    def as_dict(self) -> dict[str, int]:
        return {
            "pad_i_min": self.pad_i_min,
            "pad_i_max": self.pad_i_max,
            "pad_j_min": self.pad_j_min,
            "pad_j_max": self.pad_j_max,
            "pad_k_min": self.pad_k_min,
            "pad_k_max": self.pad_k_max,
        }


@dataclass
class VesselSegStats:
    """Per-label segmentation statistics."""

    label_id: int
    bbox: tuple[int, int, int, int, int, int]
    face_padding: BboxFacePadding
    thr_algorithm: str
    opt_thresh: float | None
    n_voxels_after_threshold: int
    n_voxels_after_island_clean: int
    n_voxels_after_region_growing: int
    warning: str | None = None


@dataclass
class LocalSegResult:
    """``seg_4dflow`` volume and per-vessel metadata."""

    segmentation: np.ndarray
    vessel_stats: list[VesselSegStats] = field(default_factory=list)


def bbox_padding_for_label(label_id: int, default_pad: int) -> BboxFacePadding:
    """Per-vessel asymmetric padding around the centerline ROI bbox."""
    d = max(0, int(default_pad))
    extra = int(VESSEL_EXTRA_PADDING)
    lid = int(label_id)

    if lid in QVTPY_ICA_BASILAR_IDS:
        return BboxFacePadding(d, d, d, d, d, 0)
    if lid == QVTPY_LMCA:
        return BboxFacePadding(0, extra, d, d, d, d)
    if lid == QVTPY_RMCA:
        return BboxFacePadding(extra, 0, d, d, d, d)
    if lid in QVTPY_ACA_IDS:
        return BboxFacePadding(0, 0, extra, d, d, d)
    return BboxFacePadding(d, d, d, d, d, d)


def _bbox_with_vessel_padding(
    roi: np.ndarray,
    shape: tuple[int, int, int],
    label_id: int,
    *,
    default_pad: int,
) -> tuple[tuple[int, int, int, int, int, int], BboxFacePadding] | None:
    """Return ``(i0, i1, j0, j1, k0, k1)`` and face padding, or None if empty."""
    m = as_backend_array(roi.astype(bool, copy=False))
    if not np.any(m):
        return None
    xs, ys, zs = np.nonzero(m)
    fp = bbox_padding_for_label(label_id, default_pad)
    nx, ny, nz = shape
    i0 = max(0, int(xs.min()) - fp.pad_i_min)
    i1 = min(nx - 1, int(xs.max()) + fp.pad_i_max)
    j0 = max(0, int(ys.min()) - fp.pad_j_min)
    j1 = min(ny - 1, int(ys.max()) + fp.pad_j_max)
    k0 = max(0, int(zs.min()) - fp.pad_k_min)
    k1 = min(nz - 1, int(zs.max()) + fp.pad_k_max)
    return (i0, i1, j0, j1, k0, k1), fp


def _bbox_with_padding(
    roi: np.ndarray,
    shape: tuple[int, int, int],
    *,
    padding: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Symmetric bbox padding (legacy helper / tests)."""
    out = _bbox_with_vessel_padding(roi, shape, QVTPY_LICA, default_pad=padding)
    return None if out is None else out[0]


def _dilated_other_centerlines_barrier(
    centerlines_mask: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int, int, int],
    *,
    radius: int,
) -> np.ndarray:
    """Forbidden slab (bool) inside *bbox*: dilated other-vessel centerlines."""
    i0, i1, j0, j1, k0, k1 = bbox
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    other = np.asarray((clm != 0) & (clm != int(label_id)), dtype=bool)
    if radius > 0:
        other = np.asarray(
            as_backend_array(
                dilate(other.astype(np.uint8), footprint=int(radius), connectivity=1)
            ),
            dtype=bool,
        )
    return other[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]


def _dilated_other_segmentation_barrier(
    seg: np.ndarray,
    label_id: int,
    *,
    radius: int,
) -> np.ndarray:
    """Full-volume forbidden mask: dilated voxels labeled as other vessels."""
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    other = np.asarray((seg_np != 0) & (seg_np != int(label_id)), dtype=bool)
    if not np.any(other):
        return np.zeros(seg_np.shape, dtype=bool)
    if radius > 0:
        other = np.asarray(
            as_backend_array(
                dilate(other.astype(np.uint8), footprint=int(radius), connectivity=1)
            ),
            dtype=bool,
        )
    return other


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
    *,
    forbidden: np.ndarray | None = None,
) -> int:
    """Write *crop_mask* into *seg* at *bbox* where ``seg == 0`` and not forbidden."""
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


def _region_grow_vessel(
    seg: np.ndarray,
    cd: np.ndarray,
    label_id: int,
    *,
    rg_intensity_frac: float,
    rg_abs_floor: float | None,
    forbidden: np.ndarray | None = None,
) -> int:
    """6-connected region growing into unlabeled, non-forbidden voxels."""
    seg_np = as_backend_array(seg)
    cd_np = as_backend_array(cd).astype(np.float64)
    forb = None if forbidden is None else as_backend_array(forbidden).astype(bool, copy=False)
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
            if forb is not None and forb[ni, nj, nk]:
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
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "otsu",
    region_growing: bool = True,
    rg_intensity_frac: float = 0.45,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    seg_min_island_fraction: float = 0.05,
    seg_bridge_open_radius: int = 0,
) -> LocalSegResult:
    """Build multilabel ``seg_4dflow`` from CD and per-label centerline backbone."""
    cd = as_backend_array(cd).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in clm.shape[:3])
    seg = np.zeros(shape, dtype=np.int32)

    label_ids = sorted(int(v) for v in np.unique(clm) if int(v) > 0)
    stats: list[VesselSegStats] = []
    opt_thresh_by_label: dict[int, float | None] = {}
    cl_rad = max(0, int(cl_barrier_radius))
    rg_rad = max(0, int(rg_barrier_radius))

    for lid in label_ids:
        roi = clm == lid
        bbox_out = _bbox_with_vessel_padding(roi, shape, lid, default_pad=crop_padding_bbox)
        if bbox_out is None:
            fp_empty = bbox_padding_for_label(lid, crop_padding_bbox)
            stats.append(
                VesselSegStats(
                    label_id=lid,
                    bbox=(0, 0, 0, 0, 0, 0),
                    face_padding=fp_empty,
                    thr_algorithm=thr_algorithm,
                    opt_thresh=None,
                    n_voxels_after_threshold=0,
                    n_voxels_after_island_clean=0,
                    n_voxels_after_region_growing=0,
                    warning="empty centerline mask for label",
                )
            )
            opt_thresh_by_label[lid] = None
            continue

        bbox, face_pad = bbox_out
        i0, i1, j0, j1, k0, k1 = bbox
        cd_crop = cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        crop_mask, opt_t, warn = _threshold_crop(cd_crop, thr_algorithm)
        opt_thresh_by_label[lid] = opt_t
        cl_barrier = _dilated_other_centerlines_barrier(
            clm, lid, bbox, radius=cl_rad
        )
        n_thr = _paste_crop_mask(seg, crop_mask, lid, bbox, forbidden=cl_barrier)

        stats.append(
            VesselSegStats(
                label_id=lid,
                bbox=bbox,
                face_padding=face_pad,
                thr_algorithm=thr_algorithm,
                opt_thresh=opt_t,
                n_voxels_after_threshold=n_thr,
                n_voxels_after_island_clean=n_thr,
                n_voxels_after_region_growing=n_thr,
                warning=warn,
            )
        )

    seg = clean_multilabel_islands(
        seg,
        min_fraction=float(seg_min_island_fraction),
        bridge_open_radius=int(seg_bridge_open_radius),
    )
    for st in stats:
        st.n_voxels_after_island_clean = int(np.count_nonzero(seg == st.label_id))

    if region_growing:
        for st in stats:
            lid = st.label_id
            floor = opt_thresh_by_label.get(lid)
            rg_forbidden = _dilated_other_segmentation_barrier(seg, lid, radius=rg_rad)
            _region_grow_vessel(
                seg,
                cd,
                lid,
                rg_intensity_frac=rg_intensity_frac,
                rg_abs_floor=floor,
                forbidden=rg_forbidden,
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
        "face_padding": st.face_padding.as_dict(),
        "thr_algorithm": st.thr_algorithm,
        "opt_thresh": st.opt_thresh,
        "n_voxels_after_threshold": st.n_voxels_after_threshold,
        "n_voxels_after_island_clean": st.n_voxels_after_island_clean,
        "n_voxels_after_region_growing": st.n_voxels_after_region_growing,
        "warning": st.warning,
    }


__all__ = [
    "BboxFacePadding",
    "LocalSegResult",
    "ThrAlgorithm",
    "VESSEL_EXTRA_PADDING",
    "VesselSegStats",
    "bbox_padding_for_label",
    "build_seg_4dflow_local",
    "vessel_stats_to_dict",
    "_bbox_with_padding",
]
