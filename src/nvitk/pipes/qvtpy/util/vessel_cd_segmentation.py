"""Per-vessel local CD crop, threshold, and optional region growing for stage 4.

Array indices ``(i, j, k)`` are treated as **(X, Y, Z)** for asymmetric bbox padding.

Region-growing intensity gate (per vessel ``L``)::

    grow_thresh = max(mean(CD on seg==L) * rg_intensity_frac(L), opt_thresh_local)

A **lower** ``rg_intensity_frac`` admits dimmer neighbours (more growth). A **higher**
value is stricter. MCA/ACA/PCA use a reduced default fraction to explore further.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.morphology import dilate
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ACA_IDS,
    QVTPY_ACOMM,
    QVTPY_ICA_BASILAR_IDS,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_RACA,
    QVTPY_RG_EXPLORE_MORE_IDS,
    QVTPY_RG_INTENSITY_FRAC_VENOUS,
    QVTPY_RG_SKIP_LABEL_IDS,
    QVTPY_RMCA,
    QVTPY_SMALL_ARTERIAL_IDS,
    QVTPY_VENOUS_LABEL_IDS,
)
from nvitk.pipes.qvtpy.util.flow_volume_masks import _binary_mask_sliding_threshold
from nvitk.pipes.qvtpy.util.mask_cleaning import keep_largest_component_per_label

setup(globals())

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]
_CROP_MIN_COMPONENT_FRAC = 0.005
_CROP_MIN_COMPONENT_FRAC_SMALL = 0.0
VESSEL_EXTRA_PADDING: int = 10
_DEFAULT_RG_INTENSITY_FRAC: float = 0.45
_RG_INTENSITY_FRAC_EXPLORE: float = 0.35
_ACOMM_BARRIER_RADIUS_DEFAULT: int = 2
_ACA_CONTRA_BARRIER_RADIUS_DEFAULT: int = 3


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
    region_growing_applied: bool = False
    rg_intensity_frac_used: float | None = None
    warning: str | None = None


@dataclass
class LocalSegResult:
    """``seg_4dflow`` volume and per-vessel metadata."""

    segmentation: np.ndarray
    vessel_stats: list[VesselSegStats] = field(default_factory=list)


def crop_min_fraction_for_label(label_id: int) -> float:
    """Min CC fraction inside the CD crop (0 for small comm/PCA vessels)."""
    if int(label_id) in QVTPY_SMALL_ARTERIAL_IDS:
        return _CROP_MIN_COMPONENT_FRAC_SMALL
    return _CROP_MIN_COMPONENT_FRAC


def resolve_venous_rg_intensity_fracs(
    overrides: dict[int, float] | None = None,
) -> dict[int, float]:
    """Merged per-sinus RG fractions (SSSV/LTSV/RTSV); STRV is never included."""
    out = dict(QVTPY_RG_INTENSITY_FRAC_VENOUS)
    if overrides:
        for lid, frac in overrides.items():
            if int(lid) in QVTPY_VENOUS_LABEL_IDS and int(lid) not in QVTPY_RG_SKIP_LABEL_IDS:
                out[int(lid)] = float(frac)
    return out


def rg_intensity_frac_for_label(
    label_id: int,
    *,
    default_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    explore_frac: float = _RG_INTENSITY_FRAC_EXPLORE,
    venous_fracs: dict[int, float] | None = None,
) -> float:
    """Per-vessel RG intensity factor (lower → more growth)."""
    lid = int(label_id)
    venous = venous_fracs if venous_fracs is not None else QVTPY_RG_INTENSITY_FRAC_VENOUS
    if lid in venous:
        return float(venous[lid])
    if lid in QVTPY_RG_EXPLORE_MORE_IDS:
        return float(explore_frac)
    return float(default_frac)


def region_growing_enabled_for_label(label_id: int) -> bool:
    """Whether region growing runs for this label (STRV is skipped)."""
    return int(label_id) not in QVTPY_RG_SKIP_LABEL_IDS


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
    """Forbidden slab (bool) inside *bbox*: dilated other-vessel centerlines."""
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
    """Full-volume forbidden mask: dilated voxels labeled as other vessels."""
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    other = np.asarray((seg_np != 0) & (seg_np != int(label_id)), dtype=bool)
    if not np.any(other):
        return np.zeros(seg_np.shape, dtype=bool)
    return _dilate_bool_mask(other, radius=radius)


def _aca_eicab_region_growing_barrier(
    eicab_qvtpy: np.ndarray,
    label_id: int,
    *,
    acomm_radius: int,
    contra_radius: int,
) -> np.ndarray:
    """Forbidden mask for ACA RG: dilated AComm + contralateral ACA from warped eICAB."""
    lid = int(label_id)
    if lid not in QVTPY_ACA_IDS:
        return np.zeros(eicab_qvtpy.shape, dtype=bool)
    eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    forb = np.zeros(eq.shape, dtype=bool)
    acomm = eq == int(QVTPY_ACOMM)
    if np.any(acomm):
        forb |= _dilate_bool_mask(acomm, radius=acomm_radius)
    contra_id = int(QVTPY_RACA) if lid == int(QVTPY_LACA) else int(QVTPY_LACA)
    contra = eq == contra_id
    if np.any(contra):
        forb |= _dilate_bool_mask(contra, radius=contra_radius)
    return forb


def _merge_forbidden(*masks: np.ndarray | None) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for m in masks:
        if m is None:
            continue
        b = np.asarray(m, dtype=bool)
        merged = b if merged is None else (merged | b)
    return merged


def _threshold_crop(
    cd_crop: np.ndarray,
    algorithm: ThrAlgorithm,
    *,
    min_component_frac: float,
) -> tuple[np.ndarray, float | None, str | None]:
    """Binary mask on *cd_crop*. Returns ``(mask, opt_thresh, warning)``."""
    cd_crop = as_backend_array(cd_crop).astype(np.float64)
    min_frac = float(min_component_frac)
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
        if min_frac > 0:
            mask = remove_small_components_by_fraction(
                mask,
                min_fraction=min_frac,
                connectivity=1,
            )
        return as_backend_array(mask).astype(bool), t, warn

    shift_hm = algorithm == "lthr"
    mask, opt_thresh = _binary_mask_sliding_threshold(
        cd_crop,
        shift_hm_flag=shift_hm,
        med_filt_flag=True,
    )
    if min_frac > 0:
        mask = remove_small_components_by_fraction(
            mask,
            min_fraction=min_frac,
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
    eicab_qvtpy: np.ndarray | None = None,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    venous_rg_intensity_fracs: dict[int, float] | None = None,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    acomm_barrier_radius: int = _ACOMM_BARRIER_RADIUS_DEFAULT,
    aca_contra_barrier_radius: int = _ACA_CONTRA_BARRIER_RADIUS_DEFAULT,
) -> LocalSegResult:
    """Build multilabel ``seg_4dflow`` from CD and per-label centerline backbone."""
    cd = as_backend_array(cd).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    eicab = (
        None
        if eicab_qvtpy is None
        else as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    )
    shape = tuple(int(s) for s in clm.shape[:3])
    seg = np.zeros(shape, dtype=np.int32)

    label_ids = sorted(int(v) for v in np.unique(clm) if int(v) > 0)
    stats: list[VesselSegStats] = []
    opt_thresh_by_label: dict[int, float | None] = {}
    cl_rad = max(0, int(cl_barrier_radius))
    rg_rad = max(0, int(rg_barrier_radius))
    venous_rg = resolve_venous_rg_intensity_fracs(venous_rg_intensity_fracs)

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
                    region_growing_applied=False,
                    warning="empty centerline mask for label",
                )
            )
            opt_thresh_by_label[lid] = None
            continue

        bbox, face_pad = bbox_out
        i0, i1, j0, j1, k0, k1 = bbox
        cd_crop = cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        crop_mask, opt_t, warn = _threshold_crop(
            cd_crop,
            thr_algorithm,
            min_component_frac=crop_min_fraction_for_label(lid),
        )
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
                region_growing_applied=False,
                warning=warn,
            )
        )

    seg = keep_largest_component_per_label(seg)
    for st in stats:
        st.n_voxels_after_island_clean = int(np.count_nonzero(seg == st.label_id))

    if region_growing:
        for st in stats:
            lid = st.label_id
            if not region_growing_enabled_for_label(lid):
                st.region_growing_applied = False
                st.rg_intensity_frac_used = None
                st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
                continue

            frac = rg_intensity_frac_for_label(
                lid,
                default_frac=rg_intensity_frac,
                explore_frac=rg_intensity_frac_explore,
                venous_fracs=venous_rg,
            )
            floor = opt_thresh_by_label.get(lid)
            rg_forbidden = _dilated_other_segmentation_barrier(seg, lid, radius=rg_rad)
            if eicab is not None and lid in QVTPY_ACA_IDS:
                aca_forb = _aca_eicab_region_growing_barrier(
                    eicab,
                    lid,
                    acomm_radius=acomm_barrier_radius,
                    contra_radius=aca_contra_barrier_radius,
                )
                rg_forbidden = _merge_forbidden(rg_forbidden, aca_forb)

            _region_grow_vessel(
                seg,
                cd,
                lid,
                rg_intensity_frac=frac,
                rg_abs_floor=floor,
                forbidden=rg_forbidden,
            )
            st.region_growing_applied = True
            st.rg_intensity_frac_used = float(frac)
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
        "n_voxels_after_largest_cc": st.n_voxels_after_island_clean,
        "n_voxels_after_island_clean": st.n_voxels_after_island_clean,
        "n_voxels_after_region_growing": st.n_voxels_after_region_growing,
        "region_growing_applied": st.region_growing_applied,
        "rg_intensity_frac_used": st.rg_intensity_frac_used,
        "warning": st.warning,
    }


__all__ = [
    "BboxFacePadding",
    "LocalSegResult",
    "ThrAlgorithm",
    "VESSEL_EXTRA_PADDING",
    "VesselSegStats",
    "_DEFAULT_RG_INTENSITY_FRAC",
    "_RG_INTENSITY_FRAC_EXPLORE",
    "bbox_padding_for_label",
    "build_seg_4dflow_local",
    "crop_min_fraction_for_label",
    "region_growing_enabled_for_label",
    "resolve_venous_rg_intensity_fracs",
    "rg_intensity_frac_for_label",
    "vessel_stats_to_dict",
    "_bbox_with_padding",
]
