"""Per-vessel local CD crop, threshold, and optional region growing for stage 4.

**Inputs**

- 3D complex-difference volume, stage-3 ``centerlines_mask``, optional warped eICAB labels.

**Outputs**

- Multilabel ``seg_4dflow`` via :func:`build_seg_4dflow_local` and per-vessel :class:`VesselSegStats`.

Array indices ``(i, j, k)`` are **(X, Y, Z)** for asymmetric bbox padding.

Region-growing intensity gate (per vessel ``L``)::

    grow_thresh = max(mean(CD on seg==L) * rg_intensity_frac(L), opt_thresh_local)

A **lower** ``rg_intensity_frac`` admits dimmer neighbours (more growth). MCA/ACA/PCA use
a reduced explore fraction by default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger
from nvitk.morphology import dilate
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected, remove_small_components_by_fraction
from nvitk.pipes.qvtpy.labels import (
    EICAB_RG_BARRIER_LABEL_IDS,
    QVTPY_ACA_IDS,
    QVTPY_ACOMM,
    QVTPY_BASILAR,
    QVTPY_COMM_IDS,
    QVTPY_ICA_BASILAR_IDS,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_NON_COMM_ARTERIAL_IDS,
    QVTPY_PCA_IDS,
    QVTPY_RACA,
    QVTPY_RICA,
    QVTPY_RG_EXPLORE_MORE_IDS,
    QVTPY_RG_INTENSITY_FRAC_VENOUS,
    QVTPY_RG_MCA_PCA_EXPLORE_IDS,
    QVTPY_RG_PCA_BASILAR_EICAB_BARRIER_IDS,
    QVTPY_RG_PCOMM_EICAB_BARRIER_IDS,
    QVTPY_RG_SKIP_LABEL_IDS,
    QVTPY_RMCA,
    QVTPY_SMALL_ARTERIAL_IDS,
    QVTPY_VENOUS_LABEL_IDS,
    QVTPY_VERTEBRAL_IDS,
)
from nvitk.pipes.qvtpy.util.vertebral_split import VertebralSplitResult, split_vertebral_from_basilar
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d
from nvitk.pipes.qvtpy.util.mask_cleaning import keep_largest_component_label_inplace
from nvitk.segmentation.region_growing import merge_forbidden, region_grow_into_label_volume

setup(globals())

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Types and defaults
# ──────────────────────────────────────────────────────────────────────────────

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]
_CROP_MIN_COMPONENT_FRAC = 0.005
_CROP_MIN_COMPONENT_FRAC_SMALL = 0.0
VESSEL_EXTRA_PADDING: int = 10
_DEFAULT_RG_INTENSITY_FRAC: float = 0.45
_RG_INTENSITY_FRAC_EXPLORE: float = 0.25
_RG_INTENSITY_FRAC_ACA: float = 0.25
# Communicating arteries: stricter (higher) gate + volume caps to curb the
# persistent "grow into almost the whole image" failure.
_RG_INTENSITY_FRAC_COMM: float = 0.60
_RG_EXPLORE_SEG_BARRIER_RADIUS: int = 1
_RG_EXPLORE_CL_BARRIER_RADIUS: int = 3
_EICAB_RG_BARRIER_RADIUS: int = 1
_ACOMM_JUNCTION_RADIUS_DEFAULT: int = 10
_ACA_OVERLAP_MIN_VOXELS_DEFAULT: int = 5

# Region-growing volume safeguards (reject catastrophic over-grow / neighbour bleed).
# A grow is rolled back when the label exceeds either cap after BFS.
_RG_MAX_GROW_FRAC_DEFAULT: float = 15.0  # grown voxels vs. pre-grow seed voxels
_RG_MAX_IMAGE_FRAC_DEFAULT: float = 0.05  # label voxels vs. whole volume
_RG_MAX_IMAGE_FRAC_EXPLORE: float = 0.02  # stricter cap for MCA/PCA (leak into basilar/ICA)
_RG_MAX_IMAGE_FRAC_COMM: float = 0.02  # stricter cap for communicating arteries
_RG_MIN_GROW_ALLOW_VOXELS: int = 2000  # always allow at least this many grown voxels
# Neighbour labels dilated as extra RG walls so PCA/PComm growth cannot slip into
# the basilar, ICA, or the opposite communicating artery through the P1/junction gap.
_RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS: int = 3


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


@dataclass(frozen=True)
class AcaSequentialGrowInfo:
    """Metadata for sequential ACA grow + optional AComm convergence correction."""

    strategy: str
    grow_order: tuple[int, ...]
    n_overlap_voxels: int
    overlap_correction_applied: bool
    acomm_junction_ijk: tuple[int, int, int] | None = None
    acomm_junction_radius_vox: int = 0
    n_convergence_voxels_corrected: int = 0
    n_stray_islands_reassigned: int = 0
    n_voronoi_voxels_reassigned: int = 0
    junction_source: str | None = None
    split_axis: str | None = None
    split_coord: int | None = None
    laca_on_low_side: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "grow_order": [int(x) for x in self.grow_order],
            "n_overlap_voxels": int(self.n_overlap_voxels),
            "overlap_correction_applied": bool(self.overlap_correction_applied),
            "acomm_junction_ijk": (
                [int(self.acomm_junction_ijk[0]), int(self.acomm_junction_ijk[1]), int(self.acomm_junction_ijk[2])]
                if self.acomm_junction_ijk is not None
                else None
            ),
            "acomm_junction_radius_vox": int(self.acomm_junction_radius_vox),
            "n_convergence_voxels_corrected": int(self.n_convergence_voxels_corrected),
            "n_stray_islands_reassigned": int(self.n_stray_islands_reassigned),
            "n_voronoi_voxels_reassigned": int(self.n_voronoi_voxels_reassigned),
            "junction_source": self.junction_source,
            "split_axis": self.split_axis,
            "split_coord": self.split_coord,
            "laca_on_low_side": self.laca_on_low_side,
        }


@dataclass
class LocalSegResult:
    """``seg_4dflow`` volume and per-vessel metadata."""

    segmentation: np.ndarray
    vessel_stats: list[VesselSegStats] = field(default_factory=list)
    aca_sequential_grow: AcaSequentialGrowInfo | None = None
    vertebral_split: VertebralSplitResult | None = None


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


def rg_abs_floor_for_label(
    label_id: int,
    opt_thresh: float | None,
) -> float | None:
    """Intensity floor for RG; MCA/PCA use fraction-only gate (no local threshold floor).

    ACA uses the local CD threshold as a hard floor so dim parenchyma / venous
    spill cannot fill when A1 seed mean alone would admit a huge bright blob.
    """
    if int(label_id) in QVTPY_RG_MCA_PCA_EXPLORE_IDS:
        return None
    return opt_thresh


def rg_intensity_frac_for_label(
    label_id: int,
    *,
    default_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    explore_frac: float = _RG_INTENSITY_FRAC_EXPLORE,
    aca_frac: float = _RG_INTENSITY_FRAC_ACA,
    venous_fracs: dict[int, float] | None = None,
) -> float:
    """Per-vessel RG intensity factor (lower → more growth)."""
    lid = int(label_id)
    venous = venous_fracs if venous_fracs is not None else QVTPY_RG_INTENSITY_FRAC_VENOUS
    if lid in venous:
        return float(venous[lid])
    if lid in QVTPY_ACA_IDS:
        return float(aca_frac)
    if lid in QVTPY_RG_MCA_PCA_EXPLORE_IDS:
        return float(explore_frac)
    return float(default_frac)


def eicab_dropped_label_barrier(
    eicab_native: np.ndarray,
    *,
    label_ids: frozenset[int] | None = None,
    radius_vox: int = _EICAB_RG_BARRIER_RADIUS,
) -> np.ndarray:
    """Forbidden mask from native eICAB ids omitted at qvtpy relabel (default LSCA/RSCA = 15/16)."""
    vol = as_backend_array(eicab_native).astype(np.int32, copy=False)
    ids = EICAB_RG_BARRIER_LABEL_IDS if label_ids is None else label_ids
    barrier = np.zeros(vol.shape, dtype=bool)
    for eid in ids:
        barrier |= as_backend_array(vol == int(eid)).astype(bool)
    return _dilate_bool_mask(barrier, radius=int(radius_vox))


def region_growing_enabled_for_label(label_id: int) -> bool:
    """Whether region growing runs for this label (all venous and comm are skipped)."""
    lid = int(label_id)
    if lid in QVTPY_RG_SKIP_LABEL_IDS or lid in QVTPY_COMM_IDS:
        return False
    return True


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
        # Symmetric padding on all axes: each ACA centerline can be thin in i or j.
        return BboxFacePadding(d, d, d, d, d, d)
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
    m = as_backend_array(mask).astype(bool)
    if radius <= 0 or not np.any(m):
        return m
    return as_backend_array(dilate(m.astype(np.uint8), footprint=int(radius), connectivity=1)).astype(bool)


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
    other = as_backend_array((clm != 0) & (clm != int(label_id))).astype(bool)
    other = _dilate_bool_mask(other, radius=radius)
    return other[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]


def _dilated_other_centerlines_barrier_full(
    centerlines_mask: np.ndarray,
    label_id: int,
    *,
    radius: int,
    exclude_label_ids: frozenset[int] | None = None,
) -> np.ndarray:
    """Full-volume forbidden mask: dilated centerlines of other vessels."""
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    lid = int(label_id)
    extra = exclude_label_ids or frozenset()
    other = np.zeros(clm.shape, dtype=bool)
    for oid in np.unique(clm):
        oid_int = int(oid)
        if oid_int == 0 or oid_int == lid or oid_int in extra:
            continue
        other |= clm == oid_int
    return _dilate_bool_mask(other, radius=int(radius))


def explore_region_grow_forbidden(
    seg: np.ndarray,
    centerlines_mask: np.ndarray,
    label_id: int,
    *,
    seg_barrier_radius: int = _RG_EXPLORE_SEG_BARRIER_RADIUS,
    cl_barrier_radius: int = _RG_EXPLORE_CL_BARRIER_RADIUS,
    exclude_label_ids: frozenset[int] | None = None,
    eicab_native: np.ndarray | None = None,
    eicab_barrier_radius: int = _EICAB_RG_BARRIER_RADIUS,
) -> np.ndarray:
    """Combined explore-group RG barriers: other seg @ *seg_barrier_radius*, other CL @ *cl_barrier_radius*."""
    lid = int(label_id)
    exclude = exclude_label_ids or frozenset()
    seg_forb = _dilated_other_segmentation_barrier_excluding(
        seg,
        lid,
        exclude_label_ids=exclude,
        radius=int(seg_barrier_radius),
    )
    cl_forb = _dilated_other_centerlines_barrier_full(
        centerlines_mask,
        lid,
        radius=int(cl_barrier_radius),
        exclude_label_ids=exclude,
    )
    merged = merge_forbidden(seg_forb, cl_forb)
    if eicab_native is not None and lid in QVTPY_RG_PCA_BASILAR_EICAB_BARRIER_IDS:
        merged = merge_forbidden(
            merged,
            eicab_dropped_label_barrier(
                eicab_native,
                radius_vox=int(eicab_barrier_radius),
            ),
        )
    assert merged is not None
    return merged


def _dilated_other_segmentation_barrier(
    seg: np.ndarray,
    label_id: int,
    *,
    radius: int,
) -> np.ndarray:
    """Full-volume forbidden mask: dilated voxels labeled as other vessels."""
    return _dilated_other_segmentation_barrier_excluding(
        seg, label_id, exclude_label_ids=frozenset(), radius=radius
    )


def _dilated_other_segmentation_barrier_excluding(
    seg: np.ndarray,
    label_id: int,
    *,
    exclude_label_ids: frozenset[int],
    radius: int,
) -> np.ndarray:
    """Dilated other-vessel mask; labels in *exclude_label_ids* are not treated as barriers."""
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    other = np.zeros(seg_np.shape, dtype=bool)
    for other_id in np.unique(seg_np):
        oid = int(other_id)
        if oid == 0 or oid == int(label_id) or oid in exclude_label_ids:
            continue
        other |= as_backend_array(seg_np == oid).astype(bool)
    if not np.any(other):
        return other
    return _dilate_bool_mask(other, radius=radius)




def _merge_forbidden(*masks: np.ndarray | None) -> np.ndarray | None:
    merged: np.ndarray | None = None
    for m in masks:
        if m is None:
            continue
        b = as_backend_array(m).astype(bool)
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
            with using("cpu"):
                t = float(threshold_otsu(to_numpy(pos)))
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
    mask, opt_thresh = binary_mask_sliding_threshold_3d(
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
    return region_grow_into_label_volume(
        seg,
        cd,
        int(label_id),
        intensity_frac=float(rg_intensity_frac),
        abs_floor=rg_abs_floor,
        forbidden=forbidden,
    )


def rg_caps_exceeded(
    n_pre: int,
    n_post: int,
    n_total: int,
    *,
    max_grow_frac: float | None,
    max_image_frac: float | None,
    min_grow_allow_voxels: int = _RG_MIN_GROW_ALLOW_VOXELS,
) -> str | None:
    """Return a rollback reason when a grown label breaches a volume cap, else None."""
    if max_image_frac is not None and n_total > 0:
        if n_post > float(max_image_frac) * float(n_total):
            return (
                f"rg rollback: {n_post} vox > {float(max_image_frac):.3f} of image "
                f"({n_total} vox)"
            )
    if (
        max_grow_frac is not None
        and n_pre > 0
        and n_post > max(float(max_grow_frac) * float(n_pre), float(min_grow_allow_voxels))
    ):
        return f"rg rollback: {n_post} vox > {float(max_grow_frac):.1f}x seed ({n_pre} vox)"
    return None


def _region_grow_vessel_capped(
    seg: np.ndarray,
    cd: np.ndarray,
    label_id: int,
    *,
    rg_intensity_frac: float,
    rg_abs_floor: float | None,
    forbidden: np.ndarray | None = None,
    max_grow_frac: float | None,
    max_image_frac: float | None,
) -> tuple[int, str | None]:
    """Region grow *label_id*; roll back added voxels if a volume cap is breached.

    Returns ``(n_label_voxels, rollback_reason)`` where the reason is ``None`` when
    the grow was kept.
    """
    seg_np = as_backend_array(seg)
    lid = int(label_id)
    pre_mask = as_backend_array(seg_np == lid).astype(bool, copy=True)
    n_pre = int(np.count_nonzero(pre_mask))
    region_grow_into_label_volume(
        seg_np,
        cd,
        lid,
        intensity_frac=float(rg_intensity_frac),
        abs_floor=rg_abs_floor,
        forbidden=forbidden,
    )
    post_mask = as_backend_array(seg_np == lid).astype(bool)
    n_post = int(np.count_nonzero(post_mask))
    reason = rg_caps_exceeded(
        n_pre,
        n_post,
        int(post_mask.size),
        max_grow_frac=max_grow_frac,
        max_image_frac=max_image_frac,
    )
    if reason is not None:
        added = post_mask & ~pre_mask
        seg_np[added] = 0
        log.warning(f"label {lid}: {reason}; kept {n_pre} seed voxels")
        return n_pre, reason
    return n_post, None


def _other_segmentation_barrier_undilated(seg: np.ndarray, label_id: int) -> np.ndarray:
    """Forbidden mask: any other label (no dilation)."""
    return _dilated_other_segmentation_barrier(seg, label_id, radius=0)


def _dilated_labels_barrier(
    seg: np.ndarray,
    label_ids: frozenset[int] | set[int] | tuple[int, ...],
    *,
    radius: int = 1,
) -> np.ndarray:
    """Dilated forbidden mask from a specific set of already-segmented label ids."""
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    barrier = np.zeros(seg_np.shape, dtype=bool)
    for lid in label_ids:
        barrier |= as_backend_array(seg_np == int(lid)).astype(bool)
    return _dilate_bool_mask(barrier, radius=int(radius))


def _dilated_pca_barrier(seg: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Dilated mask of already-segmented PCA voxels (wall for PComm growth)."""
    return _dilated_labels_barrier(seg, QVTPY_PCA_IDS, radius=int(radius))


def _segment_communicating_rg_only(
    seg: np.ndarray,
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    comm_label_ids: list[int],
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    stats_by_id: dict[int, VesselSegStats],
    eicab_native: np.ndarray | None = None,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    max_grow_frac: float | None = _RG_MAX_GROW_FRAC_DEFAULT,
    max_image_frac: float | None = _RG_MAX_IMAGE_FRAC_COMM,
) -> None:
    """Seed comm arteries (centerline + local threshold) and region-grow with walls.

    Posterior communicating arteries additionally use the registered eICAB SCA
    (native ids 15/16) and dilated PCA segmentation as walls, and every comm grow
    is bounded by :func:`rg_caps_exceeded` (stricter image-fraction cap).
    """
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in clm.shape[:3])
    sca_barrier = (
        None if eicab_native is None else eicab_dropped_label_barrier(eicab_native)
    )
    for lid in comm_label_ids:
        if not np.any(clm == lid):
            continue
        seg[clm == lid] = int(lid)
        # Densify seeds via a local CD threshold crop; centerline-only seeds are
        # too sparse to anchor a reliable intensity reference for the RG gate.
        roi = clm == lid
        bbox_out = _bbox_with_vessel_padding(roi, shape, lid, default_pad=crop_padding_bbox)
        if bbox_out is not None:
            bbox, _fp = bbox_out
            i0, i1, j0, j1, k0, k1 = bbox
            cd_crop = cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
            crop_mask, _opt, _warn = _threshold_crop(
                cd_crop,
                thr_algorithm,
                min_component_frac=crop_min_fraction_for_label(lid),
            )
            cl_barrier = _dilated_other_centerlines_barrier(
                clm, lid, bbox, radius=_RG_EXPLORE_CL_BARRIER_RADIUS
            )
            _paste_crop_mask(seg, crop_mask, lid, bbox, forbidden=cl_barrier)
        frac = _RG_INTENSITY_FRAC_COMM
        # Wall comm growth off from every other label (dilated, so it cannot creep
        # one voxel at a time along ICA/MCA/PCA). PComm additionally uses the eICAB
        # SCA territory, the dilated PCA, and the ICA/basilar trunks as walls so it
        # stops at the PCA junction instead of leaking into ICA/PCA/basilar.
        forbidden = _dilated_other_segmentation_barrier(seg, lid, radius=1)
        if lid in QVTPY_RG_PCOMM_EICAB_BARRIER_IDS:
            forbidden = merge_forbidden(
                forbidden,
                sca_barrier,
                _dilated_pca_barrier(seg),
                _dilated_labels_barrier(seg, QVTPY_ICA_BASILAR_IDS, radius=1),
            )
        n_lbl, reason = _region_grow_vessel_capped(
            seg,
            cd,
            lid,
            rg_intensity_frac=frac,
            rg_abs_floor=None,
            forbidden=forbidden,
            max_grow_frac=max_grow_frac,
            max_image_frac=max_image_frac,
        )
        st = stats_by_id.get(lid)
        if st is not None:
            st.thr_algorithm = "centerline_rg_only"
            st.region_growing_applied = True
            st.rg_intensity_frac_used = float(frac)
            st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
            st.n_voxels_after_threshold = int(np.count_nonzero(seg == lid))
            st.n_voxels_after_island_clean = st.n_voxels_after_region_growing
            if reason is not None:
                st.warning = reason
        _finalize_vessel_largest_cc(
            seg, lid, stats_by_id=stats_by_id, log_name=f"comm-{lid}"
        )


# ---------------------------------------------------------------------------
# Build seg_4dflow (per-label local threshold + region growing)
# ---------------------------------------------------------------------------


def _finalize_vessel_largest_cc(
    seg: np.ndarray,
    label_id: int,
    *,
    stats_by_id: dict[int, VesselSegStats] | None = None,
    log_name: str | None = None,
) -> int:
    """Keep largest CC for one label; optionally refresh stats and log drops."""
    lid = int(label_id)
    n_before = int(np.count_nonzero(seg == lid))
    n_after = keep_largest_component_label_inplace(seg, lid)
    dropped = n_before - n_after
    if dropped > 0 and log_name is not None:
        log.info(
            f"largest-CC {log_name} (label {lid}): {n_before} → {n_after} "
            f"(dropped {dropped} island voxel(s))"
        )
    if stats_by_id is not None:
        st = stats_by_id.get(lid)
        if st is not None:
            st.n_voxels_after_region_growing = int(n_after)
            st.n_voxels_after_island_clean = int(n_after)
    return n_after


def build_seg_4dflow_local(
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    eicab_qvtpy: np.ndarray | None = None,
    eicab_native: np.ndarray | None = None,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    rg_intensity_frac_aca: float = _RG_INTENSITY_FRAC_ACA,
    venous_rg_intensity_fracs: dict[int, float] | None = None,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    rg_max_grow_frac: float | None = _RG_MAX_GROW_FRAC_DEFAULT,
    rg_max_image_frac: float | None = _RG_MAX_IMAGE_FRAC_DEFAULT,
    venous_region_growing: bool = True,
    segment_acomm: bool = False,
    split_vertebral: bool = False,
) -> LocalSegResult:
    """Build multilabel ``seg_4dflow`` from CD and per-label centerline backbone."""
    cd = as_backend_array(cd).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    eicab = (
        None
        if eicab_qvtpy is None
        else as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    )
    eicab_ids = (
        None
        if eicab_native is None
        else as_backend_array(eicab_native).astype(np.int32, copy=False)
    )
    shape = tuple(int(s) for s in clm.shape[:3])
    seg = np.zeros(shape, dtype=np.int32)

    label_ids = sorted(int(v) for v in np.unique(clm) if int(v) > 0)
    phase1_ids = [lid for lid in label_ids if lid not in QVTPY_COMM_IDS]
    comm_ids = [lid for lid in label_ids if lid in QVTPY_COMM_IDS]
    if not segment_acomm:
        # AComm is used only to inform the ACA L/R junction; never grown as a mask.
        comm_ids = [lid for lid in comm_ids if lid != int(QVTPY_ACOMM)]
    log.step(
        f"local CD segmentation: {len(phase1_ids)} peripheral + {len(comm_ids)} "
        f"communicating label(s), thr={thr_algorithm}, RG={region_growing}"
    )
    stats: list[VesselSegStats] = []
    opt_thresh_by_label: dict[int, float | None] = {}
    cl_rad = max(0, int(cl_barrier_radius))
    rg_rad = max(0, int(rg_barrier_radius))
    venous_rg = resolve_venous_rg_intensity_fracs(venous_rg_intensity_fracs)

    for lid in phase1_ids:
        log.step(f"label {lid}: crop + {thr_algorithm} threshold")
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
        if lid in QVTPY_ACA_IDS:
            n_cl = int(np.count_nonzero(roi))
            log.info(
                f"ACA label {lid} threshold crop: cl_vox={n_cl}, "
                f"bbox=({i0}:{i1}, {j0}:{j1}, {k0}:{k1}) "
                f"size=({i1 - i0 + 1}x{j1 - j0 + 1}x{k1 - k0 + 1}), "
                f"face_pad={face_pad.as_dict()}"
            )
        cd_crop = cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
        crop_mask, opt_t, warn = _threshold_crop(
            cd_crop,
            thr_algorithm,
            min_component_frac=crop_min_fraction_for_label(lid),
        )
        opt_thresh_by_label[lid] = opt_t
        paste_cl_rad = (
            _RG_EXPLORE_CL_BARRIER_RADIUS
            if lid in QVTPY_RG_EXPLORE_MORE_IDS
            else cl_rad
        )
        cl_barrier = _dilated_other_centerlines_barrier(
            clm, lid, bbox, radius=paste_cl_rad
        )
        n_thr = _paste_crop_mask(seg, crop_mask, lid, bbox, forbidden=cl_barrier)
        if lid in QVTPY_ACA_IDS:
            n_crop = int(np.count_nonzero(crop_mask))
            n_bar = int(np.count_nonzero(cl_barrier))
            log.info(
                f"ACA label {lid} after threshold paste: "
                f"crop_fg={n_crop}, pasted={n_thr}, "
                f"cl_barrier_in_bbox={n_bar}, opt_thresh={opt_t}, warn={warn}"
            )

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

    # Largest-CC after local threshold (before RG); ACAs included so stub islands go.
    for lid in sorted(int(v) for v in np.unique(seg) if int(v) != 0):
        keep_largest_component_label_inplace(seg, int(lid))
    for st in stats:
        st.n_voxels_after_island_clean = int(np.count_nonzero(seg == st.label_id))

    aca_sequential_info: AcaSequentialGrowInfo | None = None
    stats_by_id = {st.label_id: st for st in stats}
    aca_present = [
        int(lid)
        for lid in (int(QVTPY_LACA), int(QVTPY_RACA))
        if int(lid) in stats_by_id and region_growing_enabled_for_label(int(lid))
    ]
    use_aca_sequential = bool(aca_sequential_grow) and len(aca_present) > 0

    if region_growing:
        if use_aca_sequential:
            from nvitk.pipes.qvtpy.util.aca_sequential_grow import _region_grow_acas_sequential

            log.step(
                "region growing: ACA sequential path enabled "
                f"(present={aca_present})"
            )
            aca_sequential_info = _region_grow_acas_sequential(
                seg,
                cd,
                clm,
                eicab,
                eicab_native=eicab_ids,
                opt_thresh_by_label=opt_thresh_by_label,
                rg_intensity_frac=rg_intensity_frac,
                rg_intensity_frac_aca=rg_intensity_frac_aca,
                venous_fracs=venous_rg,
                rg_barrier_radius=rg_rad,
                aca_overlap_min_voxels=aca_overlap_min_voxels,
                acomm_junction_radius=acomm_junction_radius,
                stats_by_id=stats_by_id,
                max_grow_frac=rg_max_grow_frac,
                max_image_frac=rg_max_image_frac,
            )
            for lid in aca_present:
                st = stats_by_id.get(lid)
                if st is None:
                    continue
                log.info(
                    f"ACA label {lid} stats after sequential RG: "
                    f"thr={st.n_voxels_after_threshold}, "
                    f"island={st.n_voxels_after_island_clean}, "
                    f"rg={st.n_voxels_after_region_growing}, "
                    f"frac={st.rg_intensity_frac_used}, warn={st.warning}"
                )
            for lid in (int(QVTPY_LACA), int(QVTPY_RACA)):
                if int(np.count_nonzero(seg == lid)) == 0:
                    continue
                _finalize_vessel_largest_cc(
                    seg, lid, stats_by_id=stats_by_id, log_name=f"ACA-{lid}"
                )

        for st in stats:
            lid = st.label_id
            if lid in QVTPY_COMM_IDS:
                continue
            if use_aca_sequential and lid in QVTPY_ACA_IDS:
                continue
            if lid not in QVTPY_NON_COMM_ARTERIAL_IDS:
                st.region_growing_applied = False
                st.rg_intensity_frac_used = None
                st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
                continue
            if not region_growing_enabled_for_label(lid):
                st.region_growing_applied = False
                st.rg_intensity_frac_used = None
                st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
                continue

            frac = rg_intensity_frac_for_label(
                lid,
                default_frac=rg_intensity_frac,
                explore_frac=rg_intensity_frac_explore,
                aca_frac=rg_intensity_frac_aca,
                venous_fracs=venous_rg,
            )
            floor = rg_abs_floor_for_label(lid, opt_thresh_by_label.get(lid))
            label_max_image_frac = rg_max_image_frac
            if lid in QVTPY_RG_MCA_PCA_EXPLORE_IDS or lid == int(QVTPY_BASILAR):
                rg_forbidden = explore_region_grow_forbidden(
                    seg,
                    clm,
                    lid,
                    eicab_native=eicab_ids,
                )
                # Explore vessels (MCA/PCA) and basilar leak most easily into their
                # large neighbours; add the neighbouring segmented labels as dilated
                # walls and tighten the image-fraction cap.
                if rg_max_image_frac is not None:
                    label_max_image_frac = min(
                        float(rg_max_image_frac), _RG_MAX_IMAGE_FRAC_EXPLORE
                    )
                if lid in QVTPY_PCA_IDS:
                    neighbour_ids = (
                        (QVTPY_ICA_BASILAR_IDS | QVTPY_COMM_IDS) - {int(lid)}
                    )
                    rg_forbidden = merge_forbidden(
                        rg_forbidden,
                        _dilated_labels_barrier(
                            seg,
                            frozenset(int(x) for x in neighbour_ids),
                            radius=_RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS,
                        ),
                    )
                elif lid == int(QVTPY_BASILAR):
                    # ICA trunks (ICA/basilar group minus the basilar itself) + PCA.
                    basilar_neighbours = QVTPY_PCA_IDS | (
                        QVTPY_ICA_BASILAR_IDS - {int(QVTPY_BASILAR)}
                    )
                    rg_forbidden = merge_forbidden(
                        rg_forbidden,
                        _dilated_labels_barrier(
                            seg,
                            frozenset(int(x) for x in basilar_neighbours),
                            radius=_RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS,
                        ),
                    )
            elif lid in QVTPY_ACA_IDS:
                continue
            elif lid in QVTPY_RG_EXPLORE_MORE_IDS:
                rg_forbidden = explore_region_grow_forbidden(seg, clm, lid)
            elif int(lid) in (QVTPY_LICA, QVTPY_RICA):
                rg_forbidden = merge_forbidden(
                    _dilated_other_segmentation_barrier(seg, lid, radius=rg_rad),
                    _dilated_labels_barrier(
                        seg,
                        QVTPY_COMM_IDS | QVTPY_PCA_IDS,
                        radius=_RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS,
                    ),
                )
            else:
                rg_forbidden = _dilated_other_segmentation_barrier(seg, lid, radius=rg_rad)

            _n_lbl, reason = _region_grow_vessel_capped(
                seg,
                cd,
                lid,
                rg_intensity_frac=frac,
                rg_abs_floor=floor,
                forbidden=rg_forbidden,
                max_grow_frac=rg_max_grow_frac,
                max_image_frac=label_max_image_frac,
            )
            st.region_growing_applied = True
            st.rg_intensity_frac_used = float(frac)
            st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
            if reason is not None:
                st.warning = reason
            _finalize_vessel_largest_cc(
                seg, lid, stats_by_id=stats_by_id, log_name=f"vessel-{lid}"
            )

    vertebral_info: VertebralSplitResult | None = None
    if (
        split_vertebral
        and int(QVTPY_BASILAR) in stats_by_id
        and np.any(seg == int(QVTPY_BASILAR))
    ):
        seg, vertebral_info = split_vertebral_from_basilar(seg)
        for lid in (int(QVTPY_BASILAR), *sorted(int(x) for x in QVTPY_VERTEBRAL_IDS)):
            if int(np.count_nonzero(seg == lid)) == 0:
                continue
            _finalize_vessel_largest_cc(
                seg, lid, stats_by_id=stats_by_id, log_name=f"vertebral-{lid}"
            )

    if region_growing and comm_ids:
        for lid in comm_ids:
            fp = bbox_padding_for_label(lid, crop_padding_bbox)
            stats.append(
                VesselSegStats(
                    label_id=lid,
                    bbox=(0, 0, 0, 0, 0, 0),
                    face_padding=fp,
                    thr_algorithm="centerline_rg_only",
                    opt_thresh=None,
                    n_voxels_after_threshold=0,
                    n_voxels_after_island_clean=0,
                    n_voxels_after_region_growing=0,
                    region_growing_applied=False,
                )
            )
            stats_by_id[lid] = stats[-1]
        _segment_communicating_rg_only(
            seg,
            cd,
            clm,
            comm_label_ids=comm_ids,
            rg_intensity_frac=rg_intensity_frac,
            rg_intensity_frac_explore=rg_intensity_frac_explore,
            stats_by_id=stats_by_id,
            eicab_native=eicab_ids,
            crop_padding_bbox=int(crop_padding_bbox),
            thr_algorithm=thr_algorithm,
            max_grow_frac=rg_max_grow_frac,
            max_image_frac=_RG_MAX_IMAGE_FRAC_COMM,
        )

    # Final per-label largest-CC: drop isolated voxels left on other vessel regions.
    log.step("final per-vessel largest connected component")
    for lid in sorted(int(v) for v in np.unique(seg) if int(v) != 0):
        _finalize_vessel_largest_cc(
            seg,
            lid,
            stats_by_id=stats_by_id,
            log_name=f"final-{lid}",
        )

    return LocalSegResult(
        segmentation=as_backend_array(seg.astype(np.int32, copy=False)),
        vessel_stats=stats,
        aca_sequential_grow=aca_sequential_info,
        vertebral_split=vertebral_info,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def vessel_stats_to_dict(st: VesselSegStats) -> dict[str, Any]:
    """JSON-serializable dict from :class:`VesselSegStats`."""
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
    "AcaSequentialGrowInfo",
    "_ACOMM_JUNCTION_RADIUS_DEFAULT",
    "BboxFacePadding",
    "LocalSegResult",
    "ThrAlgorithm",
    "VESSEL_EXTRA_PADDING",
    "VesselSegStats",
    "_ACA_OVERLAP_MIN_VOXELS_DEFAULT",
    "_DEFAULT_RG_INTENSITY_FRAC",
    "_RG_INTENSITY_FRAC_EXPLORE",
    "_RG_INTENSITY_FRAC_ACA",
    "_RG_INTENSITY_FRAC_COMM",
    "_RG_EXPLORE_SEG_BARRIER_RADIUS",
    "_RG_EXPLORE_CL_BARRIER_RADIUS",
    "_RG_MAX_GROW_FRAC_DEFAULT",
    "_RG_MAX_IMAGE_FRAC_DEFAULT",
    "_RG_MAX_IMAGE_FRAC_EXPLORE",
    "_RG_MAX_IMAGE_FRAC_COMM",
    "explore_region_grow_forbidden",
    "bbox_padding_for_label",
    "build_seg_4dflow_local",
    "crop_min_fraction_for_label",
    "rg_caps_exceeded",
    "region_growing_enabled_for_label",
    "resolve_venous_rg_intensity_fracs",
    "rg_abs_floor_for_label",
    "rg_intensity_frac_for_label",
    "vessel_stats_to_dict",
    "_bbox_with_padding",
]
