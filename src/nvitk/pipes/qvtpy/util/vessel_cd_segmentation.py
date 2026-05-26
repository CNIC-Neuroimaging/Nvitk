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
from nvitk.core.backend import setup
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
    QVTPY_RACA,
    QVTPY_RG_EXPLORE_MORE_IDS,
    QVTPY_RG_INTENSITY_FRAC_VENOUS,
    QVTPY_RG_MCA_PCA_EXPLORE_IDS,
    QVTPY_RG_PCA_BASILAR_EICAB_BARRIER_IDS,
    QVTPY_RG_SKIP_LABEL_IDS,
    QVTPY_RMCA,
    QVTPY_SMALL_ARTERIAL_IDS,
    QVTPY_VENOUS_LABEL_IDS,
)
from nvitk.pipes.qvtpy.util.vertebral_split import VertebralSplitResult, split_vertebral_from_basilar
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d
from nvitk.pipes.qvtpy.util.mask_cleaning import keep_largest_component_per_label
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
_RG_INTENSITY_FRAC_ACA: float = 0.35
_RG_EXPLORE_SEG_BARRIER_RADIUS: int = 1
_RG_EXPLORE_CL_BARRIER_RADIUS: int = 3
_EICAB_RG_BARRIER_RADIUS: int = 1
_ACOMM_JUNCTION_RADIUS_DEFAULT: int = 10
_ACA_OVERLAP_MIN_VOXELS_DEFAULT: int = 5


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
    grow_order: tuple[int, int]
    n_overlap_voxels: int
    overlap_correction_applied: bool
    acomm_junction_ijk: tuple[int, int, int] | None = None
    acomm_junction_radius_vox: int = 0
    n_convergence_voxels_corrected: int = 0
    n_stray_islands_reassigned: int = 0
    junction_source: str | None = None
    split_axis: str | None = None
    split_coord: int | None = None
    laca_on_low_side: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "grow_order": [int(self.grow_order[0]), int(self.grow_order[1])],
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
    """Intensity floor for RG; MCA/PCA use fraction-only gate (no local threshold floor)."""
    if int(label_id) in QVTPY_RG_MCA_PCA_EXPLORE_IDS:
        return None
    if int(label_id) in QVTPY_ACA_IDS:
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
        barrier |= np.asarray(vol == int(eid), dtype=bool)
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
        other |= np.asarray(seg_np == oid, dtype=bool)
    if not np.any(other):
        return other
    return _dilate_bool_mask(other, radius=radius)




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


def _other_segmentation_barrier_undilated(seg: np.ndarray, label_id: int) -> np.ndarray:
    """Forbidden mask: any other label (no dilation)."""
    return _dilated_other_segmentation_barrier(seg, label_id, radius=0)


def _segment_communicating_rg_only(
    seg: np.ndarray,
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    comm_label_ids: list[int],
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    stats_by_id: dict[int, VesselSegStats],
) -> None:
    """Seed comm arteries from centerlines and region-grow with undilated barriers."""
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    for lid in comm_label_ids:
        if not np.any(clm == lid):
            continue
        seg[clm == lid] = int(lid)
        frac = rg_intensity_frac_for_label(
            lid,
            default_frac=rg_intensity_frac,
            explore_frac=rg_intensity_frac_explore,
            venous_fracs=None,
        )
        forbidden = _other_segmentation_barrier_undilated(seg, lid)
        _region_grow_vessel(
            seg,
            cd,
            lid,
            rg_intensity_frac=frac,
            rg_abs_floor=None,
            forbidden=forbidden,
        )
        st = stats_by_id.get(lid)
        if st is not None:
            st.thr_algorithm = "centerline_rg_only"
            st.region_growing_applied = True
            st.rg_intensity_frac_used = float(frac)
            st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
            st.n_voxels_after_threshold = int(np.count_nonzero(seg == lid))
            st.n_voxels_after_island_clean = st.n_voxels_after_region_growing


# ---------------------------------------------------------------------------
# Build seg_4dflow (per-label local threshold + region growing)
# ---------------------------------------------------------------------------


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

    aca_sequential_info: AcaSequentialGrowInfo | None = None
    stats_by_id = {st.label_id: st for st in stats}
    use_aca_sequential = (
        bool(aca_sequential_grow)
        and int(QVTPY_LACA) in stats_by_id
        and int(QVTPY_RACA) in stats_by_id
        and region_growing_enabled_for_label(QVTPY_LACA)
        and region_growing_enabled_for_label(QVTPY_RACA)
    )

    if region_growing:
        if use_aca_sequential:
            from nvitk.pipes.qvtpy.util.aca_sequential_grow import _region_grow_acas_sequential

            aca_sequential_info = _region_grow_acas_sequential(
                seg,
                cd,
                clm,
                eicab,
                opt_thresh_by_label=opt_thresh_by_label,
                rg_intensity_frac=rg_intensity_frac,
                rg_intensity_frac_aca=rg_intensity_frac_aca,
                venous_fracs=venous_rg,
                rg_barrier_radius=rg_rad,
                aca_overlap_min_voxels=aca_overlap_min_voxels,
                acomm_junction_radius=acomm_junction_radius,
                stats_by_id=stats_by_id,
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
            if lid in QVTPY_RG_MCA_PCA_EXPLORE_IDS or lid == int(QVTPY_BASILAR):
                rg_forbidden = explore_region_grow_forbidden(
                    seg,
                    clm,
                    lid,
                    eicab_native=eicab_ids,
                )
            elif lid in QVTPY_ACA_IDS:
                continue
            elif lid in QVTPY_RG_EXPLORE_MORE_IDS:
                rg_forbidden = explore_region_grow_forbidden(seg, clm, lid)
            else:
                rg_forbidden = _dilated_other_segmentation_barrier(seg, lid, radius=rg_rad)

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

    vertebral_info: VertebralSplitResult | None = None
    if int(QVTPY_BASILAR) in stats_by_id and np.any(seg == int(QVTPY_BASILAR)):
        seg, vertebral_info = split_vertebral_from_basilar(seg)
        for lid in (QVTPY_BASILAR,):
            st = stats_by_id.get(int(lid))
            if st is not None:
                st.n_voxels_after_region_growing = int(np.count_nonzero(seg == lid))
                st.n_voxels_after_island_clean = st.n_voxels_after_region_growing

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
    "_RG_EXPLORE_SEG_BARRIER_RADIUS",
    "_RG_EXPLORE_CL_BARRIER_RADIUS",
    "explore_region_grow_forbidden",
    "bbox_padding_for_label",
    "build_seg_4dflow_local",
    "crop_min_fraction_for_label",
    "region_growing_enabled_for_label",
    "resolve_venous_rg_intensity_fracs",
    "rg_abs_floor_for_label",
    "rg_intensity_frac_for_label",
    "vessel_stats_to_dict",
    "_bbox_with_padding",
]
