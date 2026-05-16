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

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology import dilate
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected, remove_small_components_by_fraction
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
_ACOMM_JUNCTION_RADIUS_DEFAULT: int = 10
_ACA_OVERLAP_MIN_VOXELS_DEFAULT: int = 5
_ACA_SPLIT_AXIS_NAMES: tuple[str, str] = ("x", "y")


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


def _aca_seed_mask(
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    label_id: int,
) -> tuple[np.ndarray, str]:
    """Thin ACA seeds: stage-3 centerline primary, skeletonized eICAB fallback."""
    lid = int(label_id)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    seeds = np.asarray(clm == lid, dtype=bool)
    if np.any(seeds):
        return seeds, "centerline"
    if eicab_qvtpy is None:
        return seeds, "none"
    eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    raw = np.asarray(eq == lid, dtype=bool)
    if not np.any(raw):
        return seeds, "none"
    try:
        skel = skeletonize_binary(raw)
        skel_np = np.asarray(as_backend_array(skel), dtype=bool)
        if np.any(skel_np):
            return skel_np, "eicab_skeleton"
    except Exception:
        pass
    return raw, "eicab_mask"


def _acomm_junction_voxel(eicab_qvtpy: np.ndarray) -> tuple[int, int, int] | None:
    """Single-voxel AComm junction snapped to nearest eICAB AComm foreground."""
    eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    coords = np.argwhere(eq == int(QVTPY_ACOMM))
    if coords.size == 0:
        return None
    coords_np = np.asarray(to_numpy(coords), dtype=np.float64)
    com = coords_np.mean(axis=0)
    d2 = np.sum((coords_np - com) ** 2, axis=1)
    best = coords_np[int(np.argmin(d2))]
    return (int(best[0]), int(best[1]), int(best[2]))


def _infer_aca_junction(
    laca_seeds: np.ndarray,
    raca_seeds: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
) -> tuple[tuple[int, int, int] | None, str | None]:
    """Junction at eICAB AComm COM, else midpoint of closest ACA centerline approach."""
    if eicab_qvtpy is not None:
        j = _acomm_junction_voxel(eicab_qvtpy)
        if j is not None:
            return j, "eicab_acomm"
    lc = np.asarray(np.argwhere(laca_seeds), dtype=np.float64)
    rc = np.asarray(np.argwhere(raca_seeds), dtype=np.float64)
    if lc.size == 0 or rc.size == 0:
        return None, None
    d2 = np.sum((lc[:, None, :] - rc[None, :, :]) ** 2, axis=2)
    li, ri = np.unravel_index(int(np.argmin(d2)), d2.shape)
    mid = 0.5 * (lc[int(li)] + rc[int(ri)])
    nx, ny, nz = laca_seeds.shape
    return (
        (
            int(max(0, min(nx - 1, round(mid[0])))),
            int(max(0, min(ny - 1, round(mid[1])))),
            int(max(0, min(nz - 1, round(mid[2])))),
        ),
        "aca_midpoint",
    )


def _infer_aca_split_plane(
    laca_seeds: np.ndarray,
    raca_seeds: np.ndarray,
    junction: tuple[int, int, int],
) -> tuple[int, int, bool]:
    """Dominant in-plane axis and whether LACA lies on the low-index side of *junction*."""
    lc = np.asarray(np.argwhere(laca_seeds), dtype=np.float64)
    rc = np.asarray(np.argwhere(raca_seeds), dtype=np.float64)
    ji = int(junction[0])
    jj = int(junction[1])
    if lc.size == 0 or rc.size == 0:
        return 0, ji, True
    lm = lc.mean(axis=0)
    rm = rc.mean(axis=0)
    delta = lm - rm
    if abs(float(delta[0])) >= abs(float(delta[1])):
        axis = 0
        split_coord = ji
        laca_low = bool(lm[0] < rm[0])
    else:
        axis = 1
        split_coord = jj
        laca_low = bool(lm[1] < rm[1])
    return axis, split_coord, laca_low


def _aca_axis_unit_vector(seeds: np.ndarray) -> np.ndarray | None:
    """Principal axis of ACA seeds (robust when centerline stops short of junction)."""
    coords = np.asarray(np.argwhere(seeds), dtype=np.float64)
    if coords.shape[0] < 2:
        return None
    c = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    axis = np.asarray(vh[0], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return None
    return axis / n


def _aca_arm_from_junction(
    seeds: np.ndarray,
    junction: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    """Unit vector along ACA centerline axis; endpoint is farthest seed from junction."""
    coords = np.asarray(np.argwhere(seeds), dtype=np.float64)
    if coords.size == 0:
        return np.zeros(3, dtype=np.float64), None
    axis = _aca_axis_unit_vector(seeds)
    if axis is None:
        return np.zeros(3, dtype=np.float64), None
    j = np.asarray(junction, dtype=np.float64)
    d2 = np.sum((coords - j) ** 2, axis=1)
    idx = int(np.argmax(d2))
    endpoint = coords[idx]
    ep = (int(endpoint[0]), int(endpoint[1]), int(endpoint[2]))
    return axis.astype(np.float64), ep


def _distance_from_junction_voxels(
    shape: tuple[int, int, int],
    junction: tuple[int, int, int],
) -> np.ndarray:
    """EDT distance (vox) from the AComm junction seed voxel."""
    ji, jj, jk = (int(junction[0]), int(junction[1]), int(junction[2]))
    junc = np.zeros(shape, dtype=bool)
    junc[ji, jj, jk] = True
    return np.asarray(to_numpy(ndi.distance_transform_edt(~junc)), dtype=np.float32)


@dataclass(frozen=True)
class _AcaPlaneSplitResult:
    laca_mask: np.ndarray
    raca_mask: np.ndarray
    junction: tuple[int, int, int] | None
    junction_source: str | None
    split_axis: int
    split_coord: int
    laca_on_low_side: bool
    n_voxels_split: int
    n_stray_islands_reassigned: int


def _aca_seed_connected_keep_mask(
    vessel_mask: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (voxels in seed-touching CCs, stray CC voxels with no seed on that label)."""
    vessel = np.asarray(vessel_mask, dtype=bool)
    seeds = np.asarray(seed_mask, dtype=bool)
    if not np.any(vessel):
        return vessel.copy(), np.zeros_like(vessel, dtype=bool)

    labeled, n_cc = label_connected(vessel, connectivity=1)
    if n_cc <= 0:
        return vessel.copy(), np.zeros_like(vessel, dtype=bool)

    labeled_np = np.asarray(labeled, dtype=np.int32)
    keep = np.zeros_like(vessel, dtype=bool)
    for lab in range(1, int(n_cc) + 1):
        cc = labeled_np == lab
        if np.any(seeds & cc):
            keep |= cc
    stray = vessel & ~keep
    return keep, stray


def _prune_aca_stray_islands(
    laca_mask: np.ndarray,
    raca_mask: np.ndarray,
    laca_seeds: np.ndarray,
    raca_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop ACA CCs that do not touch that label's centerline seeds; swap to the other ACA."""
    l_keep, l_stray = _aca_seed_connected_keep_mask(laca_mask, laca_seeds)
    r_keep, r_stray = _aca_seed_connected_keep_mask(raca_mask, raca_seeds)
    n_reassigned = int(np.count_nonzero(l_stray) + np.count_nonzero(r_stray))

    laca_out = np.asarray(l_keep | r_stray, dtype=bool)
    raca_out = np.asarray(r_keep | l_stray, dtype=bool)
    overlap = laca_out & raca_out
    if np.any(overlap):
        d_laca = np.asarray(
            to_numpy(ndi.distance_transform_edt(~np.asarray(laca_seeds, dtype=bool))),
            dtype=np.float32,
        )
        d_raca = np.asarray(
            to_numpy(ndi.distance_transform_edt(~np.asarray(raca_seeds, dtype=bool))),
            dtype=np.float32,
        )
        coords = np.argwhere(overlap)
        dl = d_laca[coords[:, 0], coords[:, 1], coords[:, 2]]
        dr = d_raca[coords[:, 0], coords[:, 1], coords[:, 2]]
        to_laca = dl <= dr
        laca_out[coords[to_laca, 0], coords[to_laca, 1], coords[to_laca, 2]] = True
        raca_out[coords[to_laca, 0], coords[to_laca, 1], coords[to_laca, 2]] = False
        raca_out[coords[~to_laca, 0], coords[~to_laca, 1], coords[~to_laca, 2]] = True
        laca_out[coords[~to_laca, 0], coords[~to_laca, 1], coords[~to_laca, 2]] = False

    return laca_out, raca_out, n_reassigned


def _split_aca_merged_by_junction_plane(
    laca_mask: np.ndarray,
    raca_mask: np.ndarray,
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    *,
    acomm_junction_radius: int,
) -> _AcaPlaneSplitResult:
    """Plane-split the merged ACA blob near AComm, then drop seedless CC islands."""
    laca_in = np.asarray(laca_mask, dtype=bool)
    raca_in = np.asarray(raca_mask, dtype=bool)
    overlap = laca_in & raca_in
    aca_union = laca_in | raca_in
    if not np.any(overlap):
        return _AcaPlaneSplitResult(
            laca_in.copy(),
            raca_in.copy(),
            None,
            None,
            0,
            0,
            True,
            0,
            0,
        )

    laca_seeds, _ = _aca_seed_mask(centerlines_mask, eicab_qvtpy, QVTPY_LACA)
    raca_seeds, _ = _aca_seed_mask(centerlines_mask, eicab_qvtpy, QVTPY_RACA)
    junction, junction_source = _infer_aca_junction(laca_seeds, raca_seeds, eicab_qvtpy)
    if junction is None:
        raca_out = np.asarray(raca_in & ~overlap, dtype=bool)
        laca_out = np.asarray(laca_in.copy(), dtype=bool)
        return _AcaPlaneSplitResult(
            laca_out,
            raca_out,
            None,
            None,
            0,
            0,
            True,
            0,
            0,
        )

    axis, split_coord, laca_low = _infer_aca_split_plane(laca_seeds, raca_seeds, junction)
    rad = max(0, int(acomm_junction_radius))
    dist_j: np.ndarray | None = None
    if rad > 0:
        dist_j = _distance_from_junction_voxels(laca_in.shape, junction)

    if dist_j is not None and rad > 0:
        split_zone = aca_union & (dist_j <= float(rad))
    else:
        split_zone = aca_union

    laca_out = np.asarray(laca_in & ~split_zone, dtype=bool)
    raca_out = np.asarray(raca_in & ~split_zone, dtype=bool)
    n_split = 0

    d_laca = np.asarray(
        to_numpy(ndi.distance_transform_edt(~np.asarray(laca_seeds, dtype=bool))),
        dtype=np.float32,
    )
    d_raca = np.asarray(
        to_numpy(ndi.distance_transform_edt(~np.asarray(raca_seeds, dtype=bool))),
        dtype=np.float32,
    )

    if np.any(split_zone):
        coords = np.argwhere(split_zone)
        c = coords[:, axis].astype(np.int32)
        if laca_low:
            to_laca = c < split_coord
            to_raca = c > split_coord
        else:
            to_laca = c > split_coord
            to_raca = c < split_coord
        tie = ~(to_laca | to_raca)
        if np.any(tie):
            dl = d_laca[coords[:, 0], coords[:, 1], coords[:, 2]]
            dr = d_raca[coords[:, 0], coords[:, 1], coords[:, 2]]
            tie_laca = dl <= dr
            to_laca = to_laca | (tie & tie_laca)
            to_raca = to_raca | (tie & ~tie_laca)
        n_split = int(coords.shape[0])
        laca_out[coords[to_laca, 0], coords[to_laca, 1], coords[to_laca, 2]] = True
        raca_out[coords[to_raca, 0], coords[to_raca, 1], coords[to_raca, 2]] = True

    n_stray = 0
    if np.any(aca_union):
        laca_out, raca_out, n_stray = _prune_aca_stray_islands(
            laca_out, raca_out, laca_seeds, raca_seeds
        )

    return _AcaPlaneSplitResult(
        laca_out,
        raca_out,
        junction,
        junction_source,
        axis,
        split_coord,
        laca_low,
        n_split,
        n_stray,
    )


def _correct_aca_overlap_at_convergence(
    laca_mask: np.ndarray,
    raca_mask: np.ndarray,
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    *,
    acomm_junction_radius: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int] | None, int]:
    """Backward-compatible tuple return from junction-plane ACA split."""
    res = _split_aca_merged_by_junction_plane(
        laca_mask,
        raca_mask,
        centerlines_mask,
        eicab_qvtpy,
        acomm_junction_radius=acomm_junction_radius,
    )
    return res.laca_mask, res.raca_mask, res.junction, res.n_voxels_split


def _relabel_aca_masks_by_hemisphere(
    laca_mask: np.ndarray,
    raca_mask: np.ndarray,
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    *,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int] | None]:
    """Backward-compatible wrapper around junction-plane ACA split."""
    res = _split_aca_merged_by_junction_plane(
        laca_mask,
        raca_mask,
        centerlines_mask,
        eicab_qvtpy,
        acomm_junction_radius=acomm_junction_radius,
    )
    return res.laca_mask, res.raca_mask, res.junction


def _region_grow_binary_mask(
    vessel_mask: np.ndarray,
    cd: np.ndarray,
    *,
    rg_intensity_frac: float,
    rg_abs_floor: float | None,
    forbidden: np.ndarray | None = None,
) -> int:
    """6-connected RG on a boolean mask (may grow into voxels already True)."""
    print(f"Region growing binary mask with intensity fraction {rg_intensity_frac} and absolute floor {rg_abs_floor}")
    mask = np.asarray(vessel_mask, dtype=bool)
    cd_np = as_backend_array(cd).astype(np.float64)
    forb = None if forbidden is None else np.asarray(forbidden, dtype=bool)
    seeds = np.argwhere(mask)
    if seeds.size == 0:
        return 0

    seed_vals = cd_np[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    floor = float(rg_abs_floor) if rg_abs_floor is not None else 0.0
    grow_thresh = max(float(np.mean(seed_vals)) * float(rg_intensity_frac), floor)

    nx, ny, nz = mask.shape
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
            if forb is not None and forb[ni, nj, nk]:
                continue
            if cd_np[ni, nj, nk] < grow_thresh:
                continue
            if not mask[ni, nj, nk]:
                mask[ni, nj, nk] = True
                n_added += 1
            q.append((ni, nj, nk))

    return n_added


def _write_aca_masks_to_seg(
    seg: np.ndarray,
    laca_mask: np.ndarray,
    raca_mask: np.ndarray,
) -> None:
    """Replace LACA/RACA labels in *seg* from disjoint boolean masks."""
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    seg_np[seg_np == int(QVTPY_LACA)] = 0
    seg_np[seg_np == int(QVTPY_RACA)] = 0
    lm = np.asarray(laca_mask, dtype=bool)
    rm = np.asarray(raca_mask, dtype=bool)
    seg_np[lm] = int(QVTPY_LACA)
    seg_np[rm] = int(QVTPY_RACA)


def _region_grow_acas_sequential(
    seg: np.ndarray,
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    *,
    opt_thresh_by_label: dict[int, float | None],
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    venous_fracs: dict[int, float],
    rg_barrier_radius: int,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    stats_by_id: dict[int, VesselSegStats],
) -> AcaSequentialGrowInfo:
    """Grow LACA then RACA without blocking on the other ACA; fix close-approach overlap."""
    rg_rad = max(0, int(rg_barrier_radius))
    min_ov = max(0, int(aca_overlap_min_voxels))
    junc_rad = max(0, int(acomm_junction_radius))
    exclude_raca = frozenset({int(QVTPY_RACA)})
    exclude_laca = frozenset({int(QVTPY_LACA)})

    laca_mask = np.asarray(seg == int(QVTPY_LACA), dtype=bool)
    raca_mask = np.asarray(seg == int(QVTPY_RACA), dtype=bool)

    frac_l = rg_intensity_frac_for_label(
        QVTPY_LACA,
        default_frac=rg_intensity_frac,
        explore_frac=rg_intensity_frac_explore,
        venous_fracs=venous_fracs,
    )
    frac_r = rg_intensity_frac_for_label(
        QVTPY_RACA,
        default_frac=rg_intensity_frac,
        explore_frac=rg_intensity_frac_explore,
        venous_fracs=venous_fracs,
    )

    forb_laca = _dilated_other_segmentation_barrier_excluding(
        seg, QVTPY_LACA, exclude_label_ids=exclude_raca, radius=rg_rad
    )
    _region_grow_binary_mask(
        laca_mask,
        cd,
        rg_intensity_frac=frac_l,
        rg_abs_floor=opt_thresh_by_label.get(QVTPY_LACA),
        forbidden=forb_laca,
    )
    forb_raca = _dilated_other_segmentation_barrier_excluding(
        seg, QVTPY_RACA, exclude_label_ids=exclude_laca, radius=rg_rad
    )
    forb_raca = np.asarray(forb_raca, dtype=bool) & ~laca_mask
    _region_grow_binary_mask(
        raca_mask,
        cd,
        rg_intensity_frac=frac_r,
        rg_abs_floor=opt_thresh_by_label.get(QVTPY_RACA),
        forbidden=forb_raca,
    )

    overlap = laca_mask & raca_mask
    n_overlap = int(np.count_nonzero(overlap))
    split_info: _AcaPlaneSplitResult | None = None
    corrected = False

    if n_overlap > 0:
        if n_overlap >= min_ov:
            split_info = _split_aca_merged_by_junction_plane(
                laca_mask,
                raca_mask,
                centerlines_mask,
                eicab_qvtpy,
                acomm_junction_radius=junc_rad,
            )
            laca_mask = split_info.laca_mask
            raca_mask = split_info.raca_mask
            corrected = True
        else:
            raca_mask = np.asarray(raca_mask & ~overlap, dtype=bool)

    _write_aca_masks_to_seg(seg, laca_mask, raca_mask)

    for lid, frac in ((QVTPY_LACA, frac_l), (QVTPY_RACA, frac_r)):
        st = stats_by_id[lid]
        st.region_growing_applied = True
        st.rg_intensity_frac_used = float(frac)
        st.n_voxels_after_region_growing = int(np.count_nonzero(seg == int(lid)))

    axis_name = None
    if split_info is not None and 0 <= int(split_info.split_axis) < len(_ACA_SPLIT_AXIS_NAMES):
        axis_name = _ACA_SPLIT_AXIS_NAMES[int(split_info.split_axis)]

    return AcaSequentialGrowInfo(
        strategy="sequential_laca_then_raca_plane_split",
        grow_order=(int(QVTPY_LACA), int(QVTPY_RACA)),
        n_overlap_voxels=n_overlap,
        overlap_correction_applied=corrected,
        acomm_junction_ijk=None if split_info is None else split_info.junction,
        acomm_junction_radius_vox=junc_rad,
        n_convergence_voxels_corrected=0 if split_info is None else split_info.n_voxels_split,
        n_stray_islands_reassigned=0 if split_info is None else split_info.n_stray_islands_reassigned,
        junction_source=None if split_info is None else split_info.junction_source,
        split_axis=axis_name,
        split_coord=None if split_info is None else int(split_info.split_coord),
        laca_on_low_side=None if split_info is None else bool(split_info.laca_on_low_side),
    )


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
            aca_sequential_info = _region_grow_acas_sequential(
                seg,
                cd,
                clm,
                eicab,
                opt_thresh_by_label=opt_thresh_by_label,
                rg_intensity_frac=rg_intensity_frac,
                rg_intensity_frac_explore=rg_intensity_frac_explore,
                venous_fracs=venous_rg,
                rg_barrier_radius=rg_rad,
                aca_overlap_min_voxels=aca_overlap_min_voxels,
                acomm_junction_radius=acomm_junction_radius,
                stats_by_id=stats_by_id,
            )

        for st in stats:
            lid = st.label_id
            if use_aca_sequential and lid in QVTPY_ACA_IDS:
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
                venous_fracs=venous_rg,
            )
            floor = opt_thresh_by_label.get(lid)
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

    return LocalSegResult(
        segmentation=as_backend_array(seg.astype(np.int32, copy=False)),
        vessel_stats=stats,
        aca_sequential_grow=aca_sequential_info,
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
    "bbox_padding_for_label",
    "build_seg_4dflow_local",
    "crop_min_fraction_for_label",
    "region_growing_enabled_for_label",
    "resolve_venous_rg_intensity_fracs",
    "rg_intensity_frac_for_label",
    "vessel_stats_to_dict",
    "_bbox_with_padding",
]
