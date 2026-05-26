"""Sequential ACA region growing and AComm junction overlap correction (qvtpy).

Called from :func:`~nvitk.pipes.qvtpy.util.vessel_cd_segmentation.build_seg_4dflow_local`
when ``aca_sequential_grow`` is enabled: grow LACA then RACA without mutual barriers,
then split overlap at the AComm junction plane when voxel overlap exceeds a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ACA_IDS,
    QVTPY_ACOMM,
    QVTPY_LACA,
    QVTPY_RACA,
)
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    AcaSequentialGrowInfo,
    VesselSegStats,
    explore_region_grow_forbidden,
    rg_abs_floor_for_label,
    rg_intensity_frac_for_label,
)
from nvitk.segmentation.region_growing import region_grow_binary_mask

setup(globals())

_ACOMM_JUNCTION_RADIUS_DEFAULT: int = 10
_ACA_SPLIT_AXIS_NAMES: tuple[str, str] = ("x", "y")

def _aca_seed_mask(
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    label_id: int,
) -> tuple[np.ndarray, str]:
    """Thin ACA seeds: stage-3 centerline primary, skeletonized eICAB fallback."""
    lid = int(label_id)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    seeds = as_backend_array(clm == lid).astype(bool)
    if np.any(seeds):
        return seeds, "centerline"
    if eicab_qvtpy is None:
        return seeds, "none"
    eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    raw = as_backend_array(eq == lid).astype(bool)
    if not np.any(raw):
        return seeds, "none"
    try:
        skel = skeletonize_binary(raw)
        skel_np = as_backend_array(skel).astype(bool)
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
    coords_np = as_backend_array(to_numpy(coords)).astype(np.float64)
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
    lc = as_backend_array(to_numpy(np.argwhere(laca_seeds))).astype(np.float64)
    rc = as_backend_array(to_numpy(np.argwhere(raca_seeds))).astype(np.float64)
    if lc.size == 0 or rc.size == 0:
        return None, None
    d2 = np.sum((lc[:, None, :] - rc[None, :, :]) ** 2, axis=2)
    flat_idx = int(np.argmin(d2))
    li, ri = np.unravel_index(flat_idx, d2.shape)
    mid = 0.5 * (lc[int(li)] + rc[int(ri)])
    nx, ny, nz = laca_seeds.shape
    return (
        (
            int(max(0, min(nx - 1, np.round(mid[0])))),
            int(max(0, min(ny - 1, np.round(mid[1])))),
            int(max(0, min(nz - 1, np.round(mid[2])))),
        ),
        "aca_midpoint",
    )


def _infer_aca_split_plane(
    laca_seeds: np.ndarray,
    raca_seeds: np.ndarray,
    junction: tuple[int, int, int],
) -> tuple[int, int, bool]:
    """Dominant in-plane axis and whether LACA lies on the low-index side of *junction*."""
    lc = as_backend_array(np.argwhere(laca_seeds)).astype(np.float64)
    rc = as_backend_array(np.argwhere(raca_seeds)).astype(np.float64)
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
    coords = as_backend_array(np.argwhere(seeds)).astype(np.float64)
    if coords.shape[0] < 2:
        return None
    c = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(c, full_matrices=False)
    axis = as_backend_array(vh[0]).astype(np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return None
    return axis / n


def _aca_arm_from_junction(
    seeds: np.ndarray,
    junction: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    """Unit vector along ACA centerline axis; endpoint is farthest seed from junction."""
    coords = as_backend_array(np.argwhere(seeds)).astype(np.float64)
    if coords.size == 0:
        return np.zeros(3).astype(np.float64), None
    axis = _aca_axis_unit_vector(seeds)
    if axis is None:
        return np.zeros(3).astype(np.float64), None
    j = as_backend_array(junction).astype(np.float64)
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
    junc = np.zeros(shape).astype(bool)
    junc[ji, jj, jk] = True
    return as_backend_array(to_numpy(ndi.distance_transform_edt(~junc)), dtype=np.float32)


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
    vessel = as_backend_array(vessel_mask).astype(bool)
    seeds = as_backend_array(seed_mask).astype(bool)
    if not np.any(vessel):
        return vessel.copy(), np.zeros_like(vessel).astype(bool)

    labeled, n_cc = label_connected(vessel, connectivity=1)
    if n_cc <= 0:
        return vessel.copy(), np.zeros_like(vessel).astype(bool)

    labeled_np = as_backend_array(labeled).astype(np.int32)
    keep = np.zeros_like(vessel).astype(bool)
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

    laca_out = as_backend_array(l_keep | r_stray).astype(bool)
    raca_out = as_backend_array(r_keep | l_stray).astype(bool)
    overlap = laca_out & raca_out
    if np.any(overlap):
        d_laca = as_backend_array(
            to_numpy(ndi.distance_transform_edt(~as_backend_array(laca_seeds).astype(bool))),
            dtype=np.float32,
        )
        d_raca = as_backend_array(
            to_numpy(ndi.distance_transform_edt(~as_backend_array(raca_seeds).astype(bool))),
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
    laca_in = as_backend_array(laca_mask).astype(bool)
    raca_in = as_backend_array(raca_mask).astype(bool)
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
        raca_out = as_backend_array(raca_in & ~overlap).astype(bool)
        laca_out = as_backend_array(laca_in.copy()).astype(bool)
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

    laca_out = as_backend_array(laca_in & ~split_zone).astype(bool)
    raca_out = as_backend_array(raca_in & ~split_zone).astype(bool)
    n_split = 0

    d_laca = as_backend_array(
        to_numpy(ndi.distance_transform_edt(~as_backend_array(laca_seeds).astype(bool))),
        dtype=np.float32,
    )
    d_raca = as_backend_array(
        to_numpy(ndi.distance_transform_edt(~as_backend_array(raca_seeds).astype(bool))),
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


def _write_aca_masks_to_seg(
    seg: np.ndarray,
    laca_mask: np.ndarray,
    raca_mask: np.ndarray,
) -> None:
    """Replace LACA/RACA labels in *seg* from disjoint boolean masks."""
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    seg_np[seg_np == int(QVTPY_LACA)] = 0
    seg_np[seg_np == int(QVTPY_RACA)] = 0
    lm = as_backend_array(laca_mask).astype(bool)
    rm = as_backend_array(raca_mask).astype(bool)
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
    rg_intensity_frac_aca: float,
    venous_fracs: dict[int, float],
    rg_barrier_radius: int,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    stats_by_id: dict[int, VesselSegStats],
) -> AcaSequentialGrowInfo:
    """Grow LACA then RACA without blocking on the other ACA; fix close-approach overlap."""
    min_ov = max(0, int(aca_overlap_min_voxels))
    junc_rad = max(0, int(acomm_junction_radius))
    exclude_raca = frozenset({int(QVTPY_RACA)})
    exclude_laca = frozenset({int(QVTPY_LACA)})

    laca_mask = as_backend_array(seg == int(QVTPY_LACA)).astype(bool)
    raca_mask = as_backend_array(seg == int(QVTPY_RACA)).astype(bool)

    frac_l = rg_intensity_frac_for_label(
        QVTPY_LACA,
        default_frac=rg_intensity_frac,
        aca_frac=rg_intensity_frac_aca,
        venous_fracs=venous_fracs,
    )
    frac_r = rg_intensity_frac_for_label(
        QVTPY_RACA,
        default_frac=rg_intensity_frac,
        aca_frac=rg_intensity_frac_aca,
        venous_fracs=venous_fracs,
    )

    forb_laca = explore_region_grow_forbidden(
        seg,
        centerlines_mask,
        QVTPY_LACA,
        exclude_label_ids=exclude_raca,
    )
    region_grow_binary_mask(
        laca_mask,
        cd,
        intensity_frac=frac_l,
        abs_floor=rg_abs_floor_for_label(QVTPY_LACA, opt_thresh_by_label.get(QVTPY_LACA)),
        forbidden=forb_laca,
    )
    forb_raca = explore_region_grow_forbidden(
        seg,
        centerlines_mask,
        QVTPY_RACA,
        exclude_label_ids=exclude_laca,
    )
    forb_raca = as_backend_array(forb_raca).astype(bool) & ~laca_mask
    region_grow_binary_mask(
        raca_mask,
        cd,
        intensity_frac=frac_r,
        abs_floor=rg_abs_floor_for_label(QVTPY_RACA, opt_thresh_by_label.get(QVTPY_RACA)),
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
            raca_mask = as_backend_array(raca_mask & ~overlap).astype(bool)

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
