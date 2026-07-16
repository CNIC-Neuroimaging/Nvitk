"""Sequential ACA region growing and AComm junction overlap correction (qvtpy).

Called from :func:`~nvitk.pipes.qvtpy.util.vessel_cd_segmentation.build_seg_4dflow_local`
when ``aca_sequential_grow`` is enabled: grow LACA then RACA with explore-style
segmentation/centerline barriers (MCA, ICA, comms, …), allow L/R overlap only
inside a small AComm junction proximity zone, then plane-split overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected
from nvitk.pipes.qvtpy.labels import (
    EICAB_ACOMM,
    QVTPY_ACA_IDS,
    QVTPY_ACOMM,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_LPCOMM,
    QVTPY_MCA_IDS,
    QVTPY_RACA,
    QVTPY_RICA,
    QVTPY_RPCOMM,
)
from nvitk.morphology import dilate
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    AcaSequentialGrowInfo,
    VesselSegStats,
    _RG_EXPLORE_CL_BARRIER_RADIUS,
    _RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS,
    _RG_EXPLORE_SEG_BARRIER_RADIUS,
    _RG_MAX_GROW_FRAC_DEFAULT,
    _RG_MAX_IMAGE_FRAC_DEFAULT,
    _dilate_bool_mask,
    _dilated_labels_barrier,
    _dilated_other_segmentation_barrier_excluding,
    explore_region_grow_forbidden,
    merge_forbidden,
    rg_abs_floor_for_label,
    rg_caps_exceeded,
    rg_intensity_frac_for_label,
)
from nvitk.segmentation.region_growing import region_grow_binary_mask

setup(globals())

log = Logger()

_ACA_SEED_DILATE_RADIUS: int = 1
_ACA_PRUNE_MIN_RETAIN_FRAC: float = 0.55
_ACOMM_JUNCTION_RADIUS_DEFAULT: int = 10
_ACOMM_BRIDGE_DILATE_RADIUS: int = 1  # kept for ipsilateral launch dilation
_ACA_SPLIT_AXIS_NAMES: tuple[str, str] = ("x", "y")
# Proximal A1-only seeds: grow into distal ACA with ipsilateral half-space.
_ACA_LR_AXIS: int = 0  # array axis treated as L↔R (matches typical RAS lr=0)
_ACA_LR_HALFSPACE_SLACK: int = 2  # voxels of midline tolerance
_ACA_MAX_GROW_FRAC: float | None = None  # disable seed-multiple rollback for distal ACA
_ACA_ICA_TAKEOFF_CLEAR_RADIUS: int = 4  # clear parent-ICA wall near A1 / junction


def _dilate_seed_mask(seed_mask: np.ndarray, *, radius: int) -> np.ndarray:
    seeds = as_backend_array(seed_mask).astype(bool)
    if int(radius) <= 0 or not np.any(seeds):
        return seeds
    return as_backend_array(
        dilate(seeds.astype(np.uint8), footprint=int(radius), connectivity=1)
    ).astype(bool)


def _region_grow_binary_capped(
    mask: np.ndarray,
    cd: np.ndarray,
    *,
    intensity_frac: float,
    abs_floor: float | None,
    forbidden: np.ndarray | None,
    max_grow_frac: float | None,
    max_image_frac: float | None,
    seed_intensity_mask: np.ndarray | None = None,
) -> str | None:
    """Grow a boolean *mask* in place; roll back added voxels if a cap is breached."""
    pre = as_backend_array(mask).astype(bool, copy=True)
    n_pre = int(np.count_nonzero(pre))
    region_grow_binary_mask(
        mask,
        cd,
        intensity_frac=float(intensity_frac),
        abs_floor=abs_floor,
        forbidden=forbidden,
        seed_intensity_mask=seed_intensity_mask,
    )
    post = as_backend_array(mask).astype(bool)
    n_post = int(np.count_nonzero(post))
    reason = rg_caps_exceeded(
        n_pre,
        n_post,
        int(post.size),
        max_grow_frac=max_grow_frac,
        max_image_frac=max_image_frac,
    )
    if reason is not None:
        added = post & ~pre
        mask[added] = False
        return reason
    return None


def _log_aca_forbidden_diagnostics(
    name: str,
    seed_mask: np.ndarray,
    forbidden: np.ndarray,
) -> None:
    """Log how much of the ACA seed is walled in by the forbidden mask (diagnostics only)."""
    seeds = as_backend_array(seed_mask).astype(bool)
    forb = as_backend_array(forbidden).astype(bool)
    n_seed = int(np.count_nonzero(seeds))
    n_forb = int(np.count_nonzero(forb))
    n_seed_in_forb = int(np.count_nonzero(seeds & forb))
    # 6-neighbors of seeds that are free to claim (not seed, not forbidden).
    dilated = _dilate_bool_mask(seeds, radius=1)
    ring = dilated & ~seeds
    n_ring = int(np.count_nonzero(ring))
    n_ring_free = int(np.count_nonzero(ring & ~forb))
    n_ring_blocked = n_ring - n_ring_free
    frac_blocked = (float(n_ring_blocked) / float(n_ring)) if n_ring > 0 else 0.0
    log.info(
        f"ACA {name} forbidden: total={n_forb}, seed={n_seed}, "
        f"seed∩forbidden={n_seed_in_forb}, "
        f"seed-ring={n_ring} (free={n_ring_free}, blocked={n_ring_blocked}, "
        f"blocked_frac={frac_blocked:.2f})"
    )


def _aca_seed_mask(
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    label_id: int,
) -> tuple[np.ndarray, str]:
    """Thin ACA seeds: stage-3 centerline primary, union with skeletonized eICAB."""
    lid = int(label_id)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    cl_seeds = as_backend_array(clm == lid).astype(bool)
    eicab_seeds = np.zeros_like(cl_seeds).astype(bool)
    if eicab_qvtpy is not None:
        eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
        raw = as_backend_array(eq == lid).astype(bool)
        if np.any(raw):
            try:
                skel = skeletonize_binary(raw)
                skel_np = as_backend_array(skel).astype(bool)
                eicab_seeds = skel_np if np.any(skel_np) else raw
            except Exception:
                eicab_seeds = raw
    if np.any(cl_seeds) and np.any(eicab_seeds):
        merged = cl_seeds | eicab_seeds
    elif np.any(cl_seeds):
        merged = cl_seeds
    elif np.any(eicab_seeds):
        merged = eicab_seeds
    else:
        merged = cl_seeds
    merged = _dilate_seed_mask(merged, radius=_ACA_SEED_DILATE_RADIUS)
    if np.any(cl_seeds) and np.any(eicab_seeds):
        return merged, "centerline+eicab"
    if np.any(cl_seeds):
        return merged, "centerline"
    if np.any(eicab_seeds):
        return merged, "eicab_skeleton"
    return merged, "none"


def _ipsilateral_junction_launch_seeds(
    eicab_qvtpy: np.ndarray | None,
    laca_seeds: np.ndarray,
    raca_seeds: np.ndarray,
    junction: tuple[int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add thin ipsilateral AComm-side launches at the junction (not a shared fat bridge).

    LACA gets AComm voxels on the left of the junction LR plane; RACA on the right.
    This opens A2 takeoff without raising both sides' seed mean with the full bridge.
    """
    if eicab_qvtpy is None or junction is None:
        return laca_seeds, raca_seeds
    eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    acomm = as_backend_array(eq == int(QVTPY_ACOMM)).astype(bool)
    if not np.any(acomm):
        return laca_seeds, raca_seeds
    bridge = _dilate_seed_mask(acomm, radius=_ACOMM_BRIDGE_DILATE_RADIUS)
    ji = int(junction[int(_ACA_LR_AXIS)])
    coords = np.indices(bridge.shape, dtype=np.int32)
    lr = coords[int(_ACA_LR_AXIS)]
    left = bridge & (lr <= ji + int(_ACA_LR_HALFSPACE_SLACK))
    right = bridge & (lr >= ji - int(_ACA_LR_HALFSPACE_SLACK))
    return (
        as_backend_array(laca_seeds | left).astype(bool),
        as_backend_array(raca_seeds | right).astype(bool),
    )


def _lr_halfspace_forbidden(
    shape: tuple[int, int, int],
    junction: tuple[int, int, int] | None,
    label_id: int,
    *,
    slack: int = _ACA_LR_HALFSPACE_SLACK,
) -> np.ndarray:
    """Forbid contralateral LR half-space (LACA cannot cross right of junction, etc.)."""
    forb = np.zeros(shape, dtype=bool)
    if junction is None:
        return forb
    axis = int(_ACA_LR_AXIS)
    ji = int(junction[axis])
    sl = int(slack)
    if int(label_id) == int(QVTPY_LACA):
        if axis == 0:
            forb[(ji + sl + 1) :, :, :] = True
        elif axis == 1:
            forb[:, (ji + sl + 1) :, :] = True
        else:
            forb[:, :, (ji + sl + 1) :] = True
    else:
        if axis == 0:
            forb[: max(0, ji - sl), :, :] = True
        elif axis == 1:
            forb[:, : max(0, ji - sl), :] = True
        else:
            forb[:, :, : max(0, ji - sl)] = True
    return forb


def _junction_proximity_mask(
    shape: tuple[int, int, int],
    junction: tuple[int, int, int] | None,
    *,
    radius_vox: int,
) -> np.ndarray | None:
    """Voxels within *radius_vox* of the AComm junction (L/R may overlap here)."""
    if junction is None or int(radius_vox) <= 0:
        return None
    dist = _distance_from_junction_voxels(shape, junction)
    return as_backend_array(dist <= float(radius_vox)).astype(bool)


def _mask_outside_proximity(mask: np.ndarray, proximity: np.ndarray | None) -> np.ndarray:
    """Barrier from *mask* excluding the junction proximity zone."""
    m = as_backend_array(mask).astype(bool)
    if proximity is None or not np.any(proximity):
        return m
    return m & ~as_backend_array(proximity).astype(bool)


def _aca_neighbour_label_ids(label_id: int) -> frozenset[int]:
    """MCA + PComms as neighbour walls; parent ICA omitted (cleared at A1 takeoff)."""
    lid = int(label_id)
    # AComm and parent ICA intentionally omitted — ICA is cleared near takeoff;
    # AComm must not wall ACA growth.
    pcomms = frozenset({int(QVTPY_LPCOMM), int(QVTPY_RPCOMM)})
    return frozenset(int(x) for x in (QVTPY_MCA_IDS | pcomms) if int(x) != lid)


def _parent_ica_id(label_id: int) -> int:
    return int(QVTPY_LICA) if int(label_id) == int(QVTPY_LACA) else int(QVTPY_RICA)


def _acomm_non_barrier_voxels(
    shape: tuple[int, int, int],
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
    eicab_native: np.ndarray | None,
) -> np.ndarray:
    """Foreground where eICAB / qvtpy AComm must never act as an ACA RG barrier."""
    mask = np.zeros(shape, dtype=bool)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    mask |= as_backend_array(clm == int(QVTPY_ACOMM)).astype(bool)
    if eicab_qvtpy is not None:
        eq = as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
        mask |= as_backend_array(eq == int(QVTPY_ACOMM)).astype(bool)
    if eicab_native is not None:
        en = as_backend_array(eicab_native).astype(np.int32, copy=False)
        mask |= as_backend_array(en == int(EICAB_ACOMM)).astype(bool)
    if not np.any(mask):
        return mask
    return _dilate_bool_mask(mask, radius=1)


def _ica_takeoff_clear_mask(
    shape: tuple[int, int, int],
    a1_seeds: np.ndarray,
    junction: tuple[int, int, int] | None,
    *,
    radius: int = _ACA_ICA_TAKEOFF_CLEAR_RADIUS,
) -> np.ndarray:
    """Region around A1 seeds / junction where parent-ICA walls are cleared."""
    clear = _dilate_bool_mask(as_backend_array(a1_seeds).astype(bool), radius=int(radius))
    prox = _junction_proximity_mask(shape, junction, radius_vox=int(radius))
    if prox is not None:
        clear = clear | as_backend_array(prox).astype(bool)
    return clear


def _aca_region_grow_forbidden(
    seg: np.ndarray,
    centerlines_mask: np.ndarray,
    label_id: int,
    *,
    opposite_mask: np.ndarray,
    junction: tuple[int, int, int] | None,
    junction_radius: int,
    rg_barrier_radius: int,
    a1_seeds: np.ndarray,
    eicab_qvtpy: np.ndarray | None = None,
    eicab_native: np.ndarray | None = None,
) -> np.ndarray:
    """Explore-style RG barriers for one ACA with ipsilateral half-space + ICA takeoff clear."""
    lid = int(label_id)
    opp_id = int(QVTPY_RACA) if lid == int(QVTPY_LACA) else int(QVTPY_LACA)
    parent_ica = _parent_ica_id(lid)
    # Exclude opposite ACA, AComm, and parent ICA from generic other-label walls;
    # ICA is re-added then cleared only near A1 takeoff.
    exclude = frozenset({opp_id, int(QVTPY_ACOMM), parent_ica})
    rg_rad = max(0, int(rg_barrier_radius))

    forb = explore_region_grow_forbidden(
        seg,
        centerlines_mask,
        lid,
        exclude_label_ids=exclude,
        seg_barrier_radius=_RG_EXPLORE_SEG_BARRIER_RADIUS,
        cl_barrier_radius=_RG_EXPLORE_CL_BARRIER_RADIUS,
    )
    forb = merge_forbidden(
        forb,
        _dilated_other_segmentation_barrier_excluding(
            seg,
            lid,
            exclude_label_ids=exclude,
            radius=rg_rad,
        ),
        _dilated_labels_barrier(
            seg,
            _aca_neighbour_label_ids(lid),
            radius=_RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS,
        ),
    )
    # Mild parent-ICA wall, then punch a hole at A1 / junction takeoff.
    ica_wall = _dilated_labels_barrier(
        seg, frozenset({parent_ica}), radius=max(1, rg_rad - 1)
    )
    ica_cl = as_backend_array(centerlines_mask).astype(np.int32, copy=False) == parent_ica
    ica_wall = merge_forbidden(ica_wall, _dilate_bool_mask(ica_cl, radius=1))
    takeoff_clear = _ica_takeoff_clear_mask(seg.shape, a1_seeds, junction)
    if ica_wall is not None and np.any(ica_wall):
        ica_wall = as_backend_array(ica_wall).astype(bool)
        ica_wall[takeoff_clear] = False
        forb = merge_forbidden(forb, ica_wall)

    proximity = _junction_proximity_mask(seg.shape, junction, radius_vox=junction_radius)
    # Contralateral half-space (cleared inside AComm proximity).
    half = _lr_halfspace_forbidden(seg.shape, junction, lid)
    half = _mask_outside_proximity(half, proximity)
    forb = merge_forbidden(forb, half)

    opp_bar = _dilate_bool_mask(opposite_mask, radius=rg_rad)
    opp_bar = _mask_outside_proximity(opp_bar, proximity)
    if np.any(opp_bar):
        forb = merge_forbidden(forb, opp_bar)
    acomm_clear = _acomm_non_barrier_voxels(
        seg.shape,
        centerlines_mask,
        eicab_qvtpy,
        eicab_native,
    )
    if np.any(acomm_clear):
        forb = as_backend_array(forb).astype(bool)
        forb[acomm_clear] = False
    # Never forbid current A1 seeds themselves.
    forb = as_backend_array(forb).astype(bool)
    forb[as_backend_array(a1_seeds).astype(bool)] = False
    assert forb is not None
    return forb

def aca_seed_volume(
    centerlines_mask: np.ndarray,
    eicab_qvtpy: np.ndarray | None,
) -> np.ndarray:
    """Centerlines mask with ACA labels extended by eICAB skeleton seeds."""
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    out = clm.copy()
    for lid in QVTPY_ACA_IDS:
        seeds, _ = _aca_seed_mask(clm, eicab_qvtpy, int(lid))
        out[seeds] = int(lid)
    return out


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
    return as_backend_array(ndi.distance_transform_edt(~junc))


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
    """Drop ACA CCs that do not touch dilated centerline seeds; swap to the other ACA."""
    l_seed = _dilate_seed_mask(laca_seeds, radius=3)
    r_seed = _dilate_seed_mask(raca_seeds, radius=3)
    n_l_pre = int(np.count_nonzero(laca_mask))
    n_r_pre = int(np.count_nonzero(raca_mask))
    l_keep, l_stray = _aca_seed_connected_keep_mask(laca_mask, l_seed)
    r_keep, r_stray = _aca_seed_connected_keep_mask(raca_mask, r_seed)
    n_reassigned = int(np.count_nonzero(l_stray) + np.count_nonzero(r_stray))

    laca_out = as_backend_array(l_keep | r_stray).astype(bool)
    raca_out = as_backend_array(r_keep | l_stray).astype(bool)
    if n_l_pre > 0 and int(np.count_nonzero(laca_out)) < int(_ACA_PRUNE_MIN_RETAIN_FRAC * n_l_pre):
        laca_out = as_backend_array(laca_mask).astype(bool)
    if n_r_pre > 0 and int(np.count_nonzero(raca_out)) < int(_ACA_PRUNE_MIN_RETAIN_FRAC * n_r_pre):
        raca_out = as_backend_array(raca_mask).astype(bool)
    overlap = laca_out & raca_out
    if np.any(overlap):
        d_laca = as_backend_array(
            ndi.distance_transform_edt(~as_backend_array(laca_seeds).astype(bool))
        ).astype(np.float32)
        d_raca = as_backend_array(
            ndi.distance_transform_edt(~as_backend_array(raca_seeds).astype(bool))
        ).astype(np.float32)
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
        ndi.distance_transform_edt(~as_backend_array(laca_seeds).astype(bool))
    ).astype(np.float32)
    
    d_raca = as_backend_array(
        ndi.distance_transform_edt(~as_backend_array(raca_seeds).astype(bool))
    ).astype(np.float32)

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
    eicab_native: np.ndarray | None = None,
    opt_thresh_by_label: dict[int, float | None],
    rg_intensity_frac: float,
    rg_intensity_frac_aca: float,
    venous_fracs: dict[int, float],
    rg_barrier_radius: int,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    stats_by_id: dict[int, VesselSegStats],
    max_grow_frac: float | None = _RG_MAX_GROW_FRAC_DEFAULT,
    max_image_frac: float | None = _RG_MAX_IMAGE_FRAC_DEFAULT,
) -> AcaSequentialGrowInfo:
    """Grow LACA then RACA with explore-style barriers; L/R may mix only near AComm."""
    min_ov = max(0, int(aca_overlap_min_voxels))
    junc_rad = max(0, int(acomm_junction_radius))
    rg_rad = max(0, int(rg_barrier_radius))

    log.step("ACA sequential region growing (LACA → RACA)")
    # A1-only seeds for intensity mean (before ipsilateral launch).
    laca_a1, laca_src = _aca_seed_mask(centerlines_mask, eicab_qvtpy, QVTPY_LACA)
    raca_a1, raca_src = _aca_seed_mask(centerlines_mask, eicab_qvtpy, QVTPY_RACA)
    n_laca_a1 = int(np.count_nonzero(laca_a1))
    n_raca_a1 = int(np.count_nonzero(raca_a1))
    junction, junction_src = _infer_aca_junction(laca_a1, raca_a1, eicab_qvtpy)
    laca_seeds, raca_seeds = _ipsilateral_junction_launch_seeds(
        eicab_qvtpy, laca_a1, raca_a1, junction
    )
    n_laca_seed = int(np.count_nonzero(laca_seeds))
    n_raca_seed = int(np.count_nonzero(raca_seeds))
    # Distal ACA: disable seed-multiple volume rollback (A1 stubs are tiny).
    grow_cap = _ACA_MAX_GROW_FRAC
    log.info(
        f"ACA seeds: LACA src={laca_src} A1={n_laca_a1}→launch={n_laca_seed}, "
        f"RACA src={raca_src} A1={n_raca_a1}→launch={n_raca_seed}; "
        f"junction={junction} src={junction_src}, "
        f"acomm_radius={junc_rad}, rg_barrier_radius={rg_rad}, "
        f"lr_halfspace_slack={_ACA_LR_HALFSPACE_SLACK}, "
        f"explore_seg_r={_RG_EXPLORE_SEG_BARRIER_RADIUS}, "
        f"explore_cl_r={_RG_EXPLORE_CL_BARRIER_RADIUS}, "
        f"neighbour_r={_RG_EXPLORE_NEIGHBOUR_BARRIER_RADIUS}, "
        f"grow_cap={grow_cap}"
    )

    n_seg_laca0 = int(np.count_nonzero(seg == int(QVTPY_LACA)))
    n_seg_raca0 = int(np.count_nonzero(seg == int(QVTPY_RACA)))
    laca_mask = as_backend_array(seg == int(QVTPY_LACA)).astype(bool) | laca_seeds
    raca_mask = as_backend_array(seg == int(QVTPY_RACA)).astype(bool) | raca_seeds
    n_laca_pre = int(np.count_nonzero(laca_mask))
    n_raca_pre = int(np.count_nonzero(raca_mask))
    log.info(
        f"ACA pre-grow: seg LACA={n_seg_laca0} RACA={n_seg_raca0}; "
        f"mask∪seeds LACA={n_laca_pre} RACA={n_raca_pre}"
    )

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
    floor_l = rg_abs_floor_for_label(QVTPY_LACA, opt_thresh_by_label.get(QVTPY_LACA))
    floor_r = rg_abs_floor_for_label(QVTPY_RACA, opt_thresh_by_label.get(QVTPY_RACA))
    log.info(
        f"ACA RG gates: LACA frac={frac_l:.3f} floor={floor_l} (mean from A1), "
        f"RACA frac={frac_r:.3f} floor={floor_r} (mean from A1), "
        f"max_grow_frac={grow_cap}, max_image_frac={max_image_frac}"
    )

    forb_laca = _aca_region_grow_forbidden(
        seg,
        centerlines_mask,
        QVTPY_LACA,
        opposite_mask=raca_mask,
        junction=junction,
        junction_radius=junc_rad,
        rg_barrier_radius=rg_rad,
        a1_seeds=laca_a1,
        eicab_qvtpy=eicab_qvtpy,
        eicab_native=eicab_native,
    )
    _log_aca_forbidden_diagnostics("LACA", laca_mask, forb_laca)
    reason_l = _region_grow_binary_capped(
        laca_mask,
        cd,
        intensity_frac=frac_l,
        abs_floor=floor_l,
        forbidden=forb_laca,
        max_grow_frac=grow_cap,
        max_image_frac=max_image_frac,
        seed_intensity_mask=laca_a1,
    )
    n_laca_post = int(np.count_nonzero(laca_mask))
    log.info(
        f"ACA LACA grow: {n_laca_pre} → {n_laca_post} "
        f"(Δ={n_laca_post - n_laca_pre})"
        + (f"; ROLLBACK: {reason_l}" if reason_l else "; kept")
    )

    forb_raca = _aca_region_grow_forbidden(
        seg,
        centerlines_mask,
        QVTPY_RACA,
        opposite_mask=laca_mask,
        junction=junction,
        junction_radius=junc_rad,
        rg_barrier_radius=rg_rad,
        a1_seeds=raca_a1,
        eicab_qvtpy=eicab_qvtpy,
        eicab_native=eicab_native,
    )
    proximity = _junction_proximity_mask(laca_mask.shape, junction, radius_vox=junc_rad)
    forb_raca = merge_forbidden(
        forb_raca,
        _mask_outside_proximity(as_backend_array(laca_mask).astype(bool), proximity),
    )
    assert forb_raca is not None
    n_prox = int(np.count_nonzero(proximity)) if proximity is not None else 0
    log.info(f"ACA RACA opposite-LACA barrier: junction proximity voxels={n_prox}")
    _log_aca_forbidden_diagnostics("RACA", raca_mask, forb_raca)
    reason_r = _region_grow_binary_capped(
        raca_mask,
        cd,
        intensity_frac=frac_r,
        abs_floor=floor_r,
        forbidden=forb_raca,
        max_grow_frac=grow_cap,
        max_image_frac=max_image_frac,
        seed_intensity_mask=raca_a1,
    )
    n_raca_post = int(np.count_nonzero(raca_mask))
    log.info(
        f"ACA RACA grow: {n_raca_pre} → {n_raca_post} "
        f"(Δ={n_raca_post - n_raca_pre})"
        + (f"; ROLLBACK: {reason_r}" if reason_r else "; kept")
    )

    overlap = laca_mask & raca_mask
    n_overlap = int(np.count_nonzero(overlap))
    split_info: _AcaPlaneSplitResult | None = None
    corrected = False

    if n_overlap > 0:
        if n_overlap >= min_ov:
            log.info(
                f"ACA overlap={n_overlap} ≥ min={min_ov}: plane-split at AComm junction"
            )
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
            log.info(
                f"ACA plane-split: axis={split_info.split_axis} "
                f"coord={split_info.split_coord}, "
                f"split_vox={split_info.n_voxels_split}, "
                f"stray_islands={split_info.n_stray_islands_reassigned}, "
                f"junction_src={split_info.junction_source}; "
                f"LACA={int(np.count_nonzero(laca_mask))} "
                f"RACA={int(np.count_nonzero(raca_mask))}"
            )
        else:
            log.info(
                f"ACA overlap={n_overlap} < min={min_ov}: "
                "clear overlap from RACA only (no plane-split)"
            )
            raca_mask = as_backend_array(raca_mask & ~overlap).astype(bool)
    else:
        log.info("ACA overlap=0: no L/R correction")

    _write_aca_masks_to_seg(seg, laca_mask, raca_mask)
    log.info(
        f"ACA written to seg: LACA={int(np.count_nonzero(seg == int(QVTPY_LACA)))}, "
        f"RACA={int(np.count_nonzero(seg == int(QVTPY_RACA)))}"
    )

    for lid, frac, reason in (
        (QVTPY_LACA, frac_l, reason_l),
        (QVTPY_RACA, frac_r, reason_r),
    ):
        st = stats_by_id[lid]
        st.region_growing_applied = True
        st.rg_intensity_frac_used = float(frac)
        st.n_voxels_after_region_growing = int(np.count_nonzero(seg == int(lid)))
        if reason is not None:
            st.warning = reason

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
