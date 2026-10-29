"""Per-vessel local CD crop + threshold, then one blood flood, for stage 4.

**Inputs**

- 3D complex-difference volume, stage-3 ``centerlines_mask``, optional warped eICAB labels.

**Outputs**

- Multilabel ``seg_4dflow`` via :func:`build_seg_4dflow_local` and per-vessel :class:`VesselSegStats`.

Array indices ``(i, j, k)`` are **(X, Y, Z)** for asymmetric bbox padding.

Steps (:func:`build_seg_4dflow_local`):

1. **Threshold.** Each centerline label gets a local CD crop around its centerline, thresholded
   (``lsthr`` / ``lthr`` / ``otsu``), with the other vessels' centerlines as barriers; then its
   largest connected component. The communicating arteries are seeded the same way (their
   centerline plus the threshold crop).
2. **Vertebral split** of the basilar (LVA / RVA).
3. **Blood flood** (:func:`flood_expand_vessels`), the only growth step. A Frangi + GMM-hysteresis
   vessel tree of the CD is flooded from every label at once (watershed). The trunks (ICAs,
   basilar, vertebrals), the communicating arteries and the venous sinuses only fill their own
   lumen (voxels of the tree within ``flood_near_radius`` of their mask). The other arteries (ACA / MCA / PCA, their
   A2 / M2 / P2, the SCAs) grow along the tree. What an A1 / M1 / P1 floods
   beyond its own lumen is written as its A2 / M2 / P2 (eICAB's post-process does the same on TOF).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger
from nvitk.morphology import dilate, make_ball_footprint
from nvitk.morphology.components import remove_small_components_by_fraction
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ACA_A2_IDS,
    QVTPY_ACA_IDS,
    QVTPY_ACOMM,
    QVTPY_ARTERIAL_ID_TO_NAME,
    QVTPY_BASILAR,
    QVTPY_COMM_IDS,
    QVTPY_DISTAL_IDS,
    QVTPY_ICA_BASILAR_IDS,
    QVTPY_LACA,
    QVTPY_LACA_A2,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_LMCA_M2,
    QVTPY_LPCA,
    QVTPY_LPCA_P2,
    QVTPY_LSCA,
    QVTPY_MCA_IDS,
    QVTPY_PCA_IDS,
    QVTPY_PROXIMAL_TO_DISTAL,
    QVTPY_RACA,
    QVTPY_RACA_A2,
    QVTPY_RMCA,
    QVTPY_RMCA_M2,
    QVTPY_RPCA,
    QVTPY_RPCA_P2,
    QVTPY_RSCA,
    QVTPY_SCA_IDS,
    QVTPY_SMALL_ARTERIAL_IDS,
    QVTPY_VENOUS_LABEL_IDS,
    QVTPY_VERTEBRAL_IDS,
)
from nvitk.pipes.qvtpy.util.segmentation.flood_options import FLOOD_DEFAULTS
from nvitk.pipes.qvtpy.util.segmentation.vertebral_split import VertebralSplitResult, split_vertebral_from_basilar
from nvitk.filters.sliding_threshold import binary_mask_sliding_threshold_3d
from nvitk.pipes.qvtpy.util.centerline.mask_cleaning import (
    keep_component_touching_seed_inplace,
    keep_largest_component_label_inplace,
)

setup(globals())

log = Logger()

# Types and defaults

ThrAlgorithm = Literal["lsthr", "lthr", "otsu"]
_CROP_MIN_COMPONENT_FRAC = 0.005
_CROP_MIN_COMPONENT_FRAC_SMALL = 0.0
VESSEL_EXTRA_PADDING: int = 10
#: Centerline barrier radius (voxels) of the branched arteries' threshold paste (ACA / MCA / PCA and
#: their distal labels), which crowd each other near the circle.
_CL_BARRIER_RADIUS_BRANCHED: int = 3

# Blood flood (flood_expand_vessels): Frangi vesselness -> GMM hysteresis tree -> watershed markers.
FLOOD_MAX_IMAGE_FRAC_DEFAULT: float = FLOOD_DEFAULTS["flood_max_image_frac"]
FLOOD_HYST_LOW_FACTOR_DEFAULT: float = FLOOD_DEFAULTS["flood_hyst_low_factor"]
FLOOD_HYST_HIGH_FACTOR_DEFAULT: float = FLOOD_DEFAULTS["flood_hyst_high_factor"]
FLOOD_THICKEN_ITER_DEFAULT: int = FLOOD_DEFAULTS["flood_thicken_iter"]
FLOOD_LR_HALFSPACE_SLACK_DEFAULT: int = FLOOD_DEFAULTS["flood_lr_halfspace_slack"]
#: How far (voxels) the trunks and communicating arteries may flood from their own mask.
FLOOD_NEAR_RADIUS_DEFAULT: int = FLOOD_DEFAULTS["flood_near_radius"]
_DISTAL_LR_AXIS: int = 0  # array axis treated as L<->R (matches ACA / typical RAS)
# Drop the lowest vesselness voxels on the tree to keep growth tubular (anti-blob).
_DISTAL_TREE_VESSELNESS_KEEP_PERCENTILE: float = 55.0
# MCA/PCA midline punch is restricted to a dilated band around that pair's
# markers, and never clears voxels near ACA seeds (protects A2 on X-midline).
_DISTAL_LR_PUNCH_PAIR_RADIUS: int = 18
_DISTAL_ACA_PROTECT_RADIUS: int = 14
# ACA distal corridor: propagate from ACA seeds into vesselness-gated CD so A2
# can reconnect when Frangi CCs are broken at AComm.
_DISTAL_ACA_CORRIDOR_MAX_DIST_VOX: float = 45.0
_DISTAL_ACA_CORRIDOR_CD_PERCENTILE: float = 60.0
# Dilated ICA + basilar: hard walls; the growing arteries cannot claim these voxels.
_DISTAL_ICA_BASILAR_BARRIER_RADIUS: int = 3
_DISTAL_LR_PAIRS: tuple[tuple[int, int], ...] = (
    (QVTPY_LACA, QVTPY_RACA),
    (QVTPY_LMCA, QVTPY_RMCA),
    (QVTPY_LPCA, QVTPY_RPCA),
    (QVTPY_LACA_A2, QVTPY_RACA_A2),
    (QVTPY_LMCA_M2, QVTPY_RMCA_M2),
    (QVTPY_LPCA_P2, QVTPY_RPCA_P2),
    (QVTPY_LSCA, QVTPY_RSCA),
)
#: Arteries that grow along the vessel tree (the others only fill their own lumen).
_DISTAL_TARGET_IDS: frozenset[int] = frozenset(
    QVTPY_MCA_IDS | QVTPY_ACA_IDS | QVTPY_PCA_IDS | QVTPY_DISTAL_IDS | QVTPY_SCA_IDS
)
#: Labels that only fill their own lumen: trunks, vertebrals, communicating arteries and the venous
#: sinuses (a growing sinus would run out along the cortical veins).
_NEAR_FIELD_IDS: frozenset[int] = frozenset(
    QVTPY_ICA_BASILAR_IDS | QVTPY_VERTEBRAL_IDS | QVTPY_COMM_IDS | QVTPY_VENOUS_LABEL_IDS
)
_ACA_FAMILY_IDS: frozenset[int] = frozenset(QVTPY_ACA_IDS | QVTPY_ACA_A2_IDS)


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
        """Serialize the per-face bounding-box padding to a plain dict."""
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
    n_voxels_final: int
    flooded: bool = False
    warning: str | None = None


@dataclass
class LocalSegResult:
    """``seg_4dflow`` volume and per-vessel metadata."""

    segmentation: np.ndarray
    vessel_stats: list[VesselSegStats] = field(default_factory=list)
    vertebral_split: VertebralSplitResult | None = None
    flood: dict[str, Any] | None = None


def crop_min_fraction_for_label(label_id: int) -> float:
    """Min CC fraction inside the CD crop (0 for the small arteries: PCAs, comms, SCAs)."""
    if int(label_id) in QVTPY_SMALL_ARTERIAL_IDS:
        return _CROP_MIN_COMPONENT_FRAC_SMALL
    return _CROP_MIN_COMPONENT_FRAC


def bbox_padding_for_label(label_id: int, default_pad: int) -> BboxFacePadding:
    """Per-vessel asymmetric padding around the centerline ROI bbox."""
    d = max(0, int(default_pad))
    extra = int(VESSEL_EXTRA_PADDING)
    lid = int(label_id)

    if lid in QVTPY_ICA_BASILAR_IDS:
        return BboxFacePadding(d, d, d, d, d, 0)
    if lid in (QVTPY_LMCA, QVTPY_LMCA_M2):
        return BboxFacePadding(0, extra, d, d, d, d)
    if lid in (QVTPY_RMCA, QVTPY_RMCA_M2):
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
    """Binary-dilate *mask* by *radius* voxels (connectivity 1); no-op for radius <= 0 or an empty mask."""
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


# Blood flood (single growth step)


def _finalize_vessel_largest_cc(
    seg: np.ndarray,
    label_id: int,
    *,
    stats_by_id: dict[int, VesselSegStats] | None = None,
    log_name: str | None = None,
    seed_mask: np.ndarray | None = None,
) -> int:
    """Keep largest CC for one label (or the CC touching *seed_mask* if given).

    When *seed_mask* is provided (e.g. eICAB A1 for ACA), the component that
    overlaps the seed is retained even if a disconnected distal island is larger.
    """
    lid = int(label_id)
    n_before = int(np.count_nonzero(seg == lid))
    if seed_mask is not None:
        n_after = keep_component_touching_seed_inplace(seg, lid, seed_mask)
        mode = "seed-anchored-CC"
    else:
        n_after = keep_largest_component_label_inplace(seg, lid)
        mode = "largest-CC"
    dropped = n_before - n_after
    if dropped > 0 and log_name is not None:
        log.info(
            f"{mode} {log_name} (label {lid}): {n_before} → {n_after} "
            f"(dropped {dropped} island voxel(s))"
        )
    if stats_by_id is not None:
        st = stats_by_id.get(lid)
        if st is not None:
            st.n_voxels_final = int(n_after)
            st.n_voxels_after_island_clean = int(n_after)
    return n_after


def _distal_label_name(label_id: int) -> str:
    """Human vessel name for *label_id* (arterial or venous)."""
    from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

    return QVTPY_ARTERIAL_ID_TO_NAME.get(int(label_id)) or qvtpy_vessel_name(int(label_id))


def _aca_eicab_a1_seed_mask(
    eicab_qvtpy: np.ndarray | None,
    label_id: int,
) -> np.ndarray | None:
    """eICAB voxels for this ACA label (A1 / CW seed) used to anchor post-distal CC."""
    if eicab_qvtpy is None or int(label_id) not in QVTPY_ACA_IDS:
        return None
    seed = as_backend_array(eicab_qvtpy) == int(label_id)
    if not np.any(seed):
        return None
    return seed.astype(bool)


def _distal_lr_midline(
    seg: np.ndarray,
    left_id: int,
    right_id: int,
    *,
    axis: int = _DISTAL_LR_AXIS,
) -> int | None:
    """Midline coordinate on *axis* from L/R seed centroids (None if a side is missing)."""
    left = np.argwhere(seg == int(left_id))
    right = np.argwhere(seg == int(right_id))
    if left.size == 0 or right.size == 0:
        return None
    mid = 0.5 * (
        float(left[:, int(axis)].mean()) + float(right[:, int(axis)].mean())
    )
    return int(round(mid))


def _punch_lr_midline_barrier(
    tree: np.ndarray,
    mid: int,
    *,
    axis: int = _DISTAL_LR_AXIS,
    width: int = 1,
    restrict_to: np.ndarray | None = None,
) -> np.ndarray:
    """Zero a thin sagittal slab so L/R watershed basins cannot cross the midline.

    If ``restrict_to`` is set, only voxels inside that mask are cleared (so an
    MCA/PCA punch cannot carve through the ACA A2 corridor on the same X mid).
    """
    out = as_backend_array(tree).astype(bool, copy=True)
    ax = int(axis)
    w = max(0, int(width))
    lo = max(0, int(mid) - w)
    hi = min(int(out.shape[ax]), int(mid) + w + 1)
    slab = np.zeros(out.shape, dtype=bool)
    if ax == 0:
        slab[lo:hi, :, :] = True
    elif ax == 1:
        slab[:, lo:hi, :] = True
    else:
        slab[:, :, lo:hi] = True
    if restrict_to is not None:
        slab &= as_backend_array(restrict_to).astype(bool)
    out[slab] = False
    return out


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """6-connected binary dilation of *mask* by *radius* iterations; no-op for radius <= 0 or an empty
    mask."""
    m = as_backend_array(mask).astype(bool)
    r = max(0, int(radius))
    if r <= 0 or not np.any(m):
        return m
    # 6-connected dilation (base tool; brute_force chosen internally for cupy).
    return as_backend_array(dilate(m, iterations=r, connectivity=1)).astype(bool)


def _aca_distal_corridor(
    aca_seed_mask: np.ndarray,
    cd: np.ndarray,
    vesselness: np.ndarray,
    *,
    forbidden: np.ndarray,
    vesselness_floor: float,
    frangi_tree: np.ndarray | None = None,
    cd_percentile: float = _DISTAL_ACA_CORRIDOR_CD_PERCENTILE,
    max_dist_vox: float = _DISTAL_ACA_CORRIDOR_MAX_DIST_VOX,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow an ACA-only tree corridor from seeds into vesselness-gated CD.

    Reconnects distal A2 when Frangi CCs are broken, without a global CD flood.
    Also admits nearby pre-prune Frangi-tree voxels so orphaned A2 islands can
    rejoin through a short CD bridge.
    """
    seeds = as_backend_array(aca_seed_mask).astype(bool)
    if not np.any(seeds):
        return seeds, {"n_seeds": 0, "n_corridor": 0}
    cd_np = as_backend_array(cd).astype(np.float64)
    v = as_backend_array(vesselness).astype(np.float64)
    forb = as_backend_array(forbidden).astype(bool)
    cd_pos = cd_np[cd_np > 0]
    if cd_pos.size == 0:
        return seeds, {"n_seeds": int(np.count_nonzero(seeds)), "n_corridor": int(np.count_nonzero(seeds))}
    cd_thr = float(np.percentile(cd_pos, float(cd_percentile)))
    dist = ndi.distance_transform_edt(~seeds)
    near = dist <= float(max_dist_vox)
    # Soft floor: allow slightly weaker vesselness than hysteresis lowt.
    v_floor = float(vesselness_floor) * 0.5 if float(vesselness_floor) > 0 else 0.0
    gate = (cd_np >= cd_thr) & (v >= v_floor) & near & ~forb
    if frangi_tree is not None:
        # Orphaned A2 Frangi islands near ACA seeds (dropped by marker-CC prune).
        gate |= as_backend_array(frangi_tree).astype(bool) & near & ~forb
    gate |= seeds
    # 26-connectivity structuring element for the morphological reconstruction.
    structure = make_ball_footprint(3, connectivity=3)
    corridor = ndi.binary_propagation(seeds, structure=structure, mask=gate)
    meta = {
        "n_seeds": int(np.count_nonzero(seeds)),
        "n_corridor": int(np.count_nonzero(corridor)),
        "cd_threshold": cd_thr,
        "vesselness_floor": float(v_floor),
        "max_dist_vox": float(max_dist_vox),
        "n_gate": int(np.count_nonzero(gate)),
    }
    log.info(
        "distal expand: ACA corridor "
        f"seeds={meta['n_seeds']} → corridor={meta['n_corridor']} "
        f"(cd≥{cd_thr:.4g}, V≥{v_floor:.4g}, "
        f"max_dist={max_dist_vox:g})"
    )
    return corridor.astype(bool), meta


def _lr_halfspace_ok_mask(
    shape: tuple[int, int, int],
    mid: int,
    label_id: int,
    left_id: int,
    *,
    axis: int = _DISTAL_LR_AXIS,
    slack: int = FLOOD_LR_HALFSPACE_SLACK_DEFAULT,
) -> np.ndarray:
    """True where *label_id* is allowed to claim (ipsilateral half-space + slack)."""
    ok = np.ones(shape, dtype=bool)
    ax = int(axis)
    sl = int(slack)
    ji = int(mid)
    is_left = int(label_id) == int(left_id)
    if is_left:
        if ax == 0:
            ok[(ji + sl + 1) :, :, :] = False
        elif ax == 1:
            ok[:, (ji + sl + 1) :, :] = False
        else:
            ok[:, :, (ji + sl + 1) :] = False
    else:
        if ax == 0:
            ok[: max(0, ji - sl), :, :] = False
        elif ax == 1:
            ok[:, : max(0, ji - sl), :] = False
        else:
            ok[:, :, : max(0, ji - sl)] = False
    return ok


def _seed_communicating(
    seg: np.ndarray,
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    comm_label_ids: list[int],
    stats: list[VesselSegStats],
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
) -> None:
    """Seed the communicating arteries: their centerline, plus a local CD threshold crop around it
    (other centerlines dilated as barriers), into unlabelled voxels. They then only fill their own
    lumen in the flood (:func:`flood_expand_vessels`)."""
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in clm.shape[:3])
    for lid in comm_label_ids:
        roi = clm == lid
        if not np.any(roi):
            continue
        seg[roi & (seg == 0)] = int(lid)
        bbox_out = _bbox_with_vessel_padding(roi, shape, lid, default_pad=crop_padding_bbox)
        bbox, fp = bbox_out if bbox_out is not None else ((0, 0, 0, 0, 0, 0), bbox_padding_for_label(lid, 0))
        opt_t, warn = None, None
        if bbox_out is not None:
            i0, i1, j0, j1, k0, k1 = bbox
            crop_mask, opt_t, warn = _threshold_crop(
                cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1],
                thr_algorithm,
                min_component_frac=crop_min_fraction_for_label(lid),
            )
            cl_barrier = _dilated_other_centerlines_barrier(clm, lid, bbox, radius=_CL_BARRIER_RADIUS_BRANCHED)
            _paste_crop_mask(seg, crop_mask, lid, bbox, forbidden=cl_barrier)
        keep_component_touching_seed_inplace(seg, int(lid), roi)
        n = int(np.count_nonzero(seg == lid))
        stats.append(VesselSegStats(
            label_id=int(lid), bbox=bbox, face_padding=fp, thr_algorithm=f"centerline+{thr_algorithm}",
            opt_thresh=opt_t, n_voxels_after_threshold=n, n_voxels_after_island_clean=n, n_voxels_final=n,
            warning=warn,
        ))


def _near_field_fill(seg: np.ndarray, tree: np.ndarray, ids: list[int], radius: int) -> dict[str, int]:
    """The near-field part of the flood (trunks, communicating arteries, venous sinuses): the
    unlabelled vessel-tree voxels within *radius* of their own mask, each to the nearest of them
    (an artery and a sinus in contact split the voxels between them). Returns voxels added per
    label."""
    present = [int(lid) for lid in ids if np.any(seg == int(lid))]
    if not present or int(radius) <= 0:
        return {}
    markers = np.where(np.isin(seg, as_backend_array(present)), seg, 0).astype(np.int32)
    dist, idx = ndi.distance_transform_edt(markers == 0, return_indices=True)
    nearest = markers[tuple(idx)]
    claim = as_backend_array(tree).astype(bool) & (seg == 0) & (dist <= float(radius))
    seg[claim] = nearest[claim]
    out = {str(lid): int(np.count_nonzero(claim & (nearest == lid))) for lid in present}
    for lid in present:
        keep_largest_component_label_inplace(seg, int(lid))
    return out


def _keep_touching(seg: np.ndarray, label_id: int, anchor: np.ndarray) -> int:
    """Keep the components of *label_id* that touch *anchor* (26-connected); clear the rest."""
    m = seg == int(label_id)
    if not np.any(m):
        return 0
    cc, n = ndi.label(m, structure=make_ball_footprint(3, connectivity=3))
    if int(n) == 0:
        return 0
    near = ndi.binary_dilation(as_backend_array(anchor).astype(bool), structure=make_ball_footprint(3, connectivity=3))
    keep = np.unique(cc[near & m])
    keep = keep[keep > 0]
    drop = m & ~np.isin(cc, keep)
    seg[drop] = 0
    return int(np.count_nonzero(seg == int(label_id)))


def _split_distal(
    seg: np.ndarray, pre: dict[int, np.ndarray], radius: int
) -> dict[str, dict[str, int]]:
    """Write what each proximal label (A1 / M1 / P1) flooded beyond *radius* of its own pre-flood
    mask (*pre*) as its distal label (A2 / M2 / P2), and keep of each distal label the parts that
    touch its proximal one or its own pre-flood seeds."""
    out: dict[str, dict[str, int]] = {}
    for prox, dist_id in QVTPY_PROXIMAL_TO_DISTAL.items():
        own = pre.get(int(prox))
        if own is None or not np.any(own):
            continue
        far = (seg == int(prox)) & ~_dilate_bool(own, int(radius))
        n_far = int(np.count_nonzero(far))
        seg[far] = int(dist_id)
        seeds = pre.get(int(dist_id))
        anchor = (seg == int(prox)) | (seeds if seeds is not None else np.zeros_like(far))
        n_after = _keep_touching(seg, int(dist_id), anchor)
        out[_distal_label_name(int(dist_id))] = {"from": int(prox), "relabelled": n_far, "voxels": n_after}
        if n_far:
            log.info(f"flood split: {n_far} voxel(s) of {_distal_label_name(int(prox))} beyond its own lumen "
                     f"-> {_distal_label_name(int(dist_id))} ({n_after} voxels)")
    return out


def flood_expand_vessels(
    seg: np.ndarray,
    cd: np.ndarray,
    *,
    eicab_qvtpy: np.ndarray | None = None,
    max_image_frac: float = FLOOD_MAX_IMAGE_FRAC_DEFAULT,
    hyst_low_factor: float = FLOOD_HYST_LOW_FACTOR_DEFAULT,
    hyst_high_factor: float = FLOOD_HYST_HIGH_FACTOR_DEFAULT,
    frangi_sigmas: tuple[float, ...] | None = None,
    thicken_iter: int = FLOOD_THICKEN_ITER_DEFAULT,
    lr_halfspace_slack: int = FLOOD_LR_HALFSPACE_SLACK_DEFAULT,
    protrusion_filter: bool = True,
    near_radius: int = FLOOD_NEAR_RADIUS_DEFAULT,
) -> dict[str, Any]:
    """The blood flood of stage 4 (module docstring, step 3), *seg* in place.

    1. Frangi vesselness on CD (fallback: normalized CD) and a GMM + hysteresis vessel tree.
    2. The trunks, vertebrals, communicating arteries and venous sinuses (``_NEAR_FIELD_IDS``) fill
       the tree voxels within *near_radius* of their own mask.
    3. The other arteries (``_DISTAL_TARGET_IDS``: ACA / MCA / PCA, their distal labels, the SCAs)
       are watershed markers flooding the tree (only components touching them), all at once:
       - dilated ICA + basilar voxels are hard walls, as is every other label;
       - left / right pairs get a scoped midline punch (pair-local, ACA-protected) and claim
         half-space filters; the ACAs have none, but an ACA-only vesselness-gated CD corridor
         reconnects a distal A2 whose Frangi components broke at the AComm;
       - claims are capped per label and in total (*max_image_frac* of the volume);
       - each label keeps its largest component (an ACA the one on its eICAB A1).
    4. With *protrusion_filter*, a curvature / protrusion filter removes wart-like bumps, the
       pre-flood voxels protected.
    5. What an A1 / M1 / P1 flooded beyond *near_radius* of its own lumen is written as its
       A2 / M2 / P2 (:func:`_split_distal`).
    """
    from nvitk.pipes.qvtpy.util.segmentation.distal_vessel_tree import (
        _DISTAL_FRANGI_SIGMAS_DEFAULT,
        cd_vesselness,
        hysteresis_vessel_tree,
        keep_tree_components_touching_markers,
        thicken_tree_in_cd,
        watershed_labels_into_vessels,
    )
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    cd_np = as_backend_array(cd).astype(np.float64)
    eicab_np = None if eicab_qvtpy is None else as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    target_ids = sorted(int(x) for x in _DISTAL_TARGET_IDS)
    present = [lid for lid in target_ids if np.any(seg_np == lid)]
    sigmas = tuple(frangi_sigmas) if frangi_sigmas is not None else _DISTAL_FRANGI_SIGMAS_DEFAULT
    thick_n = max(0, int(thicken_iter))
    lr_slack = max(0, int(lr_halfspace_slack))
    log.step(
        "blood flood: "
        f"hyst_low={float(hyst_low_factor):.2f}, hyst_high={float(hyst_high_factor):.2f}, "
        f"max_image_frac={float(max_image_frac):.4f}, thicken_iter={thick_n}, lr_slack={lr_slack}, "
        f"near_radius={int(near_radius)}, protrusion_filter={bool(protrusion_filter)}, "
        f"frangi_sigmas={list(sigmas)}, "
        f"growing={[_distal_label_name(i) for i in present]}"
    )
    info: dict[str, Any] = {
        "enabled": True,
        "method": "frangi_hysteresis_watershed",
        "max_image_frac": float(max_image_frac),
        "hyst_low_factor": float(hyst_low_factor),
        "hyst_high_factor": float(hyst_high_factor),
        "thicken_iter": thick_n,
        "lr_halfspace_slack": lr_slack,
        "near_radius": int(near_radius),
        "protrusion_filter": bool(protrusion_filter),
        "frangi_sigmas": [float(s) for s in sigmas],
        "labels": {},
        "lr_midlines": {},
    }
    all_ids = sorted(int(v) for v in to_numpy(np.unique(seg_np)) if int(v) > 0)
    pre_masks = {lid: (seg_np == lid).copy() for lid in all_ids if lid in _DISTAL_TARGET_IDS}
    before_by_id = {lid: int(np.count_nonzero(seg_np == lid)) for lid in all_ids}

    cd_pos = cd_np[cd_np > 0]
    fg_mask = cd_np > float(np.percentile(cd_pos, 25.0)) if cd_pos.size > 0 else np.ones(cd_np.shape, dtype=bool)
    vesselness, vmode = cd_vesselness(cd_np, sigmas=sigmas)
    tree_hyst, tree_meta = hysteresis_vessel_tree(
        vesselness, fg_mask, low_factor=float(hyst_low_factor), high_factor=float(hyst_high_factor),
    )
    tree_hyst = as_backend_array(tree_hyst).astype(bool)
    info["vesselness_mode"] = vmode
    info["tree"] = tree_meta

    # 2. Trunks / vertebrals / comms / sinuses: their own lumen only.
    info["near_field"] = _near_field_fill(seg_np, tree_hyst, sorted(int(v) for v in _NEAR_FIELD_IDS), int(near_radius))
    if not present:
        log.warning("blood flood: no growing artery has seed voxels; near-field fill only")
        _record_flood_labels(info, seg_np, before_by_id)
        return info

    other = (seg_np != 0) & ~np.isin(seg_np, as_backend_array(target_ids))
    # Dilated ICA + basilar: nowhere in this band may a growing artery claim.
    ica_basilar = np.isin(seg_np, as_backend_array(sorted(int(v) for v in QVTPY_ICA_BASILAR_IDS)))
    ica_basilar_barrier = _dilate_bool(ica_basilar, _DISTAL_ICA_BASILAR_BARRIER_RADIUS)
    hard_barrier = other | ica_basilar_barrier
    info["ica_basilar_barrier"] = {
        "radius": int(_DISTAL_ICA_BASILAR_BARRIER_RADIUS),
        "n_core": int(np.count_nonzero(ica_basilar)),
        "n_dilated": int(np.count_nonzero(ica_basilar_barrier)),
    }

    markers = np.zeros(seg_np.shape, dtype=np.int32)
    for lid in present:
        markers[seg_np == lid] = lid
    # Drop Frangi CCs that never touch a growing artery (venous / noise forest).
    tree, touch_meta = keep_tree_components_touching_markers(tree_hyst, markers)
    info["tree_marker_cc"] = touch_meta
    if thick_n > 0:
        tree, thick_meta = thicken_tree_in_cd(tree, cd_np, iterations=thick_n, gate_percentile=85.0)
        info["tree_thicken"] = thick_meta
    tree = (tree & ~hard_barrier) | (markers != 0)

    # ACA corridor: reconnect A2 through vesselness-gated CD near the ACA seeds.
    aca_present = [lid for lid in present if lid in _ACA_FAMILY_IDS]
    aca_protect = np.zeros(seg_np.shape, dtype=bool)
    aca_corridor = np.zeros(seg_np.shape, dtype=bool)
    if aca_present:
        aca_seed = np.isin(seg_np, as_backend_array(aca_present)) & ~ica_basilar_barrier
        aca_protect = _dilate_bool(aca_seed, _DISTAL_ACA_PROTECT_RADIUS)
        v_floor = float(tree_meta.get("lowt", 0.0) or 0.0)
        if v_floor <= 0.0:
            v_pos = vesselness[vesselness > 0]
            v_floor = float(np.percentile(v_pos, 60.0)) if v_pos.size else 0.0
        aca_corridor, aca_corr_meta = _aca_distal_corridor(
            aca_seed, cd_np, vesselness, forbidden=hard_barrier, vesselness_floor=v_floor, frangi_tree=tree_hyst,
        )
        info["aca_corridor"] = aca_corr_meta
        tree = ((tree | aca_corridor) & ~hard_barrier) | (markers != 0)

    # Thin Frangi shell (anti-blob), but never strip the ACA corridor / protect zone.
    n_before_thin = int(np.count_nonzero(tree))
    if n_before_thin > 0:
        thr_thin = float(np.percentile(vesselness[tree], float(_DISTAL_TREE_VESSELNESS_KEEP_PERCENTILE)))
        keep_thin = (vesselness >= thr_thin) | aca_corridor | aca_protect | (markers != 0)
        tree = (tree & keep_thin) | (markers != 0) | aca_corridor
        tree = (tree & ~hard_barrier) | (markers != 0)
        info["tree_thin"] = {"percentile": float(_DISTAL_TREE_VESSELNESS_KEEP_PERCENTILE), "threshold": thr_thin,
                             "n_before": n_before_thin, "n_after": int(np.count_nonzero(tree))}

    claim_ok_by_id: dict[int, np.ndarray] = {}
    for left_id, right_id in _DISTAL_LR_PAIRS:
        if left_id not in present or right_id not in present:
            continue
        mid = _distal_lr_midline(seg_np, left_id, right_id, axis=_DISTAL_LR_AXIS)
        if mid is None:
            continue
        pair_name = f"{_distal_label_name(left_id)}/{_distal_label_name(right_id)}"
        info["lr_midlines"][pair_name] = int(mid)
        if left_id in _ACA_FAMILY_IDS:
            continue  # the ACAs hug the midline: left to the watershed
        pair_seed = (markers == left_id) | (markers == right_id)
        punch_zone = _dilate_bool(pair_seed, _DISTAL_LR_PUNCH_PAIR_RADIUS) & ~aca_protect
        tree = _punch_lr_midline_barrier(tree, mid, axis=_DISTAL_LR_AXIS, width=1, restrict_to=punch_zone)
        tree = ((tree | (markers != 0) | aca_corridor) & ~hard_barrier) | (markers != 0)
        claim_ok_by_id[left_id] = _lr_halfspace_ok_mask(seg_np.shape, mid, left_id, left_id,
                                                        axis=_DISTAL_LR_AXIS, slack=lr_slack)
        claim_ok_by_id[right_id] = _lr_halfspace_ok_mask(seg_np.shape, mid, right_id, left_id,
                                                         axis=_DISTAL_LR_AXIS, slack=lr_slack)

    n_tree = int(np.count_nonzero(tree))
    max_tree = int(max(1, round(0.01 * float(seg_np.size))))
    if n_tree > max_tree:
        thr_safe = float(np.percentile(vesselness[tree], 75.0))
        tree_tight = tree & ((vesselness >= thr_safe) | aca_corridor | aca_protect | (markers != 0))
        tree_tight = ((tree_tight | (markers != 0) | aca_corridor) & ~hard_barrier) | (markers != 0)
        tree, touch_meta2 = keep_tree_components_touching_markers(tree_tight, markers)
        tree = ((tree | aca_corridor) & ~hard_barrier) | (markers != 0)
        info["tree_safety"] = {"n_before": n_tree, "n_after": int(np.count_nonzero(tree)),
                               "vesselness_p75": thr_safe, "max_tree_vox": max_tree}
        log.warning(f"blood flood: vessel tree oversized ({n_tree} > {max_tree}); kept vesselness≥p75 "
                    f"(ACA-protected) → {info['tree_safety']['n_after']} voxels")
        n_tree = int(np.count_nonzero(tree))
        info["tree_marker_cc"] = touch_meta2
    if n_tree == 0:
        log.warning("blood flood: empty vessel tree; skipping the watershed")
        _record_flood_labels(info, seg_np, before_by_id)
        return info

    labeled = watershed_labels_into_vessels(tree, markers, connectivity=3, erode_markers=False)
    info["n_watershed_labeled"] = int(np.count_nonzero(labeled))
    max_add_total = int(max(1, round(float(max_image_frac) * float(seg_np.size))))
    added_total = 0
    capped: dict[int, bool] = {}
    for lid in present:
        before = before_by_id.get(lid, 0)
        claim = (labeled == lid) & (seg_np == 0) & ~ica_basilar_barrier
        if lid in claim_ok_by_id:
            claim = claim & claim_ok_by_id[lid]
        n_claim = int(np.count_nonzero(claim))
        per_label_cap = max(int(round(0.0015 * float(seg_np.size))), 5 * max(1, before))
        allow = min(n_claim, per_label_cap, max(0, max_add_total - added_total))
        capped[lid] = n_claim > allow
        if capped[lid]:
            claim_keep = np.zeros_like(claim)
            if allow > 0:  # the voxels nearest the label first
                coords = np.argwhere(claim)
                dist = ndi.distance_transform_edt(seg_np != lid)
                order = np.argsort(dist[coords[:, 0], coords[:, 1], coords[:, 2]])[:allow]
                sel = coords[order]
                claim_keep[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            claim = claim_keep
        n_claim = int(np.count_nonzero(claim))
        if n_claim > 0:
            seg_np[claim] = lid
            added_total += n_claim

    # Strip any spill on the ICA/basilar wall, then each label's component (ACA: on its eICAB A1).
    spill = np.isin(seg_np, as_backend_array(target_ids)) & ica_basilar_barrier & (markers == 0)
    if bool(spill.any()):
        seg_np[spill] = 0
        info["ica_basilar_barrier"]["n_spill_cleared"] = int(np.count_nonzero(spill))
    for lid in present:
        if lid in QVTPY_DISTAL_IDS:
            continue  # kept by their proximal label below
        _finalize_vessel_largest_cc(seg_np, lid, log_name=f"flood-{lid}",
                                    seed_mask=_aca_eicab_a1_seed_mask(eicab_np, lid))

    if bool(protrusion_filter):
        from nvitk.segmentation.protrusion_filter import filter_label_protrusions_in_seg

        log.step("blood flood: curvature / protrusion wart filter (pre-flood voxels protected)")
        info["protrusion_filter_result"] = filter_label_protrusions_in_seg(
            seg_np, present, protect_by_label={lid: pre_masks[lid] for lid in present if lid in pre_masks},
        )
        for lid in present:
            if lid not in QVTPY_DISTAL_IDS:
                _finalize_vessel_largest_cc(seg_np, lid, seed_mask=_aca_eicab_a1_seed_mask(eicab_np, lid))

    info["split_distal_result"] = _split_distal(seg_np, pre_masks, int(near_radius))
    for prox, dist_id in QVTPY_PROXIMAL_TO_DISTAL.items():  # a distal label stays on its proximal one
        if np.any(seg_np == int(dist_id)):
            _keep_touching(seg_np, int(dist_id), (seg_np == int(prox)) | pre_masks.get(int(dist_id),
                                                                                       np.zeros(seg_np.shape, bool)))
    _record_flood_labels(info, seg_np, before_by_id, capped)
    total_before = sum(int(v["before"]) for v in info["labels"].values())
    total_after = sum(int(v["after"]) for v in info["labels"].values())
    log.step(f"blood flood summary: {len(info['labels'])} label(s), voxels {total_before} → {total_after} "
             f"(Δ={total_after - total_before:+d})")
    return info


def _record_flood_labels(info: dict[str, Any], seg: np.ndarray, before_by_id: dict[int, int],
                         capped: dict[int, bool] | None = None) -> None:
    """Per-label before / after voxel counts of the flood into *info*."""
    ids = sorted(set(before_by_id) | {int(v) for v in to_numpy(np.unique(seg)) if int(v) > 0})
    for lid in ids:
        before, after = int(before_by_id.get(lid, 0)), int(np.count_nonzero(seg == lid))
        info["labels"][str(lid)] = {"name": _distal_label_name(lid), "before": before, "after": after,
                                    "delta": after - before, "capped": bool((capped or {}).get(lid, False))}


def build_seg_4dflow_local(
    cd: np.ndarray,
    centerlines_mask: np.ndarray,
    *,
    eicab_qvtpy: np.ndarray | None = None,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cl_barrier_radius: int = 2,
    segment_acomm: bool = False,
    split_vertebral: bool = True,
    flood_expand: bool = True,
    flood_hyst_low_factor: float = FLOOD_HYST_LOW_FACTOR_DEFAULT,
    flood_hyst_high_factor: float = FLOOD_HYST_HIGH_FACTOR_DEFAULT,
    flood_thicken_iter: int = FLOOD_THICKEN_ITER_DEFAULT,
    flood_max_image_frac: float = FLOOD_MAX_IMAGE_FRAC_DEFAULT,
    flood_lr_halfspace_slack: int = FLOOD_LR_HALFSPACE_SLACK_DEFAULT,
    flood_protrusion_filter: bool = True,
    flood_near_radius: int = FLOOD_NEAR_RADIUS_DEFAULT,
) -> LocalSegResult:
    """Build multilabel ``seg_4dflow`` from CD and the per-label centerline backbone (module
    docstring): local threshold per label, the vertebral split (*split_vertebral*), the
    communicating arteries' seeds (the AComm only with *segment_acomm*), then the blood flood
    (*flood_expand*; :func:`flood_expand_vessels` with the ``flood_*`` settings)."""
    cd = as_backend_array(cd).astype(np.float64)
    clm = as_backend_array(centerlines_mask).astype(np.int32, copy=False)
    eicab = None if eicab_qvtpy is None else as_backend_array(eicab_qvtpy).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in clm.shape[:3])
    seg = np.zeros(shape, dtype=np.int32)

    label_ids = sorted(int(v) for v in to_numpy(np.unique(clm)) if int(v) > 0)
    phase1_ids = [lid for lid in label_ids if lid not in QVTPY_COMM_IDS]
    comm_ids = [lid for lid in label_ids if lid in QVTPY_COMM_IDS]
    if not segment_acomm:
        comm_ids = [lid for lid in comm_ids if lid != int(QVTPY_ACOMM)]
    log.step(
        f"local CD segmentation: {len(phase1_ids)} label(s) + {len(comm_ids)} communicating, "
        f"thr={thr_algorithm}, flood={bool(flood_expand)}"
    )
    stats: list[VesselSegStats] = []
    cl_rad = max(0, int(cl_barrier_radius))
    branched = QVTPY_ACA_IDS | QVTPY_MCA_IDS | QVTPY_PCA_IDS | QVTPY_DISTAL_IDS

    for lid in phase1_ids:
        roi = clm == lid
        bbox_out = _bbox_with_vessel_padding(roi, shape, lid, default_pad=crop_padding_bbox)
        if bbox_out is None:
            stats.append(VesselSegStats(
                label_id=lid, bbox=(0, 0, 0, 0, 0, 0), face_padding=bbox_padding_for_label(lid, crop_padding_bbox),
                thr_algorithm=thr_algorithm, opt_thresh=None, n_voxels_after_threshold=0,
                n_voxels_after_island_clean=0, n_voxels_final=0, warning="empty centerline mask for label",
            ))
            continue
        bbox, face_pad = bbox_out
        i0, i1, j0, j1, k0, k1 = bbox
        crop_mask, opt_t, warn = _threshold_crop(
            cd[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1], thr_algorithm,
            min_component_frac=crop_min_fraction_for_label(lid),
        )
        cl_barrier = _dilated_other_centerlines_barrier(
            clm, lid, bbox, radius=_CL_BARRIER_RADIUS_BRANCHED if lid in branched else cl_rad
        )
        n_thr = _paste_crop_mask(seg, crop_mask, lid, bbox, forbidden=cl_barrier)
        stats.append(VesselSegStats(
            label_id=lid, bbox=bbox, face_padding=face_pad, thr_algorithm=thr_algorithm, opt_thresh=opt_t,
            n_voxels_after_threshold=n_thr, n_voxels_after_island_clean=n_thr, n_voxels_final=n_thr, warning=warn,
        ))

    # Largest CC after the local threshold.
    for lid in sorted(int(v) for v in to_numpy(np.unique(seg)) if int(v) != 0):
        keep_largest_component_label_inplace(seg, int(lid))
    for st in stats:
        st.n_voxels_after_island_clean = int(np.count_nonzero(seg == st.label_id))
    stats_by_id = {st.label_id: st for st in stats}

    vertebral_info: VertebralSplitResult | None = None
    if split_vertebral and int(QVTPY_BASILAR) in stats_by_id and np.any(seg == int(QVTPY_BASILAR)):
        bas_cl = np.argwhere(clm == int(QVTPY_BASILAR))
        seg, vertebral_info = split_vertebral_from_basilar(
            seg, prefer_basilar_centerline=bas_cl.astype(np.float64) if bas_cl.size else None,
        )
        if vertebral_info.split_applied:
            log.step(f"vertebral split OK: LVA={vertebral_info.lva_voxels} RVA={vertebral_info.rva_voxels} "
                     f"basilar={vertebral_info.basilar_voxels} conf={vertebral_info.confidence:.3f}")
        else:
            log.info(f"vertebral split: VAs absent ({vertebral_info.message or 'no VA bifurcation'})")
        for lid in (int(QVTPY_BASILAR), *sorted(int(x) for x in QVTPY_VERTEBRAL_IDS)):
            if np.any(seg == lid):
                _finalize_vessel_largest_cc(seg, lid, stats_by_id=stats_by_id, log_name=f"vertebral-{lid}")

    if comm_ids:
        _seed_communicating(seg, cd, clm, comm_label_ids=comm_ids, stats=stats,
                            crop_padding_bbox=int(crop_padding_bbox), thr_algorithm=thr_algorithm)
        stats_by_id = {st.label_id: st for st in stats}

    flood_info: dict[str, Any] | None = None
    if bool(flood_expand):
        flood_info = flood_expand_vessels(
            seg, cd, eicab_qvtpy=eicab,
            hyst_low_factor=float(flood_hyst_low_factor), hyst_high_factor=float(flood_hyst_high_factor),
            max_image_frac=float(flood_max_image_frac), thicken_iter=int(flood_thicken_iter),
            lr_halfspace_slack=int(flood_lr_halfspace_slack), protrusion_filter=bool(flood_protrusion_filter),
            near_radius=int(flood_near_radius),
        )
    log.step("final per-vessel connected component")
    for lid in sorted(int(v) for v in to_numpy(np.unique(seg)) if int(v) != 0):
        if lid in QVTPY_DISTAL_IDS:
            continue  # its branches meet only through its proximal label: kept by the flood
        _finalize_vessel_largest_cc(seg, lid, stats_by_id=stats_by_id, log_name=f"final-{lid}",
                                    seed_mask=_aca_eicab_a1_seed_mask(eicab, lid))
    present_ids = {int(v) for v in to_numpy(np.unique(seg)) if int(v) != 0}
    for st in stats:
        st.n_voxels_final = int(np.count_nonzero(seg == st.label_id))
        st.flooded = bool(flood_expand)
    for lid in sorted(present_ids - set(stats_by_id)):  # labels the flood created (A2 / M2 / P2)
        n = int(np.count_nonzero(seg == lid))
        stats.append(VesselSegStats(
            label_id=lid, bbox=(0, 0, 0, 0, 0, 0), face_padding=bbox_padding_for_label(lid, 0),
            thr_algorithm="flood", opt_thresh=None, n_voxels_after_threshold=0, n_voxels_after_island_clean=0,
            n_voxels_final=n, flooded=True,
        ))
    return LocalSegResult(
        segmentation=as_backend_array(seg.astype(np.int32, copy=False)),
        vessel_stats=stats,
        vertebral_split=vertebral_info,
        flood=flood_info,
    )


# Serialization


def vessel_stats_to_dict(st: VesselSegStats) -> dict[str, Any]:
    """JSON-serializable dict from :class:`VesselSegStats`."""
    i0, i1, j0, j1, k0, k1 = st.bbox
    return {
        "label_id": st.label_id,
        "name": _distal_label_name(st.label_id),
        "bbox": {"i0": i0, "i1": i1, "j0": j0, "j1": j1, "k0": k0, "k1": k1},
        "face_padding": st.face_padding.as_dict(),
        "thr_algorithm": st.thr_algorithm,
        "opt_thresh": st.opt_thresh,
        "n_voxels_after_threshold": st.n_voxels_after_threshold,
        "n_voxels_after_largest_cc": st.n_voxels_after_island_clean,
        "n_voxels_final": st.n_voxels_final,
        "flooded": st.flooded,
        "warning": st.warning,
    }


__all__ = [
    "BboxFacePadding",
    "FLOOD_HYST_HIGH_FACTOR_DEFAULT",
    "FLOOD_HYST_LOW_FACTOR_DEFAULT",
    "FLOOD_LR_HALFSPACE_SLACK_DEFAULT",
    "FLOOD_MAX_IMAGE_FRAC_DEFAULT",
    "FLOOD_NEAR_RADIUS_DEFAULT",
    "FLOOD_THICKEN_ITER_DEFAULT",
    "LocalSegResult",
    "ThrAlgorithm",
    "VESSEL_EXTRA_PADDING",
    "VesselSegStats",
    "bbox_padding_for_label",
    "build_seg_4dflow_local",
    "crop_min_fraction_for_label",
    "flood_expand_vessels",
    "vessel_stats_to_dict",
    "_bbox_with_padding",
]
