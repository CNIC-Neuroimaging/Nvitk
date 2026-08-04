"""Geometry-based identification of venous sinuses (SSSV, STRV, LTSV, RTSV).

Uses RAS orientation from the image affine (when available), splits skeleton
branches preferring significant multi-way junctions, then classic degree-3
Y/T junctions when no bifurcation is found (after pruning tiny loops / short
spurs), and assigns vessels greedily with RAS direction *and location*
priors (L/R X-hemisphere, midline SSSV/STRV, SSSV posterior to STRV) plus light
conflict guards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected
from nvitk.morphology.polyline_graph import (
    branch_polylines_from_skeleton,
    degree3_junction_nodes,
    prune_skeleton_coords_short_spurs,
    prune_skeleton_coords_tiny_loops,
    significant_bifurcation_nodes,
    three_arm_junction_nodes,
)
from nvitk.pipes.qvtpy.labels import (
    NAME_LTSV,
    NAME_RTSV,
    NAME_SSSV,
    NAME_STRV,
    MATLAB_QVT_VENOUS_VESSEL_NAMES,
    VENOUS_LABEL_BY_NAME,
)

log = Logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BRANCH_POINTS = 12
_MIN_ASSIGN_SCORE = 0.005

# Anatomical RAS direction priors (LR, AP, SI).
_RAS_DIR_PRIORS: dict[str, np.ndarray] = {
    NAME_SSSV: np.array([0.0, 0.0, 1.0], dtype=np.float64),  # SI
    NAME_STRV: np.array([0.0, 1.0, 1.0], dtype=np.float64),  # AP + SI
    NAME_LTSV: np.array([-1.0, 1.0, 0.0], dtype=np.float64),  # Left + Anterior
    NAME_RTSV: np.array([1.0, 1.0, 0.0], dtype=np.float64),  # Right + Anterior
}
# Legacy voxel-index STRV reference (used only in legacy scorer).
_STRV_REF_VOX = np.array([0.0, 1.0, 1.0], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VenousBranch:
    """One skeleton branch candidate."""

    name: str
    points: np.ndarray  # (N, 3) float32 voxel coords
    score: float


@dataclass(frozen=True)
class RasAxes:
    """Map array voxel axes to anatomical RAS directions."""

    lr_axis: int
    lr_sign: int  # +1: increasing index -> Right; -1 -> Left
    ap_axis: int
    ap_sign: int  # +1: increasing index -> Anterior
    si_axis: int
    si_sign: int  # +1: increasing index -> Superior


# ---------------------------------------------------------------------------
# RAS orientation
# ---------------------------------------------------------------------------


def _axis_for_code(codes: tuple[str, ...], letter: str) -> tuple[int, int]:
    """Return (array_axis, sign) for anatomical *letter* (R/L/A/P/S/I)."""
    up = letter.upper()
    for i, code in enumerate(codes[:3]):
        c = str(code).upper()
        if c == up:
            return i, 1
        if up == "R" and c == "L":
            return i, -1
        if up == "L" and c == "R":
            return i, -1
        if up == "A" and c == "P":
            return i, -1
        if up == "P" and c == "A":
            return i, -1
        if up == "S" and c == "I":
            return i, -1
        if up == "I" and c == "S":
            return i, -1
    return -1, 1


def resolve_ras_axes(affine: np.ndarray | None) -> RasAxes:
    """Resolve LR/AP/SI array axes and signs from a NIfTI affine (default RAS)."""
    default = RasAxes(lr_axis=0, lr_sign=1, ap_axis=1, ap_sign=1, si_axis=2, si_sign=1)
    if affine is None:
        return default
    try:
        import nibabel as nib

        aff = to_numpy(affine).astype(np.float64)
        codes = nib.orientations.aff2axcodes(aff[:3, :3])
    except Exception:
        return default

    lr_ax, lr_s = _axis_for_code(codes, "R")
    ap_ax, ap_s = _axis_for_code(codes, "A")
    si_ax, si_s = _axis_for_code(codes, "S")
    if lr_ax < 0 or ap_ax < 0 or si_ax < 0:
        return default
    return RasAxes(
        lr_axis=int(lr_ax),
        lr_sign=int(lr_s),
        ap_axis=int(ap_ax),
        ap_sign=int(ap_s),
        si_axis=int(si_ax),
        si_sign=int(si_s),
    )


def _anatomical_coords(
    points: np.ndarray,
    axes: RasAxes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (LR, AP, SI) anatomical coordinates for voxel *points* (N,3)."""
    pts = to_numpy(points).astype(np.float64)
    lr = float(axes.lr_sign) * pts[:, int(axes.lr_axis)]
    ap = float(axes.ap_sign) * pts[:, int(axes.ap_axis)]
    si = float(axes.si_sign) * pts[:, int(axes.si_axis)]
    return lr, ap, si


def _direction_anatomical(
    direction_vox: np.ndarray,
    axes: RasAxes,
) -> np.ndarray:
    """Unit direction in anatomical (LR, AP, SI) space."""
    d = to_numpy(direction_vox).astype(np.float64).ravel()[:3]
    ana = np.array(
        [
            float(axes.lr_sign) * d[int(axes.lr_axis)],
            float(axes.ap_sign) * d[int(axes.ap_axis)],
            float(axes.si_sign) * d[int(axes.si_axis)],
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(ana))
    return ana / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    """Unit-normalize 3-vector *v* (falls back to +SI if near-zero)."""
    x = to_numpy(v).astype(np.float64).ravel()[:3]
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _principal_direction(points: np.ndarray) -> np.ndarray:
    """Unit vector along the first principal component (SVD) of voxel *points*."""
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = pts - np.mean(pts, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _alignment_score(direction: np.ndarray, reference: np.ndarray) -> float:
    """Absolute cosine similarity between *direction* and *reference* (axis-agnostic, in [0, 1])."""
    d = _unit(direction)
    r = _unit(reference)
    return float(abs(np.dot(d, r)))


def _oriented_direction_ana(
    points: np.ndarray,
    axes: RasAxes,
    confluence_lr: float,
    confluence_ap: float,
    confluence_si: float,
) -> np.ndarray:
    """Principal direction in RAS, oriented away from the confluence."""
    pts = to_numpy(points).astype(np.float64)
    d = _direction_anatomical(_principal_direction(pts), axes)
    lr, ap, si = _anatomical_coords(pts, axes)
    conf = np.array(
        [float(confluence_lr), float(confluence_ap), float(confluence_si)],
        dtype=np.float64,
    )
    e0 = np.array([lr[0], ap[0], si[0]], dtype=np.float64)
    e1 = np.array([lr[-1], ap[-1], si[-1]], dtype=np.float64)
    if float(np.linalg.norm(e1 - conf)) >= float(np.linalg.norm(e0 - conf)):
        sense = e1 - e0
    else:
        sense = e0 - e1
    sense = _unit(sense)
    if float(np.dot(d, sense)) < 0.0:
        d = -d
    return d


# ---------------------------------------------------------------------------
# Skeleton branch extraction (split at all junctions / endpoints)
# ---------------------------------------------------------------------------


def extract_branch_polylines(
    venous_binary: np.ndarray,
    *,
    min_points: int = _MIN_BRANCH_POINTS,
    prefer_bifurcations: bool = True,
    prune_tiny_loops: bool = True,
    max_tiny_loop_len: int = 12,
    prune_short_spurs: bool = True,
) -> list[np.ndarray]:
    """Skeletonize each CC; one polyline per inter-junction / bifurcation chain.

    Default venous behaviour:
    - prune tiny skeleton loops and short spurs (oversegmentation artifacts)
    - prefer significant centerline bifurcations (≥3 long arms) as split points
    - if no significant bifurcation remains, keep one main path per CC
    """
    m = to_numpy(venous_binary.astype(bool, copy=False))
    if not np.any(m):
        log.info("venous extract: empty binary mask")
        return []
    labeled, _ = label_connected(m, connectivity=1)
    lab = to_numpy(labeled)
    n_cc = int(lab.max())
    log.step(
        f"venous extract: {n_cc} CC(s), min_points={int(min_points)}, "
        f"prefer_bifurcations={bool(prefer_bifurcations)}, "
        f"prune_loops={bool(prune_tiny_loops)}, prune_spurs={bool(prune_short_spurs)}"
    )
    polylines: list[np.ndarray] = []
    # Arm length gate for "significant" multi-way junction: keep modest so
    # torcular (degree ≥3 including ≥4) is found even when
    # --venous-min-branch-points is large (e.g. 30 → arm≥7, not 15).
    min_arm = max(4, min(8, int(min_points) // 4))
    # After junction split, sinus arms can be shorter than the CC gate; keep
    # them as assignable candidates (SSSV stub often < min_points).
    chain_min_points = max(12, int(min_points) // 2)
    for comp_id in range(1, n_cc + 1):
        comp = lab == comp_id
        n_vox = int(np.count_nonzero(comp))
        sk = to_numpy(skeletonize_binary(comp))
        coords = np.argwhere(sk > 0).astype(np.float32)
        n_skel = int(coords.shape[0])
        if n_skel < int(min_points):
            log.info(
                f"venous extract CC{comp_id}: skip (voxels={n_vox}, skel={n_skel} < {int(min_points)})"
            )
            continue

        n_after_loop = n_skel
        n_after_spur = n_skel
        work = coords
        # Only strip tiny noise spurs before split. Aggressive spur prune
        # (min_points//2) deletes real transverse arms and collapses SSSV↔LTSV
        # trees into a single chain (see PESA10758400).
        spur_min = max(2, min(4, int(min_points) // 8))
        if prune_tiny_loops:
            work = prune_skeleton_coords_tiny_loops(
                work, max_cycle_len=int(max_tiny_loop_len)
            )
            n_after_loop = int(work.shape[0])
        if prune_short_spurs:
            work = prune_skeleton_coords_short_spurs(
                work, min_spur_points=spur_min
            )
            n_after_spur = int(work.shape[0])

        n_bif = 0
        n_deg3 = 0
        n_three = 0
        if work.shape[0] >= int(min_points):
            if prefer_bifurcations:
                n_bif = len(
                    significant_bifurcation_nodes(work, min_arm_points=min_arm)
                )
            if n_bif == 0:
                n_deg3 = len(degree3_junction_nodes(work))
            if n_bif == 0 and n_deg3 == 0:
                n_three = len(
                    three_arm_junction_nodes(
                        work, min_arm_points=max(2, min_arm // 2)
                    )
                )
        if prefer_bifurcations and n_bif > 0:
            mode = f"multiway-split ({n_bif} junction(s))"
        elif n_deg3 > 0:
            mode = f"degree-3 junction fallback ({n_deg3} node(s))"
        elif n_three > 0:
            mode = f"3-arm junction fallback ({n_three} node(s))"
        else:
            mode = "longest-path (no bif / no Y junction)"
        log.info(
            f"venous extract CC{comp_id}: voxels={n_vox}, skel={n_skel} → "
            f"loop-prune={n_after_loop} → spur-prune={n_after_spur} "
            f"(spur_min={spur_min}); {mode}"
        )

        cc_polys = branch_polylines_from_skeleton(
            work,
            min_points=chain_min_points,
            prefer_bifurcations=bool(prefer_bifurcations),
            prune_tiny_loops=False,
            prune_short_spurs=False,
            min_bifurcation_arm_points=min_arm,
        )
        log.info(
            f"venous extract CC{comp_id}: {len(cc_polys)} chain(s) "
            f"lengths={[int(p.shape[0]) for p in cc_polys]}"
        )
        for poly in cc_polys:
            polylines.append(poly.astype(np.float32, copy=False))

    log.step(f"venous extract: {len(polylines)} candidate chain(s) total")
    return polylines


# ---------------------------------------------------------------------------
# RAS-aware SSSV / STRV / LTSV / RTSV scoring
# ---------------------------------------------------------------------------


def _lr_location_fractions(
    lr: np.ndarray,
    confluence_lr: float,
    *,
    mid_band: float,
) -> tuple[float, float, float]:
    """Fractions of points left of / right of / near confluence LR (midline X)."""
    frac_left = float(np.mean(lr < confluence_lr))
    frac_right = float(np.mean(lr > confluence_lr))
    frac_mid = float(np.mean(np.abs(lr - confluence_lr) <= mid_band))
    return frac_left, frac_right, frac_mid


def _midline_x_gate(
    lr: np.ndarray,
    cx_lr: float,
    confluence_lr: float,
    mid_band: float,
) -> float:
    """[0,1] how tightly points hug mid-X; near 0 if branch is hemispheric."""
    frac_mid = float(np.mean(np.abs(lr - confluence_lr) <= mid_band))
    # Median distance in units of mid_band (robust to long tails into a transverse).
    med_dist = float(np.median(np.abs(lr - confluence_lr)))
    prox = float(np.exp(-0.75 * (med_dist / max(mid_band, 1.0)) ** 2))
    mean_prox = float(
        np.clip(1.0 - abs(cx_lr - confluence_lr) / max(2.5 * mid_band, 1.0), 0.0, 1.0)
    )
    gate = 0.50 * frac_mid + 0.30 * prox + 0.20 * mean_prox
    # Hard reject strongly lateral branches for midline vessels.
    if frac_mid < 0.35 and med_dist > 1.25 * mid_band:
        gate *= 0.05
    elif frac_mid < 0.50 and abs(cx_lr - confluence_lr) > 1.5 * mid_band:
        gate *= 0.08
    return float(np.clip(gate, 0.0, 1.0))


def _hemisphere_x_gate(
    *,
    side: str,
    frac_left: float,
    frac_right: float,
    frac_mid: float,
    cx_lr: float,
    confluence_lr: float,
    mid_band: float,
) -> float:
    """[0,1] occupancy of left/right X half; near 0 for midline-only branches."""
    if side == "left":
        frac_side = frac_left
        on_side = cx_lr < confluence_lr
        depth = (confluence_lr - cx_lr) / max(mid_band, 1.0)
    else:
        frac_side = frac_right
        on_side = cx_lr > confluence_lr
        depth = (cx_lr - confluence_lr) / max(mid_band, 1.0)
    # Prefer branches whose mass is clearly off-midline on the correct side.
    lateral_mass = float(np.clip(frac_side - 0.5 * frac_mid, 0.0, 1.0))
    depth_score = float(np.clip(depth / 1.5, 0.0, 1.0))
    gate = 0.55 * frac_side + 0.25 * lateral_mass + 0.20 * depth_score
    if not on_side or frac_side < 0.55:
        gate *= 0.04
    elif frac_mid > 0.70 and depth < 0.5:
        # Almost entirely midline — not a transverse.
        gate *= 0.08
    return float(np.clip(gate, 0.0, 1.0))


def _score_branch_ras(
    points: np.ndarray,
    vessel_name: str,
    shape: tuple[int, int, int],
    axes: RasAxes,
    confluence_lr: float,
    confluence_ap: float,
    confluence_si: float,
) -> float:
    """Higher is better match for *vessel_name* using anatomical RAS priors.

    X-axis location is a hard multiplicative gate (not a soft additive term):
    - SSSV / STRV: must hug mid-X around confluence
    - LTSV / RTSV: must occupy left / right X half
    SSSV vs STRV still uses AP (SSSV more posterior) when both are midline.
    """
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 3:
        return 0.0

    lr, ap, si = _anatomical_coords(pts, axes)
    cx_lr, cx_ap, cx_si = float(np.mean(lr)), float(np.mean(ap)), float(np.mean(si))
    direction_ana = _oriented_direction_ana(
        pts, axes, confluence_lr, confluence_ap, confluence_si
    )
    length_score = float(pts.shape[0]) / max(shape)
    # Mid band ~12% of FOV half-width so transverses fall outside quickly.
    mid_scale = max(float(shape[int(axes.lr_axis)]) * 0.5, 1.0)
    mid_band = max(0.12 * mid_scale, 6.0)
    frac_left, frac_right, frac_mid = _lr_location_fractions(
        lr, confluence_lr, mid_band=mid_band
    )
    mid_gate = _midline_x_gate(lr, cx_lr, confluence_lr, mid_band)
    # Posterior = lower anatomical AP (RAS +A). Soft score in [0, 1].
    ap_span = max(float(shape[int(axes.ap_axis)]), 1.0)
    posterior = float(np.clip(0.5 + (confluence_ap - cx_ap) / ap_span, 0.0, 1.0))
    anterior = 1.0 - posterior

    conf_dist = np.sqrt(
        (cx_lr - confluence_lr) ** 2
        + (cx_ap - confluence_ap) ** 2
        + (cx_si - confluence_si) ** 2
    )
    conf_prox = 1.0 / (1.0 + conf_dist / max(shape))

    strv_ref_up = _unit(_RAS_DIR_PRIORS[NAME_STRV])
    strv_ref_dn = _unit(np.array([0.0, 1.0, -1.0], dtype=np.float64))

    if vessel_name == NAME_SSSV:
        superior = max(0.0, (cx_si - confluence_si) / max(shape[int(axes.si_axis)], 1))
        si_d = float(abs(direction_ana[2]))
        ap_d = float(abs(direction_ana[1]))
        lr_d = float(abs(direction_ana[0]))
        dir_score = max(0.0, si_d - 0.5 * ap_d - 0.6 * lr_d)
        toward_conf = max(0.0, -direction_ana[2]) if cx_si > confluence_si else 0.5
        content = (
            0.35 * posterior
            + 0.25 * superior
            + 0.25 * dir_score
            + 0.15 * toward_conf
        )
        return length_score * mid_gate * content

    if vessel_name == NAME_STRV:
        align_strv = max(
            float(abs(np.dot(direction_ana, strv_ref_up))),
            float(abs(np.dot(direction_ana, strv_ref_dn))),
        )
        si_d = float(abs(direction_ana[2]))
        ap_d = float(abs(direction_ana[1]))
        lr_d = float(abs(direction_ana[0]))
        balance = float(np.sqrt(max(ap_d, 1e-9) * max(si_d, 1e-9)))
        dir_score = max(
            0.0,
            0.45 * align_strv + 0.35 * balance + 0.25 * si_d - 0.70 * lr_d,
        )
        content = 0.40 * anterior + 0.35 * dir_score + 0.25 * conf_prox
        return length_score * mid_gate * content

    if vessel_name == NAME_LTSV:
        hemi = _hemisphere_x_gate(
            side="left",
            frac_left=frac_left,
            frac_right=frac_right,
            frac_mid=frac_mid,
            cx_lr=cx_lr,
            confluence_lr=confluence_lr,
            mid_band=mid_band,
        )
        ref = _unit(_RAS_DIR_PRIORS[NAME_LTSV])
        sense = max(0.0, float(np.dot(direction_ana, ref)))
        lr_run = float(abs(direction_ana[0]))
        content = 0.50 + 0.30 * sense + 0.20 * lr_run
        return length_score * hemi * content

    if vessel_name == NAME_RTSV:
        hemi = _hemisphere_x_gate(
            side="right",
            frac_left=frac_left,
            frac_right=frac_right,
            frac_mid=frac_mid,
            cx_lr=cx_lr,
            confluence_lr=confluence_lr,
            mid_band=mid_band,
        )
        ref = _unit(_RAS_DIR_PRIORS[NAME_RTSV])
        sense = max(0.0, float(np.dot(direction_ana, ref)))
        lr_run = float(abs(direction_ana[0]))
        content = 0.50 + 0.30 * sense + 0.20 * lr_run
        return length_score * hemi * content

    return length_score


def _estimate_confluence_center(
    venous_binary: np.ndarray,
    axes: RasAxes,
) -> tuple[float, float, float]:
    """Estimate confluence as superior + posterior centroid of venous foreground."""
    m = to_numpy(venous_binary.astype(bool))
    if not np.any(m):
        shape = venous_binary.shape
        return 0.0, 0.0, float(shape[int(axes.si_axis)]) * 0.5
    coords = np.argwhere(m)
    lr, ap, si = _anatomical_coords(coords.astype(np.float64), axes)
    # Weight superior and posterior voxels more heavily (confluence region).
    w = np.exp(0.15 * (si - float(np.median(si)))) * np.exp(-0.1 * (ap - float(np.min(ap))))
    w = w / (float(np.sum(w)) + 1e-12)
    return (
        float(np.sum(w * lr)),
        float(np.sum(w * ap)),
        float(np.sum(w * si)),
    )


def _score_branch_legacy(
    points: np.ndarray,
    vessel_name: str,
    shape: tuple[int, int, int],
) -> float:
    """Voxel-index scoring with hard mid-X / hemisphere gates on image X."""
    pts = to_numpy(points).astype(np.float64)
    nx, ny, nz = shape
    cx, cy, cz = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), float(np.mean(pts[:, 2]))
    direction = _principal_direction(pts)
    length_score = float(pts.shape[0]) / max(nx, ny, nz)
    mid_x = nx / 2.0
    mid_band = max(0.12 * mid_x, 6.0)
    frac_left = float(np.mean(pts[:, 0] < mid_x))
    frac_right = float(np.mean(pts[:, 0] > mid_x))
    frac_mid = float(np.mean(np.abs(pts[:, 0] - mid_x) <= mid_band))
    mid_gate = _midline_x_gate(pts[:, 0], cx, mid_x, mid_band)
    posterior = float(np.clip(1.0 - cy / max(ny, 1.0), 0.0, 1.0))
    anterior = 1.0 - posterior

    if vessel_name == NAME_SSSV:
        vertical = abs(direction[2])
        return length_score * mid_gate * (0.45 * posterior + 0.55 * vertical)
    if vessel_name == NAME_STRV:
        align = _alignment_score(direction, _STRV_REF_VOX)
        si = abs(direction[2])
        return length_score * mid_gate * (0.45 * anterior + 0.35 * align + 0.20 * si)
    if vessel_name == NAME_LTSV:
        hemi = _hemisphere_x_gate(
            side="left",
            frac_left=frac_left,
            frac_right=frac_right,
            frac_mid=frac_mid,
            cx_lr=cx,
            confluence_lr=mid_x,
            mid_band=mid_band,
        )
        transverse = abs(direction[0])
        return length_score * hemi * (0.45 + 0.55 * transverse)
    if vessel_name == NAME_RTSV:
        hemi = _hemisphere_x_gate(
            side="right",
            frac_left=frac_left,
            frac_right=frac_right,
            frac_mid=frac_mid,
            cx_lr=cx,
            confluence_lr=mid_x,
            mid_band=mid_band,
        )
        transverse = abs(direction[0])
        return length_score * hemi * (0.45 + 0.55 * transverse)
    return length_score


def _score_branch(
    points: np.ndarray,
    vessel_name: str,
    shape: tuple[int, int, int],
    axes: RasAxes,
    confluence_lr: float,
    confluence_ap: float,
    confluence_si: float,
    *,
    prefer_ras: bool,
) -> float:
    """Prefer RAS priors when affine is available; otherwise best of RAS/legacy."""
    legacy = _score_branch_legacy(points, vessel_name, shape)
    ras = _score_branch_ras(
        points,
        vessel_name,
        shape,
        axes,
        confluence_lr,
        confluence_ap,
        confluence_si,
    )
    if prefer_ras:
        # Affine-backed RAS must outweigh voxel-index legacy geometry.
        return float(0.9 * ras + 0.1 * legacy)
    return max(legacy, ras)


def _mean_ap(points: np.ndarray, axes: RasAxes) -> float:
    """Mean anatomical AP coordinate of *points*."""
    lr, ap, si = _anatomical_coords(to_numpy(points).astype(np.float64), axes)
    return float(np.mean(ap))


def _mean_lr(points: np.ndarray, axes: RasAxes) -> float:
    """Mean anatomical LR coordinate of *points*."""
    lr, ap, si = _anatomical_coords(to_numpy(points).astype(np.float64), axes)
    return float(np.mean(lr))


def _strv_geometry_ok(points: np.ndarray, axes: RasAxes) -> bool:
    """Reject STRV winners that are strongly LR (transverse-like).

    Pure-SI STRV is allowed: some subjects have a nearly vertical straight
    sinus; SSSV vs STRV is resolved by AP location (SSSV more posterior).
    """
    d = _direction_anatomical(_principal_direction(points), axes)
    ap = float(abs(d[1]))
    lr = float(abs(d[0]))
    if lr > 0.55 and lr > ap:
        return False
    return True


def _resolve_strv_conflict(
    assigned: dict[str, np.ndarray],
    assigned_idx: dict[str, int],
    candidates: list[np.ndarray],
    used: set[int],
    shape: tuple[int, int, int],
    axes: RasAxes,
    confluence_lr: float,
    confluence_ap: float,
    confluence_si: float,
    *,
    prefer_ras: bool,
    min_assign_score: float,
) -> None:
    """If STRV winner is LR-dominant (transverse-like), swap to a better unused candidate."""
    if NAME_STRV not in assigned:
        return
    cur_idx = int(assigned_idx[NAME_STRV])
    cur = candidates[cur_idx]
    if _strv_geometry_ok(cur, axes):
        return

    best_idx = -1
    best_score = -1.0
    for idx, poly in enumerate(candidates):
        if idx in used and idx != cur_idx:
            continue
        if not _strv_geometry_ok(poly, axes):
            continue
        sc = _score_branch(
            poly,
            NAME_STRV,
            shape,
            axes,
            confluence_lr,
            confluence_ap,
            confluence_si,
            prefer_ras=prefer_ras,
        )
        if sc > best_score:
            best_score = sc
            best_idx = idx
    if best_idx < 0 or best_score <= float(min_assign_score):
        return
    if best_idx == cur_idx:
        return
    used.discard(cur_idx)
    used.add(best_idx)
    assigned[NAME_STRV] = candidates[best_idx]
    assigned_idx[NAME_STRV] = best_idx
    log.info(
        f"venous assign: STRV conflict swap cand[{cur_idx}] → cand[{best_idx}] "
        f"(score={best_score:.4f}, n={int(candidates[best_idx].shape[0])})"
    )


def _resolve_sssv_strv_ap_order(
    assigned: dict[str, np.ndarray],
    assigned_idx: dict[str, int],
    axes: RasAxes,
) -> None:
    """SSSV must sit posterior (lower AP / behind) relative to STRV; swap if inverted."""
    if NAME_SSSV not in assigned or NAME_STRV not in assigned:
        return
    sssv_ap = _mean_ap(assigned[NAME_SSSV], axes)
    strv_ap = _mean_ap(assigned[NAME_STRV], axes)
    if sssv_ap <= strv_ap + 1.0:
        return
    i_s = int(assigned_idx[NAME_SSSV])
    i_t = int(assigned_idx[NAME_STRV])
    assigned[NAME_SSSV], assigned[NAME_STRV] = assigned[NAME_STRV], assigned[NAME_SSSV]
    assigned_idx[NAME_SSSV], assigned_idx[NAME_STRV] = i_t, i_s
    log.info(
        f"venous assign: SSSV↔STRV AP-order swap "
        f"(SSSV was more anterior: AP {sssv_ap:.1f} > STRV {strv_ap:.1f})"
    )


def _lr_abs_dev(points: np.ndarray, axes: RasAxes, confluence_lr: float) -> float:
    """Absolute LR distance from *points*' mean position to *confluence_lr*."""
    return abs(_mean_lr(points, axes) - float(confluence_lr))


def _resolve_sssv_vs_transverse_x(
    assigned: dict[str, np.ndarray],
    assigned_idx: dict[str, int],
    axes: RasAxes,
    confluence_lr: float,
    shape: tuple[int, int, int],
) -> None:
    """If SSSV is lateral on X and a transverse is more midline, swap them."""
    if NAME_SSSV not in assigned:
        return
    mid_band = max(0.12 * float(shape[int(axes.lr_axis)]) * 0.5, 6.0)
    sssv = assigned[NAME_SSSV]
    sssv_lr = _mean_lr(sssv, axes)
    sssv_dev = abs(sssv_lr - confluence_lr)
    if sssv_dev <= 1.25 * mid_band:
        return

    def _maybe_swap(trans_name: str) -> bool:
        """Swap SSSV with the *trans_name* transverse sinus if that transverse sits closer to
        midline than SSSV does; returns True if a swap was made."""
        if trans_name not in assigned:
            return False
        other = assigned[trans_name]
        other_dev = _lr_abs_dev(other, axes, confluence_lr)
        # Transverse should be more lateral; if it is closer to mid-X than SSSV, swap.
        if other_dev + 0.5 * mid_band >= sssv_dev:
            return False
        i_s = int(assigned_idx[NAME_SSSV])
        i_t = int(assigned_idx[trans_name])
        assigned[NAME_SSSV], assigned[trans_name] = other, sssv
        assigned_idx[NAME_SSSV], assigned_idx[trans_name] = i_t, i_s
        log.info(
            f"venous assign: SSSV↔{trans_name} mid-X swap "
            f"(SSSV |ΔLR|={sssv_dev:.1f} > {trans_name} |ΔLR|={other_dev:.1f}, "
            f"mid_band={mid_band:.1f})"
        )
        return True

    if sssv_lr > confluence_lr:
        _maybe_swap(NAME_RTSV)
    else:
        _maybe_swap(NAME_LTSV)


def _resolve_transverse_lr_order(
    assigned: dict[str, np.ndarray],
    assigned_idx: dict[str, int],
    axes: RasAxes,
    confluence_lr: float,
) -> None:
    """LTSV must be left of mid-X / confluence; RTSV right; swap if inverted."""
    if NAME_LTSV not in assigned or NAME_RTSV not in assigned:
        return
    l_lr = _mean_lr(assigned[NAME_LTSV], axes)
    r_lr = _mean_lr(assigned[NAME_RTSV], axes)
    # Already correctly ordered relative to each other and confluence.
    if l_lr < r_lr and l_lr <= confluence_lr + 1.0 and r_lr >= confluence_lr - 1.0:
        return
    if l_lr < r_lr:
        return
    i_l = int(assigned_idx[NAME_LTSV])
    i_r = int(assigned_idx[NAME_RTSV])
    assigned[NAME_LTSV], assigned[NAME_RTSV] = assigned[NAME_RTSV], assigned[NAME_LTSV]
    assigned_idx[NAME_LTSV], assigned_idx[NAME_RTSV] = i_r, i_l
    log.info(
        f"venous assign: LTSV↔RTSV LR-order swap "
        f"(LTSV was more right: LR {l_lr:.1f} ≥ RTSV {r_lr:.1f})"
    )


def assign_venous_polylines(
    candidates: list[np.ndarray],
    shape: tuple[int, int, int],
    *,
    axes: RasAxes | None = None,
    confluence: tuple[float, float, float] | None = None,
    affine: np.ndarray | None = None,
    min_assign_score: float = _MIN_ASSIGN_SCORE,
    prefer_ras: bool | None = None,
) -> dict[str, np.ndarray]:
    """Greedy SSSV→STRV→LTSV→RTSV assignment from junction-split polylines."""
    if not candidates:
        log.info("venous assign: no candidate chains")
        return {}
    axes = axes if axes is not None else resolve_ras_axes(affine)
    if prefer_ras is None:
        prefer_ras = affine is not None
    if confluence is None:
        conf_lr, conf_ap, conf_si = 0.0, float(shape[1]) * 0.4, float(shape[2]) * 0.7
    else:
        conf_lr, conf_ap, conf_si = confluence

    log.step(
        f"venous assign: {len(candidates)} candidate(s), prefer_ras={bool(prefer_ras)}, "
        f"confluence_RAS=({conf_lr:.1f},{conf_ap:.1f},{conf_si:.1f})"
    )

    assigned: dict[str, np.ndarray] = {}
    assigned_idx: dict[str, int] = {}
    used: set[int] = set()
    for name in MATLAB_QVT_VENOUS_VESSEL_NAMES:
        best_idx = -1
        best_score = -1.0
        for idx, poly in enumerate(candidates):
            if idx in used:
                continue
            sc = _score_branch(
                poly,
                name,
                shape,
                axes,
                conf_lr,
                conf_ap,
                conf_si,
                prefer_ras=prefer_ras,
            )
            if sc > best_score:
                best_score = sc
                best_idx = idx
        if best_idx >= 0 and best_score > float(min_assign_score):
            assigned[name] = candidates[best_idx]
            assigned_idx[name] = best_idx
            used.add(best_idx)
            log.info(
                f"venous assign: {name} ← cand[{best_idx}] "
                f"score={best_score:.4f} n={int(candidates[best_idx].shape[0])}"
            )
        else:
            log.info(
                f"venous assign: {name} skipped "
                f"(best_score={best_score:.4f}, min={float(min_assign_score):.4f})"
            )

    before_strv = assigned_idx.get(NAME_STRV)
    _resolve_strv_conflict(
        assigned,
        assigned_idx,
        candidates,
        used,
        shape,
        axes,
        conf_lr,
        conf_ap,
        conf_si,
        prefer_ras=prefer_ras,
        min_assign_score=min_assign_score,
    )
    after_strv = assigned_idx.get(NAME_STRV)
    if before_strv is not None and after_strv == before_strv:
        log.info("venous assign: STRV conflict guard kept original winner")

    _resolve_sssv_strv_ap_order(assigned, assigned_idx, axes)
    _resolve_sssv_vs_transverse_x(assigned, assigned_idx, axes, conf_lr, shape)
    _resolve_transverse_lr_order(assigned, assigned_idx, axes, conf_lr)

    log.step(
        f"venous assign: labeled {list(assigned.keys())} "
        f"({len(assigned)}/{len(MATLAB_QVT_VENOUS_VESSEL_NAMES)})"
    )
    return assigned


def assign_venous_branches(
    venous_binary: np.ndarray,
    *,
    min_points: int = _MIN_BRANCH_POINTS,
    min_assign_score: float = _MIN_ASSIGN_SCORE,
    affine: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Assign junction-split skeleton chains to SSSV/STRV/LTSV/RTSV (0–4 vessels).

    Prefers significant bifurcations after tiny-loop/spur pruning, then assigns
    branches greedily in vessel order using RAS direction priors (preferred when
    *affine* is set) with a light STRV conflict guard.
    """
    shape = tuple(int(s) for s in venous_binary.shape)
    n_fg = int(np.count_nonzero(venous_binary))
    log.step(
        f"venous labeling: binary fg={n_fg}, shape={shape}, "
        f"min_points={int(min_points)}, affine={'yes' if affine is not None else 'no'}"
    )
    axes = resolve_ras_axes(affine)
    log.info(
        f"venous labeling: RAS axes lr={axes.lr_axis}/{axes.lr_sign}, "
        f"ap={axes.ap_axis}/{axes.ap_sign}, si={axes.si_axis}/{axes.si_sign}"
    )
    candidates = extract_branch_polylines(venous_binary, min_points=min_points)
    if not candidates:
        log.info("venous labeling: no candidates after extract")
        return {}

    conf_lr, conf_ap, conf_si = _estimate_confluence_center(venous_binary, axes)
    return assign_venous_polylines(
        candidates,
        shape,
        axes=axes,
        confluence=(conf_lr, conf_ap, conf_si),
        affine=affine,
        min_assign_score=min_assign_score,
        prefer_ras=affine is not None,
    )


# ---- Label id mapping --------------------------------------------------------


def venous_name_to_label_id(name: str, name_to_id: dict[str, int] | None = None) -> int:
    """Map venous vessel name to fixed segmentation label id (31–34)."""
    if name_to_id and name in name_to_id:
        return int(name_to_id[name])
    key = name.strip().upper()
    if key in VENOUS_LABEL_BY_NAME:
        return int(VENOUS_LABEL_BY_NAME[key])
    for k, v in VENOUS_LABEL_BY_NAME.items():
        if k.upper() == key:
            return int(v)
    return int(VENOUS_LABEL_BY_NAME[NAME_SSSV])


__all__ = [
    "RasAxes",
    "VenousBranch",
    "assign_venous_branches",
    "assign_venous_polylines",
    "extract_branch_polylines",
    "resolve_ras_axes",
    "venous_name_to_label_id",
]
