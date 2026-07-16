"""Geometry-based identification of venous sinuses (SSSV, STRV, LTSV, RTSV).

Uses RAS orientation from the image affine (when available), splits skeleton
branches at all junctions, and assigns vessels greedily with RAS direction
priors (preferred when an affine is provided) plus a light STRV conflict guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected
from nvitk.morphology.polyline_graph import branch_polylines_from_skeleton
from nvitk.pipes.qvtpy.labels import (
    NAME_LTSV,
    NAME_RTSV,
    NAME_SSSV,
    NAME_STRV,
    MATLAB_QVT_VENOUS_VESSEL_NAMES,
    VENOUS_LABEL_BY_NAME,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BRANCH_POINTS = 12
_MIN_ASSIGN_SCORE = 0.05

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
    x = to_numpy(v).astype(np.float64).ravel()[:3]
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _principal_direction(points: np.ndarray) -> np.ndarray:
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = pts - np.mean(pts, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _alignment_score(direction: np.ndarray, reference: np.ndarray) -> float:
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
) -> list[np.ndarray]:
    """Skeletonize each CC; one polyline per inter-junction chain."""
    m = to_numpy(venous_binary.astype(bool, copy=False))
    if not np.any(m):
        return []
    labeled, _ = label_connected(m, connectivity=1)
    lab = to_numpy(labeled)
    polylines: list[np.ndarray] = []
    for comp_id in range(1, int(lab.max()) + 1):
        comp = lab == comp_id
        sk = to_numpy(skeletonize_binary(comp))
        coords = np.argwhere(sk > 0)
        if coords.shape[0] < int(min_points):
            continue
        for poly in branch_polylines_from_skeleton(
            coords.astype(np.float32),
            min_points=min_points,
        ):
            polylines.append(poly.astype(np.float32, copy=False))
    return polylines


# ---------------------------------------------------------------------------
# RAS-aware SSSV / STRV / LTSV / RTSV scoring
# ---------------------------------------------------------------------------


def _score_branch_ras(
    points: np.ndarray,
    vessel_name: str,
    shape: tuple[int, int, int],
    axes: RasAxes,
    confluence_lr: float,
    confluence_ap: float,
    confluence_si: float,
) -> float:
    """Higher is better match for *vessel_name* using anatomical RAS priors."""
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 3:
        return 0.0

    lr, ap, si = _anatomical_coords(pts, axes)
    cx_lr, cx_ap, cx_si = float(np.mean(lr)), float(np.mean(ap)), float(np.mean(si))
    direction_ana = _oriented_direction_ana(
        pts, axes, confluence_lr, confluence_ap, confluence_si
    )
    length_score = float(pts.shape[0]) / max(shape)
    mid_scale = max(abs(confluence_lr) + 1.0, 1.0)

    conf_dist = np.sqrt(
        (cx_lr - confluence_lr) ** 2
        + (cx_ap - confluence_ap) ** 2
        + (cx_si - confluence_si) ** 2
    )
    conf_prox = 1.0 / (1.0 + conf_dist / max(shape))

    strv_ref_up = _unit(_RAS_DIR_PRIORS[NAME_STRV])
    strv_ref_dn = _unit(np.array([0.0, 1.0, -1.0], dtype=np.float64))

    if vessel_name == NAME_SSSV:
        # Midline sagittal, superior; must prefer SI over AP+SI.
        midline = 1.0 - abs(cx_lr) / mid_scale
        superior = max(0.0, cx_si) / max(shape[int(axes.si_axis)], 1)
        si = float(abs(direction_ana[2]))
        ap = float(abs(direction_ana[1]))
        lr_comp = float(abs(direction_ana[0]))
        # Strong SI preference; penalize AP / LR so diagonals cannot win SSSV.
        dir_score = max(0.0, si - 0.85 * ap - 0.5 * lr_comp)
        toward_conf = max(0.0, -direction_ana[2]) if cx_si > confluence_si else 0.5
        return length_score * (
            0.2 * midline + 0.2 * superior + 0.45 * dir_score + 0.15 * toward_conf
        )

    if vessel_name == NAME_STRV:
        # Midline; AP+SI prior (either SI sense); penalize pure SI and LR-dominant.
        midline = 1.0 - abs(cx_lr) / mid_scale
        align_strv = max(
            float(abs(np.dot(direction_ana, strv_ref_up))),
            float(abs(np.dot(direction_ana, strv_ref_dn))),
        )
        si = float(abs(direction_ana[2]))
        ap = float(abs(direction_ana[1]))
        lr_comp = float(abs(direction_ana[0]))
        # Geometric AP∧SI reward; penalize SI-only and transverse-like branches.
        balance = float(np.sqrt(max(ap, 1e-9) * max(si, 1e-9)))
        dir_score = max(
            0.0,
            0.55 * align_strv + 0.45 * balance - 0.55 * max(0.0, si - ap) - 0.55 * lr_comp,
        )
        return length_score * (0.25 * midline + 0.55 * dir_score + 0.2 * conf_prox)

    if vessel_name == NAME_LTSV:
        ref = _unit(_RAS_DIR_PRIORS[NAME_LTSV])
        lateral = 1.0 if cx_lr < confluence_lr else 0.12
        # Signed sense: Left + Anterior (principal oriented away from confluence).
        signed = float(np.dot(direction_ana, ref))
        sense = max(0.0, signed)
        return length_score * lateral * (0.35 + 0.5 * sense + 0.15 * max(0.0, -direction_ana[0]))

    if vessel_name == NAME_RTSV:
        ref = _unit(_RAS_DIR_PRIORS[NAME_RTSV])
        lateral = 1.0 if cx_lr > confluence_lr else 0.12
        signed = float(np.dot(direction_ana, ref))
        sense = max(0.0, signed)
        return length_score * lateral * (0.35 + 0.5 * sense + 0.15 * max(0.0, direction_ana[0]))

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
    """Voxel-index scoring (pre-RAS heuristic); robust for SSSV / STRV."""
    pts = to_numpy(points).astype(np.float64)
    nx, ny, nz = shape
    cx, cy, cz = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), float(np.mean(pts[:, 2]))
    direction = _principal_direction(pts)
    length_score = float(pts.shape[0]) / max(nx, ny, nz)
    mid_x = nx / 2.0

    if vessel_name == NAME_SSSV:
        sagittal = 1.0 - abs(cx - mid_x) / max(mid_x, 1.0)
        vertical = abs(direction[2])
        return length_score * (0.5 * sagittal + 0.5 * vertical)
    if vessel_name == NAME_STRV:
        align = _alignment_score(direction, _STRV_REF_VOX)
        return length_score * align
    if vessel_name == NAME_LTSV:
        lateral = 1.0 if cx < mid_x else 0.2
        transverse = abs(direction[0])
        return length_score * lateral * (0.5 + 0.5 * transverse)
    if vessel_name == NAME_RTSV:
        lateral = 1.0 if cx > mid_x else 0.2
        transverse = abs(direction[0])
        return length_score * lateral * (0.5 + 0.5 * transverse)
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


def _strv_geometry_ok(points: np.ndarray, axes: RasAxes) -> bool:
    """Reject STRV winners that are pure-SI or strongly LR (transverse-like)."""
    d = _direction_anatomical(_principal_direction(points), axes)
    si = float(abs(d[2]))
    ap = float(abs(d[1]))
    lr = float(abs(d[0]))
    if si > ap + 0.15:
        return False
    if lr > 0.55 and lr > ap:
        return False
    if ap < 0.25:
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
    """If STRV winner is SI- or LR-dominant, swap to a better unused candidate."""
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
        return {}
    axes = axes if axes is not None else resolve_ras_axes(affine)
    if prefer_ras is None:
        prefer_ras = affine is not None
    if confluence is None:
        conf_lr, conf_ap, conf_si = 0.0, float(shape[1]) * 0.4, float(shape[2]) * 0.7
    else:
        conf_lr, conf_ap, conf_si = confluence

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
    return assigned


def assign_venous_branches(
    venous_binary: np.ndarray,
    *,
    min_points: int = _MIN_BRANCH_POINTS,
    min_assign_score: float = _MIN_ASSIGN_SCORE,
    affine: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Assign junction-split skeleton chains to SSSV/STRV/LTSV/RTSV (0–4 vessels).

    Splits at all skeleton forks (not only the torcular confluence), then assigns
    branches greedily in vessel order using RAS direction priors (preferred when
    *affine* is set) with a light STRV conflict guard.
    """
    shape = tuple(int(s) for s in venous_binary.shape)
    axes = resolve_ras_axes(affine)
    candidates = extract_branch_polylines(venous_binary, min_points=min_points)
    if not candidates:
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
