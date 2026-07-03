"""Geometry-based identification of venous sinuses (SSSV, STRV, LTSV, RTSV).

Uses RAS orientation from the image affine (when available), splits skeleton
branches at all junctions, and assigns vessels greedily with blended RAS +
legacy voxel-index geometry scores.
"""

from __future__ import annotations

from collections import deque
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
    d = direction / (float(np.linalg.norm(direction)) + 1e-12)
    r = reference / (float(np.linalg.norm(reference)) + 1e-12)
    return float(abs(np.dot(d, r)))


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
    """Higher is better match for *vessel_name* using anatomical coordinates."""
    pts = to_numpy(points).astype(np.float64)
    if pts.shape[0] < 3:
        return 0.0

    lr, ap, si = _anatomical_coords(pts, axes)
    cx_lr, cx_ap, cx_si = float(np.mean(lr)), float(np.mean(ap)), float(np.mean(si))
    direction_ana = _direction_anatomical(_principal_direction(pts), axes)
    length_score = float(pts.shape[0]) / max(shape)

    # Distance from branch centroid to global confluence (superior slab center).
    conf_dist = np.sqrt(
        (cx_lr - confluence_lr) ** 2
        + (cx_ap - confluence_ap) ** 2
        + (cx_si - confluence_si) ** 2
    )
    conf_prox = 1.0 / (1.0 + conf_dist / max(shape))

    if vessel_name == NAME_SSSV:
        # Midline sagittal, superior, runs inferiorly toward confluence.
        midline = 1.0 - abs(cx_lr) / max(abs(confluence_lr) + 1.0, 1.0)
        superior = max(0.0, cx_si) / max(shape[int(axes.si_axis)], 1)
        vertical = abs(direction_ana[2])
        toward_conf = max(0.0, -direction_ana[2]) if cx_si > confluence_si else 0.5
        return length_score * (0.35 * midline + 0.25 * superior + 0.2 * vertical + 0.2 * toward_conf)

    if vessel_name == NAME_STRV:
        # Midline; align with legacy AP+SI reference (validated downstream in LOC stage).
        midline = 1.0 - abs(cx_lr) / max(abs(confluence_lr) + 1.0, 1.0)
        ref_ana = _direction_anatomical(_STRV_REF_VOX, axes)
        align = _alignment_score(direction_ana, ref_ana)
        near_conf = conf_prox
        return length_score * (0.4 * midline + 0.4 * align + 0.2 * near_conf)

    if vessel_name == NAME_LTSV:
        lateral = 1.0 if cx_lr < confluence_lr else 0.15
        transverse = abs(direction_ana[0])
        away_conf = max(0.0, -direction_ana[0]) if cx_lr < confluence_lr else max(0.0, direction_ana[0])
        return length_score * lateral * (0.45 + 0.35 * transverse + 0.2 * away_conf)

    if vessel_name == NAME_RTSV:
        lateral = 1.0 if cx_lr > confluence_lr else 0.15
        transverse = abs(direction_ana[0])
        away_conf = max(0.0, direction_ana[0]) if cx_lr > confluence_lr else max(0.0, -direction_ana[0])
        return length_score * lateral * (0.45 + 0.35 * transverse + 0.2 * away_conf)

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
) -> float:
    """Blend RAS-aware and legacy scores (best of both)."""
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
    return max(legacy, ras)


def assign_venous_branches(
    venous_binary: np.ndarray,
    *,
    min_points: int = _MIN_BRANCH_POINTS,
    min_assign_score: float = _MIN_ASSIGN_SCORE,
    affine: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Assign junction-split skeleton chains to SSSV/STRV/LTSV/RTSV (0–4 vessels).

    Splits at all skeleton forks (not only the torcular confluence), then assigns
    branches greedily in vessel order using blended RAS + legacy geometry scores.
    """
    shape = tuple(int(s) for s in venous_binary.shape)
    axes = resolve_ras_axes(affine)
    candidates = extract_branch_polylines(venous_binary, min_points=min_points)
    if not candidates:
        return {}

    conf_lr, conf_ap, conf_si = _estimate_confluence_center(venous_binary, axes)
    assigned: dict[str, np.ndarray] = {}
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
            )
            if sc > best_score:
                best_score = sc
                best_idx = idx
        if best_idx >= 0 and best_score > float(min_assign_score):
            assigned[name] = candidates[best_idx]
            used.add(best_idx)
    return assigned


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
    "extract_branch_polylines",
    "resolve_ras_axes",
    "venous_name_to_label_id",
]
