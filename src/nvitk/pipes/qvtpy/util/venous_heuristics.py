"""Geometry-based identification of venous sinuses (SSSV, STRV, LTSV, RTSV)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.morphology.centerline import _centerline_longest_path, skeletonize_binary
from nvitk.morphology.components import label_connected
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

_STRV_REF = np.array([0.0, 1.0, 1.0], dtype=np.float64)
_MIN_BRANCH_POINTS = 12
_MIN_ASSIGN_SCORE = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VenousBranch:
    """One skeleton branch candidate."""

    name: str
    points: np.ndarray  # (N, 3) float32 voxel coords
    score: float


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _neighbors26(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = p
    out: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                out.append((x + dx, y + dy, z + dz))
    return out


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


def _chain_key(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Skeleton branch extraction (split at junctions / endpoints)
# ---------------------------------------------------------------------------


def _skeleton_graph(coords_xyz: np.ndarray) -> tuple[
    list[tuple[int, int, int]],
    dict[tuple[int, int, int], list[tuple[int, int, int]]],
    dict[tuple[int, int, int], int],
]:
    """Build 26-connected graph on skeleton voxel coordinates."""
    nodes = [tuple(int(v) for v in row) for row in coords_xyz]
    node_set = set(nodes)
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    deg: dict[tuple[int, int, int], int] = {}
    for n in nodes:
        nbrs = [m for m in _neighbors26(n) if m in node_set]
        adj[n] = nbrs
        deg[n] = len(nbrs)
    return nodes, adj, deg


def _branch_polylines_from_skeleton(
    coords_xyz: np.ndarray,
    *,
    min_points: int,
) -> list[np.ndarray]:
    """All chains between endpoints / junctions (splits connected sinuses at forks)."""
    if coords_xyz.shape[0] == 0:
        return []
    if coords_xyz.shape[0] <= 2:
        poly = coords_xyz.astype(np.float32, copy=False)
        return [poly] if poly.shape[0] >= int(min_points) else []

    _nodes, adj, deg = _skeleton_graph(coords_xyz)
    special = [n for n in _nodes if deg[n] != 2]
    if not special:
        poly = _centerline_longest_path(coords_xyz.astype(np.float32))
        return [poly] if poly.shape[0] >= int(min_points) else []

    seen_chains: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    polylines: list[np.ndarray] = []

    for start in special:
        for n0 in adj[start]:
            path: list[tuple[int, int, int]] = [start, n0]
            prev, cur = start, n0
            while deg[cur] == 2:
                nbrs = [x for x in adj[cur] if x != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt

            key = _chain_key(path[0], path[-1])
            if key in seen_chains:
                continue
            seen_chains.add(key)

            if len(path) >= int(min_points):
                polylines.append(np.asarray(path, dtype=np.float32))

    return polylines


def extract_branch_polylines(
    venous_binary: np.ndarray,
    *,
    min_points: int = _MIN_BRANCH_POINTS,
) -> list[np.ndarray]:
    """Skeletonize each CC and return one polyline per inter-junction chain."""
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
        for poly in _branch_polylines_from_skeleton(
            coords.astype(np.float32),
            min_points=min_points,
        ):
            polylines.append(poly.astype(np.float32, copy=False))
    return polylines


# ---------------------------------------------------------------------------
# Greedy SSSV / STRV / LTSV / RTSV assignment
# ---------------------------------------------------------------------------


def _score_branch(
    points: np.ndarray,
    vessel_name: str,
    shape: tuple[int, int, int],
) -> float:
    """Higher is better match for *vessel_name*."""
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
        align = _alignment_score(direction, _STRV_REF)
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


def assign_venous_branches(
    venous_binary: np.ndarray,
    *,
    min_points: int = _MIN_BRANCH_POINTS,
    min_assign_score: float = _MIN_ASSIGN_SCORE,
) -> dict[str, np.ndarray]:
    """Assign junction-split skeleton chains to SSSV/STRV/LTSV/RTSV (0–4 vessels).

    Connected sinuses in one foreground component are split at skeleton forks so
    e.g. SSSV meeting RTSV can yield separate centerlines. Names with no visible
    structure or score below *min_assign_score* are omitted.
    """
    shape = tuple(int(s) for s in venous_binary.shape)
    candidates = extract_branch_polylines(venous_binary, min_points=min_points)
    if not candidates:
        return {}

    assigned: dict[str, np.ndarray] = {}
    used: set[int] = set()
    for name in MATLAB_QVT_VENOUS_VESSEL_NAMES:
        best_idx = -1
        best_score = -1.0
        for idx, poly in enumerate(candidates):
            if idx in used:
                continue
            sc = _score_branch(poly, name, shape)
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
    "VenousBranch",
    "assign_venous_branches",
    "extract_branch_polylines",
    "venous_name_to_label_id",
]
