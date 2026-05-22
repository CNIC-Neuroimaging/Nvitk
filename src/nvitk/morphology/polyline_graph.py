"""Skeleton-graph polyline and junction extraction (qvtpy-style branch chains)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.morphology.centerline import _centerline_longest_path, skeletonize_binary
from nvitk.morphology.components import label_connected

ExtractionMode = Literal["junction_split", "longest_path"]


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


def _chain_key(
    a: tuple[int, int, int], b: tuple[int, int, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (a, b) if a <= b else (b, a)


def skeleton_graph(
    coords_xyz: np.ndarray,
) -> tuple[
    list[tuple[int, int, int]],
    dict[tuple[int, int, int], list[tuple[int, int, int]]],
    dict[tuple[int, int, int], int],
]:
    """Build a 26-connected graph on skeleton voxel coordinates."""
    nodes = [tuple(int(v) for v in row) for row in coords_xyz]
    node_set = set(nodes)
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    deg: dict[tuple[int, int, int], int] = {}
    for n in nodes:
        nbrs = [m for m in _neighbors26(n) if m in node_set]
        adj[n] = nbrs
        deg[n] = len(nbrs)
    return nodes, adj, deg


def branch_polylines_from_skeleton(
    coords_xyz: np.ndarray,
    *,
    min_points: int = 5,
) -> list[np.ndarray]:
    """One polyline per chain between skeleton endpoints and junctions (degree ≠ 2)."""
    if coords_xyz.shape[0] == 0:
        return []
    if coords_xyz.shape[0] <= 2:
        poly = coords_xyz.astype(np.float32, copy=False)
        return [poly] if poly.shape[0] >= int(min_points) else []

    _nodes, adj, deg = skeleton_graph(coords_xyz)
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


def junction_nodes_from_skeleton(
    coords_xyz: np.ndarray,
    *,
    min_degree: int = 3,
) -> np.ndarray:
    """Return (J, 3) junction voxel coordinates (skeleton degree >= *min_degree*)."""
    if coords_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    _nodes, _adj, deg = skeleton_graph(coords_xyz)
    md = int(min_degree)
    pts = [n for n in _nodes if deg[n] >= md]
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def _coords_from_centerline_volume(
    centerline: np.ndarray,
    *,
    label_id: int | None = None,
    reskeletonize: bool = False,
) -> np.ndarray:
    """Voxel coordinates (N, 3) from a 3D centerline mask."""
    arr = to_numpy(centerline)
    if arr.ndim != 3:
        raise ValueError("Centerline volume must be 3D.")
    if label_id is not None and int(label_id) > 0:
        mask = arr == int(label_id)
    else:
        mask = arr > 0
    if not bool(mask.any()):
        return np.zeros((0, 3), dtype=np.float32)
    if reskeletonize:
        sk = to_numpy(skeletonize_binary(mask))
        return np.argwhere(sk > 0).astype(np.float32)
    return np.argwhere(mask > 0).astype(np.float32)


def extract_polylines_from_centerline(
    centerline: np.ndarray,
    *,
    mode: ExtractionMode = "junction_split",
    min_points: int = 5,
    label_id: int | None = None,
    reskeletonize: bool = False,
    per_connected_component: bool = True,
) -> list[np.ndarray]:
    """
    Extract ordered polylines from a 3D centerline mask (not a thick vessel mask).

    Parameters
    ----------
    centerline
        3D array: binary centerline (>0) or integer label volume.
    mode
        ``junction_split`` (qvtpy-style branch chains) or ``longest_path`` (single diameter path).
    label_id
        When set, only voxels equal to this label are used.
    reskeletonize
        If True, skeletonize before graph extraction (use for thick masks only).
    per_connected_component
        When True, run extraction separately on each 6-connected foreground component.
    """
    arr = to_numpy(centerline)
    if label_id is not None and int(label_id) > 0:
        fg = arr == int(label_id)
    else:
        fg = arr > 0

    if not bool(fg.any()):
        return []

    mode = "junction_split" if str(mode) != "longest_path" else "longest_path"
    min_pts = max(2, int(min_points))

    def _extract_on_mask(mask: np.ndarray) -> list[np.ndarray]:
        coords = _coords_from_centerline_volume(
            mask.astype(np.uint8), reskeletonize=reskeletonize
        )
        if coords.shape[0] < min_pts:
            return []
        if mode == "longest_path":
            poly = _centerline_longest_path(coords)
            return [poly] if poly.shape[0] >= min_pts else []
        return branch_polylines_from_skeleton(coords, min_points=min_pts)

    if not per_connected_component:
        return _extract_on_mask(fg)

    labeled, _ = label_connected(fg.astype(np.uint8), connectivity=1)
    lab = to_numpy(labeled)
    out: list[np.ndarray] = []
    for comp_id in range(1, int(lab.max()) + 1):
        out.extend(_extract_on_mask(lab == comp_id))
    return out


def detect_junctions_from_centerline(
    centerline: np.ndarray,
    *,
    min_degree: int = 3,
    label_id: int | None = None,
    reskeletonize: bool = False,
) -> np.ndarray:
    """Junction voxels (degree >= *min_degree*) on the centerline skeleton graph."""
    coords = _coords_from_centerline_volume(
        centerline, label_id=label_id, reskeletonize=reskeletonize
    )
    return junction_nodes_from_skeleton(coords, min_degree=min_degree)


__all__ = [
    "ExtractionMode",
    "branch_polylines_from_skeleton",
    "detect_junctions_from_centerline",
    "extract_polylines_from_centerline",
    "junction_nodes_from_skeleton",
    "skeleton_graph",
]
