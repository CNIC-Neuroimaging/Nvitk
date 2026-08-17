# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/morphology/centerline.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Centerline utilities (skeletonize + polyline ordering).

These helpers are **not** GPU-native: scikit-image's 3D skeletonization is CPU
only. We accept CuPy arrays by converting to NumPy via :func:`nvitk_vendor.core.array.to_numpy`.
"""

from __future__ import annotations

from typing import Sequence

from nvitk_vendor.core.array import as_backend_array, to_numpy
from nvitk_vendor.core.backend import setup
from nvitk_vendor.core.exceptions import ValidationError
from nvitk_vendor.core.logger import Logger

setup(globals())

# ``setup(globals())`` may rebind ``np`` to CuPy; keep a host NumPy alias for
# CPU skeletonize / graph ops that require ndarray, not cupy.ndarray.
import numpy as _np

log = Logger()


def skeletonize_binary(mask) -> Any:
    """3D/2D skeletonization (CPU skimage; returns backend array of bool)."""
    skeletonize = _require_skeletonize()
    m = to_numpy(mask)
    if m.ndim not in (2, 3):
        raise ValidationError("skeletonize_binary expects a 2D or 3D mask.")
    sk = skeletonize(m.astype(np.uint8, copy=False))
    return as_backend_array(sk).astype(bool)


def skeletonize_labeled(
    mask,
    *,
    labels: Sequence[int] | None = None,
    min_points: int = 1,
) -> Any:
    """Skeletonize each label region independently; output keeps original label ids.

    For a binary mask (labels ``[1]``), skeleton voxels are written as ``1``.
    For multilabel masks, each label's skeleton voxels carry that label's id.
    """
    arr = as_backend_array(mask)
    if arr.ndim not in (2, 3):
        raise ValidationError("skeletonize_labeled expects a 2D or 3D mask.")
    arr_np = to_numpy(arr)
    labs = labels or sorted(int(v) for v in np.unique(arr_np) if int(v) != 0)
    if not labs:
        raise ValidationError("skeletonize_labeled: mask has no non-zero labels.")

    if np.issubdtype(arr_np.dtype, np.integer):
        out_dtype = arr_np.dtype
    else:
        max_lab = max(labs)
        out_dtype = np.uint8 if max_lab <= 255 else (np.uint16 if max_lab <= 65535 else np.int32)

    out = np.zeros(arr_np.shape, dtype=out_dtype)
    for lbl in labs:
        roi = arr_np == int(lbl)
        if not np.any(roi):
            continue
        sk = to_numpy(skeletonize_binary(roi)) > 0
        if int(sk.sum()) < int(min_points):
            continue
        out[sk] = int(lbl)
    if not bool(np.any(out)):
        raise ValidationError("skeletonize_labeled: no skeleton voxels produced.")
    return as_backend_array(out)


def _require_skeletonize():
    """Import and return ``skimage.morphology.skeletonize`` or raise a helpful ImportError."""
    try:
        from skimage.morphology import skeletonize
    except Exception as exc:
        import traceback
        log.exception(traceback.format_exc())
        raise ImportError(
            "Centerline computation requires scikit-image (skeletonize). "
            "Install it with: pip install scikit-image"
        ) from exc
    return skeletonize


def _neighbors26(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """The 26 face/edge/corner neighbours of voxel *p* (self excluded)."""
    x, y, z = p
    out: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                out.append((x + dx, y + dy, z + dz))
    return out


def _skeleton_graph(
    coords_xyz: np.ndarray,
) -> tuple[
    list[tuple[int, int, int]],
    dict[tuple[int, int, int], list[tuple[int, int, int]]],
    dict[tuple[int, int, int], int],
]:
    """Build the 26-connected skeleton graph: ``(nodes, adjacency, degree)`` from voxel coords."""
    nodes = [tuple(int(v) for v in row) for row in coords_xyz]
    node_set = set(nodes)
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    deg: dict[tuple[int, int, int], int] = {}
    for n in nodes:
        nbrs = [m for m in _neighbors26(n) if m in node_set]
        adj[n] = nbrs
        deg[n] = len(nbrs)
    return nodes, adj, deg


def _largest_skeleton_component(coords_xyz: np.ndarray) -> np.ndarray:
    """Keep the largest 26-connected skeleton component (drops speck CCs)."""
    if coords_xyz.shape[0] <= 2:
        return coords_xyz.astype(np.float32, copy=False)

    from collections import deque

    nodes, adj, _deg = _skeleton_graph(coords_xyz)
    visited: set[tuple[int, int, int]] = set()
    best: list[tuple[int, int, int]] = []
    for start in nodes:
        if start in visited:
            continue
        comp: list[tuple[int, int, int]] = []
        q: deque[tuple[int, int, int]] = deque([start])
        visited.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        if len(comp) > len(best):
            best = comp
    if not best:
        return coords_xyz.astype(np.float32, copy=False)
    return to_numpy(best).astype(np.float32)


def _bfs_from(
    src: tuple[int, int, int],
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> tuple[
    dict[tuple[int, int, int], int],
    dict[tuple[int, int, int], tuple[int, int, int] | None],
]:
    """Breadth-first search from *src*; returns ``(distances, parents)`` over the graph."""
    from collections import deque

    dist: dict[tuple[int, int, int], int] = {src: 0}
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {src: None}
    q: deque[tuple[int, int, int]] = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v in dist:
                continue
            dist[v] = dist[u] + 1
            parent[v] = u
            q.append(v)
    return dist, parent


def _reconstruct_path(
    end: tuple[int, int, int],
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None],
) -> list[tuple[int, int, int]]:
    """Walk *parent* pointers back from *end* to the BFS source, returned source→end."""
    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def _nearest_node(
    nodes: list[tuple[int, int, int]],
    point_xyz: np.ndarray,
) -> tuple[int, int, int]:
    """Graph node closest (squared Euclidean) to *point_xyz*."""
    p = to_numpy(point_xyz).astype(np.float64).ravel()[:3]
    best = nodes[0]
    best_d = float("inf")
    for n in nodes:
        d = (n[0] - p[0]) ** 2 + (n[1] - p[1]) ** 2 + (n[2] - p[2]) ** 2
        if d < best_d:
            best_d = float(d)
            best = n
    return best


def _path_from_node_to_farthest(
    start: tuple[int, int, int],
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]],
) -> list[tuple[int, int, int]]:
    """Longest shortest-path from *start*: path to the graph node farthest from it."""
    dist, parent = _bfs_from(start, adj)
    if not dist:
        return [start]
    far = max(dist, key=dist.get)
    return _reconstruct_path(far, parent)


def _classical_diameter_path(
    nodes: list[tuple[int, int, int]],
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]],
    endpoints: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Graph diameter via double-BFS: farthest node ``a``, then its farthest node ``b``, path a→b."""
    start = endpoints[0] if endpoints else nodes[0]
    d1, _ = _bfs_from(start, adj)
    a = max(d1, key=d1.get)
    d2, parent = _bfs_from(a, adj)
    b = max(d2, key=d2.get)
    return _reconstruct_path(b, parent)


def _prefer_coverage_cost(
    path: list[tuple[int, int, int]],
    prefer_points: np.ndarray,
) -> float:
    """Mean squared distance from each prefer point to the nearest path voxel."""
    if not path or prefer_points.size == 0:
        return float("inf")
    path_arr = to_numpy(path).astype(np.float64)
    prefs = to_numpy(prefer_points).astype(np.float64).reshape(-1, 3)
    total = 0.0
    for p in prefs:
        d2 = np.sum((path_arr - p.reshape(1, 3)) ** 2, axis=1)
        total += float(np.min(d2))
    return total / float(prefs.shape[0])


def _centerline_longest_path(
    coords_xyz: np.ndarray,
    *,
    prefer_points: np.ndarray | None = None,
) -> np.ndarray:
    """Order skeleton voxels into a main centerline polyline.

    Uses the largest 26-connected skeleton component, then the graph diameter.
    When *prefer_points* are given (e.g. a prior stage-3 polyline), prefers a
    root→farthest path that stays close to those points so proximal trunks on
    branched vessels (MCA M1) are not dropped by a distal-only diameter.
    """
    if coords_xyz.shape[0] <= 2:
        return coords_xyz.astype(np.float32, copy=False)

    coords = _largest_skeleton_component(coords_xyz)
    if coords.shape[0] <= 2:
        return coords.astype(np.float32, copy=False)

    nodes, adj, deg = _skeleton_graph(coords)
    endpoints = [n for n, d in deg.items() if d <= 1]
    if not endpoints:
        endpoints = [nodes[0]]

    diameter = _classical_diameter_path(nodes, adj, endpoints)
    prefs = None
    if prefer_points is not None:
        prefs = to_numpy(prefer_points).astype(np.float64).reshape(-1, 3)
        if prefs.shape[0] == 0:
            prefs = None

    if prefs is None:
        return to_numpy(diameter).astype(np.float32)

    # Candidate roots: prior polyline ends/mid + prior-nearest skeleton endpoints.
    seed_xyzs = [prefs[0], prefs[-1], prefs[prefs.shape[0] // 2]]
    candidates: list[list[tuple[int, int, int]]] = [diameter]
    for seed in seed_xyzs:
        root = _nearest_node(nodes, seed)
        # Prefer starting at an endpoint near the seed when available.
        near_ep = min(
            endpoints,
            key=lambda e: (e[0] - root[0]) ** 2
            + (e[1] - root[1]) ** 2
            + (e[2] - root[2]) ** 2,
        )
        for start in dict.fromkeys((near_ep, root)):
            path = _path_from_node_to_farthest(start, adj)
            if len(path) >= 2:
                candidates.append(path)

    best_path = diameter
    best_key: tuple[float, int] | None = None
    for path in candidates:
        cost = _prefer_coverage_cost(path, prefs)
        key = (cost, -len(path))
        if best_key is None or key < best_key:
            best_key = key
            best_path = path
    return to_numpy(best_path).astype(np.float32)


def _skeleton_connected_components(coords_xyz: np.ndarray) -> list[np.ndarray]:
    """Split skeleton voxels into 26-connected components (largest first)."""
    if coords_xyz.shape[0] == 0:
        return []
    if coords_xyz.shape[0] <= 2:
        return [coords_xyz.astype(np.float32, copy=False)]

    from collections import deque

    nodes, adj, _deg = _skeleton_graph(coords_xyz)
    visited: set[tuple[int, int, int]] = set()
    comps: list[list[tuple[int, int, int]]] = []
    for start in nodes:
        if start in visited:
            continue
        comp: list[tuple[int, int, int]] = []
        q: deque[tuple[int, int, int]] = deque([start])
        visited.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        if comp:
            comps.append(comp)
    comps.sort(key=len, reverse=True)
    return [to_numpy(c).astype(np.float32) for c in comps]


def unique_skeleton_edge_polylines(
    coords_xyz: np.ndarray,
    *,
    min_points: int = 2,
    prune_short_spurs: bool = False,
    prune_spur_points: int = 3,
) -> list[np.ndarray]:
    """Unique centerline edges between endpoints and junctions (no interior overlap).

    Each returned polyline is a chain between skeleton specials (deg≤1 endpoints
    and deg≥3 junction clusters). Chains share only their junction endpoints so
    the union is a connected tree/graph without duplicated trunks.
    """
    # Lazy import: polyline_graph imports helpers from this module.
    from nvitk_vendor.morphology.polyline_graph import (
        _chains_from_specials,
        _expand_reps_to_clusters,
        collapse_junction_clusters,
        junction_clusters,
        prune_skeleton_coords_short_spurs,
        skeleton_graph,
    )

    if coords_xyz.shape[0] == 0:
        return []
    coords = coords_xyz.astype(np.float32, copy=False)
    min_pts = max(2, int(min_points))
    if prune_short_spurs:
        coords = prune_skeleton_coords_short_spurs(
            coords, min_spur_points=max(2, int(prune_spur_points))
        )
    if coords.shape[0] < min_pts:
        return []
    if coords.shape[0] <= 2:
        return [coords]

    nodes, _adj, deg = skeleton_graph(coords)
    endpoints = [n for n, d in deg.items() if int(d) <= 1]
    junction_reps = collapse_junction_clusters(nodes, deg, min_degree=3)
    if not junction_reps and len(endpoints) <= 2:
        poly = _centerline_longest_path(coords)
        return [poly] if poly.shape[0] >= min_pts else []

    specials = _expand_reps_to_clusters(nodes, deg, junction_reps, min_degree=3)
    special = list(dict.fromkeys([*endpoints, *specials]))
    polylines = _chains_from_specials(coords, special, min_points=min_pts)

    # Drop tiny edges that only connect voxels inside the same junction blob.
    clusters = junction_clusters(nodes, deg, min_degree=3)
    node_to_cluster: dict[tuple[int, int, int], int] = {}
    for ci, cluster in enumerate(clusters):
        for n in cluster:
            node_to_cluster[n] = ci
    kept: list[np.ndarray] = []
    for poly in polylines:
        if poly.shape[0] < min_pts:
            continue
        a = tuple(int(v) for v in poly[0])
        b = tuple(int(v) for v in poly[-1])
        ca = node_to_cluster.get(a)
        cb = node_to_cluster.get(b)
        if ca is not None and cb is not None and ca == cb and poly.shape[0] <= 3:
            continue
        kept.append(poly.astype(np.float32, copy=False))
    return kept


def _branch_paths_from_skeleton(
    coords_xyz: np.ndarray,
    *,
    prefer_points: np.ndarray | None = None,
    min_branch_points: int = 3,
    keep_all_components: bool = True,
) -> list[np.ndarray]:
    """Decompose a skeleton into a trunk path plus side branches.

    The trunk is the main centerline (graph diameter, or the prior-biased path
    when *prefer_points* are given). Every remaining deg-1 endpoint is then
    traced back toward the already-accepted paths; the segment up to the first
    node that is already covered becomes a side branch. Branches shorter than
    *min_branch_points* are dropped. Returns ordered ``(N, 3)`` polylines with the
    trunk first (proximal→distal orientation preserved from the trunk).

    When *keep_all_components* is True (default), every skeleton CC contributes
    its own trunk + side branches (largest CC first). This preserves disconnected
    proximal roots that largest-CC-only extraction would drop.
    """
    if coords_xyz.shape[0] <= 2:
        return [coords_xyz.astype(np.float32, copy=False)]

    comps = (
        _skeleton_connected_components(coords_xyz)
        if keep_all_components
        else [_largest_skeleton_component(coords_xyz)]
    )
    paths: list[np.ndarray] = []
    for coords in comps:
        if coords.shape[0] <= 2:
            if coords.shape[0] >= 2:
                paths.append(coords.astype(np.float32, copy=False))
            continue

        nodes, adj, deg = _skeleton_graph(coords)
        trunk = _centerline_longest_path(coords, prefer_points=prefer_points)
        trunk_nodes = [tuple(int(v) for v in row) for row in to_numpy(trunk).astype(np.int64)]

        covered: set[tuple[int, int, int]] = set(trunk_nodes)
        paths.append(to_numpy(trunk).astype(np.float32))

        endpoints = sorted(n for n, d in deg.items() if d <= 1)
        for ep in endpoints:
            if ep in covered:
                continue
            seg: list[tuple[int, int, int]] = []
            prev: tuple[int, int, int] | None = None
            cur: tuple[int, int, int] | None = ep
            guard = 0
            max_steps = len(nodes) + 1
            while cur is not None and cur not in covered and guard < max_steps:
                seg.append(cur)
                guard += 1
                nxt = None
                for nb in adj.get(cur, []):
                    if nb == prev:
                        continue
                    if nb in seg:
                        continue
                    nxt = nb
                    break
                prev, cur = cur, nxt
            if cur is not None and cur in covered:
                seg.append(cur)
            if len(seg) < int(min_branch_points):
                continue
            if seg and seg[-1] in covered:
                seg = list(reversed(seg))
            paths.append(to_numpy(seg).astype(np.float32))
            for n in seg:
                covered.add(n)

    return paths if paths else [coords_xyz.astype(np.float32, copy=False)]


def compute_connected_centerline_tree(
    vessel_mask: np.ndarray,
    *,
    centerline_mask: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
    min_points: int = 2,
    min_edge_points: int = 2,
    keep_all_components: bool = True,
    bridge_max_gap: int = 0,
    bridge_tube_radius: int = 1,
    prune_short_spurs: bool = True,
    prune_spur_points: int = 3,
) -> dict[int, list[np.ndarray]]:
    """Per-label connected centerline tree as unique skeleton-edge polylines.

    This is the **default branched centerline representation** for nvitk:
    skeletonize each label, optionally MST-bridge nearby mask CCs, then extract
    every unique junction↔endpoint / junction↔junction chain. Edges share only
    junction voxels (no overlapping trunks). All meaningful skeleton CCs are
    kept by default so proximal roots are not dropped.

    Returns
    -------
    dict[int, list[np.ndarray]]
        Label → list of ordered ``(N, 3)`` polylines in voxel coordinates.
    """
    arr = as_backend_array(vessel_mask)
    if arr.ndim != 3:
        raise ValidationError("vessel_mask must be 3D for centerline computation.")
    arr_np = _np.asarray(to_numpy(arr))
    labs = labels or sorted(int(v) for v in _np.unique(arr_np) if int(v) != 0)
    out: dict[int, list[_np.ndarray]] = {}
    min_edge = max(2, int(min_edge_points))
    bridge_gap = max(0, int(bridge_max_gap))

    def _edges_for_coords(coords: _np.ndarray) -> list[_np.ndarray]:
        """Per-label worker: skeleton coords → list of polyline edge point-arrays."""
        if coords.shape[0] < int(min_points):
            return []
        comps = (
            _skeleton_connected_components(coords)
            if keep_all_components
            else [_largest_skeleton_component(coords)]
        )
        edges: list[_np.ndarray] = []
        for comp in comps:
            edges.extend(
                unique_skeleton_edge_polylines(
                    comp,
                    min_points=min_edge,
                    prune_short_spurs=bool(prune_short_spurs),
                    prune_spur_points=int(prune_spur_points),
                )
            )
        # Longest first so callers that treat paths[0] as "trunk" stay reasonable.
        edges.sort(key=lambda e: int(e.shape[0]), reverse=True)
        return edges

    if centerline_mask is not None:
        cl = as_backend_array(centerline_mask)
        if cl.shape != arr.shape:
            raise ValidationError("centerline_mask must match vessel_mask shape.")
        cl_bin = _np.asarray(to_numpy(cl)) > 0
        for lbl in labs:
            coords = _np.argwhere(cl_bin & (arr_np == int(lbl))).astype(_np.float32)
            edges = _edges_for_coords(coords)
            if edges:
                out[int(lbl)] = edges
        return out

    skeletonize = _require_skeletonize()
    for lbl in labs:
        roi = arr_np == int(lbl)
        if not bool(roi.any()):
            continue
        working = roi
        if bridge_gap > 0:
            from nvitk_vendor.morphology.mst_bridge import bridge_binary_components_mst

            working = bridge_binary_components_mst(
                working,
                max_gap=bridge_gap,
                tube_radius=max(0, int(bridge_tube_radius)),
            )
        sk = skeletonize(_np.asarray(working, dtype=_np.uint8))
        coords = _np.argwhere(_np.asarray(sk) > 0).astype(_np.float32)
        edges = _edges_for_coords(coords)
        if edges:
            out[int(lbl)] = edges
    return out


def compute_centerline_branches(
    vessel_mask: np.ndarray,
    *,
    centerline_mask: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
    min_points: int = 20,
    min_branch_points: int = 3,
    prefer_points_by_label: dict[int, np.ndarray] | None = None,
    keep_all_components: bool = True,
    bridge_max_gap: int = 0,
    bridge_tube_radius: int = 1,
    connected_edges: bool = False,
) -> dict[int, list[np.ndarray]]:
    """Per-label list of branch polylines.

    Default (*connected_edges=False*): trunk + bifurcation side branches per
    skeleton CC (legacy qvtpy / vertebral-split semantics).

    With *connected_edges=True*: unique junction-split edges from
    :func:`compute_connected_centerline_tree` (preferred for morphometrics and
    any export that must stay connected at bifurcations without overlapping
    trunks).

    All skeleton CCs are kept by default (*keep_all_components*).
    """
    if connected_edges:
        return compute_connected_centerline_tree(
            vessel_mask,
            centerline_mask=centerline_mask,
            labels=labels,
            min_points=max(2, int(min_points)),
            min_edge_points=max(2, int(min_branch_points)),
            keep_all_components=bool(keep_all_components),
            bridge_max_gap=int(bridge_max_gap),
            bridge_tube_radius=int(bridge_tube_radius),
        )

    arr = as_backend_array(vessel_mask)
    if arr.ndim != 3:
        raise ValidationError("vessel_mask must be 3D for centerline computation.")
    arr_np = _np.asarray(to_numpy(arr))
    labs = labels or sorted(int(v) for v in _np.unique(arr_np) if int(v) != 0)
    out: dict[int, list[_np.ndarray]] = {}
    prefs = prefer_points_by_label or {}
    bridge_gap = max(0, int(bridge_max_gap))

    def _branches_for(coords: _np.ndarray, lbl: int) -> None:
        """Per-label worker: accumulate branch polylines for label *lbl* into ``out``."""
        if coords.shape[0] < int(min_points):
            return
        out[int(lbl)] = _branch_paths_from_skeleton(
            coords,
            prefer_points=prefs.get(int(lbl)),
            min_branch_points=int(min_branch_points),
            keep_all_components=bool(keep_all_components),
        )

    if centerline_mask is not None:
        cl = as_backend_array(centerline_mask)
        if cl.shape != arr.shape:
            raise ValidationError("centerline_mask must match vessel_mask shape.")
        cl_bin = _np.asarray(to_numpy(cl)) > 0
        for lbl in labs:
            coords = _np.argwhere(cl_bin & (arr_np == int(lbl)))
            _branches_for(coords, int(lbl))
        return out

    skeletonize = _require_skeletonize()
    for lbl in labs:
        roi = arr_np == int(lbl)
        if not bool(roi.any()):
            continue
        working = roi
        if bridge_gap > 0:
            from nvitk_vendor.morphology.mst_bridge import bridge_binary_components_mst

            working = bridge_binary_components_mst(
                working,
                max_gap=bridge_gap,
                tube_radius=max(0, int(bridge_tube_radius)),
            )
        sk = skeletonize(_np.asarray(working, dtype=_np.uint8))
        coords = _np.argwhere(_np.asarray(sk) > 0)
        _branches_for(coords, int(lbl))
    return out


def compute_centerlines(
    vessel_mask: np.ndarray,
    *,
    centerline_mask: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
    min_points: int = 20,
    prefer_points_by_label: dict[int, np.ndarray] | None = None,
    keep_all_components: bool = False,
    bridge_max_gap: int = 0,
    bridge_tube_radius: int = 1,
) -> dict[int, np.ndarray]:
    """Return per-label ordered centerline points in voxel coordinates.

    For a single main path per label (graph diameter on the largest skeleton CC
    by default). For branched vessels that must stay connected at bifurcations
    without overlapping trunks, use :func:`compute_connected_centerline_tree`.

    Parameters
    ----------
    vessel_mask
        3D integer label mask (NumPy or CuPy).
    centerline_mask
        Optional 3D mask where centerline voxels are >0. If provided, this mask is
        intersected with each `vessel_mask == label` region and ordered.
    labels
        Optional explicit label list. Default: all non-zero labels in `vessel_mask`.
    min_points
        Skip labels whose centerline has fewer than this many voxels.
    prefer_points_by_label
        Optional prior polylines (e.g. stage-3) used to keep proximal trunks on
        branched skeletons instead of a distal-only graph diameter.
    keep_all_components
        If True, concatenate diameter paths from every skeleton CC (first = largest).
        Default False preserves historical single-path behaviour.
    bridge_max_gap
        If >0, MST-bridge nearby mask CCs of each label before skeletonize.
    """
    arr = as_backend_array(vessel_mask)
    if arr.ndim != 3:
        raise ValidationError("vessel_mask must be 3D for centerline computation.")
    arr_np = _np.asarray(to_numpy(arr))
    labs = labels or sorted(int(v) for v in _np.unique(arr_np) if int(v) != 0)
    out: dict[int, _np.ndarray] = {}
    prefs = prefer_points_by_label or {}
    bridge_gap = max(0, int(bridge_max_gap))

    def _path_for(coords: _np.ndarray, lbl: int) -> _np.ndarray | None:
        """Per-label worker: single main centerline path for label *lbl* (or ``None``)."""
        if coords.shape[0] < int(min_points):
            return None
        if keep_all_components:
            comps = _skeleton_connected_components(coords)
            paths = [
                _centerline_longest_path(c, prefer_points=prefs.get(int(lbl)))
                for c in comps
                if c.shape[0] >= 2
            ]
            if not paths:
                return None
            # Primary API stays one array: longest component path (largest CC first).
            return paths[0]
        return _centerline_longest_path(coords, prefer_points=prefs.get(int(lbl)))

    if centerline_mask is not None:
        cl = as_backend_array(centerline_mask)
        if cl.shape != arr.shape:
            raise ValidationError("centerline_mask must match vessel_mask shape.")
        cl_bin = _np.asarray(to_numpy(cl)) > 0
        for lbl in labs:
            coords = _np.argwhere(cl_bin & (arr_np == int(lbl)))
            path = _path_for(coords, int(lbl))
            if path is not None:
                out[int(lbl)] = path
        return out

    skeletonize = _require_skeletonize()
    for lbl in labs:
        roi = arr_np == int(lbl)
        if not bool(roi.any()):
            continue
        working = roi
        if bridge_gap > 0:
            from nvitk_vendor.morphology.mst_bridge import bridge_binary_components_mst

            working = bridge_binary_components_mst(
                working,
                max_gap=bridge_gap,
                tube_radius=max(0, int(bridge_tube_radius)),
            )
        sk = skeletonize(_np.asarray(working, dtype=_np.uint8))
        coords = _np.argwhere(_np.asarray(sk) > 0)
        path = _path_for(coords, int(lbl))
        if path is not None:
            out[int(lbl)] = path
    return out


def centerline_tangents(points_xyz: Any, *, k_half: int = 2) -> Any:
    """Unit tangent vectors along an ordered polyline (voxel coordinates).

    Uses central differences with ``k_half`` samples on each side; endpoints
    use one-sided differences. ``points_xyz`` shape ``(N, 3)``.
    """
    p = as_backend_array(points_xyz).astype(np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValidationError("points_xyz must be (N, 3).")
    n = p.shape[0]
    if n < 2:
        return np.zeros_like(p)

    kh = max(1, int(k_half))
    tang = np.zeros_like(p)
    for i in range(n):
        i0 = max(0, i - kh)
        i1 = min(n - 1, i + kh)
        d = p[i1] - p[i0]
        norm = np.linalg.norm(d)
        tang[i] = d / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return tang.astype(np.float32, copy=False)


__all__ = [
    "centerline_tangents",
    "compute_centerline_branches",
    "compute_centerlines",
    "compute_connected_centerline_tree",
    "skeletonize_binary",
    "skeletonize_labeled",
    "unique_skeleton_edge_polylines",
]

