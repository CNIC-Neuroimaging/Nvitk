"""Skeleton-graph polyline and junction extraction (qvtpy-style branch chains)."""

from __future__ import annotations

from collections import deque
from typing import Literal

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.morphology.centerline import _centerline_longest_path, skeletonize_binary
from nvitk.morphology.components import label_connected

setup(globals())

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


def junction_clusters(
    nodes: list[tuple[int, int, int]],
    deg: dict[tuple[int, int, int], int],
    *,
    min_degree: int = 3,
) -> list[list[tuple[int, int, int]]]:
    """26-connected clusters of skeleton junctions (deg >= *min_degree*)."""
    md = int(min_degree)
    junction = [n for n in nodes if deg[n] >= md]
    if not junction:
        return []
    jset = set(junction)
    visited: set[tuple[int, int, int]] = set()
    clusters: list[list[tuple[int, int, int]]] = []
    for start in junction:
        if start in visited:
            continue
        cluster: list[tuple[int, int, int]] = []
        q: deque[tuple[int, int, int]] = deque([start])
        visited.add(start)
        while q:
            n = q.popleft()
            cluster.append(n)
            for m in _neighbors26(n):
                if m in jset and m not in visited:
                    visited.add(m)
                    q.append(m)
        clusters.append(cluster)
    return clusters


def collapse_junction_clusters(
    nodes: list[tuple[int, int, int]],
    deg: dict[tuple[int, int, int], int],
    *,
    min_degree: int = 3,
) -> list[tuple[int, int, int]]:
    """One representative voxel per 26-connected cluster of skeleton junctions (deg >= *min_degree*)."""
    clusters = junction_clusters(nodes, deg, min_degree=int(min_degree))
    return [max(c, key=lambda n: (deg[n], n)) for c in clusters]


def _rebuild_deg(
    node_set: set[tuple[int, int, int]],
) -> dict[tuple[int, int, int], int]:
    return {n: sum(1 for m in _neighbors26(n) if m in node_set) for n in node_set}


def _arm_length_from_junction(
    junction: tuple[int, int, int],
    first: tuple[int, int, int],
    node_set: set[tuple[int, int, int]],
) -> int:
    """Steps from *junction* along *first* until the next endpoint/junction."""
    prev, cur = junction, first
    length = 1
    while True:
        nbrs = [m for m in _neighbors26(cur) if m in node_set and m != prev]
        deg = sum(1 for m in _neighbors26(cur) if m in node_set)
        if deg != 2 or not nbrs:
            return length
        prev, cur = cur, nbrs[0]
        length += 1


def _arm_components_from_cluster(
    cluster: set[tuple[int, int, int]],
    node_set: set[tuple[int, int, int]],
) -> list[set[tuple[int, int, int]]]:
    """Connected components of *node_set \\ cluster* that touch the cluster.

    Each component is one anatomical arm leaving a multi-way junction (Y or
    torcular). Counting per-neighbor from a single cluster voxel fails when
    degree > 3 or the junction is a multi-voxel blob under 26-connectivity.
    """
    outside = node_set - cluster
    if not outside:
        return []
    exits = {
        m
        for c in cluster
        for m in _neighbors26(c)
        if m in outside
    }
    if not exits:
        return []
    visited: set[tuple[int, int, int]] = set()
    arms: list[set[tuple[int, int, int]]] = []
    for seed in sorted(exits):
        if seed in visited:
            continue
        comp: set[tuple[int, int, int]] = set()
        q: deque[tuple[int, int, int]] = deque([seed])
        visited.add(seed)
        while q:
            u = q.popleft()
            comp.add(u)
            for v in _neighbors26(u):
                if v in outside and v not in visited:
                    visited.add(v)
                    q.append(v)
        # Only keep components that actually touch the junction cluster.
        if comp & exits:
            arms.append(comp)
    return arms


def significant_bifurcation_nodes(
    coords_xyz: np.ndarray,
    *,
    min_arm_points: int = 5,
    min_degree: int = 3,
) -> list[tuple[int, int, int]]:
    """Multi-way junction reps with ≥3 long arms (Y *or* torcular degree ≥4).

    Arms are counted from the whole junction *cluster*, not from a single
    representative's 26-neighbors. Fat torcular blobs often have local degree
    >3 with few long arms visible from one voxel; cluster exits fix that.
    """
    if coords_xyz.shape[0] == 0:
        return []
    nodes, _adj, deg = skeleton_graph(coords_xyz)
    node_set = set(nodes)
    clusters = junction_clusters(nodes, deg, min_degree=int(min_degree))
    min_arm = max(1, int(min_arm_points))
    out: list[tuple[int, int, int]] = []
    for cluster_list in clusters:
        cluster = set(cluster_list)
        arms = _arm_components_from_cluster(cluster, node_set)
        long_arms = sum(1 for arm in arms if len(arm) >= min_arm)
        if long_arms >= 3:
            out.append(max(cluster_list, key=lambda n: (deg[n], n)))
    return out


def _expand_reps_to_clusters(
    nodes: list[tuple[int, int, int]],
    deg: dict[tuple[int, int, int], int],
    reps: list[tuple[int, int, int]],
    *,
    min_degree: int = 3,
) -> list[tuple[int, int, int]]:
    """All voxels in junction clusters whose representative is in *reps*."""
    if not reps:
        return []
    rep_set = set(reps)
    out: list[tuple[int, int, int]] = []
    for cluster in junction_clusters(nodes, deg, min_degree=int(min_degree)):
        rep = max(cluster, key=lambda n: (deg[n], n))
        if rep in rep_set:
            out.extend(cluster)
    return out


def prune_skeleton_coords_tiny_loops(
    coords_xyz: np.ndarray,
    *,
    max_cycle_len: int = 12,
    max_handle_points: int | None = None,
) -> np.ndarray:
    """Break small skeleton cycles by removing short off-main-path handles.

    Oversegmented blobs often skeletonize into tiny loops. Only **short**
    off-main clusters on each small cycle are deleted so long anatomical
    branches (e.g. SSSV off an L↔R transverse diameter) are not stripped when
    26-connectivity creates local cycles at a junction.
    """
    if coords_xyz.shape[0] < 4:
        return coords_xyz.astype(np.float32, copy=False)
    try:
        import networkx as nx
    except Exception:
        return coords_xyz.astype(np.float32, copy=False)

    nodes, adj, _deg = skeleton_graph(coords_xyz)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for n, nbrs in adj.items():
        for m in nbrs:
            if n < m:
                G.add_edge(n, m)
    if G.number_of_edges() == 0:
        return coords_xyz.astype(np.float32, copy=False)

    try:
        cycles = [[tuple(int(v) for v in n) for n in c] for c in nx.minimum_cycle_basis(G)]
    except Exception:
        cycles = []
    if not cycles:
        try:
            cycles = [[tuple(int(v) for v in n) for n in c] for c in nx.cycle_basis(G)]
        except Exception:
            cycles = []

    main_path = _centerline_longest_path(coords_xyz.astype(np.float32))
    main_set = {tuple(int(v) for v in row) for row in main_path}

    remove: set[tuple[int, int, int]] = set()
    max_len = max(3, int(max_cycle_len))
    handle_cap = (
        int(max_handle_points)
        if max_handle_points is not None
        else max(2, max_len // 2)
    )
    for cycle in cycles:
        if len(cycle) > max_len or len(cycle) < 3:
            continue
        off_main = [n for n in cycle if n not in main_set]
        if off_main:
            # Only delete small connected off-main handles, not long branches.
            off_set = set(off_main)
            visited: set[tuple[int, int, int]] = set()
            for seed in off_main:
                if seed in visited:
                    continue
                cluster: list[tuple[int, int, int]] = []
                q = [seed]
                visited.add(seed)
                while q:
                    u = q.pop()
                    cluster.append(u)
                    for v in G.neighbors(u):
                        if v in off_set and v not in visited:
                            visited.add(v)
                            q.append(v)
                if len(cluster) <= handle_cap:
                    remove.update(cluster)
            continue
        bridge = [n for n in cycle if int(G.degree(n)) <= 2]
        if bridge:
            remove.update(bridge)

    if not remove:
        return coords_xyz.astype(np.float32, copy=False)
    kept = [n for n in nodes if n not in remove]
    if len(kept) < 2:
        return coords_xyz.astype(np.float32, copy=False)
    return to_numpy(kept).astype(np.float32)


def prune_skeleton_coords_short_spurs(
    coords_xyz: np.ndarray,
    *,
    min_spur_points: int = 3,
) -> np.ndarray:
    """Iteratively remove degree-1 chains shorter than *min_spur_points*."""
    if coords_xyz.shape[0] < 3:
        return coords_xyz.astype(np.float32, copy=False)
    node_set = {tuple(int(v) for v in row) for row in coords_xyz}
    min_len = max(1, int(min_spur_points))
    changed = True
    while changed and len(node_set) >= 2:
        changed = False
        endpoints = [
            n
            for n in list(node_set)
            if sum(1 for m in _neighbors26(n) if m in node_set) <= 1
        ]
        for ep in endpoints:
            if ep not in node_set:
                continue
            path = [ep]
            prev = None
            cur = ep
            while True:
                nbrs = [m for m in _neighbors26(cur) if m in node_set and m != prev]
                d = sum(1 for m in _neighbors26(cur) if m in node_set)
                if d > 2:
                    break
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if sum(1 for m in _neighbors26(cur) if m in node_set) != 2:
                    break
            last = path[-1]
            last_deg = sum(1 for m in _neighbors26(last) if m in node_set)
            spur = path[:-1] if last_deg >= 3 else path
            if 0 < len(spur) < min_len:
                for n in spur:
                    node_set.discard(n)
                changed = True
    if len(node_set) < 2:
        return coords_xyz.astype(np.float32, copy=False)
    return to_numpy(sorted(node_set)).astype(np.float32)


def _chains_from_specials(
    coords_xyz: np.ndarray,
    special: list[tuple[int, int, int]],
    *,
    min_points: int,
) -> list[np.ndarray]:
    """Walk chains between endpoint/junction specials.

    Intermediate non-special branch points (e.g. leftover tiny-loop junctions
    when only significant bifurcations are marked special) are traversed by
    following the longest remaining arm so chains are not fragmented.
    """
    nodes, adj, deg = skeleton_graph(coords_xyz)
    node_set = set(nodes)
    if not special:
        poly = _centerline_longest_path(coords_xyz.astype(np.float32))
        return [poly] if poly.shape[0] >= int(min_points) else []

    special_set = set(special)
    seen_chains: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    polylines: list[np.ndarray] = []

    for start in special:
        for n0 in adj.get(start, []):
            path: list[tuple[int, int, int]] = [start, n0]
            prev, cur = start, n0
            while cur not in special_set:
                nbrs = [x for x in adj.get(cur, []) if x != prev]
                if not nbrs:
                    break
                if len(nbrs) == 1:
                    nxt = nbrs[0]
                else:
                    nxt = max(
                        nbrs,
                        key=lambda n: _arm_length_from_junction(cur, n, node_set),
                    )
                path.append(nxt)
                prev, cur = cur, nxt

            key = _chain_key(path[0], path[-1])
            if key in seen_chains:
                continue
            seen_chains.add(key)
            if len(path) >= int(min_points):
                polylines.append(to_numpy(path).astype(np.float32))

    return polylines


def branch_polylines_from_skeleton(
    coords_xyz: np.ndarray,
    *,
    min_points: int = 5,
    prefer_bifurcations: bool = False,
    prune_tiny_loops: bool = False,
    max_tiny_loop_len: int = 12,
    prune_short_spurs: bool = False,
    min_bifurcation_arm_points: int | None = None,
) -> list[np.ndarray]:
    """One polyline per chain between skeleton endpoints and junctions.

    When *prefer_bifurcations* is True, significant multi-way junctions
    (clusters with ≥3 long arms — classic Y or torcular degree ≥4) are used as
    split points. The full junction *cluster* is marked special so walks cannot
    merge through fat deg>3 blobs. Tiny loops / short spurs can be pruned first.
    """
    if coords_xyz.shape[0] == 0:
        return []
    if coords_xyz.shape[0] <= 2:
        poly = coords_xyz.astype(np.float32, copy=False)
        return [poly] if poly.shape[0] >= int(min_points) else []

    coords = coords_xyz.astype(np.float32, copy=False)
    if prune_tiny_loops:
        coords = prune_skeleton_coords_tiny_loops(
            coords, max_cycle_len=int(max_tiny_loop_len)
        )
    if prune_short_spurs:
        coords = prune_skeleton_coords_short_spurs(
            coords, min_spur_points=max(2, int(min_points) // 2)
        )
    if coords.shape[0] < int(min_points):
        return []

    nodes, _adj, deg = skeleton_graph(coords)
    endpoints = [n for n, d in deg.items() if d <= 1]
    min_arm = (
        int(min_bifurcation_arm_points)
        if min_bifurcation_arm_points is not None
        else max(3, int(min_points) // 2)
    )

    if prefer_bifurcations:
        bif = significant_bifurcation_nodes(coords, min_arm_points=min_arm)
        if bif:
            # Mark every voxel in significant clusters so walks stop at the
            # torcular shell instead of traversing deg>3 interior and merging arms.
            cluster_specials = _expand_reps_to_clusters(nodes, deg, bif, min_degree=3)
            special = list(dict.fromkeys([*endpoints, *cluster_specials]))
            return _chains_from_specials(coords, special, min_points=min_points)
        # No significant multi-way junction: still expand every junction cluster
        # so deg>3 blobs cannot merge arms into a single diameter walk.
        all_reps = collapse_junction_clusters(nodes, deg, min_degree=3)
        cluster_specials = _expand_reps_to_clusters(
            nodes, deg, all_reps, min_degree=3
        )
        special = list(dict.fromkeys([*endpoints, *cluster_specials]))
        return _chains_from_specials(coords, special, min_points=min_points)

    all_reps = collapse_junction_clusters(nodes, deg, min_degree=3)
    cluster_specials = _expand_reps_to_clusters(nodes, deg, all_reps, min_degree=3)
    special = list(dict.fromkeys([*endpoints, *cluster_specials]))
    return _chains_from_specials(coords, special, min_points=min_points)


def junction_nodes_from_skeleton(
    coords_xyz: np.ndarray,
    *,
    min_degree: int = 3,
) -> np.ndarray:
    """Return (J, 3) junction voxel coordinates (skeleton degree >= *min_degree*)."""
    if coords_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    _nodes, _adj, deg = skeleton_graph(coords_xyz)
    pts = collapse_junction_clusters(_nodes, deg, min_degree=int(min_degree))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return to_numpy(pts).astype(np.float32)


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
    "collapse_junction_clusters",
    "ExtractionMode",
    "branch_polylines_from_skeleton",
    "detect_junctions_from_centerline",
    "extract_polylines_from_centerline",
    "junction_clusters",
    "junction_nodes_from_skeleton",
    "prune_skeleton_coords_short_spurs",
    "prune_skeleton_coords_tiny_loops",
    "significant_bifurcation_nodes",
    "skeleton_graph",
]
