"""Skeletonization, skeleton graph construction, pruning, paths, and topology."""

from __future__ import annotations

from collections import deque
from typing import List

import numpy as np
from skimage.morphology import skeletonize

from nvitk.measure.morphometrics_config import PRUNE_SPUR_LENGTH_MM, PRUNE_TERMINAL_SPURS
from .models import SkeletonTree

_NEI26 = [
    (i, j, k)
    for i in (-1, 0, 1)
    for j in (-1, 0, 1)
    for k in (-1, 0, 1)
    if not (i == j == k == 0)
]

def skeleton_graph(skel: np.ndarray):
    pts = np.argwhere(skel > 0)
    idx_map = {tuple(p): i for i, p in enumerate(map(tuple, pts))}
    n = len(pts)
    neighbors = [[] for _ in range(n)]
    for i, (x, y, z) in enumerate(map(tuple, pts)):
        for dx, dy, dz in _NEI26:
            j = idx_map.get((x + dx, y + dy, z + dz))
            if j is not None:
                neighbors[i].append(j)
    degree = np.array([len(nbrs) for nbrs in neighbors], dtype=int)
    return pts, neighbors, degree


def skeletonize_mask(mask_bool: np.ndarray) -> np.ndarray:
    return skeletonize(mask_bool.astype(bool)).astype(np.uint8)


def skeleton_tree_from_mask(mask_bool: np.ndarray, spacing=None) -> SkeletonTree:
    skel = skeletonize_mask(mask_bool)
    if PRUNE_TERMINAL_SPURS:
        skel = prune_short_terminal_spurs(skel, PRUNE_SPUR_LENGTH_MM, spacing=spacing)
    pts, neighbors, degree = skeleton_graph(skel)
    endpoints = list(np.where(degree == 1)[0])
    branchpoints = list(np.where(degree >= 3)[0])
    return SkeletonTree(pts, neighbors, degree, endpoints, branchpoints, None, None)


def prune_short_terminal_spurs(skel: np.ndarray, max_len_mm: float, spacing=None) -> np.ndarray:
    sk = skel.copy().astype(bool)
    if spacing is None:
        spacing = np.ones(3, dtype=float)
    spacing = np.asarray(spacing, dtype=float)

    changed = True
    while changed:
        changed = False
        pts, neighbors, degree = skeleton_graph(sk)
        if len(pts) < 3:
            break
        for ep in list(np.where(degree == 1)[0]):
            prev, cur = -1, ep
            chain = [ep]
            spur_len = 0.0
            too_long = False
            while True:
                nexts = [n for n in neighbors[cur] if n != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                spur_len += float(np.linalg.norm(
                    (pts[nxt].astype(float) - pts[cur].astype(float)) * spacing
                ))
                if spur_len >= max_len_mm:
                    too_long = True
                    break
                chain.append(nxt)
                if degree[nxt] != 2:
                    break
                prev, cur = cur, nxt
            if too_long:
                continue
            # Only remove spurs that terminate at a branchpoint (degree >= 3),
            # not short vessels that simply connect two endpoints.
            if degree[chain[-1]] >= 3:
                for idx in chain[:-1]:
                    sk[tuple(map(int, pts[idx]))] = False
                changed = True

    return sk.astype(np.uint8)


def skeleton_total_graph_length_mm(tree: SkeletonTree, spacing) -> float:
    spacing = np.asarray(spacing, dtype=float)
    total = 0.0
    for i, nbrs in enumerate(tree.neighbors):
        pi = tree.pts_vox[i].astype(float) * spacing
        for j in nbrs:
            if j <= i:
                continue
            pj = tree.pts_vox[j].astype(float) * spacing
            total += float(np.linalg.norm(pj - pi))
    return total


def bfs_path_indices(neighbors: List[List[int]], src: int, dst: int) -> List[int]:
    n = len(neighbors)
    parent = -np.ones(n, dtype=int)
    seen = np.zeros(n, dtype=bool)
    q = deque([src])
    seen[src] = True
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in neighbors[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)
    if not seen[dst]:
        return []
    path = []
    cur = dst
    while cur != -1:
        path.append(int(cur))
        if cur == src:
            break
        cur = int(parent[cur])
    return path[::-1]


def find_loop_branches(pts: np.ndarray, neighbors: List[List[int]], degree: np.ndarray):
    """
    Detect simple donut/cycle topologies in a skeleton graph.

    Returns:
      cycle_core: nodes participating in at least one cycle.
      gateways: branchpoints where the loop connects to the rest of the tree.
      loops: list of (gateway_a, gateway_b, arm1_node_indices, arm2_node_indices).
    """
    n = len(pts)
    if n < 3:
        return set(), set(), []

    n_edges = sum(len(nbrs) for nbrs in neighbors) // 2
    if n_edges <= n - 1:
        return set(), set(), []

    active = np.ones(n, dtype=bool)
    eff_deg = degree.copy().astype(int)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if active[i] and eff_deg[i] <= 1:
                active[i] = False
                for j in neighbors[i]:
                    if active[j]:
                        eff_deg[j] -= 1
                changed = True

    cycle_core = set(int(i) for i in np.where(active)[0])
    if not cycle_core:
        return set(), set(), []

    gateways = set(
        i for i in cycle_core
        if degree[i] >= 3 and any(j not in cycle_core for j in neighbors[i])
    )

    def bfs_subgraph(src: int, dst: int, allowed) -> List[int]:
        allowed = set(int(x) for x in allowed)
        if src not in allowed or dst not in allowed:
            return []
        parent = -np.ones(n, dtype=int)
        seen = np.zeros(n, dtype=bool)
        q = deque([src])
        seen[src] = True
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in neighbors[u]:
                if not seen[v] and v in allowed:
                    seen[v] = True
                    parent[v] = u
                    q.append(v)
        if not seen[dst]:
            return []
        path = []
        cur = int(dst)
        while cur != -1:
            path.append(cur)
            if cur == src:
                break
            cur = int(parent[cur])
        return path[::-1]

    def connected_components_nodes(nodes) -> List[List[int]]:
        remaining = set(int(x) for x in nodes)
        comps = []
        while remaining:
            start = int(next(iter(remaining)))
            remaining.remove(start)
            comp = [start]
            q = deque([start])
            while q:
                u = q.popleft()
                for v in neighbors[u]:
                    v = int(v)
                    if v in remaining:
                        remaining.remove(v)
                        comp.append(v)
                        q.append(v)
            comps.append(comp)
        return comps

    def fundamental_cycles_for_component(component_nodes: List[int]) -> List[set]:
        allowed = set(int(x) for x in component_nodes)
        seen = set()
        parent = {}
        tree_edges = set()
        cycles = []

        for root in sorted(allowed):
            if root in seen:
                continue
            parent[root] = -1
            seen.add(root)
            stack = [root]
            while stack:
                u = stack.pop()
                for v in neighbors[u]:
                    v = int(v)
                    if v not in allowed:
                        continue
                    edge = tuple(sorted((int(u), v)))
                    if v not in seen:
                        seen.add(v)
                        parent[v] = int(u)
                        tree_edges.add(edge)
                        stack.append(v)
                    elif parent.get(int(u), -1) != v and edge not in tree_edges:
                        # Non-tree edge: path u->v in the DFS tree plus this edge is a cycle.
                        ancestors_u = {}
                        cur = int(u)
                        step = 0
                        while cur != -1:
                            ancestors_u[cur] = step
                            cur = int(parent.get(cur, -1))
                            step += 1

                        path_u = []
                        cur = int(v)
                        while cur not in ancestors_u and cur != -1:
                            path_u.append(cur)
                            cur = int(parent.get(cur, -1))
                        if cur == -1:
                            continue
                        lca = cur

                        cycle = {int(u), int(v), int(lca)}
                        cur = int(u)
                        while cur != lca:
                            cycle.add(cur)
                            cur = int(parent.get(cur, -1))
                        for node in path_u:
                            cycle.add(int(node))

                        if len(cycle) >= 3:
                            cycles.append(cycle)

        unique = []
        seen_cycles = set()
        for cycle in cycles:
            key = frozenset(cycle)
            if key in seen_cycles:
                continue
            seen_cycles.add(key)
            unique.append(cycle)
        return unique

    def farthest_node_from(src: int, allowed) -> int:
        allowed = set(int(x) for x in allowed)
        d = {int(src): 0}
        q = deque([int(src)])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                v = int(v)
                if v in allowed and v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return int(max(d, key=d.get)) if d else int(src)

    results = []
    seen_loop_keys = set()
    for component in connected_components_nodes(cycle_core):
        for cycle_nodes in fundamental_cycles_for_component(component):
            cycle_gateways = sorted(int(x) for x in cycle_nodes if x in gateways)

            if len(cycle_gateways) >= 2:
                best_pair = None
                best_score = -1
                for i, a in enumerate(cycle_gateways):
                    for b in cycle_gateways[i + 1:]:
                        p = bfs_subgraph(a, b, cycle_nodes)
                        if not p:
                            continue
                        score = min(len(p), len(cycle_nodes) - len(p) + 2)
                        if score > best_score:
                            best_score = score
                            best_pair = (a, b)
                if best_pair is None:
                    continue
                gateway_a, gateway_b = best_pair
            elif len(cycle_gateways) == 1:
                gateway_a = int(cycle_gateways[0])
                gateway_b = farthest_node_from(gateway_a, cycle_nodes)
            else:
                # Isolated closed cycle with no external gateway: keep it for review
                # by splitting across two far-apart cycle nodes.
                gateway_a = int(next(iter(cycle_nodes)))
                gateway_b = farthest_node_from(gateway_a, cycle_nodes)

            if gateway_a == gateway_b:
                continue

            arm1 = bfs_subgraph(gateway_a, gateway_b, cycle_nodes)
            if len(arm1) < 2:
                continue
            blocked = set(arm1[1:-1])
            arm2 = bfs_subgraph(gateway_a, gateway_b, set(cycle_nodes) - blocked)
            if len(arm2) < 2 or arm2 == arm1:
                continue

            loop_key = frozenset(set(arm1) | set(arm2))
            if loop_key in seen_loop_keys:
                continue
            seen_loop_keys.add(loop_key)
            results.append((int(gateway_a), int(gateway_b), arm1, arm2))

    return cycle_core, gateways, results


def dijkstra_dist_from_root(tree: SkeletonTree, root: int, spacing) -> np.ndarray:
    # Small graph; simple Dijkstra without heap is sufficient for skeleton sizes usually encountered here.
    spacing = np.asarray(spacing, dtype=float)
    n = len(tree.pts_vox)
    dist = np.full(n, np.inf, dtype=float)
    used = np.zeros(n, dtype=bool)
    dist[root] = 0.0
    for _ in range(n):
        cand = np.where(~used)[0]
        if len(cand) == 0:
            break
        u = int(cand[np.argmin(dist[cand])])
        if not np.isfinite(dist[u]):
            break
        used[u] = True
        pu = tree.pts_vox[u].astype(float) * spacing
        for v in tree.neighbors[u]:
            pv = tree.pts_vox[v].astype(float) * spacing
            w = float(np.linalg.norm(pv - pu))
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


def skeleton_endpoints_and_path(skel: np.ndarray):
    """Original fallback for unbranched vessels: longest endpoint-to-endpoint path."""
    pts, neighbors, degree = skeleton_graph(skel)
    n = len(pts)
    if n < 2:
        return None, None, None
    endpoints = list(np.where(degree == 1)[0])
    if len(endpoints) < 2:
        d2 = np.sum((pts - pts[0]) ** 2, axis=1)
        a = int(np.argmax(d2))
        d2 = np.sum((pts - pts[a]) ** 2, axis=1)
        b = int(np.argmax(d2))
        endpoints = [a, b]
    best_len = -1
    best_path = []
    for s in endpoints:
        for t in endpoints:
            if t == s:
                continue
            path = bfs_path_indices(neighbors, s, t)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path
    if not best_path:
        return None, None, None
    ep_a = tuple(map(int, pts[best_path[0]]))
    ep_b = tuple(map(int, pts[best_path[-1]]))
    path_vox = [tuple(map(int, pts[i])) for i in best_path]
    return ep_a, ep_b, path_vox
