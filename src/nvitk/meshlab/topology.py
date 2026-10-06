"""Mesh connectivity: edges, boundaries, components, Euler characteristic, adjacency."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from nvitk.types import Mesh


def edges(mesh: Mesh) -> np.ndarray:
    """``(3M, 2)`` directed half-edges of every face, in winding order."""
    f = mesh.faces
    return np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])


def unique_edges(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """``(edges, count)``: each undirected edge once (sorted pair) and how many faces use it."""
    e = np.sort(edges(mesh), axis=1)
    if not len(e):
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq, counts


def boundary_edges(mesh: Mesh) -> np.ndarray:
    """Edges used by exactly one face (the rims of holes and open borders)."""
    uniq, counts = unique_edges(mesh)
    return uniq[counts == 1]


def non_manifold_edges(mesh: Mesh) -> np.ndarray:
    """Edges shared by more than two faces."""
    uniq, counts = unique_edges(mesh)
    return uniq[counts > 2]


def is_watertight(mesh: Mesh) -> bool:
    """Closed 2-manifold: every edge shared by exactly two faces."""
    if not mesh.n_faces:
        return False
    _, counts = unique_edges(mesh)
    return bool(np.all(counts == 2))


def boundary_loops(mesh: Mesh) -> list[np.ndarray]:
    """Vertex loops around each hole / open border (ordered vertex indices)."""
    b = boundary_edges(mesh)
    if not len(b):
        return []
    nxt: dict[int, list[int]] = {}
    for a, c in b:
        nxt.setdefault(int(a), []).append(int(c))
        nxt.setdefault(int(c), []).append(int(a))
    seen: set[tuple[int, int]] = set()
    loops = []
    for start in list(nxt):
        for first in nxt[start]:
            if (min(start, first), max(start, first)) in seen:
                continue
            loop = [start]
            prev, cur = start, first
            seen.add((min(start, first), max(start, first)))
            while cur != start and len(loop) <= len(b):
                loop.append(cur)
                options = [v for v in nxt[cur] if v != prev and (min(cur, v), max(cur, v)) not in seen]
                if not options:
                    break
                prev, cur = cur, options[0]
                seen.add((min(prev, cur), max(prev, cur)))
            loops.append(np.asarray(loop))
    return loops


def vertex_adjacency(mesh: Mesh) -> sparse.csr_matrix:
    """Symmetric ``(N, N)`` 0/1 matrix of vertices sharing an edge."""
    n = mesh.n_vertices
    e = edges(mesh)
    if not len(e):
        return sparse.csr_matrix((n, n))
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj.data[:] = 1.0
    return adj


def face_components(mesh: Mesh) -> tuple[int, np.ndarray]:
    """``(count, label per face)``: faces connected through shared vertices."""
    if not mesh.n_faces:
        return 0, np.zeros(0, dtype=np.int64)
    n_comp, vlabels = connected_components(vertex_adjacency(mesh), directed=False)
    flabels = vlabels[mesh.faces[:, 0]]
    # Renumber densely over the faces (isolated vertices own no faces).
    uniq, dense = np.unique(flabels, return_inverse=True)
    return int(len(uniq)), dense.astype(np.int64)


def euler_characteristic(mesh: Mesh) -> int:
    """``V - E + F`` over the vertices the faces use."""
    used = np.unique(mesh.faces)
    uniq, _ = unique_edges(mesh)
    return int(len(used) - len(uniq) + mesh.n_faces)


def genus(mesh: Mesh) -> float:
    """Genus of a closed surface per component: ``(2·C − χ − B) / 2`` (holes = B).

    ``B`` counts boundary loops, so an open surface (a disc) has genus 0 too. A
    sphere is 0, a torus 1. Non-integral values flag a non-manifold mesh.
    """
    n_comp, _ = face_components(mesh)
    loops = len(boundary_loops(mesh))
    return (2 * n_comp - euler_characteristic(mesh) - loops) / 2.0


__all__ = [
    "boundary_edges",
    "boundary_loops",
    "edges",
    "euler_characteristic",
    "face_components",
    "genus",
    "is_watertight",
    "non_manifold_edges",
    "unique_edges",
    "vertex_adjacency",
]
