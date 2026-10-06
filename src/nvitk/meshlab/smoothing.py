"""Mesh smoothing: Laplacian and Taubin (λ|μ, volume-preserving) on the umbrella operator."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from nvitk.meshlab.topology import boundary_edges, vertex_adjacency
from nvitk.types import Mesh


def _umbrella(mesh: Mesh) -> sparse.csr_matrix:
    """Row-normalised adjacency: each vertex's neighbour average."""
    adj = vertex_adjacency(mesh)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sparse.diags(inv) @ adj


def _fixed_mask(mesh: Mesh, keep_boundary: bool) -> np.ndarray:
    """Vertices that must not move: isolated ones, and the open border if asked."""
    fixed = np.ones(mesh.n_vertices, dtype=bool)
    fixed[np.unique(mesh.faces)] = False
    if keep_boundary:
        fixed[np.unique(boundary_edges(mesh))] = True
    return fixed


def laplacian_smooth(mesh: Mesh, *, iterations: int = 10, lam: float = 0.5, keep_boundary: bool = True) -> Mesh:
    """Move each vertex *lam* of the way to its neighbours' mean, *iterations* times.

    Shrinks the surface as it smooths; :func:`taubin_smooth` does not.
    """
    if not mesh.n_faces or iterations <= 0:
        return mesh.copy()
    op = _umbrella(mesh)
    fixed = _fixed_mask(mesh, keep_boundary)
    v = mesh.vertices.copy()
    for _ in range(int(iterations)):
        delta = op @ v - v
        delta[fixed] = 0.0
        v = v + float(lam) * delta
    return mesh.with_vertices(v)


def taubin_smooth(
    mesh: Mesh,
    *,
    iterations: int = 20,
    lam: float = 0.5,
    mu: float = -0.53,
    keep_boundary: bool = True,
) -> Mesh:
    """Taubin λ|μ smoothing: alternate a shrinking and an inflating Laplacian step.

    Removes surface noise (marching-cubes staircases) with almost no volume loss.
    ``mu`` must be negative with ``|mu| > lam``.
    """
    if not mesh.n_faces or iterations <= 0:
        return mesh.copy()
    if mu >= 0 or abs(mu) <= lam:
        raise ValueError("Taubin smoothing needs mu < 0 and |mu| > lambda.")
    op = _umbrella(mesh)
    fixed = _fixed_mask(mesh, keep_boundary)
    v = mesh.vertices.copy()
    for _ in range(int(iterations)):
        for factor in (float(lam), float(mu)):
            delta = op @ v - v
            delta[fixed] = 0.0
            v = v + factor * delta
    return mesh.with_vertices(v)


def smooth_point_data(mesh: Mesh, key: str, *, iterations: int = 5, lam: float = 0.5) -> Mesh:
    """Smooth a per-vertex array over the surface (e.g. a noisy curvature map)."""
    if key not in mesh.point_data:
        raise KeyError(f"No point data {key!r}.")
    op = _umbrella(mesh)
    vals = np.asarray(mesh.point_data[key], dtype=float)
    for _ in range(int(iterations)):
        vals = vals + float(lam) * (op @ vals - vals)
    return mesh.with_point_data(**{key: vals})


__all__ = ["laplacian_smooth", "smooth_point_data", "taubin_smooth"]
