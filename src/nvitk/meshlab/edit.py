"""Local, hand-guided edits of meshes and point clouds (around a picked point).

The operations a user does with the mouse in MeshLab: erase or keep what lies
within a sphere, take or drop the connected piece under the cursor, push or pull
the surface, smooth a spot. All take a centre in the mesh's own coordinates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nvitk.meshlab.cleaning import _rebuild, remove_unreferenced_vertices
from nvitk.meshlab.topology import face_components, vertex_adjacency
from nvitk.types import Mesh, PointCloud


def _faces_near(mesh: Mesh, centre: Sequence[float], radius: float) -> np.ndarray:
    """Boolean per face: any corner within *radius* of *centre*."""
    d = np.linalg.norm(mesh.vertices - np.asarray(centre, dtype=float), axis=1)
    return np.any(d[mesh.faces] <= float(radius), axis=1)


def erase_sphere(mesh: Mesh, centre: Sequence[float], radius: float, *, keep_inside: bool = False) -> Mesh:
    """Remove the faces touching a sphere (or keep only those, with *keep_inside*)."""
    near = _faces_near(mesh, centre, radius)
    keep = near if keep_inside else ~near
    idx = np.flatnonzero(keep)
    return remove_unreferenced_vertices(_rebuild(mesh, mesh.vertices, mesh.faces[idx], np.arange(mesh.n_vertices), idx))


def piece_at(mesh: Mesh, point: Sequence[float]) -> np.ndarray:
    """Boolean per face: the connected piece nearest to *point*."""
    _, labels = face_components(mesh)
    centres = mesh.triangles.mean(axis=1)
    nearest = int(np.argmin(np.linalg.norm(centres - np.asarray(point, dtype=float), axis=1)))
    return labels == labels[nearest]


def keep_piece(mesh: Mesh, point: Sequence[float], *, delete: bool = False) -> Mesh:
    """Keep only the piece under *point* — or delete it, with *delete*."""
    sel = piece_at(mesh, point)
    keep = ~sel if delete else sel
    idx = np.flatnonzero(keep)
    return remove_unreferenced_vertices(_rebuild(mesh, mesh.vertices, mesh.faces[idx], np.arange(mesh.n_vertices), idx))


def _falloff(d: np.ndarray, radius: float) -> np.ndarray:
    """Smooth bump: 1 at the centre, 0 at *radius* (C¹ at the rim)."""
    t = np.clip(d / max(float(radius), 1e-9), 0.0, 1.0)
    return (1.0 - t ** 2) ** 2


def sculpt(mesh: Mesh, centre: Sequence[float], radius: float, amount: float) -> Mesh:
    """Push (*amount* > 0, outward) or pull (< 0) the surface around *centre*.

    Vertices move along their normals by ``amount × falloff(distance)``, a smooth
    bump that is zero at *radius* — so the edit blends into the rest.
    """
    d = np.linalg.norm(mesh.vertices - np.asarray(centre, dtype=float), axis=1)
    w = _falloff(d, radius) * float(amount)
    return mesh.with_vertices(mesh.vertices + mesh.vertex_normals * w[:, None])


def smooth_spot(mesh: Mesh, centre: Sequence[float], radius: float, *, iterations: int = 10, strength: float = 0.5) -> Mesh:
    """Laplacian smoothing confined to a sphere (fades out towards its rim)."""
    adj = vertex_adjacency(mesh)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    d = np.linalg.norm(mesh.vertices - np.asarray(centre, dtype=float), axis=1)
    w = (_falloff(d, radius) * float(strength))[:, None]
    v = mesh.vertices.copy()
    for _ in range(int(iterations)):
        avg = (adj @ v) * inv[:, None]
        v = v + w * (avg - v)
    return mesh.with_vertices(v)


def erase_points(cloud: PointCloud, centre: Sequence[float], radius: float, *, keep_inside: bool = False) -> PointCloud:
    """Drop the points within a sphere (or keep only those)."""
    d = np.linalg.norm(cloud.points - np.asarray(centre, dtype=float), axis=1)
    inside = d <= float(radius)
    return cloud.subset(inside if keep_inside else ~inside)


def faces_of_selection(mesh: Mesh, selected: np.ndarray, *, whole: bool = True) -> np.ndarray:
    """Boolean per face from a per-vertex selection: faces with all corners selected
    (*whole*), or with any."""
    sel = np.asarray(selected, dtype=bool)[mesh.faces]
    return sel.all(axis=1) if whole else sel.any(axis=1)


def delete_selected(mesh: Mesh, selected: np.ndarray) -> Mesh:
    """Remove the selected vertices and every face touching them."""
    keep = np.flatnonzero(~faces_of_selection(mesh, selected, whole=False))
    return remove_unreferenced_vertices(_rebuild(mesh, mesh.vertices, mesh.faces[keep], np.arange(mesh.n_vertices), keep))


def keep_selected(mesh: Mesh, selected: np.ndarray) -> Mesh:
    """Only the faces whose corners are all selected."""
    keep = np.flatnonzero(faces_of_selection(mesh, selected, whole=True))
    return remove_unreferenced_vertices(_rebuild(mesh, mesh.vertices, mesh.faces[keep], np.arange(mesh.n_vertices), keep))


def grow_selection(mesh: Mesh, selected: np.ndarray, rings: int = 1) -> np.ndarray:
    """The selection grown by *rings* rings of neighbouring vertices (negative shrinks it)."""
    sel = np.asarray(selected, dtype=bool).copy()
    adj = vertex_adjacency(mesh)
    for _ in range(abs(int(rings))):
        if rings > 0:
            sel = sel | (adj @ sel.astype(np.float64) > 0)
        else:
            sel = sel & ~(adj @ (~sel).astype(np.float64) > 0)
    return sel


def connected_selection(mesh: Mesh, point: Sequence[float]) -> np.ndarray:
    """Boolean per vertex: the connected piece nearest to *point*."""
    faces = piece_at(mesh, point)
    out = np.zeros(mesh.n_vertices, dtype=bool)
    out[mesh.faces[faces].ravel()] = True
    return out


__all__ = [
    "connected_selection",
    "delete_selected",
    "erase_points",
    "erase_sphere",
    "faces_of_selection",
    "grow_selection",
    "keep_piece",
    "keep_selected",
    "piece_at",
    "sculpt",
    "smooth_spot",
]
