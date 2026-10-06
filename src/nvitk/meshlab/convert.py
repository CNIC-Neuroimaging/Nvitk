"""Conversions between nvitk surface types and PyVista / VTK, and small array helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError
from nvitk.types import Mesh, PointCloud


def require_pyvista() -> Any:
    """Import and return ``pyvista``, or raise an install hint."""
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailableError("This operation needs PyVista: pip install pyvista") from exc
    return pv


def to_pyvista(mesh: Mesh | PointCloud) -> Any:
    """A :class:`pyvista.PolyData` with the geometry and per-element arrays."""
    pv = require_pyvista()
    if isinstance(mesh, PointCloud):
        poly = pv.PolyData(np.asarray(mesh.points, dtype=float))
        for key, arr in mesh.point_data.items():
            poly.point_data[key] = arr
        return poly
    faces = np.hstack([np.full((mesh.n_faces, 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)
    for key, arr in mesh.point_data.items():
        poly.point_data[key] = arr
    for key, arr in mesh.cell_data.items():
        poly.cell_data[key] = arr
    return poly


def from_pyvista(poly: Any, *, metadata: dict[str, Any] | None = None, keep_data: bool = True) -> Mesh:
    """A :class:`Mesh` from a PyVista dataset: its surface, triangulated.

    Line and vertex cells are dropped; per-face arrays are kept only when every
    cell is a triangle (otherwise their rows would not line up with the faces).
    """
    pv = require_pyvista()
    surf = poly if isinstance(poly, pv.PolyData) else poly.extract_geometry()
    if surf.n_cells and not surf.is_all_triangles:
        surf = surf.triangulate()
    raw = np.asarray(surf.faces)
    faces = raw.reshape(-1, 4)[:, 1:] if raw.size else np.zeros((0, 3), dtype=np.int64)
    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}
    if keep_data:
        for key in surf.point_data.keys():
            arr = np.asarray(surf.point_data[key])
            if arr.shape[:1] == (surf.n_points,) and not key.lower().startswith("vtk"):
                point_data[key] = arr
        if surf.n_cells == len(faces):
            for key in surf.cell_data.keys():
                arr = np.asarray(surf.cell_data[key])
                if arr.shape[:1] == (len(faces),) and not key.lower().startswith("vtk"):
                    cell_data[key] = arr
    return Mesh(
        vertices=np.asarray(surf.points, dtype=float),
        faces=faces,
        metadata=dict(metadata or {}),
        point_data=point_data,
        cell_data=cell_data,
    )


def points_of(obj: Mesh | PointCloud | np.ndarray) -> np.ndarray:
    """``(N, 3)`` coordinates of a mesh's vertices, a cloud's points, or an array."""
    if isinstance(obj, Mesh):
        return obj.vertices
    if isinstance(obj, PointCloud):
        return obj.points
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points; got {arr.shape}.")
    return arr


def mesh_to_point_cloud(mesh: Mesh) -> PointCloud:
    """The mesh's vertices as a point cloud (with their per-vertex data and normals)."""
    data = dict(mesh.point_data)
    if mesh.n_faces:
        data.setdefault("normals", mesh.vertex_normals)
    return PointCloud(points=mesh.vertices.copy(), metadata=dict(mesh.metadata), point_data=data)


__all__ = [
    "from_pyvista",
    "mesh_to_point_cloud",
    "points_of",
    "require_pyvista",
    "to_pyvista",
]
