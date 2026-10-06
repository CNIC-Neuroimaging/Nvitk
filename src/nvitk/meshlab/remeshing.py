"""Changing a mesh's resolution or shape: decimation, subdivision, clipping, hulls."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk.meshlab.convert import from_pyvista, points_of, to_pyvista
from nvitk.types import Mesh, PointCloud

#: Subdivision schemes (VTK): ``linear`` splits faces in place, ``loop`` and
#: ``butterfly`` smooth as they refine.
SUBDIVISION_METHODS: tuple[str, ...] = ("linear", "loop", "butterfly")

#: Decimation algorithms: VTK's quadric-error ``vtkQuadricDecimation`` and the
#: topology-preserving ``vtkDecimatePro``.
DECIMATION_METHODS: tuple[str, ...] = ("quadric", "pro")


def decimate(
    mesh: Mesh,
    reduction: float = 0.5,
    *,
    method: str = "quadric",
    preserve_topology: bool = True,
) -> Mesh:
    """Remove *reduction* (0–1) of the faces while keeping the shape.

    ``"pro"`` with *preserve_topology* never opens holes or splits the surface,
    at the cost of sometimes stopping short of the target.
    """
    reduction = float(reduction)
    if not 0.0 <= reduction < 1.0:
        raise ValueError("reduction must be in [0, 1).")
    if reduction == 0.0 or not mesh.n_faces:
        return mesh.copy()
    poly = to_pyvista(mesh)
    if str(method) == "pro":
        out = poly.decimate_pro(reduction, preserve_topology=bool(preserve_topology))
    elif str(method) == "quadric":
        out = poly.decimate(reduction)
    else:
        raise ValueError(f"method must be one of {DECIMATION_METHODS}.")
    return from_pyvista(out, metadata=mesh.metadata, keep_data=False)


def subdivide(mesh: Mesh, iterations: int = 1, *, method: str = "loop") -> Mesh:
    """Split every face into four, *iterations* times (×4ⁿ faces)."""
    if str(method) not in SUBDIVISION_METHODS:
        raise ValueError(f"method must be one of {SUBDIVISION_METHODS}.")
    if iterations <= 0 or not mesh.n_faces:
        return mesh.copy()
    from nvitk.meshlab.cleaning import clean

    # VTK's subdivision filters need a clean, manifold triangle mesh.
    poly = to_pyvista(clean(mesh))
    out = poly.subdivide(int(iterations), subfilter=str(method))
    return from_pyvista(out, metadata=mesh.metadata, keep_data=False)


def clip_plane(
    mesh: Mesh,
    origin: Sequence[float],
    normal: Sequence[float],
    *,
    keep: str = "below",
    close: bool = False,
) -> Mesh:
    """Cut *mesh* with a plane, keeping the part ``"below"`` (against the normal)
    or ``"above"`` it. *close* caps the cut (a watertight result from a closed mesh)."""
    poly = to_pyvista(mesh)
    invert = str(keep) != "above"
    if close:
        out = poly.clip_closed_surface(normal=tuple(float(v) for v in (np.negative(normal) if invert else normal)),
                                       origin=tuple(float(v) for v in origin))
    else:
        out = poly.clip(normal=tuple(float(v) for v in normal), origin=tuple(float(v) for v in origin),
                        invert=invert)
    return from_pyvista(out, metadata=mesh.metadata)


def clip_box(mesh: Mesh, bounds: Sequence[float], *, inside: bool = True) -> Mesh:
    """Keep the part of *mesh* inside (or outside) ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
    poly = to_pyvista(mesh)
    out = poly.clip_box(tuple(float(v) for v in bounds), invert=bool(inside))
    return from_pyvista(out, metadata=mesh.metadata)


def convex_hull(obj: Mesh | PointCloud | np.ndarray, *, metadata: dict[str, Any] | None = None) -> Mesh:
    """Convex hull of a mesh's vertices or a point set, outward-oriented."""
    from scipy.spatial import ConvexHull

    pts = points_of(obj)
    if len(pts) < 4:
        raise ValueError("A convex hull needs at least 4 points.")
    hull = ConvexHull(pts)
    faces = hull.simplices.copy()
    # Orient each face away from the hull's interior.
    centre = pts[hull.vertices].mean(axis=0)
    tri = pts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", normals, tri[:, 0] - centre) < 0
    faces[flip] = faces[flip][:, ::-1]
    used = np.unique(faces)
    remap = np.full(len(pts), -1)
    remap[used] = np.arange(len(used))
    meta = dict(metadata if metadata is not None else getattr(obj, "metadata", {}) or {})
    return Mesh(vertices=pts[used], faces=remap[faces], metadata=meta)


def remesh_voxel(mesh: Mesh, spacing: float = 1.0, *, smooth_iterations: int = 10) -> Mesh:
    """Rebuild *mesh* through a voxel grid: watertight, uniform triangles.

    Voxelises the enclosed solid at *spacing*, runs marching cubes and a light
    Taubin smoothing. Repairs self-intersections and holes; loses details finer
    than the spacing.
    """
    from nvitk.meshlab.marching_cubes import marching_cubes_binary
    from nvitk.meshlab.smoothing import taubin_smooth
    from nvitk.meshlab.voxelize import voxelize_to_grid

    mask, affine = voxelize_to_grid(mesh, spacing=float(spacing), margin=2)
    from nvitk.types import Image

    img = Image(data=mask.astype(np.uint8), metadata={"affine": affine}, axes="XYZ")
    out = marching_cubes_binary(img, world_space=True)
    if out is None:
        raise ValueError("Nothing enclosed at that spacing: is the mesh closed?")
    out.metadata = dict(mesh.metadata)
    return taubin_smooth(out, iterations=int(smooth_iterations)) if smooth_iterations > 0 else out


__all__ = [
    "DECIMATION_METHODS",
    "SUBDIVISION_METHODS",
    "clip_box",
    "clip_plane",
    "convex_hull",
    "decimate",
    "remesh_voxel",
    "subdivide",
]


def isotropic_remesh(
    mesh: Mesh,
    target_length: float = 0.0,
    *,
    iterations: int = 10,
    feature_angle: float = 30.0,
    adaptive: bool = False,
    max_surface_distance: float = 0.0,
) -> Mesh:
    """Even, well-shaped triangles of edge length *target_length* (mm; 0 = the mean edge).

    MeshLab's isotropic explicit remeshing (PyMeshLab): edges are split,
    collapsed and flipped, and vertices relaxed and reprojected onto the original
    surface; edges sharper than *feature_angle* (degrees) are kept. *adaptive*
    adapts the length to the curvature; *max_surface_distance* (mm, 0 = off)
    bounds how far the result may drift from the input.
    """
    from nvitk.meshlab.pymeshlab_filters import run_meshlab_filter

    if not mesh.n_faces:
        raise ValueError("The mesh has no faces to remesh.")
    if target_length <= 0:
        edges = mesh.vertices[mesh.faces[:, [1, 2, 0]]] - mesh.vertices[mesh.faces]
        target_length = float(np.linalg.norm(edges, axis=2).mean())
    params: dict[str, Any] = {
        "targetlen": f"{float(target_length)}", "iterations": int(iterations),
        "featuredeg": float(feature_angle), "adaptive": bool(adaptive),
        "checksurfdist": max_surface_distance > 0,
    }
    if max_surface_distance > 0:
        params["maxsurfdist"] = f"{float(max_surface_distance)}"
    res = run_meshlab_filter(mesh, "meshing_isotropic_explicit_remeshing", params)
    out = res.current if isinstance(res.current, Mesh) else mesh
    return Mesh(out.vertices, out.faces, metadata=dict(mesh.metadata))
