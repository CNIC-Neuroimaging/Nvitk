"""Measurements on meshes and point clouds.

Units follow the coordinates: mm, mm², mm³ for world-space meshes (the default
for meshes built from images), voxels otherwise.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk.meshlab.convert import points_of, require_pyvista, to_pyvista
from nvitk.meshlab.topology import (
    boundary_loops,
    euler_characteristic,
    face_components,
    genus,
    is_watertight,
    non_manifold_edges,
    unique_edges,
)
from nvitk.types import Mesh, PointCloud

#: Curvatures :func:`curvature` computes.
CURVATURE_KINDS: tuple[str, ...] = ("mean", "gaussian", "maximum", "minimum")


def surface_area(mesh: Mesh) -> float:
    """Total face area."""
    return float(mesh.face_areas.sum()) if mesh.n_faces else 0.0


def signed_volume(mesh: Mesh) -> float:
    """Divergence-theorem volume; positive when the normals point outwards."""
    if not mesh.n_faces:
        return 0.0
    tri = mesh.triangles
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def volume(mesh: Mesh) -> float:
    """Enclosed volume (absolute; only meaningful for a closed surface)."""
    return abs(signed_volume(mesh))


def surface_centroid(mesh: Mesh) -> np.ndarray:
    """Area-weighted centre of the surface."""
    if not mesh.n_faces:
        return mesh.vertices.mean(axis=0) if mesh.n_vertices else np.zeros(3)
    areas = mesh.face_areas
    centres = mesh.triangles.mean(axis=1)
    total = areas.sum()
    return (centres * areas[:, None]).sum(axis=0) / total if total > 0 else centres.mean(axis=0)


def volume_centroid(mesh: Mesh) -> np.ndarray:
    """Centre of mass of the enclosed solid (closed surface)."""
    tri = mesh.triangles
    vols = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    total = vols.sum()
    if abs(total) < 1e-12:
        return surface_centroid(mesh)
    centres = tri.sum(axis=1) / 4.0
    return (centres * vols[:, None]).sum(axis=0) / total


def edge_lengths(mesh: Mesh) -> np.ndarray:
    """Length of every undirected edge."""
    uniq, _ = unique_edges(mesh)
    if not len(uniq):
        return np.zeros(0)
    return np.linalg.norm(mesh.vertices[uniq[:, 0]] - mesh.vertices[uniq[:, 1]], axis=1)


def principal_axes(obj: Mesh | PointCloud | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centre, axes, extents)``: PCA of the vertices / points.

    ``axes`` rows are unit vectors, largest spread first; ``extents`` the length
    along each (max − min of the projections).
    """
    pts = points_of(obj)
    centre = pts.mean(axis=0)
    centred = pts - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt.T
    extents = proj.max(axis=0) - proj.min(axis=0)
    return centre, vt, extents


def sphericity(mesh: Mesh) -> float:
    """``π^(1/3) (6V)^(2/3) / A``: 1 for a sphere, smaller for anything else."""
    area = surface_area(mesh)
    vol = volume(mesh)
    if area <= 0:
        return 0.0
    return float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)


def mesh_metrics(mesh: Mesh) -> dict[str, Any]:
    """The usual summary: size, area, volume, shape, topology, quality."""
    n_comp, _ = face_components(mesh)
    lengths = edge_lengths(mesh)
    bounds = mesh.bounds
    closed = is_watertight(mesh)
    out: dict[str, Any] = {
        "vertices": mesh.n_vertices,
        "faces": mesh.n_faces,
        "components": n_comp,
        "watertight": closed,
        "holes": len(boundary_loops(mesh)) if not closed else 0,
        "non_manifold_edges": int(len(non_manifold_edges(mesh))),
        "euler_characteristic": euler_characteristic(mesh),
        "genus": genus(mesh),
        "area": surface_area(mesh),
        "volume": volume(mesh) if closed else float("nan"),
        "sphericity": sphericity(mesh) if closed else float("nan"),
        "centroid": surface_centroid(mesh).tolist(),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "size": (bounds[1] - bounds[0]).tolist(),
        "edge_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "edge_length_min": float(lengths.min()) if len(lengths) else 0.0,
        "edge_length_max": float(lengths.max()) if len(lengths) else 0.0,
    }
    if mesh.n_faces:
        _, axes, extents = principal_axes(mesh)
        out["principal_extents"] = extents.tolist()
    return out


def point_cloud_metrics(cloud: PointCloud) -> dict[str, Any]:
    """Size, bounds, spread and spacing of a point cloud."""
    from scipy.spatial import cKDTree

    out: dict[str, Any] = {"points": cloud.n_points}
    if not cloud.n_points:
        return out
    bounds = cloud.bounds
    centre, _, extents = principal_axes(cloud)
    out.update({
        "centroid": centre.tolist(),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "size": (bounds[1] - bounds[0]).tolist(),
        "principal_extents": extents.tolist(),
    })
    if cloud.n_points > 1:
        d, _ = cKDTree(cloud.points).query(cloud.points, k=2)
        out["nn_spacing_mean"] = float(d[:, 1].mean())
        out["nn_spacing_median"] = float(np.median(d[:, 1]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Curvature
# ──────────────────────────────────────────────────────────────────────────────


def curvature(mesh: Mesh, kind: str = "mean") -> np.ndarray:
    """Per-vertex curvature (1 / mesh units): mean, gaussian, maximum or minimum.

    VTK's discrete estimators (``vtkCurvatures``): positive mean curvature on a
    convex, outward-oriented surface (``1/r`` on a sphere of radius ``r``). On a
    raw marching-cubes surface the staircase concentrates Gaussian curvature at
    the steps; smooth first (:func:`~nvitk.meshlab.taubin_smooth`).
    """
    kind = str(kind).lower()
    if kind not in CURVATURE_KINDS:
        raise ValueError(f"kind must be one of {CURVATURE_KINDS}.")
    require_pyvista()
    vals = np.asarray(to_pyvista(mesh).curvature(curv_type=kind), dtype=float)
    return np.nan_to_num(vals)


# ──────────────────────────────────────────────────────────────────────────────
# Distances
# ──────────────────────────────────────────────────────────────────────────────


def distance_to_surface(points: Any, surface: Mesh, *, signed: bool = False) -> np.ndarray:
    """Distance from each point (or vertex) to the closest point on *surface*.

    *signed*: negative inside a closed, outward-oriented surface.
    """
    import vtk

    pts = points_of(points)
    target = to_pyvista(surface)
    if signed:
        implicit = vtk.vtkImplicitPolyDataDistance()
        implicit.SetInput(target)
        return np.asarray([implicit.EvaluateFunction(p) for p in pts], dtype=float)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        out[i] = float(dist2) ** 0.5
    return out


def surface_distance_stats(a: Mesh | PointCloud, b: Mesh | PointCloud) -> dict[str, float]:
    """Symmetric surface distances between *a* and *b*.

    Mesh targets are measured point-to-surface, point clouds point-to-point:
    mean (ASSD), RMS, Hausdorff (max) and 95th percentile Hausdorff.
    """
    from scipy.spatial import cKDTree

    def _one_way(src: Mesh | PointCloud, dst: Mesh | PointCloud) -> np.ndarray:
        if isinstance(dst, Mesh) and dst.n_faces:
            return distance_to_surface(src, dst)
        d, _ = cKDTree(points_of(dst)).query(points_of(src))
        return d

    ab = _one_way(a, b)
    ba = _one_way(b, a)
    both = np.concatenate([ab, ba])
    return {
        "mean_a_to_b": float(ab.mean()),
        "mean_b_to_a": float(ba.mean()),
        "assd": float(both.mean()),
        "rms": float(np.sqrt((both ** 2).mean())),
        "hausdorff": float(both.max()),
        "hausdorff95": float(max(np.percentile(ab, 95), np.percentile(ba, 95))),
    }


def chamfer_distance(a: Any, b: Any) -> float:
    """Mean nearest-neighbour distance a→b plus b→a (point sets)."""
    from scipy.spatial import cKDTree

    pa, pb = points_of(a), points_of(b)
    d_ab, _ = cKDTree(pb).query(pa)
    d_ba, _ = cKDTree(pa).query(pb)
    return float(d_ab.mean() + d_ba.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Cross-sections
# ──────────────────────────────────────────────────────────────────────────────


def _polygon_area_3d(points: np.ndarray, normal: np.ndarray) -> float:
    """Area of a planar polygon (ordered points) with the given plane normal."""
    if len(points) < 3:
        return 0.0
    total = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    return float(abs(np.dot(total, normal)) / 2.0)


def cross_section(mesh: Mesh, origin: Sequence[float], normal: Sequence[float]) -> dict[str, Any]:
    """Cut *mesh* by a plane: contour loops, enclosed area and perimeter.

    Returns ``{"loops": [ (K, 3) arrays ], "area": …, "perimeter": …}``; the area
    is that of the outer loops minus nested ones only when they are separate
    pieces — loops are summed by absolute area, which is right for the usual
    case of disjoint cross-sections (vessels, organs).
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) or 1.0)
    poly = to_pyvista(mesh)
    cut = poly.slice(normal=tuple(normal), origin=tuple(float(v) for v in origin))
    if cut.n_points == 0:
        return {"loops": [], "area": 0.0, "perimeter": 0.0}
    stripped = cut.strip(join=True, max_length=100000)
    loops = []
    lines = np.asarray(stripped.lines)
    i = 0
    pts = np.asarray(stripped.points)
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1: i + 1 + n]
        loops.append(pts[ids])
        i += n + 1
    area = 0.0
    perimeter = 0.0
    for loop in loops:
        closed = len(loop) > 2 and np.allclose(loop[0], loop[-1])
        ring = loop[:-1] if closed else loop
        perimeter += float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum())
        area += _polygon_area_3d(ring, normal)
    return {"loops": loops, "area": area, "perimeter": perimeter}


__all__ = [
    "CURVATURE_KINDS",
    "chamfer_distance",
    "cross_section",
    "curvature",
    "distance_to_surface",
    "edge_lengths",
    "mesh_metrics",
    "point_cloud_metrics",
    "principal_axes",
    "signed_volume",
    "sphericity",
    "surface_area",
    "surface_centroid",
    "surface_distance_stats",
    "volume",
    "volume_centroid",
]
