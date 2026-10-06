"""Tubular structures on meshes: centerlines, diameter profiles, branches and bifurcations.

VMTK-style measurements on any vessel-like surface (a segmented artery, an
airway, a colon):

- :func:`mesh_centerlines` — the surface is filled into voxels, thinned to a
  skeleton (:func:`nvitk.morphology.centerline.skeletonize_binary`) and split
  into branches between endpoints and junctions
  (:func:`nvitk.morphology.polyline_graph.branch_polylines_from_skeleton`); each
  branch is smoothed and resampled in millimetres.
- :func:`branch_profiles` — at stations along each branch the surface is cut
  perpendicular to the centerline: lumen area, perimeter, equivalent / min / max
  diameter, circularity, and the inscribed radius (distance to the wall).
- :func:`branch_metrics` / :func:`bifurcations` — length, tortuosity, curvature,
  stenosis; the angles between branches where they meet.
- :func:`vessel_section_at` — one cross-section at any point, perpendicular to
  the local vessel axis (found from the wall itself, no centerline needed).
- :func:`thickness_map` / :func:`radius_map` — per-vertex local diameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from nvitk.types import Mesh


@dataclass
class VesselBranch:
    """One centerline branch, in millimetres, with its per-station profile."""

    id: int
    points: np.ndarray
    tangents: np.ndarray
    arc_length: np.ndarray
    #: Per-station arrays (``area``, ``diameter``, …) from :func:`branch_profiles`.
    profile: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.arc_length[-1]) if len(self.arc_length) else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Polyline helpers (host NumPy)
# ──────────────────────────────────────────────────────────────────────────────


def _arc(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _smooth_resample(points: np.ndarray, *, step: float, smooth: float) -> np.ndarray:
    """Spline-smooth a polyline (*smooth* × points, as SciPy's ``s``) and resample at *step* mm."""
    pts = np.asarray(points, dtype=float)
    if len(pts) >= 4 and smooth > 0:
        try:
            from scipy.interpolate import splev, splprep

            tck, _ = splprep(pts.T, s=float(smooth) * len(pts), k=min(3, len(pts) - 1))
            dense = max(len(pts) * 8, 64)
            pts = np.stack(splev(np.linspace(0, 1, dense), tck), axis=1)
        except Exception:  # noqa: BLE001 — a degenerate polyline keeps its points
            pass
    s = _arc(pts)
    if s[-1] <= 1e-9:
        return pts[:1]
    n = max(int(s[-1] / max(step, 1e-6)) + 1, 2)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, k]) for k in range(3)], axis=1)


def _tangents(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.tile([0.0, 0.0, 1.0], (len(points), 1))
    t = np.gradient(points, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return np.divide(t, n, out=np.zeros_like(t), where=n > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Centerlines
# ──────────────────────────────────────────────────────────────────────────────


def auto_voxel_size(mesh: Mesh) -> float:
    """A voxel size for skeletonising *mesh*: a fifth of a typical lumen radius.

    The radius comes from the median thickness of a sample of vertices (rays to
    the opposite wall), so thin and thick vessels get comparable resolution;
    bounded so the grid stays under ~350 voxels a side.
    """
    lo, hi = mesh.bounds
    extent = float(np.max(hi - lo))
    try:
        idx = np.random.default_rng(0).choice(mesh.n_vertices, min(400, mesh.n_vertices), replace=False)
        diam = thickness_map(mesh, vertices=idx)
        diam = diam[np.isfinite(diam) & (diam > 0)]
        guess = float(np.median(diam)) / 10.0 if diam.size else extent / 150.0
    except Exception:  # noqa: BLE001
        guess = extent / 150.0
    return float(np.clip(guess, extent / 350.0, extent / 40.0))


def mesh_centerlines(
    mesh: Mesh,
    *,
    voxel_size: float = 0.0,
    min_branch_length: float = 0.0,
    step: float = 0.0,
    smooth: float = 0.5,
    prune: bool = True,
) -> list[VesselBranch]:
    """Centerline branches of a closed tubular surface (world mm).

    *voxel_size* (mm, 0 = :func:`auto_voxel_size`) is the skeletonisation grid;
    branches shorter than *min_branch_length* (mm, 0 = 4 voxels) are dropped,
    tiny loops and short spurs pruned. Branches are resampled every *step* mm
    (0 = the voxel size) after spline smoothing (*smooth*, 0 = off).
    """
    from nvitk.core.backend import using
    from nvitk.core.array import to_numpy
    from nvitk.meshlab.voxelize import voxelize_to_grid
    from nvitk.morphology.centerline import skeletonize_binary
    from nvitk.morphology.polyline_graph import branch_polylines_from_skeleton

    vs = float(voxel_size) if voxel_size and voxel_size > 0 else auto_voxel_size(mesh)
    mask, affine = voxelize_to_grid(mesh, spacing=vs, margin=2)
    if not mask.any():
        raise ValueError("The surface encloses nothing at that voxel size — is it closed?")
    min_len = float(min_branch_length) if min_branch_length and min_branch_length > 0 else 4 * vs
    with using("cpu"):
        skel = to_numpy(skeletonize_binary(mask)).astype(bool)
        coords = np.argwhere(skel).astype(np.float32)
        polys = branch_polylines_from_skeleton(
            coords,
            min_points=max(3, int(round(min_len / vs))),
            prune_tiny_loops=bool(prune),
            prune_short_spurs=bool(prune),
        )
    step = float(step) if step and step > 0 else vs
    branches: list[VesselBranch] = []
    for poly in polys:
        poly = to_numpy(poly).astype(float)
        if len(poly) < 2:
            continue
        # Skeleton chains come ordered (walked end to end) in voxel indices.
        world = poly @ affine[:3, :3].T + affine[:3, 3]
        # Smoothing in mm²: scaled by the voxel size, or a small vessel's
        # centerline would be pulled off its axis.
        pts = _smooth_resample(world, step=step, smooth=float(smooth) * vs * vs)
        if len(pts) < 2 or _arc(pts)[-1] < min_len:
            continue
        branches.append(VesselBranch(id=len(branches) + 1, points=pts, tangents=_tangents(pts), arc_length=_arc(pts)))
    branches.sort(key=lambda b: -b.length)
    for i, b in enumerate(branches, 1):
        b.id = i
    return branches


# ──────────────────────────────────────────────────────────────────────────────
# Cross-sections
# ──────────────────────────────────────────────────────────────────────────────


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def _feret(points2d: np.ndarray) -> tuple[float, float]:
    """``(min, max)`` caliper diameter of a planar contour."""
    from scipy.spatial import ConvexHull

    if len(points2d) < 3:
        return 0.0, 0.0
    try:
        hull = points2d[ConvexHull(points2d).vertices]
    except Exception:  # noqa: BLE001 — collinear
        hull = points2d
    angles = np.linspace(0, np.pi, 90, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    proj = hull @ dirs.T
    widths = proj.max(axis=0) - proj.min(axis=0)
    diff = hull[:, None, :] - hull[None, :, :]
    return float(widths.min()), float(np.sqrt((diff ** 2).sum(-1)).max())


def section_of_loop(loop: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> dict[str, Any]:
    """Area, perimeter and diameters of one closed contour in a plane."""
    u, v = _plane_basis(normal)
    ring = loop[:-1] if len(loop) > 2 and np.allclose(loop[0], loop[-1]) else loop
    p2 = np.stack([(ring - origin) @ u, (ring - origin) @ v], axis=1)
    x, y = p2[:, 0], p2[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = float(np.linalg.norm(np.diff(np.vstack([p2, p2[:1]]), axis=0), axis=1).sum())
    dmin, dmax = _feret(p2)
    return {
        "area": float(area),
        "perimeter": perim,
        "diameter": float(2.0 * np.sqrt(area / np.pi)),
        "min_diameter": dmin,
        "max_diameter": dmax,
        "circularity": float(4.0 * np.pi * area / perim ** 2) if perim > 0 else 0.0,
        "centroid": ring.mean(axis=0),
        "contour": loop,
    }


def _loop_at(poly: Any, origin: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """The contour of a plane cut that surrounds *origin* (else the nearest one)."""
    cut = poly.slice(normal=tuple(float(v) for v in normal), origin=tuple(float(v) for v in origin))
    if cut.n_points == 0:
        return None
    stripped = cut.strip(join=True, max_length=100000)
    lines = np.asarray(stripped.lines)
    pts = np.asarray(stripped.points)
    loops = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        loops.append(pts[lines[i + 1: i + 1 + n]])
        i += n + 1
    if not loops:
        return None
    u, v = _plane_basis(normal)
    o2 = np.zeros(2)
    best, best_d = None, np.inf
    for loop in loops:
        if len(loop) < 3:
            continue
        p2 = np.stack([(loop - origin) @ u, (loop - origin) @ v], axis=1)
        if _point_in_polygon(o2, p2):
            return loop
        d = float(np.linalg.norm(p2.mean(axis=0)))
        if d < best_d:
            best, best_d = loop, d
    return best


def _distance_to_surface(points: np.ndarray, mesh: Mesh) -> np.ndarray:
    from nvitk.meshlab.measure import distance_to_surface

    return distance_to_surface(points, mesh)


def branch_profiles(
    mesh: Mesh,
    branches: Sequence[VesselBranch],
    *,
    every: float = 0.0,
) -> list[VesselBranch]:
    """Cut the surface perpendicular to each branch at stations every *every* mm.

    Fills each branch's :attr:`VesselBranch.profile` with per-station ``arc``,
    ``area``, ``perimeter``, ``diameter`` (area-equivalent), ``min_diameter``,
    ``max_diameter``, ``circularity`` and ``radius`` (inscribed: the distance from
    the centerline point to the wall). *every* = 0 uses every centerline point.
    """
    from nvitk.meshlab.convert import to_pyvista

    poly = to_pyvista(mesh)
    for b in branches:
        idx = np.arange(len(b.points))
        if every and every > 0 and len(b.points) > 1:
            targets = np.arange(0.0, b.length + 1e-9, float(every))
            idx = np.unique(np.searchsorted(b.arc_length, targets).clip(0, len(b.points) - 1))
        keys = ("arc", "area", "perimeter", "diameter", "min_diameter", "max_diameter", "circularity")
        prof: dict[str, list[float]] = {k: [] for k in keys}
        for i in idx:
            loop = _loop_at(poly, b.points[i], b.tangents[i])
            sec = section_of_loop(loop, b.points[i], b.tangents[i]) if loop is not None and len(loop) >= 3 else None
            prof["arc"].append(float(b.arc_length[i]))
            for k in keys[1:]:
                prof[k].append(float(sec[k]) if sec else np.nan)
        out = {k: np.asarray(v) for k, v in prof.items()}
        out["radius"] = _distance_to_surface(b.points[idx], mesh)
        out["index"] = idx
        b.profile = out
    return list(branches)


def _curvature(points: np.ndarray) -> np.ndarray:
    """Curvature (1/mm) along a uniformly resampled polyline."""
    if len(points) < 3:
        return np.zeros(len(points))
    s = _arc(points)
    d1 = np.gradient(points, s, axis=0)
    d2 = np.gradient(d1, s, axis=0)
    num = np.linalg.norm(np.cross(d1, d2), axis=1)
    den = np.linalg.norm(d1, axis=1) ** 3
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def branch_metrics(branch: VesselBranch) -> dict[str, float]:
    """Length, tortuosity, curvature and diameter statistics of one branch.

    Diameter statistics leave out the stations within one diameter of either end
    (sections there also cut the neighbouring branch or the cap). Stenosis is
    ``1 − min / reference`` of the area-equivalent diameter, with the branch's
    median diameter as reference.
    """
    pts = branch.points
    chord = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) > 1 else 0.0
    curv = _curvature(pts)
    out = {
        "length": branch.length,
        "chord": chord,
        "tortuosity": branch.length / chord if chord > 0 else float("nan"),
        "mean_curvature": float(np.mean(curv)) if len(curv) else 0.0,
        "max_curvature": float(np.max(curv)) if len(curv) else 0.0,
    }
    prof = branch.profile
    if prof:
        d = prof["diameter"]
        ok = np.isfinite(d)
        if ok.any():
            ref = float(np.median(d[ok]))
            # One diameter from each end the plane also cuts the neighbouring
            # branches (or the cap): leave those stations out of the statistics.
            arc = prof["arc"]
            core = ok & (arc >= ref) & (arc <= branch.length - ref)
            if core.sum() >= 2:
                ok = core
            k = int(np.nanargmin(np.where(ok, d, np.nan)))
            out.update({
                "mean_diameter": float(np.mean(d[ok])),
                "min_diameter": float(d[k]),
                "max_diameter": float(np.max(d[ok])),
                "min_diameter_at": float(prof["arc"][k]),
                "min_area": float(prof["area"][k]),
                "stenosis_percent": float(100.0 * (1.0 - d[k] / ref)) if ref > 0 else float("nan"),
                "mean_radius_inscribed": float(np.nanmean(prof["radius"])),
            })
    return out


def bifurcations(branches: Sequence[VesselBranch], *, tolerance: float = 0.0) -> list[dict[str, Any]]:
    """Where branch ends meet: the junction point, the branches, and their angles.

    Ends closer than *tolerance* mm (0 = 3× the median station spacing) form one
    junction. Each branch's direction is taken a few mm away from the junction;
    the angle of each pair is reported in degrees.
    """
    ends = []
    for b in branches:
        if len(b.points) < 2:
            continue
        ends.append((b, 0, b.points[0]))
        ends.append((b, -1, b.points[-1]))
    if not ends:
        return []
    steps = [float(np.median(np.diff(b.arc_length))) for b in branches if len(b.arc_length) > 1]
    tol = float(tolerance) if tolerance and tolerance > 0 else 3.0 * (np.median(steps) if steps else 1.0)
    groups: list[list[int]] = []
    used = set()
    for i in range(len(ends)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(ends)):
            if j not in used and np.linalg.norm(ends[i][2] - ends[j][2]) <= max(tol, 1e-6):
                group.append(j)
        if len({id(ends[g][0]) for g in group}) >= 2:
            used.update(group)
            groups.append(group)
    out = []
    for group in groups:
        centre = np.mean([ends[g][2] for g in group], axis=0)
        dirs = []
        for g in group:
            b, end, _ = ends[g]
            pts = b.points if end == 0 else b.points[::-1]
            s = _arc(pts)
            k = int(np.searchsorted(s, min(3.0 * tol, s[-1] * 0.5)))
            d = pts[max(k, 1)] - pts[0]
            dirs.append((b.id, d / (np.linalg.norm(d) or 1.0)))
        angles = {}
        for a in range(len(dirs)):
            for c in range(a + 1, len(dirs)):
                ang = float(np.degrees(np.arccos(np.clip(np.dot(dirs[a][1], dirs[c][1]), -1, 1))))
                angles[f"{dirs[a][0]}-{dirs[c][0]}"] = ang
        out.append({"point": centre, "branches": [d[0] for d in dirs], "angles": angles})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Local sections and maps
# ──────────────────────────────────────────────────────────────────────────────


def thickness_map(mesh: Mesh, *, vertices: np.ndarray | None = None, max_distance: float = 0.0) -> np.ndarray:
    """Local diameter at each vertex: distance along the inward normal to the opposite wall.

    The shape-diameter function with one ray per vertex (VTK OBB tree). For a
    tube it is the lumen diameter; ``nan`` where the ray leaves without hitting
    (an open end). *vertices* restricts it to a subset.
    """
    import vtk

    from nvitk.meshlab.convert import to_pyvista

    idx = np.arange(mesh.n_vertices) if vertices is None else np.asarray(vertices, dtype=np.int64)
    if not len(idx):
        return np.zeros(0)
    lo, hi = mesh.bounds
    reach = float(max_distance) if max_distance and max_distance > 0 else float(np.linalg.norm(hi - lo))
    tree = vtk.vtkOBBTree()
    tree.SetDataSet(to_pyvista(mesh))
    tree.BuildLocator()
    normals = mesh.vertex_normals[idx]
    origins = mesh.vertices[idx]
    eps = 1e-4 * max(reach, 1.0)
    hits = vtk.vtkPoints()
    out = np.full(len(idx), np.nan)
    for i, (p, n) in enumerate(zip(origins, normals)):
        start = p - n * eps
        end = p - n * reach
        hits.Reset()
        if tree.IntersectWithLine(start, end, hits, None) and hits.GetNumberOfPoints():
            pts = np.asarray([hits.GetPoint(j) for j in range(hits.GetNumberOfPoints())])
            out[i] = float(np.linalg.norm(pts - p, axis=1).min())
    return out


def radius_map(mesh: Mesh, branches: Sequence[VesselBranch], *, key: str = "diameter") -> np.ndarray:
    """Per-vertex value of the nearest centerline station (its *key*, e.g. diameter).

    With ``key="radius"`` (or no profiles) it is the distance from each vertex to
    the nearest centerline point — the local radius.
    """
    from scipy.spatial import cKDTree

    pts, vals = [], []
    for b in branches:
        prof = b.profile
        if prof and key in prof and key != "radius":
            pts.append(b.points[prof["index"]])
            vals.append(prof[key])
        else:
            pts.append(b.points)
            vals.append(np.full(len(b.points), np.nan))
    if not pts:
        return np.full(mesh.n_vertices, np.nan)
    allp = np.vstack(pts)
    allv = np.concatenate(vals)
    d, i = cKDTree(allp).query(mesh.vertices)
    return d if key == "radius" or np.all(np.isnan(allv)) else allv[i]


def branch_map(mesh: Mesh, branches: Sequence[VesselBranch]) -> tuple[np.ndarray, np.ndarray]:
    """``(branch id, distance)`` per vertex: the centerline branch each vertex is
    nearest to, and how far (the local radius)."""
    from scipy.spatial import cKDTree

    pts = [np.asarray(b.points, dtype=float) for b in branches if len(b.points)]
    ids = [np.full(len(b.points), b.id) for b in branches if len(b.points)]
    if not pts:
        return np.full(mesh.n_vertices, -1), np.full(mesh.n_vertices, np.nan)
    d, i = cKDTree(np.vstack(pts)).query(mesh.vertices)
    return np.concatenate(ids)[i], d


def vessel_section_at(
    mesh: Mesh,
    point: Sequence[float],
    *,
    radius: float = 0.0,
    iterations: int = 3,
) -> dict[str, Any]:
    """The cross-section through *point* perpendicular to the local vessel axis.

    No centerline needed: the axis is the main direction of the wall vertices
    near the point (a tube's longest extent), refined a few times by re-centring
    on the section's centroid. *point* may be on the wall (a click on the
    surface) or inside the lumen; *radius* (mm, 0 = from the local diameter) is
    the neighbourhood used for the axis.
    """
    from scipy.spatial import cKDTree

    from nvitk.meshlab.convert import to_pyvista

    p = np.asarray(point, dtype=float)
    tree = cKDTree(mesh.vertices)
    d0, i0 = tree.query(p)
    centre = p.copy()
    if radius and radius > 0:
        reach = float(radius)
    else:
        # The local diameter: from the nearest wall vertex across to the other side.
        diam = thickness_map(mesh, vertices=np.array([i0]))[0]
        if not np.isfinite(diam) or diam <= 0:
            diam = 2.0 * max(d0, 1e-3)
        if d0 < 0.25 * diam:
            # A click on the wall: start half a diameter inside, on the axis.
            centre = mesh.vertices[i0] - mesh.vertex_normals[i0] * diam / 2.0
        reach = 1.5 * diam
    poly = to_pyvista(mesh)
    axis = None
    sec = None
    for _ in range(max(1, int(iterations))):
        near = mesh.vertices[tree.query_ball_point(centre, reach)]
        if len(near) < 6:
            reach *= 1.5
            continue
        _, _, vt = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
        axis = vt[0]
        loop = _loop_at(poly, centre, axis)
        if loop is None or len(loop) < 3:
            break
        sec = section_of_loop(loop, centre, axis)
        centre = sec["centroid"]
        reach = max(1.5 * sec["diameter"], 1e-6)
    if sec is None or axis is None:
        raise ValueError("No closed cross-section around that point.")
    sec["axis"] = axis
    sec["centre"] = centre
    return sec


__all__ = [
    "VesselBranch",
    "auto_voxel_size",
    "bifurcations",
    "branch_map",
    "branch_metrics",
    "branch_profiles",
    "mesh_centerlines",
    "radius_map",
    "section_of_loop",
    "thickness_map",
    "vessel_section_at",
]
