# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/morphology/centering.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Re-centring of voxel-skeleton paths on the vessel lumen (the "native" centerline backend).

A voxel skeleton is medial only to within a voxel and walks the grid as a staircase; on TOF masks
(0.5 mm) that over-estimates length by ~5 %, multiplies curvature several-fold, and puts the path
off-centre so that the distance to the wall under-estimates the inscribed radius by 12-16 %
(PESA17808400 against VMTK). :func:`center_polyline` fixes the geometry by moving each point to
the centroid of the lumen cross-section cut by the plane normal to the path, alternated with
arc-length smoothing, then climbs in-plane to the maximum inscribed-sphere radius;
:func:`inscribed_radius` measures that radius as the distance to the nearest surface vertex, which
is how VMTK's ``MaximumInscribedSphereRadius`` is defined. On PESA17808400 (ICA, ACA, PCA, PComm,
SCA) the result is within 0.7-3.0 % of VMTK's MIS and -1 to +3 % of its length.

VTK + NumPy only (vendored into the Slicer module).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes for a plane with *normal*."""
    n = normal / (np.linalg.norm(normal) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Even-odd test of ``(x, y)`` against a closed 2-D polygon."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _polygon_centroid(p2: np.ndarray) -> tuple[np.ndarray, float]:
    """Area centroid and area of a closed 2-D polygon (vertex mean when degenerate)."""
    x, y = p2[:, 0], p2[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x * ys - xs * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-12:
        return p2.mean(axis=0), 0.0
    cx = float(((x + xs) * cross).sum()) / (6.0 * area)
    cy = float(((y + ys) * cross).sum()) / (6.0 * area)
    return np.array([cx, cy]), abs(area)


class _SectionCutter:
    """Plane cuts of one surface, reusing the VTK pipeline across calls."""

    def __init__(self, surface: Any) -> None:
        import vtk

        self.plane = vtk.vtkPlane()
        self.cutter = vtk.vtkCutter()
        self.cutter.SetInputData(surface)
        self.cutter.SetCutFunction(self.plane)
        self.stripper = vtk.vtkStripper()
        self.stripper.SetInputConnection(self.cutter.GetOutputPort())
        self.stripper.JoinContiguousSegmentsOn()

    def lumen_centroid(self, origin: np.ndarray, normal: np.ndarray, max_shift: float) -> np.ndarray | None:
        """Centroid of the contour surrounding *origin* (else the nearest one within *max_shift*)."""
        from vtk.util import numpy_support

        self.plane.SetOrigin(*map(float, origin))
        self.plane.SetNormal(*map(float, normal))
        self.stripper.Update()
        out = self.stripper.GetOutput()
        if out.GetNumberOfPoints() < 3:
            return None
        pts = numpy_support.vtk_to_numpy(out.GetPoints().GetData()).astype(float)
        u, v = _plane_basis(np.asarray(normal, dtype=float))
        best, best_d = None, np.inf
        lines = out.GetLines()
        lines.InitTraversal()
        import vtk

        ids = vtk.vtkIdList()
        while lines.GetNextCell(ids):
            idx = [ids.GetId(j) for j in range(ids.GetNumberOfIds())]
            if len(idx) < 4:
                continue
            ring = pts[idx]
            p2 = np.stack([(ring - origin) @ u, (ring - origin) @ v], axis=1)
            c2, area = _polygon_centroid(p2)
            if area <= 0:
                continue
            centroid = origin + c2[0] * u + c2[1] * v
            if _point_in_polygon(0.0, 0.0, p2):
                return centroid
            d = float(np.linalg.norm(c2))
            if d < best_d:
                best, best_d = centroid, d
        return best if best is not None and best_d <= max_shift else None


def _tangents(points: np.ndarray, half_window: int) -> np.ndarray:
    """Unit tangents from central differences over ±*half_window* points."""
    n = len(points)
    lo = np.clip(np.arange(n) - half_window, 0, n - 1)
    hi = np.clip(np.arange(n) + half_window, 0, n - 1)
    t = points[hi] - points[lo]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return t / norm


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at a fixed arc-length *step* (ends kept)."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= step:
        return points.copy()
    ss = np.append(np.arange(0.0, s[-1], step), s[-1])
    return np.column_stack([np.interp(ss, s, points[:, k]) for k in range(3)])


def center_polyline(
    points: np.ndarray,
    surface: Any,
    *,
    iterations: int = 5,
    smooth_mm: float = 0.3,
    sample_step_mm: float = 0.25,
    tangent_window_mm: float = 1.5,
    max_shift_mm: float | None = None,
    max_move_mm: float | None = None,
    ridge_steps: int = 8,
) -> np.ndarray:
    """Move a path onto the lumen centre line of *surface* (same coordinate frame).

    Each iteration smooths the path along arc length (*smooth_mm*), samples it every
    *sample_step_mm*, and replaces each sample by the centroid of the lumen contour cut by the
    plane normal to the path there (contours not surrounding the point are accepted only within
    *max_shift_mm*, default 2 × step). The two end points are kept, so the path still spans its
    seeds. A point never moves more than *max_move_mm* per iteration (default 4 × step): a
    skeleton is within about a voxel of the lumen centre, and near a bifurcation the plane can cut
    the crotch or the neighbouring branch, whose centroid lies millimetres away. A final in-plane
    ascent (*ridge_steps*) moves the points onto the maximum of the inscribed-sphere radius, where
    VMTK's centerline runs. Returns the path sampled every *sample_step_mm*.
    """
    from nvitk_vendor.measure.morpho.metrics import smooth_polyline

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        return pts.copy()
    step = max(float(sample_step_mm), 1e-3)
    shift = float(max_shift_mm) if max_shift_mm else 2.0 * step
    max_move = float(max_move_mm) if max_move_mm else 4.0 * step
    half = max(1, int(round(float(tangent_window_mm) / step)))
    cutter = _SectionCutter(surface)
    work = _resample(pts, step)
    for _ in range(max(1, int(iterations))):
        work = _resample(smooth_polyline(work, float(smooth_mm)), step) if smooth_mm > 0 else work
        tangents = _tangents(work, half)
        moved = work.copy()
        for i in range(1, len(work) - 1):
            c = cutter.lumen_centroid(work[i], tangents[i], shift)
            if c is not None and float(np.linalg.norm(c - work[i])) <= max_move:
                moved[i] = c
        work = moved
    if smooth_mm > 0:
        work = smooth_polyline(work, float(smooth_mm))
    if ridge_steps > 0:
        work = _ascend_inscribed_radius(work, surface, half, steps=int(ridge_steps), step_mm=0.25 * step)
    return work


def _surface_vertex_tree(surface: Any):
    from scipy.spatial import cKDTree
    from vtk.util import numpy_support

    return cKDTree(numpy_support.vtk_to_numpy(surface.GetPoints().GetData()).astype(float))


def _ascend_inscribed_radius(
    points: np.ndarray, surface: Any, half_window: int, *, steps: int, step_mm: float
) -> np.ndarray:
    """Move interior points, within their normal planes, up the inscribed-sphere radius.

    The section centroid of a bent or oval lumen is not where the largest sphere fits; VMTK's
    centerline runs along that maximum (the Voronoi ridge). Each step moves a point away from its
    nearest surface vertex, projected onto the normal plane, by at most *step_mm*, and keeps the
    move only if the radius grows.
    """
    tree = _surface_vertex_tree(surface)
    pts = points.copy()
    tangents = _tangents(pts, half_window)
    for _ in range(steps):
        d, idx = tree.query(pts[1:-1])
        away = pts[1:-1] - tree.data[idx]
        t = tangents[1:-1]
        away -= (away * t).sum(axis=1, keepdims=True) * t
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        trial = pts[1:-1] + step_mm * away / norm
        d_new, _ = tree.query(trial)
        better = d_new > d
        pts[1:-1][better] = trial[better]
    return pts


def inscribed_radius(surface: Any, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest *surface* vertex: VMTK's maximum-inscribed-sphere
    radius (its Voronoi spheres touch surface vertices), evaluated on a centred path."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    d, _ = _surface_vertex_tree(surface).query(pts)
    return np.asarray(d, dtype=float)


def centered_centerline(
    points: np.ndarray,
    surface: Any,
    *,
    iterations: int = 5,
    smooth_mm: float = 0.3,
    sample_step_mm: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`center_polyline` + :func:`inscribed_radius`: ``(points, radius)``."""
    centred = center_polyline(
        points, surface, iterations=iterations, smooth_mm=smooth_mm, sample_step_mm=sample_step_mm
    )
    return centred, inscribed_radius(surface, centred)


def center_voxel_paths(
    paths_vox: Sequence[np.ndarray],
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    surface: Any | None = None,
    iterations: int = 5,
    smooth_mm: float = 0.3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Centre voxel-index paths of a binary *mask*: ``([paths in voxel coords], [radius mm])``.

    The surface (marching cubes of the mask in ``index × spacing``) is built once when not given.
    """
    sp = np.asarray(spacing, dtype=float)
    if surface is None:
        surface = mask_surface(mask, sp)
    out_pts, out_r = [], []
    for path in paths_vox:
        p = np.asarray(path, dtype=float).reshape(-1, 3)
        if len(p) < 3:
            out_pts.append(p)
            out_r.append(inscribed_radius(surface, p * sp))
            continue
        centred, radius = centered_centerline(
            p * sp, surface, iterations=iterations, smooth_mm=smooth_mm,
            sample_step_mm=0.5 * float(np.min(sp)),
        )
        out_pts.append(centred / sp)
        out_r.append(radius)
    return out_pts, out_r


def mask_surface(mask: np.ndarray, spacing: Sequence[float]) -> Any:
    """Smoothed marching-cubes surface of a binary mask in ``index × spacing`` (origin 0)."""
    import vtk
    from vtk.util import numpy_support

    m = np.asarray(mask).astype(np.uint8)
    img = vtk.vtkImageData()
    img.SetDimensions(*m.shape)
    img.SetSpacing(*[float(s) for s in spacing])
    arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(m.ravel(order="F")), deep=True,
                                     array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(arr)
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img)
    mc.SetValue(0, 0.5)
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(mc.GetOutputPort())
    smooth.SetNumberOfIterations(15)
    smooth.SetPassBand(0.1)
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(smooth.GetOutput())
    return out


__all__ = [
    "center_polyline",
    "center_voxel_paths",
    "centered_centerline",
    "inscribed_radius",
    "mask_surface",
]
