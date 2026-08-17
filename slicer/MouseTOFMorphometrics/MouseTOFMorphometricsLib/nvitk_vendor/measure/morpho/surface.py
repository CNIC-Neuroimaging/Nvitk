# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/measure/morpho/surface.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Surface extraction, VTK polydata construction, and VTP writing."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support

from nvitk_vendor.measure.morphometrics_config import (
    REFINE_SURFACE_FOR_VMTK,
    SURFACE_REFINEMENT_SMOOTH_ITERATIONS,
    SURFACE_SUBDIVISION_LEVELS,
)
from .geometry import cumulative_s
from .models import SkeletonTree

def mask_to_surface(mask: np.ndarray, spacing, pre_refined_surface_path: Optional[str] = None):
    """Binary mask → cleaned, smoothed, largest-region triangle surface (VTK marching cubes pipeline).

    Pipeline: marching cubes → clean/triangulate → windowed-sinc smoothing →
    fill holes → keep largest connected region → optional VMTK refinement
    (:func:`refine_surface_for_vmtk`). If *pre_refined_surface_path* is given,
    the pre-refinement surface is also saved there for QC.
    """
    mask_u8 = mask.astype(np.uint8)
    image = vtk.vtkImageData()
    image.SetDimensions(mask_u8.shape)
    image.SetSpacing(*[float(s) for s in spacing])
    image.SetOrigin(0.0, 0.0, 0.0)
    vtk_array = numpy_support.numpy_to_vtk(mask_u8.ravel(order="F"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_array.SetName("Mask")
    image.GetPointData().SetScalars(vtk_array)
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, 0.5)
    mc.Update()
    clean = vtk.vtkCleanPolyData(); clean.SetInputData(mc.GetOutput()); clean.Update()
    tri = vtk.vtkTriangleFilter(); tri.SetInputData(clean.GetOutput()); tri.Update()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(tri.GetOutput())
    smooth.SetNumberOfIterations(15)
    smooth.BoundarySmoothingOff(); smooth.FeatureEdgeSmoothingOff()
    smooth.SetPassBand(0.1)
    smooth.NonManifoldSmoothingOn(); smooth.NormalizeCoordinatesOn()
    smooth.Update()
    fill = vtk.vtkFillHolesFilter(); fill.SetInputData(smooth.GetOutput()); fill.SetHoleSize(1000.0); fill.Update()
    tri2 = vtk.vtkTriangleFilter(); tri2.SetInputData(fill.GetOutput()); tri2.Update()
    clean2 = vtk.vtkCleanPolyData(); clean2.SetInputData(tri2.GetOutput()); clean2.Update()
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(clean2.GetOutput())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    clean3 = vtk.vtkCleanPolyData(); clean3.SetInputData(connectivity.GetOutput()); clean3.Update()
    if pre_refined_surface_path:
        save_vtp(clean3.GetOutput(), pre_refined_surface_path)
    return refine_surface_for_vmtk(clean3.GetOutput())


def refine_surface_for_vmtk(surface):
    """Subdivide + smooth a surface so VMTK centerline extraction has denser, more regular triangles."""
    if not REFINE_SURFACE_FOR_VMTK or int(SURFACE_SUBDIVISION_LEVELS) <= 0:
        return surface

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(surface)
    tri.Update()

    subdiv = vtk.vtkLinearSubdivisionFilter()
    subdiv.SetInputData(tri.GetOutput())
    subdiv.SetNumberOfSubdivisions(int(SURFACE_SUBDIVISION_LEVELS))
    subdiv.Update()

    refined = subdiv.GetOutput()
    if int(SURFACE_REFINEMENT_SMOOTH_ITERATIONS) > 0:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputData(refined)
        smooth.SetNumberOfIterations(int(SURFACE_REFINEMENT_SMOOTH_ITERATIONS))
        smooth.BoundarySmoothingOff()
        smooth.FeatureEdgeSmoothingOff()
        smooth.SetPassBand(0.15)
        smooth.NonManifoldSmoothingOn()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        refined = smooth.GetOutput()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(refined)
    clean.Update()
    out = clean.GetOutput()
    print(
        f"    [surface] Refined for VMTK: "
        f"{surface.GetNumberOfPoints()} pts/{surface.GetNumberOfCells()} cells -> "
        f"{out.GetNumberOfPoints()} pts/{out.GetNumberOfCells()} cells"
    )
    return out


def snap_to_surface(pt_mm: np.ndarray, surface):
    """Nearest surface-mesh point (world mm) to *pt_mm*, via a VTK point locator."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    closest_id = locator.FindClosestPoint([float(v) for v in pt_mm])
    return list(surface.GetPoint(closest_id))


def save_vtp(poly, path: str) -> None:
    """Write a VTK polydata object to a ``.vtp`` file."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(poly)
    writer.Write()


def build_polyline_polydata(points: np.ndarray, arrays: List[Tuple[np.ndarray, str]]):
    """Build a single-polyline VTK polydata from *points*, attaching each ``(array, name)`` as point data."""
    poly = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetNumberOfPoints(len(points))
    for i, xyz in enumerate(points):
        vtk_pts.SetPoint(i, float(xyz[0]), float(xyz[1]), float(xyz[2]))
    poly.SetPoints(vtk_pts)
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(points))
    for i in range(len(points)):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    poly.SetLines(cells)
    for data, name in arrays:
        vtk_arr = numpy_support.numpy_to_vtk(np.asarray(data, dtype=np.float64), deep=True)
        vtk_arr.SetName(name)
        poly.GetPointData().AddArray(vtk_arr)
    return poly


def add_string_point_array(poly, values: List[str], name: str) -> None:
    """Attach a per-point string array *name* to a VTK polydata's point data."""
    arr = vtk.vtkStringArray()
    arr.SetName(name)
    arr.SetNumberOfValues(len(values))
    for i, value in enumerate(values):
        arr.SetValue(i, str(value))
    poly.GetPointData().AddArray(arr)


def build_radius_tube_polydata(points: np.ndarray, radius: np.ndarray, arrays: List[Tuple[np.ndarray, str]]):
    """Build a VTK tube mesh around a centerline, varying tube radius by the per-point *radius* array."""
    poly = build_polyline_polydata(points, arrays)
    radius_arr = numpy_support.numpy_to_vtk(np.asarray(radius, dtype=np.float64), deep=True)
    radius_arr.SetName("EffectiveRadius")
    poly.GetPointData().SetScalars(radius_arr)
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetNumberOfSides(24)
    tube.CappingOn()
    tube.SetRadius(1.0)
    tube.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tube.Update()
    clean = vtk.vtkCleanPolyData(); clean.SetInputData(tube.GetOutput()); clean.Update()
    return clean.GetOutput()


def default_tree_point_metadata(n: int, label: str = "trunk", path: str = "") -> dict:
    """Constant-valued per-point tree metadata (label/path/depth) for *n* centerline points."""
    path = str(path or "")
    return {
        "tree_label": np.array([label] * n, dtype=object),
        "tree_path": np.array([path] * n, dtype=object),
        "tree_depth": np.full(n, len([p for p in path.split(".") if p]), dtype=float),
    }


def add_tree_metadata_point_arrays(poly, metadata: dict) -> None:
    """Attach tree label/path/depth arrays from *metadata* (see :func:`default_tree_point_metadata`) to a polydata."""
    if "tree_depth" in metadata:
        arr = numpy_support.numpy_to_vtk(np.asarray(metadata["tree_depth"], dtype=np.float64), deep=True)
        arr.SetName("TreeDepth")
        poly.GetPointData().AddArray(arr)
    if "tree_label" in metadata:
        add_string_point_array(poly, np.asarray(metadata["tree_label"], dtype=object).tolist(), "TreeLabel")
    if "tree_path" in metadata:
        add_string_point_array(poly, np.asarray(metadata["tree_path"], dtype=object).tolist(), "TreePath")


def extract_point_data_array(poly, name: str, n_points: int) -> np.ndarray:
    """Read a named point-data array from a VTK polydata as a 1-D float array (NaNs if absent).

    Vector arrays (e.g. normals) are reduced to their magnitude.
    """
    arr = poly.GetPointData().GetArray(name)
    if arr is None:
        return np.full(n_points, np.nan)
    values = numpy_support.vtk_to_numpy(arr).astype(float)
    return values if values.ndim == 1 else np.linalg.norm(values, axis=1)


def resample_point_data_by_arclength(old_pts: np.ndarray, new_pts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate a per-point scalar array from *old_pts* onto *new_pts*, matched by arc length."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(old_pts):
        return np.full(len(new_pts), np.nan, dtype=float)
    if len(old_pts) < 2 or len(new_pts) == 0:
        return values.copy() if len(values) == len(new_pts) else np.full(len(new_pts), np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full(len(new_pts), np.nan, dtype=float)
    old_s = cumulative_s(np.asarray(old_pts, dtype=float))
    new_s = cumulative_s(np.asarray(new_pts, dtype=float))
    return np.interp(new_s, old_s[valid], values[valid])


def _polygon_area_3d(pts3d: np.ndarray, normal: np.ndarray) -> float:
    """Area of a planar 3-D polygon: project onto its plane (given by *normal*) and shoelace it."""
    if len(pts3d) < 3:
        return 0.0
    n_hat = normal / (np.linalg.norm(normal) + 1e-15)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, tmp); u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    x = pts3d @ u; y = pts3d @ v
    n = len(x)
    area = sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))
    return abs(area) * 0.5


def compute_cross_section_radius(surface, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Effective vessel radius at each centerline point: cut *surface* by the local normal plane,
    take the largest closed cross-section loop's area, and convert area → equivalent-circle radius.
    """
    n = len(pts)
    areas = np.full(n, np.nan)
    tangents = np.zeros((n, 3))
    for i in range(n):
        a, b = max(0, i - 1), min(n - 1, i + 1)
        t = pts[b] - pts[a]
        tn = np.linalg.norm(t)
        if tn > 1e-10:
            tangents[i] = t / tn
    plane = vtk.vtkPlane()
    cutter = vtk.vtkCutter(); cutter.SetInputData(surface); cutter.SetCutFunction(plane)
    stripper = vtk.vtkStripper(); stripper.SetInputConnection(cutter.GetOutputPort())
    for i in range(n):
        t = tangents[i]
        if np.linalg.norm(t) < 1e-10:
            continue
        plane.SetOrigin(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]))
        plane.SetNormal(float(t[0]), float(t[1]), float(t[2]))
        stripper.Update()
        result = stripper.GetOutput()
        if result.GetNumberOfPoints() < 3:
            continue
        all_pts = numpy_support.vtk_to_numpy(result.GetPoints().GetData())
        best_area = 0.0
        lines = result.GetLines(); lines.InitTraversal()
        id_list = vtk.vtkIdList()
        while lines.GetNextCell(id_list):
            idx = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
            if len(idx) < 3:
                continue
            a_val = _polygon_area_3d(all_pts[idx], t)
            if a_val > best_area:
                best_area = a_val
        if best_area > 0.0:
            areas[i] = best_area
    radii = np.sqrt(np.maximum(areas, 0.0) / np.pi)
    return radii, areas


def build_donut_loop_debug_polydata(loop_rows: List[dict], tree: SkeletonTree, spacing):
    """Build a multi-polyline VTK debug mesh of donut-loop arms, tagged with loop/arm index and endpoint role."""
    all_points = []
    loop_ids = []
    arm_ids = []
    point_roles = []
    cell_offsets = []
    spacing = np.asarray(spacing, dtype=float)
    offset = 0
    for row in loop_rows:
        nodes = row["node_indices"]
        pts_mm = tree.pts_vox[nodes].astype(float) * spacing
        cell_offsets.append((offset, len(pts_mm)))
        all_points.append(pts_mm)
        loop_ids.extend([row["loop_index"]] * len(pts_mm))
        arm_ids.extend([row["arm_index"]] * len(pts_mm))
        roles = np.zeros(len(pts_mm), dtype=float)
        if len(roles):
            roles[0] = 1.0
            roles[-1] = 2.0
        point_roles.extend(roles.tolist())
        offset += len(pts_mm)
    if not all_points:
        return vtk.vtkPolyData()
    points = np.vstack(all_points)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetNumberOfPoints(len(points))
    for i, xyz in enumerate(points):
        vtk_pts.SetPoint(i, float(xyz[0]), float(xyz[1]), float(xyz[2]))
    lines = vtk.vtkCellArray()
    for start, length in cell_offsets:
        lines.InsertNextCell(length)
        for i in range(length):
            lines.InsertCellPoint(start + i)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetLines(lines)
    for data, name in [
        (loop_ids, "DonutLoopIndex"),
        (arm_ids, "DonutArmIndex"),
        (point_roles, "GatewayRole"),
    ]:
        arr = numpy_support.numpy_to_vtk(np.asarray(data, dtype=np.float64), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    return poly
