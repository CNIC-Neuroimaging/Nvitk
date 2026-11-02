# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/vmtk/surface.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""VMTK surface tools (non-interactive ``vmtksurface*`` scripts) on ``vtkPolyData``."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._core import array_from_image, image_from_array, point_array, run_script

SMOOTHING_METHODS: tuple[str, ...] = ("taubin", "laplace")
SUBDIVISION_METHODS: tuple[str, ...] = ("linear", "butterfly", "loop")
CAPPING_METHODS: tuple[str, ...] = ("simple", "centerpoint", "smooth", "annular", "concaveannular")
CURVATURE_TYPES: tuple[str, ...] = ("mean", "gaussian", "maximum", "minimum")


def smooth(
    surface: Any,
    *,
    method: str = "taubin",
    iterations: int = 30,
    passband: float = 0.1,
    relaxation: float = 0.01,
    boundary_smoothing: bool = True,
) -> Any:
    """``vmtksurfacesmoothing``: Taubin (windowed-sinc, volume preserving; *passband*) or Laplacian
    (*relaxation*) smoothing."""
    if method not in SMOOTHING_METHODS:
        raise ValueError(f"method must be one of {SMOOTHING_METHODS}")
    return run_script(
        "vmtkSurfaceSmoothing",
        Surface=surface,
        Method=method,
        NumberOfIterations=int(iterations),
        PassBand=float(passband),
        RelaxationFactor=float(relaxation),
        BoundarySmoothing=1 if boundary_smoothing else 0,
    ).Surface


def remesh(
    surface: Any,
    *,
    target_edge_length: float | None = None,
    target_area: float | None = None,
    iterations: int = 10,
    preserve_boundary: bool = False,
) -> Any:
    """``vmtksurfaceremeshing``: isotropic triangle remeshing to a target edge length (mm) or area
    (mm²); with neither, the mean edge length of the input is kept."""
    members: dict[str, Any] = {
        "Surface": surface,
        "NumberOfIterations": int(iterations),
        "PreserveBoundaryEdges": 1 if preserve_boundary else 0,
    }
    if target_area:
        members.update(ElementSizeMode="area", TargetArea=float(target_area))
    else:
        if not target_edge_length:
            target_edge_length = mean_edge_length(surface)
        members.update(ElementSizeMode="edgelength", TargetEdgeLength=float(target_edge_length))
    members["CleanOutput"] = 1
    # The remesher can leave unreferenced points behind; drop them.
    return _clean(run_script("vmtkSurfaceRemeshing", **members).Surface)


def _clean(surface: Any) -> Any:
    """Merge duplicate and drop unused points (``vtkCleanPolyData``).

    Not chained after a ``vtkTriangleFilter``: on VMTK remesher output that pipeline segfaults in
    VTK 9.6 (each filter alone is fine), and the remesher already emits triangles.
    """
    import vtk

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surface)
    clean.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(clean.GetOutput())
    return out


def mean_edge_length(surface: Any) -> float:
    """Mean triangle edge length of *surface*."""
    import vtk

    edges = vtk.vtkExtractEdges()
    edges.SetInputData(surface)
    edges.Update()
    out = edges.GetOutput()
    total, n = 0.0, out.GetNumberOfCells()
    for c in range(n):
        ids = out.GetCell(c).GetPointIds()
        p0 = np.asarray(out.GetPoint(ids.GetId(0)))
        p1 = np.asarray(out.GetPoint(ids.GetId(1)))
        total += float(np.linalg.norm(p1 - p0))
    return total / max(n, 1)


def decimate(surface: Any, *, target_reduction: float = 0.5, boundary_vertex_deletion: bool = False) -> Any:
    """``vmtksurfacedecimation``: remove *target_reduction* (0-1) of the triangles."""
    return run_script(
        "vmtkSurfaceDecimation",
        Surface=surface,
        TargetReduction=float(target_reduction),
        BoundaryVertexDeletion=1 if boundary_vertex_deletion else 0,
    ).Surface


def subdivide(surface: Any, *, subdivisions: int = 1, method: str = "butterfly") -> Any:
    """``vmtksurfacesubdivision`` (linear / butterfly / loop)."""
    if method not in SUBDIVISION_METHODS:
        raise ValueError(f"method must be one of {SUBDIVISION_METHODS}")
    return run_script(
        "vmtkSurfaceSubdivision", Surface=_clean(surface), NumberOfSubdivisions=int(subdivisions), Method=method
    ).Surface


def cap(surface: Any, *, method: str = "centerpoint", rings: int = 8) -> tuple[Any, np.ndarray]:
    """``vmtksurfacecapper``: close every open boundary. Returns the capped surface and its
    per-cell ``CellEntityIds`` (1 = wall, >=2 = one id per cap)."""
    if method not in CAPPING_METHODS:
        raise ValueError(f"method must be one of {CAPPING_METHODS}")
    from ._core import cell_array

    out = run_script(
        "vmtkSurfaceCapper",
        Surface=surface,
        Method=method,
        NumberOfRings=int(rings),
        Interactive=0,
        CellEntityIdsArrayName="CellEntityIds",
    ).Surface
    return out, cell_array(out, "CellEntityIds", default=1)


def remove_kites(surface: Any, *, size_factor: float = 0.1) -> Any:
    """``vmtksurfacekiteremoval``: collapse the small "kite" triangle pairs marching cubes leaves."""
    return run_script("vmtkSurfaceKiteRemoval", Surface=surface, SizeFactor=float(size_factor)).Surface


def connectivity(surface: Any, *, closest_point: Sequence[float] | None = None) -> Any:
    """``vmtksurfaceconnectivity``: the largest connected piece, or the one closest to a point."""
    members: dict[str, Any] = {"Surface": surface, "CleanOutput": 1}
    if closest_point is not None:
        members.update(Method="closest", ClosestPoint=[float(v) for v in closest_point])
    else:
        members.update(Method="largest")
    return run_script("vmtkSurfaceConnectivity", **members).Surface


def curvature(surface: Any, *, kind: str = "mean", absolute: bool = False, median_filter: bool = False) -> np.ndarray:
    """``vmtksurfacecurvature``: per-vertex mean / Gaussian / max / min curvature (1/mm).

    VTK's discrete estimator (``vtkCurvatures``); *median_filter* smooths it over each vertex's
    one-ring, which helps on marching-cubes surfaces.
    """
    if kind not in CURVATURE_TYPES:
        raise ValueError(f"kind must be one of {CURVATURE_TYPES}")
    out = run_script(
        "vmtkSurfaceCurvature",
        Surface=surface,
        CurvatureType=kind,
        AbsoluteCurvature=1 if absolute else 0,
        MedianFiltering=1 if median_filter else 0,
    ).Surface
    return point_array(out, "Curvature")


def normals(surface: Any, *, auto_orient: bool = True) -> Any:
    """``vmtksurfacenormals``: consistent (optionally outward auto-oriented) point normals ``Normals``."""
    return run_script(
        "vmtkSurfaceNormals", Surface=surface, AutoOrientNormals=1 if auto_orient else 0, NormalsArrayName="Normals"
    ).Surface


def flow_extensions(
    surface: Any,
    centerlines: Any,
    *,
    extension_ratio: float = 10.0,
    adaptive_length: bool = True,
    mode: str = "centerlinedirection",
) -> Any:
    """``vmtkflowextensions``: extrude every open boundary (inlet/outlets) along the centerline
    direction by *extension_ratio* × the boundary radius (CFD-ready inflow/outflow tubes)."""
    return run_script(
        "vmtkFlowExtensions",
        Surface=surface,
        Centerlines=centerlines,
        ExtensionMode=mode,
        AdaptiveExtensionLength=1 if adaptive_length else 0,
        ExtensionRatio=float(extension_ratio),
        Interactive=0,
    ).Surface


def mass_properties(surface: Any) -> dict[str, Any]:
    """``vmtksurfacemassproperties``: area (mm²), enclosed volume (mm³), shape index, centre of mass."""
    s = run_script("vmtkSurfaceMassProperties", Surface=surface)
    return {
        "surface_area_mm2": float(s.SurfaceArea),
        "volume_mm3": float(s.Volume),
        "shape_index": float(s.ShapeIndex),
        "center_of_mass": np.asarray(s.CenterOfMass, dtype=float),
    }


def marching_cubes(
    volume: np.ndarray,
    *,
    level: float = 0.5,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    largest_only: bool = False,
) -> Any:
    """``vmtkmarchingcubes`` on an ``(X, Y, Z)`` array (``index × spacing + origin`` frame)."""
    img = image_from_array(np.asarray(volume, dtype=np.float32), spacing, origin)
    return run_script(
        "vmtkMarchingCubes", Image=img, Level=float(level), Connectivity=1 if largest_only else 0
    ).Surface


def to_binary_image(surface: Any, *, spacing: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``vmtksurfacetobinaryimage``: voxelise a closed surface. Returns ``(mask, spacing, origin)``
    with the mask indexed ``(X, Y, Z)``."""
    img = run_script(
        "vmtkSurfaceToBinaryImage", Surface=surface, PolyDataToImageDataSpacing=[float(spacing)] * 3,
        InsideValue=1, OutsideValue=0,
    ).Image
    return (
        array_from_image(img).astype(bool),
        np.asarray(img.GetSpacing(), dtype=float),
        np.asarray(img.GetOrigin(), dtype=float),
    )


__all__ = [
    "CAPPING_METHODS",
    "CURVATURE_TYPES",
    "SMOOTHING_METHODS",
    "SUBDIVISION_METHODS",
    "cap",
    "connectivity",
    "curvature",
    "decimate",
    "flow_extensions",
    "marching_cubes",
    "mass_properties",
    "mean_edge_length",
    "normals",
    "remesh",
    "remove_kites",
    "smooth",
    "subdivide",
    "to_binary_image",
]
