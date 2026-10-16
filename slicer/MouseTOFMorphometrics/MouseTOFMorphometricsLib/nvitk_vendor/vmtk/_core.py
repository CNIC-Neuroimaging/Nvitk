# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/vmtk/_core.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""VMTK import guard, script runner, and NumPy ↔ VTK conversion helpers.

Depends only on NumPy and VTK (VMTK is imported lazily), so the package stays importable — and
vendorable into the Slicer module — without the rest of nvitk_vendor.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Iterable, Sequence

import numpy as np
import vtk
from vtk.util import numpy_support


class VmtkUnavailableError(ImportError):
    """VMTK is not installed in this environment."""


_VMTK_STATE: dict[str, Any] = {}


def _import_vmtk() -> tuple[Any, Any] | None:
    """``(vmtkscripts, vtkvmtk)`` or ``None`` (cached)."""
    if "mods" not in _VMTK_STATE:
        try:
            from vmtk import vmtkscripts, vtkvmtk  # noqa: PLC0415

            _VMTK_STATE["mods"] = (vmtkscripts, vtkvmtk)
        except Exception as exc:  # noqa: BLE001 — any import failure means "not usable"
            _VMTK_STATE["mods"] = None
            _VMTK_STATE["error"] = exc
    return _VMTK_STATE["mods"]


def vmtk_available() -> bool:
    """True when ``vmtk`` (scripts + compiled ``vtkvmtk``) imports."""
    return _import_vmtk() is not None


def require_vmtk() -> tuple[Any, Any]:
    """``(vmtkscripts, vtkvmtk)``, or :class:`VmtkUnavailableError` with an install hint."""
    mods = _import_vmtk()
    if mods is None:
        raise VmtkUnavailableError(
            "VMTK is not installed. nvitk's environments get it from recipe/post-link.sh "
            "(cp311 wheel built by recipe/vmtk/build_wheel.sh; `pip install vmtk==1.5.2` on "
            f"CPython >= 3.12). Import error: {_VMTK_STATE.get('error')!r}"
        ) from _VMTK_STATE.get("error")
    return mods


def vmtk_version() -> str:
    """Installed VMTK version (``""`` when unavailable)."""
    try:
        from importlib.metadata import version

        return version("vmtk")
    except Exception:  # noqa: BLE001
        return ""


def run_script(name: str, *, quiet: bool = True, **members: Any) -> Any:
    """Instantiate ``vmtkscripts.<name>``, set *members*, ``Execute()`` and return the script.

    VMTK's scripts log progress through ``print``; ``quiet`` swallows it (errors still raise).
    """
    vmtkscripts, _ = require_vmtk()
    script = getattr(vmtkscripts, name)()
    for key, value in members.items():
        if not hasattr(script, key):
            raise AttributeError(f"{name} has no member {key!r}")
        setattr(script, key, value)
    sink = io.StringIO()
    if quiet:
        # Scripts bind sys.stdout at construction (and sub-scripts they create inherit it), so
        # both the script's own stream and the global one are redirected.
        script.LogOn = 0
        script.OutputStream = sink
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        script.Execute()
    return script


# ──────────────────────────────────────────────────────────────────────────────
# NumPy ↔ VTK
# ──────────────────────────────────────────────────────────────────────────────


def points_of(poly: Any) -> np.ndarray:
    """``(N, 3)`` float coordinates of a VTK dataset (empty when it has no points)."""
    if poly is None or poly.GetNumberOfPoints() == 0:
        return np.empty((0, 3), dtype=float)
    return numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)


def point_array(poly: Any, name: str, *, default: float = np.nan) -> np.ndarray:
    """Point-data array *name* as NumPy (filled with *default* when absent)."""
    n = 0 if poly is None else int(poly.GetNumberOfPoints())
    arr = None if poly is None else poly.GetPointData().GetArray(name)
    if arr is None:
        return np.full(n, default, dtype=float)
    return numpy_support.vtk_to_numpy(arr).astype(float)


def cell_array(poly: Any, name: str, *, default: float = np.nan) -> np.ndarray:
    """Cell-data array *name* as NumPy (filled with *default* when absent)."""
    n = 0 if poly is None else int(poly.GetNumberOfCells())
    arr = None if poly is None else poly.GetCellData().GetArray(name)
    if arr is None:
        return np.full(n, default, dtype=float)
    return numpy_support.vtk_to_numpy(arr).astype(float)


def line_point_ids(poly: Any) -> list[np.ndarray]:
    """Point ids of every line cell of *poly*, in order."""
    out: list[np.ndarray] = []
    if poly is None:
        return out
    for c in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(c)
        if cell.GetCellType() not in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
            continue
        ids = cell.GetPointIds()
        out.append(np.asarray([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64))
    return out


def polylines_of(poly: Any, arrays: Sequence[str] = ()) -> list[dict[str, np.ndarray]]:
    """Each line cell as ``{"points": (n, 3), <array>: (n,) …}``."""
    pts = points_of(poly)
    data = {name: point_array(poly, name) for name in arrays}
    out = []
    for ids in line_point_ids(poly):
        item = {"points": pts[ids]}
        for name, values in data.items():
            item[name] = values[ids] if values.ndim == 1 else values[ids]
        out.append(item)
    return out


def polyline_polydata(
    lines: Iterable[np.ndarray],
    arrays: dict[str, Sequence[np.ndarray]] | None = None,
) -> Any:
    """A ``vtkPolyData`` with one poly-line cell per entry of *lines* and per-point *arrays*."""
    lines = [np.asarray(line, dtype=float).reshape(-1, 3) for line in lines]
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    offset = 0
    for line in lines:
        for p in line:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        cell = vtk.vtkPolyLine()
        cell.GetPointIds().SetNumberOfIds(len(line))
        for i in range(len(line)):
            cell.GetPointIds().SetId(i, offset + i)
        cells.InsertNextCell(cell)
        offset += len(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)
    for name, per_line in (arrays or {}).items():
        values = np.concatenate([np.asarray(v, dtype=float) for v in per_line]) if per_line else np.empty(0)
        set_point_array(poly, name, values)
    return poly


def set_point_array(poly: Any, name: str, values: np.ndarray) -> None:
    """Attach (or replace) point-data array *name*."""
    arr = numpy_support.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    arr.SetName(name)
    poly.GetPointData().RemoveArray(name)
    poly.GetPointData().AddArray(arr)


def image_from_array(
    array: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    name: str = "ImageScalars",
) -> Any:
    """``vtkImageData`` from an ``(X, Y, Z)`` array (x fastest, as nibabel arrays are indexed)."""
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"image_from_array expects a 3D array, got {arr.shape}.")
    img = vtk.vtkImageData()
    img.SetDimensions(*[int(s) for s in arr.shape])
    img.SetSpacing(*[float(s) for s in spacing])
    img.SetOrigin(*[float(o) for o in origin])
    flat = np.ascontiguousarray(arr.ravel(order="F"))
    if flat.dtype == bool:
        flat = flat.astype(np.uint8)
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True)
    vtk_arr.SetName(name)
    img.GetPointData().SetScalars(vtk_arr)
    return img


def array_from_image(img: Any) -> np.ndarray:
    """``(X, Y, Z)`` array of a ``vtkImageData``'s active scalars."""
    dims = img.GetDimensions()
    scalars = img.GetPointData().GetScalars()
    flat = numpy_support.vtk_to_numpy(scalars)
    if flat.ndim > 1 and flat.shape[1] == 1:
        flat = flat[:, 0]
    return np.asarray(flat).reshape(dims, order="F")


def closest_surface_point(surface: Any, point: Sequence[float]) -> np.ndarray:
    """Nearest surface vertex to *point* (how VMTK seeds are snapped)."""
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    return np.asarray(surface.GetPoint(locator.FindClosestPoint([float(v) for v in point])), dtype=float)


def surface_distance(surface: Any, points: np.ndarray) -> np.ndarray:
    """Unsigned distance from each point to *surface* (``vtkImplicitPolyDataDistance``).

    Evaluated on a centerline this is the maximum-inscribed-sphere radius VMTK reports: on the
    VMTK centerline of PESA17808400 it matched ``MaximumInscribedSphereRadius`` to within 0.01 mm.
    """
    dist = vtk.vtkImplicitPolyDataDistance()
    dist.SetInput(surface)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.abs(np.asarray([dist.EvaluateFunction(*map(float, p)) for p in pts], dtype=float))


__all__ = [
    "VmtkUnavailableError",
    "array_from_image",
    "cell_array",
    "closest_surface_point",
    "image_from_array",
    "line_point_ids",
    "point_array",
    "points_of",
    "polyline_polydata",
    "polylines_of",
    "require_vmtk",
    "run_script",
    "set_point_array",
    "surface_distance",
    "vmtk_available",
    "vmtk_version",
]
