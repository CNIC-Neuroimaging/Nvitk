# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/vmtk/flow.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""4D-flow particle tracing with VMTK's ``vtkvmtkStaticTemporalStreamTracer``.

The tracer behind ``vmtkparticletracer``: an RK45 stream tracer whose velocity is interpolated in
space within cells *and* linearly in time between the per-phase velocity arrays, optionally
wrapping over the cardiac cycle.

- **pathlines**: particles released at one cardiac phase and advected through the time-varying
  field (periodic over the cycle) for ``max_length`` seconds;
- **streamlines**: the same tracer on one frozen phase (a single-row time table makes the field
  steady), integrated for ``max_length`` millimetres.

Integration is in physical units, positions in mm (``voxel × spacing``), velocity in mm/s, time in
seconds, and the traces come back in voxel coordinates, clipped to the vessel mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ._core import require_vmtk

IntegrationDirection = Literal["forward", "backward", "both"]
SeedMode = Literal["volume", "planar"]
TraceMode = Literal["streamlines", "pathlines"]
ColorMetric = Literal["speed", "integration_time", "arc_length", "fixed"]
PlaneSide = Literal["min", "max"]

TRACE_MODES: tuple[str, ...] = ("streamlines", "pathlines")
INTEGRATION_DIRECTIONS: tuple[str, ...] = ("forward", "backward", "both")
SEED_MODES: tuple[str, ...] = ("planar", "volume")
COLOR_METRICS: tuple[str, ...] = ("speed", "integration_time", "arc_length", "fixed")


@dataclass(frozen=True)
class FlowTraceParams:
    """Mask-seeded particle tracing settings."""

    n_seeds: int = 64
    #: Streamlines: maximum path length (mm). Pathlines: integration horizon (seconds).
    max_length: float = 35.0
    stream_seed: int | None = 42
    label_ids: tuple[int, ...] | None = None
    trace_mode: TraceMode = "streamlines"
    integration_direction: IntegrationDirection = "forward"
    seed_mode: SeedMode = "planar"
    seed_plane_axis: int = 2
    seed_plane_side: PlaneSide = "min"
    #: Time between cardiac phases (seconds).
    dt_seconds: float = 1.0
    #: Seeds slower than this (mm/s) at release are dropped (as ``vmtkparticletracer``'s MinSpeed).
    min_speed: float = 1.0
    resample_paths: bool = False
    resample_spacing_vox: float = 0.5


@dataclass
class FlowTrace:
    """One traced particle path: voxel coordinates with per-point time (s) and speed (mm/s)."""

    points: np.ndarray
    time: np.ndarray
    speed: np.ndarray

    @property
    def n_points(self) -> int:
        return int(len(self.points))


# Seeds / ROI


def mask_roi(mask: np.ndarray, label_ids: Sequence[int] | None) -> np.ndarray:
    """Boolean ROI: the union of *label_ids*, or every non-zero voxel."""
    m = np.asarray(mask)
    if label_ids:
        return np.isin(m, [int(v) for v in label_ids])
    return m > 0


def stream_seed_cloud(
    mask: np.ndarray,
    n_seeds: int,
    seed: int | None,
    *,
    label_ids: Sequence[int] | None = None,
    seed_mode: SeedMode = "volume",
    seed_plane_axis: int = 2,
    seed_plane_side: PlaneSide = "min",
) -> np.ndarray | None:
    """Random subsample of ROI voxel indices ``(N, 3)``; ``planar`` keeps the min/max slab along an axis."""
    if int(n_seeds) <= 0:
        return None
    coords = np.argwhere(mask_roi(mask, label_ids))
    if coords.shape[0] == 0:
        return None
    if seed_mode == "planar":
        coords = _planar_seed_candidates(coords, axis=int(seed_plane_axis), side=str(seed_plane_side))
    n_eff = min(int(n_seeds), int(coords.shape[0]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(coords.shape[0], size=n_eff, replace=False)
    return coords[pick].astype(np.float64)


def _planar_seed_candidates(coords: np.ndarray, *, axis: int, side: str) -> np.ndarray:
    """ROI voxels on the min/max face along *axis* (slab widened up to 3 voxels when sparse)."""
    ax = int(axis) % 3
    plane_val = int(np.max(coords[:, ax])) if side == "max" else int(np.min(coords[:, ax]))
    for tol in (0, 1, 2, 3):
        slab = coords[np.abs(coords[:, ax] - plane_val) <= tol]
        if slab.shape[0] >= 4:
            return slab
    return coords


# Tracing


def _velocity_image(
    frames: list[np.ndarray],
    frame_ids: list[int],
    roi: np.ndarray,
    spacing: np.ndarray,
    margin: int = 2,
):
    """Velocity image cropped to the ROI box: one 3-component point array ``velocity_<id>`` per
    phase, zero outside the (1-voxel dilated) ROI. Returns ``(vtkImageData, offset_voxels)``."""
    import vtk
    from scipy import ndimage as ndi
    from vtk.util import numpy_support

    lo = np.maximum(np.argwhere(roi).min(axis=0) - margin, 0)
    hi = np.minimum(np.argwhere(roi).max(axis=0) + margin + 1, roi.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    keep = ndi.binary_dilation(roi[box], iterations=1)
    img = vtk.vtkImageData()
    img.SetDimensions(*[int(b - a) for a, b in zip(lo, hi)])
    img.SetSpacing(*[float(s) for s in spacing])
    img.SetOrigin(*[float(v) for v in lo * spacing])
    for fid, frame in zip(frame_ids, frames):
        v = np.asarray(frame[box], dtype=np.float64) * keep[..., None]
        flat = np.ascontiguousarray(v.reshape(-1, 3, order="F"))
        arr = numpy_support.numpy_to_vtk(flat, deep=True)
        arr.SetName(f"velocity_{int(fid)}")
        img.GetPointData().AddArray(arr)
    return img, lo


def _longest_inside_run(points_vox: np.ndarray, roi: np.ndarray) -> slice | None:
    """Index slice of the longest run of consecutive points whose nearest voxel is in *roi*."""
    idx = np.clip(np.round(points_vox).astype(np.int64), 0, np.asarray(roi.shape) - 1)
    inside = roi[idx[:, 0], idx[:, 1], idx[:, 2]]
    best, best_len, start = None, 0, None
    for i, ok in enumerate(np.append(inside, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best, best_len = slice(start, i), i - start
            start = None
    return best if best_len >= 2 else None


def trace_flow(
    velocity: np.ndarray,
    mask: np.ndarray,
    params: FlowTraceParams,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    time_start: int = 0,
) -> list[FlowTrace]:
    """Trace particles through a velocity field with VMTK.

    *velocity* is ``(X, Y, Z, 3)`` (one phase) or ``(X, Y, Z, T, 3)`` in mm/s, components along the
    array axes; *mask* is the ``(X, Y, Z)`` vessel labels. Streamlines use phase *time_start*;
    pathlines are released at that phase and integrated periodically over the cycle.
    """
    import vtk
    from vtk.util import numpy_support

    _vmtkscripts, vtkvmtk = require_vmtk()
    vel = np.asarray(velocity, dtype=np.float32)
    if vel.ndim == 4:
        vel = vel[:, :, :, None, :]
    if vel.ndim != 5 or vel.shape[-1] != 3:
        raise ValueError("velocity must be (X, Y, Z, 3) or (X, Y, Z, T, 3).")
    n_time = int(vel.shape[3])
    t0 = int(np.clip(time_start, 0, n_time - 1))
    spacing = np.asarray(spacing, dtype=float)
    roi = mask_roi(mask, params.label_ids)
    seeds = stream_seed_cloud(
        mask, params.n_seeds, params.stream_seed, label_ids=params.label_ids,
        seed_mode=params.seed_mode, seed_plane_axis=params.seed_plane_axis,
        seed_plane_side=params.seed_plane_side,
    )
    if seeds is None:
        return []
    if float(params.min_speed) > 0:
        s_idx = np.round(seeds).astype(np.int64)
        speed0 = np.linalg.norm(vel[s_idx[:, 0], s_idx[:, 1], s_idx[:, 2], t0, :], axis=1)
        seeds = seeds[speed0 >= float(params.min_speed)]
        if len(seeds) == 0:
            return []

    pathlines = params.trace_mode == "pathlines" and n_time >= 2
    dt = max(float(params.dt_seconds), 1e-6)
    table = vtk.vtkTable()
    index_col = vtk.vtkIntArray()
    index_col.SetName("index")
    time_col = vtk.vtkDoubleArray()
    time_col.SetName("time")
    table.AddColumn(index_col)
    table.AddColumn(time_col)
    if pathlines:
        frame_ids = list(range(n_time))
        rows = [(t, t * dt) for t in frame_ids] + [(0, n_time * dt)]  # close the cycle
        seed_time = t0 * dt
    else:
        frame_ids = [t0]
        rows = [(t0, 0.0)]  # one row: a steady field
        seed_time = 0.0
    table.SetNumberOfRows(len(rows))
    for r, (fid, tm) in enumerate(rows):
        table.SetValue(r, 0, vtk.vtkVariant(int(fid)))
        table.SetValue(r, 1, vtk.vtkVariant(float(tm)))

    image, offset = _velocity_image([vel[:, :, :, f, :] for f in frame_ids], frame_ids, roi, spacing)
    # VTK's cell locator misses points lying exactly on the image faces (a planar seed slab at an
    # array border), so seeds sit a quarter voxel inside the volume.
    seeds = np.clip(seeds, 0.25, np.asarray(roi.shape, dtype=float) - 1.25)
    seed_pts = vtk.vtkPoints()
    for s in seeds:
        seed_pts.InsertNextPoint(*(s * spacing))
    source = vtk.vtkPolyData()
    source.SetPoints(seed_pts)

    voxel = float(np.min(spacing))
    tracer = vtkvmtk.vtkvmtkStaticTemporalStreamTracer()
    tracer.SetInputData(image)
    tracer.SetSourceData(source)
    tracer.SetTimeStepsTable(table)
    tracer.SetUseVectorComponents(0)
    tracer.SetVectorPrefix("velocity_")
    tracer.SetSeedTime(seed_time)
    tracer.SetPeriodic(1 if pathlines else 0)
    tracer.SetIntegratorTypeToRungeKutta45()
    tracer.SetIntegrationStepUnit(1)  # LENGTH_UNIT: steps in mm
    tracer.SetInitialIntegrationStep(0.25 * voxel)
    tracer.SetMinimumIntegrationStep(0.05 * voxel)
    tracer.SetMaximumIntegrationStep(1.0 * voxel)
    tracer.SetMaximumNumberOfSteps(100000)
    # VMTK's stagnation test compares displacement/step (dimensionless here) with TerminalSpeed, so
    # it is not a speed in mm/s; slow particles are handled by the seed filter above instead.
    tracer.SetTerminalSpeed(1e-12)
    tracer.SetComputeVorticity(0)
    if pathlines:
        # Integrate forward in time; the horizon (seconds) is applied below.
        tracer.SetIntegrationDirectionToForward()
        tracer.SetMaximumPropagation(1e9)
    else:
        direction = str(params.integration_direction or "forward")
        {"forward": tracer.SetIntegrationDirectionToForward,
         "backward": tracer.SetIntegrationDirectionToBackward}.get(direction, tracer.SetIntegrationDirectionToBoth)()
        tracer.SetMaximumPropagation(float(params.max_length))
    tracer.Update()
    out = tracer.GetOutput()

    pts_mm = numpy_support.vtk_to_numpy(out.GetPoints().GetData()).astype(float) if out.GetNumberOfPoints() else np.empty((0, 3))
    time_arr = out.GetPointData().GetArray("IntegrationTime")
    times = numpy_support.vtk_to_numpy(time_arr).astype(float) if time_arr is not None else np.zeros(len(pts_mm))
    traces: list[FlowTrace] = []
    for c in range(out.GetNumberOfCells()):
        ids = out.GetCell(c).GetPointIds()
        idx = np.asarray([ids.GetId(i) for i in range(ids.GetNumberOfIds())], dtype=np.int64)
        if len(idx) < 2:
            continue
        p = pts_mm[idx]
        t = times[idx]
        if pathlines:
            keep = t <= seed_time + float(params.max_length) + 1e-9
            p, t = p[keep], t[keep]
        if len(p) < 2:
            continue
        run = _longest_inside_run(p / spacing, roi)
        if run is None:
            continue
        p, t = p[run], t[run]
        traces.append(FlowTrace(points=p / spacing, time=t - t[0], speed=_speed(p, t)))
    if params.resample_paths:
        traces = [resample_trace(tr, spacing_vox=float(params.resample_spacing_vox), spacing=spacing) for tr in traces]
    return traces


def _speed(points_mm: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Speed (mm/s) at each point from the integrated positions and times."""
    if len(points_mm) < 2:
        return np.zeros(len(points_mm))
    ds = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    dt = np.diff(time_s)
    seg = np.where(np.abs(dt) > 1e-12, ds / np.where(np.abs(dt) > 1e-12, np.abs(dt), 1.0), 0.0)
    out = np.empty(len(points_mm))
    out[0] = seg[0]
    out[-1] = seg[-1]
    out[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return out


def resample_trace(trace: FlowTrace, *, spacing_vox: float, spacing: Sequence[float]) -> FlowTrace:
    """Resample a trace to ~uniform arc length (voxel units), interpolating time and speed."""
    pts = np.asarray(trace.points, dtype=float)
    if len(pts) < 2:
        return trace
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    step = max(float(spacing_vox), 1e-3)
    if s[-1] <= step:
        return trace
    ss = np.arange(0.0, s[-1] + 0.5 * step, step)
    new = np.column_stack([np.interp(ss, s, pts[:, k]) for k in range(3)])
    return FlowTrace(points=new, time=np.interp(ss, s, trace.time), speed=np.interp(ss, s, trace.speed))


def vertex_scalars(traces: Sequence[FlowTrace], metric: ColorMetric, *, spacing: Sequence[float] = (1, 1, 1)) -> list[np.ndarray]:
    """Per-vertex colouring values: speed (mm/s), integration time (s), arc length (mm), or zeros."""
    sp = np.asarray(spacing, dtype=float)
    out = []
    for tr in traces:
        if metric == "speed":
            out.append(np.asarray(tr.speed, dtype=float))
        elif metric == "integration_time":
            out.append(np.asarray(tr.time, dtype=float))
        elif metric == "arc_length":
            seg = np.linalg.norm(np.diff(np.asarray(tr.points) * sp, axis=0), axis=1)
            out.append(np.concatenate([[0.0], np.cumsum(seg)]))
        else:
            out.append(np.zeros(tr.n_points))
    return out


def mean_speeds(traces: Sequence[FlowTrace]) -> np.ndarray:
    """Mean speed (mm/s) of each trace."""
    return np.asarray([float(np.mean(t.speed)) if t.n_points else 0.0 for t in traces], dtype=float)


__all__ = [
    "COLOR_METRICS",
    "INTEGRATION_DIRECTIONS",
    "SEED_MODES",
    "TRACE_MODES",
    "ColorMetric",
    "FlowTrace",
    "FlowTraceParams",
    "IntegrationDirection",
    "PlaneSide",
    "SeedMode",
    "TraceMode",
    "mask_roi",
    "mean_speeds",
    "resample_trace",
    "stream_seed_cloud",
    "trace_flow",
    "vertex_scalars",
]
