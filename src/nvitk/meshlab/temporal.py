"""Meshes over time (3D+t): build, measure, track, smooth and resample a :class:`MeshSeries`.

A series either has one topology for every frame (a tracked surface — vertex
``i`` is the same material point throughout) or a mesh of its own per frame
(marching cubes of each time point). Displacement, temporal smoothing and frame
interpolation need the former; :func:`propagate_mesh` turns the latter into it
by carrying one frame's mesh through the others.

In the GUI a series is one Surface layer whose frame follows the viewer's time
slider (:mod:`nvitk.gui.mesh.layers`); :func:`to_napari_4d` gives the 4D
``(t, x, y, z)`` form for export.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.types import Image, Mesh, MeshSeries


def _time_axis(image: Image) -> int:
    """The time axis of a 4D image (``T``/``C`` in its axes, else the last)."""
    axes = str(image.axes or "").upper()
    if len(axes) == image.ndim:
        for ch in ("T", "C"):
            if ch in axes:
                return axes.index(ch)
    return image.ndim - 1


def series_from_image(
    image: Image,
    *,
    label_id: int | None = None,
    level: float | None = None,
    step_size: int = 1,
    smooth_iterations: int = 0,
    world_space: bool = True,
) -> MeshSeries:
    """One surface per time frame of a 3D+t mask (or image, at an iso *level*).

    *label_id* picks one label of a label map (``None``: every non-zero voxel).
    *level* thresholds an intensity image instead. Frames with nothing to mesh
    get an empty mesh, so frame indices stay aligned with the image.
    """
    from nvitk.meshlab.marching_cubes import marching_cubes_binary
    from nvitk.meshlab.smoothing import taubin_smooth

    data = to_numpy(image.data)
    if data.ndim != 4:
        raise ValueError("series_from_image needs a 3D+t image.")
    t_ax = _time_axis(image)
    meta = dict(image.metadata or {})
    spatial_axes = "".join(ch for ch in str(image.axes or "XYZT").upper() if ch not in ("T", "C"))[:3] or "XYZ"
    frames: list[Mesh] = []
    for k in range(data.shape[t_ax]):
        key = [slice(None)] * 4
        key[t_ax] = k
        vol = data[tuple(key)]
        if level is not None:
            mask = vol >= float(level)
        elif label_id is not None:
            mask = vol == int(label_id)
        else:
            mask = vol != 0
        frame_img = Image(
            data=mask.astype(np.uint8),
            metadata={k2: v for k2, v in meta.items() if k2 in ("affine", "spacing", "x_res", "y_res", "z_res")},
            axes=spatial_axes,
        )
        mesh = marching_cubes_binary(frame_img, step_size=int(step_size), world_space=world_space)
        if mesh is None:
            mesh = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int32))
        elif smooth_iterations > 0:
            mesh = taubin_smooth(mesh, iterations=int(smooth_iterations))
        mesh.metadata.update({"frame": k, "name": f"frame_{k}"})
        frames.append(mesh)
    t_res = float(image.temporal_resolution or 1.0)
    name = f"{image.name or 'mask'}" + (f"_label{label_id}" if label_id is not None else "")
    return MeshSeries(
        frames=frames,
        times=np.arange(len(frames)) * t_res,
        metadata={"name": name, "t_res": t_res, "space": "world" if world_space else "voxel"},
    )


def series_metrics(series: MeshSeries) -> dict[str, np.ndarray]:
    """Per-frame ``time``, ``area``, ``volume`` (closed frames) and ``centroid`` (T, 3)."""
    from nvitk.meshlab.measure import surface_area, surface_centroid, volume
    from nvitk.meshlab.topology import is_watertight

    area, vol, cen = [], [], []
    for mesh in series:
        area.append(surface_area(mesh))
        vol.append(volume(mesh) if mesh.n_faces and is_watertight(mesh) else np.nan)
        cen.append(surface_centroid(mesh) if mesh.n_vertices else np.full(3, np.nan))
    return {
        "time": np.asarray(series.times, dtype=float),
        "area": np.asarray(area),
        "volume": np.asarray(vol),
        "centroid": np.asarray(cen),
    }


def vertex_displacement(series: MeshSeries, reference: int = 0) -> np.ndarray:
    """``(T, N)`` distance of each vertex from where it was at frame *reference*."""
    stack = series.vertex_stack()
    return np.linalg.norm(stack - stack[int(reference)], axis=2)


def vertex_velocity(series: MeshSeries) -> np.ndarray:
    """``(T, N)`` speed of each vertex (central differences over the frame times)."""
    stack = series.vertex_stack()
    t = np.asarray(series.times, dtype=float)
    if len(t) < 2:
        return np.zeros(stack.shape[:2])
    vel = np.gradient(stack, t, axis=0)
    return np.linalg.norm(vel, axis=2)


def smooth_in_time(series: MeshSeries, sigma_frames: float = 1.0, *, cyclic: bool = True) -> MeshSeries:
    """Gaussian smoothing of every vertex trajectory along time.

    *cyclic* wraps around (a cardiac cycle); otherwise the ends are held.
    """
    from scipy.ndimage import gaussian_filter1d

    stack = series.vertex_stack()
    smoothed = gaussian_filter1d(stack, float(sigma_frames), axis=0, mode="wrap" if cyclic else "nearest")
    return MeshSeries.from_vertex_stack(smoothed, series[0].faces, times=series.times, metadata=series.metadata)


def interpolate_frames(
    series: MeshSeries,
    n_frames: int | None = None,
    *,
    factor: float = 2.0,
    cyclic: bool = False,
) -> MeshSeries:
    """Resample a tracked series in time (linear in each vertex trajectory).

    *n_frames* (or *factor* × the current count) frames span the same time range;
    *cyclic* treats the last frame as followed by the first (cardiac cycle).
    """
    stack = series.vertex_stack()
    t = np.asarray(series.times, dtype=float)
    n_old = len(t)
    n_new = int(n_frames) if n_frames else max(2, int(round(n_old * float(factor))))
    step = (t[-1] - t[0]) / (n_old - 1) if n_old > 1 else 1.0
    if cyclic:
        t_ext = np.append(t, t[-1] + step)
        stack_ext = np.concatenate([stack, stack[:1]], axis=0)
        new_t = t[0] + np.arange(n_new) * (t_ext[-1] - t[0]) / n_new
    else:
        t_ext, stack_ext = t, stack
        new_t = np.linspace(t[0], t[-1], n_new)
    pos = np.interp(new_t, t_ext, np.arange(len(t_ext)))
    lo = np.floor(pos).astype(int).clip(0, len(t_ext) - 1)
    hi = np.minimum(lo + 1, len(t_ext) - 1)
    w = (pos - lo)[:, None, None]
    out = (1 - w) * stack_ext[lo] + w * stack_ext[hi]
    meta = dict(series.metadata)
    if n_new > 1:
        meta["t_res"] = float(new_t[1] - new_t[0])
    return MeshSeries.from_vertex_stack(out, series[0].faces, times=new_t, metadata=meta)


def closest_points(points: np.ndarray, surface: Mesh) -> np.ndarray:
    """The closest point on *surface* to each of *points*."""
    import vtk

    from nvitk.meshlab.convert import to_pyvista

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(to_pyvista(surface))
    locator.BuildLocator()
    out = np.empty_like(points, dtype=float)
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)
    for i, p in enumerate(points):
        locator.FindClosestPoint(p, closest, cell_id, sub_id, dist2)
        out[i] = closest
    return out


def propagate_mesh(
    series: MeshSeries,
    *,
    reference: int = 0,
    iterations: int = 3,
    smooth_iterations: int = 5,
) -> MeshSeries:
    """Carry frame *reference*'s mesh through every other frame (shared topology).

    Each frame starts from the previous frame's result (outward from *reference*
    in both directions); vertices are pulled onto that frame's surface by
    closest-point projection, alternated with Taubin smoothing so the mesh slides
    rather than bunching. Gives vertex correspondence to a per-frame
    (marching-cubes) series, so displacement and temporal smoothing apply.
    """
    from nvitk.meshlab.smoothing import taubin_smooth

    ref = series[int(reference)]
    if not ref.n_faces:
        raise ValueError("The reference frame is empty.")
    out: list[np.ndarray | None] = [None] * len(series)
    out[int(reference)] = ref.vertices.copy()

    def _fit(start: np.ndarray, target: Mesh) -> np.ndarray:
        if not target.n_faces:
            return start.copy()
        v = start.copy()
        for _ in range(max(1, int(iterations))):
            v = closest_points(v, target)
            if smooth_iterations > 0:
                v = taubin_smooth(ref.with_vertices(v), iterations=int(smooth_iterations)).vertices
        return closest_points(v, target)

    for k in range(int(reference) + 1, len(series)):
        out[k] = _fit(out[k - 1], series[k])
    for k in range(int(reference) - 1, -1, -1):
        out[k] = _fit(out[k + 1], series[k])
    return MeshSeries.from_vertex_stack(
        np.stack(out), ref.faces, times=series.times, metadata=dict(series.metadata),
    )


def to_napari_4d(
    series: MeshSeries,
    *,
    values: Sequence[np.ndarray] | np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Napari 4D Surface data for a series: ``(vertices (ΣN, 4), faces[, values])``.

    Frame ``k``'s vertices get time coordinate ``k``; give the layer
    ``scale=(t_res, 1, 1, 1)`` to place frames in time. Napari selects a frame's
    faces by *exact* equality between those integers and the slider position
    divided by the scale, which a non-integral step (``3 * 0.1 / 0.1``) can miss —
    the GUI therefore shows series as a 3D layer swapped per frame
    (:func:`nvitk.gui.mesh.layers.add_mesh_series_layer`); this export is for
    files and other viewers.
    """
    verts, faces, vals = [], [], []
    offset = 0
    for k, mesh in enumerate(series):
        t = np.full((mesh.n_vertices, 1), float(k))
        verts.append(np.hstack([t, mesh.vertices]))
        faces.append(mesh.faces + offset)
        if values is not None:
            vals.append(np.asarray(values[k], dtype=float).reshape(-1))
        offset += mesh.n_vertices
    v = np.vstack(verts) if verts else np.zeros((0, 4))
    f = np.vstack(faces).astype(np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    if values is not None:
        return v, f, np.concatenate(vals)
    return v, f


def from_napari_4d(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> MeshSeries:
    """Split 4D ``(t, x, y, z)`` Surface data back into one mesh per time value."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    times = np.unique(vertices[:, 0])
    frames = []
    for t in times:
        idx = np.flatnonzero(vertices[:, 0] == t)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[idx] = np.arange(len(idx))
        sel = np.all(np.isin(faces, idx), axis=1)
        frames.append(Mesh(vertices=vertices[idx, 1:], faces=remap[faces[sel]], metadata=dict(metadata or {})))
    return MeshSeries(frames=frames, times=times, metadata=dict(metadata or {}))


__all__ = [
    "closest_points",
    "from_napari_4d",
    "interpolate_frames",
    "propagate_mesh",
    "series_from_image",
    "series_metrics",
    "smooth_in_time",
    "to_napari_4d",
    "vertex_displacement",
    "vertex_velocity",
]
