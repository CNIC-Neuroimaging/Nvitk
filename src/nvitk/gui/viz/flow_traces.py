"""Napari 4D-flow particle traces (VMTK streamlines / pathlines), phase-synced Vectors lines.

Tracing is :func:`nvitk.vmtk.flow.trace_flow` (VMTK's temporal stream tracer, physical units):
streamlines of the frozen phase under the dims slider, or pathlines released at that phase and
advected periodically through the cardiac cycle. Every phase's traces are cached, so scrubbing and
playback only re-draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.gui.core.spatial import layer_affine, layer_spacing
from nvitk.gui.viz.layers import (
    _install_phase_dims,
    _move_time_axis_last,
    _repair_time_dim_range,
    _time_axis_index_from_layer,
    _time_index_from_viewer,
)
from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
from nvitk.vmtk.flow import (
    COLOR_METRICS,
    INTEGRATION_DIRECTIONS,
    SEED_MODES,
    TRACE_MODES,
    ColorMetric,
    FlowTrace,
    FlowTraceParams,
    trace_flow,
    vertex_scalars,
)

FLOW_TRACES_LAYER = "Flow traces (VMTK)"

COLORMAP_CHOICES = ("turbo", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "hsv")


def _unique_layer_name(viewer: Any, base: str = FLOW_TRACES_LAYER) -> str:
    """Return *base* or ``base (N)`` so each tool run gets a distinct layer name."""
    existing = {str(getattr(lyr, "name", "")) for lyr in viewer.layers}
    if base not in existing:
        return base
    n = 2
    while f"{base} ({n})" in existing:
        n += 1
    return f"{base} ({n})"


@dataclass
class FlowTraceCache:
    """Velocity field, trace settings and the per-phase trace cache."""

    velocity: np.ndarray
    mask: np.ndarray
    spacing: tuple[float, float, float]
    params: FlowTraceParams
    n_time: int
    edge_width: float
    opacity: float
    colormap: str
    color_metric: ColorMetric
    per_vertex_color: bool
    speed_lo: float
    speed_hi: float
    _frames: dict[int, tuple[list[np.ndarray], list]] = field(default_factory=dict, repr=False)


def _global_speed_limits(velocity: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """2nd/98th percentile speed within *mask* across the whole field (fixed speed colouring)."""
    mag = np.linalg.norm(velocity.astype(np.float64), axis=-1)
    roi = mask > 0
    vals = mag[roi] if (mag.ndim == 4 and roi.any()) else mag.ravel()
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(vals, 2.0)), float(np.percentile(vals, 98.0))
    return (lo, hi if hi > lo else lo + 1.0)


def _scalar_to_rgba(values: np.ndarray, lo: float, hi: float, cmap_name: str) -> np.ndarray:
    """Map scalars to RGBA through a Matplotlib colormap."""
    import matplotlib
    import matplotlib.colors as mcolors

    cmap = matplotlib.colormaps[str(cmap_name or "turbo")]
    return np.asarray(cmap(mcolors.Normalize(vmin=float(lo), vmax=float(hi))(values)), dtype=np.float32)


def _scalar_limits(cache: FlowTraceCache, scalars: list[np.ndarray]) -> tuple[float, float]:
    metric = str(cache.color_metric)
    if metric == "fixed":
        return (0.0, 1.0)
    if metric == "speed":
        return (cache.speed_lo, cache.speed_hi)
    flat = np.concatenate([s for s in scalars if s.size]) if scalars else np.array([])
    if flat.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.percentile(flat, 2.0)), float(np.percentile(flat, 98.0))
    return (lo, hi if hi > lo else lo + 1.0)


def _paths_and_colors(traces: list[FlowTrace], cache: FlowTraceCache) -> tuple[list[np.ndarray], list]:
    """Display paths (per segment when colouring per vertex) and their RGBA colours."""
    if not traces:
        return [], []
    scalars = vertex_scalars(traces, cache.color_metric, spacing=cache.spacing)
    lo, hi = _scalar_limits(cache, scalars)
    fixed = np.array([0.2, 0.75, 1.0, 1.0], dtype=np.float32)
    paths: list[np.ndarray] = []
    colors: list = []
    for tr, sc in zip(traces, scalars):
        pts = np.asarray(tr.points, dtype=np.float32)
        if len(pts) < 2:
            continue
        if cache.per_vertex_color:
            seg_vals = 0.5 * (sc[:-1] + sc[1:])
            rgba = [fixed] * (len(pts) - 1) if cache.color_metric == "fixed" else list(
                _scalar_to_rgba(seg_vals, lo, hi, cache.colormap))
            for i in range(len(pts) - 1):
                paths.append(pts[i:i + 2])
                colors.append(rgba[i])
        else:
            paths.append(pts)
            colors.append(fixed if cache.color_metric == "fixed"
                          else _scalar_to_rgba(np.array([float(np.mean(sc))]), lo, hi, cache.colormap)[0])
    return paths, colors


def _paths_to_vector_data(paths: list[np.ndarray]) -> np.ndarray:
    """Voxel-space polylines -> Napari vectors ``(N, 2, 3)`` (tail, direction)."""
    tails, dirs = [], []
    for pts in paths:
        p = np.asarray(pts, dtype=np.float64)
        for i in range(len(p) - 1):
            tails.append(p[i])
            dirs.append(p[i + 1] - p[i])
    if not tails:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([np.asarray(tails, dtype=np.float32), np.asarray(dirs, dtype=np.float32)], axis=1)


def _vector_colors(paths: list[np.ndarray], colors: list) -> list:
    """One colour per vector segment."""
    out = []
    for p, c in zip(paths, colors):
        out.extend([c] * max(0, len(p) - 1))
    return out


def build_flow_trace_cache(
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    *,
    phase_layer: Any = None,
    label_ids: list[int] | None = None,
    trace_mode: str = "streamlines",
    n_seeds: int = 64,
    max_length: float = 35.0,
    stream_seed: int | None = 42,
    integration_direction: str = "forward",
    seed_mode: str = "planar",
    seed_plane_axis: int = 2,
    seed_plane_side: str = "min",
    dt_seconds: float = 1.0,
    min_speed: float = 1.0,
    resample_paths: bool = False,
    resample_spacing_vox: float = 0.5,
    edge_width: float = 0.25,
    opacity: float = 0.55,
    colormap: str = "turbo",
    color_metric: str = "speed",
    per_vertex_color: bool = True,
) -> FlowTraceCache:
    """Velocity (mm/s) from the three phase layers and the trace settings."""
    time_axis = _time_axis_index_from_layer(phase_layer) if phase_layer is not None else -1
    ap_np, rl_np, fh_np = _move_time_axis_last(ap, rl, fh, time_axis=time_axis)
    vx, vy, vz = velocity_mm_s_from_phases(
        as_backend_array(ap_np).astype(np.float64),
        as_backend_array(rl_np).astype(np.float64),
        as_backend_array(fh_np).astype(np.float64),
    )
    velocity = np.stack([to_numpy(v).astype(np.float32) for v in (vx, vy, vz)], axis=-1)
    mask_np = to_numpy(mask)
    if mask_np.ndim != 3 or tuple(mask_np.shape) != tuple(velocity.shape[:3]):
        raise ValueError(
            f"Mask shape {tuple(mask_np.shape)} does not match the phase grid {tuple(velocity.shape[:3])}; "
            "align the mask to the phase grid first."
        )
    sp = layer_spacing(phase_layer) if phase_layer is not None else None
    spacing = tuple(float(s) for s in (sp[:3] if sp is not None and len(sp) >= 3 else (1.0, 1.0, 1.0)))
    mode = "pathlines" if str(trace_mode).strip().lower() == "pathlines" else "streamlines"
    direction = str(integration_direction or "forward").strip().lower()
    metric = str(color_metric or "speed").strip().lower()
    params = FlowTraceParams(
        n_seeds=int(n_seeds),
        max_length=float(max_length),
        stream_seed=int(stream_seed) if stream_seed is not None else None,
        label_ids=tuple(int(x) for x in label_ids) if label_ids else None,
        trace_mode=mode,  # type: ignore[arg-type]
        integration_direction=direction if direction in INTEGRATION_DIRECTIONS else "forward",  # type: ignore[arg-type]
        seed_mode="planar" if str(seed_mode).strip().lower() == "planar" else "volume",
        seed_plane_axis=int(seed_plane_axis),
        seed_plane_side="max" if str(seed_plane_side).strip().lower() == "max" else "min",
        dt_seconds=float(dt_seconds),
        min_speed=float(min_speed),
        resample_paths=bool(resample_paths),
        resample_spacing_vox=float(resample_spacing_vox),
    )
    lo, hi = _global_speed_limits(velocity, mask_np)
    return FlowTraceCache(
        velocity=velocity, mask=mask_np, spacing=spacing, params=params, n_time=int(velocity.shape[3]),
        edge_width=float(edge_width), opacity=float(opacity), colormap=str(colormap or "turbo"),
        color_metric=metric if metric in COLOR_METRICS else "speed",  # type: ignore[arg-type]
        per_vertex_color=bool(per_vertex_color), speed_lo=lo, speed_hi=hi,
    )


def flow_trace_frame(cache: FlowTraceCache, time_index: int) -> tuple[list[np.ndarray], list]:
    """Display paths and RGBA colours for one cardiac phase (cached)."""
    t = int(np.clip(time_index, 0, max(0, cache.n_time - 1)))
    if t not in cache._frames:
        traces = trace_flow(cache.velocity, cache.mask, cache.params, spacing=cache.spacing, time_start=t)
        cache._frames[t] = _paths_and_colors(traces, cache)
    return cache._frames[t]


def precompute_flow_trace_frames(cache: FlowTraceCache) -> None:
    """Trace every cardiac phase up front (makes scrubbing instant)."""
    for t in range(int(cache.n_time)):
        flow_trace_frame(cache, t)


@dataclass
class FlowTracePlayback:
    """Dims-synced flow trace Vectors layer."""

    cache: FlowTraceCache
    phase_layer: Any
    layer: Any
    dims_callback: Callable[[Any], None]


def stop_flow_trace_playback(viewer: Any) -> None:
    """Disconnect the dims hook of the previous flow-trace overlay."""
    state = getattr(viewer, "_nvitk_flow_trace_state", None)
    if state is None:
        return
    try:
        viewer.dims.events.current_step.disconnect(state.dims_callback)
    except Exception:
        pass
    setattr(viewer, "_nvitk_flow_trace_state", None)


def add_animated_flow_traces_layer(
    viewer: Any,
    cache: FlowTraceCache,
    *,
    phase_layer: Any,
    spatial_reference_layer: Any = None,
    name: str = FLOW_TRACES_LAYER,
    initial_time: int = 0,
    sync_dims: bool = True,
) -> FlowTracePlayback:
    """Add the traces as a Vectors layer that follows the dims slider / play bar."""
    stop_flow_trace_playback(viewer)
    spatial_ref = spatial_reference_layer or phase_layer
    t0 = int(np.clip(initial_time, 0, max(0, cache.n_time - 1)))
    paths, colors = flow_trace_frame(cache, t0)
    kwargs: dict[str, Any] = {
        "name": _unique_layer_name(viewer) if name == FLOW_TRACES_LAYER else name,
        "vector_style": "line",
        "length": 1.0,
        "edge_width": cache.edge_width,
        "opacity": cache.opacity,
    }
    aff = layer_affine(spatial_ref)
    if aff is not None:
        kwargs["affine"] = aff
    if paths:
        kwargs["edge_color"] = _vector_colors(paths, colors)
    layer = viewer.add_vectors(_paths_to_vector_data(paths), **kwargs)
    _install_phase_dims(viewer, phase_layer)
    try:
        layer.editable = False
        layer.metadata = {
            "nvitk_flow_traces": {
                "engine": "vmtk",
                "trace_mode": str(cache.params.trace_mode),
                "color_metric": str(cache.color_metric),
                "colormap": str(cache.colormap),
                "speed_lo_mm_s": float(cache.speed_lo),
                "speed_hi_mm_s": float(cache.speed_hi),
            }
        }
    except Exception:
        pass
    try:
        viewer.layers.selection.active = phase_layer
    except Exception:
        pass

    def _on_dims(_event: Any = None) -> None:
        """Show the traces of the viewer's current cardiac phase."""
        if not getattr(layer, "visible", True):
            return
        _repair_time_dim_range(viewer, phase_layer)
        t = _time_index_from_viewer(viewer, phase_layer, cache.n_time)
        p, c = flow_trace_frame(cache, t)
        layer.data = _paths_to_vector_data(p)
        if p:
            layer.edge_color = _vector_colors(p, c)

    if sync_dims:
        viewer.dims.events.current_step.connect(_on_dims)
    playback = FlowTracePlayback(cache=cache, phase_layer=phase_layer, layer=layer, dims_callback=_on_dims)
    setattr(viewer, "_nvitk_flow_trace_state", playback)
    return playback


__all__ = [
    "COLORMAP_CHOICES",
    "COLOR_METRICS",
    "FLOW_TRACES_LAYER",
    "SEED_MODES",
    "TRACE_MODES",
    "FlowTraceCache",
    "FlowTracePlayback",
    "add_animated_flow_traces_layer",
    "build_flow_trace_cache",
    "flow_trace_frame",
    "precompute_flow_trace_frames",
    "stop_flow_trace_playback",
]
