"""Napari 4D flow streamlines / pathlines overlay (phase-synced Vectors lines)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.gui.core.spatial import layer_affine
from nvitk.gui.viz.layers import (
    _install_phase_dims,
    _move_time_axis_last,
    _repair_time_dim_range,
    _time_axis_index_from_layer,
    _time_index_from_viewer,
)
from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
from nvitk.viz.streamlines import (
    ColorMetric,
    FlowTraceParams,
    compute_pathlines,
    compute_streamlines,
    sample_vel_trilinear,
    vertex_scalars_for_polylines,
)

FLOW_STREAMLINES_LAYER = "Flow streamlines"


def _unique_flow_streamlines_layer_name(viewer: Any, base: str = FLOW_STREAMLINES_LAYER) -> str:
    """Return *base* or ``base (N)`` so each tool run gets a distinct layer name."""
    existing = {str(getattr(lyr, "name", "")) for lyr in viewer.layers}
    if base not in existing:
        return base
    n = 2
    while f"{base} ({n})" in existing:
        n += 1
    return f"{base} ({n})"

COLORMAP_CHOICES = (
    "turbo",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "hsv",
)


@dataclass
class FlowStreamlineCache:
    """Precomputed velocity volume and trace display settings."""

    velocity: np.ndarray
    mask: np.ndarray
    params: FlowTraceParams
    n_time: int
    edge_width: float
    opacity: float
    colormap: str
    color_metric: ColorMetric
    per_vertex_color: bool
    speed_lo: float
    speed_hi: float
    _polyline_cache: dict[tuple, tuple[list[np.ndarray], list]] = field(
        default_factory=dict, repr=False
    )
    _cache_order: list[tuple] = field(default_factory=list, repr=False)


def _global_speed_limits(velocity: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    mag = np.linalg.norm(to_numpy(velocity).astype(np.float64), axis=-1)
    roi = to_numpy(mask) > 0
    if mag.ndim == 4:
        vals = mag[roi] if roi.any() else mag.ravel()
    else:
        vals = mag.ravel()
    if vals.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    return (lo, hi if hi > lo else lo + 1.0)


def _scalar_to_rgba(value: float, lo: float, hi: float, cmap_name: str) -> np.ndarray:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap(str(cmap_name or "turbo"))
    norm = mcolors.Normalize(vmin=float(lo), vmax=float(hi))
    return np.asarray(cmap(norm(float(value))), dtype=np.float32)


def _vertex_scalars_pathline_speed(
    polylines: list[np.ndarray],
    velocity_xyzt: np.ndarray,
    time_start: int,
) -> list[np.ndarray]:
    nt = int(velocity_xyzt.shape[3])
    t0 = int(np.clip(time_start, 0, nt - 1))
    out: list[np.ndarray] = []
    for poly in polylines:
        vals = []
        for k, p in enumerate(to_numpy(poly).astype(np.float64)):
            tt = min(t0 + int(k), nt - 1)
            v = sample_vel_trilinear(velocity_xyzt[..., tt, :], p)
            vals.append(float(np.linalg.norm(v)))
        out.append(np.asarray(vals, dtype=np.float64))
    return out


def _scalar_limits(
    cache: FlowStreamlineCache,
    scalar_lists: list[np.ndarray],
) -> tuple[float, float]:
    metric = str(cache.color_metric)
    if metric == "fixed":
        return (0.0, 1.0)
    if metric == "speed":
        return (cache.speed_lo, cache.speed_hi)
    flat = np.concatenate([s for s in scalar_lists if s.size > 0]) if scalar_lists else np.array([])
    if flat.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(flat, 2.0))
    hi = float(np.percentile(flat, 98.0))
    return (lo, hi if hi > lo else lo + 1.0)


def _paths_and_colors(
    polylines: list[np.ndarray],
    scalar_lists: list[np.ndarray],
    cache: FlowStreamlineCache,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not polylines:
        return [], []
    lo, hi = _scalar_limits(cache, scalar_lists)
    if str(cache.color_metric) == "fixed":
        fixed = np.array([0.2, 0.75, 1.0, 1.0], dtype=np.float32)
        if cache.per_vertex_color:
            paths: list[np.ndarray] = []
            colors: list[np.ndarray] = []
            for poly in polylines:
                if len(poly) < 2:
                    continue
                for i in range(len(poly) - 1):
                    paths.append(poly[i : i + 2])
                    colors.append(fixed)
            return paths, colors
        return polylines, [fixed] * len(polylines)

    if cache.per_vertex_color:
        paths = []
        colors = []
        for poly, scalars in zip(polylines, scalar_lists):
            if len(poly) < 2:
                continue
            for i in range(len(poly) - 1):
                paths.append(poly[i : i + 2])
                val = 0.5 * (float(scalars[i]) + float(scalars[i + 1]))
                colors.append(_scalar_to_rgba(val, lo, hi, cache.colormap))
        return paths, colors

    colors = []
    for scalars in scalar_lists:
        val = float(np.mean(scalars)) if scalars.size else lo
        colors.append(_scalar_to_rgba(val, lo, hi, cache.colormap))
    return polylines, colors


def _paths_to_vector_data(paths: list[np.ndarray]) -> np.ndarray:
    """Convert voxel-space path polylines to Napari vectors (N, 2, 3)."""
    tails: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    for path in paths:
        pts = to_numpy(path).astype(np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
            continue
        if pts.shape[0] == 2:
            tails.append(pts[0, :3])
            dirs.append(pts[1, :3] - pts[0, :3])
            continue
        for i in range(int(pts.shape[0]) - 1):
            tails.append(pts[i, :3])
            dirs.append(pts[i + 1, :3] - pts[i, :3])
    if not tails:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack(
        [np.asarray(tails, dtype=np.float32), np.asarray(dirs, dtype=np.float32)],
        axis=1,
    )


def _path_colors_to_vector_colors(
    paths: list[np.ndarray],
    colors: list[np.ndarray],
) -> list[np.ndarray]:
    """Expand one RGBA color per path into one color per vector segment."""
    out: list[np.ndarray] = []
    for path, color in zip(paths, colors):
        pts = to_numpy(path)
        n_seg = 1 if pts.shape[0] == 2 else max(0, int(pts.shape[0]) - 1)
        out.extend([color] * n_seg)
    return out


def build_flow_streamline_cache(
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
    resample_paths: bool = False,
    resample_spacing_vox: float = 0.5,
    edge_width: float = 0.25,
    opacity: float = 0.55,
    colormap: str = "turbo",
    color_metric: str = "speed",
    per_vertex_color: bool = True,
) -> FlowStreamlineCache:
    """Build velocity volume and trace parameters."""
    time_axis = _time_axis_index_from_layer(phase_layer) if phase_layer is not None else -1
    ap_np, rl_np, fh_np = _move_time_axis_last(ap, rl, fh, time_axis=time_axis)
    ap_a = as_backend_array(ap_np).astype(np.float64)
    rl_a = as_backend_array(rl_np).astype(np.float64)
    fh_a = as_backend_array(fh_np).astype(np.float64)
    mask_np = to_numpy(mask)
    if mask_np.ndim > 3:
        raise ValueError("Mask must be a 3D spatial volume for flow traces.")
    spatial_shape = tuple(int(x) for x in ap_a.shape[:3])
    if tuple(mask_np.shape) != spatial_shape:
        raise ValueError(
            f"Mask shape {tuple(mask_np.shape)} does not match AP spatial shape {spatial_shape}; "
            "align the mask to the phase grid first."
        )
    vx, vy, vz = velocity_mm_s_from_phases(ap_a, rl_a, fh_a)
    n_time = int(vx.shape[3])
    velocity = np.stack(
        [
            to_numpy(vx).astype(np.float32),
            to_numpy(vy).astype(np.float32),
            to_numpy(vz).astype(np.float32),
        ],
        axis=-1,
    )
    lbl_tuple = tuple(int(x) for x in label_ids) if label_ids else None
    mode: Literal["streamlines", "pathlines"] = (
        "pathlines" if str(trace_mode).strip().lower() == "pathlines" else "streamlines"
    )
    direction: Literal["forward", "backward", "both"] = "forward"
    d = str(integration_direction or "forward").strip().lower()
    if d in ("forward", "backward", "both"):
        direction = d  # type: ignore[assignment]
    seed_m: Literal["volume", "planar"] = (
        "planar" if str(seed_mode).strip().lower() == "planar" else "volume"
    )
    side: Literal["min", "max"] = (
        "max" if str(seed_plane_side).strip().lower() == "max" else "min"
    )
    metric: ColorMetric = "speed"
    m = str(color_metric or "speed").strip().lower()
    if m in ("speed", "integration_time", "arc_length", "fixed"):
        metric = m  # type: ignore[assignment]

    params = FlowTraceParams(
        n_seeds=int(n_seeds),
        max_length=float(max_length),
        stream_seed=int(stream_seed) if stream_seed is not None else None,
        label_ids=lbl_tuple,
        trace_mode=mode,
        integration_direction=direction,
        seed_mode=seed_m,
        seed_plane_axis=int(seed_plane_axis),
        seed_plane_side=side,
        dt_seconds=float(dt_seconds),
        resample_paths=bool(resample_paths),
        resample_spacing_vox=float(resample_spacing_vox),
    )
    speed_lo, speed_hi = _global_speed_limits(velocity, mask_np)
    return FlowStreamlineCache(
        velocity=velocity,
        mask=mask_np,
        params=params,
        n_time=n_time,
        edge_width=float(edge_width),
        opacity=float(opacity),
        colormap=str(colormap or "turbo"),
        color_metric=metric,
        per_vertex_color=bool(per_vertex_color),
        speed_lo=speed_lo,
        speed_hi=speed_hi,
    )


def _cache_key(cache: FlowStreamlineCache, time_index: int) -> tuple:
    p = cache.params
    return (
        int(time_index),
        str(p.trace_mode),
        int(p.n_seeds),
        float(p.max_length),
        int(p.stream_seed) if p.stream_seed is not None else None,
        p.label_ids,
        str(p.integration_direction),
        str(p.seed_mode),
        int(p.seed_plane_axis),
        str(p.seed_plane_side),
        float(p.dt_seconds),
        bool(p.resample_paths),
        float(p.resample_spacing_vox),
        str(cache.color_metric),
        bool(cache.per_vertex_color),
        str(cache.colormap),
    )


def _put_polyline_cache(
    cache: FlowStreamlineCache,
    key: tuple,
    value: tuple[list[np.ndarray], list],
) -> None:
    cache._polyline_cache[key] = value
    if key not in cache._cache_order:
        cache._cache_order.append(key)


def precompute_flow_streamline_frames(cache: FlowStreamlineCache) -> None:
    """Integrate traces for every cardiac phase (makes scrubbing instant)."""
    for t in range(int(cache.n_time)):
        flow_streamline_frame(cache, t)


def flow_streamline_frame(
    cache: FlowStreamlineCache,
    time_index: int,
) -> tuple[list[np.ndarray], list]:
    """Return display paths and RGBA edge colors for one cardiac phase."""
    t = int(np.clip(time_index, 0, max(0, cache.n_time - 1)))
    key = _cache_key(cache, t)
    if key in cache._polyline_cache:
        return cache._polyline_cache[key]

    if cache.params.trace_mode == "pathlines":
        polylines = compute_pathlines(
            cache.velocity, cache.mask, cache.params, time_start=t
        )
        if cache.color_metric == "speed":
            scalar_lists = _vertex_scalars_pathline_speed(polylines, cache.velocity, t)
        else:
            scalar_lists = vertex_scalars_for_polylines(
                polylines,
                velocity_xyz=cache.velocity[..., t, :],
                color_metric=cache.color_metric,
                dt_seconds=cache.params.dt_seconds,
                trace_mode="pathlines",
            )
    else:
        vel_t = cache.velocity[..., t, :]
        polylines = compute_streamlines(vel_t, cache.mask, cache.params)
        scalar_lists = vertex_scalars_for_polylines(
            polylines,
            velocity_xyz=vel_t,
            color_metric=cache.color_metric,
            dt_seconds=cache.params.dt_seconds,
            trace_mode="streamlines",
        )

    paths, colors = _paths_and_colors(polylines, scalar_lists, cache)
    value = (paths, colors)
    _put_polyline_cache(cache, key, value)
    return value


@dataclass
class FlowStreamlinePlayback:
    """Dims-synced flow trace Vectors layer."""

    cache: FlowStreamlineCache
    phase_layer: Any
    layer: Any
    dims_callback: Callable[[Any], None]


def stop_flow_streamline_playback(viewer: Any) -> None:
    """Disconnect dims hooks from a prior flow-trace overlay."""
    state = getattr(viewer, "_nvitk_flow_streamline_state", None)
    if state is None:
        return
    try:
        viewer.dims.events.current_step.disconnect(state.dims_callback)
    except Exception:
        pass
    setattr(viewer, "_nvitk_flow_streamline_state", None)


def _update_flow_streamline_layer(
    layer: Any,
    cache: FlowStreamlineCache,
    time_index: int,
    *,
    viewer: Any = None,
    phase_layer: Any = None,
) -> None:
    paths, colors = flow_streamline_frame(cache, time_index)
    layer.data = _paths_to_vector_data(paths)
    if paths and colors:
        layer.edge_color = _path_colors_to_vector_colors(paths, colors)
    if viewer is not None and phase_layer is not None:
        _repair_time_dim_range(viewer, phase_layer)


def add_animated_flow_streamlines_layer(
    viewer: Any,
    cache: FlowStreamlineCache,
    *,
    phase_layer: Any,
    spatial_reference_layer: Any = None,
    name: str = FLOW_STREAMLINES_LAYER,
    initial_time: int = 0,
    sync_dims: bool = True,
) -> FlowStreamlinePlayback:
    """Add flow traces; sync paths to the Napari dims slider / play bar."""
    # Only the latest run follows the dims slider; older layers stay as-is.
    stop_flow_streamline_playback(viewer)

    spatial_ref = spatial_reference_layer or phase_layer
    t0 = int(np.clip(initial_time, 0, max(0, cache.n_time - 1)))
    paths, colors = flow_streamline_frame(cache, t0)
    vector_data = _paths_to_vector_data(paths)

    layer_name = (
        _unique_flow_streamlines_layer_name(viewer)
        if name == FLOW_STREAMLINES_LAYER
        else name
    )

    kwargs: dict[str, Any] = {
        "name": layer_name,
        "vector_style": "line",
        "length": 1.0,
        "edge_width": cache.edge_width,
        "opacity": cache.opacity,
    }
    aff = layer_affine(spatial_ref)
    if aff is not None:
        kwargs["affine"] = aff
    if paths and colors:
        kwargs["edge_color"] = _path_colors_to_vector_colors(paths, colors)
    layer = viewer.add_vectors(vector_data, **kwargs)

    _install_phase_dims(viewer, phase_layer)
    try:
        layer.editable = False
        layer.metadata = {
            "nvitk_flow_streamlines": {
                "trace_mode": str(cache.params.trace_mode),
                "color_metric": str(cache.color_metric),
                "colormap": str(cache.colormap),
                "per_vertex_color": bool(cache.per_vertex_color),
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
        if not getattr(layer, "visible", True):
            return
        _repair_time_dim_range(viewer, phase_layer)
        t = _time_index_from_viewer(viewer, phase_layer, cache.n_time)
        _update_flow_streamline_layer(
            layer, cache, t, viewer=viewer, phase_layer=phase_layer
        )

    if sync_dims:
        viewer.dims.events.current_step.connect(_on_dims)

    playback = FlowStreamlinePlayback(
        cache=cache,
        phase_layer=phase_layer,
        layer=layer,
        dims_callback=_on_dims,
    )
    setattr(viewer, "_nvitk_flow_streamline_state", playback)
    return playback


__all__ = [
    "COLORMAP_CHOICES",
    "FLOW_STREAMLINES_LAYER",
    "FlowStreamlineCache",
    "FlowStreamlinePlayback",
    "add_animated_flow_streamlines_layer",
    "build_flow_streamline_cache",
    "flow_streamline_frame",
    "precompute_flow_streamline_frames",
    "stop_flow_streamline_playback",
]
