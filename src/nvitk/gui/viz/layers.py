"""Napari-native visualization (hotspots, 4D flow vectors) without PyVista."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.gui.core.spatial import layer_affine, layer_spacing
from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
from nvitk.viz.pet_hotspots import HotspotMode, _roi_mask, _select_hotspots

HOTSPOTS_LAYER = "SUV hotspots"
FLOW_VECTORS_LAYER = "Flow velocity"
DEFAULT_FLOW_EDGE_WIDTH = 0.3
DEFAULT_HOTSPOT_POINT_SIZE = 6.0
DEFAULT_HOTSPOT_COLORMAP = "viridis"


@dataclass
class FlowVectorCache:
    """Precomputed subsampled flow glyphs for all cardiac phases."""

    positions: np.ndarray
    velocities: np.ndarray
    magnitudes: np.ndarray
    n_time: int
    max_arrow_voxels: float
    speed_percentile: float


def hotspot_points_from_volumes(
    suv: np.ndarray,
    mask: np.ndarray,
    *,
    label_ids = None,
    hotspot = "top_percent",
    top_percent = 0.1,
    top_k = None,
    threshold = None,
    max_points = 20000,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Return Napari point coordinates (N,3), SUV values, and feature columns.

    Subsamples to *max_points* by highest SUV when needed.
    """
    suv_arr = to_numpy(suv).astype(np.float64)
    mask_arr = to_numpy(mask)
    roi = _roi_mask(mask_arr, label_ids)
    hot = _select_hotspots(
        suv_arr,
        roi,
        hotspot=hotspot,
        top_percent=top_percent,
        top_k=top_k,
        suv_threshold=threshold,
    )
    coords = np.argwhere(hot)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64), {}

    vals = suv_arr[hot].astype(np.float64)
    if coords.shape[0] > max_points:
        order = np.argsort(vals)[::-1][: int(max_points)]
        coords = coords[order]
        vals = vals[order]

    features = {
        "suv": vals,
        "i": coords[:, 0].astype(int),
        "j": coords[:, 1].astype(int),
        "k": coords[:, 2].astype(int),
    }
    return coords.astype(np.float64), vals, features


@dataclass
class HotspotPointsState:
    """Tracks SUV hotspot layer and Napari 0.7 style-sync callbacks."""

    layer: Any
    disconnect_style_sync = None


def stop_hotspot_points_sync(viewer: Any) -> None:
    """Disconnect style-sync hooks from a prior SUV hotspot layer."""
    state = getattr(viewer, "_nvitk_hotspot_points_state", None)
    if state is None:
        return
    if state.disconnect_style_sync is not None:
        try:
            state.disconnect_style_sync()
        except Exception:
            pass
    setattr(viewer, "_nvitk_hotspot_points_state", None)


def _point_feature_columns(layer: Any) -> set[str]:
    """Column names on a Napari Points ``features`` table (dict or DataFrame)."""
    feat = getattr(layer, "features", None)
    if feat is None:
        return set()
    cols = getattr(feat, "columns", None)
    if cols is not None:
        return {str(c) for c in cols}
    if isinstance(feat, dict):
        return {str(k) for k in feat.keys()}
    return set()


def install_points_style_sync(
    layer: Any,
    *,
    sync_face_color: bool = False,
) -> Callable[[], None]:
    """
    Napari 0.5+: broadcast layer-panel style controls to all points.

    Without this, the GUI sliders only affect the next added or selected points.
    """
    def _sync_size(_event: Any = None) -> None:
        """Broadcast the layer-panel's current size to every existing point."""
        if len(layer.data) == 0:
            return
        layer.size = float(layer.current_size)

    def _sync_symbol(_event: Any = None) -> None:
        """Broadcast the layer-panel's current symbol to every existing point."""
        if len(layer.data) == 0:
            return
        layer.symbol = layer.current_symbol

    def _sync_border(_event: Any = None) -> None:
        """Broadcast the layer-panel's current border width to every existing point."""
        if len(layer.data) == 0:
            return
        layer.border_width = float(layer.current_border_width)

    def _sync_face_color(_event: Any = None) -> None:
        """Broadcast the layer-panel's current face color to every existing point, unless it names a
        feature column (in which case colormap-driven coloring should be left alone)."""
        if len(layer.data) == 0:
            return
        val = layer.current_face_color
        if isinstance(val, str) and val in _point_feature_columns(layer):
            return
        layer.face_color = val

    layer.events.current_size.connect(_sync_size)
    layer.events.current_symbol.connect(_sync_symbol)
    layer.events.current_border_width.connect(_sync_border)
    callbacks = [
        (layer.events.current_size, _sync_size),
        (layer.events.current_symbol, _sync_symbol),
        (layer.events.current_border_width, _sync_border),
    ]
    if sync_face_color and hasattr(layer.events, "current_face_color"):
        layer.events.current_face_color.connect(_sync_face_color)
        callbacks.append((layer.events.current_face_color, _sync_face_color))

    def disconnect() -> None:
        """Detach all the style-sync callbacks installed above."""
        for evt, cb in callbacks:
            try:
                evt.disconnect(cb)
            except Exception:
                pass

    return disconnect


def init_points_layer_style(
    layer: Any,
    *,
    size: float,
    symbol: str,
    face_color: Any,
) -> None:
    """Set scalar style on all points and align Napari ``current_*`` controls."""
    layer.size = float(size)
    layer.symbol = symbol
    layer.face_color = face_color
    if hasattr(layer, "current_size"):
        layer.current_size = float(size)
    if hasattr(layer, "current_symbol"):
        layer.current_symbol = symbol
    if hasattr(layer, "current_face_color"):
        layer.current_face_color = face_color


def add_hotspot_points_layer(
    viewer: Any,
    coords: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    reference_layer: Any,
    name = HOTSPOTS_LAYER,
    point_size = DEFAULT_HOTSPOT_POINT_SIZE,
    colormap = DEFAULT_HOTSPOT_COLORMAP,
) -> Any:
    """Add (replacing any prior) a Points layer for PET hotspot *coords*, colored by SUV via
    *colormap* (or plain red if there are no points), and install style-sync callbacks."""
    stop_hotspot_points_sync(viewer)
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)
    size = float(point_size)
    kwargs = {"size": size, "symbol": "o", "border_width_is_relative": False}
    if coords.shape[0] == 0:
        kwargs["face_color"] = "red"
    else:
        kwargs["face_color"] = "suv"
        kwargs["face_colormap"] = str(colormap or DEFAULT_HOTSPOT_COLORMAP)
        vals = to_numpy(features["suv"]).astype(np.float64)
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        kwargs["face_contrast_limits"] = (lo, hi if hi > lo else lo + 1.0)
    aff = layer_affine(reference_layer)
    if aff is not None:
        kwargs["affine"] = aff
    layer = viewer.add_points(coords, name=name, features=features, **kwargs)
    layer.current_size = size
    layer.size = size
    disconnect = install_points_style_sync(layer)
    setattr(
        viewer,
        "_nvitk_hotspot_points_state",
        HotspotPointsState(layer=layer, disconnect_style_sync=disconnect),
    )
    return layer


def _subsample_mask_indices(mask: np.ndarray, max_points: int, label_ids: list[int] | None) -> np.ndarray:
    """Voxel indices inside the ROI defined by *mask*/*label_ids*, evenly subsampled down to at most
    *max_points*."""
    roi = _roi_mask(mask, label_ids)
    idx = np.argwhere(roi)
    if idx.shape[0] <= max_points:
        return idx
    step = max(1, idx.shape[0] // int(max_points))
    return idx[::step][: int(max_points)]


def _time_axis_index_from_layer(layer: Any) -> int:
    """Array axis index for cardiac phase (``T`` / ``C``) on a 4D phase layer."""
    from nvitk.gui.core.orientation import _axes_string_from_layer

    axes = (_axes_string_from_layer(layer) or "").upper()
    nd = int(getattr(getattr(layer, "data", None), "ndim", 0) or 0)
    if len(axes) == nd:
        if "T" in axes:
            return int(axes.index("T"))
        if "C" in axes:
            return int(axes.index("C"))
    return max(0, nd - 1)


def _move_time_axis_last(*arrays: np.ndarray, time_axis: int) -> tuple[np.ndarray, ...]:
    """Ensure the cardiac-phase axis is last (required by velocity_mm_s_from_phases)."""
    out = []
    for arr in arrays:
        a = to_numpy(arr)
        if int(time_axis) != a.ndim - 1 and a.ndim >= 4:
            a = np.moveaxis(a, int(time_axis), -1)
        out.append(a)
    return tuple(out)


def _displacement_from_velocity(
    velocities_mm_s: np.ndarray,
    *,
    max_arrow_voxels: float,
    speed_percentile = 95.0,
) -> np.ndarray:
    """Unit direction × capped length; color still uses true speed in features."""
    vel = to_numpy(velocities_mm_s).astype(np.float64)
    if vel.size == 0:
        return vel
    mag = np.linalg.norm(vel, axis=1, keepdims=True)
    mag_safe = np.maximum(mag, 1e-9)
    direction = vel / mag_safe
    ref = float(np.percentile(mag.ravel(), float(speed_percentile)))
    ref = max(ref, 1e-6)
    rel = np.clip(mag / ref, 0.0, 1.0)
    return (direction * rel * float(max_arrow_voxels)).astype(np.float64)


def flow_vector_frame(
    cache: FlowVectorCache,
    time_index: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build Napari vectors data (N, 2, 3) and features for one cardiac phase."""
    t = int(np.clip(time_index, 0, cache.n_time - 1))
    pos = cache.positions.astype(np.float64, copy=False)
    vel = cache.velocities[t].astype(np.float64, copy=False)
    mag = cache.magnitudes[t].astype(np.float64, copy=False)
    disp = _displacement_from_velocity(
        vel,
        max_arrow_voxels=cache.max_arrow_voxels,
        speed_percentile=cache.speed_percentile,
    )
    # Napari Vectors: row 0 = tail, row 1 = projection (not endpoint); end = tail + length * proj.
    data = np.stack([pos, disp], axis=1)
    features = {
        "speed": mag,
        "vx_mm_s": vel[:, 0],
        "vy_mm_s": vel[:, 1],
        "vz_mm_s": vel[:, 2],
        "phase": np.full(mag.shape[0], t, dtype=np.int32),
    }
    return data, features


def flow_vectors_all_times(
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    *,
    phase_layer = None,
    label_ids = None,
    max_points = 4000,
    max_arrow_voxels = 5.0,
    speed_percentile = 95.0,
) -> FlowVectorCache:
    """Precompute subsampled velocity glyphs for every cardiac phase."""
    time_axis = _time_axis_index_from_layer(phase_layer) if phase_layer is not None else -1
    ap_np, rl_np, fh_np = _move_time_axis_last(ap, rl, fh, time_axis=time_axis)
    ap_a = as_backend_array(ap_np).astype(np.float64)
    rl_a = as_backend_array(rl_np).astype(np.float64)
    fh_a = as_backend_array(fh_np).astype(np.float64)
    mask_np = to_numpy(mask)
    if mask_np.ndim > 3:
        raise ValueError("Mask must be a 3D spatial volume for flow vectors.")
    spatial_shape = tuple(int(x) for x in ap_a.shape[:3])
    if tuple(mask_np.shape) != spatial_shape:
        raise ValueError(
            f"Mask shape {tuple(mask_np.shape)} does not match AP spatial shape {spatial_shape}; "
            "align the mask to the phase grid first."
        )
    vx, vy, vz = velocity_mm_s_from_phases(ap_a, rl_a, fh_a)
    n_time = int(vx.shape[3])
    idx = _subsample_mask_indices(mask, max_points, label_ids)
    if idx.size == 0:
        return FlowVectorCache(
            positions=np.zeros((0, 3), dtype=np.float64),
            velocities=np.zeros((n_time, 0, 3), dtype=np.float64),
            magnitudes=np.zeros((n_time, 0), dtype=np.float64),
            n_time=n_time,
            max_arrow_voxels=float(max_arrow_voxels),
            speed_percentile=float(speed_percentile),
        )

    pos = idx.astype(np.float64)
    n_pts = int(pos.shape[0])
    velocities = np.zeros((n_time, n_pts, 3), dtype=np.float64)
    magnitudes = np.zeros((n_time, n_pts), dtype=np.float64)
    ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]
    for t in range(n_time):
        vec = np.stack(
            [
                to_numpy(vx[ii, jj, kk, t]),
                to_numpy(vy[ii, jj, kk, t]),
                to_numpy(vz[ii, jj, kk, t]),
            ],
            axis=1,
        ).astype(np.float64)
        velocities[t] = vec
        magnitudes[t] = np.linalg.norm(vec, axis=1)

    return FlowVectorCache(
        positions=pos,
        velocities=velocities,
        magnitudes=magnitudes,
        n_time=n_time,
        max_arrow_voxels=float(max_arrow_voxels),
        speed_percentile=float(speed_percentile),
    )


def flow_vectors_at_time(
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    time_index: int,
    *,
    phase_layer = None,
    label_ids = None,
    max_points = 4000,
    max_arrow_voxels = 5.0,
    speed_percentile = 95.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions Nx3, displacement Nx3, speed N) for one cardiac phase."""
    cache = flow_vectors_all_times(
        ap,
        rl,
        fh,
        mask,
        phase_layer=phase_layer,
        label_ids=label_ids,
        max_points=max_points,
        max_arrow_voxels=max_arrow_voxels,
        speed_percentile=speed_percentile,
    )
    t = int(np.clip(time_index, 0, max(0, cache.n_time - 1)))
    if cache.positions.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    vel = cache.velocities[t]
    disp = _displacement_from_velocity(
        vel,
        max_arrow_voxels=cache.max_arrow_voxels,
        speed_percentile=cache.speed_percentile,
    )
    return cache.positions, disp, cache.magnitudes[t]


def add_flow_vectors_layer(
    viewer: Any,
    positions: np.ndarray,
    vectors: np.ndarray,
    *,
    reference_layer: Any,
    name = FLOW_VECTORS_LAYER,
    features = None,
    speed_limits = None,
    colormap = "turbo",
) -> Any:
    """Add (replacing any prior) a Vectors layer for the given arrow *positions*/*vectors*, colored by
    speed via *colormap* if a ``"speed"`` feature is supplied."""
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)
    kwargs = {
        "name": name,
        "vector_style": "arrow",
        "edge_width": DEFAULT_FLOW_EDGE_WIDTH,
        "length": 1.0,
    }
    if features and "speed" in features:
        kwargs["features"] = features
        kwargs["edge_color"] = "speed"
        kwargs["edge_colormap"] = colormap
        speeds = to_numpy(features["speed"]).astype(np.float64)
        if speed_limits is not None:
            kwargs["edge_contrast_limits"] = speed_limits
        elif speeds.size:
            lo = float(np.percentile(speeds, 2.0))
            hi = float(np.percentile(speeds, 98.0))
            kwargs["edge_contrast_limits"] = (lo, hi if hi > lo else lo + 1.0)
    aff = layer_affine(reference_layer)
    if aff is not None:
        kwargs["affine"] = aff
    data = np.stack([positions, vectors], axis=1)
    return viewer.add_vectors(data, **kwargs)


def _time_index_from_viewer(viewer: Any, phase_layer: Any, n_time: int) -> int:
    """Read cardiac phase from the 4D phase layer dims (not the vectors layer)."""
    t_ax = _time_axis_index_from_layer(phase_layer)
    steps = tuple(int(x) for x in viewer.dims.current_step)
    if t_ax < len(steps):
        return int(np.clip(steps[t_ax], 0, max(0, n_time - 1)))
    return 0


def _phase_dim_steps(viewer: Any, phase_layer: Any) -> list[int]:
    """Current integer slice indices for every axis of *phase_layer*."""
    ndim = int(phase_layer.data.ndim)
    shape = tuple(int(x) for x in phase_layer.data.shape)
    steps = list(int(round(float(x))) for x in viewer.dims.current_step)
    if len(steps) < ndim:
        steps.extend([0] * (ndim - len(steps)))
    return [int(np.clip(steps[i], 0, max(0, shape[i] - 1))) for i in range(ndim)]


def _install_phase_dims(viewer: Any, phase_layer: Any) -> None:
    """One-time 4D dims setup after adding vectors: ranges/labels, keep camera + scrubber."""
    from napari.components.dims import RangeTuple
    from nvitk.gui.core.orientation import (
        _axes_string_from_layer,
        _layer_display_scale,
        ensure_4d_scale_only_layer,
        napari_dim_order,
    )

    if getattr(phase_layer, "data", None) is None or int(phase_layer.data.ndim) <= 3:
        return

    ensure_4d_scale_only_layer(phase_layer)
    ndim = int(phase_layer.data.ndim)
    shape = tuple(int(x) for x in phase_layer.data.shape)
    axes_str = _axes_string_from_layer(phase_layer)
    steps = _phase_dim_steps(viewer, phase_layer)
    sc = _layer_display_scale(phase_layer, ndim)
    ranges = tuple(
        RangeTuple(0.0, float(max(0, n - 1)) * sc[i], sc[i]) for i, n in enumerate(shape)
    )

    viewer.dims.ndim = max(int(viewer.dims.ndim), ndim)
    full_range = list(viewer.dims.range)
    if len(full_range) < ndim:
        full_range.extend([RangeTuple(0.0, 0.0, 1.0)] * (ndim - len(full_range)))
    for i in range(ndim):
        full_range[i] = ranges[i]
    viewer.dims.range = tuple(full_range[: viewer.dims.ndim])

    viewer.dims.order = napari_dim_order(axes_str, layer_affine(phase_layer), ndim)
    if axes_str and len(axes_str) == ndim:
        viewer.dims.axis_labels = tuple(axes_str)
    viewer.dims.current_step = tuple(steps[: viewer.dims.ndim])


def _repair_time_dim_range(viewer: Any, phase_layer: Any) -> None:
    """Fix cardiac-phase slider extent if vector layer updates polluted dims.range."""
    from napari.components.dims import RangeTuple
    from nvitk.gui.core.orientation import _layer_display_scale, ensure_4d_scale_only_layer

    if getattr(phase_layer, "data", None) is None or int(phase_layer.data.ndim) <= 3:
        return
    t_ax = _time_axis_index_from_layer(phase_layer)
    n_time = int(phase_layer.data.shape[t_ax])
    if t_ax >= len(viewer.dims.range):
        return
    ensure_4d_scale_only_layer(phase_layer)
    sc = _layer_display_scale(phase_layer, int(phase_layer.data.ndim))
    step = float(sc[t_ax])
    expected_stop = float(max(0, n_time - 1)) * step
    rng = viewer.dims.range[t_ax]
    if abs(float(rng.stop) - expected_stop) <= max(step, 1e-6):
        return
    full = list(viewer.dims.range)
    full[t_ax] = RangeTuple(0.0, expected_stop, step)
    viewer.dims.range = tuple(full)


def _topmost_phase_layer(viewer: Any) -> Any | None:
    """Topmost 4D+ Image/Labels layer (cardiac-phase volume), if any."""
    for lyr in reversed(list(getattr(viewer, "layers", []) or [])):
        if type(lyr).__name__ not in ("Image", "Labels"):
            continue
        data = getattr(lyr, "data", None)
        if int(getattr(data, "ndim", 0) or 0) > 3:
            return lyr
    return None


def repair_time_dim_for_viewer(viewer: Any) -> None:
    """Force the cardiac-phase slider to the true phase count.

    Adding 3D overlays/volumes next to a 4D ``XYZT`` layer makes Napari right-align
    their spatial axes onto the 4D time world-axis, inflating the slider to thousands
    of steps. Re-applying the phase layer's own time extent restores the real count.
    """
    phase_layer = _topmost_phase_layer(viewer)
    if phase_layer is None:
        return
    try:
        _repair_time_dim_range(viewer, phase_layer)
    except Exception:
        pass


def _global_speed_limits(cache: FlowVectorCache) -> tuple[float, float]:
    """2nd/98th percentile speed magnitude across all cached cardiac phases, for fixed-range coloring."""
    if cache.magnitudes.size == 0:
        return (0.0, 1.0)
    lo = float(np.percentile(cache.magnitudes, 2.0))
    hi = float(np.percentile(cache.magnitudes, 98.0))
    return (lo, hi if hi > lo else lo + 1.0)


@dataclass
class FlowVectorPlayback:
    """Dims-synced flow vector layer (updates with Napari time slider / play bar)."""

    cache: FlowVectorCache
    phase_layer: Any
    layer: Any
    dims_callback: Callable[[Any], None]


def stop_flow_vector_playback(viewer: Any) -> None:
    """Disconnect dims hooks from a prior flow-vector overlay."""
    state = getattr(viewer, "_nvitk_flow_vector_state", None)
    if state is None:
        return
    try:
        viewer.dims.events.current_step.disconnect(state.dims_callback)
    except Exception:
        pass
    setattr(viewer, "_nvitk_flow_vector_state", None)


def _update_flow_vector_layer(
    layer: Any,
    cache: FlowVectorCache,
    time_index: int,
    *,
    viewer = None,
    phase_layer = None,
) -> None:
    """Recompute and push the flow-vector geometry/features for *time_index* onto *layer*, repairing
    the viewer's time-dim range against *phase_layer* if both are given."""
    data, features = flow_vector_frame(cache, time_index)
    layer.data = data
    layer.features = features
    if viewer is not None and phase_layer is not None:
        _repair_time_dim_range(viewer, phase_layer)


def add_animated_flow_vectors_layer(
    viewer: Any,
    cache: FlowVectorCache,
    *,
    phase_layer: Any,
    spatial_reference_layer = None,
    name = FLOW_VECTORS_LAYER,
    initial_time = 0,
    sync_dims = True,
    colormap = "turbo",
) -> FlowVectorPlayback:
    """Add flow vectors; sync glyph data to the Napari dims slider / play bar."""
    stop_flow_vector_playback(viewer)

    spatial_ref = spatial_reference_layer or phase_layer
    t0 = int(np.clip(initial_time, 0, max(0, cache.n_time - 1)))
    data, features = flow_vector_frame(cache, t0)
    limits = _global_speed_limits(cache)

    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)

    kwargs = {
        "name": name,
        "vector_style": "arrow",
        "edge_width": DEFAULT_FLOW_EDGE_WIDTH,
        "length": 1.0,
        "features": features,
        "edge_color": "speed",
        "edge_colormap": colormap,
        "edge_contrast_limits": limits,
    }
    aff = layer_affine(spatial_ref)
    if aff is not None:
        kwargs["affine"] = aff
    layer = viewer.add_vectors(data, **kwargs)

    _install_phase_dims(viewer, phase_layer)
    try:
        viewer.layers.selection.active = phase_layer
    except Exception:
        pass

    def _on_dims(_event: Any = None) -> None:
        """Refresh the flow-vector layer for the viewer's current cardiac phase, when visible."""
        if not getattr(layer, "visible", True):
            return
        _repair_time_dim_range(viewer, phase_layer)
        t = _time_index_from_viewer(viewer, phase_layer, cache.n_time)
        _update_flow_vector_layer(layer, cache, t, viewer=viewer, phase_layer=phase_layer)

    if sync_dims:
        viewer.dims.events.current_step.connect(_on_dims)

    playback = FlowVectorPlayback(
        cache=cache,
        phase_layer=phase_layer,
        layer=layer,
        dims_callback=_on_dims,
    )
    setattr(viewer, "_nvitk_flow_vector_state", playback)
    return playback


def voxel_spacing_from_layer(layer: Any) -> tuple[float, float, float]:
    """*layer*'s (x, y, z) voxel spacing, or ``(1.0, 1.0, 1.0)`` if unavailable."""
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)
