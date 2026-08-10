"""Napari vessel cross-section viewer (centerline pick + oblique 2D panel)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.measure.cross_section import cross_section_at_loc, cross_section_at_point, masked_plane_velocity_series
from nvitk.measure.hemodynamics import flow_pulsatile_ml_s, mean_flow_ml_min
from nvitk.morphology import compute_centerlines
from nvitk.morphology.centerline import centerline_tangents
from nvitk.viz.centerline_pick import (
    CenterlinePick,
    choose_plane_normal_sense,
    frame_from_tangent,
    pick_centerline,
    refine_pick_to_vertex_if_closer,
    smooth_polyline_display,
    tangent_from_centerline,
    tangent_window_indices,
    unit_vector,
)

from nvitk.gui.core.spatial import (
    layer_spatial_kwargs,
    layer_spacing,
    world_to_data_coords,
)
from nvitk.gui.viz.layers import DEFAULT_FLOW_EDGE_WIDTH
from nvitk.gui.viz.cross_section_panel import CrossSectionPanel, attach_cross_section_dock
from nvitk.gui.viz.loc_points import (
    LOC_SNAP_DISTANCE_VOX,
    LocPose,
    nearest_loc_pose,
    parse_loc_poses,
)

XS_CENTERLINES = "Vessel centerlines (xs)"
XS_CL_POINTS = "Centerline points (xs)"
XS_PICK = "Cross-section pick"
XS_PLANE = "Cross-section plane (xs)"
XS_NORMAL = "Cross-section normal (xs)"
XS_TANGENT = "Tangent segment (xs)"
XS_SEG = "Vessel seg (xs)"
XS_OVERLAY_META = "nvitk_vessel_xs_overlay"

_XS_COLOR_SELECTED = "red"
_XS_COLOR_TANGENT_NEIGHBOR = "lime"

_TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _remove_layers_named(viewer: Any, names: set[str]) -> None:
    """Remove every layer in *viewer* whose name is in *names*."""
    for lyr in list(viewer.layers):
        if lyr.name in names:
            viewer.layers.remove(lyr)


def _intensity_volume(layer: Any) -> np.ndarray:
    """*layer*'s data as a float64 host array, taking the magnitude of complex data."""
    data = to_numpy(layer.data)
    if np.iscomplexobj(data):
        data = np.abs(data)
    return np.asarray(data, dtype=np.float64)


def _overlay_metadata() -> dict[str, Any]:
    """Layer metadata tag marking a layer as a vessel cross-section overlay (for cleanup)."""
    return {XS_OVERLAY_META: True}


def _overlay_spatial_kwargs(reference_layer: Any) -> dict[str, Any]:
    """Spatial (scale/affine) kwargs to align an overlay layer with *reference_layer*."""
    return layer_spatial_kwargs(reference_layer)


def _configure_overlay_points_layer(layer: Any) -> None:
    """Make a Points overlay layer non-editable and lock its interaction mode to pan/zoom."""
    try:
        layer.editable = False
    except Exception:
        pass
    try:
        layer.mode = "pan_zoom"
    except Exception:
        pass


def _configure_overlay_shapes_layer(layer: Any) -> None:
    """Make a Shapes overlay layer non-editable."""
    try:
        layer.editable = False
    except Exception:
        pass


def _is_left_mouse_button(event: Any) -> bool:
    """True if the mouse *event* was triggered by the left button (or its identity is unknown)."""
    btn = getattr(event, "button", None)
    if btn in (0, 1, None):
        return True
    name = str(btn).lower()
    return name in ("left", "lbutton", "mouse1")


def _overlay_edge_width(layer: Any) -> float:
    """Line-overlay edge width scaled to *layer*'s finest voxel spacing (or a default if unavailable)."""
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return max(0.12, float(min(sp)) * 0.35)
    return DEFAULT_FLOW_EDGE_WIDTH


def _overlay_point_size(layer: Any) -> float:
    """Point-overlay marker size scaled to *layer*'s finest voxel spacing (or 1.0 if unavailable)."""
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return max(0.4, float(min(sp)) * 1.2)
    return 1.0


def _connect_pick_callback(target: Any, callback: Any) -> None:
    """Register *callback* as the first mouse-drag handler on *target* (viewer or layer)."""
    try:
        target.mouse_drag_callbacks.insert(0, callback)
    except Exception:
        target.mouse_drag_callbacks.append(callback)


def _disconnect_pick_callback(target: Any, callback: Any) -> None:
    """Remove *callback* from *target*'s mouse-drag handlers, if present."""
    if target is None or callback is None:
        return
    try:
        if callback in target.mouse_drag_callbacks:
            target.mouse_drag_callbacks.remove(callback)
    except Exception:
        pass


def _plane_square_corners(
    center: np.ndarray,
    tangent: np.ndarray,
    radius_vox: float,
) -> np.ndarray:
    """Four corner points of a square patch centered at *center*, spanning the plane perpendicular to
    *tangent* with half-width *radius_vox*, for drawing the cross-section plane overlay."""
    from nvitk.measure.cross_section import plane_basis_from_tangent

    u, v = plane_basis_from_tangent(tangent)
    c = to_numpy(center).astype(np.float64).reshape(3)
    r = float(radius_vox)
    corners = []
    for du, dv in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        corners.append(c + du * r * u + dv * r * v)
    return np.asarray(corners, dtype=np.float32)


def _pick_max_distance_vox(params: dict[str, Any]) -> float:
    """3D snap radius (vox) for fallback when view-line pick misses."""
    return max(3.5, float(params.get("radius_vox", 12.0)) * 0.45)


def _pick_max_ray_distance_vox(params: dict[str, Any]) -> float:
    """Perpendicular tolerance (vox) from view line to centerline."""
    return max(5.0, float(params.get("radius_vox", 12.0)) * 0.65)


def _pick_max_anchor_distance_vox(params: dict[str, Any]) -> float:
    """In-plane distance (vox) from click to snap, ignoring depth along the view."""
    return max(6.0, float(params.get("radius_vox", 12.0)) * 0.85)


def _normal_display_half_length_vox(layer: Any, radius_vox: float) -> float:
    """Half-length (voxels) of the drawn normal-direction segment, scaled to *layer*'s spacing."""
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return max(0.8, float(min(sp)) * 1.8)
    return max(1.0, float(radius_vox) * 0.35)


def _tangent_display_half_length_vox(layer: Any, radius_vox: float) -> float:
    """Half-length (voxels) of the drawn tangent-direction segment (longer than the normal segment)."""
    return max(2.5, _normal_display_half_length_vox(layer, radius_vox) * 2.5)


def _normal_half_line_data(
    center_vox: np.ndarray,
    tangent: np.ndarray,
    *,
    half_length_vox: float,
) -> np.ndarray:
    """Single segment along +normal in layer data (voxel) coordinates."""
    c = to_numpy(center_vox).astype(np.float64).reshape(3)
    t = unit_vector(to_numpy(tangent).astype(np.float64).reshape(3))
    half_len = float(half_length_vox)
    return np.stack([c, c + half_len * t], axis=0).astype(np.float32)


def _arrow_chevrons(
    tip: np.ndarray,
    direction: np.ndarray,
    *,
    size_vox: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Two short paths forming an arrowhead at *tip* pointing along *direction*."""
    t = unit_vector(to_numpy(direction).astype(np.float64).reshape(3))
    u, _v = frame_from_tangent(t)
    tip = to_numpy(tip).astype(np.float64).reshape(3)
    size = float(size_vox)
    base = tip - size * t
    wing = size * 0.42
    left = base + wing * (u + 0.35 * t)
    right = base + wing * (-u + 0.35 * t)
    return (
        np.stack([tip, left], axis=0).astype(np.float32),
        np.stack([tip, right], axis=0).astype(np.float32),
    )


def _tangent_display_paths_data(
    center_vox: np.ndarray,
    tangent: np.ndarray,
    *,
    reference_layer: Any,
    radius_vox: float,
) -> list[np.ndarray]:
    """Extended tangent line plus arrowheads at both ends (data coordinates)."""
    c = to_numpy(center_vox).astype(np.float64).reshape(3)
    t = unit_vector(to_numpy(tangent).astype(np.float64).reshape(3))
    half_len = _tangent_display_half_length_vox(reference_layer, radius_vox)
    arrow_size = max(0.6, half_len * 0.22)
    start = c - half_len * t
    end = c + half_len * t
    main = np.stack([start, end], axis=0).astype(np.float32)
    a1, a2 = _arrow_chevrons(end, t, size_vox=arrow_size)
    b1, b2 = _arrow_chevrons(start, -t, size_vox=arrow_size)
    return [main, a1, a2, b1, b2]


def _centerlines_layer_visible(centerlines_layer: Any | None) -> bool:
    """True if *centerlines_layer* exists and is currently visible."""
    if centerlines_layer is None:
        return False
    return bool(getattr(centerlines_layer, "visible", True))


def _voxel_spacing(layer: Any) -> tuple[float, float, float]:
    """*layer*'s (x, y, z) voxel spacing, or ``(1.0, 1.0, 1.0)`` if unavailable."""
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)


def _params_from_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw tool-panel *params* dict into the typed cross-section computation parameters
    (radius, resolution, interpolation, resegment/supersampling flags, threshold algorithm)."""
    resegment = bool(params.get("measure_resegment", False))
    supersampling = bool(params.get("cs_supersampling", True))
    # Keep interp_vals for supersampled grid even when not resegmenting, so the
    # stage-4 mask is nearest-neighbor upsampled onto the finer plane.
    default_interp = 4 if (resegment or supersampling) else 1
    # Match stage-6 LOC measurement: linear plane interpolation for velocity sampling
    # (``cross_section_plane_interp=1``). Previously this was forced to 0 whenever
    # ``measure_resegment`` was off, which systematically shifted mean flow vs the CSV.
    plane_order = params.get("cross_section_plane_interp", params.get("plane_interp_order"))
    try:
        plane_interp_order = int(plane_order) if plane_order is not None else 1
    except (TypeError, ValueError):
        plane_interp_order = 1
    if plane_interp_order not in (0, 1):
        plane_interp_order = 1
    return {
        "radius_vox": float(params.get("cross_section_radius_vox") or 12.0),
        "cross_section_res": int(params.get("cross_section_res") or 0),
        "interp_vals": int(params.get("interp_vals") or default_interp),
        "plane_interp_order": plane_interp_order,
        "measure_resegment": resegment,
        "cs_supersampling": supersampling,
        "thr_algorithm": str(params.get("thr_algorithm") or "lsthr"),
        "centerline_window": int(str(params.get("centerline_window") or "5")),
        "show_segmentation_3d": bool(params.get("show_segmentation_3d", True)),
        "loc_snap_distance_vox": float(
            params.get("loc_snap_distance_vox") or LOC_SNAP_DISTANCE_VOX
        ),
    }


def _label_color_map(centerlines: dict[int, np.ndarray]) -> dict[int, str]:
    """Assign a stable Tab10 hex color to each centerline label id, in sorted-id order."""
    return {
        int(lbl): _TAB10[i % len(_TAB10)]
        for i, lbl in enumerate(sorted(centerlines.keys()))
    }


def centerlines_dict_from_arterial_branches(
    arterial: dict[int, list[tuple[str, Any]]],
    *,
    min_points: int = 3,
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, str]]:
    """Flatten stage-4/6 named branches into pickable centerline keys.

    Returns
    -------
    centerlines
        Synthetic int key → polyline (exact stage polylines, not re-extracted).
    volume_label_by_key
        Synthetic key → parent qvtpy seg label (for plane label constraints).
    branch_name_by_key
        Synthetic key → branch name (``LMCA-M2a``, …).
    """
    centerlines: dict[int, np.ndarray] = {}
    volume_label_by_key: dict[int, int] = {}
    branch_name_by_key: dict[int, str] = {}
    key = 1
    for parent_lid in sorted(int(k) for k in arterial.keys()):
        for name, pts in arterial.get(parent_lid) or []:
            arr = to_numpy(pts).astype(np.float32, copy=False).reshape(-1, 3)
            if arr.shape[0] < int(min_points):
                continue
            centerlines[key] = arr
            volume_label_by_key[key] = int(parent_lid)
            branch_name_by_key[key] = str(name)
            key += 1
    return centerlines, volume_label_by_key, branch_name_by_key


def append_venous_centerlines(
    centerlines: dict[int, np.ndarray],
    volume_label_by_key: dict[int, int],
    branch_name_by_key: dict[int, str],
    venous: dict[str, Any] | None,
    *,
    venous_label_by_name: dict[str, int] | None = None,
    min_points: int = 3,
) -> int:
    """Append named venous polylines onto an existing XS centerline map.

    Returns the number of venous vessels added.
    """
    if not venous:
        return 0
    from nvitk.pipes.qvtpy.util.centerline.venous_heuristics import venous_name_to_label_id

    key = (max(centerlines.keys()) + 1) if centerlines else 1
    n_added = 0
    for name in sorted(str(n) for n in venous.keys()):
        pts = venous.get(name)
        if pts is None:
            continue
        arr = to_numpy(pts).astype(np.float32, copy=False).reshape(-1, 3)
        if arr.shape[0] < int(min_points):
            continue
        lid = int(venous_name_to_label_id(name, venous_label_by_name))
        centerlines[key] = arr
        volume_label_by_key[key] = lid
        branch_name_by_key[key] = str(name)
        key += 1
        n_added += 1
    return n_added


def build_centerlines_dict(
    centerline_mask: np.ndarray,
    *,
    segmentation: np.ndarray | None = None,
    min_points: int = 5,
) -> dict[int, np.ndarray]:
    """Ordered polylines per label from centerline (+ optional seg) masks.

    Fallback when stage-4 ``centerlines_seg_branches.json`` is unavailable.
    Prefer :func:`centerlines_dict_from_arterial_branches` so overlays match
    stage 4/6 bifurcation geometry.
    """
    cl = to_numpy(centerline_mask)
    labels = sorted(int(v) for v in np.unique(cl) if int(v) != 0)
    if not labels:
        return {}
    if segmentation is not None:
        seg = to_numpy(segmentation).astype(np.int32, copy=False)
        return compute_centerlines(
            seg,
            centerline_mask=cl,
            labels=labels,
            min_points=min_points,
        )
    out: dict[int, np.ndarray] = {}
    for lbl in labels:
        roi = np.zeros(cl.shape, dtype=np.int32)
        roi[cl == int(lbl)] = int(lbl)
        part = compute_centerlines(
            roi,
            centerline_mask=cl,
            labels=[int(lbl)],
            min_points=min_points,
        )
        if int(lbl) in part:
            out[int(lbl)] = part[int(lbl)]
    return out


def _stack_centerline_points(
    centerlines: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 3) points and parallel vessel label id per row."""
    pts_list: list[np.ndarray] = []
    labels: list[int] = []
    for lbl in sorted(centerlines.keys()):
        pts = centerlines[lbl]
        if pts is None or pts.shape[0] == 0:
            continue
        for row in np.asarray(pts, dtype=np.float32):
            pts_list.append(row)
            labels.append(int(lbl))
    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.stack(pts_list, axis=0), np.asarray(labels, dtype=np.int32)


def _default_centerline_point_colors(
    point_labels: np.ndarray,
    label_colors: dict[int, str],
) -> list[str]:
    """Per-point face colors from each point's centerline label id."""
    return [label_colors[int(lbl)] for lbl in point_labels]


def _centerline_point_colors_for_pick(
    point_labels: np.ndarray,
    pick: CenterlinePick,
    centerlines: dict[int, np.ndarray],
    *,
    window: int,
    label_colors: dict[int, str],
) -> list[str]:
    """Per-point colors highlighting the picked point and its tangent-window neighbors on top of the
    default per-label coloring."""
    colors = _default_centerline_point_colors(point_labels, label_colors)
    pts = centerlines.get(int(pick.label))
    if pts is None:
        return colors
    a, b = tangent_window_indices(pts, pick.index, window=window)
    local_idx = 0
    for row, lbl in enumerate(point_labels):
        if int(lbl) != int(pick.label):
            continue
        if local_idx == int(pick.index):
            colors[row] = _XS_COLOR_SELECTED
        elif a <= local_idx <= b:
            colors[row] = _XS_COLOR_TANGENT_NEIGHBOR
        local_idx += 1
    return colors


def _add_centerline_points_layer(
    viewer: Any,
    centerlines: dict[int, np.ndarray],
    *,
    reference_layer: Any,
) -> tuple[Any, np.ndarray, dict[int, str]] | tuple[None, np.ndarray, dict[int, str]]:
    """Add a non-editable Points overlay for every centerline point, colored by vessel label; returns
    ``(layer_or_None, point_labels, label_colors)``."""
    stacked, point_labels = _stack_centerline_points(centerlines)
    label_colors = _label_color_map(centerlines)
    if stacked.shape[0] == 0:
        return None, point_labels, label_colors
    kwargs = {
        "name": XS_CL_POINTS,
        "size": max(0.25, _overlay_point_size(reference_layer) * 0.55),
        "face_color": _default_centerline_point_colors(point_labels, label_colors),
        "symbol": "disc",
        "metadata": _overlay_metadata(),
    }
    kwargs.update(_overlay_spatial_kwargs(reference_layer))
    layer = viewer.add_points(stacked.astype(np.float64), **kwargs)
    _configure_overlay_points_layer(layer)
    return layer, point_labels, label_colors


def _add_centerline_paths(
    viewer: Any,
    centerlines: dict[int, np.ndarray],
    *,
    reference_layer: Any,
    smooth_display: bool = True,
) -> Any:
    """Add a non-editable Shapes path overlay for every centerline (one path per label, color-coded),
    optionally smoothing the display polyline; returns the layer or ``None`` if there's nothing to draw."""
    paths = []
    colors = []
    for i, lbl in enumerate(sorted(centerlines.keys())):
        pts = centerlines[lbl]
        if pts is None or pts.shape[0] < 2:
            continue
        disp = (
            smooth_polyline_display(pts)
            if smooth_display
            else np.asarray(pts, dtype=np.float32)
        )
        paths.append(disp.astype(np.float32))
        colors.append(_TAB10[i % len(_TAB10)])
    if not paths:
        return None
    kwargs = {
        "edge_width": _overlay_edge_width(reference_layer),
        "opacity": 0.9,
        "name": XS_CENTERLINES,
        "metadata": _overlay_metadata(),
    }
    kwargs.update(_overlay_spatial_kwargs(reference_layer))
    try:
        layer = viewer.add_shapes(
            paths,
            shape_type="path",
            edge_color=colors,
            **kwargs,
        )
        _configure_overlay_shapes_layer(layer)
        return layer
    except TypeError:
        kwargs.pop("edge_color", None)
        return viewer.add_shapes(paths, shape_type="path", **kwargs)


def _add_seg_labels(
    viewer: Any,
    seg: np.ndarray,
    *,
    reference_layer: Any,
) -> Any:
    """Add *seg* as a translucent Labels overlay, aligned to *reference_layer*'s spatial metadata."""
    kwargs = {"name": XS_SEG, "opacity": 0.25}
    kwargs.update(layer_spatial_kwargs(reference_layer))
    return viewer.add_labels(seg.astype(np.int32), **kwargs)


def shutdown_vessel_cross_sections(app_state: dict[str, Any]) -> None:
    """Disconnect pick callback and drop dock reference."""
    session = app_state.pop("vessel_xs", None)
    if not isinstance(session, dict):
        return
    viewer = session.get("viewer")
    cb = session.get("callback")
    intensity_layer = session.get("intensity_layer")
    if viewer is not None and cb is not None:
        _disconnect_pick_callback(viewer, cb)
        _disconnect_pick_callback(intensity_layer, cb)
    prev_ndisplay = session.get("prev_ndisplay")
    if viewer is not None and prev_ndisplay is not None:
        try:
            viewer.dims.ndisplay = int(prev_ndisplay)
        except Exception:
            pass
    dock = session.get("dock")
    if dock is not None:
        try:
            dock.close()
        except Exception:
            pass


def _resolve_loc_centerline(
    loc: LocPose,
    centerlines: dict[int, np.ndarray],
    branch_name_by_key: Mapping[int, str],
    volume_label_by_key: Mapping[int, int],
) -> tuple[int, int] | None:
    """Best ``(centerline_key, index)`` for *loc*: name match first, then vessel_id, then nearest."""
    name_u = loc.vessel_name.upper()
    candidates: list[int] = []
    for key, name in branch_name_by_key.items():
        nu = str(name).upper()
        if nu == name_u or nu.startswith(name_u) or name_u.startswith(nu):
            candidates.append(int(key))
    if not candidates and loc.vessel_id:
        for key, lid in volume_label_by_key.items():
            if int(lid) == int(loc.vessel_id):
                candidates.append(int(key))
    if not candidates:
        candidates = [int(k) for k in centerlines.keys()]

    best: tuple[int, int, float] | None = None
    for key in candidates:
        pts = centerlines.get(int(key))
        if pts is None or pts.shape[0] < 1:
            continue
        d2 = np.sum((pts.astype(np.float64) - loc.center.reshape(1, 3)) ** 2, axis=1)
        idx = int(np.argmin(d2))
        dist = float(d2[idx])
        if loc.centerline_index is not None and 0 <= int(loc.centerline_index) < pts.shape[0]:
            # Prefer the recorded index when it is close to the LOC center.
            alt = int(loc.centerline_index)
            if float(d2[alt]) <= dist * 1.25 + 1.0:
                idx, dist = alt, float(d2[alt])
        if best is None or dist < best[2]:
            best = (int(key), idx, dist)
    if best is None:
        return None
    return best[0], best[1]


def _flow_at_pose(
    center: np.ndarray,
    tangent: np.ndarray,
    *,
    state: dict[str, Any],
    volume_label_id: int,
    offset: int,
    index: int,
) -> dict[str, Any] | None:
    """One flow-waveform entry at an exact center/tangent pose (stage-6–compatible sampling)."""
    vx = state.get("vx")
    vy = state.get("vy")
    vz = state.get("vz")
    if vx is None or vy is None or vz is None:
        return None
    p = state["params"]
    cd = state["cd"]
    mag = state.get("mag", cd)
    vel_mag = state.get("vel_mag", cd)
    try:
        cs = cross_section_at_loc(
            center,
            tangent,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=state["voxel_spacing"],
            radius_vox=p["radius_vox"],
            interp_vals=p["interp_vals"],
            cross_section_res=p["cross_section_res"],
            plane_interp_order=p["plane_interp_order"],
            cs_supersampling=p.get("cs_supersampling", False),
            measure_resegment=p["measure_resegment"],
            thr_algorithm=p["thr_algorithm"],  # type: ignore[arg-type]
            volume_seg=state.get("segmentation"),
            volume_label_id=int(volume_label_id),
            label_constrain=not p["measure_resegment"],
        )
    except Exception:
        return None
    if cs.area_mm2 <= 0.0 or not bool(np.any(cs.mask_2d)):
        return None
    vel_ts = masked_plane_velocity_series(
        vx, vy, vz, cs, plane_interp_order=p["plane_interp_order"],
    )
    flow_ts = np.abs(flow_pulsatile_ml_s(vel_ts, cs.area_mm2))
    return {
        "offset": int(offset),
        "index": int(index),
        "flow_ml_s": to_numpy(flow_ts),
        "mean_flow_ml_min": float(mean_flow_ml_min(flow_ts)),
        "area_mm2": float(cs.area_mm2),
    }


def _neighbor_flow_waveforms(
    pick: CenterlinePick,
    centerlines: dict[int, np.ndarray],
    state: dict[str, Any],
    *,
    loc: LocPose | None = None,
) -> list[dict[str, Any]]:
    """Flow Q(t) at the picked station and up to ±2 neighbors along the centerline.

    Each entry carries the pulsatile series in ml/s plus its cardiac time-averaged
    magnitude in **mL/min** (``mean_flow_ml_min``), the unit the panel reports and
    the one the DB / literature bands use.

    When *loc* is set (QC tab with ``locs.csv``), offset 0 uses the **exact** stage-5
    LOC center + saved tangent — the same pose stage 6 wrote into ``loc_measurements.csv``.
    Neighbors still walk the matched centerline with recomputed tangents.
    """
    vx = state.get("vx")
    vy = state.get("vy")
    vz = state.get("vz")
    if vx is None or vy is None or vz is None:
        return []

    branch_name_by_key = state.get("branch_name_by_key") or {}
    volume_label_by_key = state.get("volume_label_by_key") or {}

    if loc is not None:
        resolved = _resolve_loc_centerline(
            loc, centerlines, branch_name_by_key, volume_label_by_key,
        )
        if resolved is not None:
            cl_key, cl_idx = resolved
        else:
            cl_key, cl_idx = int(pick.label), int(pick.index)
        pts = centerlines.get(int(cl_key))
        vol_lid = int(loc.vessel_id or volume_label_by_key.get(int(cl_key), cl_key))
        out: list[dict[str, Any]] = []
        # Exact LOC pose for the selected station.
        selected = _flow_at_pose(
            loc.center,
            loc.tangent,
            state=state,
            volume_label_id=vol_lid,
            offset=0,
            index=int(loc.centerline_index if loc.centerline_index is not None else cl_idx),
        )
        if selected is not None:
            out.append(selected)
        if pts is None or pts.shape[0] < 1:
            return out
        tangents = centerline_tangents(pts, k_half=2)
        for offset in (-2, -1, 1, 2):
            idx = int(cl_idx) + int(offset)
            if idx < 0 or idx >= pts.shape[0]:
                continue
            entry = _flow_at_pose(
                pts[idx],
                tangents[idx],
                state=state,
                volume_label_id=vol_lid,
                offset=offset,
                index=idx,
            )
            if entry is not None:
                out.append(entry)
        out.sort(key=lambda e: int(e.get("offset", 0)))
        return out

    pts = centerlines.get(int(pick.label))
    if pts is None or pts.shape[0] < 1:
        return []
    vol_lid = int(volume_label_by_key.get(int(pick.label), pick.label))
    tangents = centerline_tangents(pts, k_half=2)
    out = []
    for offset in (-2, -1, 0, 1, 2):
        idx = int(pick.index) + int(offset)
        if idx < 0 or idx >= pts.shape[0]:
            continue
        entry = _flow_at_pose(
            pts[idx],
            tangents[idx],
            state=state,
            volume_label_id=vol_lid,
            offset=offset,
            index=idx,
        )
        if entry is not None:
            out.append(entry)
    return out


def install_vessel_cross_sections(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    intensity_layer: Any,
    centerline_mask: np.ndarray,
    segmentation: np.ndarray | None,
    params: dict[str, Any],
    vx: np.ndarray | None = None,
    vy: np.ndarray | None = None,
    vz: np.ndarray | None = None,
    arterial_branches: dict[int, list[tuple[str, Any]]] | None = None,
    venous_centerlines: dict[str, Any] | None = None,
    venous_label_by_name: dict[str, int] | None = None,
    locs: Sequence[Mapping[str, Any]] | Sequence[LocPose] | None = None,
    mag: np.ndarray | None = None,
    vel_mag: np.ndarray | None = None,
) -> None:
    """Register Napari layers, dock, and 3D click-to-inspect behavior.

    When *arterial_branches* is provided (stage-4/6 ``centerlines_seg_branches``),
    those exact named polylines are used so the overlay coincides with pipeline
    centerlines. Otherwise polylines are re-extracted from *centerline_mask*.

    Optional *venous_centerlines* (stage-3 name → polyline) are always appended
    when present so venous vessels are pickable alongside arterial branches.

    Optional *locs* (stage-5 ``locs.csv`` rows or :class:`LocPose` list) enables
    snapping to the exact LOC pose stage 6 measured — used by the QC tab loader.
    The standalone visualization tool omits *locs* and keeps centerline-only picks.
    """
    shutdown_vessel_cross_sections(app_state)

    cd = _intensity_volume(intensity_layer)
    p = _params_from_dict(params)
    loc_poses: list[LocPose] = []
    if locs:
        if locs and isinstance(locs[0], LocPose):
            loc_poses = list(locs)  # type: ignore[arg-type]
        else:
            loc_poses = parse_loc_poses(locs)  # type: ignore[arg-type]
    volume_label_by_key: dict[int, int] = {}
    branch_name_by_key: dict[int, str] = {}
    from_stage_branches = False
    if arterial_branches:
        centerlines, volume_label_by_key, branch_name_by_key = (
            centerlines_dict_from_arterial_branches(arterial_branches, min_points=3)
        )
        from_stage_branches = bool(centerlines)
    if not from_stage_branches:
        centerlines = build_centerlines_dict(
            centerline_mask,
            segmentation=segmentation,
            min_points=5,
        )
        volume_label_by_key = {int(k): int(k) for k in centerlines}
        branch_name_by_key = {int(k): str(k) for k in centerlines}
    n_venous = append_venous_centerlines(
        centerlines,
        volume_label_by_key,
        branch_name_by_key,
        venous_centerlines,
        venous_label_by_name=venous_label_by_name,
        min_points=3,
    )
    if n_venous:
        # Venous polylines are exact stage-3 geometry — keep display unsmoothed
        # whenever any stage polylines are present.
        from_stage_branches = True
    if not centerlines:
        raise ValueError("No centerlines found (check centerline mask labels and min length).")

    names = {XS_CENTERLINES, XS_CL_POINTS, XS_PICK, XS_PLANE, XS_NORMAL, XS_TANGENT, XS_SEG}
    _remove_layers_named(viewer, names)

    centerlines_layer = _add_centerline_paths(
        viewer,
        centerlines,
        reference_layer=intensity_layer,
        # Keep stage-4/6 geometry exact (no display smoothing drift).
        smooth_display=not from_stage_branches,
    )
    cl_points_layer, cl_point_labels, cl_label_colors = _add_centerline_points_layer(
        viewer, centerlines, reference_layer=intensity_layer
    )
    if segmentation is not None and p["show_segmentation_3d"]:
        _add_seg_labels(viewer, segmentation, reference_layer=intensity_layer)

    pick_layer = viewer.add_points(
        np.zeros((0, 3), dtype=np.float64),
        name=XS_PICK,
        size=_overlay_point_size(intensity_layer),
        face_color="red",
        symbol="o",
        metadata=_overlay_metadata(),
        **_overlay_spatial_kwargs(intensity_layer),
    )
    _configure_overlay_points_layer(pick_layer)
    plane_layer: Any | None = None
    normal_layer: Any | None = None
    tangent_layer: Any | None = None
    edge_w = _overlay_edge_width(intensity_layer)

    panel = CrossSectionPanel()
    dock = attach_cross_section_dock(viewer, panel)
    voxel_sp = _voxel_spacing(intensity_layer)

    state: dict[str, Any] = {
        "cd": cd,
        "mag": mag if mag is not None else cd,
        "vel_mag": vel_mag if vel_mag is not None else cd,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "centerlines": centerlines,
        "volume_label_by_key": volume_label_by_key,
        "branch_name_by_key": branch_name_by_key,
        "from_stage_branches": from_stage_branches,
        "segmentation": segmentation,
        "params": p,
        "voxel_spacing": voxel_sp,
        "locs": loc_poses,
        "pick_layer": pick_layer,
        "plane_layer": plane_layer,
        "normal_layer": normal_layer,
        "tangent_layer": tangent_layer,
        "cl_points_layer": cl_points_layer,
        "cl_point_labels": cl_point_labels,
        "cl_label_colors": cl_label_colors,
        "panel": panel,
        "intensity_layer": intensity_layer,
        "centerlines_layer": centerlines_layer,
        "centerline_mask": centerline_mask,
        "branch_names": sorted(branch_name_by_key.values()),
    }

    def _update_plane_and_normal(pick: CenterlinePick, tangent: np.ndarray) -> None:
        """Create (once) or update the cross-section plane square and normal-direction line overlays
        for the current *pick*/*tangent*."""
        nonlocal plane_layer, normal_layer
        corners = _plane_square_corners(pick.point, tangent, p["radius_vox"])
        normal_data = _normal_half_line_data(
            pick.point,
            tangent,
            half_length_vox=_normal_display_half_length_vox(
                intensity_layer, p["radius_vox"]
            ),
        )
        shape_kwargs = {
            "shape_type": "polygon",
            "edge_color": "red",
            "face_color": [1.0, 0.0, 0.0, 0.08],
            "edge_width": edge_w,
            "metadata": _overlay_metadata(),
            **_overlay_spatial_kwargs(intensity_layer),
        }
        if plane_layer is None:
            plane_layer = viewer.add_shapes(
                [corners],
                name=XS_PLANE,
                **shape_kwargs,
            )
            _configure_overlay_shapes_layer(plane_layer)
            state["plane_layer"] = plane_layer
        else:
            plane_layer.data = [corners]
        normal_shape_kwargs = {
            "shape_type": "path",
            "edge_color": "red",
            "edge_width": edge_w * 1.5,
            "name": XS_NORMAL,
            "metadata": _overlay_metadata(),
            **_overlay_spatial_kwargs(intensity_layer),
        }
        if normal_layer is None:
            normal_layer = viewer.add_shapes(
                [normal_data],
                **normal_shape_kwargs,
            )
            _configure_overlay_shapes_layer(normal_layer)
            state["normal_layer"] = normal_layer
        else:
            normal_layer.data = [normal_data]

    def _update_tangent_visual(pick: CenterlinePick, tangent: np.ndarray) -> None:
        """Highlight the picked centerline point and its tangent-window neighbors, and create/update
        the arrowed tangent-line overlay for the current *pick*/*tangent*."""
        nonlocal tangent_layer
        cl_layer = state.get("cl_points_layer")
        point_labels = state.get("cl_point_labels")
        label_colors = state.get("cl_label_colors")
        if (
            cl_layer is not None
            and point_labels is not None
            and label_colors is not None
            and int(point_labels.shape[0]) > 0
        ):
            cl_layer.face_color = _centerline_point_colors_for_pick(
                point_labels,
                pick,
                centerlines,
                window=p["centerline_window"],
                label_colors=label_colors,
            )
        tang_paths = _tangent_display_paths_data(
            pick.point,
            tangent,
            reference_layer=intensity_layer,
            radius_vox=p["radius_vox"],
        )
        tang_edge_w = edge_w * 2.0
        shape_kwargs = {
            "shape_type": "path",
            "edge_color": "red",
            "edge_width": tang_edge_w,
            "name": XS_TANGENT,
            "metadata": _overlay_metadata(),
            **_overlay_spatial_kwargs(intensity_layer),
        }
        if tangent_layer is None:
            tangent_layer = viewer.add_shapes(
                tang_paths,
                **shape_kwargs,
            )
            _configure_overlay_shapes_layer(tangent_layer)
            state["tangent_layer"] = tangent_layer
        else:
            tangent_layer.data = tang_paths
            tangent_layer.edge_width = tang_edge_w

    def _apply_pick(
        pick: CenterlinePick,
        click_xyz: np.ndarray,
        *,
        loc: LocPose | None = None,
    ) -> None:
        """Compute the oblique cross-section at *pick*, update all overlay layers (pick marker, plane,
        normal, tangent, highlighted centerline points), and render the slice/waveforms in the dock.

        When *loc* is provided, the plane and selected-station flow use the exact stage-5
        LOC center + saved tangent (stage-6 parity). Otherwise the tangent is recomputed
        from the centerline (standalone visualization tool).
        """
        pts = centerlines[pick.label]
        if loc is not None:
            center = np.asarray(loc.center, dtype=np.float64).reshape(3)
            tang = np.asarray(loc.tangent, dtype=np.float64).reshape(3)
            # Rebuild a pick whose point is the exact LOC center for overlays.
            pick = CenterlinePick(
                label=int(pick.label),
                index=int(pick.index),
                point=center.astype(np.float32, copy=False),
                distance_sq=float(pick.distance_sq),
            )
            vol_lid = int(loc.vessel_id or volume_label_by_key.get(int(pick.label), pick.label))
            branch_name = loc.label()
            title_extra = "  LOC pose"
        else:
            center = np.asarray(pick.point, dtype=np.float64).reshape(3)
            tang = tangent_from_centerline(
                pts,
                pick.index,
                window=p["centerline_window"],
            )
            tang = choose_plane_normal_sense(
                tang,
                pick.point,
                click_xyz,
                centerline_pts=pts,
                index=pick.index,
            )
            vol_lid = int(volume_label_by_key.get(int(pick.label), pick.label))
            branch_name = str(branch_name_by_key.get(int(pick.label), pick.label))
            title_extra = ""
        try:
            # Prefer the same cross_section_at_loc path used for flow so the
            # displayed mask/area matches the Q̄ readout.
            result = cross_section_at_loc(
                center,
                tang,
                mag=state.get("mag", state["cd"]),
                cd=state["cd"],
                vel_mag=state.get("vel_mag", state["cd"]),
                voxel_spacing=state["voxel_spacing"],
                radius_vox=p["radius_vox"],
                interp_vals=p["interp_vals"],
                cross_section_res=p["cross_section_res"],
                plane_interp_order=p["plane_interp_order"],
                cs_supersampling=p.get("cs_supersampling", False),
                measure_resegment=p["measure_resegment"],
                thr_algorithm=p["thr_algorithm"],  # type: ignore[arg-type]
                volume_seg=state["segmentation"],
                volume_label_id=vol_lid,
                label_constrain=not p["measure_resegment"],
            )
        except Exception as exc:
            # Fallback for display-only if loc path fails (e.g. missing seg).
            try:
                result = cross_section_at_point(
                    center,
                    tang,
                    cd=state["cd"],
                    voxel_spacing=state["voxel_spacing"],
                    radius_vox=p["radius_vox"],
                    interp_vals=p["interp_vals"],
                    cross_section_res=p["cross_section_res"],
                    plane_interp_order=p["plane_interp_order"],
                    cs_supersampling=p.get("cs_supersampling", False),
                    measure_resegment=p["measure_resegment"],
                    thr_algorithm=p["thr_algorithm"],  # type: ignore[arg-type]
                    volume_seg=state["segmentation"],
                    volume_label_id=vol_lid,
                )
            except Exception:
                panel.clear(f"Cross-section failed: {exc}")
                return

        pick_layer.data = np.asarray(pick.point, dtype=np.float64).reshape(1, 3)
        _update_plane_and_normal(pick, tang)
        _update_tangent_visual(pick, tang)

        intensity_2d = result.intensity_2d
        if intensity_2d is None or int(getattr(np.asarray(intensity_2d), "ndim", 0)) != 2:
            # Never fall back to the 3D CD volume for the 2D dock.
            intensity_2d = np.zeros_like(result.mask_2d, dtype=np.float64)
        else:
            intensity_2d = np.asarray(intensity_2d)
            mask_arr = np.asarray(result.mask_2d)
            if intensity_2d.shape != mask_arr.shape:
                # Should not happen after cross_section_at_point fix; guard the dock.
                panel.clear(
                    f"Intensity/mask shape mismatch: {intensity_2d.shape} vs {mask_arr.shape}"
                )
                return
        idx_txt = (
            f"cl_idx {loc.centerline_index}"
            if loc is not None and loc.centerline_index is not None
            else f"index {pick.index}"
        )
        title = (
            f"{branch_name}  {idx_txt}{title_extra}\n"
            f"area {result.area_mm2:.2f} mm²  circularity {result.circularity:.3f}"
        )
        waveforms = _neighbor_flow_waveforms(
            pick, centerlines, state, loc=loc,
        )
        panel.show_slice(intensity_2d, result.mask_2d, title=title, waveforms=waveforms)
        if dock is not None:
            try:
                dock.show()
                dock.raise_()
            except Exception:
                pass

    def _pick_view_line_from_event(event: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """View line in voxel space: through click, direction into the scene.

        Prefer Napari ``get_ray_intersections`` on the intensity layer so the ray
        matches the viewer's transform / displayed dims (same stack as status hover).
        """
        from nvitk.gui.core.label_pick import view_ray_via_layer

        return view_ray_via_layer(intensity_layer, event, viewer=viewer)

    def _try_pick_from_event(event: Any, *, ndisplay: int) -> CenterlinePick | None:
        """Attempt to pick the nearest centerline point/segment from a mouse *event*, using the 3D
        view-ray when in 3D display mode, else a direct data-coordinate distance search."""
        pos = getattr(event, "position", None)
        if pos is None:
            pos = getattr(getattr(viewer, "cursor", None), "position", None)
        xyz = world_to_data_coords(intensity_layer, pos)
        if xyz is None:
            return None
        use_line = int(ndisplay) == 3
        line_origin, line_dir = (
            _pick_view_line_from_event(event) if use_line else (None, None)
        )
        if use_line and line_dir is None:
            use_line = False
        return pick_centerline(
            xyz.astype(np.float32, copy=False),
            centerlines,
            max_distance_vox=_pick_max_distance_vox(p),
            max_ray_distance_vox=_pick_max_ray_distance_vox(p),
            max_anchor_distance_vox=_pick_max_anchor_distance_vox(p),
            centerline_mask=state.get("centerline_mask"),
            ray_origin=line_origin,
            ray_direction=line_dir,
            use_view_line=use_line,
        )

    def _on_mouse_pick(viewer_obj: Any, event: Any) -> None:
        """Left-click handler in 3D view: pick the nearest centerline point/segment near the click and
        apply the cross-section pick, when picking is enabled and centerlines are visible.

        When stage-5 LOCs were passed in (QC tab), a click near a LOC snaps to that exact
        pose so flow matches ``loc_measurements.csv``.
        """
        if getattr(event, "type", None) != "mouse_press":
            return
        if not _is_left_mouse_button(event):
            return
        if int(getattr(viewer_obj.dims, "ndisplay", 2)) != 3:
            return
        panel_widget = state.get("panel")
        if panel_widget is not None and not panel_widget.is_picking_enabled():
            return
        if not _centerlines_layer_visible(state.get("centerlines_layer")):
            return

        click_xyz = world_to_data_coords(
            intensity_layer, getattr(event, "position", None)
        )
        snap_dist = float(p.get("loc_snap_distance_vox") or LOC_SNAP_DISTANCE_VOX)
        loc_hit = nearest_loc_pose(
            state.get("locs") or [],
            click_xyz if click_xyz is not None else np.zeros(3),
            max_distance_vox=snap_dist,
        ) if click_xyz is not None else None

        # Prefer a direct LOC hit (clicking the red LOC points) before centerline snap.
        if loc_hit is not None:
            resolved = _resolve_loc_centerline(
                loc_hit, centerlines, branch_name_by_key, volume_label_by_key,
            )
            if resolved is not None:
                cl_key, cl_idx = resolved
                pick = CenterlinePick(
                    label=int(cl_key),
                    index=int(cl_idx),
                    point=loc_hit.center.astype(np.float32, copy=False),
                    distance_sq=0.0,
                )
                _apply_pick(pick, click_xyz.astype(np.float32, copy=False), loc=loc_hit)
                try:
                    event.handled = True
                except Exception:
                    pass
                yield
                while getattr(event, "type", None) == "mouse_move":
                    yield
                return

        pick = _try_pick_from_event(
            event, ndisplay=int(getattr(viewer_obj.dims, "ndisplay", 2))
        )
        if pick is None:
            return
        if click_xyz is None:
            click_xyz = pick.point
        else:
            click_xyz = click_xyz.astype(np.float32, copy=False)
        pick = refine_pick_to_vertex_if_closer(
            pick,
            centerlines,
            click_xyz,
            max_distance_vox=max(2.0, _pick_max_distance_vox(p) * 0.35),
        )
        # Snap centerline picks that land next to a LOC onto the LOC pose.
        loc_near = nearest_loc_pose(
            state.get("locs") or [],
            pick.point,
            max_distance_vox=snap_dist,
        )
        _apply_pick(pick, click_xyz, loc=loc_near)
        try:
            event.handled = True
        except Exception:
            pass
        yield
        while getattr(event, "type", None) == "mouse_move":
            yield

    _connect_pick_callback(viewer, _on_mouse_pick)

    prev_ndisplay = int(getattr(viewer.dims, "ndisplay", 2))
    if prev_ndisplay != 3:
        try:
            viewer.dims.ndisplay = 3
        except Exception:
            pass

    try:
        intensity_layer.mode = "pan_zoom"
        viewer.layers.selection.active = intensity_layer
        for lyr in (cl_points_layer, pick_layer, centerlines_layer):
            if lyr is None:
                continue
            if getattr(lyr, "name", "") in (XS_CL_POINTS, XS_PICK):
                _configure_overlay_points_layer(lyr)
            else:
                _configure_overlay_shapes_layer(lyr)
    except Exception:
        pass

    try:
        from nvitk.gui.viz.layers import repair_time_dim_for_viewer

        repair_time_dim_for_viewer(viewer)
    except Exception:
        pass

    app_state["vessel_xs"] = {
        **state,
        "viewer": viewer,
        "callback": _on_mouse_pick,
        "dock": dock,
        "intensity_layer_name": intensity_layer.name,
        "prev_ndisplay": prev_ndisplay,
    }


__all__ = [
    "XS_CENTERLINES",
    "XS_CL_POINTS",
    "XS_PICK",
    "XS_PLANE",
    "XS_NORMAL",
    "XS_TANGENT",
    "XS_SEG",
    "build_centerlines_dict",
    "centerlines_dict_from_arterial_branches",
    "append_venous_centerlines",
    "install_vessel_cross_sections",
    "shutdown_vessel_cross_sections",
]
