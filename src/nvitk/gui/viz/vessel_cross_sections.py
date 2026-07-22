"""Napari vessel cross-section viewer (centerline pick + oblique 2D panel)."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.measure.cross_section import cross_section_at_loc, cross_section_at_point, masked_plane_velocity_series
from nvitk.measure.hemodynamics import flow_pulsatile_ml_s
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
    layer_affine,
    layer_spatial_kwargs,
    layer_spacing,
    world_to_data_coords,
)
from nvitk.gui.viz.layers import DEFAULT_FLOW_EDGE_WIDTH
from nvitk.gui.viz.cross_section_panel import CrossSectionPanel, attach_cross_section_dock

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
    for lyr in list(viewer.layers):
        if lyr.name in names:
            viewer.layers.remove(lyr)


def _intensity_volume(layer: Any) -> np.ndarray:
    data = to_numpy(layer.data)
    if np.iscomplexobj(data):
        data = np.abs(data)
    return np.asarray(data, dtype=np.float64)


def _overlay_metadata() -> dict[str, Any]:
    return {XS_OVERLAY_META: True}


def _overlay_spatial_kwargs(reference_layer: Any) -> dict[str, Any]:
    return layer_spatial_kwargs(reference_layer)


def _configure_overlay_points_layer(layer: Any) -> None:
    try:
        layer.editable = False
    except Exception:
        pass
    try:
        layer.mode = "pan_zoom"
    except Exception:
        pass


def _configure_overlay_shapes_layer(layer: Any) -> None:
    try:
        layer.editable = False
    except Exception:
        pass


def _is_left_mouse_button(event: Any) -> bool:
    btn = getattr(event, "button", None)
    if btn in (0, 1, None):
        return True
    name = str(btn).lower()
    return name in ("left", "lbutton", "mouse1")


def _overlay_edge_width(layer: Any) -> float:
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return max(0.12, float(min(sp)) * 0.35)
    return DEFAULT_FLOW_EDGE_WIDTH


def _overlay_point_size(layer: Any) -> float:
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return max(0.4, float(min(sp)) * 1.2)
    return 1.0


def _connect_pick_callback(target: Any, callback: Any) -> None:
    try:
        target.mouse_drag_callbacks.insert(0, callback)
    except Exception:
        target.mouse_drag_callbacks.append(callback)


def _disconnect_pick_callback(target: Any, callback: Any) -> None:
    if target is None or callback is None:
        return
    try:
        if callback in target.mouse_drag_callbacks:
            target.mouse_drag_callbacks.remove(callback)
    except Exception:
        pass


def _world_to_layer_data(layer: Any, position: Any) -> np.ndarray | None:
    """Unclipped world→data coords for *layer*'s 3D spatial grid.

    Uses ``layer.world_to_data`` (per-layer transform), which correctly handles a
    3D layer embedded in a higher-dim viewer, then keeps the trailing 3 axes.
    """
    if position is None:
        return None
    try:
        data_pos = layer.world_to_data(position)
        pos = to_numpy(data_pos).astype(np.float64).ravel()
    except Exception:
        pos = to_numpy(position).astype(np.float64).ravel()
        aff = layer_affine(layer)
        if aff is not None and pos.size >= 3:
            inv = np.linalg.inv(to_numpy(aff).astype(np.float64))
            homog = np.array([pos[-3], pos[-2], pos[-1], 1.0], dtype=np.float64)
            pos = (inv @ homog)[:3]
    if pos.size < 3:
        return None
    return pos[-3:].astype(np.float64)


def _view_ray_in_layer_data(
    layer: Any,
    position: Any,
    view_direction: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Ray (origin, into-scene unit direction) in *layer* data coords.

    Mapping both the click point and a point one step along the view direction with
    the layer's own ``world_to_data`` keeps origin and direction on the same axes,
    even when a higher-dim (4D) layer shifts Napari's displayed world axes.
    """
    if position is None:
        return None, None
    origin = _world_to_layer_data(layer, position)
    if origin is None:
        return None, None
    if view_direction is None:
        return origin, None
    pos_arr = to_numpy(position).astype(np.float64).ravel()
    vd_arr = to_numpy(view_direction).astype(np.float64).ravel()
    n = min(pos_arr.size, vd_arr.size)
    if n == 0:
        return origin, None
    tip = _world_to_layer_data(layer, pos_arr[:n] + vd_arr[:n])
    if tip is None:
        return origin, None
    direction = -(tip - origin)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return origin, None
    return origin, (direction / norm).astype(np.float64)


def _plane_square_corners(
    center: np.ndarray,
    tangent: np.ndarray,
    radius_vox: float,
) -> np.ndarray:
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
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return max(0.8, float(min(sp)) * 1.8)
    return max(1.0, float(radius_vox) * 0.35)


def _tangent_display_half_length_vox(layer: Any, radius_vox: float) -> float:
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
    if centerlines_layer is None:
        return False
    return bool(getattr(centerlines_layer, "visible", True))


def _voxel_spacing(layer: Any) -> tuple[float, float, float]:
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)


def _params_from_dict(params: dict[str, Any]) -> dict[str, Any]:
    resegment = bool(params.get("measure_resegment", True))
    interp = bool(params.get("interpolate_plane", True)) and resegment
    return {
        "radius_vox": float(params.get("cross_section_radius_vox") or 12.0),
        "cross_section_res": int(params.get("cross_section_res") or 0),
        "interp_vals": int(params.get("interp_vals") or 4) if resegment else 1,
        "plane_interp_order": 1 if interp else 0,
        "measure_resegment": resegment,
        "thr_algorithm": str(params.get("thr_algorithm") or "lsthr"),
        "centerline_window": int(str(params.get("centerline_window") or "5")),
        "show_segmentation_3d": bool(params.get("show_segmentation_3d", True)),
    }


def _label_color_map(centerlines: dict[int, np.ndarray]) -> dict[int, str]:
    return {
        int(lbl): _TAB10[i % len(_TAB10)]
        for i, lbl in enumerate(sorted(centerlines.keys()))
    }


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
    return [label_colors[int(lbl)] for lbl in point_labels]


def _centerline_point_colors_for_pick(
    point_labels: np.ndarray,
    pick: CenterlinePick,
    centerlines: dict[int, np.ndarray],
    *,
    window: int,
    label_colors: dict[int, str],
) -> list[str]:
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


def build_centerlines_dict(
    centerline_mask: np.ndarray,
    *,
    segmentation: np.ndarray | None = None,
    min_points: int = 5,
) -> dict[int, np.ndarray]:
    """Ordered polylines per label from centerline (+ optional seg) masks."""
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


def _add_centerline_paths(
    viewer: Any,
    centerlines: dict[int, np.ndarray],
    *,
    reference_layer: Any,
) -> Any:
    paths = []
    colors = []
    for i, lbl in enumerate(sorted(centerlines.keys())):
        pts = centerlines[lbl]
        if pts is None or pts.shape[0] < 2:
            continue
        disp = smooth_polyline_display(pts)
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


def _neighbor_flow_waveforms(
    pick: CenterlinePick,
    centerlines: dict[int, np.ndarray],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flow Q(t) at the picked station and up to ±2 neighbors along the centerline."""
    vx = state.get("vx")
    vy = state.get("vy")
    vz = state.get("vz")
    if vx is None or vy is None or vz is None:
        return []
    pts = centerlines.get(int(pick.label))
    if pts is None or pts.shape[0] < 1:
        return []
    p = state["params"]
    tangents = centerline_tangents(pts, k_half=2)
    out: list[dict[str, Any]] = []
    for offset in (-2, -1, 0, 1, 2):
        idx = int(pick.index) + int(offset)
        if idx < 0 or idx >= pts.shape[0]:
            continue
        try:
            cs = cross_section_at_loc(
                pts[idx],
                tangents[idx],
                mag=state["cd"],
                cd=state["cd"],
                vel_mag=state["cd"],
                voxel_spacing=state["voxel_spacing"],
                radius_vox=p["radius_vox"],
                cross_section_res=p["cross_section_res"],
                plane_interp_order=p["plane_interp_order"],
                measure_resegment=p["measure_resegment"],
                thr_algorithm=p["thr_algorithm"],  # type: ignore[arg-type]
                volume_seg=state.get("segmentation"),
                volume_label_id=int(pick.label),
                label_constrain=not p["measure_resegment"],
            )
        except Exception:
            continue
        if cs.area_mm2 <= 0.0 or not bool(np.any(cs.mask_2d)):
            continue
        vel_ts = masked_plane_velocity_series(
            vx,
            vy,
            vz,
            cs,
            plane_interp_order=p["plane_interp_order"],
        )
        flow_ts = np.abs(flow_pulsatile_ml_s(vel_ts, cs.area_mm2))
        out.append(
            {
                "offset": int(offset),
                "index": int(idx),
                "flow_ml_s": flow_ts,
            }
        )
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
) -> None:
    """Register Napari layers, dock, and 3D click-to-inspect behavior."""
    shutdown_vessel_cross_sections(app_state)

    cd = _intensity_volume(intensity_layer)
    p = _params_from_dict(params)
    centerlines = build_centerlines_dict(
        centerline_mask,
        segmentation=segmentation,
        min_points=5,
    )
    if not centerlines:
        raise ValueError("No centerlines found (check centerline mask labels and min length).")

    names = {XS_CENTERLINES, XS_CL_POINTS, XS_PICK, XS_PLANE, XS_NORMAL, XS_TANGENT, XS_SEG}
    _remove_layers_named(viewer, names)

    centerlines_layer = _add_centerline_paths(
        viewer, centerlines, reference_layer=intensity_layer
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
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "centerlines": centerlines,
        "segmentation": segmentation,
        "params": p,
        "voxel_spacing": voxel_sp,
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
    }

    def _update_plane_and_normal(pick: CenterlinePick, tangent: np.ndarray) -> None:
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

    def _apply_pick(pick: CenterlinePick, click_xyz: np.ndarray) -> None:
        pts = centerlines[pick.label]
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
        try:
            result = cross_section_at_point(
                pick.point,
                tang,
                cd=state["cd"],
                voxel_spacing=state["voxel_spacing"],
                radius_vox=p["radius_vox"],
                interp_vals=p["interp_vals"],
                cross_section_res=p["cross_section_res"],
                plane_interp_order=p["plane_interp_order"],
                measure_resegment=p["measure_resegment"],
                thr_algorithm=p["thr_algorithm"],  # type: ignore[arg-type]
                volume_seg=state["segmentation"],
                volume_label_id=pick.label,
            )
        except Exception as exc:
            panel.clear(f"Cross-section failed: {exc}")
            return

        pick_layer.data = np.asarray(pick.point, dtype=np.float64).reshape(1, 3)
        _update_plane_and_normal(pick, tang)
        _update_tangent_visual(pick, tang)

        intensity_2d = result.intensity_2d
        if intensity_2d is None:
            intensity_2d = state["cd"]
        title = (
            f"Label {pick.label}  index {pick.index}\n"
            f"area {result.area_mm2:.2f} mm²  circularity {result.circularity:.3f}"
        )
        waveforms = _neighbor_flow_waveforms(pick, centerlines, state)
        panel.show_slice(intensity_2d, result.mask_2d, title=title, waveforms=waveforms)
        if dock is not None:
            try:
                dock.show()
                dock.raise_()
            except Exception:
                pass

    def _pick_view_line_from_event(event: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """View line in voxel space: through click, direction into the scene.

        Both the ray origin and direction are mapped with ``intensity_layer.world_to_data``
        so the pick stays consistent even when extra 4D layers embed the intensity layer
        on different world axes (Napari right-aligns dims, shifting ``dims_displayed``).
        """
        pos = getattr(event, "position", None)
        if pos is None:
            pos = getattr(getattr(viewer, "cursor", None), "position", None)
        view_dir = getattr(event, "view_direction", None)
        if view_dir is None:
            view_dir = getattr(getattr(viewer, "dims", None), "view_direction", None)
        return _view_ray_in_layer_data(intensity_layer, pos, view_dir)

    def _try_pick_from_event(event: Any, *, ndisplay: int) -> CenterlinePick | None:
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
        pick = _try_pick_from_event(
            event, ndisplay=int(getattr(viewer_obj.dims, "ndisplay", 2))
        )
        if pick is None:
            return
        click_xyz = world_to_data_coords(
            intensity_layer, getattr(event, "position", None)
        )
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
        _apply_pick(pick, click_xyz)
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
        "viewer": viewer,
        "callback": _on_mouse_pick,
        "intensity_layer": intensity_layer,
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
    "install_vessel_cross_sections",
    "shutdown_vessel_cross_sections",
]
