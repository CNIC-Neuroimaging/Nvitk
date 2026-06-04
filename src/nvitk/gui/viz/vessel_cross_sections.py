"""Napari vessel cross-section viewer (centerline pick + oblique 2D panel)."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.measure.cross_section import cross_section_at_point
from nvitk.morphology import compute_centerlines
from nvitk.viz.centerline_pick import (
    CenterlinePick,
    choose_plane_normal_sense,
    pick_centerline,
    smooth_polyline_display,
    tangent_from_centerline,
    unit_vector,
)

from nvitk.gui.core.spatial import (
    data_indices_to_world,
    layer_spatial_kwargs,
    layer_spacing,
    view_direction_into_scene,
    world_to_data_coords,
)
from nvitk.gui.viz.layers import DEFAULT_FLOW_EDGE_WIDTH
from nvitk.gui.viz.cross_section_panel import CrossSectionPanel, attach_cross_section_dock

XS_CENTERLINES = "Vessel centerlines (xs)"
XS_PICK = "Cross-section pick"
XS_PLANE = "Cross-section plane (xs)"
XS_NORMAL = "Cross-section normal (xs)"
XS_SEG = "Vessel seg (xs)"
XS_OVERLAY_META = "nvitk_vessel_xs_overlay"

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


def _normal_half_line_world(
    center_vox: np.ndarray,
    tangent: np.ndarray,
    *,
    reference_layer: Any,
    half_length_vox: float,
) -> np.ndarray:
    """Single red segment along +normal only (world coordinates for Shapes)."""
    c = to_numpy(center_vox).astype(np.float64).reshape(3)
    t = unit_vector(to_numpy(tangent).astype(np.float64).reshape(3))
    half_len = float(half_length_vox)
    seg = np.stack([c, c + half_len * t], axis=0)
    return data_indices_to_world(seg, reference_layer)


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
        paths.append(data_indices_to_world(disp, reference_layer))
        colors.append(_TAB10[i % len(_TAB10)])
    if not paths:
        return None
    kwargs = {
        "edge_width": _overlay_edge_width(reference_layer),
        "opacity": 0.9,
        "name": XS_CENTERLINES,
        "metadata": _overlay_metadata(),
    }
    try:
        layer = viewer.add_shapes(
            paths,
            shape_type="path",
            edge_color=colors,
            **kwargs,
        )
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


def install_vessel_cross_sections(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    intensity_layer: Any,
    centerline_mask: np.ndarray,
    segmentation: np.ndarray | None,
    params: dict[str, Any],
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

    names = {XS_CENTERLINES, XS_PICK, XS_PLANE, XS_NORMAL, XS_SEG}
    _remove_layers_named(viewer, names)

    centerlines_layer = _add_centerline_paths(
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
    )
    plane_layer: Any | None = None
    normal_layer: Any | None = None
    edge_w = _overlay_edge_width(intensity_layer)

    panel = CrossSectionPanel()
    dock = attach_cross_section_dock(viewer, panel)
    voxel_sp = _voxel_spacing(intensity_layer)

    state: dict[str, Any] = {
        "cd": cd,
        "centerlines": centerlines,
        "segmentation": segmentation,
        "params": p,
        "voxel_spacing": voxel_sp,
        "pick_layer": pick_layer,
        "plane_layer": plane_layer,
        "normal_layer": normal_layer,
        "panel": panel,
        "intensity_layer": intensity_layer,
        "centerlines_layer": centerlines_layer,
        "centerline_mask": centerline_mask,
    }

    def _update_plane_and_normal(pick: CenterlinePick, tangent: np.ndarray) -> None:
        nonlocal plane_layer, normal_layer
        corners = _plane_square_corners(pick.point, tangent, p["radius_vox"])
        corners_w = data_indices_to_world(corners, intensity_layer)
        normal_w = _normal_half_line_world(
            pick.point,
            tangent,
            reference_layer=intensity_layer,
            half_length_vox=_normal_display_half_length_vox(
                intensity_layer, p["radius_vox"]
            ),
        )
        if plane_layer is None:
            plane_layer = viewer.add_shapes(
                [corners_w],
                shape_type="polygon",
                edge_color="red",
                face_color=[1.0, 0.0, 0.0, 0.08],
                edge_width=edge_w,
                name=XS_PLANE,
                metadata=_overlay_metadata(),
            )
            state["plane_layer"] = plane_layer
        else:
            plane_layer.data = [corners_w]
        if normal_layer is None:
            normal_layer = viewer.add_shapes(
                [normal_w],
                shape_type="path",
                edge_color="red",
                edge_width=edge_w * 1.5,
                name=XS_NORMAL,
                metadata=_overlay_metadata(),
            )
            state["normal_layer"] = normal_layer
        else:
            normal_layer.data = [normal_w]

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

        pick_layer.data = data_indices_to_world(
            np.asarray(pick.point, dtype=np.float64).reshape(1, 3),
            intensity_layer,
        )
        _update_plane_and_normal(pick, tang)

        intensity_2d = result.intensity_2d
        if intensity_2d is None:
            intensity_2d = state["cd"]
        title = (
            f"Label {pick.label}  index {pick.index}\n"
            f"area {result.area_mm2:.2f} mm²  circularity {result.circularity:.3f}"
        )
        panel.show_slice(intensity_2d, result.mask_2d, title=title)
        if dock is not None:
            try:
                dock.show()
                dock.raise_()
            except Exception:
                pass

    def _pick_view_line_from_event(event: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        """View line in voxel space: through click, direction into the scene."""
        pos = getattr(event, "position", None)
        if pos is None:
            pos = getattr(getattr(viewer, "cursor", None), "position", None)
        origin = world_to_data_coords(intensity_layer, pos)
        if origin is None:
            return None, None
        view_dir = getattr(getattr(viewer, "dims", None), "view_direction", None)
        if view_dir is None:
            view_dir = getattr(event, "view_direction", None)
        dir_data = view_direction_into_scene(intensity_layer, view_dir, event)
        return origin, dir_data

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
    "XS_PICK",
    "XS_PLANE",
    "XS_NORMAL",
    "XS_SEG",
    "build_centerlines_dict",
    "install_vessel_cross_sections",
    "shutdown_vessel_cross_sections",
]
