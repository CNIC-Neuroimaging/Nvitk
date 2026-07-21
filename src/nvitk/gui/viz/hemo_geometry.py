"""Napari overlays for PITC / PWV vessel hemodynamics geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.gui.core.spatial import layer_affine
from nvitk.gui.viz.layers import init_points_layer_style, install_points_style_sync
from nvitk.pipes.qvtpy.util.vessel_hemodynamics import RegionGeometryViz

HEMO_INIT_LAYER = "PITC/PWV root init"
HEMO_PATHS_LAYER = "PITC/PWV centerlines"
HEMO_PITC_STATIONS_LAYER = "PITC stations"
HEMO_PWV_STATIONS_LAYER = "PWV stations"
HEMO_OVERLAY_META = "nvitk_hemo_overlay"

_REGION_COLORS = {
    "L_ICA": "#1f77b4",
    "R_ICA": "#d62728",
    "Basilar": "#2ca02c",
}


@dataclass
class HemoOverlayState:
    """Tracks style-sync callbacks for active hemodynamics overlays."""

    disconnectors: list[Any]


def _overlay_metadata() -> dict[str, Any]:
    return {HEMO_OVERLAY_META: True}


def clear_hemo_geometry_layers(viewer: Any) -> None:
    """Remove prior PITC/PWV geometry overlays."""
    state = getattr(viewer, "_nvitk_hemo_overlay_state", None)
    if state is not None:
        for disconnect in getattr(state, "disconnectors", []) or []:
            try:
                disconnect()
            except Exception:
                pass
        setattr(viewer, "_nvitk_hemo_overlay_state", None)
    for lyr in list(viewer.layers):
        meta = getattr(lyr, "metadata", {}) or {}
        if meta.get(HEMO_OVERLAY_META):
            viewer.layers.remove(lyr)


def _add_paths_layer(
    viewer: Any,
    paths: list[np.ndarray],
    edge_colors: list[str],
    *,
    reference_layer: Any | None,
) -> None:
    if not paths:
        return
    kwargs: dict[str, Any] = {
        "name": HEMO_PATHS_LAYER,
        "shape_type": "path",
        "edge_color": edge_colors,
        "edge_width": 0.35,
        "opacity": 0.9,
        "metadata": _overlay_metadata(),
    }
    aff = layer_affine(reference_layer) if reference_layer is not None else None
    if aff is not None:
        kwargs["affine"] = aff
    layer = viewer.add_shapes(paths, **kwargs)
    try:
        layer.editable = False
    except Exception:
        pass


def _stack_features(
    rows: Iterable[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows_list = list(rows)
    if not rows_list:
        return np.zeros((0, 3), dtype=np.float64), {
            k: np.asarray([], dtype=np.float64) for k in feature_names
        }
    coords = np.asarray(
        [
            [float(r["centerline_x"]), float(r["centerline_y"]), float(r["centerline_z"])]
            for r in rows_list
        ],
        dtype=np.float64,
    )
    features: dict[str, np.ndarray] = {}
    for name in feature_names:
        vals = [r.get(name) for r in rows_list]
        if name in ("region_id", "vessel_name"):
            features[name] = np.asarray([str(v) for v in vals], dtype=object)
        elif name == "used_for_pwv":
            features[name] = np.asarray([bool(v) for v in vals], dtype=bool)
        elif name in ("vessel_id", "station_index"):
            features[name] = np.asarray([int(v) for v in vals], dtype=np.int32)
        else:
            features[name] = np.asarray([float(v) for v in vals], dtype=np.float64)
    return coords, features


def _add_points_layer(
    viewer: Any,
    coords: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    name: str,
    reference_layer: Any | None,
    size: float,
    symbol: str,
    face_color: Any,
    face_colormap: str | None = None,
    face_contrast_limits: tuple[float, float] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "name": name,
        "features": features,
        "size": float(size),
        "symbol": symbol,
        "border_width": 0,
        "border_width_is_relative": False,
        "metadata": _overlay_metadata(),
    }
    feature_color = (
        coords.shape[0] > 0
        and isinstance(face_color, str)
        and face_color in features
    )
    if coords.shape[0] == 0 or isinstance(face_color, str) and face_color.startswith("#"):
        kwargs["face_color"] = face_color
    else:
        kwargs["face_color"] = face_color
        if face_colormap:
            kwargs["face_colormap"] = face_colormap
        if face_contrast_limits is not None:
            kwargs["face_contrast_limits"] = face_contrast_limits
    aff = layer_affine(reference_layer) if reference_layer is not None else None
    if aff is not None:
        kwargs["affine"] = aff
    layer = viewer.add_points(coords, **kwargs)
    if feature_color:
        layer.size = float(size)
        layer.symbol = symbol
        if hasattr(layer, "current_size"):
            layer.current_size = float(size)
        if hasattr(layer, "current_symbol"):
            layer.current_symbol = symbol
    else:
        init_points_layer_style(layer, size=float(size), symbol=symbol, face_color=face_color)
    disconnect = install_points_style_sync(layer, sync_face_color=True)
    return layer, disconnect


def _finite_limits(features: dict[str, np.ndarray], key: str) -> tuple[float, float] | None:
    vals = to_numpy(features.get(key, np.asarray([], dtype=np.float64))).astype(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def add_hemo_geometry_layers(
    viewer: Any,
    regions: list[RegionGeometryViz],
    *,
    reference_layer: Any | None,
    mode: str,
    face_key: str,
    point_size: float = 2.5,
) -> None:
    """Add PITC or PWV geometry overlays to Napari."""
    clear_hemo_geometry_layers(viewer)
    disconnectors: list[Any] = []
    paths: list[np.ndarray] = []
    path_colors: list[str] = []
    init_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    for region in regions:
        color = _REGION_COLORS.get(region.region_id, "#9467bd")
        init_rows.append(
            {
                "centerline_x": float(region.root_init_xyz[0]),
                "centerline_y": float(region.root_init_xyz[1]),
                "centerline_z": float(region.root_init_xyz[2]),
                "region_id": region.region_id,
                "vessel_name": "root_init",
                "vessel_id": int(region.root_label),
                "station_index": 0,
                "distance_mm": 0.0,
            }
        )
        for vessel in region.vessels.values():
            if vessel.polyline_oriented.shape[0] >= 2:
                paths.append(vessel.polyline_oriented.astype(np.float32))
                path_colors.append(color)
            for station in vessel.stations:
                if mode == "pwv" and not station.used_for_pwv:
                    continue
                station_rows.append(
                    {
                        "centerline_x": station.centerline_x,
                        "centerline_y": station.centerline_y,
                        "centerline_z": station.centerline_z,
                        "region_id": region.region_id,
                        "vessel_id": int(station.vessel_id),
                        "vessel_name": station.vessel_name,
                        "station_index": int(station.station_index),
                        "distance_mm": float(station.distance_mm),
                        "pi": float(station.pi),
                        "quality": float(station.quality),
                        "area_mm2": float(station.area_mm2),
                        "used_for_pwv": bool(station.used_for_pwv),
                        "pwv_weight_area": float(station.pwv_weight_area),
                        "pwv_weight_quality": float(station.pwv_weight_quality),
                        "pwv_xcor_time_s": float(station.pwv_xcor_time_s),
                        "pwv_time_to_upstroke_s": float(station.pwv_time_to_upstroke_s),
                    }
                )
    _add_paths_layer(viewer, paths, path_colors, reference_layer=reference_layer)
    init_coords, init_features = _stack_features(
        init_rows, ("region_id", "vessel_id", "vessel_name", "station_index", "distance_mm")
    )
    init_layer, init_disc = _add_points_layer(
        viewer,
        init_coords,
        init_features,
        name=HEMO_INIT_LAYER,
        reference_layer=reference_layer,
        size=max(3.5, float(point_size) * 1.6),
        symbol="star",
        face_color="#2ca02c",
    )
    disconnectors.append(init_disc)
    station_features_all = (
        "region_id",
        "vessel_id",
        "vessel_name",
        "station_index",
        "distance_mm",
        "pi",
        "quality",
        "area_mm2",
        "used_for_pwv",
        "pwv_weight_area",
        "pwv_weight_quality",
        "pwv_xcor_time_s",
        "pwv_time_to_upstroke_s",
    )
    station_coords, station_features = _stack_features(station_rows, station_features_all)
    face_key = str(face_key or ("quality" if mode == "pitc" else "pwv_weight_area"))
    cmap = "viridis" if mode == "pitc" else "magma"
    limits = _finite_limits(station_features, face_key)
    station_name = HEMO_PITC_STATIONS_LAYER if mode == "pitc" else HEMO_PWV_STATIONS_LAYER
    station_layer, station_disc = _add_points_layer(
        viewer,
        station_coords,
        station_features,
        name=station_name,
        reference_layer=reference_layer,
        size=float(point_size),
        symbol="disc",
        face_color=face_key if station_coords.shape[0] else "#d62728",
        face_colormap=cmap,
        face_contrast_limits=limits,
    )
    disconnectors.append(station_disc)
    try:
        init_layer.mode = "pan_zoom"
        station_layer.mode = "pan_zoom"
    except Exception:
        pass
    setattr(viewer, "_nvitk_hemo_overlay_state", HemoOverlayState(disconnectors=disconnectors))


__all__ = [
    "HEMO_INIT_LAYER",
    "HEMO_PATHS_LAYER",
    "HEMO_PITC_STATIONS_LAYER",
    "HEMO_PWV_STATIONS_LAYER",
    "add_hemo_geometry_layers",
    "clear_hemo_geometry_layers",
]
