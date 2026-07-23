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
HEMO_EXCLUDED_LAYER = "PITC/PWV excluded branches"
HEMO_STATIONS_LAYER = "PITC/PWV stations"
# Legacy names (cleared if present from older tool runs).
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
    """Tracks style-sync callbacks and the active stations layer."""

    disconnectors: list[Any]
    station_layer: Any | None = None
    mode: str = "pitc"
    default_face_key: str = "quality"


_NUMERIC_STATION_FEATURES: tuple[str, ...] = (
    "distance_mm",
    "pi",
    "quality",
    "area_mm2",
    "pwv_weight_area",
    "pwv_weight_quality",
    "pwv_xcor_time_s",
    "pwv_time_to_upstroke_s",
    "pwv_bjornfoot_weighted_rms",
    "pwv_bjornfoot_delay_residual_s",
    "pwv_bjornfoot_waveform_corr",
)


def station_feature_choices(layer: Any) -> list[str]:
    """Numeric feature columns available for station face coloring."""
    feat = getattr(layer, "features", None)
    if feat is None:
        return []
    raw_cols = getattr(feat, "columns", None)
    cols = list(raw_cols) if raw_cols is not None else []
    if not cols and isinstance(feat, dict):
        cols = list(feat.keys())
    column_names = {str(column) for column in cols}
    return [c for c in _NUMERIC_STATION_FEATURES if c in column_names]


def color_stations_by_feature(
    layer: Any,
    feature: str,
    *,
    mode: str = "pitc",
    cmap_name: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float] | None:
    """Color a stations Points layer by a numeric feature.

    Recent Napari builds only expose a solid face-color swatch in layer controls,
    so this helper is the supported way to switch feature encodings after a run.

    Colors are assigned as explicit per-point RGBA rather than Napari's
    feature-colormap mode: PWV metrics carry NaN for stations not used in the fit,
    and Napari's colormap mapping renders nothing when any value is NaN. NaN values
    are drawn transparent so the remaining stations stay colored.

    Returns the ``(lo, hi)`` contrast limits actually applied, or ``None`` if the
    layer had no points.
    """
    key = str(feature)
    choices = station_feature_choices(layer)
    if key not in choices:
        raise ValueError(f"Feature {key!r} not available on stations layer.")
    feat = layer.features
    if hasattr(feat, "__getitem__"):
        vals = to_numpy(feat[key]).astype(np.float64)
    else:
        vals = to_numpy(feat.get(key, [])).astype(np.float64)
    vals = vals.reshape(-1)
    finite_mask = np.isfinite(vals)
    finite = vals[finite_mask]
    if finite.size:
        auto_lo = float(np.min(finite))
        auto_hi = float(np.max(finite))
        if auto_hi <= auto_lo:
            auto_hi = auto_lo + 1.0
    else:
        auto_lo, auto_hi = 0.0, 1.0
    lo = float(auto_lo if vmin is None else vmin)
    hi = float(auto_hi if vmax is None else vmax)
    if hi <= lo:
        hi = lo + 1.0
    cmap = str(cmap_name) if cmap_name else ("viridis" if mode == "pitc" else "magma")
    rgba = _feature_rgba(vals, finite_mask, lo, hi, cmap)
    if rgba.shape[0] == 0:
        return None
    if hasattr(layer, "face_color_mode"):
        try:
            layer.face_color_mode = "direct"
        except Exception:
            pass
    layer.face_color = rgba
    if hasattr(layer, "face_colormap"):
        layer.face_colormap = cmap
    if hasattr(layer, "face_contrast_limits"):
        layer.face_contrast_limits = (lo, hi)
    if hasattr(layer, "refresh_colors"):
        try:
            layer.refresh_colors(update_color_mapping=False)
        except TypeError:
            layer.refresh_colors()
    return lo, hi


def _feature_rgba(
    values: np.ndarray,
    finite_mask: np.ndarray,
    lo: float,
    hi: float,
    cmap_name: str,
) -> np.ndarray:
    """Per-point RGBA for *values*; non-finite entries are transparent."""
    import matplotlib as mpl

    try:
        cmap = mpl.colormaps[cmap_name]
    except (KeyError, AttributeError):
        cmap = mpl.cm.get_cmap(cmap_name)
    norm = np.zeros_like(values, dtype=np.float64)
    span = hi - lo
    if span > 0:
        norm = np.clip((values - lo) / span, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    rgba = np.asarray(cmap(norm), dtype=np.float64)
    if rgba.ndim == 1:
        rgba = rgba.reshape(1, -1)
    rgba[~finite_mask] = (0.0, 0.0, 0.0, 0.0)
    return rgba


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
    name: str,
    reference_layer: Any | None,
    edge_width: float = 0.35,
) -> None:
    if not paths:
        return
    kwargs: dict[str, Any] = {
        "name": name,
        "shape_type": "path",
        "edge_color": edge_colors,
        "edge_width": edge_width,
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
    sync_face_color: bool = True,
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
    if coords.shape[0] == 0 or (isinstance(face_color, str) and face_color.startswith("#")):
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
    layer.size = float(size)
    layer.symbol = symbol
    if hasattr(layer, "current_size"):
        layer.current_size = float(size)
    if hasattr(layer, "current_symbol"):
        layer.current_symbol = symbol
    if feature_color:
        # Recent Napari layer controls only expose a solid face-color swatch.
        # Keep colormap mode here; feature switching is done via our dock combo.
        try:
            if face_colormap:
                layer.face_colormap = face_colormap
            if face_contrast_limits is not None:
                layer.face_contrast_limits = face_contrast_limits
            layer.face_color = str(face_color)
            if hasattr(layer, "face_color_mode"):
                layer.face_color_mode = "colormap"
            if hasattr(layer, "refresh_colors"):
                try:
                    layer.refresh_colors(update_color_mapping=True)
                except TypeError:
                    layer.refresh_colors()
        except Exception:
            pass
        disconnect = install_points_style_sync(layer, sync_face_color=False)
    else:
        init_points_layer_style(layer, size=float(size), symbol=symbol, face_color=face_color)
        disconnect = install_points_style_sync(
            layer, sync_face_color=bool(sync_face_color)
        )
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
    face_key: str | None = None,
    point_size: float = 2.5,
) -> None:
    """Add PITC or PWV geometry overlays to Napari."""
    clear_hemo_geometry_layers(viewer)
    disconnectors: list[Any] = []
    paths: list[np.ndarray] = []
    path_colors: list[str] = []
    excluded_paths: list[np.ndarray] = []
    init_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    default_face = str(face_key or "quality")
    for region in regions:
        color = _REGION_COLORS.get(region.region_id, "#9467bd")
        extras = list(getattr(region, "root_init_extra_xyz", None) or [])
        init_points = extras if extras else [region.root_init_xyz]
        for ip, xyz in enumerate(init_points):
            init_rows.append(
                {
                    "centerline_x": float(xyz[0]),
                    "centerline_y": float(xyz[1]),
                    "centerline_z": float(xyz[2]),
                    "region_id": region.region_id,
                    "vessel_name": "root_init" if ip == 0 else f"root_init_{ip + 1}",
                    "vessel_id": int(region.root_label),
                    "station_index": int(ip),
                    "distance_mm": 0.0,
                }
            )
        for vessel in region.vessels.values():
            if vessel.polyline_oriented.shape[0] >= 2:
                paths.append(vessel.polyline_oriented.astype(np.float32))
                path_colors.append(color)
            for station in vessel.stations:
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
                        "pwv_bjornfoot_weighted_rms": float(
                            station.pwv_bjornfoot_weighted_rms
                        ),
                        "pwv_bjornfoot_delay_residual_s": float(
                            station.pwv_bjornfoot_delay_residual_s
                        ),
                        "pwv_bjornfoot_waveform_corr": float(
                            station.pwv_bjornfoot_waveform_corr
                        ),
                    }
                )
        for seg in region.excluded_segments:
            arr = to_numpy(seg).astype(np.float32, copy=False)
            if arr.shape[0] >= 2:
                excluded_paths.append(arr)
    _add_paths_layer(
        viewer,
        paths,
        path_colors,
        name=HEMO_PATHS_LAYER,
        reference_layer=reference_layer,
    )
    _add_paths_layer(
        viewer,
        excluded_paths,
        ["#d62728"] * len(excluded_paths),
        name=HEMO_EXCLUDED_LAYER,
        reference_layer=reference_layer,
        edge_width=0.25,
    )
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
        "pwv_bjornfoot_weighted_rms",
        "pwv_bjornfoot_delay_residual_s",
        "pwv_bjornfoot_waveform_corr",
    )
    station_coords, station_features = _stack_features(station_rows, station_features_all)
    cmap = "viridis"
    limits = _finite_limits(station_features, default_face)
    station_layer, station_disc = _add_points_layer(
        viewer,
        station_coords,
        station_features,
        name=HEMO_STATIONS_LAYER,
        reference_layer=reference_layer,
        size=float(point_size),
        symbol="disc",
        face_color=default_face if station_coords.shape[0] else "#d62728",
        face_colormap=cmap,
        face_contrast_limits=limits,
    )
    disconnectors.append(station_disc)
    if station_coords.shape[0] and default_face in station_feature_choices(station_layer):
        try:
            color_stations_by_feature(
                station_layer, default_face, mode=mode, cmap_name=cmap
            )
        except Exception:
            pass
    try:
        init_layer.mode = "pan_zoom"
        station_layer.mode = "pan_zoom"
    except Exception:
        pass
    setattr(
        viewer,
        "_nvitk_hemo_overlay_state",
        HemoOverlayState(
            disconnectors=disconnectors,
            station_layer=station_layer,
            mode=str(mode),
            default_face_key=str(default_face),
        ),
    )
    return station_layer


__all__ = [
    "HEMO_EXCLUDED_LAYER",
    "HEMO_INIT_LAYER",
    "HEMO_PATHS_LAYER",
    "HEMO_STATIONS_LAYER",
    "HEMO_PITC_STATIONS_LAYER",
    "HEMO_PWV_STATIONS_LAYER",
    "add_hemo_geometry_layers",
    "clear_hemo_geometry_layers",
    "color_stations_by_feature",
    "station_feature_choices",
]
