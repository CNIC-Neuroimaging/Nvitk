"""Visualization helpers."""

from __future__ import annotations

from .ct_windows import (
    CT_WINDOWS,
    DEFAULT_WINDOW_KEY,
    CTWindow,
    get_window,
    limits_for,
    suggest_window,
    window_from_limits,
    window_keys,
)

from .atlas_sources import desikan_atlas_path, vascular_atlas_path
from .brainshow import (
    ResolvedAtlas,
    atlas_indices_for_region,
    brainshow,
    build_index_to_value,
    build_volume_stat_image,
    normalize_region_key,
    resolve_atlas,
)
from .pet_hotspots import HotspotMode, show_hotspots
from .flowshow import (
    FlowshowAnimationOptions,
    FlowshowVectorOptions,
    VectorColorMode,
    flowshow,
)
from .streamlines import (
    ColorMetric,
    FlowTraceParams,
    StreamlineParams,
    compute_pathlines,
    compute_streamlines,
    resample_polyline,
    sample_vel_trilinear,
    stream_seed_cloud,
    streamline_mean_speeds,
    vertex_scalars_for_polylines,
)

__all__ = [
    "CT_WINDOWS",
    "DEFAULT_WINDOW_KEY",
    "CTWindow",
    "get_window",
    "limits_for",
    "suggest_window",
    "window_from_limits",
    "window_keys",
    "brainshow",
    "ResolvedAtlas",
    "atlas_indices_for_region",
    "desikan_atlas_path",
    "normalize_region_key",
    "resolve_atlas",
    "vascular_atlas_path",
    "build_index_to_value",
    "build_volume_stat_image",
    "show_hotspots",
    "HotspotMode",
    "flowshow",
    "FlowshowVectorOptions",
    "FlowshowAnimationOptions",
    "VectorColorMode",
    "ColorMetric",
    "FlowTraceParams",
    "StreamlineParams",
    "compute_pathlines",
    "compute_streamlines",
    "resample_polyline",
    "sample_vel_trilinear",
    "stream_seed_cloud",
    "streamline_mean_speeds",
    "vertex_scalars_for_polylines",
]
