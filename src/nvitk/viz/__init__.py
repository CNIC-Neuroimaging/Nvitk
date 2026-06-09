"""Visualization helpers."""

from __future__ import annotations

from .brainshow import (
    ResolvedAtlas,
    brainshow,
    build_index_to_value,
    build_volume_stat_image,
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
    "brainshow",
    "ResolvedAtlas",
    "resolve_atlas",
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
