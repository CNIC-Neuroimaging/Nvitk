"""
Backward-compatible re-exports for brain visualization helpers.

Prefer importing from :mod:`nvitk.viz.brainshow`.
"""

from __future__ import annotations

from nvitk.viz.brainshow import (
    ResolvedAtlas,
    brainshow,
    build_index_to_value,
    build_volume_stat_image,
    resolve_atlas,
)

__all__ = [
    "brainshow",
    "ResolvedAtlas",
    "resolve_atlas",
    "build_index_to_value",
    "build_volume_stat_image",
]
