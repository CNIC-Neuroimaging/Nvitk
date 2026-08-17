# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/measure/morpho/models.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Dataclasses and typed containers for the cleaned centerline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

@dataclass
class VesselInfo:
    """Static vessel metadata (naming, laterality, territory, flow topology) from the label map."""

    name: str
    full_name: str
    side: str
    pair: Optional[str]
    territory: str
    flow_from: str
    flow_to: List[str]
    no_upstream_start: Optional[str]


@dataclass
class StenosisResult:
    """Per-vessel stenosis detection: reference/minimum radius and flagged segments."""

    r_ref: float
    r_min: float
    percent_stenosis_max: float
    segments_s_mm: List[Tuple[float, float]]
    segments_point_idx: List[Tuple[int, int]]
    r_ref_per_point: np.ndarray


@dataclass
class EnlargementResult:
    """Per-vessel enlargement (aneurysm-like) detection: reference/max radius and flagged segments."""

    r_ref: float
    r_max: float
    percent_enlargement_max: float
    segments_s_mm: List[Tuple[float, float]]
    segments_point_idx: List[Tuple[int, int]]
    r_ref_per_point: np.ndarray


@dataclass
class SkeletonTree:
    """Skeleton graph: voxel points, adjacency/degree, endpoints/branchpoints, and root distances."""

    pts_vox: np.ndarray
    neighbors: List[List[int]]
    degree: np.ndarray
    endpoints: List[int]
    branchpoints: List[int]
    root: Optional[int]
    dist_from_root_mm: Optional[np.ndarray]
