"""Dataclasses and typed containers for the cleaned centerline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

@dataclass
class VesselInfo:
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
    r_ref: float
    r_min: float
    percent_stenosis_max: float
    segments_s_mm: List[Tuple[float, float]]
    segments_point_idx: List[Tuple[int, int]]
    r_ref_per_point: np.ndarray


@dataclass
class EnlargementResult:
    r_ref: float
    r_max: float
    percent_enlargement_max: float
    segments_s_mm: List[Tuple[float, float]]
    segments_point_idx: List[Tuple[int, int]]
    r_ref_per_point: np.ndarray


@dataclass
class SkeletonTree:
    pts_vox: np.ndarray
    neighbors: List[List[int]]
    degree: np.ndarray
    endpoints: List[int]
    branchpoints: List[int]
    root: Optional[int]
    dist_from_root_mm: Optional[np.ndarray]
