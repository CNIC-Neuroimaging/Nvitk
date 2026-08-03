"""Morphological operators (dilation, erosion, opening, closing, fill-holes).

All functions are backend-aware (NumPy or CuPy), accept either an
:class:`~nvitk.types.Image` or a raw array, and return the same type as the
input (no in-place mutation of the caller's data).

This package re-exports the public API from :mod:`nvitk.morphology.binary` and
:mod:`nvitk.morphology._common`; implementation lives in those submodules.
"""

from __future__ import annotations

from ._common import make_ball_footprint
from .binary import close, dilate, erode, fill_holes, open
from .centerline import (
    compute_centerline_branches,
    compute_centerlines,
    compute_connected_centerline_tree,
    skeletonize_binary,
    skeletonize_labeled,
    unique_skeleton_edge_polylines,
)
from .centerline_siphon import (
    GenusReport,
    SiphonCorrectionResult,
    clean_ica_mask_after_centerline,
    clean_mask_geodesic_cl,
    compute_corrected_centerline,
    compute_mask_genus,
    correct_siphon_centerlines,
    prune_skeleton_shortest_arc,
    recover_lumen_thickness,
    recover_lumen_thickness_symmetric,
    refine_mask_lumen_gaps,
)
from .components import (
    keep_component_closest_to_center,
    keep_components_touching_seeds,
    keep_largest_components,
    label_connected,
    remove_small_components,
    remove_small_components_by_fraction,
)
from .mst_bridge import (
    bridge_binary_components_mst,
    draw_tube_3d,
    fill_multilabel_gaps_mst,
)

__all__ = [
    "GenusReport",
    "SiphonCorrectionResult",
    "clean_ica_mask_after_centerline",
    "clean_mask_geodesic_cl",
    "close",
    "compute_centerline_branches",
    "compute_centerlines",
    "compute_connected_centerline_tree",
    "compute_corrected_centerline",
    "unique_skeleton_edge_polylines",
    "compute_mask_genus",
    "correct_siphon_centerlines",
    "dilate",
    "erode",
    "fill_holes",
    "keep_component_closest_to_center",
    "keep_components_touching_seeds",
    "keep_largest_components",
    "label_connected",
    "make_ball_footprint",
    "bridge_binary_components_mst",
    "draw_tube_3d",
    "fill_multilabel_gaps_mst",
    "open",
    "prune_skeleton_shortest_arc",
    "recover_lumen_thickness",
    "recover_lumen_thickness_symmetric",
    "refine_mask_lumen_gaps",
    "remove_small_components",
    "remove_small_components_by_fraction",
    "skeletonize_binary",
    "skeletonize_labeled",
]
