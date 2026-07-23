"""eICAB-inspired distal vessel tree — thin re-export of :mod:`nvitk.segmentation.blood_flood`.

Kept for qvtpy import stability. New code should import from
``nvitk.segmentation.blood_flood`` directly.
"""

from __future__ import annotations

from nvitk.segmentation.blood_flood import (
    FRANGI_SIGMAS_DEFAULT as _DISTAL_FRANGI_SIGMAS_DEFAULT,
    HYST_HIGH_FACTOR_DEFAULT as _DISTAL_HYST_HIGH_FACTOR_DEFAULT,
    HYST_LOW_FACTOR_DEFAULT as _DISTAL_HYST_LOW_FACTOR_DEFAULT,
    apply_hysteresis_threshold_3d,
    cd_vesselness,
    hysteresis_vessel_tree,
    keep_tree_components_touching_markers,
    thicken_tree_in_cd,
    watershed_labels_into_vessels,
)

__all__ = [
    "apply_hysteresis_threshold_3d",
    "cd_vesselness",
    "hysteresis_vessel_tree",
    "keep_tree_components_touching_markers",
    "thicken_tree_in_cd",
    "watershed_labels_into_vessels",
    "_DISTAL_FRANGI_SIGMAS_DEFAULT",
    "_DISTAL_HYST_HIGH_FACTOR_DEFAULT",
    "_DISTAL_HYST_LOW_FACTOR_DEFAULT",
]
