"""Shared helpers for qvtpy stages 3–6 (no package-level re-exports).

Submodules (import explicitly):

- :mod:`eicab_masks` — resolve CW/WB eICAB NIfTI paths.
- :mod:`flow_volume_masks` — global CD sliding-threshold vessel mask.
- :mod:`mask_cleaning` — multilabel island removal and area opening.
- :mod:`venous_heuristics` — geometry-based venous sinus centerlines.
- :mod:`centerline_io` — load/write centerline masks and polylines.
- :mod:`vessel_cd_segmentation` — per-vessel local ``seg_4dflow`` builder.
- :mod:`aca_sequential_grow` — sequential ACA region growing.
- :mod:`vertebral_split` — basilar → LVA/RVA via inferior VB centerline bifurcation.
- :mod:`loc_selection` — QVTplus-style LOC placement.
- :mod:`cross_section` — re-export of :mod:`nvitk.measure.cross_section`.
- :mod:`measure_qc` — stage-6 cross-section QC PNGs.
"""

from __future__ import annotations

__all__ = [
    "cross_section",
    "eicab_masks",
    "flow_volume_masks",
    "loc_selection",
    "mask_cleaning",
    "venous_heuristics",
    "vessel_cd_segmentation",
]
