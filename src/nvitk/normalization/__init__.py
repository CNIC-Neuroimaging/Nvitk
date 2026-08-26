"""Intensity normalisation utilities.

:mod:`~nvitk.normalization.intensity` harmonises calibrated (CT/HU) and uncalibrated
(MR/TOF) volumes onto a shared range, so a mixed-modality dataset can be trained through a
single input channel. PET SUV scaling still lives in :mod:`nvitk.measure.suv`; further
normalisation helpers (white-stripe, histogram matching) belong here as they are factored out
of the pipelines.
"""

from __future__ import annotations

from .intensity import (
    CTA_WINDOW,
    MR_PERCENTILES,
    TARGET_RANGE,
    harmonize_modality,
    robust_scale,
    window_ct,
)

__all__ = [
    "CTA_WINDOW",
    "MR_PERCENTILES",
    "TARGET_RANGE",
    "harmonize_modality",
    "robust_scale",
    "window_ct",
]
