"""Image restoration / denoising utilities.

Exposes:

* :func:`bilateral` — backend-aware bilateral filter (skimage CPU / CUDA GPU).
* :func:`n4_bias_field_correction` — ANTsPy N4 bias-field correction.
* :func:`mri_super_resolution` — ANTsPyNet MRI super-resolution.
"""

from __future__ import annotations

from .bilateral import (
    bilateral,
    bilateral_2d,
    bilateral_3d,
    estimate_bilateral_parameters,
)
from .mri_super_resolution import (
    MRI_SR_EXPANSION_FACTORS,
    MRI_SR_FEATURES,
    mri_super_resolution,
)
from .n4_bias import n4_bias_field_correction

__all__ = [
    "MRI_SR_EXPANSION_FACTORS",
    "MRI_SR_FEATURES",
    "bilateral",
    "bilateral_2d",
    "bilateral_3d",
    "estimate_bilateral_parameters",
    "mri_super_resolution",
    "n4_bias_field_correction",
]
