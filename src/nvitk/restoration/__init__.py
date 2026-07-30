"""Image restoration / denoising utilities.

Exposes:

* :func:`bilateral` — backend-aware bilateral filter (skimage CPU / CUDA GPU).
* :func:`n4_bias_field_correction` — ANTsPy N4 bias-field correction.
"""

from __future__ import annotations

from .bilateral import (
    bilateral,
    bilateral_2d,
    bilateral_3d,
    estimate_bilateral_parameters,
)
from .n4_bias import n4_bias_field_correction

__all__ = [
    "bilateral",
    "bilateral_2d",
    "bilateral_3d",
    "estimate_bilateral_parameters",
    "n4_bias_field_correction",
]
