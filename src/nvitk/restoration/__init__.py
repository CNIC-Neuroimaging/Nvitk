"""Image restoration / denoising utilities.

Currently exposes a backend-aware :func:`bilateral` filter:

* On the NumPy backend, it defers to :mod:`skimage.restoration.denoise_bilateral`.
* On the CuPy backend, it dispatches to custom CUDA raw kernels (2-D/3-D, with
  optional shared-memory variants) defined in :mod:`nvitk.restoration._cuda_kernels`.
"""

from __future__ import annotations

from .bilateral import (
    bilateral,
    bilateral_2d,
    bilateral_3d,
    estimate_bilateral_parameters,
)

__all__ = [
    "bilateral",
    "bilateral_2d",
    "bilateral_3d",
    "estimate_bilateral_parameters",
]
