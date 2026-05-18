"""Light smoothing of native black-blood volumes before lumen segmentation."""

from __future__ import annotations

from typing import Literal

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup

setup(globals())

VwiPreprocess = Literal["none", "median", "gaussian"]


def preprocess_vwi_bb(
    wvi: np.ndarray,
    method: VwiPreprocess = "median",
    *,
    median_size: int = 3,
    gaussian_sigma: float = 0.8,
) -> np.ndarray:
    """Return a smoothed copy of *wvi* for segmentation (original not modified)."""
    if method == "none":
        return as_backend_array(wvi).astype(np.float64, copy=True)
    vol = as_backend_array(wvi).astype(np.float64, copy=False)
    if method == "median":
        size = int(median_size)
        if size < 1:
            return vol.copy()
        if size % 2 == 0:
            size += 1
        return as_backend_array(
            ndi.median_filter(vol, size=size, mode="nearest")
        ).astype(np.float64, copy=False)
    if method == "gaussian":
        sigma = float(gaussian_sigma)
        if sigma <= 0.0:
            return vol.copy()
        return as_backend_array(
            ndi.gaussian_filter(vol, sigma=sigma, mode="nearest")
        ).astype(np.float64, copy=False)
    raise ValueError(f"Unknown preprocess method: {method!r}")
