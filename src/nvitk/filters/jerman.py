"""Jerman multiscale vesselness filter (IEEE TMI 2016).

Implements the enhancement function of Jerman et al. using Hessian
eigenvalues at multiple Gaussian scales. Unlike Frangi / Hybrid Hessian,
the response is regularized via ``tau`` for more uniform vessel intensity
across scales and contrast (useful for TOF-MRA preprocessing).

Reference
---------
T. Jerman, F. Pernuš, B. Likar, Ž. Špiclin,
"Enhancement of Vascular Structures in 3D and 2D Angiographic Images",
IEEE Trans. Med. Imaging, 35(9):2107–2118, 2016.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.filters.hessian import HESSIAN_SIGMAS_DEFAULT, parse_sigmas
from nvitk.types import Image

JERMAN_SIGMAS_DEFAULT: tuple[float, ...] = HESSIAN_SIGMAS_DEFAULT
JERMAN_TAU_DEFAULT: float = 0.5
_EIGVAL_TOL = 1e-10


def _jerman_response(
    lambda2: np.ndarray,
    lambda3: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """Vesselness at one scale from Hessian eigenvalues (eq. 13–14).

    Expects dark-ridge convention: tubular structures have ``lambda2, lambda3 > 0``.
    ``lambda2`` / ``lambda3`` are the two largest-|eigenvalue| components
    (for 2-D, pass ``lambda3 = lambda2``).
    """
    lambda2_raw = np.asarray(lambda2)
    lambda3_raw = np.asarray(lambda3)

    # Clip only for safe division; polarity / tau use raw values.
    lambda2_c = np.maximum(lambda2_raw, _EIGVAL_TOL)
    lambda3_c = np.maximum(lambda3_raw, _EIGVAL_TOL)

    tau_threshold = float(tau) * float(np.maximum(lambda3_raw, 0).max())
    lambda_rho = np.full_like(lambda3_c, tau_threshold, dtype=np.float64)
    lambda_rho[lambda3_raw <= 0] = 0.0
    mask_hi = lambda3_raw > tau_threshold
    lambda_rho[mask_hi] = lambda3_raw[mask_hi]

    denom = (lambda2_c + lambda_rho) ** 3
    response = (lambda2_c**2) * (lambda_rho - lambda2_c) * 27.0 / denom

    # Saturate to 1 where λ2 ≥ λρ/2 (ideal tubular geometry).
    response[np.logical_and(lambda2_raw >= lambda_rho / 2.0, lambda_rho > 0)] = 1.0
    response[np.logical_or(lambda2_raw <= 0, lambda_rho <= 0)] = 0.0
    response[~np.isfinite(response)] = 0.0
    return np.clip(response, 0.0, 1.0).astype(np.float64, copy=False)


def jerman_filter(
    image: Image | np.ndarray,
    *,
    sigmas: Iterable[float] | Sequence[float] | None = None,
    tau: float = JERMAN_TAU_DEFAULT,
    black_ridges: bool = True,
    mode: str = "reflect",
    cval: float = 0.0,
) -> Image | np.ndarray:
    """Jerman vesselness / ridge filter for 2-D or 3-D images.

    Parameters
    ----------
    image
        2-D or 3-D intensity volume (:class:`~nvitk.types.Image` or ndarray).
    sigmas
        Gaussian scales. Default ``(1, 3, 5, 7, 9)``.
    tau
        Regularization in ``[0.5, 1]``. Lower → stronger / more intense response.
    black_ridges
        Detect dark ridges when True; bright ridges when False (e.g. TOF vessels).
    mode, cval
        Boundary mode for Gaussian derivatives.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    if not (0.5 <= float(tau) <= 1.0):
        raise ValueError(f"`tau` must be in [0.5, 1.0], got {tau!r}")

    if isinstance(image, Image):
        arr = to_numpy(image.data)
        wrap = True
    else:
        arr = to_numpy(image)
        wrap = False

    if arr.ndim not in (2, 3):
        raise ValueError(f"jerman_filter expects 2-D or 3-D data, got {arr.ndim}D")

    scales = (
        tuple(float(s) for s in sigmas)
        if sigmas is not None
        else JERMAN_SIGMAS_DEFAULT
    )
    if not scales:
        raise ValueError("sigmas must be a non-empty sequence of positive floats")

    data = np.asarray(arr, dtype=np.float64)
    if not black_ridges:
        data = -data

    filtered_max = np.zeros_like(data, dtype=np.float64)
    for sigma in scales:
        eigvals = hessian_matrix_eigvals(
            hessian_matrix(
                data,
                float(sigma),
                mode=str(mode or "reflect"),
                cval=float(cval),
                use_gaussian_derivatives=True,
            )
        )
        # Sort by absolute magnitude: |λ1| ≤ |λ2| ≤ |λ3|.
        eigvals = np.take_along_axis(eigvals, np.abs(eigvals).argsort(0), 0)
        if data.ndim == 2:
            lambda2_raw = eigvals[1]
            lambda3_raw = lambda2_raw
        else:
            lambda2_raw, lambda3_raw = eigvals[1], eigvals[2]

        vals = _jerman_response(lambda2_raw, lambda3_raw, tau=float(tau))
        filtered_max = np.maximum(filtered_max, vals)

    out = np.asarray(filtered_max, dtype=np.float32)
    if wrap:
        return image.with_data(out)
    return out


__all__ = [
    "JERMAN_SIGMAS_DEFAULT",
    "JERMAN_TAU_DEFAULT",
    "jerman_filter",
    "parse_sigmas",
]
