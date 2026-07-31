"""Hybrid Hessian vessel / ridge filter (``skimage.filters.hessian``)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.types import Image

HESSIAN_SIGMAS_DEFAULT: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)


def parse_sigmas(text: str | None, *, default: Sequence[float] = HESSIAN_SIGMAS_DEFAULT) -> tuple[float, ...]:
    """Parse comma/semicolon-separated sigma scales; empty → *default*."""
    if text is None or not str(text).strip():
        return tuple(float(s) for s in default)
    return tuple(float(x) for x in str(text).replace(";", ",").split(",") if x.strip())


def hessian_filter(
    image: Image | np.ndarray,
    *,
    sigmas: Iterable[float] | Sequence[float] | None = None,
    black_ridges: bool = True,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 15.0,
    mode: str = "reflect",
    cval: float = 0.0,
) -> Image | np.ndarray:
    """Hybrid Hessian filter (skimage) for ridges / vessels in 2-D or 3-D.

    Thin wrapper around :func:`skimage.filters.hessian`. Prefer *black_ridges*
    for dark vessels on a bright background; set ``False`` for bright ridges
    (e.g. TOF / CD vessels).

    Parameters
    ----------
    image
        2-D or 3-D intensity volume (:class:`~nvitk.types.Image` or ndarray).
    sigmas
        Gaussian scales. Default ``(1, 3, 5, 7, 9)`` (skimage uses ``range(1, 10, 2)``).
    black_ridges
        Detect dark ridges when True; bright ridges when False.
    alpha, beta, gamma
        Passed through to skimage (blob / plate / structure sensitivity).
    mode, cval
        Boundary mode for Gaussian derivatives.
    """
    from skimage.filters import hessian

    if isinstance(image, Image):
        arr = to_numpy(image.data)
        wrap = True
    else:
        arr = to_numpy(image)
        wrap = False

    if arr.ndim not in (2, 3):
        raise ValueError(f"hessian_filter expects 2-D or 3-D data, got {arr.ndim}D")

    scales = (
        tuple(float(s) for s in sigmas)
        if sigmas is not None
        else HESSIAN_SIGMAS_DEFAULT
    )
    if not scales:
        raise ValueError("sigmas must be a non-empty sequence of positive floats")

    out = hessian(
        np.asarray(arr, dtype=np.float64),
        sigmas=scales,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        black_ridges=bool(black_ridges),
        mode=str(mode or "reflect"),
        cval=float(cval),
    )
    out = np.asarray(out, dtype=np.float32)
    if wrap:
        return image.with_data(out)
    return out


__all__ = [
    "HESSIAN_SIGMAS_DEFAULT",
    "hessian_filter",
    "parse_sigmas",
]
