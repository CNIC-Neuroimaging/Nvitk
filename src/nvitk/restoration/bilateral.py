"""
Backend-aware bilateral filtering.

Public API
----------
- :func:`bilateral`                  ─ high-level 2-D / 3-D dispatcher
- :func:`bilateral_2d`               ─ force a 2-D filter (with optional stack mode)
- :func:`bilateral_3d`               ─ force a true 3-D filter
- :func:`estimate_bilateral_parameters` ─ auto-selection helper for sigmas

Backend selection
-----------------
The concrete implementation is chosen from :func:`nvitk.core.backend.get_current_backend`:

- ``numpy``  → :func:`skimage.restoration.denoise_bilateral` (CPU, slice-by-slice for 3-D).
- ``cupy``   → custom CUDA raw kernels from :mod:`._cuda_kernels` (GPU, native 2-D/3-D).

Inputs can be raw ``numpy``/``cupy`` arrays or :class:`nvitk.types.Image`. The
returned type mirrors the input (Image → Image, array → array).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import get_current_backend, setup
from nvitk.types import Image

setup(globals())


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def estimate_bilateral_parameters(
    image: Any,
    *,
    spatial_pct: float = 0.02,
    spatial_min: float = 1.0,
    spatial_max: float = 5.0,
) -> Tuple[float, float]:
    """Return ``(sigma_spatial, sigma_color)`` for *image*.

    ``sigma_spatial`` is a fraction of the shortest spatial dimension (clamped
    to ``[spatial_min, spatial_max]``). ``sigma_color`` defaults to
    :func:`numpy.std` (legacy scikit-image convention).

    Accepts :class:`Image`, NumPy or CuPy arrays.
    """
    arr = as_backend_array(image.data if isinstance(image, Image) else image)
    shape = tuple(int(s) for s in arr.shape)
    min_dim = min(shape)

    sigma_spatial = max(spatial_min, min(min_dim * float(spatial_pct), spatial_max))
    sigma_color = float(to_numpy(arr.std()))
    if sigma_color <= 0.0:
        sigma_color = 1.0
    return float(sigma_spatial), float(sigma_color)


# ---------------------------------------------------------------------------
# CPU path: scikit-image
# ---------------------------------------------------------------------------


def _bilateral_cpu_2d(
    arr_np: Any,
    *,
    sigma_spatial: float,
    sigma_color: float,
) -> Any:
    """Apply skimage's ``denoise_bilateral`` to a 2-D NumPy slice."""
    from skimage.restoration import denoise_bilateral

    # skimage expects finite float arrays. ``channel_axis=None`` forces
    # grayscale treatment. ``mode='reflect'`` mirrors the borderless handling
    # in the CUDA kernel.
    return denoise_bilateral(
        arr_np.astype("float32", copy=False),
        sigma_color=float(sigma_color),
        sigma_spatial=float(sigma_spatial),
        channel_axis=None,
        mode="reflect",
    ).astype("float32", copy=False)


def _bilateral_cpu(
    arr_np: Any,
    *,
    sigma_spatial: float,
    sigma_color: float,
    do_3d: bool,
    axis: int,
) -> Any:
    """Dispatch CPU bilateral filtering to skimage (2-D or slice-by-slice 3-D)."""
    import numpy as _host_np

    if arr_np.ndim == 2:
        return _bilateral_cpu_2d(
            arr_np,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
        )

    if do_3d:
        # skimage has no native 3-D bilateral; fall back to per-slice along *axis*.
        # A true 3-D CPU filter is significantly slower; advise users to pick GPU.
        pass

    out = _host_np.empty_like(arr_np, dtype="float32")
    n = arr_np.shape[axis]
    for i in range(n):
        slicer = [slice(None)] * arr_np.ndim
        slicer[axis] = i
        slc = arr_np[tuple(slicer)]
        out[tuple(slicer)] = _bilateral_cpu_2d(
            slc,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
        )
    return out


# ---------------------------------------------------------------------------
# GPU path: custom CUDA raw kernels
# ---------------------------------------------------------------------------


def _launch_cuda_2d(
    arr_cp: Any,
    *,
    sigma_spatial: float,
    sigma_color: float,
    kernel_radius: int,
) -> Any:
    """Launch ``bilateral_filter_2d`` on a 2-D CuPy array."""
    from ._cuda_kernels import get_kernel

    import cupy as cp  # safe: we only get here when backend == "cupy"

    image = arr_cp.astype(cp.float32, copy=False)
    output = cp.empty_like(image)

    height, width = image.shape
    inv_spatial = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_color = 1.0 / (2.0 * sigma_color * sigma_color)

    block = (16, 16)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    get_kernel("bilateral_filter_2d")(
        grid,
        block,
        (
            image,
            output,
            int(height),
            int(width),
            int(kernel_radius),
            float(inv_spatial),
            float(inv_color),
        ),
    )
    return output


def _launch_cuda_3d(
    arr_cp: Any,
    *,
    sigma_spatial: float,
    sigma_color: float,
    kernel_radius: int,
) -> Any:
    """Launch ``bilateral_filter_3d`` on a 3-D CuPy array."""
    from ._cuda_kernels import get_kernel

    import cupy as cp

    image = arr_cp.astype(cp.float32, copy=False)
    output = cp.empty_like(image)

    depth, height, width = image.shape
    inv_spatial = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_color = 1.0 / (2.0 * sigma_color * sigma_color)

    block = (8, 8, 8)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        (depth + block[2] - 1) // block[2],
    )

    get_kernel("bilateral_filter_3d")(
        grid,
        block,
        (
            image,
            output,
            int(depth),
            int(height),
            int(width),
            int(kernel_radius),
            float(inv_spatial),
            float(inv_color),
        ),
    )
    return output


def _bilateral_gpu(
    arr_cp: Any,
    *,
    sigma_spatial: float,
    sigma_color: float,
    kernel_radius: int,
    do_3d: bool,
    axis: int,
) -> Any:
    import cupy as cp

    if arr_cp.ndim == 2:
        return _launch_cuda_2d(
            arr_cp,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            kernel_radius=kernel_radius,
        )

    if do_3d:
        return _launch_cuda_3d(
            arr_cp,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            kernel_radius=kernel_radius,
        )

    out = cp.empty_like(arr_cp, dtype=cp.float32)
    n = arr_cp.shape[axis]
    for i in range(n):
        slicer = [slice(None)] * arr_cp.ndim
        slicer[axis] = i
        slc = arr_cp[tuple(slicer)]
        out[tuple(slicer)] = _launch_cuda_2d(
            slc,
            sigma_spatial=sigma_spatial,
            sigma_color=sigma_color,
            kernel_radius=kernel_radius,
        )
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def bilateral(
    image: Image | Any,
    *,
    sigma_spatial: Optional[float] = None,
    sigma_color: Optional[float] = None,
    kernel_radius: Optional[int] = None,
    do_3d: bool = False,
    axis: int = 0,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> Image | Any:
    """Edge-preserving bilateral filter.

    Parameters
    ----------
    image
        2-D or 3-D input (:class:`Image` or raw array).
    sigma_spatial, sigma_color
        Bilateral parameters; auto-estimated from *image* when omitted
        (see :func:`estimate_bilateral_parameters`).
    kernel_radius
        Half-width of the spatial kernel. Defaults to ``ceil(2 * sigma_spatial)``.
    do_3d
        If True and the input is 3-D, run a native 3-D filter. If False, apply
        a 2-D filter slice-by-slice along *axis* (scikit-image semantics).
    axis
        Slicing axis for stack-by-stack 3-D filtering (ignored when ``do_3d``).
    backend
        Force ``"numpy"`` (skimage) or ``"cupy"`` (CUDA kernels); ``"auto"``
        (default) follows :func:`get_current_backend`.

    Returns
    -------
    Image | ndarray
        Same wrapping as *image* and same backend as the input (when
        ``backend='auto'``) or the requested backend otherwise.
    """
    if isinstance(image, Image):
        arr = image.data
    else:
        arr = image

    if arr.ndim not in (2, 3):
        raise ValueError(f"bilateral expects 2-D or 3-D input; got ndim={arr.ndim}")

    chosen = backend if backend != "auto" else get_current_backend()
    if chosen not in {"numpy", "cupy"}:
        raise ValueError(f"Unknown backend '{chosen}' (expected 'numpy' or 'cupy').")

    if sigma_spatial is None or sigma_color is None:
        est_spatial, est_color = estimate_bilateral_parameters(arr)
        if sigma_spatial is None:
            sigma_spatial = est_spatial
        if sigma_color is None:
            sigma_color = est_color

    if kernel_radius is None:
        kernel_radius = max(1, int(math.ceil(2.0 * float(sigma_spatial))))

    orig_dtype = arr.dtype

    if chosen == "cupy":
        arr_cp = as_backend_array(arr)
        out = _bilateral_gpu(
            arr_cp,
            sigma_spatial=float(sigma_spatial),
            sigma_color=float(sigma_color),
            kernel_radius=int(kernel_radius),
            do_3d=bool(do_3d),
            axis=int(axis),
        )
    else:
        arr_np = to_numpy(arr)
        out = _bilateral_cpu(
            arr_np,
            sigma_spatial=float(sigma_spatial),
            sigma_color=float(sigma_color),
            do_3d=bool(do_3d),
            axis=int(axis),
        )

    if out.dtype != orig_dtype:
        out = out.astype(orig_dtype, copy=False)

    if isinstance(image, Image):
        return image.with_data(out)
    return out


def bilateral_2d(
    image: Image | Any,
    *,
    sigma_spatial: Optional[float] = None,
    sigma_color: Optional[float] = None,
    kernel_radius: Optional[int] = None,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> Image | Any:
    """2-D bilateral filter (fails on 3-D inputs; use :func:`bilateral` instead)."""
    arr = image.data if isinstance(image, Image) else image
    if arr.ndim != 2:
        raise ValueError(f"bilateral_2d expects a 2-D array, got ndim={arr.ndim}")
    return bilateral(
        image,
        sigma_spatial=sigma_spatial,
        sigma_color=sigma_color,
        kernel_radius=kernel_radius,
        do_3d=False,
        axis=0,
        backend=backend,
    )


def bilateral_3d(
    image: Image | Any,
    *,
    sigma_spatial: Optional[float] = None,
    sigma_color: Optional[float] = None,
    kernel_radius: Optional[int] = None,
    backend: Literal["auto", "numpy", "cupy"] = "auto",
) -> Image | Any:
    """True 3-D bilateral filter. Falls back to slice-by-slice on the CPU backend."""
    arr = image.data if isinstance(image, Image) else image
    if arr.ndim != 3:
        raise ValueError(f"bilateral_3d expects a 3-D array, got ndim={arr.ndim}")
    return bilateral(
        image,
        sigma_spatial=sigma_spatial,
        sigma_color=sigma_color,
        kernel_radius=kernel_radius,
        do_3d=True,
        axis=0,
        backend=backend,
    )


__all__ = [
    "bilateral",
    "bilateral_2d",
    "bilateral_3d",
    "estimate_bilateral_parameters",
]
