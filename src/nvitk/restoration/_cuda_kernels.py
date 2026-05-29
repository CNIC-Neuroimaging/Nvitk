"""
CUDA raw kernels for GPU-accelerated bilateral filtering.

These kernels are ported from the legacy BioImaging/src/imaging/filters/kernels.py
(see the ``nvitk.restoration.bilateral`` dispatcher for their public surface).

They are compiled lazily on first use so that importing this module does **not**
require a CUDA-capable CuPy install. Importers that need a kernel should go
through :func:`get_kernel`; importing ``cupy`` outside that path is avoided so
CPU-only installs remain functional.
"""

from __future__ import annotations

from typing import Dict


_BILATERAL_2D_SRC = r"""
extern "C" __global__
void bilateral_filter_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int height,
    const int width,
    const int kernel_radius,
    const float inv_sigma_spatial_sq,
    const float inv_sigma_color_sq
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int center_idx = y * width + x;
    float center_value = input[center_idx];
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
            int nx = x + kx;
            int ny = y + ky;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int neighbor_idx = ny * width + nx;
            float neighbor_value = input[neighbor_idx];

            float spatial_dist = (float)(kx * kx + ky * ky);
            float spatial_w = expf(-spatial_dist * inv_sigma_spatial_sq);

            float color_diff = center_value - neighbor_value;
            float color_exp = -(color_diff * color_diff) * inv_sigma_color_sq;
            if (color_exp < -50.0f) color_exp = -50.0f;
            float color_w = expf(color_exp);

            float w = spatial_w * color_w;
            if (isfinite(w) && w > 0.0f) {
                sum += neighbor_value * w;
                weight_sum += w;
            }
        }
    }

    if (weight_sum > 1e-10f && isfinite(sum) && isfinite(weight_sum)) {
        float r = sum / weight_sum;
        output[center_idx] = isfinite(r) ? r : center_value;
    } else {
        output[center_idx] = center_value;
    }
}
"""


_BILATERAL_3D_SRC = r"""
extern "C" __global__
void bilateral_filter_3d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int depth,
    const int height,
    const int width,
    const int kernel_radius,
    const float inv_sigma_spatial_sq,
    const float inv_sigma_color_sq
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    int center_idx = z * height * width + y * width + x;
    float center_value = input[center_idx];
    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int kz = -kernel_radius; kz <= kernel_radius; ++kz) {
        for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
            for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                int nz = z + kz;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height
                    || nz < 0 || nz >= depth) continue;

                int neighbor_idx = nz * height * width + ny * width + nx;
                float neighbor_value = input[neighbor_idx];

                float spatial_dist = (float)(kx * kx + ky * ky + kz * kz);
                float spatial_w = expf(-spatial_dist * inv_sigma_spatial_sq);

                float color_diff = center_value - neighbor_value;
                float color_exp = -(color_diff * color_diff) * inv_sigma_color_sq;
                if (color_exp < -50.0f) color_exp = -50.0f;
                float color_w = expf(color_exp);

                float w = spatial_w * color_w;
                if (isfinite(w) && w > 0.0f) {
                    sum += neighbor_value * w;
                    weight_sum += w;
                }
            }
        }
    }

    if (weight_sum > 1e-10f && isfinite(sum) && isfinite(weight_sum)) {
        float r = sum / weight_sum;
        output[center_idx] = isfinite(r) ? r : center_value;
    } else {
        output[center_idx] = center_value;
    }
}
"""


_KERNEL_SOURCES: dict[str, tuple[str, str]] = {
    "bilateral_filter_2d": (_BILATERAL_2D_SRC, "bilateral_filter_2d"),
    "bilateral_filter_3d": (_BILATERAL_3D_SRC, "bilateral_filter_3d"),
}


_KERNEL_CACHE: Dict[str, object] = {}


def get_kernel(name: str):
    """Compile and cache a CuPy raw kernel by *name*.

    Raises :class:`RuntimeError` if CuPy is not available.
    """
    if name in _KERNEL_CACHE:
        return _KERNEL_CACHE[name]

    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError(
            "CuPy is required for GPU bilateral filtering. "
            "Install with `pip install cupy-cuda12x` (or matching your CUDA)."
        ) from exc

    try:
        src, entry = _KERNEL_SOURCES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown bilateral kernel '{name}'. Available: {sorted(_KERNEL_SOURCES)}") from exc

    kernel = cp.RawKernel(src, entry)
    _KERNEL_CACHE[name] = kernel
    return kernel


__all__ = ["get_kernel"]
