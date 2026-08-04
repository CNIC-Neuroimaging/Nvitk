"""Internal helpers for the :mod:`nvitk.morphology` module."""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.backend import get_current_backend, setup, to_cupy, to_numpy
from nvitk.types import Image

setup(globals())


def _as_array(obj: Image | Any) -> Any:
    """Unwrap an :class:`Image` to its voxel array; pass raw arrays through."""
    return obj.data if isinstance(obj, Image) else obj


def _wrap_like(original: Image | Any, data: Any) -> Image | Any:
    """Re-wrap *data* as an :class:`Image` when *original* was one; else return *data*."""
    if isinstance(original, Image):
        return original.with_data(data)
    return data


def _coerce_to_current_backend(arr: Any) -> Any:
    """Move *arr* onto the currently-selected NumPy/CuPy backend."""
    backend = get_current_backend()
    if backend == "numpy":
        return to_numpy(arr)
    return to_cupy(arr)


def _iterate_structure_hostside(structure: Any, iterations: int) -> Any:
    """Run :func:`scipy.ndimage.iterate_structure` on NumPy then move back."""
    from scipy.ndimage import iterate_structure as _iter

    host = to_numpy(structure).astype(bool)
    grown = _iter(host, iterations=iterations)
    if get_current_backend() == "cupy":
        return to_cupy(grown.astype("uint8")).astype(bool)
    return grown


def make_ball_footprint(
    ndim: int,
    radius: int = 1,
    *,
    connectivity: int = 2,
) -> Any:
    """Return an ``ndi.generate_binary_structure``-based ball of *radius*.

    The result is on the current backend.
    """
    struct = ndi.generate_binary_structure(ndim, connectivity)
    if radius <= 1:
        return struct
    return _iterate_structure_hostside(struct, radius)


def _resolve_structure(
    ndim: int,
    footprint: int | Any | None,
    *,
    connectivity: int,
    isotropic: bool,
    spacing: Sequence[float] | None,
) -> Any:
    """Produce the final structuring element for morphology ops."""
    if footprint is None:
        return ndi.generate_binary_structure(ndim, connectivity)
    if isinstance(footprint, int):
        if isotropic:
            if spacing is None or len(spacing) != ndim:
                raise ValueError(
                    "isotropic=True requires spacing of length ndim "
                    f"(got ndim={ndim}, spacing={spacing})."
                )
            base = ndi.generate_binary_structure(ndim, connectivity)
            grown = _iterate_structure_hostside(base, footprint)
            max_s = max(spacing)
            zoom = tuple(max_s / float(s) for s in spacing)
            zoomed = ndi.zoom(grown.astype("uint8"), zoom=zoom, order=0)
            return zoomed > 0
        base = ndi.generate_binary_structure(ndim, connectivity)
        return _iterate_structure_hostside(base, footprint)
    return _coerce_to_current_backend(footprint)


__all__ = [
    "_as_array",
    "_wrap_like",
    "_coerce_to_current_backend",
    "_iterate_structure_hostside",
    "_resolve_structure",
    "make_ball_footprint",
]
