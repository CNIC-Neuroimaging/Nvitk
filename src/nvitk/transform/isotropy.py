"""Make an anisotropic 3D :class:`~nvitk.types.Image` voxel-isotropic via ``ndi.zoom``."""

from __future__ import annotations

from typing import Any

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def _resolve_spacing(image: Image) -> tuple[float, float, float]:
    """Return the 3-D voxel spacing (mm), raising if the image has no usable spacing metadata."""
    spacing = image.spacing
    if spacing is None or len(spacing) < 3:
        raise ValueError(
            "Image must expose a 3D spacing in its metadata "
            "(either 'spacing' or x_res/y_res/z_res)."
        )
    return float(spacing[0]), float(spacing[1]), float(spacing[2])


def isotropy(
    image: Image,
    *,
    axis: int | None = None,
    factor: float | None = None,
    order: int = 1,
    mode: str = "nearest",
    prefilter: bool = True,
) -> Image:
    """
    Resample *image* along the anisotropic axis to make voxels isotropic.

    Parameters
    ----------
    image
        3D :class:`~nvitk.types.Image` carrying a spacing triplet.
    axis
        Explicit axis (0, 1, 2) to resample. When omitted, uses the axis with
        the **largest** spacing (i.e. lowest sampling density), matching the
        legacy BioImaging behavior.
    factor
        Explicit zoom factor on the chosen axis. When omitted, uses
        ``max(spacing) / min(spacing)``.
    order
        Spline interpolation order for ``ndi.zoom``. Use 0 for label masks.
    mode
        Boundary mode (``'nearest'`` by default).
    prefilter
        Passed through to ``ndi.zoom``.

    Returns
    -------
    Image
        A new image with updated data and spacing/affine. Axes/orientation
        are preserved.
    """
    if image.ndim != 3:
        raise ValueError("isotropy() is defined for 3D images only.")

    x_res, y_res, z_res = _resolve_spacing(image)
    spacings = (x_res, y_res, z_res)

    if axis is None:
        axis = int(max(range(3), key=lambda i: spacings[i]))
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}.")

    max_dim = max(spacings)
    min_dim = min(spacings)
    if factor is None:
        if min_dim <= 0:
            raise ValueError(f"Invalid spacing {spacings}: spacings must be positive.")
        factor = float(max_dim / min_dim)

    zoom_factors: list[float] = [1.0, 1.0, 1.0]
    zoom_factors[axis] = float(factor)

    data = ndi.zoom(
        image.data, tuple(zoom_factors), order=order, mode=mode, prefilter=prefilter
    )

    out = image.with_data(data)

    new_spacing = list(spacings)
    new_spacing[axis] = float(spacings[axis] / factor)
    out.spacing = tuple(new_spacing)

    affine = image.affine
    if affine is not None:
        new_affine = to_numpy(affine).astype(float).copy()
        new_affine[:3, axis] = new_affine[:3, axis] / float(factor)
        out.metadata["affine"] = new_affine

    return out


__all__ = ["isotropy"]
