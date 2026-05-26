"""Affine voxel-to-voxel resampling between two :class:`~nvitk.types.Image` grids."""

from __future__ import annotations

from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def _require_affine(image: Image, name: str) -> np.ndarray:
    affine = image.affine
    if affine is None:
        raise ValueError(f"Image '{name}' has no affine in its metadata.")
    affine = to_numpy(affine).astype(float)
    if affine.shape != (4, 4):
        raise ValueError(f"Affine for '{name}' must be (4, 4); got {affine.shape}.")
    return affine


def resample_to(
    source: Image,
    target: Image,
    *,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool | None = None,
) -> Image:
    """
    Resample *source* onto *target*'s voxel grid via affine composition.

    The mapping is ``T = inv(target.affine) @ source.affine``; scipy's
    ``affine_transform`` expects the output→input map so we invert it.

    Parameters
    ----------
    source
        Image whose voxels are resampled.
    target
        Image whose grid (shape + affine) defines the output.
    order
        Spline interpolation order. **Use 0 for label/mask data** (nearest
        neighbor) and 1 for intensity data.
    mode, cval
        Boundary handling passed to ``ndi.affine_transform``.
    prefilter
        Defaults to ``False`` when ``order == 0`` and ``True`` otherwise.

    Returns
    -------
    Image
        A new image whose data lives on ``target``'s grid and carries
        ``target``'s affine/spacing/orientation but ``source``'s other metadata.
    """
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("resample_to currently supports 3D images only.")

    aff_source = _require_affine(source, "source")
    aff_target = _require_affine(target, "target")

    # Host-side 4x4 linear algebra; cheap.
    aff_target_inv = np.linalg.inv(aff_target)
    transform_matrix = aff_target_inv @ aff_source
    inv_linear = np.linalg.inv(transform_matrix[:3, :3])
    inv_offset = -inv_linear @ transform_matrix[:3, 3]

    if prefilter is None:
        prefilter = order != 0

    resampled = ndi.affine_transform(
        source.data,
        matrix=as_backend_array(inv_linear),
        offset=as_backend_array(inv_offset),
        output_shape=target.shape,
        order=int(order),
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    md = dict(source.metadata or {})
    target_md = target.metadata or {}
    md["affine"] = to_numpy(aff_target).astype(float)
    for key in ("spacing", "x_res", "y_res", "z_res", "orientation"):
        if key in target_md:
            md[key] = target_md[key]
        else:
            md.pop(key, None)
    md["shape"] = tuple(getattr(resampled, "shape", ()))

    return Image(
        data=resampled,
        metadata=md,
        axes=source.axes if source.axes is not None else target.axes,
        name=source.name,
        orientation=target.orientation,
    )


def resample_pet_to_mask(pet: Image, mask: Image, *, order: int = 1) -> Image:
    """Trilinear resample of a PET image onto a mask's grid."""
    return resample_to(pet, mask, order=order, prefilter=order != 0)


def resample_mask_to_pet(mask: Image, pet: Image, *, order: int = 0) -> Image:
    """Nearest-neighbor resample of a mask onto a PET image's grid."""
    return resample_to(mask, pet, order=order, prefilter=False)


__all__ = [
    "resample_to",
    "resample_pet_to_mask",
    "resample_mask_to_pet",
]
