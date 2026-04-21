"""
Geometric transform utilities for 3D/4D clinical imaging.

Submodules:

- :mod:`nvitk.transform.isotropy` — make an image voxel-isotropic by zooming the
  anisotropic axis.
- :mod:`nvitk.transform.resampling` — affine-based voxel resampling between two
  :class:`~nvitk.types.Image` grids.
- :mod:`nvitk.transform.rotation` — detect and correct rigid Z-rotation between
  image grids.
"""

from __future__ import annotations

from .isotropy import isotropy
from .resampling import resample_mask_to_pet, resample_pet_to_mask, resample_to
from .rotation import check_and_correct_rotation, correct_z_rotation

__all__ = [
    "isotropy",
    "resample_to",
    "resample_pet_to_mask",
    "resample_mask_to_pet",
    "correct_z_rotation",
    "check_and_correct_rotation",
]
