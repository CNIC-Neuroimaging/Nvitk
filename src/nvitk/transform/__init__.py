"""
Geometric transform utilities for 3D/4D clinical imaging.

Submodules:

- :mod:`nvitk.transform.isotropy` — make an image voxel-isotropic by zooming the
  anisotropic axis.
- :mod:`nvitk.transform.resampling` — affine-based voxel resampling between two
  :class:`~nvitk.types.Image` grids.
- :mod:`nvitk.transform.rotate` — rotate a volume around a spatial axis.
- :mod:`nvitk.transform.reorient` — permute / flip / match reference or mouse preset.
- :mod:`nvitk.transform.swap_axes` — swap or permute array axes.
- :mod:`nvitk.transform.rotation` — detect and correct rigid Z-rotation between
  image grids.
"""

from __future__ import annotations

from .isotropy import isotropy
from .oblique import oblique_slice
from .reorient import mouse_reorient_volume, reorient_volume
from .resampling import resample_mask_to_pet, resample_pet_to_mask, resample_to
from .rotate import rotate_volume
from .rotation import check_and_correct_rotation, correct_z_rotation
from .swap_axes import permute_axes, swap_axes

__all__ = [
    "isotropy",
    "oblique_slice",
    "resample_to",
    "resample_pet_to_mask",
    "resample_mask_to_pet",
    "rotate_volume",
    "reorient_volume",
    "mouse_reorient_volume",
    "permute_axes",
    "swap_axes",
    "correct_z_rotation",
    "check_and_correct_rotation",
]
