"""
Writers for NIfTI, TIFF, MetaImage, and PIL-supported formats (used by :func:`nvitk.io.imsave`).
"""

from __future__ import annotations

from .mha import write_mha
from .nifti import write_nifti
from .pil import write_pil
from .tiff import write_tiff

__all__ = [
    "write_nifti",
    "write_tiff",
    "write_mha",
    "write_pil",
]
