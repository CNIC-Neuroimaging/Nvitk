"""
Low-level readers by format (also invoked via :mod:`nvitk.io.imageio`).

Prefer :func:`nvitk.io.imread` unless you need a specific reader directly.
"""

from __future__ import annotations

from .b2nd import b2nd_properties_path, load_b2nd_properties, read_b2nd
from .dicom import read_dicom
from .mha import read_mha
from .nd2 import read_nd2
from .nifti import nifti_metadata_json_path, read_nifti
from .pil import read_pil
from .pkl import read_pkl
from .tiff import read_tiff

__all__ = [
    "read_nifti",
    "nifti_metadata_json_path",
    "read_dicom",
    "read_tiff",
    "read_nd2",
    "read_mha",
    "read_pil",
    "read_b2nd",
    "read_pkl",
    "b2nd_properties_path",
    "load_b2nd_properties",
]
