"""
CLI-style converters (DICOM→NIfTI, vendor-specific pipelines, STL, etc.).

Each submodule exposes a callable with its own parameters; see their docstrings.
"""

from __future__ import annotations

from .dcm2nii import dcm2nii
from .nikon2nifti import nikon2nifti
from .phase2volume import phase2volume
from .stl2nifti import stl2nifti

__all__ = [
    "dcm2nii",
    "nikon2nifti",
    "phase2volume",
    "stl2nifti",
]
