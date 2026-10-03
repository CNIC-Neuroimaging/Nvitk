"""
CLI-style converters (DICOM→NIfTI, vendor-specific pipelines, STL, etc.).

Each submodule exposes a callable with its own parameters; see their docstrings.
"""

from __future__ import annotations

from ._dicom_phases import stack_cardiac_phases
from ._dicom_spectral import classify_spectral, stack_monoenergetic
from .dcm2nii import dcm2nii
from .nikon2nifti import nikon2nifti
from .phase2volume import phase2volume
from .stl2nifti import stl2nifti

__all__ = [
    "dcm2nii",
    "nikon2nifti",
    "phase2volume",
    "classify_spectral",
    "stack_cardiac_phases",
    "stack_monoenergetic",
    "stl2nifti",
]
