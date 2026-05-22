"""
Image I/O: :func:`imread` / :func:`imsave`, axis helpers, and optional format conversors.

Typical usage: ``from nvitk.io import imread, imsave`` or import conversors by name.
"""

from __future__ import annotations

from .conversors import dcm2nii, nikon2nifti, phase2volume, stl2nifti
from .imageio import convert_image, imread, imsave, imshow, swapaxes
from nvitk.viz.brainshow import brainshow

__all__ = [
    "imread",
    "imsave",
    "imshow",
    "brainshow",
    "swapaxes",
    "convert_image",
    "dcm2nii",
    "nikon2nifti",
    "phase2volume",
    "stl2nifti",
]
