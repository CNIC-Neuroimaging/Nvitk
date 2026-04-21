"""Detect and correct rigid Z-rotation misalignment between image grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as _host_np

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


@dataclass
class RotationAnalysis:
    """Result of :func:`check_and_correct_rotation`."""

    z_rotation_degrees: float
    needs_correction: bool
    recommendation: str
    rotation_matrix: _host_np.ndarray
    corrected_image: Image | None = None


def _rotate_affine_z(affine: _host_np.ndarray, degrees: float) -> _host_np.ndarray:
    angle_rad = float(_host_np.radians(-degrees))
    cos_a = float(_host_np.cos(angle_rad))
    sin_a = float(_host_np.sin(angle_rad))
    rot = _host_np.array(
        [
            [cos_a, -sin_a, 0.0, 0.0],
            [sin_a, cos_a, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot @ _host_np.asarray(affine, dtype=float)


def correct_z_rotation(image: Image | None, affine: _host_np.ndarray, rotation_degrees: float):
    """
    Rotate *image* and its *affine* by ``-rotation_degrees`` around the Z axis.

    Parameters
    ----------
    image
        Optional :class:`~nvitk.types.Image` to rotate. If ``None`` only the
        affine is updated.
    affine
        4x4 voxel-to-world affine (NumPy).
    rotation_degrees
        Rotation to undo (the function applies ``-rotation_degrees``). A fast
        path is used for |180°|.

    Returns
    -------
    (Image | None, np.ndarray)
        Rotated image (or ``None``) and updated affine.
    """
    corrected_affine = _rotate_affine_z(affine, rotation_degrees)

    if image is None:
        return None, corrected_affine

    if abs(rotation_degrees - 180.0) < 0.01 or abs(rotation_degrees + 180.0) < 0.01:
        flipped = np.flip(np.flip(image.data, axis=0), axis=1)
    else:
        flipped = ndi.rotate(
            image.data,
            -float(rotation_degrees),
            axes=(0, 1),
            reshape=False,
            order=3,
            mode="constant",
            cval=0,
        )

    out = image.with_data(flipped)
    out.metadata["affine"] = corrected_affine
    return out, corrected_affine


def check_and_correct_rotation(
    pet_affine: _host_np.ndarray | Any,
    mri_affine: _host_np.ndarray | Any,
    pet_image: Image | None = None,
    *,
    threshold_deg: float = 5.0,
) -> RotationAnalysis:
    """
    Compare two affines and optionally correct Z-rotation on *pet_image*.

    Uses the normalized-column relative rotation ``R_mri @ R_pet.T``; the Z
    rotation is the ``arctan2`` of the 2x2 upper-left block. A near-180°
    reflection case is detected via the signs of ``R_relative``.

    Parameters
    ----------
    pet_affine, mri_affine
        4x4 voxel-to-world affines. CuPy arrays are materialized to NumPy.
    pet_image
        Optional PET image to correct when rotation exceeds *threshold_deg*.
    threshold_deg
        Tolerance (degrees) under which correction is skipped.
    """
    pet_affine = to_numpy(pet_affine)
    mri_affine = to_numpy(mri_affine)

    R_pet = _host_np.asarray(pet_affine[:3, :3].copy(), dtype=float)
    R_mri = _host_np.asarray(mri_affine[:3, :3].copy(), dtype=float)

    for i in range(3):
        R_pet[:, i] = R_pet[:, i] / _host_np.linalg.norm(R_pet[:, i])
        R_mri[:, i] = R_mri[:, i] / _host_np.linalg.norm(R_mri[:, i])

    R_relative = R_mri @ R_pet.T

    z_rotation_rad = _host_np.arctan2(R_relative[1, 0], R_relative[0, 0])
    z_rotation_deg = float(_host_np.degrees(z_rotation_rad))

    if R_relative[0, 0] < -0.9 and R_relative[1, 1] < -0.9:
        z_rotation_deg = 180.0

    needs_correction = abs(z_rotation_deg) > threshold_deg

    if abs(z_rotation_deg - 180.0) < threshold_deg or abs(z_rotation_deg + 180.0) < threshold_deg:
        recommendation = (
            "180 degree Z-axis rotation detected. This is likely due to different "
            "scanner conventions. Apply correction before resampling for accurate "
            "SUV measurements."
        )
        z_rotation_deg = 180.0
        needs_correction = True
    elif needs_correction:
        recommendation = (
            f"{z_rotation_deg:.1f} degree Z-axis rotation detected. "
            "Consider applying rotation correction for better alignment."
        )
    else:
        recommendation = "Minimal rotation detected. No correction needed."

    corrected_image: Image | None = None
    if pet_image is not None and needs_correction:
        corrected_image, _ = correct_z_rotation(pet_image, pet_affine, z_rotation_deg)

    return RotationAnalysis(
        z_rotation_degrees=z_rotation_deg,
        needs_correction=needs_correction,
        recommendation=recommendation,
        rotation_matrix=R_relative,
        corrected_image=corrected_image,
    )


__all__ = [
    "RotationAnalysis",
    "correct_z_rotation",
    "check_and_correct_rotation",
]
