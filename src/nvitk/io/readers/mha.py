"""MetaImage (``.mha`` / ``.mhd``) reader via SimpleITK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError

from .._common import RAS_LPS_FLIP, reorder_axes

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


def _build_affine(
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: tuple[float, ...],
) -> np.ndarray:
    """Build a voxel→world affine from MHA/MHD spacing, origin, and (row-major) direction cosines.

    The result maps ITK's ``(x, y, z)`` index order onto **RAS** millimetres, matching what
    :func:`~nvitk.io.readers.nifti.read_nifti` produces. ITK's own values are LPS, so the world
    axes are flipped on the way out; without that an image read from MetaImage and the same image
    read from NIfTI describe mirrored worlds while both claim to be affines, and anything
    comparing or overlaying them is quietly wrong about left and right.
    """
    dim = len(spacing)
    affine = np.eye(4, dtype=float)
    if dim == 0:
        return affine

    direction_matrix = np.asarray(direction, dtype=float).reshape(dim, dim)
    scale = np.diag(np.asarray(spacing, dtype=float))
    transform = direction_matrix @ scale

    affine[:dim, :dim] = transform
    affine[:dim, 3] = np.asarray(origin, dtype=float)
    return RAS_LPS_FLIP @ affine if dim == 3 else affine


def read_mha(path: str, *, axes: str | None = None, **_: Any):
    """Load MetaImage volume as numpy array with ``axes``, ``affine``, spacing, and origin metadata."""
    if sitk is None:
        raise BackendUnavailableError('SimpleITK is not installed. Please install it with "pip install SimpleITK".')

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    img = sitk.ReadImage(str(p))
    data = sitk.GetArrayFromImage(img)

    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())
    direction = tuple(float(v) for v in img.GetDirection())

    # SimpleITK hands back ZYX, but its spacing, origin and direction all index (x, y, z). Left
    # that way the array's axis 0 is Z while the affine's column 0 is X, so anything pairing them
    # positionally -- the GUI's axis table, for one -- reports each axis with another's direction
    # and voxel size. Transposing to XYZ makes the two agree, and matches what read_nifti returns
    # so the same volume read from either format lands in the same space with the same axis order.
    if data.ndim == 3:
        data = reorder_axes(data, "ZYX", "XYZ")
        axes_prev = "XYZ"
    else:
        axes_prev = "TZYX" if data.ndim == 4 else "".join(f"D{i}" for i in range(data.ndim))

    metadata: dict[str, Any] = {
        "axes": axes_prev,
        "shape": tuple(data.shape),
        # ITK's own values, kept in ITK's order and convention: the writer prefers them, which
        # makes a MetaImage round trip exact rather than a decomposition of the affine.
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "affine": _build_affine(spacing, origin, direction),
        "world": "ras",
    }

    if len(spacing) > 0:
        metadata["x_res"] = spacing[0]
    if len(spacing) > 1:
        metadata["y_res"] = spacing[1]
    if len(spacing) > 2:
        metadata["z_res"] = spacing[2]

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
