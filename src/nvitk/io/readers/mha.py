from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError

from .._common import reorder_axes

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


def _build_affine(
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    direction: tuple[float, ...],
) -> np.ndarray:
    dim = len(spacing)
    affine = np.eye(4, dtype=float)
    if dim == 0:
        return affine

    direction_matrix = np.asarray(direction, dtype=float).reshape(dim, dim)
    scale = np.diag(np.asarray(spacing, dtype=float))
    transform = direction_matrix @ scale

    affine[:dim, :dim] = transform
    affine[:dim, 3] = np.asarray(origin, dtype=float)
    return affine


def read_mha(path: str, *, axes: str | None = None, **_: Any):
    if sitk is None:
        raise BackendUnavailableError('SimpleITK is not installed. Please install it with "pip install SimpleITK".')

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    img = sitk.ReadImage(str(p))
    data = sitk.GetArrayFromImage(img)

    # SimpleITK returns ZYX for 3D arrays.
    axes_prev = "ZYX" if data.ndim == 3 else ("TZYX" if data.ndim == 4 else "".join(f"D{i}" for i in range(data.ndim)))
    spacing = tuple(float(v) for v in img.GetSpacing())
    origin = tuple(float(v) for v in img.GetOrigin())
    direction = tuple(float(v) for v in img.GetDirection())

    metadata: dict[str, Any] = {
        "axes": axes_prev,
        "shape": tuple(data.shape),
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "affine": _build_affine(spacing, origin, direction),
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
