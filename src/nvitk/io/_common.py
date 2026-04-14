from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import UnsupportedFormatError, ValidationError


_TYPE_ALIASES = {
    "nii": "nifti",
    "nii.gz": "nifti",
    "nifti": "nifti",
    "dicom": "dicom",
    "dcm": "dicom",
    "tif": "tiff",
    "tiff": "tiff",
    "nd2": "nd2",
    "mha": "mha",
    "mhd": "mha",
    "png": "pil",
    "jpg": "pil",
    "jpeg": "pil",
    "bmp": "pil",
    "gif": "pil",
}


def normalize_type(force_type: str | None) -> str | None:
    if force_type is None:
        return None
    key = force_type.strip().lower().lstrip(".")
    return _TYPE_ALIASES.get(key, key)


def reorder_axes(data: Any, axes_prev: str, axes_new: str) -> Any:
    if axes_prev == axes_new:
        return data

    if len(axes_prev) != getattr(data, "ndim", -1):
        raise ValidationError(f"axes_prev '{axes_prev}' does not match data ndim={getattr(data, 'ndim', None)}")
    if len(axes_new) != getattr(data, "ndim", -1):
        raise ValidationError(f"axes_new '{axes_new}' does not match data ndim={getattr(data, 'ndim', None)}")
    if sorted(axes_prev) != sorted(axes_new):
        raise ValidationError(f"Cannot reorder axes from '{axes_prev}' to '{axes_new}'")

    perm = [axes_prev.index(ax) for ax in axes_new]
    try:
        return data.transpose(perm)
    except Exception:
        return np.transpose(data, perm)


def _path_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def guess_read_type(path: str | Path, force_type: str | None = None) -> str:
    normalized = normalize_type(force_type)
    if normalized:
        return normalized

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(path))

    if p.is_dir():
        # By convention, reading a directory means DICOM source.
        return "dicom"

    suffix = _path_suffix(p).lstrip(".")
    out = _TYPE_ALIASES.get(suffix)
    if out:
        return out

    raise UnsupportedFormatError(f"Unsupported input format for path: {path}")


def guess_write_type(path: str | Path, force_type: str | None = None) -> str:
    normalized = normalize_type(force_type)
    if normalized:
        return normalized

    p = Path(path)
    suffix = _path_suffix(p).lstrip(".")
    out = _TYPE_ALIASES.get(suffix)
    if out:
        return out

    raise UnsupportedFormatError(
        f"Unsupported output format for path: {path}. "
        "Use force_type='nifti'|'tiff'|'mha'|'pil'."
    )


def default_nifti_axes(ndim: int) -> str:
    if ndim == 2:
        return "XY"
    if ndim == 3:
        return "XYZ"
    if ndim == 4:
        return "XYZT"
    if ndim == 5:
        return "XYZCT"
    return "".join(f"D{i}" for i in range(ndim))


def orientation_codes_from_affine(affine: Any) -> str | None:
    """Return axis codes like ``\"RAS\"`` / ``\"LPS\"`` from a 4x4 voxel-to-world affine (nibabel)."""
    try:
        import nibabel as nib
    except Exception:
        return None
    aff = np.asarray(affine, dtype=float)
    if aff.shape != (4, 4):
        return None
    try:
        codes = nib.orientations.aff2axcodes(aff)
    except Exception:
        return None
    return "".join(str(c) for c in codes)