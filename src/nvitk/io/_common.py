"""Shared I/O helpers: format aliases, axis reordering, NIfTI axis labels, orientation from affine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import UnsupportedFormatError, ValidationError


# ──────────────────────────────────────────────────────────────────────────────
# Format aliases
# ──────────────────────────────────────────────────────────────────────────────

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
    "b2nd": "b2nd",
    "blosc2": "b2nd",
    "pkl": "pkl",
    "pickle": "pkl",
}


def normalize_type(force_type: str | None) -> str | None:
    """Map *force_type* aliases (``nii``, ``dcm``, …) to canonical reader names, or None."""
    if force_type is None:
        return None
    key = force_type.strip().lower().lstrip(".")
    return _TYPE_ALIASES.get(key, key)


def reorder_axes(data: Any, axes_prev: str, axes_new: str) -> Any:
    """
    Permute *data* so axis labels match *axes_new* (same multiset of letters as *axes_prev*).

    Raises
    ------
    ValidationError
        If lengths or letter sets differ from ``data.ndim``.
    """
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
    """File suffix for *path*, treating ``.nii.gz`` as a unit."""
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def guess_read_type(path: str | Path, force_type: str | None = None) -> str:
    """
    Resolve which reader registry key to use (``nifti``, ``dicom``, …).

    If *force_type* is set, it wins after :func:`normalize_type`. Otherwise uses extension;
    directories default to ``dicom`` unless they hold NIfTI or Blosc2 (``.b2nd``) volumes.
    """
    normalized = normalize_type(force_type)
    if normalized:
        return normalized

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(path))

    if p.is_dir():
        # Prefer NIfTI when the folder clearly contains volumes and no DICOM files.
        has_nifti = False
        has_dicom = False
        has_b2nd = False
        try:
            for child in p.iterdir():
                if not child.is_file():
                    continue
                name = child.name.lower()
                if name.endswith(".nii.gz") or name.endswith(".nii"):
                    has_nifti = True
                elif name.endswith(".dcm") or name == "dicomdir":
                    has_dicom = True
                elif name.endswith(".b2nd"):
                    has_b2nd = True
                if has_nifti and has_dicom:
                    break
        except OSError:
            pass
        if has_nifti and not has_dicom:
            return "nifti"
        if has_b2nd and not has_dicom and not has_nifti:
            return "b2nd"
        # By convention, reading a directory means DICOM source.
        return "dicom"

    suffix = _path_suffix(p).lstrip(".")
    out = _TYPE_ALIASES.get(suffix)
    if out:
        return out

    raise UnsupportedFormatError(f"Unsupported input format for path: {path}")


def guess_write_type(path: str | Path, force_type: str | None = None) -> str:
    """Like :func:`guess_read_type` but for output paths (writer registry keys)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Spatial metadata
# ──────────────────────────────────────────────────────────────────────────────


def default_nifti_axes(ndim: int) -> str:
    """Default axis label string (``XY``, ``XYZ``, ``XYZT``, …) for *ndim*."""
    if ndim == 2:
        return "XY"
    if ndim == 3:
        return "XYZ"
    if ndim == 4:
        return "XYZT"
    if ndim == 5:
        return "XYZCT"
    return "".join(f"D{i}" for i in range(ndim))


#: nibabel/NIfTI affines are RAS+; ITK (SimpleITK, MetaImage, NRRD) is LPS. The two differ by a
#: sign flip on the first two world axes, and the matrix is its own inverse, so one constant
#: serves both directions. Getting this wrong mirrors left against right without changing a
#: single voxel -- which for a segmentation whose classes are lateralised is the worst kind of
#: silent error, since the result still looks like an anatomically plausible mask.
RAS_LPS_FLIP: Any = np.diag([-1.0, -1.0, 1.0, 1.0])


def itk_geometry_from_affine(
    affine: Any, axes: str, *, world: str = "ras"
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Decompose a voxel-to-world *affine* into ITK ``(spacing, origin, direction)``.

    *affine* maps voxel indices **in the order named by** *axes* (so ``"XYZ"`` for an image read
    from NIfTI) onto world millimetres in the *world* convention. ITK indexes ``(x, y, z)`` and
    works in LPS, so the columns are permuted into that order and the world axes flipped when
    *world* is RAS.

    Returns
    -------
    tuple
        ``(spacing, origin, direction)`` ready for ``SetSpacing`` / ``SetOrigin`` /
        ``SetDirection``; *direction* is the row-major 3x3 as ITK wants it.

    Raises
    ------
    ValidationError
        On a degenerate affine -- a zero-length column has no direction to recover, and silently
        substituting one would put the volume somewhere arbitrary.
    """
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4):
        raise ValidationError(f"Expected a 4x4 affine, got {matrix.shape}.")
    if str(world).lower() == "ras":
        matrix = RAS_LPS_FLIP @ matrix

    spatial = [a for a in axes.upper() if a in "XYZ"]
    if len(spatial) != 3:
        raise ValidationError(f"Need three spatial axes to build an ITK geometry, got {axes!r}.")
    # ITK's index order is x, y, z; the affine's columns follow *axes*.
    order = [spatial.index(a) for a in "XYZ"]
    linear = matrix[:3, :3][:, order]

    spacing = np.linalg.norm(linear, axis=0)
    if not np.all(spacing > 0):
        raise ValidationError(f"Affine has a zero-length axis; spacing would be {spacing}.")
    direction = linear / spacing

    return (
        tuple(float(v) for v in spacing),
        tuple(float(v) for v in matrix[:3, 3]),
        tuple(float(v) for v in direction.reshape(-1)),
    )


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
