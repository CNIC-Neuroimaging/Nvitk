"""
Write volumes to NIfTI (and optional JSON sidecar / header extension) via nibabel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError, ValidationError

from .._common import default_nifti_axes, reorder_axes


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_NIFTI_RESERVED_KEYS = {
    "axes",
    "shape",
    "affine",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _extract_zooms(metadata: dict[str, Any], ndim: int) -> tuple[float, ...]:
    defaults = [1.0, 1.0, 1.0, 1.0]
    values = [
        metadata.get("x_res", metadata.get("dx")),
        metadata.get("y_res", metadata.get("dy")),
        metadata.get("z_res", metadata.get("dz")),
        metadata.get("t_res", metadata.get("temporal_resolution", metadata.get("dt"))),
    ]
    for i, value in enumerate(values):
        if value is None:
            continue
        try:
            defaults[i] = float(value)
        except Exception:
            continue
    return tuple(defaults[:ndim])


# ──────────────────────────────────────────────────────────────────────────────
# write_nifti
# ──────────────────────────────────────────────────────────────────────────────


def write_nifti(
    path: str,
    data: Any,
    *,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
    save_metadata_extension: bool = True,
    **_: Any,
) -> None:
    """
    Save *data* to *path* as NIfTI; writes a sibling ``.json`` when extra metadata is JSON-safe.

    Reorders from ``metadata['axes']`` to *axes* when both differ. Zooms come from ``x_res`` /
    ``y_res`` / ``z_res`` / ``t_res`` (or aliases). With ``save_metadata_extension=True``, a
    nibabel header extension may embed additional JSON metadata.
    """
    try:
        import nibabel as nib
    except Exception as exc:
        raise BackendUnavailableError('nibabel is not installed. Please install it with "pip install nibabel".') from exc

    metadata = dict(metadata or {})
    arr = to_numpy(data)

    axes_prev = metadata.get("axes", default_nifti_axes(arr.ndim))
    axes_new = axes or default_nifti_axes(arr.ndim)
    if len(axes_prev) != arr.ndim:
        raise ValidationError(f"axes='{axes_prev}' does not match data ndim={arr.ndim}")

    if axes_prev != axes_new:
        arr = reorder_axes(arr, axes_prev, axes_new)
        metadata["axes"] = axes_new
    else:
        metadata["axes"] = axes_prev

    affine = metadata.get("affine")
    if affine is None:
        affine = np.eye(4, dtype=float)
    else:
        affine = np.asarray(affine, dtype=float)
        if affine.shape != (4, 4):
            raise ValidationError(f"Affine matrix must have shape (4,4), got {affine.shape}")

    nifti_img = nib.Nifti1Image(arr, affine)
    zooms = _extract_zooms(metadata, arr.ndim)
    try:
        nifti_img.header.set_zooms(zooms)
    except Exception:
        pass

    if save_metadata_extension:
        payload = {
            str(k): _jsonable(v)
            for k, v in metadata.items()
            if k not in _NIFTI_RESERVED_KEYS and not str(k).startswith("_")
        }
        if payload:
            encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            # Use code 16 (comment) to keep nibabel interoperability for JSON payloads.
            nifti_img.header.extensions.append(nib.nifti1.Nifti1Extension(16, encoded))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, str(out))
