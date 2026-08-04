"""
NIfTI reader (nibabel): array load, affine/orientation, zooms, JSON sidecar and header extensions.

Use :func:`read_nifti` or :func:`nvitk.io.imread` with ``force_type='nifti'``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError

from .._common import default_nifti_axes, orientation_codes_from_affine, reorder_axes


# ──────────────────────────────────────────────────────────────────────────────
# Constants & path helpers
# ──────────────────────────────────────────────────────────────────────────────

_NIFTI_JSON_STRIP_KEYS = {
    "axes",
    "shape",
    "affine",
    "x_res",
    "y_res",
    "z_res",
    "t_res",
    "temporal_resolution",
    "orientation",
    "spacing",
}


def _sorted_nifti_files_in_dir(directory: Path) -> list[Path]:
    """Alphabetically sorted list of ``.nii``/``.nii.gz`` files directly under *directory* (hidden files excluded)."""
    out: list[Path] = []
    for child in directory.iterdir():
        if not child.is_file():
            continue
        low = child.name.lower()
        if low.startswith("."):
            continue
        if low.endswith(".nii.gz") or low.endswith(".nii"):
            out.append(child)
    return sorted(out, key=lambda x: x.name.lower())


# -----------------------------------------------------------------------------
# Single-file resolution
# -----------------------------------------------------------------------------


def _resolve_nifti_file_path(path: Path) -> Path:
    """Resolve *path* to a single NIfTI file: pass through a file, or pick the first match in a directory."""
    if path.is_file():
        return path
    if path.is_dir():
        files = _sorted_nifti_files_in_dir(path)
        if not files:
            raise FileNotFoundError(f"No NIfTI (.nii/.nii.gz) files in directory: {path}")
        return files[0]
    raise FileNotFoundError(path)


def _json_sidecar_path(nifti_file: Path) -> Path:
    """Derive the ``<stem>.json`` sidecar path for a ``.nii``/``.nii.gz`` file."""
    name = nifti_file.name
    low = name.lower()
    if low.endswith(".nii.gz"):
        return nifti_file.with_name(name[:-7] + ".json")
    if low.endswith(".nii"):
        return nifti_file.with_suffix(".json")
    return nifti_file.with_suffix(".json")


def nifti_metadata_json_path(nifti_file: str | Path) -> Path:
    """Sibling path ``<stem>.json`` for a ``.nii`` / ``.nii.gz`` file (the JSON need not exist)."""
    return _json_sidecar_path(Path(nifti_file))


def _merge_sidecar_dict(metadata: dict[str, Any], raw: Any) -> None:
    """In-place: merge a JSON sidecar dict into *metadata*, skipping nvitk-reserved keys (e.g. affine/spacing)."""
    if not isinstance(raw, dict):
        return
    extra = {k: v for k, v in raw.items() if k not in _NIFTI_JSON_STRIP_KEYS}
    metadata.update(extra)


# ──────────────────────────────────────────────────────────────────────────────
# read_nifti
# ──────────────────────────────────────────────────────────────────────────────


def read_nifti(
    path: str,
    *,
    axes: str | None = None,
    metadata_json: str | Path | None = None,
    **_: Any,
):
    """
    Load a NIfTI file or the first ``.nii`` / ``.nii.gz`` in a directory via nibabel.

    Populates ``axes``, ``shape``, ``affine``, ``orientation`` (when derivable), voxel sizes,
    and merges a sibling JSON sidecar if present (or *metadata_json* when given). NIfTI header
    extension payloads that decode as JSON are merged without overwriting core spatial keys.
    Optional *axes* reorders the array and updates metadata.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Voxel array and metadata dict for :class:`~nvitk.types.image.Image` construction.
    """
    try:
        import nibabel as nib
    except Exception as exc:
        raise BackendUnavailableError('nibabel is not installed. Please install it with "pip install nibabel".') from exc

    p_in = Path(path)
    if not p_in.exists():
        raise FileNotFoundError(path)

    p = _resolve_nifti_file_path(p_in)

    proxy = nib.load(str(p))
    data = np.asarray(proxy.dataobj)
    metadata: dict[str, Any] = {
        "axes": default_nifti_axes(data.ndim),
        "shape": tuple(data.shape),
        "affine": proxy.affine,
    }
    oc = orientation_codes_from_affine(proxy.affine)
    if oc is not None:
        metadata["orientation"] = oc

    zooms = proxy.header.get_zooms()[: data.ndim]
    if len(zooms) > 0:
        metadata["x_res"] = float(zooms[0])
    if len(zooms) > 1:
        metadata["y_res"] = float(zooms[1])
    if len(zooms) > 2:
        metadata["z_res"] = float(zooms[2])
    if len(zooms) > 3:
        metadata["t_res"] = float(zooms[3])
        metadata["temporal_resolution"] = float(zooms[3])

    json_path: Path | None = Path(metadata_json) if metadata_json is not None else None
    if json_path is None:
        candidate = _json_sidecar_path(p)
        if candidate.is_file():
            json_path = candidate
    if json_path is not None:
        if not json_path.is_file():
            raise FileNotFoundError(str(json_path))
        with json_path.open(encoding="utf-8") as f:
            sidecar = json.load(f)
        _merge_sidecar_dict(metadata, sidecar)

    for extension in proxy.header.extensions:
        try:
            try:
                content_bytes = extension.get_content()
            except Exception:
                content_bytes = getattr(extension, "_raw", None)
            if isinstance(content_bytes, (bytes, bytearray)):
                payload = bytes(content_bytes).rstrip(b"\x00")
                extension_metadata = json.loads(payload.decode("utf-8"))
                if isinstance(extension_metadata, dict):
                    if "axes" in extension_metadata:
                        extension_metadata.pop("axes")
                    if "shape" in extension_metadata:
                        extension_metadata.pop("shape")
                    if "affine" in extension_metadata:
                        extension_metadata.pop("affine")
                    if "x_res" in extension_metadata:
                        extension_metadata.pop("x_res")
                    if "y_res" in extension_metadata:
                        extension_metadata.pop("y_res")
                    if "z_res" in extension_metadata:
                        extension_metadata.pop("z_res")
                    if "t_res" in extension_metadata:
                        extension_metadata.pop("t_res")
                    if "temporal_resolution" in extension_metadata:
                        extension_metadata.pop("temporal_resolution")
                    if "orientation" in extension_metadata:
                        extension_metadata.pop("orientation")
                    metadata.update(extension_metadata)
        except Exception:
            continue

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    aff = metadata.get("affine")
    oc2 = orientation_codes_from_affine(aff) if aff is not None else None
    if oc2 is not None:
        metadata["orientation"] = oc2

    return data, metadata
