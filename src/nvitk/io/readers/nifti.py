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
# Colour (RGB/RGBA) channel detection
# ──────────────────────────────────────────────────────────────────────────────

#: Trailing-axis lengths that can hold colour samples (RGB / RGBA).
_RGB_CHANNEL_SIZES = (3, 4)

#: Per-axis voxel-size metadata keys, in array-axis order.
_RESOLUTION_KEYS = ("x_res", "y_res", "z_res", "t_res")


def _channel_axis_from_axes(axes: Any, shape: tuple[int, ...]) -> int | None:
    """Index of the ``C`` axis in a recorded axis string, when it fits *shape* and is RGB-sized."""
    if not isinstance(axes, str):
        return None
    axes_upper = axes.upper()
    if len(axes_upper) != len(shape) or axes_upper.count("C") != 1:
        return None
    index = axes_upper.index("C")
    return index if shape[index] in _RGB_CHANNEL_SIZES else None


def _guessed_channel_axis(data: np.ndarray) -> int | None:
    """Trailing axis of a 3-D array that holds colour samples rather than slices.

    Requires 8-bit samples and a plausible image plane, so that a three-slice
    analysis map (e.g. a Zeiss thickness volume, stored 16-bit) is not mistaken
    for a colour image.
    """
    if data.ndim != 3 or data.shape[-1] not in _RGB_CHANNEL_SIZES:
        return None
    if data.dtype != np.uint8:
        return None
    if min(data.shape[0], data.shape[1]) < 8:
        return None
    return data.ndim - 1


def _resolve_channel_axis(data: np.ndarray, stored_axes: Any, rgb: bool | None) -> int | None:
    """Array axis holding colour samples: from a recorded ``C`` label, else *rgb*, else a guess."""
    index = _channel_axis_from_axes(stored_axes, data.shape)
    if index is not None:
        return index
    if rgb is False:
        return None
    if data.ndim < 3 or data.shape[-1] not in _RGB_CHANNEL_SIZES:
        return None
    return data.ndim - 1 if rgb else _guessed_channel_axis(data)


# ──────────────────────────────────────────────────────────────────────────────
# read_nifti
# ──────────────────────────────────────────────────────────────────────────────


def read_nifti(
    path: str,
    *,
    axes: str | None = None,
    metadata_json: str | Path | None = None,
    rgb: bool | None = None,
    **_: Any,
):
    """
    Load a NIfTI file or the first ``.nii`` / ``.nii.gz`` in a directory via nibabel.

    Populates ``axes``, ``shape``, ``affine``, ``orientation`` (when derivable), voxel sizes,
    and merges a sibling JSON sidecar if present (or *metadata_json* when given). NIfTI header
    extension payloads that decode as JSON are merged without overwriting core spatial keys.
    Optional *axes* reorders the array and updates metadata.

    A trailing axis of 3 or 4 is ambiguous - colour samples, or that many slices - so when one
    is present the reader records its verdict as ``rgb`` and labels a colour axis ``C`` rather
    than ``Z``, leaving nothing for the viewer to guess. Pass *rgb* to decide explicitly.

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

    # The recorded axis string is not adopted wholesale (the array on disk is in NIfTI
    # order), but it is the most reliable way to spot a colour axis.
    stored_axes: Any = None

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
        if isinstance(sidecar, dict):
            stored_axes = sidecar.get("axes", stored_axes)
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
                        stored_axes = extension_metadata.pop("axes")
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

    channel_axis = _resolve_channel_axis(data, stored_axes, rgb)
    if data.ndim >= 3 and data.shape[-1] in _RGB_CHANNEL_SIZES:
        metadata["rgb"] = channel_axis is not None
    if channel_axis is not None:
        axis_labels = list(metadata["axes"])
        axis_labels[channel_axis] = "C"
        metadata["axes"] = "".join(axis_labels)
        # A colour axis has no voxel size; the NIfTI zoom for it is a placeholder.
        metadata.pop(_RESOLUTION_KEYS[channel_axis], None)

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    aff = metadata.get("affine")
    oc2 = orientation_codes_from_affine(aff) if aff is not None else None
    if oc2 is not None:
        metadata["orientation"] = oc2

    return data, metadata
