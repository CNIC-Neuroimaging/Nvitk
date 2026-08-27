"""
Blosc2 NDArray (``.b2nd``) reader for nnU-Net / nnSSL preprocessed volumes.

A preprocessed case is a compressed ``<name>.b2nd`` array shaped ``(C, Z, Y, X)`` next to a
``<name>.pkl`` sidecar that holds the SimpleITK geometry of the source scan plus the
crop/resample bookkeeping. Neither file records the *final* voxel spacing, so this reader
rebuilds it from the sidecar and — when it can be located by walking up the tree — the
``*Plans*.json`` whose ``data_identifier`` names one of the parent folders. The array is
returned in ``XYZ`` order with a world affine, matching :func:`~nvitk.io.readers.read_nifti`.

Use :func:`read_b2nd` or :func:`nvitk.io.imread` with ``force_type='b2nd'``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError, ValidationError

from .._common import orientation_codes_from_affine, reorder_axes

try:
    import blosc2
except Exception:
    blosc2 = None


# How far up the tree to look for the plans JSON, and the size above which a candidate JSON is
# assumed to be something else (nnssl writes multi-MB ``pretrain_data__*.json`` files around).
_MAX_PLAN_SEARCH_DEPTH = 8
_MAX_PLAN_BYTES = 4 * 1024 * 1024

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar properties
# ──────────────────────────────────────────────────────────────────────────────


def b2nd_properties_path(b2nd_file: str | Path) -> Path:
    """Sibling ``<stem>.pkl`` properties path for a ``.b2nd`` file (the pickle need not exist)."""
    return Path(b2nd_file).with_suffix(".pkl")


def load_b2nd_properties(path: str | Path) -> dict[str, Any]:
    """
    Load a preprocessing properties pickle (spacing, ``sitk_stuff``, crop bbox, shapes).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with p.open("rb") as f:
        props = pickle.load(f)
    return dict(props) if isinstance(props, dict) else {}


def _properties_for(b2nd_file: Path, properties: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the properties dict for *b2nd_file*: explicit dict/path wins, else the sidecar if readable."""
    if isinstance(properties, dict):
        return dict(properties)
    if properties is not None:
        return load_b2nd_properties(properties)
    candidate = b2nd_properties_path(b2nd_file)
    if not candidate.is_file():
        return {}
    try:
        return load_b2nd_properties(candidate)
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Plans discovery (target spacing + transpose_forward)
# ──────────────────────────────────────────────────────────────────────────────


def _plan_candidates(b2nd_file: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(data_identifier_folder_name, plans_json)`` pairs for the ancestors of *b2nd_file*."""
    for depth, folder in enumerate(b2nd_file.parents):
        if depth > _MAX_PLAN_SEARCH_DEPTH:
            return
        parent = folder.parent
        if parent == folder:
            return
        try:
            candidates = sorted(c for c in parent.glob("*.json") if "plans" in c.name.lower())
        except OSError:
            continue
        for candidate in candidates:
            yield folder.name, candidate


def _find_configuration_plan(b2nd_file: Path) -> dict[str, Any] | None:
    """
    Locate the plans configuration that produced *b2nd_file*.

    Matches a ``configurations[*].data_identifier`` against an ancestor folder name (nnU-Net /
    nnssl write preprocessed cases under ``<plans_name>_<configuration>/``). Returns the target
    spacing, ``transpose_forward``, and provenance, or ``None`` when no plans file is found.
    """
    for data_identifier, plans_file in _plan_candidates(b2nd_file):
        try:
            if plans_file.stat().st_size > _MAX_PLAN_BYTES:
                continue
            with plans_file.open(encoding="utf-8") as f:
                plans = json.load(f)
        except Exception:
            continue
        if not isinstance(plans, dict):
            continue
        configurations = plans.get("configurations")
        if not isinstance(configurations, dict):
            continue
        for name, config in configurations.items():
            if isinstance(config, dict) and config.get("data_identifier") == data_identifier:
                return {
                    "plans_file": str(plans_file),
                    "plans_name": plans.get("plans_name"),
                    "configuration": name,
                    "spacing": config.get("spacing"),
                    "transpose_forward": plans.get("transpose_forward"),
                }
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Geometry reconstruction
# ──────────────────────────────────────────────────────────────────────────────


def _transpose_forward(plan: dict[str, Any] | None) -> tuple[int, int, int]:
    """``transpose_forward`` from *plan* (the axis permutation applied before cropping), or identity."""
    raw = (plan or {}).get("transpose_forward")
    if isinstance(raw, (list, tuple)) and len(raw) == 3 and sorted(int(v) for v in raw) == [0, 1, 2]:
        return tuple(int(v) for v in raw)  # type: ignore[return-value]
    return (0, 1, 2)


def _spatial_perm_to_xyz(transpose_forward: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Permutation taking the stored spatial axes to SimpleITK index order ``(x, y, z)``.

    Stored axis ``a`` is pre-transpose axis ``transpose_forward[a]``, and the pre-transpose array
    is ``(z, y, x)`` — i.e. SimpleITK axis ``2 - transpose_forward[a]``.
    """
    inverse = [0, 0, 0]
    for stored_axis, pre_axis in enumerate(transpose_forward):
        inverse[pre_axis] = stored_axis
    return (inverse[2], inverse[1], inverse[0])


def _resolve_spacing(
    properties: dict[str, Any],
    plan: dict[str, Any] | None,
    spatial_shape: tuple[int, ...],
    transpose_forward: tuple[int, int, int],
) -> tuple[tuple[float, float, float] | None, str]:
    """
    Voxel spacing of the *stored* array, in stored spatial-axis order, with its provenance.

    Preprocessing resamples to the plans' target spacing, so that value is authoritative when the
    plans file is found. Otherwise the original spacing is scaled by the shape change the
    resampling caused (exact up to nnU-Net's shape rounding), or used as-is when nothing was
    resampled.
    """
    target = (plan or {}).get("spacing")
    if isinstance(target, (list, tuple)) and len(target) == len(spatial_shape):
        return tuple(float(v) for v in target), "plans"  # type: ignore[return-value]

    raw = properties.get("spacing")
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != 3 or len(spatial_shape) != 3:
        return None, "unknown"
    original = [float(raw[i]) for i in transpose_forward]

    before = properties.get("shape_after_cropping_and_before_resampling")
    if isinstance(before, (list, tuple)) and len(before) == 3:
        before_shape = tuple(int(v) for v in before)
        if before_shape == tuple(int(v) for v in spatial_shape):
            return tuple(original), "properties"  # type: ignore[return-value]
        scaled = [
            original[i] * before_shape[i] / float(spatial_shape[i])
            for i in range(3)
        ]
        return tuple(scaled), "derived"  # type: ignore[return-value]

    return tuple(original), "properties"  # type: ignore[return-value]


def _sitk_geometry(
    properties: dict[str, Any],
    spacing_stored: tuple[float, float, float] | None,
    transpose_forward: tuple[int, int, int],
) -> dict[str, Any] | None:
    """
    Rebuild the SimpleITK geometry (LPS) of the preprocessed array from the source geometry.

    Cropping to the nonzero bounding box shifts the origin by the crop offset in *original*
    voxels; resampling replaces the spacing but leaves origin and direction untouched. Returns
    ``spacing`` / ``origin`` / ``direction`` in ``(x, y, z)`` order plus the 4x4 LPS affine, or
    ``None`` when the sidecar carries no ``sitk_stuff``.
    """
    sitk_stuff = properties.get("sitk_stuff")
    if not isinstance(sitk_stuff, dict) or spacing_stored is None:
        return None
    direction = sitk_stuff.get("direction")
    origin = sitk_stuff.get("origin")
    source_spacing = sitk_stuff.get("spacing")
    if direction is None or origin is None or source_spacing is None:
        return None
    if len(direction) != 9 or len(origin) != 3 or len(source_spacing) != 3:
        return None

    cosines = np.asarray(direction, dtype=float).reshape(3, 3)
    origin_vec = np.asarray(origin, dtype=float)
    source_spacing_vec = np.asarray(source_spacing, dtype=float)

    # Crop offset: bbox is in stored spatial-axis order; undo the transpose, then reverse
    # (z, y, x) → (x, y, z) to reach SimpleITK index order.
    offset_pre = np.zeros(3, dtype=float)
    bbox = properties.get("bbox_used_for_cropping")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 3:
        for stored_axis, bounds in enumerate(bbox):
            offset_pre[transpose_forward[stored_axis]] = float(bounds[0])
    origin_cropped = origin_vec + cosines @ (source_spacing_vec * offset_pre[::-1])

    spacing_pre = np.empty(3, dtype=float)
    for stored_axis, value in enumerate(spacing_stored):
        spacing_pre[transpose_forward[stored_axis]] = float(value)
    spacing_sitk = spacing_pre[::-1]

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = cosines @ np.diag(spacing_sitk)
    affine_lps[:3, 3] = origin_cropped

    return {
        "spacing": tuple(float(v) for v in spacing_sitk),
        "origin": tuple(float(v) for v in origin_cropped),
        "direction": tuple(float(v) for v in cosines.reshape(-1)),
        "affine_lps": affine_lps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Array loading
# ──────────────────────────────────────────────────────────────────────────────


def _b2nd_files_in_dir(directory: Path) -> list[Path]:
    """Alphabetically sorted ``.b2nd`` files directly under *directory* (hidden files excluded)."""
    return sorted(
        (c for c in directory.iterdir() if c.is_file() and c.suffix.lower() == ".b2nd" and not c.name.startswith(".")),
        key=lambda x: x.name.lower(),
    )


def _select_channel(
    array: Any,
    shape: tuple[int, ...],
    channel: int | None,
    squeeze_channel: bool,
) -> tuple[np.ndarray, bool]:
    """
    Materialize the Blosc2 array, optionally keeping a single channel.

    Slicing before materializing lets Blosc2 decompress only the requested channel. Returns the
    array and whether a leading channel axis is still present.
    """
    if len(shape) != 4:
        return np.asarray(array[...]), False
    if channel is not None:
        index = int(channel)
        if not -shape[0] <= index < shape[0]:
            raise IndexError(f"channel={channel} out of range for {shape[0]} channel(s)")
        return np.asarray(array[index]), False
    if shape[0] == 1 and squeeze_channel:
        return np.asarray(array[0]), False
    return np.asarray(array[...]), True


def _to_xyz(data: np.ndarray, perm: tuple[int, int, int], has_channel: bool) -> tuple[np.ndarray, str]:
    """Reorder stored ``(C,)ZYX`` data to ``XYZ`` / ``XYZC`` (a view — no copy) and return its axis string."""
    if has_channel:
        return np.transpose(data, (perm[0] + 1, perm[1] + 1, perm[2] + 1, 0)), "XYZC"
    return np.transpose(data, perm), "XYZ"


def _blosc2_storage_info(array: Any) -> dict[str, Any]:
    """Best-effort chunk/block/compression details of a Blosc2 array, for the properties panel."""
    info: dict[str, Any] = {}
    for key, attr in (("b2nd_chunks", "chunks"), ("b2nd_blocks", "blocks")):
        value = getattr(array, attr, None)
        if value is not None:
            info[key] = tuple(int(v) for v in value)
    schunk = getattr(array, "schunk", None)
    ratio = getattr(schunk, "cratio", None)
    if ratio is not None:
        info["b2nd_compression_ratio"] = float(ratio)
    return info


# ──────────────────────────────────────────────────────────────────────────────
# read_b2nd
# ──────────────────────────────────────────────────────────────────────────────


def read_b2nd(
    path: str,
    *,
    axes: str | None = None,
    channel: int | None = None,
    squeeze_channel: bool = True,
    properties: str | Path | dict[str, Any] | None = None,
    world: str = "ras",
    **_: Any,
):
    """
    Load a Blosc2 ``.b2nd`` preprocessed volume (or every ``.b2nd`` in a directory).

    Data is returned in ``XYZ`` order (``XYZC`` when several channels are kept) with ``affine``,
    ``orientation``, voxel sizes, and SimpleITK ``spacing`` / ``origin`` / ``direction`` rebuilt
    from the ``.pkl`` sidecar and the plans file. The raw sidecar is preserved under
    ``preprocessing_properties``.

    Parameters
    ----------
    path
        A ``.b2nd`` file, or a directory (every ``.b2nd`` inside is read).
    axes
        If set, reorder the output to this axis string.
    channel
        Keep only this channel of a ``(C, Z, Y, X)`` array; only that channel is decompressed.
    squeeze_channel
        Drop a length-1 channel axis (default) so single-modality cases load as 3D volumes.
    properties
        Sidecar path or an already-loaded properties dict; defaults to the sibling ``.pkl``.
    world
        ``ras`` (default, matching the NIfTI reader) or ``lps`` (SimpleITK's convention) for the
        world space the ``affine`` maps into.

    Returns
    -------
    tuple[numpy.ndarray, dict] or list[tuple[numpy.ndarray, dict]]
        Voxel array and metadata, or one such pair per file when *path* is a directory.
    """
    if blosc2 is None:
        raise BackendUnavailableError('blosc2 is not installed. Please install it with "pip install blosc2".')

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.is_dir():
        files = _b2nd_files_in_dir(p)
        if not files:
            raise FileNotFoundError(f"No Blosc2 (.b2nd) files in directory: {p}")
        if properties is not None:
            raise ValidationError(
                f"'properties' applies to a single case, but {p} holds {len(files)} .b2nd file(s); "
                "read them one at a time to override the sidecar."
            )
        return [
            read_b2nd(
                str(f),
                axes=axes,
                channel=channel,
                squeeze_channel=squeeze_channel,
                world=world,
            )
            for f in files
        ]

    array = blosc2.open(urlpath=str(p), mode="r", mmap_mode="r")
    stored_shape = tuple(int(v) for v in array.shape)
    if len(stored_shape) not in (3, 4):
        raise ValidationError(
            f"Expected a 3D (Z,Y,X) or 4D (C,Z,Y,X) .b2nd array, got shape {stored_shape}: {p}"
        )

    data, has_channel = _select_channel(array, stored_shape, channel, squeeze_channel)
    spatial_shape = stored_shape[1:] if len(stored_shape) == 4 else stored_shape

    props = _properties_for(p, properties)
    plan = _find_configuration_plan(p)
    transpose_forward = _transpose_forward(plan)
    spacing_stored, spacing_source = _resolve_spacing(props, plan, spatial_shape, transpose_forward)
    geometry = _sitk_geometry(props, spacing_stored, transpose_forward)

    data, axes_prev = _to_xyz(data, _spatial_perm_to_xyz(transpose_forward), has_channel)

    metadata: dict[str, Any] = {
        "axes": axes_prev,
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "filename": p.name,
        "name": p.stem,
    }
    metadata.update(_blosc2_storage_info(array))

    if geometry is not None:
        affine = geometry["affine_lps"]
        if str(world).lower() == "ras":
            affine = _LPS_TO_RAS @ affine
        metadata["affine"] = affine
        metadata["world"] = str(world).lower()
        metadata["spacing"] = geometry["spacing"]
        metadata["origin"] = geometry["origin"]
        metadata["direction"] = geometry["direction"]
        orientation = orientation_codes_from_affine(affine)
        if orientation is not None:
            metadata["orientation"] = orientation
        metadata["x_res"], metadata["y_res"], metadata["z_res"] = geometry["spacing"]
    elif spacing_stored is not None:
        # No source geometry: spacing is still known, but only as an axis-aligned scale.
        perm = _spatial_perm_to_xyz(transpose_forward)
        metadata["spacing"] = tuple(float(spacing_stored[a]) for a in perm)
        metadata["x_res"], metadata["y_res"], metadata["z_res"] = metadata["spacing"]

    if has_channel:
        metadata["t_res"] = 1.0
        metadata["channels"] = int(stored_shape[0])
    elif channel is not None and len(stored_shape) == 4:
        metadata["channel"] = int(channel) % int(stored_shape[0])

    if props:
        metadata["preprocessing_properties"] = props
        metadata["spacing_source"] = spacing_source
    if plan is not None:
        metadata["plans_file"] = plan["plans_file"]
        metadata["configuration"] = plan["configuration"]
        if plan.get("plans_name"):
            metadata["plans_name"] = plan["plans_name"]

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
