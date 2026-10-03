"""
Cardiac phase stacking — single-phase reconstructions → one 3D+t volume.

Description
-----------
A prospectively gated cardiac CT (or a retrospective one reconstructed at a few
phases) arrives as one *series per phase*: ``IMR, 73%``, ``IMR, 78%``,
``IMR, 83%`` — the same acquisition and reconstruction, different R-R phase.
Each converts to its own 3D NIfTI. This module finds those siblings among the
converted outputs and writes them as one ``X x Y x Z x T`` volume ordered by
phase, so the GUI can play them as a cine and any 3D+t tool can use them.

Grouping
--------
Two outputs are phases of one stack when they share study, frame of reference,
modality, voxel grid (shape + affine), reconstruction (kernel, filter, image type,
slice thickness) and their series description *with the phase token removed*.
The phase comes from ``NominalPercentageOfCardiacPhase`` when present, otherwise
from a ``NN%`` token in the description or the image comments. Different
reconstructions of the same phases (``IMR`` vs ``MCR, IMR`` vs ``IMR SHARP``)
therefore form separate stacks, never one mixed one.

Time axis
---------
Phases are in percent of the R-R interval. With a known ``HeartRate`` the time
axis is in seconds (``RR = 60 / HR``); without one it stays in percent of R-R and
the sidecar says so (``t_units = "percent_RR"``). The per-frame phases are always
recorded as ``cardiac_phases_percent``.

I/O: NIfTI (+ JSON sidecar or embedded JSON extension) in and out, via nibabel.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None

log = Logger()

#: ``73%``, ``73 %``, ``73.5%`` — the phase token in a series description.
_PHASE_TOKEN = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _sidecar_path(nifti_path: Path) -> Path:
    """``name.json`` beside ``name.nii[.gz]``."""
    name = nifti_path.name
    stem = name[:-7] if name.endswith(".nii.gz") else name[:-4] if name.endswith(".nii") else nifti_path.stem
    return nifti_path.with_name(f"{stem}.json")


def read_header_metadata(nifti_path: str | Path) -> dict[str, Any]:
    """The DICOM metadata stored with a converted NIfTI, without loading voxels.

    The JSON sidecar when there is one, else the JSON extension
    :func:`~nvitk.io.conversors._dicom_conversion._save_image_with_metadata` embeds.
    """
    path = Path(nifti_path)
    side = _sidecar_path(path)
    if side.is_file():
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    if nib is None:
        return {}
    try:
        header = nib.load(str(path)).header
    except Exception:  # noqa: BLE001
        return {}
    for ext in header.extensions:
        try:
            raw = ext.get_content()
        except Exception:  # noqa: BLE001
            raw = getattr(ext, "_raw", None)
        if isinstance(raw, (bytes, bytearray)):
            try:
                data = json.loads(bytes(raw).rstrip(b"\x00").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
    return {}


def cardiac_phase_percent(md: dict[str, Any]) -> float | None:
    """The R-R phase (%) of a series, from the DICOM tag or a ``NN%`` token."""
    for key in ("NominalPercentageOfCardiacPhase", "(0020,9241)"):
        try:
            val = float(md.get(key))
            if 0.0 <= val <= 100.0:
                return val
        except (TypeError, ValueError):
            pass
    for key in ("SeriesDescription", "ImageComments", "series_description"):
        match = _PHASE_TOKEN.search(str(md.get(key) or ""))
        if match:
            val = float(match.group(1))
            if 0.0 <= val <= 100.0:
                return val
    return None


def _strip_phase(text: Any) -> str:
    """*text* with its phase token removed and whitespace/commas normalised."""
    out = _PHASE_TOKEN.sub("", str(text or ""))
    out = re.sub(r"[,\s]+", " ", out).strip(" ,")
    return out


def _image_type_key(md: dict[str, Any]) -> str:
    """``ImageType`` as a stable string."""
    raw = md.get("ImageType")
    if isinstance(raw, (list, tuple)):
        return "\\".join(str(v).upper() for v in raw)
    return str(raw or "").upper()


def stack_key(md: dict[str, Any], shape: tuple[int, ...], affine: np.ndarray) -> tuple:
    """Everything two phases of one stack must share (see *Grouping* above)."""
    return (
        str(md.get("StudyInstanceUID") or ""),
        str(md.get("FrameOfReferenceUID") or ""),
        str(md.get("Modality") or "").upper(),
        _strip_phase(md.get("SeriesDescription")),
        str(md.get("ConvolutionKernel") or ""),
        str(md.get("FilterType") or ""),
        _image_type_key(md),
        str(md.get("SliceThickness") or ""),
        tuple(int(s) for s in shape),
        tuple(np.round(np.asarray(affine, dtype=float).ravel(), 3)),
    )


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def find_phase_groups(paths: Sequence[str | Path], *, min_phases: int = 2) -> list[list[tuple[float, Path, dict]]]:
    """
    Group converted 3D NIfTIs into cardiac phase stacks.

    Returns
    -------
    list of groups
        Each group is ``[(phase_percent, path, metadata), …]`` sorted by phase,
        with at least *min_phases* distinct phases. Outputs that are not 3D, have
        no phase, or have no sibling are left out.
    """
    if nib is None:
        return []
    buckets: dict[tuple, list[tuple[float, Path, dict]]] = {}
    for raw in paths:
        path = Path(raw)
        try:
            img = nib.load(str(path))
        except Exception:  # noqa: BLE001 — a file another step wrote badly is not ours to fail on
            continue
        if len(img.shape) != 3:
            continue
        md = read_header_metadata(path)
        phase = cardiac_phase_percent(md)
        if phase is None:
            continue
        key = stack_key(md, tuple(img.shape), img.affine)
        buckets.setdefault(key, []).append((phase, path, md))
    groups: list[list[tuple[float, Path, dict]]] = []
    for items in buckets.values():
        items.sort(key=lambda it: it[0])
        # One file per phase: a re-run that wrote "x.nii.gz" and "x_1.nii.gz"
        # must not put the same phase into the stack twice.
        unique: dict[float, tuple[float, Path, dict]] = {}
        for item in items:
            unique.setdefault(round(item[0], 3), item)
        kept = sorted(unique.values(), key=lambda it: it[0])
        if len(kept) >= int(min_phases):
            groups.append(kept)
    return groups


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _phase_label(phase: float) -> str:
    """``73`` or ``73.5`` — a phase for a file name."""
    return f"{phase:g}".replace(".", "p")


def _stack_name(group: Sequence[tuple[float, Path, dict]], compress: bool) -> str:
    """``IMR_CT_phases_73-78-83.nii.gz`` for a group."""
    md = group[0][2]
    desc = _strip_phase(md.get("SeriesDescription")) or "series"
    desc = re.sub(r"[^\w\-]+", "_", desc).strip("_") or "series"
    mod = str(md.get("Modality") or "UNK").upper()
    phases = "-".join(_phase_label(p) for p, _, _ in group)
    ext = ".nii.gz" if compress else ".nii"
    return f"{desc}_{mod}_phases_{phases}{ext}"


def write_volume_stack(
    paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    metadata: dict[str, Any],
    extra: dict[str, Any],
    axis_step: float,
    time_units: str,
    save_metadata: bool = True,
) -> str:
    """
    Stack same-grid 3D NIfTIs along a new last axis and save one 4D NIfTI.

    The shared writer behind the cardiac phase and monoenergetic stacks. The first
    file's header and affine are kept; voxel data keep their stored dtype (int16 stays
    int16). ``pixdim[4]`` is *axis_step* with NIfTI time units *time_units* (``"sec"``
    or ``"unknown"`` for a non-time axis). *metadata* updated with *extra* is embedded
    and, with *save_metadata*, written as the JSON sidecar. Returns the written path.
    """
    if nib is None:
        raise RuntimeError("nibabel is required to write volume stacks.")
    first = nib.load(str(paths[0]))
    shape = tuple(int(s) for s in first.shape)
    dtype = first.get_data_dtype()
    volume = np.empty(shape + (len(paths),), dtype=dtype)
    for t, path in enumerate(paths):
        img = nib.load(str(path))
        if tuple(img.shape) != shape:
            raise ValueError(f"{Path(path).name}: shape {img.shape} != {shape}.")
        # dataobj, not get_fdata: keep int16 int16 (unscaled, as written).
        volume[..., t] = np.asanyarray(img.dataobj, dtype=dtype)

    md = dict(metadata)
    md.update(extra)
    md.setdefault("dynamic", True)
    md["t_res"] = float(axis_step)
    md["temporal_resolution"] = float(axis_step)
    md["stack_sources"] = [Path(p).name for p in paths]

    image = nib.Nifti1Image(volume, first.affine, header=first.header.copy())
    # The copied header still carries the first input's embedded metadata; a stack
    # must describe itself only (header-only readers take the first extension).
    del image.header.extensions[:]
    image.header.set_xyzt_units("mm", time_units)
    image.header.set_zooms(tuple(list(first.header.get_zooms()[:3]) + [float(axis_step)]))
    image.set_sform(first.affine, code=1)
    image.set_qform(first.affine, code=1)
    try:
        image.header.extensions.append(
            nib.nifti1.Nifti1Extension(16, json.dumps(md, default=str).encode("utf-8"))
        )
    except Exception:  # noqa: BLE001 — metadata is a convenience; the volume is the product
        pass
    out = Path(output_path)
    nib.save(image, str(out))
    if save_metadata:
        _sidecar_path(out).write_text(json.dumps(md, indent=2, default=str), encoding="utf-8")
    return str(out)


def write_phase_stack(
    group: Sequence[tuple[float, Path, dict]],
    output_path: str | Path,
    *,
    save_metadata: bool = True,
) -> str:
    """
    Stack *group* (from :func:`find_phase_groups`) along a new last axis and save it.

    With a known heart rate the axis is in seconds (``RR = 60 / HR``), otherwise in
    percent of R-R. Returns the written path.
    """
    phases = [float(p) for p, _, _ in group]
    md = dict(group[0][2])
    hr = None
    try:
        hr = float(md.get("HeartRate"))
        if not np.isfinite(hr) or hr <= 0:
            hr = None
    except (TypeError, ValueError):
        hr = None
    if hr is not None:
        rr = 60.0 / hr
        times = [(p - phases[0]) / 100.0 * rr for p in phases]
        t_units, nifti_t_units = "s", "sec"
    else:
        times = [p - phases[0] for p in phases]
        t_units, nifti_t_units = "percent_RR", "unknown"
    diffs = np.diff(times)
    return write_volume_stack(
        [path for _, path, _ in group],
        output_path,
        metadata=md,
        extra={
            "axes": "XYZT",
            "phase_stack": True,
            "n_timepoints": len(group),
            "cardiac_phases_percent": phases,
            ("frame_times_s" if t_units == "s" else "frame_offsets_percent_RR"): [round(t, 6) for t in times],
            "t_units": t_units,
            "phase_sources": [str(path.name) for _, path, _ in group],
            "SeriesDescription": f"{_strip_phase(md.get('SeriesDescription'))} "
                                 f"(phases {', '.join(f'{p:g}%' for p in phases)})",
        },
        axis_step=float(np.median(diffs)) if diffs.size else 1.0,
        time_units=nifti_t_units,
        save_metadata=save_metadata,
    )


def stack_cardiac_phases(
    paths: Sequence[str | Path],
    output_folder: str | Path | None = None,
    *,
    compress: bool = True,
    save_metadata: bool = True,
    min_phases: int = 2,
    skip_existing: bool = False,
) -> list[str]:
    """
    Find cardiac phase siblings among *paths* and write one 3D+t stack per group.

    Parameters
    ----------
    paths
        Converted NIfTI files (e.g. what ``dcm2nii`` returned), or a directory's
        contents. Non-NIfTI paths are ignored.
    output_folder
        Where stacks go; default beside the first file of each group.
    compress, save_metadata, skip_existing
        As for the converter.
    min_phases
        Smallest group worth stacking.

    Returns
    -------
    list of str
        The stack files written (or found, with *skip_existing*).
    """
    nifti = [Path(p) for p in paths if str(p).endswith((".nii", ".nii.gz"))]
    written: list[str] = []
    for group in find_phase_groups(nifti, min_phases=min_phases):
        folder = Path(output_folder) if output_folder else group[0][1].parent
        os.makedirs(folder, exist_ok=True)
        target = folder / _stack_name(group, compress)
        if skip_existing and target.exists():
            log.info("Skipping existing phase stack: %s", target)
            written.append(str(target))
            continue
        out = write_phase_stack(group, target, save_metadata=save_metadata)
        log.info(
            "Stacked %d cardiac phases (%s) → %s",
            len(group), ", ".join(f"{p:g}%" for p, _, _ in group), out,
        )
        written.append(out)
    return written


__all__ = [
    "write_volume_stack",
    "cardiac_phase_percent",
    "find_phase_groups",
    "read_header_metadata",
    "stack_cardiac_phases",
    "stack_key",
    "write_phase_stack",
]
