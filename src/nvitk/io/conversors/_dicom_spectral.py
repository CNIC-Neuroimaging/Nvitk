"""
Spectral (dual-energy / photon-counting) CT *results* — classification and energy stacks.

Description
-----------
A spectral CT exports its results as ordinary CT series once a workstation has generated
them from the raw spectral data (on Philips: from the Spectral Base Images, see
:mod:`._dicom_philips`): virtual monoenergetic images at chosen keV, iodine density maps,
iodine-removed (virtual non-contrast) images, effective atomic number, electron density, uric
acid and calcium-suppressed images. They convert like any CT series; this module adds what
makes them usable afterwards:

- :func:`classify_spectral` names the result type, its units and — for a monoenergetic image —
  the energy, so every output carries ``spectral_result`` / ``spectral_energy_kev`` /
  ``spectral_units`` in its metadata and an unambiguous file name;
- :func:`stack_monoenergetic` turns monoenergetic series of one reconstruction at several
  energies into one ``X x Y x Z x E`` volume ordered by keV — the spectral counterpart of a
  cardiac phase stack — so the GUI's curve tool plots the spectral attenuation curve (HU vs keV)
  of any region.

Detection order
---------------
1. The DICOM multi-energy attributes: ``MonoenergeticEnergyEquivalent`` (0018,937C), searched at
   the top level and inside the multi-energy / per-frame sequences, and the multi-energy
   ``ImageType`` values (``VMI``, ``MAT_SPECIFIC``, ``MAT_REMOVED``, ``EFF_ATOMIC_NUM``,
   ``ELECTRON_DENSITY`` …).
2. Otherwise the series description / image comments, with patterns covering the usual vendor
   wording (``MonoE 70keV[HU]``, ``VMI 40 keV``, ``Iodine Density``, ``Iodine no Water``,
   ``Z Effective``, ``Electron Density``, ``VNC`` / ``Virtual Non-Contrast``, ``Uric Acid``,
   ``Calcium Suppression`` / ``CaSupp``).

.. note::
   The description patterns follow common vendor naming but have **not yet been checked
   against exported Philips spectral-result series** — the ES17774 study ships only the raw
   Spectral Base Images. ``classify_spectral`` records which source it used
   (``spectral_source``) so a misclassification is visible in the sidecar.

I/O: pydicom datasets / metadata dicts in; NIfTI (+ JSON) out for the energy stacks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger

log = Logger()

# ---------------------------------------------------------------------------
# Result catalogue
# ---------------------------------------------------------------------------

#: result key → (human label, default units, short file-name label)
SPECTRAL_RESULTS: dict[str, tuple[str, str, str]] = {
    "monoe": ("Virtual monoenergetic", "HU", "monoE"),
    "iodine_density": ("Iodine density", "mg/mL", "iodine"),
    "iodine_no_water": ("Iodine (no water)", "mg/mL", "iodineNoWater"),
    "vnc": ("Virtual non-contrast", "HU", "VNC"),
    "z_effective": ("Effective atomic number", "Z", "Zeff"),
    "electron_density": ("Electron density", "%EDW", "electronDensity"),
    "uric_acid": ("Uric acid", "", "uricAcid"),
    "calcium_suppressed": ("Calcium suppression", "HU", "CaSupp"),
}

#: ``70 keV``, ``70keV``, ``70.5 kev`` — the energy token.
_KEV = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*kev", re.IGNORECASE)

#: Description patterns, most specific first (so "iodine no water" is not "iodine").
_DESCRIPTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iodine_no_water", re.compile(r"iodine.{0,12}(no|without|w/o)[\s_-]*water", re.I)),
    ("vnc", re.compile(r"\bvnc\b|virtual[\s_-]*non[\s_-]*contrast|iodine[\s_-]*(removed|removal)", re.I)),
    ("iodine_density", re.compile(r"iodine", re.I)),
    ("z_effective", re.compile(r"\bz[\s_-]*eff|effective[\s_-]*(atomic|z)", re.I)),
    ("electron_density", re.compile(r"electron[\s_-]*density|%\s*edw", re.I)),
    ("uric_acid", re.compile(r"uric[\s_-]*acid", re.I)),
    ("calcium_suppressed", re.compile(r"calcium[\s_-]*suppress|\bca[\s_-]*supp", re.I)),
    ("monoe", re.compile(r"\bmono[\s_-]*e\b|monoe|mono[\s_-]*energ|\bvmi\b|\bmonochromatic\b", re.I)),
)

#: DICOM multi-energy ``ImageType`` values → result key.
_IMAGE_TYPE_RESULTS: dict[str, str] = {
    "VMI": "monoe",
    "MAT_REMOVED": "vnc",
    "EFF_ATOMIC_NUM": "z_effective",
    "EFFECTIVE_ATOMIC_NUMBER": "z_effective",
    "ELECTRON_DENSITY": "electron_density",
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _find_tag(obj: Any, keyword: str, depth: int = 0) -> Any:
    """First value of *keyword* in a pydicom dataset or nested metadata dict (depth-limited)."""
    if obj is None or depth > 6:
        return None
    try:
        if keyword in obj:
            value = obj[keyword]
            return getattr(value, "value", value)
    except Exception:  # noqa: BLE001
        pass
    try:
        items = obj.values() if isinstance(obj, dict) else [el.value for el in obj if el.VR == "SQ"]
    except Exception:  # noqa: BLE001
        return None
    for value in items:
        if isinstance(value, (list, tuple)) or type(value).__name__ == "Sequence":
            for item in value:
                found = _find_tag(item, keyword, depth + 1)
                if found is not None:
                    return found
        elif isinstance(value, dict):
            found = _find_tag(value, keyword, depth + 1)
            if found is not None:
                return found
    return None


def _text(obj: Any, key: str) -> str:
    """A text attribute from a dataset or dict, as a string."""
    try:
        value = obj.get(key, "") if obj is not None else ""
    except Exception:  # noqa: BLE001
        value = ""
    return str(getattr(value, "value", value) or "")


def _image_type(obj: Any) -> list[str]:
    try:
        raw = obj.get("ImageType", None) if obj is not None else None
    except Exception:  # noqa: BLE001
        raw = None
    raw = getattr(raw, "value", raw)
    if isinstance(raw, str):
        raw = raw.split("\\")
    return [str(v).strip().upper() for v in (raw or [])]


def classify_spectral(ds: Any = None, md: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    What spectral result a series is, or ``None`` for a conventional image.

    Parameters
    ----------
    ds
        A representative pydicom dataset of the series (preferred: it has every tag).
    md
        The series metadata dict, used when *ds* is not available (e.g. reading a
        converted NIfTI's sidecar back).

    Returns
    -------
    dict or None
        ``{"result", "label", "units", "energy_kev", "source"}``.
    """
    sources = [s for s in (ds, md) if s is not None]
    if not sources:
        return None
    # ---- 1. Standard multi-energy attributes ----------------------------------
    energy = None
    for src in sources:
        val = _find_tag(src, "MonoenergeticEnergyEquivalent")
        try:
            energy = float(val) if val not in (None, "") else None
        except (TypeError, ValueError):
            energy = None
        if energy:
            break
    result, source = None, None
    if energy:
        result, source = "monoe", "MonoenergeticEnergyEquivalent"
    else:
        for src in sources:
            tokens = _image_type(src)
            hit = next((_IMAGE_TYPE_RESULTS[t] for t in tokens if t in _IMAGE_TYPE_RESULTS), None)
            if hit:
                result, source = hit, "ImageType"
                break
    # ---- 2. Description / comments --------------------------------------------
    text = " ".join(
        _text(src, key) for src in sources for key in ("SeriesDescription", "ImageComments")
    )
    if result is None:
        for key, pattern in _DESCRIPTION_PATTERNS:
            if pattern.search(text):
                result, source = key, "description"
                break
    if result is None:
        # A bare "70 keV" in the description of a CT series is a monoenergetic image.
        if _KEV.search(text) and "CT" in {_text(s, "Modality").upper() for s in sources}:
            result, source = "monoe", "description (keV)"
    if result is None:
        return None
    if result == "monoe" and not energy:
        match = _KEV.search(text)
        energy = float(match.group(1)) if match else None
    label, units, _short = SPECTRAL_RESULTS[result]
    rescale_type = next((_text(s, "RescaleType") for s in sources if _text(s, "RescaleType")), "")
    if rescale_type and rescale_type.upper() not in ("US", "UNSPECIFIED"):
        units = rescale_type
    return {
        "result": result,
        "label": label,
        "units": units,
        "energy_kev": energy,
        "source": source,
    }


def spectral_file_label(info: dict[str, Any]) -> str:
    """Short file-name label: ``monoE70keV``, ``iodine``, ``Zeff`` …"""
    short = SPECTRAL_RESULTS.get(info.get("result", ""), ("", "", info.get("result", "")))[2]
    energy = info.get("energy_kev")
    if info.get("result") == "monoe" and energy:
        return f"{short}{float(energy):g}keV".replace(".", "p")
    return short


def annotate_spectral(md: dict[str, Any], ds: Any = None) -> dict[str, Any]:
    """Add ``spectral_*`` keys to *md* when the series is a spectral result (in place)."""
    info = classify_spectral(ds, md)
    if info is None:
        return md
    md["spectral_result"] = info["result"]
    md["spectral_result_label"] = info["label"]
    md["spectral_units"] = info["units"]
    md["spectral_source"] = info["source"]
    md["spectral_file_label"] = spectral_file_label(info)
    if info.get("energy_kev"):
        md["spectral_energy_kev"] = float(info["energy_kev"])
    return md


# ---------------------------------------------------------------------------
# Energy stacks
# ---------------------------------------------------------------------------


def _strip_energy(text: Any) -> str:
    """*text* without its keV token, whitespace normalised."""
    out = _KEV.sub("", str(text or ""))
    return re.sub(r"[,\s]+", " ", out).strip(" ,")


def energy_stack_key(md: dict[str, Any], shape: tuple[int, ...], affine: np.ndarray) -> tuple:
    """What monoenergetic images of one stack share: study, frame, recon, grid."""
    return (
        str(md.get("StudyInstanceUID") or ""),
        str(md.get("FrameOfReferenceUID") or ""),
        _strip_energy(md.get("SeriesDescription")),
        str(md.get("ConvolutionKernel") or ""),
        str(md.get("SliceThickness") or ""),
        tuple(int(s) for s in shape),
        tuple(np.round(np.asarray(affine, dtype=float).ravel(), 3)),
    )


def find_energy_groups(
    paths: Sequence[str | Path], *, min_levels: int = 2
) -> list[list[tuple[float, Path, dict]]]:
    """Group converted 3D monoenergetic NIfTIs into energy stacks (``[(keV, path, md), …]``)."""
    from ._dicom_phases import read_header_metadata

    try:
        import nibabel as nib
    except Exception:  # pragma: no cover
        return []
    buckets: dict[tuple, list[tuple[float, Path, dict]]] = {}
    for raw in paths:
        path = Path(raw)
        if not str(path).endswith((".nii", ".nii.gz")):
            continue
        try:
            img = nib.load(str(path))
        except Exception:  # noqa: BLE001
            continue
        if len(img.shape) != 3:
            continue
        md = read_header_metadata(path)
        energy = md.get("spectral_energy_kev")
        if energy in (None, ""):
            info = classify_spectral(md=md)
            energy = info.get("energy_kev") if info and info["result"] == "monoe" else None
        if energy in (None, ""):
            continue
        key = energy_stack_key(md, tuple(img.shape), img.affine)
        buckets.setdefault(key, []).append((float(energy), path, md))
    groups = []
    for items in buckets.values():
        unique: dict[float, tuple[float, Path, dict]] = {}
        for item in sorted(items, key=lambda it: it[0]):
            unique.setdefault(round(item[0], 3), item)
        kept = sorted(unique.values(), key=lambda it: it[0])
        if len(kept) >= int(min_levels):
            groups.append(kept)
    return groups


def _energy_stack_name(group: Sequence[tuple[float, Path, dict]], compress: bool) -> str:
    md = group[0][2]
    desc = _strip_energy(md.get("SeriesDescription")) or "monoE"
    desc = re.sub(r"[^\w\-]+", "_", desc).strip("_") or "monoE"
    mod = str(md.get("Modality") or "CT").upper()
    energies = "-".join(f"{e:g}".replace(".", "p") for e, _, _ in group)
    return f"{desc}_{mod}_monoE_{energies}keV{'.nii.gz' if compress else '.nii'}"


def stack_monoenergetic(
    paths: Sequence[str | Path],
    output_folder: str | Path | None = None,
    *,
    compress: bool = True,
    save_metadata: bool = True,
    min_levels: int = 2,
    skip_existing: bool = False,
) -> list[str]:
    """
    Stack monoenergetic outputs of one reconstruction into ``X x Y x Z x E`` volumes.

    The fourth axis is energy, ascending; ``spectral_energies_kev`` lists the levels and the
    axis step is recorded in keV (``t_units = "keV"``), so nvitk's time tools treat the volume
    like a series whose "frames" are energies — the curve tool then draws HU against keV.

    Returns the stack files written (or found, with *skip_existing*).
    """
    from ._dicom_phases import write_volume_stack

    written: list[str] = []
    for group in find_energy_groups(paths, min_levels=min_levels):
        folder = Path(output_folder) if output_folder else group[0][1].parent
        os.makedirs(folder, exist_ok=True)
        target = folder / _energy_stack_name(group, compress)
        if skip_existing and target.exists():
            written.append(str(target))
            continue
        energies = [float(e) for e, _, _ in group]
        diffs = np.diff(energies)
        md0 = dict(group[0][2])
        # Per-level keys describe one input, not the stack.
        for key in ("spectral_energy_kev", "spectral_file_label", "InstanceNumber"):
            md0.pop(key, None)
        out = write_volume_stack(
            [p for _, p, _ in group],
            target,
            metadata=md0,
            extra={
                "axes": "XYZT",
                "spectral_stack": True,
                "spectral_result": "monoe",
                "spectral_units": md0.get("spectral_units", "HU"),
                "spectral_energies_kev": energies,
                "n_timepoints": len(group),
                "t_units": "keV",
                "SeriesDescription": f"{_strip_energy(md0.get('SeriesDescription'))} "
                                     f"(monoE {', '.join(f'{e:g}' for e in energies)} keV)",
            },
            axis_step=float(np.median(diffs)) if diffs.size else 1.0,
            time_units="unknown",
            save_metadata=save_metadata,
        )
        log.info("Stacked %d monoenergetic levels (%s keV) → %s",
                 len(group), ", ".join(f"{e:g}" for e in energies), out)
        written.append(out)
    return written


__all__ = [
    "SPECTRAL_RESULTS",
    "annotate_spectral",
    "classify_spectral",
    "energy_stack_key",
    "find_energy_groups",
    "spectral_file_label",
    "stack_monoenergetic",
]
