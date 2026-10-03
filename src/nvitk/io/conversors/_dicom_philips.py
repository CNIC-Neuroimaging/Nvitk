"""
Philips CT specifics for the DICOM converter: spectral base images and heart rate.

Description
-----------
Two things a Philips spectral cardiac CT export (``Spectral CT`` / IQon) carries that the
generic path gets wrong:

1. **Spectral Base Images (SBI).** Series described ``SBI(V5.1)_CB_0.67_C|IMR, 78%`` are
   Secondary Captures with ``ImageType = DERIVED\\SECONDARY\\SBI\\SBI_CSPN``: one instance per
   axial position of the reconstruction they belong to, every one with a *different* ``Rows``
   (100 … 940 x 512). They are not pictures. The "pixels" are Philips' compressed spectral
   base data — byte entropy ~5.9 bits against ~0.6 for a real CT slice, and the row count
   tracks how much there was to compress (short at the scan ends, longest through the body).
   Only a Philips spectral workstation can turn them into monoenergetic, iodine or Z-effective
   images. Converting them yields noise — and, with per-geometry splitting, dozens of noise
   files per series — so they are recognised and skipped.

2. **Heart rate.** The standard ``HeartRate`` (0018,1088) is present but *empty* on these
   exports; Philips records the rate at the start of the scan in private (01F1,1045)
   ``Initial Heart Rate``. It is what turns a cardiac phase stack's R-R percentages into
   seconds, so it is copied into ``HeartRate`` when the standard tag says nothing.

I/O: pydicom datasets in, plain values out.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

#: Private (01F1,1045) — Philips CT "Initial Heart Rate" (bpm).
_PHILIPS_INITIAL_HR_TAG = (0x01F1, 0x1045)


def _image_type_tokens(ds: Any) -> list[str]:
    """Uppercased ``ImageType`` values."""
    try:
        return [str(v).strip().upper() for v in (ds.get("ImageType", None) or [])]
    except Exception:  # noqa: BLE001
        return []


def is_philips_spectral_base_image(ds: Any) -> bool:
    """True for a Philips Spectral Base Image instance (compressed spectral payload).

    Recognised by ``ImageType`` carrying ``SBI`` (``DERIVED\\SECONDARY\\SBI\\SBI_CSPN``), or by
    an ``SBI(…)`` image comment / series description on a Philips Secondary Capture.
    """
    tokens = _image_type_tokens(ds)
    if "SBI" in tokens or any(t.startswith("SBI_") for t in tokens):
        return True
    manufacturer = str(ds.get("Manufacturer", "") or "").upper()
    if "PHILIPS" not in manufacturer:
        return False
    for key in ("ImageComments", "SeriesDescription"):
        if str(ds.get(key, "") or "").strip().upper().startswith("SBI("):
            return True
    return False


def is_spectral_base_series(ds_list: Sequence[Any]) -> bool:
    """True when every instance of a series is a spectral base image."""
    return bool(ds_list) and all(is_philips_spectral_base_image(ds) for ds in ds_list)


def philips_heart_rate(ds: Any) -> float | None:
    """The scan's heart rate (bpm) from Philips private (01F1,1045), or ``None``."""
    if "PHILIPS" not in str(ds.get("Manufacturer", "") or "").upper():
        return None
    try:
        elem = ds.get(_PHILIPS_INITIAL_HR_TAG)
        if elem is None:
            return None
        value = elem.value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("ascii", "ignore")
        hr = float(str(value).strip())
    except Exception:  # noqa: BLE001
        return None
    return hr if 20.0 <= hr <= 300.0 else None


def apply_heart_rate_alias(md: dict[str, Any], ds: Any = None) -> dict[str, Any]:
    """Fill ``md["HeartRate"]`` from the vendor tag when the standard one is empty.

    Leaves a real ``HeartRate`` alone. Records where the value came from in
    ``HeartRateSource``. Returns *md* (modified in place).
    """
    current = md.get("HeartRate")
    try:
        if current not in (None, "", "None") and float(current) > 0:
            return md
    except (TypeError, ValueError):
        pass
    hr = philips_heart_rate(ds) if ds is not None else None
    if hr is None:
        raw = md.get("(01F1,1045)")
        try:
            hr = float(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            hr = None
    if hr is not None:
        md["HeartRate"] = float(hr)
        md["HeartRateSource"] = "Philips (01F1,1045) Initial Heart Rate"
    return md


# ---------------------------------------------------------------------------
# Keeping the SBI data: manifest and copy
# ---------------------------------------------------------------------------

#: ``SBI(V5.1)_CB_0.67_C|IMR, 78%`` → version, kernel, thickness, reference description.
_SBI_COMMENT = re.compile(
    r"SBI\((?P<version>[^)]*)\)_(?P<kernel>[^_|]+)_(?P<thickness>[\d.]+)[^|]*\|(?P<reference>.*)$",
    re.IGNORECASE,
)

#: Name of the manifest written beside the NIfTI outputs.
SBI_MANIFEST_NAME = "spectral_base_images.json"


def sbi_comment_fields(ds: Any) -> dict[str, str]:
    """Version / kernel / slice thickness / reference recon parsed from an SBI comment."""
    text = str(ds.get("ImageComments", "") or ds.get("SeriesDescription", "") or "")
    match = _SBI_COMMENT.search(text)
    if not match:
        return {"comment": text}
    return {"comment": text, **{k: v.strip() for k, v in match.groupdict().items()}}


def sbi_slice_positions(ds_list: Sequence[Any]) -> list[float]:
    """Sorted slice positions (z of ``ImagePositionPatient``, mm) of a series."""
    out = []
    for ds in ds_list:
        ipp = ds.get("ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            out.append(round(float(ipp[2]), 2))
    return sorted(out)


def sbi_series_summary(ds_list: Sequence[Any]) -> dict[str, Any]:
    """Everything worth recording about one SBI series, for the manifest."""
    first = ds_list[0]
    fields = sbi_comment_fields(first)
    files = sorted(str(getattr(ds, "filename", "") or "") for ds in ds_list)
    total = 0
    for f in files:
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    positions = sbi_slice_positions(ds_list)
    return {
        "series_number": str(first.get("SeriesNumber", "")),
        "series_instance_uid": str(first.get("SeriesInstanceUID", "")),
        "series_description": str(first.get("SeriesDescription", "")),
        "sbi_version": fields.get("version", ""),
        "kernel": fields.get("kernel", ""),
        "slice_thickness_mm": fields.get("thickness", str(first.get("SliceThickness", ""))),
        "reference_description": fields.get("reference", ""),
        "frame_of_reference_uid": str(first.get("FrameOfReferenceUID", "")),
        "study_instance_uid": str(first.get("StudyInstanceUID", "")),
        "n_instances": len(ds_list),
        "z_range_mm": [positions[0], positions[-1]] if positions else None,
        "total_bytes": total,
        "files": files,
    }


def write_sbi_manifest(entries: Sequence[dict[str, Any]], output_folder: str) -> str:
    """Write :data:`SBI_MANIFEST_NAME` with *entries* and how to use them; returns its path."""
    import json

    path = os.path.join(output_folder, SBI_MANIFEST_NAME)
    doc = {
        "about": (
            "Philips Spectral Base Images (SBI): compressed proprietary spectral data stored as "
            "Secondary Captures, one per axial slice of the reconstruction named in "
            "'reference_series'. They are not pixels and cannot be converted to NIfTI directly. "
            "Load the original DICOM (including these files) on a Philips spectral workstation "
            "(IntelliSpace Portal / Spectral Diagnostic Suite / Spectral Magic Glass) to generate "
            "monoenergetic, iodine, Z-effective, VNC or electron-density series, export them as "
            "DICOM, and convert those with dcm2nii (--stack-energies stacks monoE levels)."
        ),
        "series": list(entries),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return path


def copy_sbi_series(ds_list: Sequence[Any], dest_dir: str, *, skip_existing: bool = False) -> int:
    """Copy one SBI series' DICOM files into *dest_dir* (names kept); returns how many."""
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for ds in ds_list:
        src = str(getattr(ds, "filename", "") or "")
        if not src or not os.path.isfile(src):
            continue
        dst = os.path.join(dest_dir, os.path.basename(src))
        if skip_existing and os.path.exists(dst):
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied


__all__ = [
    "SBI_MANIFEST_NAME",
    "copy_sbi_series",
    "sbi_comment_fields",
    "sbi_series_summary",
    "sbi_slice_positions",
    "write_sbi_manifest",
    "apply_heart_rate_alias",
    "is_philips_spectral_base_image",
    "is_spectral_base_series",
    "philips_heart_rate",
]
