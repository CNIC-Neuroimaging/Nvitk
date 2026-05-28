"""Stage0: DICOM → NIfTI conversion for gpetpy inputs.

Inputs (per subject):
- ``.../IA_PET_V5/CT`` (DICOM)
- ``.../IA_PET_V5/PET`` (DICOM)
- ``.../PESA_Brain/3D_T1`` (DICOM)

Outputs:
- ``NIFTI_ROOT/<batch>/<subject>/CT.nii.gz``
- ``NIFTI_ROOT/<batch>/<subject>/PT.nii.gz``
- ``NIFTI_ROOT/<batch>/<subject>/T1.nii.gz``
and ``conversion_manifest.json`` alongside them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.io.conversors.dcm2nii import dcm2nii

from .layout import GpetLayout

log = Logger()


@dataclass(frozen=True)
class ConvertResult:
    ct: Path
    pet: Path
    t1: Path
    manifest: Path


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _any_nifti_in_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for p in path.iterdir():
        if p.is_file() and _is_nifti(p):
            return True
    return False


def _convert_one(
    *,
    dicom_dir: Path,
    out_path: Path,
    skip_existing: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and out_path.is_file():
        return out_path
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if skip_existing and _any_nifti_in_dir(out_path.parent) and out_path.exists():
        return out_path

    result = dcm2nii(
        str(dicom_dir),
        str(out_path),
        custom_naming="Modality_SeriesNumber",
        force_ras=True,
        compress=True,
        save_metadata=True,
        skip_existing=skip_existing,
    )
    if isinstance(result, list):
        # explicit_output_path should force a single return, but be defensive.
        if not result:
            raise RuntimeError(f"No NIfTI produced for {dicom_dir}")
        return Path(result[0])
    return Path(result)


def run_subject(
    subject: str,
    lay: GpetLayout,
    *,
    skip_existing: bool = True,
) -> ConvertResult:
    """Convert one subject into canonical CT/PT/T1 NIfTIs under the layout."""
    subj = str(subject).strip()
    if not subj:
        raise ValueError("subject must be non-empty")

    lay.nifti_dir.mkdir(parents=True, exist_ok=True)

    ct = _convert_one(dicom_dir=lay.dicom_ia_ct_dir(), out_path=lay.nifti_ct(), skip_existing=skip_existing)
    pet = _convert_one(dicom_dir=lay.dicom_ia_pet_dir(), out_path=lay.nifti_pet(), skip_existing=skip_existing)
    t1 = _convert_one(
        dicom_dir=lay.dicom_pesabrain_t1_dir(),
        out_path=lay.nifti_t1(),
        skip_existing=skip_existing,
    )

    manifest = lay.nifti_dir / "conversion_manifest.json"
    payload = {
        "subject": subj,
        "batch": lay.batch,
        "created_at": datetime.now().isoformat(),
        "outputs": {"CT": str(ct), "PT": str(pet), "T1": str(t1)},
        "inputs": {
            "dicom": {
                "IA_PET_V5/CT": str(lay.dicom_ia_ct_dir()),
                "IA_PET_V5/PET": str(lay.dicom_ia_pet_dir()),
                "PESA_Brain/3D_T1": str(lay.dicom_pesabrain_t1_dir()),
            }
        },
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info("[%s] gpetpy stage0 convert OK", subj)
    return ConvertResult(ct=ct, pet=pet, t1=t1, manifest=manifest)

