"""Black-blood stage 0 (convert): DICOM → canonical ``BlackBlood/vwi_bb.nii.gz``."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

from nvitk.core.logger import Logger
from nvitk.io.conversors.dcm2nii import dcm2nii

from . import config as cfg

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Output filenames
# ──────────────────────────────────────────────────────────────────────────────

VWI_BB_NIFTI = "vwi_bb.nii.gz"
VWI_BB_JSON = "vwi_bb.json"
BLACK_BLOOD_DIR = "BlackBlood"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _iter_subjects(dicom_root: Path) -> list[str]:
    """Sorted subject directory names under *dicom_root*."""
    if not dicom_root.is_dir():
        return []
    return sorted(p.name for p in dicom_root.iterdir() if p.is_dir())


def _iter_nifti(folder: Path) -> list[Path]:
    """Sorted ``.nii`` / ``.nii.gz`` files in *folder*."""
    if not folder.is_dir():
        return []
    return sorted([*folder.glob("*.nii"), *folder.glob("*.nii.gz")])


def _nifti_stem(path: Path) -> str:
    """Filename stem with ``.nii`` / ``.nii.gz`` suffix removed."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def _matching_json(nifti_path: Path, search_dir: Path) -> Path | None:
    """Locate a JSON sidecar for *nifti_path* under *search_dir*, if present."""
    stem = _nifti_stem(nifti_path)
    for candidate in (search_dir / f"{stem}.json", search_dir / f"{nifti_path.name}.json"):
        if candidate.is_file():
            return candidate
    hits = sorted(search_dir.glob(f"{stem}*.json"))
    return hits[0] if hits else None


def vwi_bb_nifti_path(nifti_root: Path, subject: str) -> Path:
    """Canonical black-blood NIfTI path for *subject*."""
    return nifti_root / subject / BLACK_BLOOD_DIR / VWI_BB_NIFTI


def vwi_bb_json_path(nifti_root: Path, subject: str) -> Path:
    """Canonical black-blood JSON sidecar path for *subject*."""
    return nifti_root / subject / BLACK_BLOOD_DIR / VWI_BB_JSON


# ---------------------------------------------------------------------------
# Per-subject convert and batch runner
# ---------------------------------------------------------------------------


def convert_subject(
    subject: str,
    *,
    dicom_root: Path,
    nifti_root: Path,
    skip_existing: bool = False,
) -> Path:
    """Convert ``vwi_bb`` DICOMs and write canonical ``BlackBlood/vwi_bb`` outputs."""
    subj_dicom = dicom_root / subject / cfg.STAGE0_DICOM_SLOT
    if not subj_dicom.is_dir():
        raise FileNotFoundError(f"DICOM subject dir not found: {subj_dicom}")

    dest_nifti = vwi_bb_nifti_path(nifti_root, subject)
    dest_json = vwi_bb_json_path(nifti_root, subject)
    dest_nifti.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and dest_nifti.is_file():
        log.info(f"[{subject}] stage0 convert: skip existing -> {dest_nifti}")
        return dest_nifti

    tmp_dir = nifti_root / subject / ".bb_convert_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"black_blood stage0 convert | subject={subject}")
    dcm2nii(
        str(subj_dicom),
        str(tmp_dir),
        custom_naming="AccessionNumber_SeriesDescription_SeriesNumber",
        rescale_type="FP",
        force_ras=True,
        compress=True,
        save_metadata=True,
    )

    niftis = _iter_nifti(tmp_dir)
    if not niftis:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"No NIfTI produced under {tmp_dir} for {subject}")
    if len(niftis) > 1:
        log.warning(
            f"[{subject}] multiple NIfTI in convert tmp; using {niftis[0].name}"
        )

    src_nifti = niftis[0]
    if dest_nifti.exists():
        dest_nifti.unlink()
    shutil.move(str(src_nifti), str(dest_nifti))

    src_json = _matching_json(src_nifti, tmp_dir)
    if src_json is not None:
        if dest_json.exists():
            dest_json.unlink()
        shutil.move(str(src_json), str(dest_json))
    else:
        log.warning(f"[{subject}] no JSON sidecar found in {tmp_dir}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return dest_nifti


def run_subject(
    subject: str,
    *,
    dicom_root: Path,
    nifti_root: Path,
    skip_existing: bool = False,
) -> Path:
    """CLI entry: convert one subject (alias for :func:`convert_subject`)."""
    return convert_subject(
        subject,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        skip_existing=skip_existing,
    )


def report_subjects(nifti_root: Path, subjects: Iterable[str]) -> dict[str, Any]:
    """Print and return ``vwi_bb.nii.gz`` completeness for *subjects*."""
    root = Path(nifti_root)
    complete: list[str] = []
    incomplete: list[str] = []
    for subj in subjects:
        p = vwi_bb_nifti_path(root, subj)
        if p.is_file():
            complete.append(subj)
        else:
            incomplete.append(subj)
    print()
    print("NIfTI completeness report (black_blood vwi_bb)")
    print(f"  root: {root}")
    print(f"  complete: {len(complete)} / {len(list(subjects))}")
    if incomplete:
        print("-- Missing vwi_bb.nii.gz --")
        for subj in incomplete:
            print(f"  {subj}")
    return {"complete": complete, "incomplete": incomplete}
