"""qvtpy stage 0: DICOM -> NIfTI conversion and reorganization.

- conversion (dicom2nifti) with specific flags
- reorganization into a layout compatible with :mod:`nvitk.io.conversors.phase2volume`:

  * ``4DFlow/AP``, ``4DFlow/RL``, ``4DFlow/FH`` — magnitude ``*_m.nii`` and phase ``*_ph.nii``
    (from ``*_M_FFE`` / ``*_PHASE`` NIfTI), with matching ``*.json`` metadata beside them
  * ``TOF/TOF.nii`` plus any TOF-series ``*.json`` (original names)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import click

from nvitk.core.logger import Logger
from nvitk.io.conversors.dcm2nii import dcm2nii
from nvitk.io.conversors.phase2volume import phase2volume

from . import config as cfg

log = Logger()


def _iter_subjects(dicom_root: Path) -> list[str]:
    """Subject ids: one folder per subject under *dicom_root*."""
    if not dicom_root.exists():
        return []
    return sorted(p.name for p in dicom_root.iterdir() if p.is_dir())


def _iter_nifti(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted([*folder.glob("*.nii"), *folder.glob("*.nii.gz")])


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def _flow_direction(stem: str) -> str | None:
    u = stem.upper()
    if "_AP_" in u:
        return "AP"
    if "_RL_" in u:
        return "RL"
    if "_FH_" in u:
        return "FH"
    return None


def _classify_flow_stem(stem: str) -> tuple[str, str] | None:
    """Return (AP|RL|FH, 'm'|'ph') for 4DFlow magnitude / phase stems, else None."""
    direction = _flow_direction(stem)
    if direction is None:
        return None
    su = stem.upper()
    if su.endswith("_M_FFE"):
        return (direction, "m")
    if su.endswith("_PHASE"):
        return (direction, "ph")
    return None


def _flow_dest_nifti_name(stem: str, kind: str) -> str:
    su = stem.upper()
    if kind == "m":
        if not su.endswith("_M_FFE"):
            raise ValueError(stem)
        base = stem[: len(stem) - len("_M_FFE")]
    else:
        if not su.endswith("_PHASE"):
            raise ValueError(stem)
        base = stem[: len(stem) - len("_PHASE")]
    return f"{base}_{kind}.nii"


def _is_tof_stem(stem: str) -> bool:
    """Heuristic for TOF / angio series (not 4DFlow)."""
    if _classify_flow_stem(stem) is not None:
        return False
    u = stem.upper()
    return any(
        k in u
        for k in (
            "TOF",
            "MRA",
            "ANGIO",
            "TOF3D",
            "CS3DI",
            "3DI",
        )
    )


def _export_nifti_move(src: Path, dst: Path) -> None:
    """Write *dst* as uncompressed NIfTI (.nii), then remove *src* if different path."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log.warning(f"Replacing existing file: {dst}")
        dst.unlink()
    try:
        import nibabel as nib

        img = nib.load(str(src))
        nib.save(img, str(dst))
    except Exception as exc:
        raise RuntimeError(f"Failed to convert NIfTI {src} -> {dst}: {exc}") from exc
    if src.resolve() != dst.resolve():
        src.unlink(missing_ok=True)


def convert_subject(
    subject_dicom_dir: Path,
    subject_out_dir: Path,
    *,
    skip_existing: bool = False,
) -> None:
    """Run DICOM->NIfTI conversion using required flags."""
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and any(_iter_nifti(subject_out_dir)):
        log.info(f"[{subject_dicom_dir.name}] stage0: skipping existing conversion")
        return

    dcm2nii(
        str(subject_dicom_dir),
        str(subject_out_dir),
        custom_naming="AccessionNumber_SeriesDescription_SeriesNumber",
        rescale_type="FP",
        force_ras=True,
        compress=True,
        save_metadata=True,
    )


def reorganize_subject(subject_out_dir: Path) -> None:
    """Move flat dcm2nii outputs into ``4DFlow/{AP,RL,FH}/`` and ``TOF/`` with QVT naming."""
    flow_root = subject_out_dir / "4DFlow"
    tof_dir = subject_out_dir / "TOF"
    for sub in ("AP", "RL", "FH"):
        (flow_root / sub).mkdir(parents=True, exist_ok=True)
    tof_dir.mkdir(parents=True, exist_ok=True)

    top_files = sorted(p for p in subject_out_dir.iterdir() if p.is_file())
    niftis = [p for p in top_files if p.name.endswith(".nii.gz") or p.name.endswith(".nii")]
    jsons = [p for p in top_files if p.suffix == ".json"]

    for src in niftis:
        stem = _nifti_stem(src)
        flow = _classify_flow_stem(stem)
        if flow is not None:
            direction, kind = flow
            dest_name = _flow_dest_nifti_name(stem, kind)
            dest = flow_root / direction / dest_name
            _export_nifti_move(src, dest)
            continue

        if _is_tof_stem(stem):
            dest = tof_dir / "TOF.nii"
            if dest.exists() and src.resolve() != dest.resolve():
                log.warning(f"Multiple TOF-like series; overwriting {dest} with {src.name}")
            _export_nifti_move(src, dest)
            continue

        log.warning(f"[{subject_out_dir.name}] leaving unclassified NIfTI at subject root: {src.name}")

    for js in jsons:
        stem = js.stem
        flow = _classify_flow_stem(stem)
        if flow is not None:
            direction, _kind = flow
            dest = flow_root / direction / js.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                log.warning(f"Replacing existing JSON: {dest}")
                dest.unlink()
            shutil.move(str(js), str(dest))
            continue

        if _is_tof_stem(stem):
            dest = tof_dir / js.name
            if dest.exists():
                log.warning(f"Replacing existing JSON: {dest}")
                dest.unlink()
            shutil.move(str(js), str(dest))
            continue

        log.warning(f"[{subject_out_dir.name}] leaving unclassified JSON at subject root: {js.name}")


def run_subject(
    subject: str,
    *,
    dicom_root: Path,
    nifti_root: Path,
    compute_phase_derived: bool = False,
    skip_existing: bool = False,
) -> Path:
    """Convert + reorganize for one subject. Returns the subject NIfTI folder."""
    subj_dicom = dicom_root / subject
    if not subj_dicom.exists():
        raise FileNotFoundError(f"DICOM subject dir not found: {subj_dicom}")

    subj_nifti = nifti_root / subject
    log.info(f"qvtpy stage0 | subject={subject}")
    log.info(f"  dicom: {subj_dicom}")
    log.info(f"  nifti: {subj_nifti}")

    convert_subject(subj_dicom, subj_nifti, skip_existing=skip_existing)
    reorganize_subject(subj_nifti)

    if compute_phase_derived:
        try:
            phase2volume(subj_nifti, multifile=False)
        except Exception as exc:
            print(traceback.format_exc())
            log.warning(f"[{subject}] phase2volume failed: {exc}")

    return subj_nifti


@click.command("qvtpy-stage0")
@click.option("--dicom-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_DICOM_ROOT)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_NIFTI_ROOT)
@click.option(
    "--subject",
    default=None,
    help="Single subject id. If omitted, all subfolders of --dicom-root are processed.",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--compute-phase-derived", is_flag=True, default=False)
def main(
    dicom_root: Path,
    nifti_root: Path,
    subject: str | None,
    skip_existing: bool,
    compute_phase_derived: bool,
) -> None:
    Logger()
    if subject:
        run_subject(
            subject,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            compute_phase_derived=compute_phase_derived,
            skip_existing=skip_existing,
        )
        return

    subjects = _iter_subjects(dicom_root)
    if not subjects:
        raise click.ClickException(
            f"No subject folders found under dicom_root={dicom_root}. "
            "Pass --subject or point --dicom-root at a directory of subject subfolders."
        )
    for subj in subjects:
        run_subject(
            subj,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            compute_phase_derived=compute_phase_derived,
            skip_existing=skip_existing,
        )


__all__ = ["run_subject", "main"]
