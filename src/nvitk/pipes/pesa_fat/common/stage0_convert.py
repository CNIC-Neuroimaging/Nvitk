"""PESA-Fat stage 0 (shared): per-subject DICOM -> NIfTI conversion and renaming.

Port of :code:`BioImaging/src/pesa_fat/ct_pet/0_conversion_and_organization.sh`.

Responsibilities (applied to a single ``PESA*`` subject at a time):

1. Invoke :func:`nvitk.io.conversors.dcm2nii.dcm2nii` with the flags the
   legacy shell script used (``--force-ras --multifile --naming
   Modality_SeriesNumber``). The call produces
   ``NIFTI_ROOT/<batch>/<SUBJECT>/`` with files named
   ``<Modality>_<SeriesNumber>[_<suffix>].nii[.gz]``.
2. Rename the per-subject NIfTI files into the canonical stage-0 layout:

      * ``CT_<series>.nii``           -> ``CT.nii``
      * ``PT_<series>.nii``           -> ``PT.nii``
      * ``MR_<series>_<metric>.nii``  -> ``DIXON_<REGION>_<metric>.nii``
        (region from the series-number last digit: 1=HEAD, 2=THORAX, 3=LEGS)

The script is idempotent: converted subject folders that already contain
NIfTI files are skipped; already-renamed targets are kept (``Path.replace`` is
only invoked when the destination does not yet exist).

Runnable as a standalone CLI
(``python -m nvitk.pipes.pesa_fat.common.stage0_convert --batch X --subject PESA001``)
and importable as :func:`run_subject` by the batch master.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click

from nvitk.core.logger import Logger
from nvitk.io.conversors.dcm2nii import dcm2nii
from nvitk.pipes.pesa_fat.common.paths import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_NIFTI_ROOT,
    BatchLayout,
    layout,
    parse_subjects,
)


log = Logger()


_REGION_BY_LAST_DIGIT: dict[str, str] = {"1": "HEAD", "2": "THORAX", "3": "LEGS"}


# ---------------------------------------------------------------------------
# Rename policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenameResult:
    """Outcome of :func:`_plan_rename` for a single file."""

    src: Path
    dst: Path | None
    reason: str


def _plan_rename(src: Path) -> RenameResult:
    name = src.name
    if name.startswith("CT_") and (name.endswith(".nii") or name.endswith(".nii.gz")):
        ext = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        return RenameResult(src, src.with_name(f"CT{ext}"), "ct")
    if name.startswith("PT_") and (name.endswith(".nii") or name.endswith(".nii.gz")):
        ext = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        return RenameResult(src, src.with_name(f"PT{ext}"), "pt")
    if name.startswith("MR_") and (name.endswith(".nii") or name.endswith(".nii.gz")):
        ext = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        rest = name[len("MR_") :]
        rest = rest[: -len(ext)]
        parts = rest.split("_", 1)
        if len(parts) < 2:
            return RenameResult(src, None, "mr-missing-suffix")
        series, suffix = parts
        last_digit = series[-1] if series else ""
        region = _REGION_BY_LAST_DIGIT.get(last_digit)
        if region is None:
            return RenameResult(src, None, f"mr-unknown-series:{series}")
        return RenameResult(src, src.with_name(f"DIXON_{region}_{suffix}{ext}"), "mr")
    return RenameResult(src, None, "unknown-prefix")


def _iter_nifti_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted([*folder.glob("*.nii"), *folder.glob("*.nii.gz")])


def rename_subject_folder(folder: Path) -> dict[str, int]:
    """Apply the stage-0 rename policy to every NIfTI file in ``folder``."""
    counts = {"renamed": 0, "kept": 0, "skipped": 0}
    for src in _iter_nifti_files(folder):
        plan = _plan_rename(src)
        if plan.dst is None:
            counts["skipped"] += 1
            log.debug(f"skip: {src.name} ({plan.reason})")
            continue
        if plan.dst == src:
            counts["kept"] += 1
            continue
        if plan.dst.exists():
            counts["kept"] += 1
            log.debug(f"dst exists: {plan.dst.name} (leaving {src.name} alone)")
            continue
        src.replace(plan.dst)
        counts["renamed"] += 1
        log.info(f"  {src.name} -> {plan.dst.name}")
    return counts


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------


def _subject_has_nifti_outputs(folder: Path) -> bool:
    if not folder.exists():
        return False
    for p in folder.iterdir():
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            return True
    return False


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    naming: str = "Modality_SeriesNumber",
    force_ras: bool = True,
    skip_existing: bool = True,
    compress: bool = True,
    
) -> Path:
    """Run stage 0 for a single PESA* subject.

    Returns the subject NIfTI directory (created if missing). Raises
    :class:`FileNotFoundError` if the DICOM subject directory does not exist.
    """
    subject_dicom = lay.subject_dicom_dir(subject)
    if not subject_dicom.exists():
        raise FileNotFoundError(
            f"DICOM subject directory not found: {subject_dicom}"
        )

    subject_nifti = lay.subject_nifti_dir(subject)

    if skip_existing and _subject_has_nifti_outputs(subject_nifti):
        log.info(f"[{subject}] already converted, skipping conversion")
    else:
        subject_nifti.mkdir(parents=True, exist_ok=True)
        log.info(f"[{subject}] converting ...")
        dcm2nii(
            str(subject_dicom),
            str(subject_nifti),
            custom_naming=naming,
            force_ras=force_ras,
            compress=compress,
            skip_existing=skip_existing,
        )

    summary = rename_subject_folder(subject_nifti)
    log.info(
        f"[{subject}] rename done: "
        + ", ".join(f"{k}={v}" for k, v in summary.items())
    )
    return subject_nifti


def run_batch(
    batch: str,
    *,
    dicom_root: Path | str | None = None,
    nifti_root: Path | str | None = None,
    subjects: list[str] | None = None,
    naming: str = "Modality_SeriesNumber",
    force_ras: bool = True,
    skip_existing: bool = True,
    compress: bool = True,
) -> BatchLayout:
    """Run stage 0 for every PESA* subject in ``batch`` (local / sequential)."""
    lay = layout(batch, dicom_root=dicom_root, nifti_root=nifti_root)
    lay.nifti_dir.mkdir(parents=True, exist_ok=True)

    if not lay.dicom_dir.exists():
        raise FileNotFoundError(f"DICOM batch directory not found: {lay.dicom_dir}")

    subject_dirs = lay.subject_dicom_dirs()
    if subjects:
        wanted = set(subjects)
        subject_dirs = [d for d in subject_dirs if d.name in wanted]
    if not subject_dirs:
        raise FileNotFoundError(
            f"No PESA* subject directories found in {lay.dicom_dir} (filter={subjects})"
        )

    log.info("=" * 74)
    log.info(
        f"PESA-Fat stage 0 | batch={batch} | {len(subject_dirs)} subject(s)"
    )
    log.info(f"  DICOM: {lay.dicom_dir}")
    log.info(f"  NIFTI: {lay.nifti_dir}")
    log.info("=" * 74)

    for subject_dir in subject_dirs:
        try:
            run_subject(
                subject_dir.name,
                lay,
                naming=naming,
                force_ras=force_ras,
                skip_existing=skip_existing,
                compress=compress,
            )
        except Exception as exc:
            log.error(f"[{subject_dir.name}] stage 0 failed: {exc}")

    log.info(f"Stage 0 complete for batch {batch}")
    return lay


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("stage0-convert")
@click.option("--batch", required=True, help="Batch name (e.g. '202602_Week4').")
@click.option(
    "--subject",
    default=None,
    help="Single PESA* subject to process (SGE chain entry point).",
)
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated PESA* subjects for local mode (omit to process all).",
)
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=None,
    help=f"DICOM root; default {DEFAULT_DICOM_ROOT}.",
)
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path),
    default=None,
    help=f"NIfTI root; default {DEFAULT_NIFTI_ROOT}.",
)
@click.option("--naming", default="Modality_SeriesNumber", show_default=True)
@click.option("--no-force-ras", is_flag=True, help="Disable RAS reorientation.")
@click.option(
    "--no-skip-existing",
    is_flag=True,
    help="Re-convert already populated subjects.",
)
@click.option(
    "--no-compress",
    is_flag=True,
    help="Save uncompressed .nii (default is .nii.gz).",
)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str,
    subject: str | None,
    subjects: str | None,
    dicom_root: Path | None,
    nifti_root: Path | None,
    naming: str,
    no_force_ras: bool,
    no_skip_existing: bool,
    no_compress: bool,
    log_level: str,
) -> None:
    """Convert and re-organise a PESA-Fat DICOM batch (single-subject or full batch)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    if subject:
        lay = layout(batch, dicom_root=dicom_root, nifti_root=nifti_root)
        run_subject(
            subject,
            lay,
            naming=naming,
            force_ras=not no_force_ras,
            skip_existing=not no_skip_existing,
            compress=not no_compress,
        )
        return

    subj_list = parse_subjects(subjects)
    run_batch(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        subjects=subj_list,
        naming=naming,
        force_ras=not no_force_ras,
        skip_existing=not no_skip_existing,
        compress=not no_compress,
    )


if __name__ == "__main__":
    main()
