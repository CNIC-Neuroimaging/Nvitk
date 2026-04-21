"""CT-PET v5 stage 1 (per-subject): TotalSegmentator on CT input.

Runs every :data:`CT_TASKS` task sequentially for a single PESA* subject via
:func:`nvitk.segmentation.total_segmentator.run_totalsegmentator`. All outputs
are written under ``RESULTS/<batch>/res_segmentation_ct/<SUBJECT>/CT/`` with
the task name as the file stem (``<task>.nii.gz``).

Stage orchestration (loops over subjects, local vs. SGE dispatch) lives in
:mod:`nvitk.pipes.pesa_fat.ct_pet_v5.run`; this module is a pure worker.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import (
    BatchLayout,
    DEFAULT_MODEL_ROOT,
    layout,
    resolve_nii,
)
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as cfg
from nvitk.segmentation.total_segmentator import run_totalsegmentator


log = Logger()


def _subject_output_dir(lay: BatchLayout, subject: str) -> Path:
    """``RESULTS/<batch>/res_segmentation_ct/<subject>/CT/``."""
    return lay.results_dir / cfg.STAGE1_DIR / subject / "CT"


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    device: str = "gpu",
    model_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Run every :data:`cfg.CT_TASKS` task on a single subject's CT NIfTI.

    Returns the per-subject output directory.
    """
    subject_nifti = lay.subject_nifti_dir(subject)
    if not subject_nifti.exists():
        raise FileNotFoundError(f"Subject NIfTI dir not found: {subject_nifti}")

    input_ct = resolve_nii(subject_nifti, cfg.INPUT_STEM)
    output_dir = _subject_output_dir(lay, subject)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = model_dir or lay.model_dir or DEFAULT_MODEL_ROOT
    log.info("=" * 74)
    log.info(
        f"CT-PET v5 stage 1 | subject={subject} | device={device} "
        f"| tasks={[t.name for t in cfg.CT_TASKS]}"
    )
    log.info(f"  input : {input_ct}")
    log.info(f"  output: {output_dir}")
    log.info(f"  models: {model_dir}")
    log.info("=" * 74)

    for task in cfg.CT_TASKS:
        out_file = output_dir / f"{task.name}.nii"
        out_file_gz = output_dir / f"{task.name}.nii.gz"
        if not overwrite and (out_file.exists() or out_file_gz.exists()):
            log.info(f"[{subject}] {task.name:<28} -> up to date")
            continue

        log.info(f"[{subject}] {task.name:<28} -> running")
        try:
            run_totalsegmentator(
                input_ct,
                out_file,
                task=task.name,
                device=device,
                roi_subset=list(task.roi_subset) or None,
                multilabel=True,
                statistics=False,
                model_dir=model_dir,
                check=True,
                capture_output=False,
            )
        except Exception as exc:
            log.error(f"[{subject}] {task.name} failed: {exc}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("ctpet-v5-stage1")
@click.option("--batch", required=True)
@click.option("--subject", required=True, help="PESA* subject name.")
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--device",
    type=click.Choice(["gpu", "cpu"]),
    default="gpu",
    show_default=True,
)
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--log-level", default="INFO")
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    log_level: str,
) -> None:
    """CT-PET v5 stage 1 worker: run all TotalSegmentator tasks for a subject."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
        model_root=model_dir,
    )
    run_subject(
        subject,
        lay,
        device=device,
        model_dir=model_dir or lay.model_dir,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
