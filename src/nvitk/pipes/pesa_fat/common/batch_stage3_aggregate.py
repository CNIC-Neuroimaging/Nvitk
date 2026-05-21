"""Batch stage-3 Excel aggregation (invoked inside Singularity on the cluster).

Concatenates per-subject ``per_subject/<subj>.xlsx`` files into
``<batch>_SummaryCodebook.xlsx`` for each requested pipeline, mirroring
local :func:`nvitk.pipes.pesa_fat.common.stage3_batch_summary.aggregate_stage3_summary`.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import layout, parse_subjects
from nvitk.pipes.pesa_fat.common.stage3_batch_summary import aggregate_stage3_summary

log = Logger()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@backend_click_option()
@click.option("--batch", required=True, help="Batch name (e.g. 202602_Week4).")
@click.option(
    "--subjects",
    required=True,
    help="Comma-separated PESA* subjects (same list as the batch submission).",
)
@click.option(
    "--pipelines",
    required=True,
    help="Comma-separated: ct-pet-v5 and/or dixon-v5.",
)
@click.option("--dicom-root", type=click.Path(path_type=Path), required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--model-dir", type=click.Path(path_type=Path), required=True)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    batch: str,
    subjects: str,
    pipelines: str,
    dicom_root: Path,
    nifti_root: Path,
    results_root: Path,
    model_dir: Path,
    log_level: str,
) -> None:
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    subj_list = parse_subjects(subjects)
    if not subj_list:
        raise click.UsageError("--subjects must list at least one subject")

    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
        model_root=model_dir,
    )

    for raw in pipelines.split(","):
        p = raw.strip().lower()
        if not p:
            continue
        if p not in ("ct-pet-v5", "dixon-v5"):
            raise click.BadParameter(f"Unknown pipeline {raw!r}")
        aggregate_stage3_summary(lay, subj_list, p)


if __name__ == "__main__":
    main()
