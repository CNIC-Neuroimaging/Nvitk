"""Dixon v5 stage 1 (per-subject): TotalSegmentator on MR Dixon contrasts.

Runs the region-specific task plan (see :mod:`config`) for a single PESA*
subject. For each ``(region, task)`` pair it reads the matching Dixon
contrast (``DIXON_<REGION>_<SUFFIX>.nii[.gz]``) and writes the multilabel
segmentation to
``RESULTS/<batch>/res_segmentation_dixon/<SUBJECT>/DIXON_<REGION>/<task>.nii``.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import (
    BatchLayout,
    DEFAULT_MODEL_ROOT,
    layout,
    resolve_nii_optional,
)
from nvitk.pipes.pesa_fat.dixon_v5 import config as cfg
from nvitk.segmentation.total_segmentator import run_totalsegmentator


log = Logger()


def _find_input(subject_nifti_dir: Path, region: str, suffix: str) -> Path | None:
    return resolve_nii_optional(subject_nifti_dir, f"{cfg.INPUT_PREFIX}_{region}_{suffix}")


def _subject_region_output_dir(lay: BatchLayout, subject: str, region: str) -> Path:
    return (
        lay.results_dir
        / cfg.STAGE1_DIR
        / subject
        / f"{cfg.INPUT_PREFIX}_{region}"
    )


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    device: str = "gpu",
    model_dir: Path | None = None,
    overwrite: bool = True,
    regions: tuple[str, ...] = cfg.REGION_ORDER,
) -> dict[str, Path]:
    """Run every MR task for a single subject across the requested regions.

    Returns a ``{region: output_dir}`` mapping.
    """
    subject_nifti = lay.subject_nifti_dir(subject)
    if not subject_nifti.exists():
        raise FileNotFoundError(f"Subject NIfTI dir not found: {subject_nifti}")

    model_dir = model_dir or lay.model_dir or DEFAULT_MODEL_ROOT
    out_dirs: dict[str, Path] = {}

    log.info("=" * 74)
    log.info(
        f"Dixon v5 stage 1 | subject={subject} | device={device} "
        f"| regions={','.join(regions)}"
    )
    log.info(f"  input : {subject_nifti}")
    log.info(f"  models: {model_dir}")
    log.info("=" * 74)

    for region in regions:
        tasks = cfg.REGIONS[region]
        out_dir = _subject_region_output_dir(lay, subject, region)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[region] = out_dir

        for task in tasks:
            in_img = _find_input(subject_nifti, region, task.input_suffix)
            if in_img is None:
                log.warning(
                    f"[{subject}] {region}/{task.name}: missing "
                    f"{cfg.INPUT_PREFIX}_{region}_{task.input_suffix}.nii(.gz), skipping"
                )
                continue

            out_file = out_dir / f"{task.name}.nii"
            out_file_gz = out_dir / f"{task.name}.nii.gz"
            if not overwrite and (out_file.exists() or out_file_gz.exists()):
                log.info(f"[{subject}] {region}/{task.name:<28} -> up to date")
                continue

            log.info(f"[{subject}] {region}/{task.name:<28} -> running")
            try:
                run_totalsegmentator(
                    in_img,
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
                log.error(f"[{subject}] {region}/{task.name} failed: {exc}")

    return out_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("dixon-v5-stage1")
@click.option("--batch", required=True)
@click.option("--subject", required=True)
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
@click.option(
    "--regions",
    default=",".join(cfg.REGION_ORDER),
    show_default=True,
    help="Comma-separated regions to process.",
)
@click.option("--overwrite", is_flag=True, default=True)
@click.option("--log-level", default="INFO")
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    device: str,
    model_dir: Path | None,
    regions: str,
    overwrite: bool,
    log_level: str,
) -> None:
    """Dixon v5 stage 1 worker (single subject across the requested regions)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
        model_root=model_dir,
    )
    region_tuple = tuple(r.strip().upper() for r in regions.split(",") if r.strip())
    unknown = set(region_tuple) - set(cfg.REGIONS)
    if unknown:
        raise click.BadParameter(
            f"Unknown regions {unknown}. Valid: {tuple(cfg.REGIONS)}"
        )
    run_subject(
        subject,
        lay,
        device=device,
        model_dir=model_dir or lay.model_dir,
        overwrite=overwrite,
        regions=region_tuple,
    )


if __name__ == "__main__":
    main()
