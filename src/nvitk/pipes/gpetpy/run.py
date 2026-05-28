"""gPET pipeline master (gpetpy).

Stages (select with ``--stages``; default ``download,convert,stage1``)
---------------------------------------------------------------------
- ``download`` — XNAT → DICOM (local only; requires ``--input-source xnat``).
- ``convert`` — DICOM → NIfTI (CT/PT/T1).
- ``stage1`` — PET brain crop (CT TotalSegmentator mask → PET grid → fixed Z window).
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger

from .layout import DEFAULT_DICOM_ROOT, DEFAULT_NIFTI_ROOT, DEFAULT_RESULTS_ROOT, GpetLayout
from .stage0_download import download_from_xnat
from .stage0_convert import run_subject as convert_subject
from .stage1_crop_brain_pet import run_subject as crop_subject

log = Logger()


STAGES_ORDERED: tuple[str, ...] = ("download", "convert", "stage1")
STAGE_ALIASES: dict[str, str] = {
    "download": "download",
    "stage0_d": "download",
    "stage0_download": "download",
    "convert": "convert",
    "stage0_c": "convert",
    "stage0_convert": "convert",
    "stage0": "convert",
    "stage1": "stage1",
    "crop": "stage1",
    "crop_brain": "stage1",
}


def _parse_stages(spec: str) -> list[str]:
    tokens = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        raise click.ClickException("--stages cannot be empty.")
    canonical: set[str] = set()
    for tok in tokens:
        key = tok.replace("-", "_")
        if key not in STAGE_ALIASES:
            raise click.ClickException(
                f"Unknown stage {tok!r}. Valid: {', '.join(sorted(set(STAGE_ALIASES.keys())))}"
            )
        canonical.add(STAGE_ALIASES[key])
    return [s for s in STAGES_ORDERED if s in canonical]


def _parse_subjects(value: str | None) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


@click.command("nvitk-gpetpy", context_settings={"help_option_names": ["-h", "--help"]})
@backend_click_option()
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated subject ids (required for --input-source xnat).",
)
@click.option(
    "--stages",
    default="download,convert,stage1",
    show_default=True,
    help="Comma-separated stages: download,convert,stage1",
)
@click.option(
    "--input-source",
    type=click.Choice(["paths", "xnat"], case_sensitive=False),
    default="xnat",
    show_default=True,
)
@click.option("--xnat-config", "xnat_config_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--batch",
    default="auto",
    show_default=True,
    help="Batch name (e.g. 202602_Week1). Use 'auto' with --input-source xnat.",
)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option("--device", type=click.Choice(["gpu", "cpu"]), default="gpu", show_default=True)
@click.option("--model-dir", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("--log-level", default="INFO", show_default=True)
def main(
    *,
    backend: str,  # noqa: ARG001
    subjects: str | None,
    stages: str,
    input_source: str,
    xnat_config_path: Path | None,
    batch: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    device: str,
    model_dir: Path | None,
    overwrite: bool,
    log_level: str,
) -> None:
    """Run gpetpy pipeline stages for selected subjects."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())

    stages_sel = _parse_stages(stages)
    subj_list = _parse_subjects(subjects)

    src = str(input_source).strip().lower()
    if src == "xnat":
        if not subj_list:
            raise click.ClickException("--input-source xnat requires --subjects.")
        # Download returns per subject (batch, layout) and may derive different batches.
        dl = download_from_xnat(
            subjects=subj_list,
            batch=batch,
            dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
            nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
            results_root=results_root or DEFAULT_RESULTS_ROOT,
            xnat_config_path=xnat_config_path,
        ) if "download" in stages_sel else {}

        # Determine layouts for subsequent stages: if download ran, use returned batches.
        layouts: dict[str, GpetLayout] = {}
        if dl:
            for s, (b, lay) in dl.items():
                layouts[s] = lay
        else:
            # Without download, require explicit batch (not auto) so we can locate files.
            if str(batch).strip().lower() == "auto":
                raise click.ClickException("--batch auto requires running the download stage in xnat mode.")
            for s in subj_list:
                layouts[s] = GpetLayout(
                    batch=str(batch).strip(),
                    subject=s,
                    dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
                    nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
                    results_root=results_root or DEFAULT_RESULTS_ROOT,
                )

        for subj, lay in layouts.items():
            if "convert" in stages_sel:
                convert_subject(subj, lay, skip_existing=not overwrite)
            if "stage1" in stages_sel:
                crop_subject(
                    subj,
                    lay,
                    device=device,
                    model_dir=model_dir,
                    overwrite=overwrite,
                )
        return

    # paths mode: assume batch + dicom/nifti already laid out
    if not subj_list:
        raise click.ClickException("--subjects is required.")
    if str(batch).strip().lower() == "auto":
        raise click.ClickException("--batch auto is only valid with --input-source xnat.")
    for s in subj_list:
        lay = GpetLayout(
            batch=str(batch).strip(),
            subject=s,
            dicom_root=dicom_root or DEFAULT_DICOM_ROOT,
            nifti_root=nifti_root or DEFAULT_NIFTI_ROOT,
            results_root=results_root or DEFAULT_RESULTS_ROOT,
        )
        if "convert" in stages_sel:
            convert_subject(s, lay, skip_existing=not overwrite)
        if "stage1" in stages_sel:
            crop_subject(s, lay, device=device, model_dir=model_dir, overwrite=overwrite)


if __name__ == "__main__":
    main()
