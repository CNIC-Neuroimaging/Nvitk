"""Black-blood pipeline runner (all stages via ``nvitk-pesa-brain-bb``)."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_brain.black_blood import config as cfg
from nvitk.pipes.pesa_brain.black_blood import (
    stage0_convert,
    stage0_download,
    stage1_registration,
    stage2_bb_segmentation,
)
from nvitk.pipes.pesa_brain.black_blood.util import paths
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import SegStrategy
from nvitk.pipes.pesa_brain.black_blood.util.eicab_masks import EicabMaskKind

log = Logger()

STAGE_DOWNLOAD = "stage0_d"
STAGE_CONVERT = "stage0_c"
STAGE_REG = "stage1"
STAGE_SEG = "stage2"

_STAGE_ALIASES: dict[str, str] = {
    "stage0_d": STAGE_DOWNLOAD,
    "stage0d": STAGE_DOWNLOAD,
    "stage0_download": STAGE_DOWNLOAD,
    "download": STAGE_DOWNLOAD,
    "stage0_c": STAGE_CONVERT,
    "stage0c": STAGE_CONVERT,
    "stage0_convert": STAGE_CONVERT,
    "stage0": STAGE_CONVERT,
    "convert": STAGE_CONVERT,
    "stage1": STAGE_REG,
    "stage1_registration": STAGE_REG,
    "registration": STAGE_REG,
    "reg": STAGE_REG,
    "stage2": STAGE_SEG,
    "stage2_bb_segmentation": STAGE_SEG,
    "segmentation": STAGE_SEG,
    "seg": STAGE_SEG,
}

_DEFAULT_STAGES = f"{STAGE_CONVERT},{STAGE_REG},{STAGE_SEG}"


def _normalize_stages(stages: str) -> list[str]:
    out: list[str] = []
    for raw in stages.split(","):
        s = raw.strip().lower()
        if not s:
            continue
        key = _STAGE_ALIASES.get(s, s)
        if key not in (STAGE_DOWNLOAD, STAGE_CONVERT, STAGE_REG, STAGE_SEG):
            raise click.BadParameter(
                f"Unknown stage {raw!r}. Valid: stage0_d, stage0_c, stage1, stage2 "
                "(aliases: download, convert, reg, seg)."
            )
        if key not in out:
            out.append(key)
    return out or [STAGE_CONVERT, STAGE_REG, STAGE_SEG]


def _parse_subjects(
    subjects: str | None,
    subjects_file: Path | None,
    nifti_root: Path,
) -> list[str]:
    if subjects_file is not None:
        return stage0_download.load_subjects(subjects=None, subjects_file=subjects_file)
    if subjects:
        return [s.strip() for s in subjects.split(",") if s.strip()]
    if not nifti_root.is_dir():
        return []
    return sorted(
        p.name
        for p in nifti_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


@click.command("nvitk-pesa-brain-bb")
@click.option("--subjects", default=None, help="Comma-separated subject ids.")
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Subject list file (.txt / .csv / .xlsx) for download stage.",
)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--eicab-results-root", type=click.Path(path_type=Path), default=None)
@click.option("--qvtpy-results-root", type=click.Path(path_type=Path), default=None)
@click.option("--vwi-bb-rel-path", default=None)
@click.option("--wvi-rel-path", default=None, hidden=True)
@click.option("--eicab-subdir", default=None)
@click.option(
    "--eicab-mask",
    type=click.Choice(["cw", "wb"]),
    default="cw",
    show_default=True,
    help="eICAB multilabel mask for centerlines/segmentation (CW or WB).",
)
@click.option("--stages", default=_DEFAULT_STAGES, show_default=True)
@click.option(
    "--with-download",
    is_flag=True,
    default=False,
    help="Prepend stage0_d (XNAT download) to --stages.",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--xnat-config", type=click.Path(path_type=Path), default=None)
@click.option("--server", type=str, default=None)
@click.option("--project", type=str, default=None)
@click.option("--user", type=str, default=None)
@click.option("--password", type=str, default=None)
@click.option("--netrc-file", type=click.Path(path_type=Path), default=None)
@click.option("--report", is_flag=True, default=False, help="QC report after download/convert.")
@click.option("--dof", type=int, default=6, show_default=True)
@click.option("--cost", default="normmi", show_default=True)
@click.option(
    "--seg-strategy",
    type=click.Choice(["crop-resegment", "centerline-growth"]),
    default="crop-resegment",
    show_default=True,
)
@click.option(
    "--thr-algorithm",
    type=click.Choice(["otsu", "lsthr", "lthr"]),
    default="otsu",
    show_default=True,
)
@click.option("--crop-padding-bbox", type=int, default=3, show_default=True)
@click.option("--cl-barrier-radius", type=int, default=2, show_default=True)
@click.option("--min-component-frac", type=float, default=0.005, show_default=True)
@click.option("--rg-intensity-frac", type=float, default=1.5, show_default=True)
@click.option("--rg-barrier-radius", type=int, default=2, show_default=True)
def main(
    subjects: str | None,
    subjects_file: Path | None,
    dicom_root: Path | None,
    nifti_root: Path | None,
    output_root: Path | None,
    eicab_results_root: Path | None,
    qvtpy_results_root: Path | None,
    vwi_bb_rel_path: str | None,
    wvi_rel_path: str | None,
    eicab_subdir: str | None,
    eicab_mask: str,
    stages: str,
    with_download: bool,
    skip_existing: bool,
    xnat_config: Path | None,
    server: str | None,
    project: str | None,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    report: bool,
    dof: int,
    cost: str,
    seg_strategy: str,
    thr_algorithm: str,
    crop_padding_bbox: int,
    cl_barrier_radius: int,
    min_component_frac: float,
    rg_intensity_frac: float,
    rg_barrier_radius: int,
) -> None:
    """Run black-blood stages: download, convert, registration, segmentation."""
    dicom = Path(dicom_root or cfg.DEFAULT_DICOM_ROOT)
    nifti = paths.require_path(nifti_root or cfg.DEFAULT_NIFTI_ROOT, "nifti_root")
    out = paths.require_path(output_root or cfg.DEFAULT_RESULTS_ROOT, "output_root")
    eicab_root = paths.resolve_eicab_results_root(
        eicab_results_root or qvtpy_results_root
    )
    paths.require_vwi_bb_rel_path(vwi_bb_rel_path or wvi_rel_path)
    rel = vwi_bb_rel_path or wvi_rel_path
    mask_kind: EicabMaskKind = eicab_mask.lower()  # type: ignore[assignment]

    stages_sel = _normalize_stages(stages)
    if with_download and STAGE_DOWNLOAD not in stages_sel:
        stages_sel = [STAGE_DOWNLOAD, *stages_sel]

    subj_list = _parse_subjects(subjects, subjects_file, nifti)
    if not subj_list:
        raise click.ClickException(
            f"No subjects resolved (use --subjects, --subjects-file, or populate {nifti})."
        )

    strategy: SegStrategy = seg_strategy  # type: ignore[assignment]

    if STAGE_DOWNLOAD in stages_sel:
        from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

        dl_subjects = (
            stage0_download.load_subjects(subjects=subjects, subjects_file=subjects_file)
            if subjects_file is not None or subjects
            else subj_list
        )
        profile = load_xnat_profile(xnat_config)
        conn = resolve_xnat_connection(
            profile,
            server=server,
            project=project,
            user=user,
            password=password,
            netrc_file=str(netrc_file) if netrc_file else None,
        )
        stage0_download.run_download(
            dl_subjects,
            dicom_root=dicom,
            xnat_config=conn,
            skip_existing=skip_existing,
            report=report,
        )

    for subj in subj_list:
        log.info(
            f"=== black_blood | subject={subj} | stages={stages_sel} | "
            f"eicab_mask={mask_kind} ==="
        )
        try:
            if STAGE_CONVERT in stages_sel:
                stage0_convert.run_subject(
                    subj,
                    dicom_root=dicom,
                    nifti_root=nifti,
                    skip_existing=skip_existing,
                )
            if STAGE_REG in stages_sel:
                stage1_registration.run_subject(
                    subj,
                    nifti_root=nifti,
                    output_root=out,
                    eicab_results_root=eicab_root,
                    skip_existing=skip_existing,
                    vwi_bb_rel=rel,
                    eicab_subdir=eicab_subdir,
                    dof=dof,
                    cost=cost,
                )
            if STAGE_SEG in stages_sel:
                stage2_bb_segmentation.run_subject(
                    subj,
                    nifti_root=nifti,
                    output_root=out,
                    eicab_results_root=eicab_root,
                    seg_strategy=strategy,
                    skip_existing=skip_existing,
                    vwi_bb_rel=rel,
                    eicab_subdir=eicab_subdir,
                    eicab_mask=mask_kind,
                    thr_algorithm=thr_algorithm,  # type: ignore[arg-type]
                    crop_padding_bbox=crop_padding_bbox,
                    cl_barrier_radius=cl_barrier_radius,
                    min_component_frac=min_component_frac,
                    rg_intensity_frac=rg_intensity_frac,
                    rg_barrier_radius=rg_barrier_radius,
                )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            log.error(f"[{subj}] failed: {exc}")

    if report and STAGE_CONVERT in stages_sel:
        stage0_convert.report_subjects(nifti, subj_list)


if __name__ == "__main__":
    main()
