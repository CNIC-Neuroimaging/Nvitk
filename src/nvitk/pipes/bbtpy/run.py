"""
Black-blood pipeline runner (``nvitk-bbtpy``).

Stages (``--stages``, comma-separated)
--------------------------------------
``stage0_d`` / ``download``
    XNAT → DICOM (BrainVIEW VWI_BB, variant priority strong > default > weak).

``stage0_c`` / ``convert`` / ``stage0``
    DICOM → ``{nifti_root}/{subject}/BlackBlood/vwi_bb.nii.gz``.

``stage1`` / ``reg`` / ``registration``
    Rigid FLIRT: eICAB ``TOF_resampled`` (moving) → native ``vwi_bb`` (fixed).
    Writes ``tof_to_vwi_bb.mat`` and warped TOF QC volume.

``stage2`` / ``seg`` / ``segmentation``
    Warp eICAB to ``vwi_bb``, build centerlines (QC), then per-vessel dilated eICAB ROI
    hypointense thresholding → ``seg_bb.nii.gz``.

Default stages (no download): ``stage0_c,stage1,stage2``.

Data layout
-----------
- Native BB: ``{nifti_root}/{subject}/BlackBlood/vwi_bb.nii.gz``
- eICAB (qvtpy): ``{eicab_results_root}/{subject}/eicab/*_eICAB_{CW|WB}.nii.gz``
- Results: ``{output_root}/{subject}/bbtpy/``

Segmentation: dilated eICAB mask ROI + hypointense threshold (``util/bb_vessel_segmentation.py``).
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger, PipelineRunTracker
from nvitk.pipes.bbtpy import config as cfg
from nvitk.pipes.bbtpy import (
    stage0_convert,
    stage0_download,
    stage1_registration,
    stage2_bb_segmentation,
)
from nvitk.pipes.bbtpy.util import paths
from nvitk.pipes.bbtpy.util.eicab_masks import EicabMaskKind

log = Logger()

# ---------------------------------------------------------------------------
# Stage identifiers and aliases
# ---------------------------------------------------------------------------

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

_STAGE_LABELS: dict[str, str] = {
    STAGE_DOWNLOAD: "XNAT download (VWI_BB)",
    STAGE_CONVERT: "DICOM → vwi_bb NIfTI",
    STAGE_REG: "FLIRT TOF_resampled → vwi_bb",
    STAGE_SEG: "centerlines + BB segmentation",
}


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


@click.command("nvitk-bbtpy")
@backend_click_option()
@click.option(
    "--subjects",
    default=None,
    help="Comma-separated subject IDs. If omitted, all folders under --nifti-root are used.",
)
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Subject list (.txt / .csv / .xlsx); used for download and explicit cohort runs.",
)
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=None,
    help=f"XNAT download root [{cfg.DEFAULT_DICOM_ROOT}].",
)
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path),
    default=None,
    help=f"NIfTI tree with BlackBlood/vwi_bb per subject [{cfg.DEFAULT_NIFTI_ROOT}].",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Pipeline outputs (registration, segmentation) [{cfg.DEFAULT_RESULTS_ROOT}].",
)
@click.option(
    "--eicab-results-root",
    type=click.Path(path_type=Path),
    default=None,
    help=f"qvtpy eICAB results root [{cfg.DEFAULT_EICAB_RESULTS_ROOT}].",
)
@click.option(
    "--qvtpy-results-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Alias for --eicab-results-root (legacy name).",
)
@click.option(
    "--vwi-bb-rel-path",
    default=None,
    help=f"Relative path to native BB volume under each subject [{cfg.VWI_BB_REL_PATH}].",
)
@click.option("--wvi-rel-path", default=None, hidden=True)
@click.option(
    "--eicab-subdir",
    default=None,
    help=f"eICAB folder name under subject results [{cfg.EICAB_SUBDIR}].",
)
@click.option(
    "--eicab-mask",
    type=click.Choice(["cw", "wb"]),
    default="cw",
    show_default=True,
    help=(
        "eICAB multilabel for centerlines: Circle of Willis (cw) or whole-brain (wb). "
        "Falls back to the other mask with a warning if the requested file is missing."
    ),
)
@click.option(
    "--stages",
    default=_DEFAULT_STAGES,
    show_default=True,
    help="Comma-separated stages: stage0_d, stage0_c, stage1, stage2 (see module docstring).",
)
@click.option(
    "--with-download",
    is_flag=True,
    default=False,
    help="Prepend stage0_d (XNAT download) to --stages.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip steps when output artifacts already exist.",
)
@click.option(
    "--xnat-config",
    type=click.Path(path_type=Path),
    default=None,
    help="XNAT profile YAML (server, project, credentials).",
)
@click.option("--server", type=str, default=None, help="Override XNAT server URL.")
@click.option("--project", type=str, default=None, help="Override XNAT project ID.")
@click.option("--user", type=str, default=None, help="Override XNAT username.")
@click.option("--password", type=str, default=None, help="Override XNAT password.")
@click.option(
    "--netrc-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional netrc for XNAT authentication.",
)
@click.option(
    "--report",
    is_flag=True,
    default=False,
    help="After convert: print subjects missing vwi_bb under --nifti-root.",
)
@click.option(
    "--dof",
    type=int,
    default=6,
    show_default=True,
    help="FLIRT degrees of freedom (stage1: TOF_resampled → vwi_bb).",
)
@click.option(
    "--cost",
    default="normmi",
    show_default=True,
    help="FLIRT cost function (stage1).",
)
@click.option(
    "--eicab-dilate",
    type=int,
    default=4,
    show_default=True,
    help="Stage2: dilate each warped eICAB label before ROI thresholding.",
)
@click.option(
    "--thr-algorithm",
    type=click.Choice(["lsthr", "lthr", "otsu"]),
    default="otsu",
    show_default=True,
    help="Stage2: hypointense threshold inside dilated eICAB ROI (lsthr/lthr/otsu).",
)
@click.option(
    "--min-component-frac",
    type=float,
    default=0.005,
    show_default=True,
    help="Stage2: drop threshold islands smaller than this fraction of foreground.",
)
@click.option(
    "--min-centerline-points",
    type=int,
    default=3,
    show_default=True,
    help="Minimum skeleton points per eICAB label to keep a centerline.",
)
@click.option(
    "--vwi-preprocess",
    type=click.Choice(["none", "median", "gaussian"]),
    default="median",
    show_default=True,
    help="Light smoothing of vwi_bb before lumen segmentation.",
)
@click.option(
    "--vwi-median-size",
    type=int,
    default=3,
    show_default=True,
    help="Median filter kernel size (odd) when --vwi-preprocess=median.",
)
@click.option(
    "--vwi-gaussian-sigma",
    type=float,
    default=0.8,
    show_default=True,
    help="Gaussian sigma (voxels) when --vwi-preprocess=gaussian.",
)
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
    eicab_dilate: int,
    thr_algorithm: str,
    min_component_frac: float,
    min_centerline_points: int,
    vwi_preprocess: str,
    vwi_median_size: int,
    vwi_gaussian_sigma: float,
) -> None:
    """Black-blood (bbtpy): VWI_BB download, TOF→BB registration, centerline segmentation."""
    dicom = Path(dicom_root or cfg.DEFAULT_DICOM_ROOT)
    nifti = paths.require_path(nifti_root or cfg.DEFAULT_NIFTI_ROOT, "nifti_root")
    out = paths.require_path(output_root or cfg.DEFAULT_RESULTS_ROOT, "output_root")
    eicab_root = paths.resolve_eicab_results_root(
        eicab_results_root or qvtpy_results_root
    )
    paths.require_vwi_bb_rel_path(vwi_bb_rel_path or wvi_rel_path)
    rel = vwi_bb_rel_path or wvi_rel_path
    mask_kind: EicabMaskKind = eicab_mask.lower()

    stages_sel = _normalize_stages(stages)
    if with_download and STAGE_DOWNLOAD not in stages_sel:
        stages_sel = [STAGE_DOWNLOAD, *stages_sel]

    subj_list = _parse_subjects(subjects, subjects_file, nifti)
    if not subj_list:
        raise click.ClickException(
            f"No subjects resolved (use --subjects, --subjects-file, or populate {nifti})."
        )

    with PipelineRunTracker(
        log,
        "bbtpy",
        subj_list,
        stages_sel,
        stage_labels=_STAGE_LABELS,
    ) as run:
        if STAGE_DOWNLOAD in stages_sel:
            from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

            dl_subjects = (
                stage0_download.load_subjects(
                    subjects=subjects, subjects_file=subjects_file
                )
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

            def _download() -> None:
                stage0_download.run_download(
                    dl_subjects,
                    dicom_root=dicom,
                    xnat_config=conn,
                    skip_existing=skip_existing,
                    report=report,
                )

            run.run_stage(
                "(cohort)",
                STAGE_DOWNLOAD,
                _download,
                detail=f"{len(dl_subjects)} subject(s)",
            )

        for subj in subj_list:
            if STAGE_CONVERT in stages_sel:
                run.run_stage(
                    subj,
                    STAGE_CONVERT,
                    lambda s=subj: stage0_convert.run_subject(
                        s,
                        dicom_root=dicom,
                        nifti_root=nifti,
                        skip_existing=skip_existing,
                    ),
                )
            if STAGE_REG in stages_sel:
                run.run_stage(
                    subj,
                    STAGE_REG,
                    lambda s=subj: stage1_registration.run_subject(
                        s,
                        nifti_root=nifti,
                        output_root=out,
                        eicab_results_root=eicab_root,
                        skip_existing=skip_existing,
                        vwi_bb_rel=rel,
                        eicab_subdir=eicab_subdir,
                        dof=dof,
                        cost=cost,
                    ),
                )
            if STAGE_SEG in stages_sel:
                run.run_stage(
                    subj,
                    STAGE_SEG,
                    lambda s=subj: stage2_bb_segmentation.run_subject(
                        s,
                        nifti_root=nifti,
                        output_root=out,
                        eicab_results_root=eicab_root,
                        skip_existing=skip_existing,
                        vwi_bb_rel=rel,
                        eicab_subdir=eicab_subdir,
                        eicab_mask=mask_kind,
                        eicab_dilate=eicab_dilate,
                        thr_algorithm=thr_algorithm,
                        min_component_frac=min_component_frac,
                        min_centerline_points=min_centerline_points,
                        vwi_preprocess=vwi_preprocess,
                        vwi_median_size=vwi_median_size,
                        vwi_gaussian_sigma=vwi_gaussian_sigma,
                    ),
                )

    if report and STAGE_CONVERT in stages_sel:
        stage0_convert.report_subjects(nifti, subj_list)


if __name__ == "__main__":
    main()
