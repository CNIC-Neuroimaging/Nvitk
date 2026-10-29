"""qvtpy stage 7: TOF eICAB morphometrics (cow_morpho port).

**Inputs**

- eICAB **CW** (Circle-of-Willis; WB fallback with a warning) multilabel mask under
  ``<output_root>/<subject>/eicab/``. The default ``eicab_cw`` morphometrics profile is tuned for
  CW masks (cow_morpho); whole-brain masks need their own profile.

**Outputs**

- ``vessel_summary.csv`` / ``segments.csv`` / ``lesions.csv`` / ``subject_summary.csv``,
  ``case_metrics_donut_tree.xlsx``, centerline/surface VTPs (scanner RAS), tortuosity tables,
  radius histograms and ``morphometrics_provenance.json`` under
  ``<output_root>/<subject>/qvtpy/stage7_morphometrics/``.
"""

from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.measure.morphometrics import run_morphometrics_case
from nvitk.measure.morphometrics_config import (
    CENTERLINE_BACKENDS,
    DEFAULT_PROFILE,
    LESION_DETECTION_METHODS,
    PROFILES,
    MorphometricsConfig,
    morphometrics_config,
)
from nvitk.measure.morpho.topology_io import default_eicab_topology_path
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.eicab.morpho_paths import (
    STAGE7_SKIP_MARKER,
    resolve_stage7_seg_mask,
    stage7_dir,
)
from nvitk.pipes.qvtpy.util.io.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)

log = Logger()


def _default_nvitk_src_dir() -> Path:
    """The nvitk package's ``src`` directory (parent of the ``nvitk`` module), for container mounts."""
    return Path(nvitk.__file__).resolve().parent.parent


def run_subject(
    subject: str,
    *,
    output_root: Path,
    skip_existing: bool = False,
    eicab_mask_preference: str = "wb",
    use_postprocessed_mask: bool = True,
    taubin: bool | None = None,
    lesion_detection_method: str | None = None,
    n_workers: int | None = None,
    profile: str = DEFAULT_PROFILE,
    centerline_backend: str | None = None,
    morpho_config: MorphometricsConfig | None = None,
) -> Path:
    """Run stage-7 TOF morphometrics for one subject (*morpho_config* overrides *profile*)."""
    out_dir = stage7_dir(output_root, subject)
    excel_path = out_dir / STAGE7_SKIP_MARKER
    if skip_existing and excel_path.is_file():
        log.info(f"[{subject}] stage7 morphometrics: skip -> {out_dir}")
        return out_dir

    mask_res = resolve_stage7_seg_mask(
        output_root,
        subject,
        preference="cw" if eicab_mask_preference.lower() == "cw" else "wb",
        prefer_postprocessed=use_postprocessed_mask,
    )
    if mask_res.fallback:
        log.warning(f"[{subject}] stage7: {mask_res.fallback_reason}")

    log.info(
        f"[{subject}] stage7 morphometrics: input={mask_res.path.name} "
        f"(pp={mask_res.postprocessed}, used={mask_res.used})"
    )
    if taubin is None:  # eICAB's own mask is not drawn at the lumen's half maximum: smooth it
        taubin = not mask_res.postprocessed
    if lesion_detection_method:
        morpho_config = (morpho_config or morphometrics_config(profile)).with_params(
            LESION_DETECTION_METHOD=str(lesion_detection_method).lower())
    run_morphometrics_case(
        mask_res.path,
        out_dir,
        mapping_json=str(default_eicab_topology_path()),
        case_out_dir_override=out_dir,
        n_workers=n_workers,
        profile=profile,
        config=morpho_config,
        centerline_backend=centerline_backend,
        input_already_smoothed=not bool(taubin),
        skip_if_excel_exists=skip_existing,
        case_id=subject,
    )
    log.info(f"[{subject}] stage7 morphometrics -> {excel_path}")

    try:
        from nvitk.pipes.qvtpy.common.morpho_db_publish import maybe_publish_stage7_on_sge

        maybe_publish_stage7_on_sge(subject_uid=subject, stage7_dir=out_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[{subject}] stage7 DB publish hook skipped: {exc}")

    return out_dir


def _subject_sge_spec(
    subject: str,
    *,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    eicab_mask_preference: str = "wb",
    use_postprocessed_mask: bool = True,
    taubin: bool | None = None,
    lesion_detection_method: str | None = None,
    n_workers: int | None = None,
    profile: str = DEFAULT_PROFILE,
    centerline_backend: str | None = None,
    backend: str = "cpu",
) -> tuple[StageSpec, ClusterPaths]:
    """Build the SGE stage spec and cluster paths for running stage-7 morphometrics on *subject*."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage7_morphometrics"),
        *sge_backend_cli_args(backend),
        "--subject",
        shlex.quote(subject),
        "--output-root",
        shlex.quote(binds.output),
        "--eicab-mask-preference",
        shlex.quote(eicab_mask_preference),
    ]
    if use_postprocessed_mask:
        parts.append("--use-postprocessed-mask")
    else:
        parts.append("--no-use-postprocessed-mask")
    if taubin is not None:
        parts.append("--taubin" if taubin else "--no-taubin")
    if lesion_detection_method:
        parts.extend(["--lesion-detection-method", shlex.quote(str(lesion_detection_method))])
    if skip_existing:
        parts.append("--skip-existing")
    if n_workers is not None:
        parts.extend(["--n-workers", str(int(n_workers))])
    parts.extend(["--profile", shlex.quote(profile)])
    if centerline_backend:
        parts.extend(["--centerline-backend", shlex.quote(centerline_backend)])
    python_cmd = " ".join(parts)
    paths = ClusterPaths(
        src=src_p,
        container=container,
        models=None,
        data_root=output_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    resources = sge_qvtpy_stage_resources(backend)
    if n_workers is not None and int(n_workers) > 1:
        # One CPU slot per parallel vessel job (qsub -pe smp N).
        resources = replace(resources, pe_smp=max(int(resources.pe_smp or 1), int(n_workers)))
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage7_{subject}",
        python_cmd=python_cmd,
        resources=resources,
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return spec, paths


def build_subject_sge_command(
    subject: str,
    *,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    eicab_mask_preference: str = "wb",
    use_postprocessed_mask: bool = True,
    taubin: bool | None = None,
    lesion_detection_method: str | None = None,
    n_workers: int | None = None,
    profile: str = DEFAULT_PROFILE,
    centerline_backend: str | None = None,
    backend: str = "cpu",
) -> str:
    """Return the host shell command for one stage7 array/SGE task."""
    from nvitk.cluster.sge import build_singularity_command

    spec, paths = _subject_sge_spec(
        subject,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        eicab_mask_preference=eicab_mask_preference,
        use_postprocessed_mask=use_postprocessed_mask,
        taubin=taubin,
        lesion_detection_method=lesion_detection_method,
        n_workers=n_workers,
        profile=profile,
        centerline_backend=centerline_backend,
        backend=backend,
    )
    return build_singularity_command(spec, paths)


def submit_subject_sge(
    subject: str,
    *,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
    eicab_mask_preference: str = "wb",
    use_postprocessed_mask: bool = True,
    taubin: bool | None = None,
    lesion_detection_method: str | None = None,
    n_workers: int | None = None,
    profile: str = DEFAULT_PROFILE,
    centerline_backend: str | None = None,
    backend: str = "cpu",
) -> str:
    """Emit or submit one stage-7 SGE job. Returns qsub job id."""
    spec, paths = _subject_sge_spec(
        subject,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        eicab_mask_preference=eicab_mask_preference,
        use_postprocessed_mask=use_postprocessed_mask,
        taubin=taubin,
        lesion_detection_method=lesion_detection_method,
        n_workers=n_workers,
        profile=profile,
        centerline_backend=centerline_backend,
        backend=backend,
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage7-morphometrics")
@backend_click_option()
@click.option("--subject", required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option(
    "--eicab-mask-preference",
    type=click.Choice(["cw", "wb"], case_sensitive=False),
    default="wb",
    show_default=True,
    help="eICAB mask to measure. wb: stage 1's distal-grown *_eICAB_WB_pp (A1/A2, M1/M2, P1/P2 as "
    "separate vessels), else *_CW_pp; cw: *_CW_pp. Without --use-postprocessed-mask, eICAB's own mask.",
)
@click.option(
    "--profile",
    type=click.Choice(sorted(PROFILES), case_sensitive=False),
    default=DEFAULT_PROFILE,
    show_default=True,
    help="Morphometrics parameter profile (nvitk.measure.morphometrics_config).",
)
@click.option(
    "--centerline-backend",
    type=click.Choice(CENTERLINE_BACKENDS, case_sensitive=False),
    default=None,
    help="Override the profile's centerline backend (vmtk / native / raycast — experimental).",
)
@click.option(
    "--use-postprocessed-mask/--no-use-postprocessed-mask",
    default=True,
    show_default=True,
    help="Prefer stage 1's ``*_eICAB_{WB,CW}_pp`` masks when available.",
)
@click.option(
    "--taubin/--no-taubin",
    default=None,
    help="Taubin-smooth the label mask before measuring. Default: off for stage 1's post-processed "
    "masks (already drawn at the local half maximum), on for eICAB's own mask.",
)
@click.option(
    "--lesion-detection-method",
    type=click.Choice(LESION_DETECTION_METHODS, case_sensitive=False),
    default=None,
    help="Override the profile's stenosis / enlargement detection (robust_reference / taper).",
)
@click.option(
    "--n-workers",
    type=click.IntRange(min=1),
    default=None,
    help="Vessels measured in parallel: one process per label/component job (1 = one at a time; "
         "default min(4, CPUs/2)). On SGE, N > 1 requests -pe smp N.",
)
def main(
    backend: str,
    subject: str,
    output_root: Path,
    skip_existing: bool,
    eicab_mask_preference: str,
    use_postprocessed_mask: bool,
    taubin: bool | None,
    lesion_detection_method: str | None,
    n_workers: int | None,
    profile: str,
    centerline_backend: str | None,
) -> None:
    """CLI entry point: run stage-7 TOF morphometrics for a single subject."""
    del backend
    run_subject(
        subject,
        output_root=output_root,
        skip_existing=skip_existing,
        eicab_mask_preference=eicab_mask_preference,
        use_postprocessed_mask=use_postprocessed_mask,
        taubin=taubin,
        lesion_detection_method=lesion_detection_method,
        n_workers=n_workers,
        profile=profile,
        centerline_backend=centerline_backend,
    )


if __name__ == "__main__":
    main()
