"""qvtpy stage 7: TOF eICAB morphometrics (cow_morpho port).

**Inputs**

- eICAB WB (or CW fallback) multilabel mask under ``<output_root>/<subject>/eicab/``.

**Outputs**

- ``case_metrics_donut_tree.xlsx``, centerline/surface VTPs, tortuosity tables, and
  radius histograms under ``<output_root>/<subject>/qvtpy/stage7_morphometrics/``.
"""

from __future__ import annotations

import shlex
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
from nvitk.measure.morphometrics_config import MorphometricsConfig
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
    return Path(nvitk.__file__).resolve().parent.parent


def run_subject(
    subject: str,
    *,
    output_root: Path,
    skip_existing: bool = False,
    eicab_mask_preference: str = "wb",
    use_postprocessed_mask: bool = True,
    input_already_smoothed: bool = False,
    n_workers: int | None = None,
    morpho_config: MorphometricsConfig | None = None,
) -> Path:
    """Run stage-7 TOF morphometrics for one subject."""
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
    run_morphometrics_case(
        mask_res.path,
        out_dir,
        case_out_dir_override=out_dir,
        n_workers=n_workers,
        config=morpho_config,
        input_already_smoothed=input_already_smoothed,
        skip_if_excel_exists=skip_existing,
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
    input_already_smoothed: bool = False,
    n_workers: int | None = None,
    backend: str = "cpu",
) -> tuple[StageSpec, ClusterPaths]:
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
    if input_already_smoothed:
        parts.append("--input-already-smoothed")
    if skip_existing:
        parts.append("--skip-existing")
    if n_workers is not None:
        parts.extend(["--n-workers", str(int(n_workers))])
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
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage7_{subject}",
        python_cmd=python_cmd,
        resources=sge_qvtpy_stage_resources(backend),
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
    input_already_smoothed: bool = False,
    n_workers: int | None = None,
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
        input_already_smoothed=input_already_smoothed,
        n_workers=n_workers,
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
    input_already_smoothed: bool = False,
    n_workers: int | None = None,
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
        input_already_smoothed=input_already_smoothed,
        n_workers=n_workers,
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
)
@click.option(
    "--use-postprocessed-mask/--no-use-postprocessed-mask",
    default=True,
    show_default=True,
    help="Prefer ``*_eICAB_CW_pp.nii.gz`` when available.",
)
@click.option(
    "--input-already-smoothed",
    is_flag=True,
    default=False,
    help="Skip Taubin smoothing (input is already smoothed).",
)
@click.option("--n-workers", type=int, default=None, help="Parallel label/component workers.")
def main(
    backend: str,
    subject: str,
    output_root: Path,
    skip_existing: bool,
    eicab_mask_preference: str,
    use_postprocessed_mask: bool,
    input_already_smoothed: bool,
    n_workers: int | None,
) -> None:
    del backend
    run_subject(
        subject,
        output_root=output_root,
        skip_existing=skip_existing,
        eicab_mask_preference=eicab_mask_preference,
        use_postprocessed_mask=use_postprocessed_mask,
        input_already_smoothed=input_already_smoothed,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
