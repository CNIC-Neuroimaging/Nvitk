#!/usr/bin/env python3
"""Compute 4DFlow derived NIfTIs on an existing qvtpy NIfTI tree (stage0_c derivatives only).

Writes ``Angiography_3D``, ``ComplexDifference_3D``, and ``ComplexDifference_4D`` under
``<nifti_root>/<subject>/4DFlow/`` using the same :func:`~nvitk.io.conversors.phase2volume`
logic as :mod:`nvitk.pipes.qvtpy.stage0_convert` (no DICOM conversion).

Examples::

    # All subjects under the local NIfTI root missing any derived volume
    python scripts/pesa_brain/qvtpy_compute_derived.py --skip-existing

    # Explicit cohort on the cluster NIfTI root
    python scripts/pesa_brain/qvtpy_compute_derived.py \\
        --submit sge \\
        --nifti-root /data_lab_MCC/imarcoss/LabMCC/DATA/NIFTI \\
        --subjects PESA5745609,PESA123 \\
        --skip-existing
"""

from __future__ import annotations

import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, TextIO

import click

from nvitk.cluster.remote_submit import run_sge_script_ssh_capture
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_script_argv,
    submit_stage,
    write_script_header,
)
from nvitk.cluster.sge_remote import publish_sge_driver_script, resolve_sge_script_paths
from nvitk.cluster.sge_chunk import parse_sge_submission_job_ids
from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.io._common import default_nifti_axes
from nvitk.io.conversors.phase2volume import (
    compute_phase_derivatives,
    discover_phase_inputs,
    resolve_venc_mm_s,
)
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.stage0_download import load_subjects
from nvitk.pipes.qvtpy.util.cluster_upload import prompt_ssh_credentials
from nvitk.pipes.qvtpy.util.paths import (
    CLUSTER_HOST_ALIASES,
    layout_cluster,
    layout_local,
)
from nvitk.pipes.qvtpy.util.sge_backend import (
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)

log = Logger()

DERIVED_STEMS: tuple[str, ...] = (
    "Angiography_3D",
    "ComplexDifference_3D",
    "ComplexDifference_4D",
)

_SCRIPT_REL = Path("scripts/pesa_brain/qvtpy_compute_derived.py")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_in_container(*, repo_bind: str = "/nvitk/repo") -> str:
    return f"{repo_bind.rstrip('/')}/{_SCRIPT_REL.as_posix()}"


def _derived_file(flow_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.nii.gz", f"{stem}.nii"):
        path = flow_dir / name
        if path.is_file():
            return path
    return None


def derived_complete(flow_dir: Path) -> bool:
    """True when all requested derived stems exist under *flow_dir*."""
    return all(_derived_file(flow_dir, stem) is not None for stem in DERIVED_STEMS)


def missing_derived(flow_dir: Path) -> list[str]:
    return [stem for stem in DERIVED_STEMS if _derived_file(flow_dir, stem) is None]


def _list_subjects_from_nifti_root(nifti_root: Path) -> list[str]:
    if not nifti_root.is_dir():
        raise FileNotFoundError(f"NIfTI root not found: {nifti_root}")
    return sorted(
        p.name
        for p in nifti_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def resolve_subjects(
    *,
    nifti_root: Path,
    subjects: str | None,
    subjects_file: Path | None,
) -> list[str]:
    if subjects is not None or subjects_file is not None:
        return load_subjects(subjects=subjects, subjects_file=subjects_file)
    return _list_subjects_from_nifti_root(nifti_root)


def filter_subjects_needing_derivatives(
    subjects: Iterable[str],
    *,
    nifti_root: Path,
    skip_existing: bool,
) -> tuple[list[str], list[str]]:
    """Return (to_run, skipped) subject ids."""
    to_run: list[str] = []
    skipped: list[str] = []
    for subject in subjects:
        flow_dir = nifti_root / subject / "4DFlow"
        if skip_existing and derived_complete(flow_dir):
            skipped.append(subject)
        else:
            to_run.append(subject)
    return to_run, skipped


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    dicom_root: Path | None = None,
    skip_existing: bool = False,
    phase_background_correction: bool = True,
    phase_bg_poly_order: int = 2,
    phase_bg_static_percentile: float = 25.0,
    cd_4d_background_correction: bool | None = None,
    backend: str = "gpu",
) -> list[Path]:
    """Compute derived 4DFlow volumes for one subject."""
    subj_dir = nifti_root / subject
    flow_dir = subj_dir / "4DFlow"
    if not flow_dir.is_dir():
        raise FileNotFoundError(f"Missing 4DFlow directory: {flow_dir}")

    if skip_existing and derived_complete(flow_dir):
        log.info(f"[{subject}] skip-existing: derived volumes present")
        return []

    missing = missing_derived(flow_dir)
    if missing:
        log.info(f"[{subject}] computing missing derived: {', '.join(missing)}")

    inputs = discover_phase_inputs(subj_dir)
    angio_image = imread(inputs.angio_path)
    ap_image = imread(inputs.ap_phase_path)
    rl_image = imread(inputs.rl_phase_path)
    fh_image = imread(inputs.fh_phase_path)

    dicom_search_dir = (dicom_root / subject) if dicom_root is not None else None
    actual_venc, venc_src = resolve_venc_mm_s(
        ap_dir=inputs.ap_dir,
        default_mm_s=700.0,
        ap_phase_metadata=dict(ap_image.metadata or {}),
        magnitude_metadata=dict(angio_image.metadata or {}),
        dicom_search_dir=dicom_search_dir,
        log_context=subject,
    )
    if venc_src != "default":
        log.info(f"[{subject}] VENC={actual_venc:.1f} mm/s from {venc_src}")

    outputs = compute_phase_derivatives(
        angio_image.data,
        ap_image.data,
        rl_image.data,
        fh_image.data,
        venc=actual_venc,
        background_phase_correction=phase_background_correction,
        bg_poly_order=phase_bg_poly_order,
        bg_static_percentile=phase_bg_static_percentile,
        cd_4d_background_correction=cd_4d_background_correction,
    )

    base_meta = dict(angio_image.metadata or {})
    written: list[Path] = []
    for name in DERIVED_STEMS:
        array = outputs.get(name)
        if array is None:
            log.warning(f"[{subject}] phase2volume returned no array for {name}")
            continue
        output_path = flow_dir / f"{name}.nii.gz"
        output_metadata = dict(base_meta)
        output_metadata["axes"] = default_nifti_axes(array.ndim)
        output_metadata["shape"] = tuple(array.shape)
        if array.ndim < 4:
            output_metadata.pop("t_res", None)
            output_metadata.pop("temporal_resolution", None)
        imsave(output_path, array, metadata=output_metadata)
        written.append(output_path)
        log.info(f"[{subject}] wrote {output_path.name}")

    return written


def _emit_subject_sge(
    fh: TextIO,
    subject: str,
    *,
    nifti_root: Path,
    repo_root: Path,
    src_dir: Path,
    container: Path,
    dicom_root: Path | None,
    skip_existing: bool,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    no_cd_4d_background_correction: bool,
    backend: str,
) -> str:
    binds = SingularityBinds()
    repo_bind = "/nvitk/repo"
    cmd_parts: list[str] = [
        *python_script_argv(_script_in_container(repo_bind=repo_bind)),
        "--submit",
        "local",
        "--backend",
        shlex.quote(backend),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
    ]
    if skip_existing:
        cmd_parts.append("--skip-existing")
    if phase_background_correction:
        cmd_parts.append("--phase-background-correction")
    cmd_parts.extend(
        [
            "--phase-bg-poly-order",
            str(int(phase_bg_poly_order)),
            "--phase-bg-static-percentile",
            str(float(phase_bg_static_percentile)),
        ]
    )
    if no_cd_4d_background_correction:
        cmd_parts.append("--no-cd-4d-background-correction")
    if dicom_root is not None:
        cmd_parts.extend(["--dicom-root", "/nvitk/dicom"])

    extra_binds: list[tuple[Path, str]] = [(repo_root.resolve(), repo_bind)]
    if dicom_root is not None:
        extra_binds.append((dicom_root.resolve(), "/nvitk/dicom"))

    paths = ClusterPaths(
        src=src_dir.resolve(),
        container=container.resolve(),
        models=None,
        data_root=nifti_root.resolve(),
        output_root=nifti_root.resolve(),
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_derived_{subject}",
        python_cmd=" ".join(cmd_parts),
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
        extra_host_binds=tuple(extra_binds),
    )
    return submit_stage(spec, paths, emit=fh)


def submit_subjects_sge(
    subjects: list[str],
    *,
    nifti_root: Path,
    repo_root: Path,
    src_dir: Path,
    container: Path,
    dicom_root: Path | None,
    skip_existing: bool,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    no_cd_4d_background_correction: bool,
    backend: str,
    ssh_host: str | None,
    ssh_user: str | None,
) -> tuple[int, list[str]]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_basename = f"submit_qvtpy_derived_{ts}.sh"
    local_script_path, remote_script_path = resolve_sge_script_paths(
        None,
        remote_scripts_dir=cfg.SGE_SCRIPTS_DIR,
        default_basename=script_basename,
    )

    host, user, password = prompt_ssh_credentials(
        remote_host=ssh_host,
        remote_user=ssh_user,
        host_aliases=CLUSTER_HOST_ALIASES,
    )

    with open(local_script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=cfg.SGE_LOG_DIR,
            err_dir=cfg.SGE_ERR_DIR,
            title=f"qvtpy derived volumes ({len(subjects)} subject(s))",
        )
        for subject in subjects:
            _emit_subject_sge(
                fh,
                subject,
                nifti_root=nifti_root,
                repo_root=repo_root,
                src_dir=src_dir,
                container=container,
                dicom_root=dicom_root,
                skip_existing=skip_existing,
                phase_background_correction=phase_background_correction,
                phase_bg_poly_order=phase_bg_poly_order,
                phase_bg_static_percentile=phase_bg_static_percentile,
                no_cd_4d_background_correction=no_cd_4d_background_correction,
                backend=backend,
            )

    log.info(f"  local script : {local_script_path}")
    log.info(f"  cluster path : {remote_script_path}")

    cluster_exec_path = publish_sge_driver_script(
        local_script_path,
        remote_script_path,
        host=host,
        user=user,
        password=password,
    )
    exit_code, stdout, stderr = run_sge_script_ssh_capture(
        host,
        user,
        password,
        cluster_exec_path,
        local_script_path=local_script_path,
    )
    if stderr.strip():
        log.info(stderr.strip())
    job_ids = parse_sge_submission_job_ids(stdout, stderr)
    return exit_code, job_ids


@click.command()
@backend_click_option()
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Subject NIfTI tree (default: local or cluster layout from config).",
)
@click.option(
    "--dicom-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional DICOM root for VENC lookup (``<dicom-root>/<subject>/``).",
)
@click.option(
    "--subjects",
    default=None,
    help="Comma/space separated subject IDs. If omitted, all folders under --nifti-root.",
)
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Text/CSV/XLSX file with subject IDs.",
)
@click.option(
    "--subject",
    default=None,
    help="Single subject (used by SGE workers; do not combine with --subjects).",
)
@click.option("--skip-existing", is_flag=True, default=False, show_default=True)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"]),
    default="local",
    show_default=True,
    help="Run locally or submit one SGE job per subject.",
)
@click.option(
    "--container",
    type=click.Path(path_type=Path),
    default=cfg.CONTAINER_PATH,
    show_default=True,
    help="Singularity image for --submit sge.",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host path mounted at /nvitk/src/ (default: <repo>/src or qvtpy NVITK_SRC_DIR).",
)
@click.option(
    "--ssh-host",
    default=None,
    help="Cluster login host for --submit sge (prompted if omitted).",
)
@click.option(
    "--ssh-user",
    default=None,
    help="SSH username for --submit sge (prompted if omitted).",
)
@click.option(
    "--phase-background-correction/--no-phase-background-correction",
    "phase_background_correction",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option("--phase-bg-poly-order", type=int, default=2, show_default=True)
@click.option("--phase-bg-static-percentile", type=float, default=25.0, show_default=True)
@click.option(
    "--no-cd-4d-background-correction",
    is_flag=True,
    default=False,
    help="Disable per-frame polynomial background on ComplexDifference_4D.",
)
def main(
    nifti_root: Path | None,
    dicom_root: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    subject: str | None,
    skip_existing: bool,
    submit: str,
    container: Path,
    src_dir: Path | None,
    ssh_host: str | None,
    ssh_user: str | None,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    no_cd_4d_background_correction: bool,
    backend: str,
) -> None:
    if subjects_file is not None and subjects is not None:
        raise click.ClickException("Provide only one of --subjects or --subjects-file.")
    if subject is not None and (subjects is not None or subjects_file is not None):
        raise click.ClickException("Use either --subject or --subjects/--subjects-file.")

    repo_root = _repo_root()
    src_p = Path(src_dir).expanduser().resolve() if src_dir is not None else (repo_root / "src")
    cd_4d_bpc = False if no_cd_4d_background_correction else None

    if submit == "sge":
        paths = layout_cluster(
            nifti_root=nifti_root,
            dicom_root=dicom_root if dicom_root is not None else None,
        )
    else:
        paths = layout_local(
            nifti_root=nifti_root,
            dicom_root=dicom_root if dicom_root is not None else None,
        )

    nifti_root_eff = paths.nifti_root.expanduser().resolve()
    dicom_root_eff = (
        paths.dicom_root.expanduser().resolve() if dicom_root is not None else None
    )

    worker_kwargs = dict(
        nifti_root=nifti_root_eff,
        dicom_root=dicom_root_eff,
        skip_existing=skip_existing,
        phase_background_correction=phase_background_correction,
        phase_bg_poly_order=phase_bg_poly_order,
        phase_bg_static_percentile=phase_bg_static_percentile,
        cd_4d_background_correction=cd_4d_bpc,
        backend=backend,
    )

    if subject is not None:
        run_subject(subject, **worker_kwargs)
        return

    subject_list = resolve_subjects(
        nifti_root=nifti_root_eff,
        subjects=subjects,
        subjects_file=subjects_file,
    )
    to_run, skipped = filter_subjects_needing_derivatives(
        subject_list,
        nifti_root=nifti_root_eff,
        skip_existing=skip_existing,
    )

    log.info(f"qvtpy derived | nifti={nifti_root_eff} | submit={submit}")
    log.info(
        f"  subjects: {len(subject_list)} total, {len(to_run)} to run, "
        f"{len(skipped)} skipped (skip-existing={skip_existing})"
    )

    if not to_run:
        click.echo("Nothing to do.")
        return

    if submit == "local":
        ok = 0
        failed: list[str] = []
        for subj in to_run:
            try:
                run_subject(subj, **worker_kwargs)
                ok += 1
            except Exception as exc:
                log.exception(f"[{subj}] derived computation failed: {exc}")
                failed.append(subj)
        click.echo(
            f"Done: {ok}/{len(to_run)} subject(s) processed"
            + (f", {len(failed)} failed" if failed else "")
        )
        if failed:
            raise SystemExit(1)
        return

    exit_code, job_ids = submit_subjects_sge(
        to_run,
        nifti_root=nifti_root_eff,
        repo_root=repo_root,
        src_dir=src_p,
        container=container,
        dicom_root=dicom_root_eff,
        skip_existing=skip_existing,
        phase_background_correction=phase_background_correction,
        phase_bg_poly_order=phase_bg_poly_order,
        phase_bg_static_percentile=phase_bg_static_percentile,
        no_cd_4d_background_correction=no_cd_4d_background_correction,
        backend=backend,
        ssh_host=ssh_host,
        ssh_user=ssh_user,
    )
    if exit_code != 0:
        raise click.ClickException(f"SGE driver failed (exit {exit_code}).")
    click.echo(f"Submitted {len(job_ids)} job(s): {', '.join(job_ids) or '(none parsed)'}")


if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except SystemExit as exc:
        raise SystemExit(exc.code) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
