"""qvtpy stage 1: eICAB Circle-of-Willis / TOF segmentation on stage-0 NIfTI layout.

**Inputs**

- ``<nifti_root>/<subject>/TOF/TOF.nii.gz`` from :mod:`nvitk.pipes.qvtpy.stage0_convert`.

**Outputs**

- eICAB NIfTIs under ``<output_root>/<subject>/<eicab-subdir>/`` (default
  :data:`~nvitk.pipes.qvtpy.config.STAGE1_EICAB_DIR`), including ``TOF_resampled`` for stage 2.

**Execution**

- ``--submit local`` — :func:`nvitk.segmentation.eicab.runner.run_eicab` via Singularity.
- ``--submit sge`` — host ``singularity run`` eICAB (+ optional nvitk post-steps) per subject.
- ``--post-process-eicab`` (default on) — Otsu ICA resegment + ICA RG; writes ``*_pp`` mask and ``centerlines_mask_pp.nii.gz`` (originals unchanged).
- ``--only-pp`` — skip eICAB inference; run ICA post-process on existing outputs only.
"""

from __future__ import annotations

import getpass
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
    write_script_header,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.segmentation.eicab import config as eicab_cfg
from nvitk.segmentation.eicab.cluster import submit_eicab_job
from nvitk.segmentation.eicab.runner import run_eicab

from . import config as cfg
from .util.eicab_masks import find_tof_resampled_volume, resolve_eicab_mask
from .util.eicab_postprocess import postprocess_eicab_directory
from .util.sge_backend import sge_backend_cli_args, sge_stage_extra_env

log = Logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


# ---------------------------------------------------------------------------
# Path helpers + TOF discovery
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    """Host directory mounted at ``/nvitk/src/`` (contains a ``nvitk/`` package tree)."""
    return Path(nvitk.__file__).resolve().parent.parent


def _default_emit_script(label: str) -> Path:
    stem = _SAFE.sub("_", label)[:60] or "batch"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eicab_cfg.DEFAULT_SGE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return eicab_cfg.DEFAULT_SGE_SCRIPTS_DIR / f"submit_qvtpy_eicab_{stem}_{ts}.sh"


def _iter_subjects_nifti(nifti_root: Path) -> list[str]:
    """Subject ids: one folder per subject under *nifti_root*."""
    if not nifti_root.exists():
        return []
    return sorted(p.name for p in nifti_root.iterdir() if p.is_dir())


def find_tof_volume(subject_nifti_dir: Path) -> Path | None:
    """Return path to TOF NIfTI after stage0 reorganization, or None."""
    tof_dir = subject_nifti_dir / "TOF"
    if not tof_dir.is_dir():
        return None
    for name in ("TOF.nii.gz", "TOF.nii"):
        p = tof_dir / name
        if p.is_file():
            return p
    candidates = sorted(
        p
        for p in tof_dir.iterdir()
        if p.is_file() and (p.suffix == ".nii" or p.name.endswith(".nii.gz"))
    )
    return candidates[0] if candidates else None


def _output_has_segmentation(out_dir: Path) -> bool:
    if not out_dir.is_dir():
        return False
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.endswith(".nii.gz") or p.suffix == ".nii":
            return True
    return False


def _eicab_out_dir(
    output_root: Path,
    subject: str,
    eicab_subdir: str | None,
) -> Path:
    subdir = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    return output_root / subject / subdir


def _resolve_pipeline_container(pipeline_container: Path | None) -> Path:
    """Outer SGE image (Python 3 + singularity client); inner eICAB stays in eicab .sif."""
    if pipeline_container is not None:
        return Path(pipeline_container)
    return Path(cfg.CONTAINER_PATH)


def _require_eicab_for_postprocess(out_dir: Path) -> None:
    """Raise if *out_dir* lacks eICAB multilabel + TOF_resampled for post-process."""
    if not _output_has_segmentation(out_dir):
        raise FileNotFoundError(
            f"No eICAB segmentation NIfTI under {out_dir}. "
            "Run stage1 without --only-pp first."
        )
    if find_tof_resampled_volume(out_dir) is None:
        raise FileNotFoundError(
            f"No TOF_resampled under {out_dir} (required for ICA post-process)."
        )
    resolve_eicab_mask(out_dir, preference="cw")


def run_postprocess_only(
    subject: str,
    *,
    output_root: Path,
    eicab_subdir: str | None = None,
) -> Path:
    """ICA Otsu resegment + region growing on existing eICAB outputs."""
    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)
    _require_eicab_for_postprocess(out_dir)
    log.info(f"qvtpy stage1 eICAB post-process only | subject={subject}")
    log.info(f"  output: {out_dir}")
    tof_resampled = find_tof_resampled_volume(out_dir)
    postprocess_eicab_directory(out_dir, tof_path=tof_resampled)
    return out_dir


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    skip_existing: bool = False,
    resolution: float = 0.5,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    eicab_container: Path | None = None,
    tmp_dir: Path | None = None,
    vasculature_dir: Path | None = None,
    keep_aux_outputs: bool = False,
    post_process_eicab: bool = True,
    only_pp: bool = False,
) -> Path:
    """Run eICAB locally for one subject. Returns the eICAB output directory."""
    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)

    if only_pp:
        return run_postprocess_only(
            subject,
            output_root=output_root,
            eicab_subdir=eicab_subdir,
        )

    subj_nifti = nifti_root / subject
    if not subj_nifti.is_dir():
        raise FileNotFoundError(f"NIfTI subject dir not found: {subj_nifti}")

    tof = find_tof_volume(subj_nifti)
    if tof is None:
        raise FileNotFoundError(
            f"No TOF NIfTI under {subj_nifti / 'TOF'} (expected TOF/TOF.nii.gz from stage0)."
        )

    tmp = Path(tmp_dir) if tmp_dir is not None else (out_dir / ".eicab_tmp")
    container = Path(eicab_container) if eicab_container is not None else eicab_cfg.CONTAINER_PATH
    vas_host = (
        Path(vasculature_dir).expanduser()
        if vasculature_dir is not None
        else Path(eicab_cfg.DEFAULT_VASCULATURE_HOST_DIR).expanduser()
    )

    if skip_existing and _output_has_segmentation(out_dir):
        log.info(f"[{subject}] stage1 eICAB: skipping existing output -> {out_dir}")
        return out_dir

    log.info(f"qvtpy stage1 eICAB (local) | subject={subject}")
    log.info(f"  input : {tof}")
    log.info(f"  output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_eicab(
        tof,
        out_dir,
        resolution=resolution,
        simple_segmentation=simple_segmentation,
        attention=attention,
        device=device,
        container=container,
        tmp_dir=tmp,
        keep_aux_outputs=keep_aux_outputs,
        vasculature_host_path=vas_host,
        capture_output=False,
    )
    if post_process_eicab:
        log.step(f"[{subject}] eICAB ICA post-process (Otsu + RG → *_pp + centerlines_mask_pp)")
        tof_resampled = find_tof_resampled_volume(out_dir)
        postprocess_eicab_directory(
            out_dir,
            tof_path=tof_resampled,
        )
    return out_dir


# ---------------------------------------------------------------------------
# SGE submission
# ---------------------------------------------------------------------------


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    skip_existing: bool = False,
    resolution: float = 0.5,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    eicab_container: Path | None = None,
    pipeline_container: Path | None = None,
    src_dir: Path | None = None,
    tmp_dir: Path | None = None,
    vasculature_dir: Path | None = None,
    log_dir: Path | None = None,
    err_dir: Path | None = None,
    keep_aux_outputs: bool = False,
    post_process_eicab: bool = True,
    only_pp: bool = False,
    resources: SgeResources | None = None,
    hold_jid: str | None = None,
    backend: str = "gpu",
    dry_run: bool = False,
    emit: TextIO | None = None,
    job_name: str | None = None,
) -> str:
    """Submit one stage1 eICAB SGE job. Returns the qsub job id (or '' when *emit* is set).

    The local TOF file does not need to exist yet: when a stage0_c job is
    chained via ``hold_jid``, the TOF is produced on the cluster before this
    job starts. We discover the actual file when present, otherwise fall back
    to the canonical stage0_c output path ``{nifti_root}/{subject}/TOF/TOF.nii.gz``.
    """
    subj_nifti = nifti_root / subject
    tof = find_tof_volume(subj_nifti) if subj_nifti.is_dir() else None
    if tof is None:
        tof = subj_nifti / "TOF" / "TOF.nii.gz"
        log.info(
            f"[{subject}] stage1 eICAB: input TOF not present yet; "
            f"emitting predicted path {tof} (produced by stage0_c at run time)."
        )

    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if only_pp:
        return _submit_postprocess_only_sge(
            subject,
            nifti_root=nifti_root,
            output_root=output_root,
            eicab_subdir=eicab_subdir,
            pipeline_container=pipeline_container,
            src_dir=src_dir,
            hold_jid=hold_jid,
            backend=backend,
            dry_run=dry_run,
            emit=emit,
            job_name=job_name,
        )

    if skip_existing and _output_has_segmentation(out_dir):
        log.info(f"[{subject}] stage1 eICAB: skipping existing output -> {out_dir}")
        return ""

    tmp = Path(tmp_dir) if tmp_dir is not None else (out_dir / ".eicab_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    eicab_c = Path(eicab_container) if eicab_container is not None else eicab_cfg.CONTAINER_PATH
    pipeline_c = _resolve_pipeline_container(pipeline_container)
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    vas_host = (
        Path(vasculature_dir).expanduser()
        if vasculature_dir is not None
        else Path(eicab_cfg.DEFAULT_VASCULATURE_HOST_DIR).expanduser()
    )
    ld = Path(log_dir) if log_dir is not None else eicab_cfg.SGE_LOG_DIR
    ed = Path(err_dir) if err_dir is not None else eicab_cfg.SGE_ERR_DIR

    res = resources or SgeResources(
        project=eicab_cfg.SGE_PROJECT,
        account=eicab_cfg.SGE_ACCOUNT,
        ngpu=eicab_cfg.SGE_NGPU,
        h_vmem=eicab_cfg.SGE_H_VMEM,
        queue=eicab_cfg.SGE_QUEUE,
    )
    jn = job_name or f"{cfg.SGE_JOB_PREFIX}_stage1_eicab_{_SAFE.sub('_', subject)[:40]}"

    log.info(f"qvtpy stage1 eICAB (sge) | subject={subject}")
    log.info(f"  input : {tof}")
    log.info(f"  output: {out_dir}")
    log.info(f"  outer container (run_job): {pipeline_c}")
    log.info(f"  inner container (eICAB):   {eicab_c}")

    return submit_eicab_job(
        job_name=jn,
        input_nifti=tof.resolve(),
        output_dir=out_dir.resolve(),
        tmp_dir=tmp.resolve(),
        eicab_container=eicab_c.resolve(),
        src_dir=src_p.resolve(),
        pipeline_container=pipeline_c.resolve(),
        input_root=nifti_root.resolve(),
        output_root=output_root.resolve(),
        vasculature_host=vas_host,
        log_dir=ld,
        err_dir=ed,
        resolution=resolution,
        device=device,
        simple_segmentation=simple_segmentation,
        attention=attention,
        keep_aux_outputs=keep_aux_outputs,
        post_process_eicab=post_process_eicab,
        backend=backend,
        resources=res,
        hold_jid=hold_jid,
        dry_run=dry_run,
        emit=emit,
    )


def _submit_postprocess_only_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None,
    pipeline_container: Path | None,
    src_dir: Path | None,
    hold_jid: str | None,
    backend: str,
    dry_run: bool,
    emit: TextIO | None,
    job_name: str | None,
) -> str:
    """SGE job: ``python -m nvitk.pipes.qvtpy.stage1_eicab --only-pp`` inside pipeline container."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    pipeline_c = (
        Path(pipeline_container)
        if pipeline_container is not None
        else eicab_cfg.PIPELINE_CONTAINER_PATH
    )
    binds = SingularityBinds()
    parts: list[str] = [
        *python_module_argv("nvitk.pipes.qvtpy.stage1_eicab"),
        *sge_backend_cli_args(backend),
        "--only-pp",
        "--submit",
        "local",
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
    ]
    if eicab_subdir:
        parts.extend(["--eicab-subdir", shlex.quote(eicab_subdir.strip())])
    python_cmd = " ".join(parts)

    paths = ClusterPaths(
        src=src_p,
        container=pipeline_c,
        models=None,
        data_root=nifti_root,
        output_root=output_root,
        log_dir=eicab_cfg.SGE_LOG_DIR,
        err_dir=eicab_cfg.SGE_ERR_DIR,
    )
    jn = job_name or f"{cfg.SGE_JOB_PREFIX}_stage1_eicab_pp_{_SAFE.sub('_', subject)[:36]}"
    spec = StageSpec(
        job_name=jn,
        python_cmd=python_cmd,
        resources=SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=0,
            h_vmem=cfg.SGE_H_VMEM,
            queue=cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    log.info(f"qvtpy stage1 eICAB post-process only (sge) | subject={subject}")
    log.info(f"  output: {_eicab_out_dir(output_root, subject, eicab_subdir)}")
    return submit_stage(spec, paths, hold_jid=hold_jid, dry_run=dry_run, emit=emit)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("qvtpy-stage1-eicab")
@backend_click_option()
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_NIFTI_ROOT,
    show_default=True,
    help="Root of per-subject NIfTI trees (stage0 output).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=cfg.DEFAULT_RESULTS_ROOT,
    show_default=True,
    help="Parent directory for per-subject eICAB outputs.",
)
@click.option(
    "--eicab-subdir",
    type=str,
    default=None,
    help=f"Subdirectory under output-root/<subject>/ (default: {cfg.STAGE1_EICAB_DIR!r}).",
)
@click.option(
    "--subject",
    default=None,
    help="Single subject id. If omitted, all subfolders of --nifti-root are processed.",
)
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Run locally or submit per-subject SGE jobs (canonical eICAB pipeline-container wrapper).",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--resolution", type=float, default=0.5, show_default=True)
@click.option("--simple-segmentation", is_flag=True, default=False)
@click.option("--attention", is_flag=True, default=False)
@click.option(
    "--device",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="cpu",
    show_default=True,
)
@click.option(
    "--container",
    "eicab_container",
    type=click.Path(path_type=Path),
    default=None,
    help="eICAB Singularity image (default: nvitk.segmentation.eicab.config.CONTAINER_PATH).",
)
@click.option(
    "--pipeline-container",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Outer pipeline Singularity image with Python 3 (default: qvtpy CONTAINER_PATH).",
)
@click.option(
    "--src-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Host tree mounted at /nvitk/src/ (default: parent of the installed nvitk package).",
)
@click.option(
    "--tmp-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Temp directory for eICAB (default: <output>/<subject>/<eicab-subdir>/.eicab_tmp).",
)
@click.option(
    "--vasculature-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Host tree bind-mounted to /programs/Neuro/vasculature2 (eICAB requirement).",
)
@click.option("--log-dir", type=click.Path(path_type=Path), default=None, help="(sge) qsub stdout logs.")
@click.option("--err-dir", type=click.Path(path_type=Path), default=None, help="(sge) qsub stderr logs.")
@click.option(
    "--keep-aux-outputs",
    is_flag=True,
    default=False,
    help="Keep auxiliary eICAB intermediates (default: prune to CoW/WB NIfTIs).",
)
@click.option(
    "--post-process-eicab/--no-post-process-eicab",
    default=True,
    show_default=True,
    help="After eICAB: Otsu ICA resegment + ICA RG; writes *_pp mask and centerlines_mask_pp.nii.gz.",
)
@click.option(
    "--only-pp",
    is_flag=True,
    default=False,
    help="Skip eICAB inference; run ICA post-process on existing outputs in output-root/<subject>/eicab/.",
)
@click.option(
    "--emit-script",
    type=click.Path(path_type=Path),
    default=None,
    help="(sge) Bash submission script path (default: under eicab.config.DEFAULT_SGE_SCRIPTS_DIR).",
)
@click.option(
    "--no-remote",
    is_flag=True,
    help="(sge) After writing the submission script, do not run it via SSH.",
)
@click.option("--remote-host", default=None, help="(sge) SSH hostname or alias from CLUSTER_HOST_ALIASES.")
@click.option("--remote-user", default=None, help="(sge) SSH username (else prompt).")
@click.option("--dry-run", is_flag=True, default=False, help="(sge) Write the script but do not SSH-execute it.")
def main(
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None,
    subject: str | None,
    submit: str,
    skip_existing: bool,
    resolution: float,
    simple_segmentation: bool,
    attention: bool,
    device: str,
    eicab_container: Path | None,
    pipeline_container: Path | None,
    src_dir: Path | None,
    tmp_dir: Path | None,
    vasculature_dir: Path | None,
    log_dir: Path | None,
    err_dir: Path | None,
    keep_aux_outputs: bool,
    post_process_eicab: bool,
    only_pp: bool,
    emit_script: Path | None,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
    dry_run: bool,
) -> None:
    Logger()

    subjects = [subject] if subject else _iter_subjects_nifti(nifti_root)
    if not subjects:
        raise click.ClickException(
            f"No subject folders found under nifti_root={nifti_root}. "
            "Pass --subject or point --nifti-root at a directory of subject subfolders."
        )

    mode = submit.lower()
    if mode == "local":
        for subj in subjects:
            try:
                run_subject(
                    subj,
                    nifti_root=nifti_root,
                    output_root=output_root,
                    eicab_subdir=eicab_subdir,
                    skip_existing=skip_existing,
                    resolution=resolution,
                    simple_segmentation=simple_segmentation,
                    attention=attention,
                    device=device,
                    eicab_container=eicab_container,
                    tmp_dir=tmp_dir,
                    vasculature_dir=vasculature_dir,
                    keep_aux_outputs=keep_aux_outputs,
                    post_process_eicab=post_process_eicab,
                    only_pp=only_pp,
                )
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB skipped: {exc}")
        return

    # SGE: always emit a script, then SSH-execute (unless --no-remote / --dry-run).
    label = subjects[0] if len(subjects) == 1 else f"batch_{len(subjects)}"
    script_path = Path(emit_script) if emit_script is not None else _default_emit_script(label)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, "w", encoding="utf-8") as fh:
        write_script_header(
            fh,
            log_dir=Path(log_dir) if log_dir is not None else eicab_cfg.SGE_LOG_DIR,
            err_dir=Path(err_dir) if err_dir is not None else eicab_cfg.SGE_ERR_DIR,
            title=f"qvtpy stage1 eICAB (n={len(subjects)})",
        )
        for subj in subjects:
            try:
                submit_subject_sge(
                    subj,
                    nifti_root=nifti_root,
                    output_root=output_root,
                    eicab_subdir=eicab_subdir,
                    skip_existing=skip_existing,
                    resolution=resolution,
                    simple_segmentation=simple_segmentation,
                    attention=attention,
                    device=device,
                    eicab_container=eicab_container,
                    pipeline_container=pipeline_container,
                    src_dir=src_dir,
                    tmp_dir=tmp_dir,
                    vasculature_dir=vasculature_dir,
                    log_dir=log_dir,
                    err_dir=err_dir,
                    keep_aux_outputs=keep_aux_outputs,
                    post_process_eicab=post_process_eicab,
                    only_pp=only_pp,
                    backend=backend,
                    dry_run=False,
                    emit=fh,
                )
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB emit skipped: {exc}")

    log.info("=" * 78)
    log.info(f"qvtpy stage1 SGE script written: {script_path}")
    log.info(f"On the cluster login node: bash {script_path}")
    log.info("=" * 78)

    if dry_run:
        log.info("Dry-run: script written; skipping SSH execution.")
        return

    if no_remote:
        log.info("Skipping remote SSH (--no-remote).")
        return

    log.reset(restart_progress=False)
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host_resolved = eicab_cfg.CLUSTER_HOST_ALIASES.get(host_key, host_key)
    user = remote_user or click.prompt("SSH user")
    password = getpass.getpass("SSH password: ")
    ok = run_sge_script_ssh(host_resolved, user, password, script_path)
    if not ok:
        log.warning(
            f"Remote execution did not complete successfully. Run manually: bash {script_path}"
        )


__all__ = [
    "find_tof_resampled_volume",
    "find_tof_volume",
    "run_postprocess_only",
    "run_subject",
    "submit_subject_sge",
    "main",
]


if __name__ == "__main__":
    main()
