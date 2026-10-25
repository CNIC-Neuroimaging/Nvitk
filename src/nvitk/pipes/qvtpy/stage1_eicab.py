"""qvtpy stage 1: eICAB Circle-of-Willis / TOF segmentation on stage-0 NIfTI layout.

**Inputs**

- ``<nifti_root>/<subject>/TOF/TOF.nii.gz`` from :mod:`nvitk.pipes.qvtpy.stage0_convert`.

**Outputs**

- eICAB NIfTIs under ``<output_root>/<subject>/<eicab-subdir>/`` (default
  :data:`~nvitk.pipes.qvtpy.config.STAGE1_EICAB_DIR`), including ``TOF_resampled`` for stage 2.

**Execution**

- ``--submit local`` — :func:`nvitk.segmentation.eicab.runner.run_eicab` via Singularity.
- ``--submit sge`` — host ``singularity run`` eICAB (+ optional nvitk post-steps) per subject.
- ``--post-process-eicab`` (default on) — :mod:`~nvitk.pipes.qvtpy.util.eicab.eicab_postprocess`: local
  re-segmentation of every label, extension of the ICAs / basilar below eICAB's box, and (with
  ``--eicab-pp-siphon``, default on) the ICA siphon loop cuts; writes the ``*_pp`` mask,
  ``centerlines_mask_pp.nii.gz`` and ``eicab_postprocess.json`` (originals unchanged).
- ``--only-pp`` — skip eICAB inference; run ICA post-process on existing outputs only.
- ``--reuse-eicab`` — reuse an existing eICAB CW mask (+ ``TOF_resampled``) and run the
  post-process only; without one, fall back to eICAB in CW-only mode (the whole-brain flood
  fill is skipped, see ``--eicab-wb``). The branch taken is logged; on SGE the check runs at
  submit time (post-process-only job) and again inside the eICAB job.
"""

from __future__ import annotations

import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.sge_remote import publish_sge_driver_script, resolve_sge_script_paths
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
from nvitk.segmentation.eicab.cluster import build_eicab_host_shell_cmd, submit_eicab_job
from nvitk.segmentation.eicab.runner import run_eicab

from . import config as cfg
from .util.eicab.eicab_masks import (
    eicab_pp_path,
    find_eicab_cw_mask,
    find_tof_resampled_volume,
    resolve_eicab_mask,
)
from .util.eicab.eicab_postprocess import postprocess_eicab_directory
from .util.io.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)

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
    """True if *out_dir* contains at least one NIfTI file anywhere under it (eICAB has produced output)."""
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
    """eICAB output directory for *subject* under *output_root*, using *eicab_subdir* if given."""
    subdir = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    return output_root / subject / subdir


def find_existing_eicab(out_dir: Path) -> tuple[Path, Path] | None:
    """The eICAB CW mask and ``TOF_resampled`` already in *out_dir* (the post-process inputs),
    or ``None`` when either is missing."""
    cw = find_eicab_cw_mask(out_dir)
    if cw is None:
        return None
    tof_resampled = find_tof_resampled_volume(out_dir)
    if tof_resampled is None:
        return None
    return cw, tof_resampled


def _eicab_whole_brain(eicab_wb: bool | None, reuse_eicab: bool) -> bool:
    """Whether eICAB runs its whole-brain flood fill: *eicab_wb* when given, else on unless
    *reuse_eicab* (whose fallback run only needs the CW mask)."""
    return (not reuse_eicab) if eicab_wb is None else bool(eicab_wb)


def _eicab_scope(whole_brain: bool) -> str:
    """Log wording for the eICAB run scope."""
    return "CW + whole brain" if whole_brain else "CW only, whole-brain flood fill skipped"


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
    pp_siphon: bool = True,
    pp_distal: bool = True,
) -> Path:
    """The eICAB post-process on existing eICAB outputs."""
    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)
    _require_eicab_for_postprocess(out_dir)
    log.info(f"qvtpy stage1 eICAB post-process only | subject={subject}")
    log.info(f"  output: {out_dir}")
    tof_resampled = find_tof_resampled_volume(out_dir)
    postprocess_eicab_directory(out_dir, tof_path=tof_resampled, siphon=pp_siphon, distal=pp_distal)
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
    pp_siphon: bool = True,
    pp_distal: bool = True,
    reuse_eicab: bool = False,
    eicab_wb: bool | None = None,
) -> Path:
    """Run eICAB locally for one subject. Returns the eICAB output directory.

    *reuse_eicab*: when *out_dir* already holds the eICAB CW mask and ``TOF_resampled``, skip
    eICAB and run the post-process only; otherwise fall back to eICAB. *eicab_wb*: run eICAB's
    whole-brain flood fill (default: on, off for the *reuse_eicab* fallback).
    """
    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)
    whole_brain = _eicab_whole_brain(eicab_wb, reuse_eicab)

    if reuse_eicab and not only_pp:
        existing = find_existing_eicab(out_dir)
        if existing is not None:
            cw, tof_resampled = existing
            log.info(
                f"[{subject}] stage1: reusing the existing eICAB CW mask {cw.name} "
                f"(+ {tof_resampled.name}) in {out_dir}: eICAB skipped"
            )
            if not post_process_eicab:
                log.info(f"[{subject}] stage1: post-process disabled: nothing to do")
                return out_dir
            if skip_existing and eicab_pp_path(cw).is_file():
                log.info(f"[{subject}] stage1: skipping the existing post-process -> {eicab_pp_path(cw).name}")
                return out_dir
            log.step(f"[{subject}] eICAB post-process (re-segment + extend + siphon → *_pp + centerlines_mask_pp)")
            postprocess_eicab_directory(out_dir, tof_path=tof_resampled, siphon=pp_siphon, distal=pp_distal)
            return out_dir
        log.info(
            f"[{subject}] stage1: no eICAB CW mask in {out_dir}: "
            f"falling back to eICAB ({_eicab_scope(whole_brain)})"
        )

    if only_pp:
        return run_postprocess_only(
            subject,
            output_root=output_root,
            eicab_subdir=eicab_subdir,
            pp_siphon=pp_siphon,
            pp_distal=pp_distal,
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

    # With --reuse-eicab, reaching this point means no usable CW mask: partial outputs are redone.
    if skip_existing and not reuse_eicab and _output_has_segmentation(out_dir):
        log.info(f"[{subject}] stage1 eICAB: skipping existing output -> {out_dir}")
        return out_dir

    log.info(f"qvtpy stage1 eICAB (local) | subject={subject}")
    log.info(f"  input : {tof}")
    log.info(f"  output: {out_dir}")
    log.info(f"  scope : {_eicab_scope(whole_brain)}")

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
        cw_only=not whole_brain,
    )
    if not whole_brain:
        if find_eicab_cw_mask(out_dir) is None:
            raise FileNotFoundError(f"eICAB (CW only) wrote no *_eICAB_CW mask under {out_dir}")
        log.info(
            f"[{subject}] stage1: eICAB CW only: no *_eICAB_WB mask "
            "(stage 3's venous exclusion falls back to the CW labels)"
        )
    if post_process_eicab:
        log.step(f"[{subject}] eICAB post-process (re-segment + extend + siphon → *_pp + centerlines_mask_pp)")
        tof_resampled = find_tof_resampled_volume(out_dir)
        postprocess_eicab_directory(
            out_dir,
            tof_path=tof_resampled,
            siphon=pp_siphon,
            distal=pp_distal,
        )
    return out_dir


# ---------------------------------------------------------------------------
# SGE submission
# ---------------------------------------------------------------------------


def _resolve_stage1_tof(nifti_root: Path, subject: str) -> Path:
    """Discover TOF when present; else canonical stage0_c path for held SGE jobs."""
    subj_nifti = nifti_root / subject
    tof = find_tof_volume(subj_nifti) if subj_nifti.is_dir() else None
    if tof is None:
        tof = subj_nifti / "TOF" / "TOF.nii.gz"
        log.info(
            f"[{subject}] stage1 eICAB: input TOF not present yet; "
            f"emitting predicted path {tof} (produced by stage0_c at run time)."
        )
    return tof


def _postprocess_only_sge_spec(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    pipeline_container: Path | None = None,
    src_dir: Path | None = None,
    backend: str = "gpu",
    job_name: str | None = None,
    pp_siphon: bool = True,
    pp_distal: bool = True,
) -> tuple[StageSpec, ClusterPaths]:
    """StageSpec/ClusterPaths for ``--only-pp`` inside the pipeline container."""
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
    if not pp_siphon:
        parts.append("--no-eicab-pp-siphon")
    if not pp_distal:
        parts.append("--no-eicab-pp-distal")
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
        resources=sge_qvtpy_stage_resources(backend, request_gpu=False),
        binds=binds,
        use_nv=sge_stage_use_nv(backend, request_gpu=False),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return spec, paths


def build_subject_sge_command(
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
    thread_limit: int | None = None,
    sge_pe_smp: int | None = None,
    local_metric_scratch: bool | None = None,
    vasculature_dir: Path | None = None,
    log_dir: Path | None = None,
    err_dir: Path | None = None,
    keep_aux_outputs: bool = False,
    post_process_eicab: bool = True,
    only_pp: bool = False,
    resources: SgeResources | None = None,
    backend: str = "gpu",
    job_name: str | None = None,
    pp_siphon: bool = True,
    pp_distal: bool = True,
    reuse_eicab: bool = False,
    eicab_wb: bool | None = None,
) -> str:
    """Return the host shell command for one stage1 array/SGE task.

    Normal eICAB path: host ``singularity run`` chain via
    :func:`~nvitk.segmentation.eicab.cluster.build_eicab_host_shell_cmd`.
    ``only_pp``: pipeline-container singularity command (same as
    :func:`_submit_postprocess_only_sge`). *reuse_eicab*: an existing CW mask found now gives
    the post-process-only command; otherwise the eICAB command checks again at run time.
    """
    del skip_existing, resources, log_dir, err_dir  # submit-only / unused for cmd build
    whole_brain = _eicab_whole_brain(eicab_wb, reuse_eicab)
    if reuse_eicab and not only_pp:
        reuse = _reuse_at_submit(subject, _eicab_out_dir(output_root, subject, eicab_subdir), whole_brain)
        if reuse and not post_process_eicab:
            return shlex.join(["echo", f"[nvitk stage1] {subject}: existing eICAB CW mask, post-process disabled"])
        only_pp = reuse
    if only_pp:
        from nvitk.cluster.sge import build_singularity_command

        spec, paths = _postprocess_only_sge_spec(
            subject,
            nifti_root=nifti_root,
            output_root=output_root,
            eicab_subdir=eicab_subdir,
            pipeline_container=pipeline_container,
            src_dir=src_dir,
            backend=backend,
            job_name=job_name,
            pp_siphon=pp_siphon,
            pp_distal=pp_distal,
        )
        return build_singularity_command(spec, paths)

    tof = _resolve_stage1_tof(nifti_root, subject)
    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)
    tmp = Path(tmp_dir) if tmp_dir is not None else (out_dir / ".eicab_tmp")
    eicab_c = Path(eicab_container) if eicab_container is not None else eicab_cfg.CONTAINER_PATH
    pipeline_c = _resolve_pipeline_container(pipeline_container)
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    vas_host = (
        Path(vasculature_dir).expanduser()
        if vasculature_dir is not None
        else Path(eicab_cfg.DEFAULT_VASCULATURE_HOST_DIR).expanduser()
    )
    metric_scratch = (
        eicab_cfg.EICAB_LOCAL_METRIC_SCRATCH
        if local_metric_scratch is None
        else local_metric_scratch
    )
    return build_eicab_host_shell_cmd(
        input_nifti=tof.resolve(),
        output_dir=out_dir.resolve(),
        tmp_dir=tmp.resolve(),
        eicab_container=eicab_c.resolve(),
        src_dir=src_p.resolve(),
        pipeline_container=pipeline_c.resolve(),
        input_root=nifti_root.resolve(),
        output_root=output_root.resolve(),
        vasculature_host=vas_host,
        resolution=resolution,
        device=device,
        simple_segmentation=simple_segmentation,
        attention=attention,
        keep_aux_outputs=keep_aux_outputs,
        post_process_eicab=post_process_eicab,
        pp_siphon=pp_siphon,
        pp_distal=pp_distal,
        backend=backend,
        thread_limit=thread_limit,
        sge_pe_smp=sge_pe_smp,
        local_metric_scratch=metric_scratch,
        whole_brain=whole_brain,
        reuse_existing=reuse_eicab,
    )


def _reuse_at_submit(subject: str, out_dir: Path, whole_brain: bool) -> bool:
    """Submit-time ``--reuse-eicab`` check (logged): True when *out_dir* already holds the CW
    mask and ``TOF_resampled``, so a post-process-only job is enough."""
    existing = find_existing_eicab(out_dir)
    if existing is not None:
        log.info(
            f"[{subject}] stage1 (sge): reusing the existing eICAB CW mask {existing[0].name} "
            f"in {out_dir}: post-process-only job, eICAB skipped"
        )
        return True
    log.info(
        f"[{subject}] stage1 (sge): no eICAB CW mask in {out_dir} yet: eICAB job "
        f"({_eicab_scope(whole_brain)}); the job checks again at run time"
    )
    return False


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
    thread_limit: int | None = None,
    sge_pe_smp: int | None = None,
    local_metric_scratch: bool | None = None,
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
    pp_siphon: bool = True,
    pp_distal: bool = True,
    reuse_eicab: bool = False,
    eicab_wb: bool | None = None,
) -> str:
    """Submit one stage1 eICAB SGE job. Returns the qsub job id (or '' when *emit* is set).

    *reuse_eicab* / *eicab_wb*: see :func:`run_subject`; an existing CW mask found at submit
    time gives a post-process-only job, and the eICAB job checks again at run time.

    The local TOF file does not need to exist yet: when a stage0_c job is
    chained via ``hold_jid``, the TOF is produced on the cluster before this
    job starts. We discover the actual file when present, otherwise fall back
    to the canonical stage0_c output path ``{nifti_root}/{subject}/TOF/TOF.nii.gz``.
    """
    tof = _resolve_stage1_tof(nifti_root, subject)

    out_dir = _eicab_out_dir(output_root, subject, eicab_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    whole_brain = _eicab_whole_brain(eicab_wb, reuse_eicab)

    if reuse_eicab and not only_pp:
        reuse = _reuse_at_submit(subject, out_dir, whole_brain)
        if reuse and not post_process_eicab:
            log.info(f"[{subject}] stage1 (sge): post-process disabled: nothing to submit")
            return ""
        only_pp = reuse

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
            pp_siphon=pp_siphon,
            pp_distal=pp_distal,
        )

    if skip_existing and not reuse_eicab and _output_has_segmentation(out_dir):
        log.info(f"[{subject}] stage1 eICAB: skipping existing output -> {out_dir}")
        return ""

    tmp = Path(tmp_dir) if tmp_dir is not None else (out_dir / ".eicab_tmp")
    if emit is None:
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
    if thread_limit:
        log.info(
            f"  thread limit  : {thread_limit} "
            "(OMP/BLAS + VED via sitecustomize / NVITK_CPU_LIMIT)"
        )
    if sge_pe_smp:
        log.info(f"  sge pe smp    : {sge_pe_smp} (qsub -pe smp {sge_pe_smp})")
    metric_scratch = (
        eicab_cfg.EICAB_LOCAL_METRIC_SCRATCH
        if local_metric_scratch is None
        else local_metric_scratch
    )
    log.info(
        "  metric/cwd : wipe metric_space + --pwd /tmp/ved_cwd "
        "(isolate ComputeVED Scale_* from parallel jobs)"
    )
    if metric_scratch:
        log.info(
            f"  metric scratch: {eicab_cfg.EICAB_METRIC_SCRATCH_ROOT}/nvitk_eicab_$JOB_ID/"
            "{{metric_space,cwd}} (bind over /output/metric_space + /tmp/ved_cwd)"
        )
    log.info(f"  outer container (run_job): {pipeline_c}")
    log.info(f"  inner container (eICAB):   {eicab_c}")
    log.info(f"  scope : {_eicab_scope(whole_brain)}")

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
        pp_siphon=pp_siphon,
        pp_distal=pp_distal,
        backend=backend,
        resources=res,
        thread_limit=thread_limit,
        sge_pe_smp=sge_pe_smp,
        local_metric_scratch=metric_scratch,
        whole_brain=whole_brain,
        reuse_existing=reuse_eicab,
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
    pp_siphon: bool = True,
    pp_distal: bool = True,
) -> str:
    """SGE job: ``python -m nvitk.pipes.qvtpy.stage1_eicab --only-pp`` inside pipeline container."""
    spec, paths = _postprocess_only_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        eicab_subdir=eicab_subdir,
        pipeline_container=pipeline_container,
        src_dir=src_dir,
        backend=backend,
        job_name=job_name,
        pp_siphon=pp_siphon,
        pp_distal=pp_distal,
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
    "--eicab-thread-limit",
    type=int,
    default=None,
    help=(
        "(sge) Cap OMP/BLAS threads inside the eICAB container. "
        "Default: pipelines.eicab.eicab_thread_limit in .nvitk/sge.json."
    ),
)
@click.option(
    "--sge-pe-smp",
    type=int,
    default=None,
    help=(
        "(sge) Optional qsub -pe smp N (cluster-specific parallel environment). "
        "Omit unless your queue supports it; use --eicab-thread-limit instead."
    ),
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
    help="After eICAB: local re-segmentation of every label, extension of the ICAs / basilar below eICAB's box "
         "and the ICA siphon cuts; writes the *_pp mask, centerlines_mask_pp.nii.gz and eicab_postprocess.json.",
)
@click.option(
    "--eicab-pp-siphon/--no-eicab-pp-siphon",
    default=True,
    show_default=True,
    help="Include the ICA siphon correction (minimum cut) in the eICAB post-process.",
)
@click.option(
    "--eicab-pp-distal/--no-eicab-pp-distal",
    default=True,
    show_default=True,
    help="Grow the ACAs / MCAs / PCAs / SCAs distally along the Frangi tree in the eICAB post-process.",
)
@click.option(
    "--only-pp",
    is_flag=True,
    default=False,
    help="Skip eICAB inference; run the eICAB post-process on existing outputs in output-root/<subject>/eicab/.",
)
@click.option(
    "--reuse-eicab/--no-reuse-eicab",
    default=False,
    show_default=True,
    help="Reuse an existing eICAB CW mask (+ TOF_resampled) in output-root/<subject>/<eicab-subdir>/ and run "
         "the post-process only; without one, fall back to eICAB (CW only unless --eicab-wb). Logged per subject.",
)
@click.option(
    "--eicab-wb/--no-eicab-wb",
    default=None,
    help="Run eICAB's whole-brain (WB) flood fill after the CW mask. Default: on, off for the --reuse-eicab "
         "fallback. Without it there is no *_eICAB_WB mask (stage 3's venous exclusion uses the CW labels).",
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
    eicab_thread_limit: int | None,
    sge_pe_smp: int | None,
    vasculature_dir: Path | None,
    log_dir: Path | None,
    err_dir: Path | None,
    keep_aux_outputs: bool,
    post_process_eicab: bool,
    eicab_pp_siphon: bool,
    eicab_pp_distal: bool,
    only_pp: bool,
    reuse_eicab: bool,
    eicab_wb: bool | None,
    emit_script: Path | None,
    no_remote: bool,
    remote_host: str | None,
    remote_user: str | None,
    dry_run: bool,
    backend: str = "gpu",
) -> None:
    """CLI entry point: run eICAB (and optional post-processing) for one or every subject under
    ``nifti_root``, either locally or by submitting SGE jobs."""
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
                    pp_siphon=eicab_pp_siphon,
                    pp_distal=eicab_pp_distal,
                    only_pp=only_pp,
                    reuse_eicab=reuse_eicab,
                    eicab_wb=eicab_wb,
                )
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB skipped: {exc}")
        return

    # SGE: emit locally, upload over sshfs, then SSH-run on the cluster.
    label = subjects[0] if len(subjects) == 1 else f"batch_{len(subjects)}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_script_path, remote_script_path = resolve_sge_script_paths(
        Path(emit_script) if emit_script is not None else None,
        remote_scripts_dir=cfg.SGE_SCRIPTS_DIR,
        default_basename=f"submit_qvtpy_eicab_{label}_{ts}.sh",
    )

    with open(local_script_path, "w", encoding="utf-8") as fh:
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
                    thread_limit=eicab_thread_limit,
                    sge_pe_smp=sge_pe_smp,
                    vasculature_dir=vasculature_dir,
                    log_dir=log_dir,
                    err_dir=err_dir,
                    keep_aux_outputs=keep_aux_outputs,
                    post_process_eicab=post_process_eicab,
                    pp_siphon=eicab_pp_siphon,
                    pp_distal=eicab_pp_distal,
                    only_pp=only_pp,
                    reuse_eicab=reuse_eicab,
                    eicab_wb=eicab_wb,
                    backend=backend,
                    dry_run=False,
                    emit=fh,
                )
            except (FileNotFoundError, OSError) as exc:
                log.warning(f"[{subj}] stage1 eICAB emit skipped: {exc}")

    log.info("=" * 78)
    log.info(f"qvtpy stage1 SGE script written locally: {local_script_path}")
    log.info(f"Cluster path: {remote_script_path}")
    log.info("=" * 78)

    if dry_run:
        log.info("Dry-run: script written; skipping SSH execution.")
        return

    if no_remote:
        log.info("Skipping remote SSH (--no-remote).")
        log.info(f"Upload to cluster and run: bash {remote_script_path}")
        return

    log.reset(restart_progress=False)
    from nvitk.pipes.qvtpy.util.io.cluster_upload import prompt_ssh_credentials

    host_resolved, user, password = prompt_ssh_credentials(
        remote_host=remote_host,
        remote_user=remote_user,
        host_aliases=cfg.CLUSTER_HOST_ALIASES,
    )
    cluster_exec_path = publish_sge_driver_script(
        local_script_path,
        remote_script_path,
        host=host_resolved,
        user=user,
        password=password,
    )
    ok = run_sge_script_ssh(
        host_resolved,
        user,
        password,
        cluster_exec_path,
        local_script_path=local_script_path,
    )
    if not ok:
        log.warning(
            f"Remote execution did not complete successfully. Run manually on the cluster: "
            f"bash {cluster_exec_path}"
        )


__all__ = [
    "build_subject_sge_command",
    "find_existing_eicab",
    "find_tof_resampled_volume",
    "find_tof_volume",
    "run_postprocess_only",
    "run_subject",
    "submit_subject_sge",
    "main",
]


if __name__ == "__main__":
    main()
