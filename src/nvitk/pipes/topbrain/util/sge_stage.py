"""Cohort-scoped SGE job construction for ToPBrain stages.

Description
-----------
Every stage of this pipeline runs once for the whole cohort rather than once per subject, and
every stage needs the same set of roots visible inside the container. Rather than repeat the
``StageSpec``/``ClusterPaths`` boilerplate in eight stage modules, they share
:func:`build_stage_spec`.

Container mount map
-------------------
:class:`~nvitk.cluster.sge.SingularityBinds` only names four mounts (src, data, output,
models), but this pipeline has ten roots. The framework roots are therefore mounted through
``StageSpec.extra_host_binds`` at fixed, documented locations::

    /nvitk/src/     ← nvitk source checkout
    /nvitk/data/    ← challenge_root   (read-only release)
    /nvitk/output/  ← results_root
    /models/        ← model_root
    /nnunet/{raw,preprocessed,results}
    /nnssl/{raw,preprocessed,results}
    /corpus/        ← corpus_root

The **invariant** every stage must honour: the worker command is built from
:func:`container_layout`, never from host paths. The host↔container mapping lives here and in
:class:`~nvitk.cluster.sge.ClusterPaths`; a host path leaking into a worker argv is a bug that
only shows up on the cluster.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Sequence, TextIO

import nvitk
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    build_singularity_command,
    submit_stage,
)
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain.util.paths import TopBrainPaths
from nvitk.pipes.topbrain.util.sge_backend import (
    sge_stage_extra_env,
    sge_stage_use_nv,
    sge_topbrain_stage_resources,
)

#: Container mount point for each root that is not one of the four standard binds.
EXTRA_MOUNTS: dict[str, str] = {
    "nnunet_raw": "/nnunet/raw",
    "nnunet_preprocessed": "/nnunet/preprocessed",
    "nnunet_results": "/nnunet/results",
    "nnssl_raw": "/nnssl/raw",
    "nnssl_preprocessed": "/nnssl/preprocessed",
    "nnssl_results": "/nnssl/results",
    "corpus_root": "/corpus",
}


def default_nvitk_src_dir() -> Path:
    """Repo ``src/`` directory inferred from the installed ``nvitk`` package location."""
    return Path(nvitk.__file__).resolve().parent.parent


def container_layout() -> TopBrainPaths:
    """The roots as the worker sees them *inside* the container.

    Stage workers receive these, never host paths — see the module docstring.
    """
    binds = SingularityBinds()
    return TopBrainPaths(
        challenge_root=Path(binds.data),
        results_root=Path(binds.output),
        model_root=Path(binds.models),
        **{key: Path(mount) for key, mount in EXTRA_MOUNTS.items()},
    )


def host_binds(paths: TopBrainPaths) -> tuple[tuple[Path, str], ...]:
    """``extra_host_binds`` pairs mapping the host's framework roots to :data:`EXTRA_MOUNTS`."""
    return tuple((getattr(paths, key), mount) for key, mount in EXTRA_MOUNTS.items())


def build_stage_spec(
    stage: str,
    argv: Sequence[str],
    *,
    paths: TopBrainPaths,
    container: Path,
    src_dir: Path | None = None,
    backend: str = "gpu",
    request_gpu: bool | None = None,
    h_vmem: str | None = None,
    pe_smp: int | None = None,
    job_suffix: str = "",
) -> tuple[StageSpec, ClusterPaths]:
    """Build the ``(StageSpec, ClusterPaths)`` pair for one cohort-scoped stage.

    Parameters
    ----------
    stage
        Stage id (``stage0`` …) — becomes part of the job name.
    argv
        The worker command, already shell-quoted, built against :func:`container_layout`.
    request_gpu
        Force the GPU request on or off independently of *backend*. Stage 0 does array work
        that benefits from CuPy but needs no CUDA allocation from SGE, for instance.
    job_suffix
        Appended to the job name to keep concurrent variants (label sets, losses, folds)
        distinguishable in ``qstat``.
    """
    binds = SingularityBinds()
    name = f"{cfg.SGE_JOB_PREFIX}_{stage}"
    if job_suffix:
        name = f"{name}_{job_suffix}"

    cluster_paths = ClusterPaths(
        src=Path(src_dir) if src_dir is not None else default_nvitk_src_dir(),
        container=Path(container),
        models=paths.model_root,
        data_root=paths.challenge_root,
        output_root=paths.results_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=name[:63],
        python_cmd=" ".join(argv),
        resources=sge_topbrain_stage_resources(
            backend, request_gpu=request_gpu, h_vmem=h_vmem, pe_smp=pe_smp
        ),
        binds=binds,
        use_nv=sge_stage_use_nv(backend, request_gpu=request_gpu),
        extra_env=sge_stage_extra_env(binds.src, backend),
        extra_host_binds=host_binds(paths),
    )
    return spec, cluster_paths


def build_stage_command(stage: str, argv: Sequence[str], **kwargs) -> str:
    """Host shell command for one stage — the ``singularity exec`` line, unsubmitted.

    Used by the master when assembling a multi-stage script rather than submitting directly.
    """
    spec, cluster_paths = build_stage_spec(stage, argv, **kwargs)
    return build_singularity_command(spec, cluster_paths)


def submit_stage_job(
    stage: str,
    argv: Sequence[str],
    *,
    hold_jid: str | Sequence[str] | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
    **kwargs,
) -> str:
    """Emit or submit one stage as a standalone SGE job; returns the job id (or ``""``)."""
    spec, cluster_paths = build_stage_spec(stage, argv, **kwargs)
    return submit_stage(spec, cluster_paths, hold_jid=hold_jid, dry_run=dry_run, emit=emit)


def quote_path(value: Path | str) -> str:
    """Shell-quote a path for inclusion in a worker argv."""
    return shlex.quote(str(value))


__all__ = [
    "EXTRA_MOUNTS",
    "build_stage_command",
    "build_stage_spec",
    "container_layout",
    "default_nvitk_src_dir",
    "host_binds",
    "quote_path",
    "submit_stage_job",
]
