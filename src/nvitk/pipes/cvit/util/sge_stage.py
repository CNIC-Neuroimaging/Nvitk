"""
Cohort-scoped SGE job construction for CViT stages.

Description
-----------
Same model as the topbrain pipeline (one job per stage, ``-hold_jid`` chains, everything run
inside the nvitk Singularity image). The pipeline-agnostic pieces — source / container
resolution, the unbound-path guard, the login-node driver — are reused from
:mod:`nvitk.pipes.topbrain.util.sge_stage`; only the mount map and the resources are CViT's.

Container mount map
-------------------
::

    /nvitk/src/     ← nvitk source checkout
    /nvitk/data/    ← data_root        (labelled data, read-only)
    /nvitk/output/  ← results_root
    /models/        ← model_root
    /nnunet/{raw,preprocessed,results}
    /nnssl/{raw,preprocessed,results}
    /corpus/        ← corpus_root

**Invariant**: worker commands are built from :func:`container_layout`, never from host paths;
:func:`build_stage_spec` refuses to submit an argv holding an absolute path nothing mounts.
User-supplied paths outside the fixed roots (an images directory, a corpus source, a pretrained
checkpoint) are identity-bound via ``data_paths``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    build_singularity_command,
    python_module_argv,
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.topbrain.util.sge_backend import (
    sge_backend_cli_args,
    sge_stage_extra_env,
    sge_stage_use_nv,
    torch_device_for_backend,
)
from nvitk.pipes.topbrain.util.sge_stage import (
    find_unbound_paths,
    quote_path,
    resolve_container,
    resolve_src_dir,
    run_driver_script as _run_driver_script,
)
from nvitk.pipes.topbrain.util.paths import tree_visible

log = Logger()

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


def container_layout() -> CViTPaths:
    """The roots as a worker sees them inside the container."""
    binds = SingularityBinds()
    return CViTPaths(
        data_root=Path(binds.data),
        results_root=Path(binds.output),
        model_root=Path(binds.models),
        **{key: Path(mount) for key, mount in EXTRA_MOUNTS.items()},
    )


def root_args(paths: CViTPaths) -> list[str]:
    """``--<root>`` flags passing every root of *paths* to a worker CLI."""
    out: list[str] = []
    for key in pth.ROOT_KEYS:
        out += [f"--{key.replace('_', '-')}", quote_path(getattr(paths, key))]
    return out


def host_binds(paths: CViTPaths) -> tuple[tuple[Path, str], ...]:
    return tuple((Path(getattr(paths, key)), mount) for key, mount in EXTRA_MOUNTS.items())


def to_container_path(paths: CViTPaths, host_path: Path | str) -> Path | None:
    """Rewrite a host path under a fixed root into its container path (``None`` if outside)."""
    inside = container_layout()
    candidate = Path(host_path).expanduser()
    for key in pth.ROOT_KEYS:
        root = Path(getattr(paths, key))
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            continue
        return Path(getattr(inside, key)) / rel
    return None


def plan_data_binds(paths: CViTPaths, data_paths: Iterable[Path | str]) -> tuple[tuple[Path, str], ...]:
    """Identity binds for user paths not already reachable through a fixed root."""
    fixed = [Path(getattr(paths, key)) for key in pth.ROOT_KEYS]
    planned: dict[str, tuple[Path, str]] = {}
    for raw in data_paths:
        p = Path(raw).expanduser()
        if any(p == r or r in p.parents for r in fixed):
            continue
        if tree_visible(p) and not p.exists():
            raise FileNotFoundError(f"{p} does not exist (it must be reachable from the cluster).")
        planned[str(p)] = (p, str(p))
    return tuple(planned.values())


def container_mount_points(extra: Sequence[tuple[Path, str]] = ()) -> tuple[str, ...]:
    binds = SingularityBinds()
    return (binds.src, binds.data, binds.output, binds.models, *EXTRA_MOUNTS.values(),
            *(str(m) for _, m in extra))


def stage_resources(backend: str, *, request_gpu: bool | None, h_vmem: str | None,
                    pe_smp: int | None) -> SgeResources:
    gpu = request_gpu if request_gpu is not None else str(backend).lower() == "gpu"
    return SgeResources(
        project=cfg.SGE_PROJECT,
        account=cfg.SGE_ACCOUNT,
        ngpu=(int(cfg.SGE_NGPU) or 1) if gpu else 0,
        h_vmem=h_vmem if h_vmem is not None else cfg.SGE_H_VMEM,
        queue=cfg.SGE_QUEUE,
        pe_smp=int(pe_smp) if pe_smp is not None else None,
    )


def build_stage_spec(
    stage: str,
    argv: Sequence[str],
    *,
    paths: CViTPaths,
    container: Path,
    src_dir: Path | None = None,
    backend: str = "gpu",
    request_gpu: bool | None = None,
    h_vmem: str | None = None,
    pe_smp: int | None = None,
    job_suffix: str = "",
    data_paths: Iterable[Path | str] = (),
) -> tuple[StageSpec, ClusterPaths]:
    """``(StageSpec, ClusterPaths)`` for one cohort-scoped CViT stage.

    Raises
    ------
    ValueError
        If the worker argv references an absolute path that would not exist in the container.
    """
    binds = SingularityBinds()
    name = f"{cfg.SGE_JOB_PREFIX}_{stage}" + (f"_{job_suffix}" if job_suffix else "")
    cluster_paths = ClusterPaths(
        src=resolve_src_dir(src_dir),
        container=resolve_container(container),
        models=paths.model_root,
        data_root=paths.data_root,
        output_root=paths.results_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    extra = host_binds(paths) + plan_data_binds(paths, data_paths)
    unbound = find_unbound_paths(argv, container_mount_points(extra))
    if unbound:
        raise ValueError(
            f"{stage}: the worker command references path(s) that would not exist inside the "
            f"container: {unbound}. Build them from container_layout() or pass them as data_paths."
        )
    spec = StageSpec(
        job_name=name[:63],
        python_cmd=" ".join(argv),
        resources=stage_resources(backend, request_gpu=request_gpu, h_vmem=h_vmem, pe_smp=pe_smp),
        binds=binds,
        use_nv=sge_stage_use_nv(backend, request_gpu=request_gpu),
        extra_env=sge_stage_extra_env(binds.src, backend),
        extra_host_binds=extra,
    )
    return spec, cluster_paths


def submit_stage_job(stage: str, argv: Sequence[str], *, hold_jid=None, dry_run: bool = False,
                     emit: TextIO | None = None, **kwargs: Any) -> str:
    """Emit or submit one stage job; returns its job id (``""`` when only emitted)."""
    spec, cluster_paths = build_stage_spec(stage, argv, **kwargs)
    return submit_stage(spec, cluster_paths, hold_jid=hold_jid, dry_run=dry_run, emit=emit)


def build_stage_command(stage: str, argv: Sequence[str], **kwargs: Any) -> str:
    spec, cluster_paths = build_stage_spec(stage, argv, **kwargs)
    return build_singularity_command(spec, cluster_paths)


def run_driver_script(emit_blocks, **kwargs: Any) -> list[str]:
    """Topbrain's login-node driver, configured with CViT's SGE settings."""
    return _run_driver_script(emit_blocks, config=cfg, host_aliases=pth.CLUSTER_HOST_ALIASES, **kwargs)



__all__ = [
    "EXTRA_MOUNTS",
    "build_stage_command",
    "build_stage_spec",
    "container_layout",
    "host_binds",
    "plan_data_binds",
    "python_module_argv",
    "quote_path",
    "root_args",
    "run_driver_script",
    "sge_backend_cli_args",
    "stage_resources",
    "submit_stage_job",
    "to_container_path",
    "torch_device_for_backend",
]
