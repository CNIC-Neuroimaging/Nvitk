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

User-supplied data paths
------------------------
Some options name a location the pipeline cannot know in advance — an extra corpus source, an
annotated cohort, a loss-config file, a baseline run. Those genuinely are host paths, and the
ten fixed roots cannot cover them.

They are handled by **identity binding**: ``-B /host/path:/host/path``, so the path means the
same thing inside the container as outside and the argv needs no rewriting. That is only sound
because under ``--submit sge`` the paths in question are already cluster paths — the pipeline
resolves the ``cluster_*`` layout, and a workstation-only path passed there would be wrong for
the job whatever we did with it. :func:`plan_data_binds` skips anything already covered by a
fixed root, so ``--corpus-source topbrain`` adds no second mount of the release.

Enforcement
-----------
:func:`find_unbound_paths` scans every built argv for absolute paths and
:func:`build_stage_spec` refuses to submit when one is not reachable inside the container. This
is what keeps the invariant from quietly eroding: a new flag that takes a path fails at
submission with a message naming it, rather than after an hour in the GPU queue.
"""

from __future__ import annotations

import os
import shlex
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, TextIO

import nvitk
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    build_singularity_command,
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain.util import paths as pth
from nvitk.pipes.topbrain.util.paths import TopBrainPaths
from nvitk.pipes.topbrain.util.sge_backend import (
    sge_stage_extra_env,
    sge_stage_use_nv,
    sge_topbrain_stage_resources,
)

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


def default_nvitk_src_dir() -> Path:
    """Repo ``src/`` directory inferred from the installed ``nvitk`` package location."""
    return Path(nvitk.__file__).resolve().parent.parent


def resolve_src_dir(src_dir: Path | str | None = None) -> Path:
    """The ``src/`` tree to bind into the container, in precedence order.

    ``--src-dir`` > ``sge.json`` ``paths.nvitk_src_dir`` > the installed package location.

    The configured value wins over the installed one because a cluster job should run the
    *deployed* checkout, not whatever happens to be importable on the submitting workstation.
    That only holds if the deployment is current, so the resolved path is logged at submission
    and sanity-checked: a tree without ``nvitk/__init__.py`` is rejected here rather than
    surfacing as ``ModuleNotFoundError`` inside the container.

    Raises
    ------
    FileNotFoundError
        If the resolved directory is missing or is not an nvitk source tree.
    """
    from nvitk.pipes.topbrain import config as _cfg

    if src_dir is not None:
        resolved, origin = Path(src_dir), "--src-dir"
    else:
        configured = _cfg.NVITK_SRC_DIR
        if configured:
            resolved, origin = Path(configured), "sge.json paths.nvitk_src_dir"
        else:
            resolved, origin = default_nvitk_src_dir(), "installed nvitk package"

    # These checks answer "did you point me at a real checkout?", and only this host's
    # filesystem can answer it. When the cluster storage is not mounted here the question is
    # unanswerable, not failed: the job runs on a node that does see it. Refusing to submit
    # would be this host vetoing a path it has no view of. The check stays exactly as strict
    # wherever the tree *is* visible, which is where a typo can still be caught.
    if not pth.tree_visible(resolved):
        log.warning(
            "Cannot see %s from this host, so the nvitk source tree for the job is taken on "
            "trust. Make sure it is deployed there and up to date.", resolved,
        )
    elif not (resolved / "nvitk" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"nvitk source tree for the cluster job not found at {resolved} (from {origin}): "
            f"it has no nvitk/__init__.py. Point --src-dir at a checkout, or fix "
            f"pipelines' paths.nvitk_src_dir in sge.json."
        )
    local = default_nvitk_src_dir()
    if resolved.resolve() != local.resolve():
        log.info("Binding nvitk source from %s (%s), not the local checkout at %s.",
                 resolved, origin, local)
        warn_if_stale(resolved, local)
    return resolved


#: Package subtrees whose modification times decide whether a deployment looks stale. The
#: pipeline and the core it leans on; scanning the whole tree would be slower for no more signal.
_STALENESS_SUBTREES: tuple[str, ...] = ("nvitk/pipes/topbrain", "nvitk/core")


def _newest_source_mtime(root: Path) -> float:
    """Most recent ``.py`` modification time under :data:`_STALENESS_SUBTREES` of *root*."""
    newest = 0.0
    for subtree in _STALENESS_SUBTREES:
        for path in (root / subtree).rglob("*.py"):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:  # a file vanishing mid-scan is not worth failing a submission for
                continue
    return newest


def warn_if_stale(deployed: Path, local: Path) -> bool:
    """Warn when *deployed* looks older than *local*; returns whether it does.

    The failure this prevents is specific and otherwise baffling: you add a flag locally, submit,
    and the job dies with ``no such option: --sampling`` because the cluster ran a deployment
    from last week. Comparing newest-source timestamps catches it at submission, where the fix
    is one ``rsync``.

    Only a warning — running an older deployment on purpose is legitimate, and a hard failure
    would make that impossible.
    """
    deployed_mtime, local_mtime = _newest_source_mtime(deployed), _newest_source_mtime(local)
    if not deployed_mtime or not local_mtime or deployed_mtime >= local_mtime:
        return False
    from datetime import datetime

    log.warning(
        "The deployed source at %s looks STALE: its newest module is from %s, the local "
        "checkout has changes from %s. Jobs will run the deployed code, so anything added "
        "locally since then will fail with 'no such option'. Sync it first:\n"
        "    rsync -a --delete %s/ %s/",
        deployed,
        datetime.fromtimestamp(deployed_mtime).isoformat(timespec="minutes"),
        datetime.fromtimestamp(local_mtime).isoformat(timespec="minutes"),
        str(local).rstrip("/"), str(deployed).rstrip("/"),
    )
    return True


def resolve_container(container: Path | str) -> Path:
    """Validate the Singularity image exists and is readable on this host.

    Checked at submission because the alternative is what it replaces: the job queues, starts,
    and singularity reports ``could not open image ... no such file or directory`` into an
    ``.err`` file, having consumed a slot and told you nothing about which images do exist.

    Raises
    ------
    FileNotFoundError
        Naming the missing image and listing the ``.sif`` files that *are* in its directory,
        newest first, so the fix is visible in the error itself.
    """
    image = Path(container).expanduser()
    if image.is_file() and os.access(image, os.R_OK):
        return image
    if not pth.tree_visible(image.parent):
        log.warning(
            "Cannot see %s from this host; the container is taken on trust and will be "
            "resolved on the compute node.", image,
        )
        return image

    available: list[str] = []
    parent = image.parent
    if parent.is_dir():
        candidates = sorted(
            (c for c in parent.glob("*.sif") if os.access(c, os.R_OK)),
            key=lambda c: c.stat().st_mtime, reverse=True,
        )
        available = [f"{c.name} ({datetime.fromtimestamp(c.stat().st_mtime):%Y-%m-%d})"
                     for c in candidates[:8]]

    reason = "is not readable" if image.exists() else "does not exist"
    hint = (
        f" Images available in {parent}: {', '.join(available)}."
        if available else f" No readable .sif images found in {parent}."
    )
    raise FileNotFoundError(
        f"Container image {image} {reason}.{hint} Pass --container, or fix "
        f"pipelines.{cfg.PIPELINE_NAME}.default_sge_container_root in sge.json."
    )


def container_mount_points(extra: Sequence[tuple[Path, str]] = ()) -> tuple[str, ...]:
    """Every path that resolves inside the container, for :func:`find_unbound_paths`."""
    binds = SingularityBinds()
    return (
        binds.src, binds.data, binds.output, binds.models,
        *EXTRA_MOUNTS.values(),
        *(str(mount) for _, mount in extra),
    )


def plan_data_binds(
    paths: TopBrainPaths, data_paths: Iterable[Path | str]
) -> tuple[tuple[Path, str], ...]:
    """Identity binds for user-supplied *data_paths* not already inside a fixed root.

    Returns ``(host, container)`` pairs where the two are equal — see the module docstring on
    why identity mapping is the right call for paths the pipeline did not choose.

    Raises
    ------
    FileNotFoundError
        If a path does not exist on the submitting host. Under ``--submit sge`` these are
        cluster paths, and the workstation sees the same mounts, so a miss here means the job
        would not find it either — much better caught now than after the queue wait.
    """
    fixed = [
        Path(paths.challenge_root), Path(paths.results_root), Path(paths.model_root),
        *(Path(getattr(paths, key)) for key in EXTRA_MOUNTS),
    ]
    planned: dict[str, tuple[Path, str]] = {}
    for raw in data_paths:
        candidate = Path(raw).expanduser()
        if not pth.tree_visible(candidate):
            log.warning(
                "Cannot see %s from this host; binding it on trust. If the path is wrong the "
                "job will fail on the node instead of here.", candidate,
            )
            resolved = candidate
            if not any(resolved == root or root in resolved.parents for root in fixed):
                planned[str(resolved)] = (resolved, str(resolved))
            continue
        if not candidate.exists():
            raise FileNotFoundError(
                f"{candidate} does not exist on this host. Under --submit sge the paths on the "
                f"command line must be the cluster's (the pipeline resolves the cluster "
                f"layout), and they must be reachable from here to be bind-mounted."
            )
        resolved = candidate.resolve()
        if any(resolved == root or root in resolved.parents for root in fixed):
            continue  # already reachable through one of the fixed mounts
        planned[str(resolved)] = (resolved, str(resolved))
    return tuple(planned.values())


#: Separators that can join several paths inside one CLI token, e.g.
#: ``name=/images:/labels:mr`` or ``pesa_tof=/root:glob``.
_SPEC_SEPARATORS = str.maketrans({"=": " ", ":": " ", ",": " "})


def find_unbound_paths(argv: Sequence[str], mounts: Sequence[str]) -> list[str]:
    """Absolute paths in *argv* that no container mount would make reachable.

    Deliberately crude and deliberately noisy: it splits each token on the separators the
    pipeline's compound specs use and flags anything starting with ``/`` that is not under a
    mount. A false positive costs one explicit bind; a false negative costs an hour of queue
    time and a confusing "file not found" from inside a container.
    """
    unbound: list[str] = []
    for token in argv:
        for part in str(token).strip("'\"").translate(_SPEC_SEPARATORS).split():
            part = part.strip("'\"")
            if not part.startswith("/"):
                continue
            if any(part == m.rstrip("/") or part.startswith(m.rstrip("/") + "/")
                   for m in mounts):
                continue
            unbound.append(part)
    return sorted(set(unbound))


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
    data_paths: Iterable[Path | str] = (),
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
    data_paths
        Host locations named by user-supplied options (extra corpus sources, annotated
        cohorts, a loss-config file, a baseline run). Identity-bound unless a fixed root
        already covers them — see the module docstring.

    Raises
    ------
    ValueError
        If the assembled argv references an absolute path that nothing would mount. Catching
        it here is the whole point: the alternative is a job that queues for an hour and then
        cannot find its data.
    """
    binds = SingularityBinds()
    name = f"{cfg.SGE_JOB_PREFIX}_{stage}"
    if job_suffix:
        name = f"{name}_{job_suffix}"

    cluster_paths = ClusterPaths(
        src=resolve_src_dir(src_dir),
        container=resolve_container(container),
        models=paths.model_root,
        data_root=paths.challenge_root,
        output_root=paths.results_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )

    extra_binds = host_binds(paths) + plan_data_binds(paths, data_paths)
    unbound = find_unbound_paths(argv, container_mount_points(extra_binds))
    if unbound:
        raise ValueError(
            f"{stage}: the worker command references path(s) that would not exist inside the "
            f"container: {unbound}. Either build the argument from container_layout(), or "
            f"pass the host path through build_stage_spec(data_paths=...) so it is "
            f"bind-mounted."
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
        extra_host_binds=extra_binds,
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
    "container_mount_points",
    "default_nvitk_src_dir",
    "find_unbound_paths",
    "host_binds",
    "plan_data_binds",
    "quote_path",
    "resolve_container",
    "resolve_src_dir",
    "submit_stage_job",
    "warn_if_stale",
]
