"""
TensorBoard monitoring for the ToPBrain training stages.

Description
-----------
Thin glue between :mod:`nvitk.core.tensorboard` — which does the actual log parsing, event
writing and serving — and this pipeline's layout, label sets and SGE submission.

Where the events live
---------------------
``<results_root>/tensorboard/{stage1,stage2}/<dataset>/<run>/fold_<n>/``

``results_root`` rather than either framework's results root, because the two stages train
with different frameworks and one server must show both. It is also already bind-mounted into
the container (``/nvitk/output``), so the same tree is reachable from a cluster node and,
through the mount, from the workstation.

Watching a cluster run
----------------------
Three arrangements, all supported:

``--tensorboard --submit sge`` *(default)*
    Each stage job mirrors its own training logs into the shared tree as it trains. Nothing
    listens on the cluster; the workstation serves the mounted directory with
    ``nvitk-tensorboard``, whose exact command line is logged at submission.
``--tensorboard-serve cluster``
    Additionally submits a small CPU-only job running the server on a compute node with
    ``--bind_all``. It writes ``tensorboard_server.json`` naming the node it landed on, from
    which the ``ssh -L`` tunnel is assembled. Costs a slot for as long as it runs.
``nvitk-tensorboard --mirror … --logdir …``
    Nothing on the cluster is involved at all — the workstation parses the training logs
    across the mount itself. Works for runs that were launched without ``--tensorboard``, and
    for finished ones.

Only one process mirrors a given tree at a time; the rest stand by
(:class:`~nvitk.core.tensorboard.MirrorLock`).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence, TextIO

from nvitk.core.logger import Logger
from nvitk.core.tensorboard import (
    DEFAULT_INTERVAL,
    DEFAULT_PORT,
    SERVER_SIDECAR,
    TensorBoardServer,
    TensorBoardWatcher,
    resolve_port,
    tensorboard_available,
)
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.util import paths as pth
from nvitk.pipes.topbrain.util.paths import TENSORBOARD_DIR, TopBrainPaths

log = Logger()

#: Which framework results root each training stage writes its logs under. The key is also the
#: first component of the TensorBoard run name, so stage 1 and stage 2 stay visually separate.
STAGE_SOURCE_ROOTS: dict[str, str] = {
    "stage1": "nnssl_results",
    "stage2": "nnunet_results",
}

#: ``--tensorboard-serve`` choices. ``auto`` resolves against ``--submit`` — see
#: :func:`resolve_serve_mode`.
SERVE_MODES: tuple[str, ...] = ("auto", "local", "cluster", "none")


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────


def tensorboard_root(results_root: Path) -> Path:
    """The event tree a TensorBoard server should be pointed at."""
    return Path(results_root) / TENSORBOARD_DIR


def server_sidecar_path(results_root: Path) -> Path:
    """Where a cluster-side server records the node and port it came up on."""
    return tensorboard_root(results_root) / SERVER_SIDECAR


def stage_sources(paths: TopBrainPaths, stages: Sequence[str]) -> dict[str, Path]:
    """Label → framework results root, for the training stages in *stages*.

    Stages that do not train (0, 3-5) contribute nothing; passing them is harmless so callers
    can forward their whole stage selection.
    """
    return {
        stage: getattr(paths, STAGE_SOURCE_ROOTS[stage])
        for stage in stages
        if stage in STAGE_SOURCE_ROOTS
    }


def dice_class_names(label_set: str | None) -> tuple[str, ...]:
    """Foreground class names in label order, for the per-class pseudo-dice series.

    nnU-Net's ``Pseudo dice`` list follows ``label_manager.foreground_labels``, i.e. label
    values ascending — the same order :func:`~nvitk.pipes.topbrain.labels.label_map` sorts to.
    Naming matters here: on a 36-class vessel model, ``class_21`` says nothing while
    ``22_L-Pcom`` tells you immediately which side road the model is failing to find.
    """
    if not label_set:
        return ()
    try:
        return tuple(name for _, name in sorted(lbl.label_map(label_set).items()))
    except ValueError:  # unknown label set — numbered series are still useful
        log.debug("No label map for %r; per-class dice series will be numbered.", label_set)
        return ()


# ──────────────────────────────────────────────────────────────────────────────
# Mirroring (used by the training stages)
# ──────────────────────────────────────────────────────────────────────────────


@contextmanager
def monitoring(
    paths: TopBrainPaths,
    *,
    stages: Sequence[str],
    enabled: bool = False,
    label_set: str | None = None,
    event_root: Path | None = None,
    interval: float = DEFAULT_INTERVAL,
) -> Iterator[TensorBoardWatcher | None]:
    """Mirror the given stages' training logs for the duration of the block.

    A no-op when *enabled* is false or TensorBoard is not installed — monitoring must never be
    the reason a training run does not start, so a missing package downgrades to a warning.

    Parameters
    ----------
    stages
        Stage ids whose results roots to watch, e.g. ``("stage2",)``.
    event_root
        Overrides ``<results_root>/tensorboard``. Passed by the SGE workers, whose
        ``results_root`` is the container-side mount.

    Examples
    --------
    >>> with monitoring(paths, stages=("stage2",), enabled=True, label_set="ta36"):
    ...     nnunet_run.train_pretrained(...)  # doctest: +SKIP
    """
    if not enabled:
        yield None
        return
    if not tensorboard_available():
        log.warning(
            "--tensorboard requested but the 'tensorboard' package is not installed; "
            "training continues without it (pip install tensorboard)."
        )
        yield None
        return

    watcher = TensorBoardWatcher(
        stage_sources(paths, stages),
        event_root or tensorboard_root(paths.results_root),
        class_names=dice_class_names(label_set) or None,
        interval=interval,
    )
    watcher.start()
    try:
        yield watcher
    finally:
        watcher.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────────────────────────────────────


def resolve_serve_mode(serve: str, *, submit: str) -> str:
    """Resolve ``auto`` against the submission target.

    ``auto`` starts a local server for a local run, and starts **nothing** for an SGE run: the
    workstation has the cluster mounted, so serving the mount is both cheaper than a cluster
    slot and more reliable than reaching a compute node's port. The command to do so is logged
    instead. ``--tensorboard-serve cluster`` asks for the cluster-side server explicitly.
    """
    mode = str(serve or "auto").strip().lower()
    if mode not in SERVE_MODES:
        raise ValueError(f"Unknown --tensorboard-serve {serve!r}; expected {SERVE_MODES}.")
    if mode != "auto":
        return mode
    return "local" if str(submit).lower() != "sge" else "none"


def workstation_view(cluster: TopBrainPaths, local: TopBrainPaths) -> tuple[TopBrainPaths, bool]:
    """Which layout this workstation should read for a run submitted to the cluster.

    On this lab's setup the cluster storage is NFS-mounted at the *same absolute paths* on the
    workstation, so the ``cluster_*`` roots — the ones the jobs actually write to — are directly
    readable here. The ``local_*`` roots are a separate working copy and contain nothing a
    cluster job produced, so pointing TensorBoard at them would show an empty page.

    Probing the filesystem rather than assuming either way keeps this correct on a machine
    where the mount is absent, at the cost of one ``stat``.

    Returns
    -------
    paths
        The layout to read from here.
    mounted
        Whether that layout is the cluster mount (``True``) or the local fallback.
    """
    if pth.tree_visible(cluster.results_root) or Path(cluster.challenge_root).is_dir():
        return cluster, True
    log.warning(
        "Cluster results root %s is not visible from this host; falling back to the local "
        "roots, which will not contain anything the cluster jobs write.", cluster.results_root,
    )
    return local, False


def local_serve_command(
    event_root: Path,
    *,
    sources: dict[str, Path] | None = None,
    port: int = DEFAULT_PORT,
) -> str:
    """The ``nvitk-tensorboard`` line that serves *event_root* from this workstation.

    Includes ``--mirror`` roots so the command also works when the run was submitted without
    ``--tensorboard``: the mirror then does the parsing locally, across the mount.
    """
    parts = ["nvitk-tensorboard", "--logdir", str(event_root), "--port", str(int(port))]
    for label, root in (sources or {}).items():
        parts.extend(["--mirror", f"{label}={root}"])
    return " ".join(parts)


@contextmanager
def local_server(
    results_root: Path, *, enabled: bool = True, port: int = DEFAULT_PORT
) -> Iterator[TensorBoardServer | None]:
    """Serve ``<results_root>/tensorboard`` on this host for the duration of the block."""
    if not enabled or not tensorboard_available():
        yield None
        return
    event_root = tensorboard_root(results_root)
    server = TensorBoardServer(
        event_root, port=resolve_port(port, host="127.0.0.1"), host="127.0.0.1"
    )
    try:
        server.start()
        yield server
    finally:
        server.stop()


def watch_and_serve(
    paths: TopBrainPaths,
    *,
    stages: Sequence[str],
    label_set: str | None = None,
    port: int = DEFAULT_PORT,
    interval: float = DEFAULT_INTERVAL,
    mirror: bool = True,
) -> None:
    """Mirror and serve from this host, blocking until interrupted.

    The workstation-side counterpart to a cluster run: *paths* is the **local** layout, i.e.
    the cluster roots as they are mounted here, so this reads the same training logs the
    compute nodes are writing. Mirroring locally as well as in the job is harmless — whichever
    process holds the lock does the work and the other stands by.

    Raises
    ------
    RuntimeError
        If TensorBoard is not installed, naming the package.
    """
    if not tensorboard_available():
        raise RuntimeError(
            "TensorBoard is not installed in this environment (pip install tensorboard)."
        )
    event_root = tensorboard_root(paths.results_root)
    watcher = (
        TensorBoardWatcher(
            stage_sources(paths, stages), event_root,
            class_names=dice_class_names(label_set) or None, interval=interval,
        )
        if mirror else None
    )
    if watcher is not None:
        watcher.start()
    server = TensorBoardServer(
        event_root, port=resolve_port(port, host="127.0.0.1"), host="127.0.0.1"
    )
    try:
        server.start()
        log.info("Ctrl-C to stop serving (submitted cluster jobs keep running).")
        server.wait()
    except KeyboardInterrupt:
        log.info("Stopping TensorBoard.")
    finally:
        server.stop()
        if watcher is not None:
            watcher.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Cluster-side server job
# ──────────────────────────────────────────────────────────────────────────────


def _server_worker_argv(*, port: int, interval: float, label_set: str | None) -> list[str]:
    """Worker argv for the TensorBoard server job, against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv
    from nvitk.pipes.topbrain.util.sge_stage import container_layout, quote_path

    inside = container_layout()
    event_root = tensorboard_root(inside.results_root)
    argv = [
        *python_module_argv("nvitk.core.tensorboard"),
        "--logdir", quote_path(event_root),
        "--port", str(int(port)),
        "--bind-all",
        "--interval", str(float(interval)),
        "--sidecar", quote_path(event_root / SERVER_SIDECAR),
    ]
    # Mirror as well, so the job is self-sufficient if the stages were submitted without
    # --tensorboard. The lock keeps it from duplicating a stage job's own mirroring.
    for label, root in stage_sources(inside, tuple(STAGE_SOURCE_ROOTS)).items():
        argv.extend(["--mirror", quote_path(f"{label}={root}")])
    for name in dice_class_names(label_set):
        argv.extend(["--class-name", quote_path(name)])
    return argv


def submit_server_sge(
    *,
    paths: TopBrainPaths,
    container: Path,
    src_dir: Path | None = None,
    port: int = DEFAULT_PORT,
    interval: float = DEFAULT_INTERVAL,
    label_set: str | None = None,
    local_results_root: Path | None = None,
    h_vmem: str = "8G",
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit the cluster-side TensorBoard server; returns its job id.

    Deliberately submitted **without** ``-hold_jid``: it must come up while the training jobs
    are still queued, not after them. It is also CPU-only — a server holding a GPU for the
    length of a training run would be an expensive way to draw graphs.

    The job runs until it is killed (``qdel``) or the queue's wall-clock limit ends it. Its
    node and port land in ``<results_root>/tensorboard/tensorboard_server.json``, which the
    workstation can read across the mount.

    Parameters
    ----------
    paths
        The **cluster** layout — what the job itself sees.
    local_results_root
        The same results root as it is mounted on this workstation, used only to tell the user
        where to read the sidecar from here. The two rarely coincide, and quoting the cluster
        path at a local shell is the usual way this step goes wrong.
    """
    from nvitk.pipes.topbrain.util.sge_stage import submit_stage_job

    job_id = submit_stage_job(
        "tensorboard",
        _server_worker_argv(port=port, interval=interval, label_set=label_set),
        paths=paths, container=container, src_dir=src_dir,
        backend="cpu", request_gpu=False, h_vmem=h_vmem, pe_smp=1,
        hold_jid=None, dry_run=dry_run, emit=emit,
    )
    log.info("TensorBoard server job %s (CPU-only, no hold — it comes up while the training "
             "jobs queue). Stop it with 'qdel %s'.", job_id or "(emitted)", job_id or "<job-id>")
    log.info("  it writes the node it landed on to %s", server_sidecar_path(paths.results_root))
    if local_results_root is not None:
        log.info("  read from here as %s", server_sidecar_path(local_results_root))
    log.info("  then tunnel: ssh -N -L %d:<node>:%d <cluster-login-host> "
             "and open http://localhost:%d", port, port, port)
    return job_id


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_PORT",
    "SERVE_MODES",
    "STAGE_SOURCE_ROOTS",
    "dice_class_names",
    "local_serve_command",
    "local_server",
    "monitoring",
    "resolve_serve_mode",
    "server_sidecar_path",
    "stage_sources",
    "submit_server_sge",
    "tensorboard_root",
    "watch_and_serve",
    "workstation_view",
]
