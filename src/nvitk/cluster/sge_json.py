"""Read ``sge.json`` and merge its overlays onto Python defaults.

Where the file lives is decided by :mod:`nvitk.core.config_paths`; this module only knows how
to interpret its contents.
"""

from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import Any, Mapping

from nvitk.core import config_paths

SGE_JSON_NAME = "sge.json"


def sge_json_path() -> Path | None:
    """Locate ``sge.json``; see :func:`nvitk.core.config_paths.describe_search` for where."""
    return config_paths.config_file(SGE_JSON_NAME)


def load_sge_document() -> dict[str, Any]:
    """Parsed ``sge.json``, or an empty dict when it is absent or unreadable.

    Cached by :mod:`~nvitk.core.config_paths`, so the repeated ``paths_section()`` /
    ``defaults_section()`` / ``pipeline_section()`` calls throughout the codebase no longer
    re-read and re-parse the file each time.
    """
    return config_paths.load_json(SGE_JSON_NAME)


def paths_section() -> dict[str, Any]:
    """The ``paths`` block of ``sge.json`` (empty dict if absent)."""
    return dict(load_sge_document().get("paths", {}))


def resolve_nvitk_src_dir(*, fallback: Path | None = None) -> Path:
    """Cluster/host nvitk source tree from ``paths.nvitk_src_dir`` in ``sge.json``."""
    paths = paths_section()
    raw = paths.get("nvitk_src_dir")
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return fallback
    return Path(__file__).resolve().parents[1]


def gui_sge_job_root() -> str:
    """Default remote staging root for GUI SGE jobs (``paths.gui_sge_job_root``)."""
    raw = paths_section().get("gui_sge_job_root")
    if raw is None or not str(raw).strip():
        return ""
    return str(raw).strip().rstrip("/")


def sge_scripts_dir() -> str:
    """Cluster directory holding submitted driver scripts (``paths.sge_scripts_dir``).

    The pipelines already resolve this key for their own ``submit_*.sh``; the GUI
    reads the same one so every cluster launch leaves its script in one place
    instead of one copy per job directory.
    """
    raw = paths_section().get("sge_scripts_dir")
    if raw is None or not str(raw).strip():
        return ""
    return str(raw).strip().rstrip("/")


#: Default local base for sshfs mountpoints when ``paths.sshfs_mount_root`` is unset.
DEFAULT_SSHFS_MOUNT_ROOT = "~/.cache/nvitk/sshfs"

#: Seconds an idle sshfs mount is kept alive before being torn down
#: (``paths.sshfs_idle_ttl_seconds``).
DEFAULT_SSHFS_IDLE_TTL_SECONDS: float = 300.0

#: ``-o`` options applied to every sshfs mount when ``paths.sshfs_options`` is unset.
#: ``reconnect`` plus the keepalives matter for long pipeline runs: without them a transient
#: network drop leaves a mountpoint whose every syscall fails until it is unmounted by hand.
DEFAULT_SSHFS_OPTIONS: tuple[str, ...] = (
    "reconnect",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=3",
)


def sshfs_mount_root() -> Path:
    """Local directory sshfs mountpoints are created under (``paths.sshfs_mount_root``)."""
    raw = paths_section().get("sshfs_mount_root")
    if raw is None or not str(raw).strip():
        raw = DEFAULT_SSHFS_MOUNT_ROOT
    return Path(os.path.expanduser(str(raw).strip()))


def resolve_host_alias(host: str) -> str:
    """Resolve a short cluster name (``samwise``) through ``paths.cluster_host_aliases``.

    Lives here rather than in :mod:`nvitk.cluster.remote_transfer` because
    :mod:`nvitk.cluster.sshfs` needs it too, and that module cannot import ``remote_transfer``
    without a cycle. ``remote_transfer.resolve_cluster_host`` delegates to this.
    """
    key = str(host or "").strip()
    if not key:
        return key
    aliases = merge_cluster_host_aliases({}, paths_section(), {})
    return aliases.get(key, key)


def sshfs_reuse_existing_mounts() -> bool:
    """Whether to reuse an sshfs mount somebody else already made (``paths.sshfs_reuse_existing_mounts``).

    On by default. A workstation that already keeps cluster storage mounted -- say
    ``samwise:/BIOIT_IMAGE`` under ``~/NetVolumes`` -- should not gain a second connection to
    the same tree just because nvitk wants a subdirectory of it.
    """
    raw = paths_section().get("sshfs_reuse_existing_mounts")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def sshfs_idle_ttl_seconds() -> float:
    """Seconds an unused sshfs mount is kept before it is torn down.

    Mounting costs a full SSH handshake plus FUSE setup -- one to three seconds. Anything that
    checks the cluster on a timer (the GUI polls ``output/.done`` every five seconds) would
    otherwise remount on every tick. Keeping an idle mount briefly makes those checks free;
    the TTL stops a long-lived process from holding mounts it has finished with.

    ``0`` restores unmount-as-soon-as-unused.
    """
    raw = paths_section().get("sshfs_idle_ttl_seconds")
    if raw is None or not str(raw).strip():
        return DEFAULT_SSHFS_IDLE_TTL_SECONDS
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return DEFAULT_SSHFS_IDLE_TTL_SECONDS


def sshfs_options() -> list[str]:
    """``-o`` options for every sshfs mount (``paths.sshfs_options``, else the defaults)."""
    raw = paths_section().get("sshfs_options")
    if not isinstance(raw, (list, tuple)) or not raw:
        return list(DEFAULT_SSHFS_OPTIONS)
    return [str(opt).strip() for opt in raw if str(opt).strip()]


def _cluster_root(value: Any) -> str:
    """Normalise a configured *cluster* path to an absolute POSIX root, or ``""``.

    Non-absolute values are dropped rather than expanded. ``~`` on a cluster path would
    expand to the *workstation's* home directory here, which is exactly the local/remote
    confusion the sshfs layer exists to remove. ``/`` itself is dropped too: it is never a
    legitimate mount root.
    """
    text = str(value or "").strip().replace("\\", "/")
    if not text.startswith("/"):
        return ""
    normalised = posixpath.normpath(text)
    return "" if normalised == "/" else normalised


def sshfs_extra_roots() -> list[str]:
    """Extra cluster roots the sshfs layer may mount (``paths.sshfs_extra_roots``).

    An escape hatch for cluster directories no ``sge.json`` key names, so adding one does
    not require a code change.
    """
    raw = paths_section().get("sshfs_extra_roots")
    if not isinstance(raw, (list, tuple)):
        return []
    return [root for root in (_cluster_root(item) for item in raw) if root]


def cluster_working_roots() -> list[str]:
    """Every cluster directory nvitk is configured to touch, longest path first.

    This is the allowlist the sshfs layer mounts from. Only roots that appear in
    ``sge.json`` are reachable, so a stray path can never end up mounting the cluster's
    whole filesystem. Collected from the SGE ``paths`` block, from every
    ``pipelines.*.cluster_*`` data root, and from :func:`sshfs_extra_roots`.

    Sorted longest-first so :func:`nvitk.cluster.sshfs.resolve_mount_root` picks the
    narrowest root containing a path rather than the first one that happens to match.
    """
    roots: set[str] = set()
    paths = paths_section()
    for key in ("gui_sge_job_root", "sge_scripts_dir", "sge_log_root", "sge_err_root"):
        root = _cluster_root(paths.get(key))
        if root:
            roots.add(root)
    pipes = load_sge_document().get("pipelines")
    if isinstance(pipes, dict):
        for section in pipes.values():
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if not str(key).startswith("cluster_"):
                    continue
                root = _cluster_root(value)
                if root:
                    roots.add(root)
    roots.update(sshfs_extra_roots())
    return sorted(roots, key=lambda root: (-len(root), root))


def resolve_nvitk_container(
    *, pipe: Mapping[str, Any] | None = None, fallback: Path | None = None
) -> Path | None:
    """Cluster nvitk Singularity image from pipeline override, ``sge.json``, or the registry.

    ``None`` when nothing is configured. There is deliberately no built-in image path: one
    would be specific to a single institution's filesystem, and silently returning it makes an
    unconfigured install fail later with a confusing "no such file" instead of saying which
    setting is missing. Callers that must have an image should pass the result through
    :func:`nvitk.core.config_paths.require`.
    """
    if pipe:
        for key in ("default_sge_container_root", "sge_container_root", "container_path"):
            raw = pipe.get(key)
            if raw is not None and str(raw).strip():
                return Path(os.path.expanduser(str(raw).strip()))
    paths = paths_section()
    raw = paths.get("nvitk_container")
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    try:
        from nvitk.registry.containers import resolve_nvitk_cluster_sif

        reg_path = resolve_nvitk_cluster_sif()
        if reg_path is not None:
            return reg_path
    except Exception:
        pass
    return fallback


def defaults_section() -> dict[str, Any]:
    """The ``defaults`` block of ``sge.json`` (empty dict if absent)."""
    return dict(load_sge_document().get("defaults", {}))


def pipeline_section(pipeline_id: str) -> dict[str, Any]:
    """The ``pipelines[pipeline_id]`` block of ``sge.json`` (empty dict if absent)."""
    doc = load_sge_document()
    pipes = doc.get("pipelines")
    if not isinstance(pipes, dict):
        return {}
    raw = pipes.get(pipeline_id, {})
    return dict(raw) if isinstance(raw, dict) else {}


def merged_pipeline_flat(pipeline_id: str) -> dict[str, Any]:
    """``defaults`` shallow-updated by ``pipelines[pipeline_id]``."""
    out = defaults_section()
    out.update(pipeline_section(pipeline_id))
    return out


def _p(path_like: Any) -> Path:
    """Coerce a string/Path-like value to a user-expanded :class:`Path`."""
    return Path(os.path.expanduser(str(path_like)))


def resolve_log_err_dirs(
    *,
    paths: Mapping[str, Any],
    pipe: Mapping[str, Any],
    fallback_log: Path,
    fallback_err: Path,
) -> tuple[Path, Path]:
    """Resolve SGE log/err dirs: explicit paths, or ``sge_*_root`` + optional ``*_subdir``."""
    log_dir = pipe.get("sge_log_dir") or paths.get("sge_log_dir")
    err_dir = pipe.get("sge_err_dir") or paths.get("sge_err_dir")
    if log_dir:
        lg = _p(log_dir)
    else:
        root = paths.get("sge_log_root")
        sub = pipe.get("log_subdir") if "log_subdir" in pipe else pipe.get("sge_log_subdir")
        if root:
            r = _p(root)
            if sub is not None and str(sub).strip():
                lg = r / str(sub).strip()
            else:
                lg = r
        else:
            lg = fallback_log
    if err_dir:
        er = _p(err_dir)
    else:
        root = paths.get("sge_err_root")
        sub = pipe.get("err_subdir") if "err_subdir" in pipe else pipe.get("sge_err_subdir")
        if root:
            r = _p(root)
            if sub is not None and str(sub).strip():
                er = r / str(sub).strip()
            else:
                er = r
        else:
            er = fallback_err
    return lg, er


def merge_cluster_host_aliases(
    base: dict[str, str],
    paths: Mapping[str, Any],
    pipe: Mapping[str, Any],
) -> dict[str, str]:
    """Merge ``cluster_host_aliases`` maps from *paths* then *pipe* on top of *base* (later sources win)."""
    out = dict(base)
    for section in (paths, pipe):
        extra = section.get("cluster_host_aliases")
        if isinstance(extra, dict):
            for k, v in extra.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
    return out
