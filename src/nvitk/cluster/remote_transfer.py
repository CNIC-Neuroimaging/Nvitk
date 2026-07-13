"""SSH/SFTP helpers for uploading GUI SGE jobs to a cluster login node."""

from __future__ import annotations

import os
import shlex
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from nvitk.cluster import sge_json


def resolve_cluster_host(host: str) -> str:
    """Resolve short names (e.g. ``samwise``) via ``.nvitk/sge.json`` aliases."""
    key = str(host or "").strip()
    if not key:
        return key
    paths = sge_json.paths_section()
    aliases = sge_json.merge_cluster_host_aliases({}, paths, {})
    return aliases.get(key, key)


def _require_paramiko():
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "paramiko is required for remote SGE upload (pip install 'nvitk[cluster]')."
        ) from exc


@contextmanager
def sftp_session(
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    timeout: float | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Yield ``(ssh_client, sftp)`` connected to the cluster login node."""
    _require_paramiko()
    import paramiko

    host_resolved = resolve_cluster_host(host)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host_resolved,
        port=port,
        username=user,
        password=password,
        timeout=timeout,
        allow_agent=False,
        look_for_keys=False,
    )
    try:
        sftp = client.open_sftp()
        try:
            yield client, sftp
        finally:
            sftp.close()
    finally:
        client.close()


def ensure_remote_dir(sftp: Any, remote_path: str) -> None:
    """Create *remote_path* on the server if missing (POSIX, best-effort)."""
    parts = [p for p in remote_path.replace("\\", "/").split("/") if p]
    cur = ""
    if remote_path.startswith("/"):
        cur = "/"
    for part in parts:
        cur = f"{cur.rstrip('/')}/{part}"
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def remote_path_exists(sftp: Any, remote_path: str) -> bool:
    try:
        sftp.stat(remote_path)
        return True
    except OSError:
        return False


def read_remote_text(sftp: Any, remote_path: str) -> str:
    with sftp.open(remote_path, "r") as fh:
        return fh.read().decode("utf-8", errors="replace")


def upload_file(
    sftp: Any,
    local_path: Path,
    remote_path: str,
) -> None:
    parent = remote_path.rsplit("/", 1)[0]
    if parent:
        ensure_remote_dir(sftp, parent)
    sftp.put(str(local_path), remote_path)


def download_remote_file(sftp: Any, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))


def download_directory_sftp(sftp: Any, remote_root: str, local_root: Path) -> int:
    """Recursively download *remote_root* into *local_root*. Returns file count."""
    remote_root = remote_root.rstrip("/")
    if not remote_path_exists(sftp, remote_root):
        return 0
    local_root = local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for dirpath, _dirnames, filenames in _walk_remote(sftp, remote_root):
        rel = (
            Path(".")
            if dirpath.rstrip("/") == remote_root
            else Path(dirpath).relative_to(remote_root)
        )
        for name in filenames:
            remote_path = f"{dirpath.rstrip('/')}/{name}"
            local_path = local_root / rel / name
            download_remote_file(sftp, remote_path, local_path)
            n_files += 1
    return n_files


def _walk_remote(sftp: Any, remote_root: str) -> Iterator[tuple[str, list[str], list[str]]]:
    """Yield ``(dirpath, dirnames, filenames)`` tuples like :func:`os.walk`."""
    pending: list[str] = [remote_root.rstrip("/")]
    while pending:
        current = pending.pop()
        try:
            entries = sftp.listdir_attr(current)
        except OSError:
            continue
        dirnames: list[str] = []
        filenames: list[str] = []
        for entry in entries:
            name = entry.filename
            if name in (".", ".."):
                continue
            remote_path = f"{current.rstrip('/')}/{name}"
            if stat.S_ISDIR(entry.st_mode):
                dirnames.append(name)
                pending.append(remote_path)
            else:
                filenames.append(name)
        yield current, dirnames, filenames


def download_directory(
    *,
    host: str,
    user: str,
    password: str,
    remote_root: str,
    local_root: Path,
    port: int = 22,
    timeout: float | None = None,
) -> int:
    """Recursively download *remote_root* to *local_root* via SFTP."""
    with sftp_session(
        host=host,
        user=user,
        password=password,
        port=port,
        timeout=timeout,
    ) as (_client, sftp):
        return download_directory_sftp(sftp, remote_root, local_root)


def download_remote_files(
    *,
    host: str,
    user: str,
    password: str,
    remote_files: list[tuple[str, Path]],
    port: int = 22,
) -> None:
    """Download ``(remote_path, local_path)`` pairs via SFTP."""
    with sftp_session(host=host, user=user, password=password, port=port) as (_client, sftp):
        for remote_path, local_path in remote_files:
            download_remote_file(sftp, remote_path, local_path)


def upload_directory(
    *,
    host: str,
    user: str,
    password: str,
    local_root: Path,
    remote_root: str,
    port: int = 22,
    timeout: float | None = None,
) -> None:
    """Recursively upload *local_root* to *remote_root* via SFTP."""
    with sftp_session(
        host=host,
        user=user,
        password=password,
        port=port,
        timeout=timeout,
    ) as (_client, sftp):
        ensure_remote_dir(sftp, remote_root.rstrip("/"))
        local_root = local_root.resolve()
        for dirpath, _dirnames, filenames in os.walk(local_root):
            rel = Path(dirpath).relative_to(local_root)
            remote_dir = f"{remote_root.rstrip('/')}/{rel.as_posix()}".rstrip("/")
            if remote_dir:
                ensure_remote_dir(sftp, remote_dir)
            for name in filenames:
                lp = Path(dirpath) / name
                rp = f"{remote_dir}/{name}" if remote_dir else f"{remote_root.rstrip('/')}/{name}"
                sftp.put(str(lp), rp)


def upload_staged_job(
    *,
    host: str,
    user: str,
    password: str,
    local_staging: Path,
    remote_job_root: str,
    port: int = 22,
) -> None:
    """Upload a GUI job staging tree to the cluster job directory."""
    upload_directory(
        host=host,
        user=user,
        password=password,
        local_root=local_staging,
        remote_root=remote_job_root.rstrip("/"),
        port=port,
    )


def ssh_exec(
    *,
    host: str,
    user: str,
    password: str,
    command: str,
    port: int = 22,
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Run *command* on the login node; return ``(exit_code, stdout, stderr)``."""
    with sftp_session(
        host=host,
        user=user,
        password=password,
        port=port,
        timeout=timeout,
    ) as (client, _sftp):
        _stdin, stdout, stderr = client.exec_command(command)
        out_b = stdout.read()
        err_b = stderr.read()
        code = stdout.channel.recv_exit_status()
        return (
            int(code),
            out_b.decode(errors="replace"),
            err_b.decode(errors="replace"),
        )


def _normalize_remote_path(path: str) -> str:
    return str(path or "").strip().rstrip("/")


def is_safe_gui_job_root(remote_job_root: str) -> bool:
    """Return True when *remote_job_root* is under configured ``gui_sge_job_root``."""
    root = _normalize_remote_path(remote_job_root)
    if not root.startswith("/"):
        return False
    base = sge_json.gui_sge_job_root()
    if not base:
        return True
    base_n = _normalize_remote_path(base)
    return root == base_n or root.startswith(base_n + "/")


def remove_remote_job_tree(
    *,
    host: str,
    user: str,
    password: str,
    remote_job_root: str,
    port: int = 22,
) -> tuple[int, str, str]:
    """Delete a remote job directory after verified retrieval (path guard enforced)."""
    root = _normalize_remote_path(remote_job_root)
    if not root:
        raise ValueError("Remote job path is empty.")
    if not is_safe_gui_job_root(root):
        raise ValueError(
            f"Refusing to delete {root!r}: path is outside configured gui_sge_job_root."
        )
    cmd = f"rm -rf {shlex.quote(root)}"
    return ssh_exec(host=host, user=user, password=password, command=cmd, port=port)


__all__ = [
    "download_directory",
    "download_directory_sftp",
    "download_remote_file",
    "download_remote_files",
    "ensure_remote_dir",
    "is_safe_gui_job_root",
    "read_remote_text",
    "remote_path_exists",
    "remove_remote_job_tree",
    "resolve_cluster_host",
    "sftp_session",
    "ssh_exec",
    "upload_directory",
    "upload_file",
    "upload_staged_job",
]
