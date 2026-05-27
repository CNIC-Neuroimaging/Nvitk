"""SSH/SFTP helpers for uploading GUI SGE jobs to a cluster login node."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def upload_file(
    sftp: Any,
    local_path: Path,
    remote_path: str,
) -> None:
    parent = remote_path.rsplit("/", 1)[0]
    if parent:
        ensure_remote_dir(sftp, parent)
    sftp.put(str(local_path), remote_path)


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
        finally:
            sftp.close()
    finally:
        client.close()


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


__all__ = [
    "ensure_remote_dir",
    "resolve_cluster_host",
    "upload_directory",
    "upload_file",
    "upload_staged_job",
]
