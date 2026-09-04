"""SSH/SFTP helpers for uploading GUI SGE jobs to a cluster login node."""

from __future__ import annotations

import os
import shlex
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

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
    """Raise a clear install hint unless ``paramiko`` is available (needed for remote SGE transfer)."""
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
    """True if *remote_path* exists on the SFTP server."""
    try:
        sftp.stat(remote_path)
        return True
    except OSError:
        return False


def read_remote_text(sftp: Any, remote_path: str) -> str:
    """Read a remote text file's full contents (UTF-8, replacing undecodable bytes)."""
    with sftp.open(remote_path, "r") as fh:
        return fh.read().decode("utf-8", errors="replace")


def upload_file(
    sftp: Any,
    local_path: Path,
    remote_path: str,
) -> None:
    """Upload *local_path* to *remote_path*, creating the remote parent directory if needed."""
    parent = remote_path.rsplit("/", 1)[0]
    if parent:
        ensure_remote_dir(sftp, parent)
    sftp.put(str(local_path), remote_path)


def download_remote_file(sftp: Any, remote_path: str, local_path: Path) -> None:
    """Download *remote_path* to *local_path*, creating the local parent directory if needed."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))


def sync_remote_glob(
    client: Any,
    sftp: Any,
    *,
    remote_root: str,
    local_root: Path,
    pattern: str,
) -> tuple[int, int]:
    """Mirror files matching *pattern* under *remote_root* into *local_root*.

    Enumeration is a single ``find`` over the SSH channel rather than a recursive SFTP walk,
    which costs a round trip per directory and would be painful on a results tree that is one
    directory per dataset per run per fold. Only files whose size or mtime differs from the
    local copy are fetched, so a caller polling on an interval re-downloads the handful of logs
    that actually grew.

    Relative paths are preserved, so the local tree has the same shape as the remote one and
    anything that discovers runs by structure keeps working against it.

    Returns
    -------
    tuple
        ``(seen, fetched)`` -- files matched remotely, and of those, files transferred.
    """
    root = _normalize_remote_path(remote_root).rstrip("/")
    command = (
        f"find {shlex.quote(root)} -name {shlex.quote(pattern)} -type f "
        f"-printf '%s\\t%T@\\t%p\\n'"
    )
    _stdin, stdout, _stderr = client.exec_command(command)
    payload = stdout.read().decode(errors="replace")
    # find exits non-zero for any unreadable subtree it walked past; the paths it *did* print
    # are still good, so the status is deliberately not checked.
    stdout.channel.recv_exit_status()

    seen = fetched = 0
    for line in payload.splitlines():
        fields = line.split("\t", 2)
        if len(fields) != 3:
            continue
        raw_size, raw_mtime, remote_path = fields
        try:
            size, mtime = int(raw_size), float(raw_mtime)
        except ValueError:
            continue
        seen += 1
        destination = Path(local_root) / remote_path[len(root):].lstrip("/")
        try:
            existing = destination.stat()
            if existing.st_size == size and existing.st_mtime >= mtime:
                continue
        except OSError:
            pass  # absent locally, or unreadable -- either way, fetch it
        download_remote_file(sftp, remote_path, destination)
        os.utime(destination, (mtime, mtime))
        fetched += 1
    return seen, fetched


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


def upload_files(
    sftp: Any,
    pairs: Sequence[tuple[Path, str]],
    *,
    skip_existing: bool = True,
    on_progress: Any = None,
) -> tuple[int, int]:
    """Upload an explicit ``(local, remote)`` list; return ``(uploaded, skipped)``.

    Neither existing helper fits a filtered transfer: :func:`upload_file` takes one path, and
    :func:`upload_directory` sends a whole tree with no skip logic. A cohort analysis selects a few
    hundred volumes out of several thousand in the same directory, so the *list* is the unit.

    ``skip_existing`` compares the remote size against the local one. Re-running the same cohort
    for a second contrast would otherwise re-send every volume — minutes of transfer for files that
    are already there. Size rather than checksum: an SFTP stat is one round trip, hashing a few
    hundred volumes is not.

    ``on_progress(done, total)`` is called as the transfer advances; a silent twenty-minute upload
    is indistinguishable from a hang.
    """
    total = len(pairs)
    uploaded = skipped = 0
    seen_dirs: set[str] = set()

    for index, (local_path, remote_path) in enumerate(pairs, start=1):
        local = Path(local_path)
        if not local.is_file():
            raise FileNotFoundError(f"Cannot upload, not a file: {local}")

        parent = remote_path.rsplit("/", 1)[0]
        if parent and parent not in seen_dirs:
            ensure_remote_dir(sftp, parent)
            seen_dirs.add(parent)

        if skip_existing:
            try:
                if sftp.stat(remote_path).st_size == local.stat().st_size:
                    skipped += 1
                    if on_progress is not None:
                        on_progress(index, total)
                    continue
            except IOError:
                pass  # not there, or unstatable — upload it

        sftp.put(str(local), remote_path)
        uploaded += 1
        if on_progress is not None:
            on_progress(index, total)

    return uploaded, skipped


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
    """Strip whitespace and any trailing slash from a POSIX remote path."""
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
    "upload_files",
    "upload_staged_job",
]
