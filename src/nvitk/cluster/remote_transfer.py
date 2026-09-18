"""Cluster data transfer over sshfs, and SSH command execution for job control.

Cluster storage is isolated: it is not mounted on the workstation, so no local path ever
names a cluster file. Every read and write therefore goes through an sshfs mount of a
*configured* cluster root -- see :mod:`nvitk.cluster.sshfs`. There is deliberately no SFTP
fallback; one transport means one set of failure modes.

Two planes
----------
*Data* moves through the mount: :class:`ClusterSession` maps a cluster path to its local view
and ordinary filesystem calls do the rest. *Commands* still go over SSH, because sshfs cannot
run ``qsub``, ``qstat`` or ``bash submit.sh``. :func:`ssh_exec` is that half, and it is also
why ``paramiko`` is still a dependency.

Mount roots are opened lazily
-----------------------------
A session does not mount anything until a path is requested, then mounts the narrowest
configured root containing it and keeps it for the session's lifetime. Callers that touch
several roots get several mounts without asking for them; callers that touch one path pay for
one mount. Mounts are reference counted globally, so nesting sessions over the same root
mounts it once.
"""

from __future__ import annotations

import os
import shlex
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from nvitk.cluster import sge_json
from nvitk.cluster.sshfs import (
    SshfsError,
    SshfsMount,
    SshfsRootNotAllowed,
    assert_no_mountpoint_leak,
    cluster_mount,
    resolve_mount_root,
)
from nvitk.core.logger import Logger

log = Logger()


def resolve_cluster_host(host: str) -> str:
    """Resolve short names (e.g. ``samwise``) via ``.nvitk/sge.json`` aliases."""
    key = str(host or "").strip()
    if not key:
        return key
    paths = sge_json.paths_section()
    aliases = sge_json.merge_cluster_host_aliases({}, paths, {})
    return aliases.get(key, key)


def _require_paramiko() -> None:
    """Raise a clear install hint unless ``paramiko`` is available (needed for SSH exec)."""
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "paramiko is required to run commands on the cluster login node "
            "(pip install 'nvitk[cluster]')."
        ) from exc


def _copy_file(source: Path, destination: Path) -> None:
    """Copy *source* to *destination*, preserving mtime where the server permits it.

    ``copystat`` is best-effort on purpose: some SSH servers refuse ``chmod``/``utime`` through
    sshfs, and failing the whole transfer over a timestamp would be absurd. The mtime matters
    only to the size/mtime skip logic in :func:`upload_files` and :func:`sync_remote_glob`,
    which both degrade to "copy it again" rather than to anything incorrect.
    """
    shutil.copyfile(str(source), str(destination))
    try:
        shutil.copystat(str(source), str(destination))
    except OSError:
        pass


class ClusterSession:
    """Credentials plus the sshfs mounts opened to serve them.

    Replaces the old ``sftp_session`` handle. Where that yielded an SFTP channel, this yields
    a path mapper: :meth:`local` turns a cluster path into the local path that reads and
    writes it, and everything else in this module is built on that.
    """

    def __init__(self, *, host: str, user: str, password: str, port: int = 22) -> None:
        """Record the connection; mounts are opened on first use, not here."""
        self.host = resolve_cluster_host(host)
        self.user = str(user)
        self.port = int(port)
        self._password = password
        self._stack = ExitStack()
        self._mounts: dict[str, SshfsMount] = {}

    def mount_for(self, remote_path: str | Path) -> SshfsMount:
        """The mount serving *remote_path*, opening it if this session has not yet."""
        root = resolve_mount_root(remote_path)
        handle = self._mounts.get(root)
        if handle is None:
            handle = self._stack.enter_context(
                cluster_mount(
                    host=self.host,
                    user=self.user,
                    password=self._password,
                    remote_root=root,
                )
            )
            self._mounts[root] = handle
        return handle

    def local(self, remote_path: str | Path) -> Path:
        """Local path that reads and writes the cluster path *remote_path*."""
        return self.mount_for(remote_path).local(remote_path)

    def mounts(self) -> list[SshfsMount]:
        """Mounts this session currently holds."""
        return list(self._mounts.values())

    def close(self) -> None:
        """Release every mount this session opened."""
        self._stack.close()
        self._mounts.clear()

    def __enter__(self) -> "ClusterSession":
        """Enter the session; mounts are still opened lazily."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release the session's mounts."""
        self.close()


@contextmanager
def cluster_session(
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    remote_root: str | Path | None = None,
) -> Iterator[ClusterSession]:
    """Yield a :class:`ClusterSession` for the cluster login node.

    *remote_root* pre-mounts one root when the caller already knows which it needs, so the
    first failure is a mount error at the top of the operation rather than partway through a
    loop. Omit it to let each path mount its own root on demand.
    """
    session = ClusterSession(host=host, user=user, password=password, port=port)
    try:
        if remote_root is not None:
            session.mount_for(remote_root)
        yield session
    finally:
        session.close()


def ensure_remote_dir(session: ClusterSession, remote_path: str | Path) -> Path:
    """Create *remote_path* on the cluster (parents included); return its local path."""
    local = session.local(remote_path)
    local.mkdir(parents=True, exist_ok=True)
    return local


def remote_path_exists(session: ClusterSession, remote_path: str | Path) -> bool:
    """True if *remote_path* exists on the cluster.

    A path outside every configured mount root raises rather than returning False: that is a
    configuration error, not an absent file, and silently reporting "missing" would send the
    caller looking in the wrong place.
    """
    local = session.local(remote_path)
    try:
        return local.exists()
    except OSError:
        return False


def remote_listdir(session: ClusterSession, remote_path: str | Path) -> list[str]:
    """Entry names directly under *remote_path*, or ``[]`` when it is absent or unreadable."""
    try:
        return sorted(entry.name for entry in session.local(remote_path).iterdir())
    except OSError:
        return []


def read_remote_text(session: ClusterSession, remote_path: str | Path) -> str:
    """Read a cluster text file in full (UTF-8, replacing undecodable bytes)."""
    return session.local(remote_path).read_text(encoding="utf-8", errors="replace")


def upload_file(
    session: ClusterSession,
    local_path: Path,
    remote_path: str | Path,
) -> None:
    """Upload *local_path* to *remote_path*, creating the remote parent directory if needed."""
    destination = session.local(remote_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_file(Path(local_path), destination)


def download_remote_file(
    session: ClusterSession,
    remote_path: str | Path,
    local_path: Path,
) -> None:
    """Download *remote_path* to *local_path*, creating the local parent directory if needed."""
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_file(session.local(remote_path), destination)


def upload_files(
    session: ClusterSession,
    pairs: Sequence[tuple[Path, str]],
    *,
    skip_existing: bool = True,
    on_progress: Any = None,
) -> tuple[int, int]:
    """Upload an explicit ``(local, remote)`` list; return ``(uploaded, skipped)``.

    Neither other helper fits a filtered transfer: :func:`upload_file` takes one path, and
    :func:`upload_directory` sends a whole tree with no skip logic. A cohort analysis selects
    a few hundred volumes out of several thousand in the same directory, so the *list* is the
    unit.

    ``skip_existing`` compares the remote size against the local one. Re-running the same
    cohort for a second contrast would otherwise re-send every volume -- minutes of transfer
    for files that are already there. Size rather than checksum: a stat through the mount is
    one round trip, hashing a few hundred volumes is not.

    ``on_progress(done, total)`` is called as the transfer advances; a silent twenty-minute
    upload is indistinguishable from a hang.
    """
    total = len(pairs)
    uploaded = skipped = 0
    seen_dirs: set[Path] = set()

    for index, (local_path, remote_path) in enumerate(pairs, start=1):
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(f"Cannot upload, not a file: {source}")

        destination = session.local(remote_path)
        parent = destination.parent
        if parent not in seen_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            seen_dirs.add(parent)

        if skip_existing:
            try:
                if destination.stat().st_size == source.stat().st_size:
                    skipped += 1
                    if on_progress is not None:
                        on_progress(index, total)
                    continue
            except OSError:
                pass  # not there, or unstatable -- upload it

        _copy_file(source, destination)
        uploaded += 1
        if on_progress is not None:
            on_progress(index, total)

    return uploaded, skipped


def upload_directory_tree(
    session: ClusterSession,
    local_root: Path,
    remote_root: str | Path,
) -> int:
    """Recursively upload *local_root* into *remote_root*. Returns the file count."""
    source_root = Path(local_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {source_root}")
    destination_root = session.local(remote_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for dirpath, _dirnames, filenames in os.walk(source_root):
        relative = Path(dirpath).relative_to(source_root)
        target_dir = destination_root if relative == Path(".") else destination_root / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            _copy_file(Path(dirpath) / name, target_dir / name)
            count += 1
    return count


def download_directory_tree(
    session: ClusterSession,
    remote_root: str | Path,
    local_root: Path,
) -> int:
    """Recursively download *remote_root* into *local_root*. Returns the file count."""
    source_root = session.local(remote_root)
    if not source_root.is_dir():
        return 0
    destination_root = Path(local_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for dirpath, _dirnames, filenames in os.walk(source_root):
        relative = Path(dirpath).relative_to(source_root)
        target_dir = destination_root if relative == Path(".") else destination_root / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            _copy_file(Path(dirpath) / name, target_dir / name)
            count += 1
    return count


#: Retained name for :func:`download_directory_tree`. The old SFTP-handle spelling is still
#: imported in a few pipelines; the transport changed but the operation did not.
download_directory_sftp = download_directory_tree


def sync_remote_glob(
    session: ClusterSession,
    *,
    remote_root: str | Path,
    local_root: Path,
    pattern: str,
) -> tuple[int, int]:
    """Mirror files matching *pattern* under *remote_root* into *local_root*.

    Only files whose size or mtime differs from the local copy are fetched, so a caller
    polling on an interval re-downloads the handful of logs that actually grew rather than
    the whole tree. Relative paths are preserved, so the local tree has the same shape as the
    remote one and anything that discovers runs by structure keeps working against it.

    Returns
    -------
    tuple
        ``(seen, fetched)`` -- files matched remotely, and of those, files transferred.
    """
    source_root = session.local(remote_root)
    destination_root = Path(local_root)
    if not source_root.is_dir():
        return 0, 0

    seen = fetched = 0
    for remote_file in sorted(source_root.rglob(pattern)):
        try:
            if not remote_file.is_file():
                continue
            stat = remote_file.stat()
        except OSError:
            continue
        seen += 1
        destination = destination_root / remote_file.relative_to(source_root)
        try:
            existing = destination.stat()
            if existing.st_size == stat.st_size and existing.st_mtime >= stat.st_mtime:
                continue
        except OSError:
            pass  # absent locally, or unreadable -- either way, fetch it
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(remote_file, destination)
        try:
            os.utime(destination, (stat.st_mtime, stat.st_mtime))
        except OSError:
            pass
        fetched += 1
    return seen, fetched


def upload_directory(
    *,
    host: str,
    user: str,
    password: str,
    local_root: Path,
    remote_root: str,
    port: int = 22,
    timeout: float | None = None,
) -> int:
    """Recursively upload *local_root* to *remote_root* over sshfs. Returns the file count."""
    with cluster_session(
        host=host, user=user, password=password, port=port, remote_root=remote_root
    ) as session:
        return upload_directory_tree(session, Path(local_root), remote_root)


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
    """Recursively download *remote_root* to *local_root* over sshfs. Returns the file count."""
    with cluster_session(
        host=host, user=user, password=password, port=port, remote_root=remote_root
    ) as session:
        return download_directory_tree(session, remote_root, Path(local_root))


def download_remote_files(
    *,
    host: str,
    user: str,
    password: str,
    remote_files: list[tuple[str, Path]],
    port: int = 22,
) -> None:
    """Download ``(remote_path, local_path)`` pairs over sshfs."""
    with cluster_session(host=host, user=user, password=password, port=port) as session:
        for remote_path, local_path in remote_files:
            download_remote_file(session, remote_path, local_path)


def upload_staged_job(
    *,
    host: str,
    user: str,
    password: str,
    local_staging: Path,
    remote_job_root: str,
    port: int = 22,
    remote_script_path: str | None = None,
) -> None:
    """Upload a GUI job staging tree to the cluster job directory.

    *remote_script_path* additionally places ``submit.sh`` there, which is how the GUI keeps
    its driver scripts in the configured ``sge_scripts_dir`` alongside every pipeline's rather
    than scattering one copy per job directory. The copy inside the job root is kept as well:
    it is what makes a failed job reproducible by hand from the directory that holds its
    inputs.
    """
    root = _normalize_remote_path(remote_job_root)
    with cluster_session(host=host, user=user, password=password, port=port) as session:
        upload_directory_tree(session, Path(local_staging), root)
        if not remote_script_path:
            return
        script = Path(local_staging) / "submit.sh"
        if not script.is_file():
            return
        upload_file(session, script, _normalize_remote_path(str(remote_script_path)))


@contextmanager
def ssh_client(
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    timeout: float | None = None,
) -> Iterator[Any]:
    """Yield a connected Paramiko ``SSHClient`` for running commands on the login node.

    Command execution is the one thing sshfs cannot do, so this is the only remaining use of
    Paramiko: ``qsub``, ``qstat``, ``bash submit.sh`` and ``rm -rf`` of a job tree. It moves
    no file data.
    """
    _require_paramiko()
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=resolve_cluster_host(host),
            port=port,
            username=user,
            password=password,
            timeout=timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        yield client
    finally:
        client.close()


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
    with ssh_client(
        host=host, user=user, password=password, port=port, timeout=timeout
    ) as client:
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
    """Delete a remote job directory after verified retrieval (path guard enforced).

    Deletion runs as ``rm -rf`` over SSH rather than a recursive unlink through the mount:
    it executes on the cluster, so it needs no mount at all, and it costs one round trip
    instead of one per file.
    """
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
    "ClusterSession",
    "SshfsError",
    "SshfsRootNotAllowed",
    "assert_no_mountpoint_leak",
    "cluster_session",
    "download_directory",
    "download_directory_sftp",
    "download_directory_tree",
    "download_remote_file",
    "download_remote_files",
    "ensure_remote_dir",
    "is_safe_gui_job_root",
    "read_remote_text",
    "remote_listdir",
    "remote_path_exists",
    "remove_remote_job_tree",
    "resolve_cluster_host",
    "ssh_client",
    "ssh_exec",
    "sync_remote_glob",
    "upload_directory",
    "upload_directory_tree",
    "upload_file",
    "upload_files",
    "upload_staged_job",
]
