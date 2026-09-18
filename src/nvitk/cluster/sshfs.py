"""Mount cluster storage on the workstation over sshfs.

Why this exists
---------------
Cluster storage is not a network directory mounted on the workstation. It is reachable only
over SSH, so every local read or write of a cluster path must go through a transport. This
module is that transport: an sshfs mount of a *configured* cluster root, plus the mapping
between a cluster absolute path and its local view underneath the mountpoint.

The mount is workstation-side only
----------------------------------
Jobs run on the cluster and must see real cluster paths. A mountpoint path in an emitted
driver script, a ``qsub`` argument, or a ``singularity -B`` bind would name a directory that
does not exist on the compute node, and the job would fail with a confusing "no such file".
:meth:`SshfsMount.local` is the only place a cluster path should become a local one, and
:func:`assert_no_mountpoint_leak` is the guard that keeps the two apart.

Only configured roots are mountable
-----------------------------------
:func:`resolve_mount_root` picks the narrowest root from
:func:`nvitk.cluster.sge_json.cluster_working_roots` that contains the requested path, and
raises for anything else. Mounting the cluster's filesystem root is never possible, so a
mistyped or attacker-supplied path cannot expose the whole cluster.

Reconnection
------------
``reconnect`` is in the default options, but sshfs cannot re-authenticate a password-based
mount after the connection drops -- it has no password to replay. A dropped mount therefore
stays dead until it is remounted, which is why :func:`is_live` exists and why
:func:`cluster_mount` replaces a stale mount instead of handing it back.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import os
import posixpath
import re
import getpass
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from nvitk.cluster import sge_json
from nvitk.core.logger import Logger

log = Logger()

#: Seconds to wait for ``sshfs`` itself to return. The mount is asynchronous, so this bounds
#: the handshake, not the transfer.
DEFAULT_MOUNT_TIMEOUT: float = 30.0

#: Seconds to wait for a liveness ``stat`` before calling a mountpoint dead.
DEFAULT_PROBE_TIMEOUT: float = 5.0

#: ``errno`` values a dead or unreachable sshfs mount reports.
_DEAD_MOUNT_ERRNOS = frozenset(
    {errno.ENOTCONN, errno.ETIMEDOUT, errno.EIO, errno.EHOSTDOWN, errno.EHOSTUNREACH}
)


class SshfsError(RuntimeError):
    """Base class for sshfs transport failures."""


class SshfsRootNotAllowed(SshfsError):
    """A cluster path lies outside every configured cluster working root."""


class SshfsMountError(SshfsError):
    """The ``sshfs`` command failed, or the mountpoint never became usable."""


def _normalise_remote(remote_path: str | Path) -> str:
    """Normalise a cluster path to an absolute POSIX path with no trailing slash."""
    text = str(remote_path).strip().replace("\\", "/")
    if not text.startswith("/"):
        raise SshfsError(
            f"Cluster paths must be absolute POSIX paths, got {text!r}. "
            "A relative path here usually means a local path leaked into cluster code."
        )
    return posixpath.normpath(re.sub(r"^/+", "/", text))


@dataclass(frozen=True)
class SshfsMount:
    """One live sshfs mount of a cluster directory, and the path mapping it provides.

    *owned* records whether this process ran the ``sshfs`` command. A mount left behind by
    another process (or an earlier run) is reused but never unmounted, so one pipeline cannot
    pull the filesystem out from under another.
    """

    host: str
    user: str
    remote_root: str
    mountpoint: Path
    owned: bool

    @property
    def key(self) -> tuple[str, str, str]:
        """Registry key identifying this mount: ``(user, host, remote_root)``."""
        return (self.user, self.host, self.remote_root)

    def contains(self, remote_path: str | Path) -> bool:
        """True when *remote_path* falls inside this mount's remote root."""
        target = _normalise_remote(remote_path)
        return target == self.remote_root or target.startswith(self.remote_root + "/")

    def local(self, remote_path: str | Path) -> Path:
        """The local path under this mountpoint that reads and writes *remote_path*.

        This is the only sanctioned cluster-to-local conversion. The result is for local I/O
        on this workstation and must never be written into a script or a ``qsub`` argument.
        """
        target = _normalise_remote(remote_path)
        if target == self.remote_root:
            return self.mountpoint
        if not target.startswith(self.remote_root + "/"):
            raise SshfsRootNotAllowed(
                f"{target!r} is not under this mount's root {self.remote_root!r}. "
                "Open a mount for the right root instead of reusing this one."
            )
        return self.mountpoint / target[len(self.remote_root) + 1 :]

    def remote(self, local_path: str | Path) -> str:
        """The cluster path that *local_path* (somewhere under this mountpoint) refers to.

        ``abspath`` rather than ``resolve``: resolving symlinks under a live sshfs mount
        costs a network round trip per component, and this is pure path arithmetic.
        """
        absolute = Path(os.path.abspath(str(local_path)))
        try:
            relative = absolute.relative_to(self.mountpoint)
        except ValueError as exc:
            raise SshfsError(
                f"{absolute} is not under the mountpoint {self.mountpoint}."
            ) from exc
        text = relative.as_posix()
        return self.remote_root if text in ("", ".") else f"{self.remote_root}/{text}"

    def ensure_dir(self, remote_path: str | Path) -> Path:
        """Create *remote_path* on the cluster through the mount; return its local path."""
        local = self.local(remote_path)
        local.mkdir(parents=True, exist_ok=True)
        return local


def require_sshfs() -> str:
    """Absolute path to the ``sshfs`` binary, or raise naming how to install it."""
    binary = shutil.which("sshfs")
    if binary:
        return binary
    raise SshfsMountError(
        "sshfs is required for all cluster data transfer but was not found on PATH. "
        "Install it with: sudo apt install sshfs   (Debian/Ubuntu) "
        "or: sudo dnf install fuse-sshfs   (Fedora/RHEL)."
    )


def _fusermount_binaries() -> list[str]:
    """Available unmount helpers, FUSE 3 first."""
    found = [shutil.which(name) for name in ("fusermount3", "fusermount")]
    return [path for path in found if path]


def _mount_slug(remote_root: str) -> str:
    """Filesystem-safe, human-recognisable directory name for *remote_root*.

    The hash suffix keeps two different roots from sharing a mountpoint once sanitising has
    flattened their punctuation -- mounting one root over another's data would be silent
    corruption, not an error.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", remote_root.strip("/")) or "root"
    digest = hashlib.sha256(remote_root.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[-64:].strip('_')}-{digest}"


def mountpoint_for(host: str, user: str, remote_root: str | Path) -> Path:
    """Local mountpoint nvitk uses for *remote_root* on *user*@*host*.

    Deterministic, so a mount left up by an earlier run is found and reused rather than
    duplicated beside itself.
    """
    root = _normalise_remote(remote_root)
    return sge_json.sshfs_mount_root() / f"{user}@{host}" / _mount_slug(root)


def _unescape_mount_field(field: str) -> str:
    """Decode the octal escapes ``/proc/mounts`` uses for space, tab, newline and backslash."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _iter_proc_mounts() -> Iterator[tuple[str, str, str, str]]:
    """Yield ``(device, mountpoint, fstype, options)`` for every current mount."""
    try:
        text = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        yield (
            _unescape_mount_field(fields[0]),
            _unescape_mount_field(fields[1]),
            fields[2],
            fields[3],
        )


def is_mounted(mountpoint: str | Path) -> bool:
    """True when an sshfs filesystem is currently mounted at *mountpoint*."""
    target = os.path.abspath(str(mountpoint))
    return any(
        mount_point == target and fstype.startswith("fuse.sshfs")
        for _device, mount_point, fstype, _options in _iter_proc_mounts()
    )


@dataclass(frozen=True)
class ExternalMount:
    """An sshfs mount this process did not create, as read from ``/proc/mounts``."""

    user: str
    host: str
    remote_root: str
    mountpoint: Path
    read_only: bool


def _parse_sshfs_device(device: str) -> tuple[str, str, str] | None:
    """Split an sshfs device string into ``(user, host, remote_root)``.

    sshfs spells its device ``[user@]host:[remote_dir]``. The host precedes the first colon,
    so the colon is split first and the optional ``user@`` taken off the left -- doing it the
    other way round would misread a remote path that happens to contain ``@``.

    ``None`` when the entry cannot be used for mapping: no remote directory, or a relative
    one, which is resolved against the remote home and so cannot be matched against the
    absolute cluster paths nvitk deals in.
    """
    text = str(device or "").strip()
    bracket = text.find("]:")
    if bracket != -1:  # bracketed IPv6 literal, with or without a user: user@[::1]:/path
        head, remote = text[: bracket + 1], text[bracket + 2 :]
    else:
        head, separator, remote = text.partition(":")
        if not separator:
            return None
    user, _at, host = head.rpartition("@")
    if not host or not remote.startswith("/"):
        return None
    root = posixpath.normpath(re.sub(r"^/+", "/", remote))
    return user, host, root


def iter_external_sshfs_mounts() -> Iterator[ExternalMount]:
    """Every sshfs mount currently on this host, whoever created it."""
    for device, mount_point, fstype, options in _iter_proc_mounts():
        if not fstype.startswith("fuse.sshfs"):
            continue
        parsed = _parse_sshfs_device(device)
        if parsed is None:
            continue
        user, host, root = parsed
        flags = options.split(",")
        yield ExternalMount(
            # sshfs defaults to the local username when the device names no user.
            user=user or getpass.getuser(),
            host=host,
            remote_root=root,
            mountpoint=Path(mount_point),
            read_only="ro" in flags,
        )


def find_external_mount(
    *, host: str, user: str, remote_path: str | Path
) -> SshfsMount | None:
    """An sshfs mount someone else already made that exposes *remote_path*, or ``None``.

    This is what stops nvitk opening a second connection to storage the workstation already
    has mounted. A user who keeps ``samwise:/BIOIT_IMAGE`` under ``~/NetVolumes`` gets
    ``/BIOIT_IMAGE/nvitk-sge/gui`` served from there instead of a fresh sshfs of the subtree.

    Matching normalises both sides: the host through the ``sge.json`` alias table (an existing
    mount usually names the cluster by its ssh alias while nvitk resolves it to an address),
    and an absent username to the local one, which is what sshfs itself defaults to.

    Read-only mounts are skipped. nvitk uploads as well as reads, and a half-usable mount
    would fail later at a confusing place rather than here.

    The returned mount is always ``owned=False``, so nothing in this module will ever unmount
    something the user set up.
    """
    if not sge_json.sshfs_reuse_existing_mounts():
        return None
    target = _normalise_remote(remote_path)
    want_host = sge_json.resolve_host_alias(host)
    want_user = str(user or "").strip() or getpass.getuser()

    best: SshfsMount | None = None
    for candidate in iter_external_sshfs_mounts():
        if sge_json.resolve_host_alias(candidate.host) != want_host:
            continue
        if candidate.user != want_user:
            continue
        root = candidate.remote_root
        if not (target == root or target.startswith(root.rstrip("/") + "/")):
            continue
        if candidate.read_only:
            log.info(
                "Not reusing read-only sshfs mount %s for %s; nvitk needs to write.",
                candidate.mountpoint, target,
            )
            continue
        if not is_live(candidate.mountpoint):
            continue
        # Narrowest wins, matching resolve_mount_root: the closest mount means the shortest
        # path translation and the least surprising scope.
        if best is None or len(root) > len(best.remote_root):
            best = SshfsMount(
                host=want_host,
                user=want_user,
                remote_root=root,
                mountpoint=candidate.mountpoint,
                owned=False,
            )
    return best


def is_live(mountpoint: str | Path, *, timeout: float = DEFAULT_PROBE_TIMEOUT) -> bool:
    """True when *mountpoint* is mounted *and* responding.

    A password-authenticated sshfs mount cannot re-authenticate after its SSH connection
    drops, so it can stay listed in ``/proc/mounts`` while every syscall against it fails
    with ``ENOTCONN`` -- or hangs. The probe therefore runs on a daemon thread and is
    abandoned on timeout rather than waited on, so a wedged mount degrades to "not live"
    instead of hanging the caller forever.
    """
    if not is_mounted(mountpoint):
        return False
    outcome: list[bool] = []

    def probe() -> None:
        try:
            os.stat(str(mountpoint))
            outcome.append(True)
        except OSError as exc:
            if exc.errno not in _DEAD_MOUNT_ERRNOS:
                # Mounted, answering, but this path is odd (e.g. EACCES). Still a live mount.
                outcome.append(True)
            else:
                outcome.append(False)

    worker = threading.Thread(target=probe, name="nvitk-sshfs-probe", daemon=True)
    worker.start()
    worker.join(timeout)
    return bool(outcome) and outcome[0]


def unmount(mountpoint: str | Path, *, lazy: bool = True) -> bool:
    """Unmount *mountpoint*. Returns True when it is no longer mounted afterwards."""
    target = str(mountpoint)
    if not is_mounted(target):
        return True
    attempts: list[list[str]] = [[binary, "-u", target] for binary in _fusermount_binaries()]
    umount = shutil.which("umount")
    if umount:
        attempts.append([umount, "-l", target] if lazy else [umount, target])
    for argv in attempts:
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 or not is_mounted(target):
            return True
    log.warning("Could not unmount sshfs mountpoint %s", target)
    return False


def _mount_options(extra: Sequence[str] | None = None) -> list[str]:
    """Configured sshfs options plus ``password_stdin`` and anything in *extra*."""
    options = list(sge_json.sshfs_options())
    options.extend(str(opt).strip() for opt in (extra or ()) if str(opt).strip())
    if not any(opt.split("=", 1)[0] == "password_stdin" for opt in options):
        options.append("password_stdin")
    seen: set[str] = set()
    ordered: list[str] = []
    for opt in options:
        if opt not in seen:
            seen.add(opt)
            ordered.append(opt)
    return ordered


def _explain_mount_failure(host: str, stderr: str) -> str:
    """Turn sshfs's terse stderr into something the user can act on."""
    lowered = stderr.lower()
    if "host key verification failed" in lowered or "remote host identification" in lowered:
        return (
            f"sshfs could not verify the host key for {host}. nvitk will not silently "
            f"disable host-key checking. Trust the host once with:\n"
            f"    ssh-keyscan -H {host} >> ~/.ssh/known_hosts\n"
            f"or connect interactively once with: ssh {host}"
        )
    if "permission denied" in lowered:
        return f"sshfs authentication to {host} failed (permission denied). Check the password."
    if "read: connection reset" in lowered or "connection refused" in lowered:
        return f"sshfs could not reach {host} over SSH."
    if "fuse" in lowered and "device" in lowered:
        return (
            "FUSE is unavailable: /dev/fuse is missing or this user may not use it. "
            "Check that the fuse module is loaded and the user is permitted to mount."
        )
    return stderr.strip() or "sshfs exited non-zero with no diagnostic output."


def mount(
    *,
    host: str,
    user: str,
    password: str,
    remote_root: str | Path,
    options: Sequence[str] | None = None,
    timeout: float = DEFAULT_MOUNT_TIMEOUT,
) -> SshfsMount:
    """Mount *remote_root* from the cluster and return the resulting :class:`SshfsMount`.

    Reuses a live mount already present at the expected mountpoint (``owned=False``). A
    mountpoint that is mounted but not responding is unmounted first, because handing back a
    wedged mount would surface much later as an unexplained hang.

    *host* must already be resolved through
    :func:`nvitk.cluster.remote_transfer.resolve_cluster_host`; this function does not read
    the alias table, so the registry key is always the real hostname.
    """
    root = _normalise_remote(remote_root)
    if root == "/":
        raise SshfsRootNotAllowed("Refusing to mount the cluster filesystem root.")
    reused = find_external_mount(host=host, user=user, remote_path=root)
    if reused is not None:
        log.info(
            "Reusing existing sshfs mount for %s: %s:%s -> %s (not created by nvitk, "
            "so it will not be unmounted)",
            root, reused.user, reused.remote_root, reused.mountpoint,
        )
        return reused

    binary = require_sshfs()
    point = mountpoint_for(host, user, root)

    if is_mounted(point):
        if is_live(point):
            log.info("Reusing existing sshfs mount %s -> %s", point, root)
            return SshfsMount(
                host=host, user=user, remote_root=root, mountpoint=point, owned=False
            )
        log.warning("sshfs mountpoint %s is stale; remounting.", point)
        unmount(point)

    point.mkdir(parents=True, exist_ok=True)
    if any(point.iterdir()):
        raise SshfsMountError(
            f"Mountpoint {point} is not empty and nothing is mounted there. "
            "Move or remove its contents; mounting over them would hide your data."
        )

    argv = [binary, "-o", ",".join(_mount_options(options)), f"{user}@{host}:{root}", str(point)]
    log.info("Mounting cluster storage over sshfs: %s@%s:%s -> %s", user, host, root, point)
    try:
        done = subprocess.run(
            argv,
            input=f"{password}\n",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        unmount(point)
        raise SshfsMountError(
            f"sshfs timed out after {timeout:.0f}s mounting {user}@{host}:{root}."
        ) from exc
    except OSError as exc:
        raise SshfsMountError(f"Could not run sshfs: {exc}") from exc

    if done.returncode != 0:
        raise SshfsMountError(
            f"sshfs failed to mount {user}@{host}:{root} at {point} "
            f"(exit {done.returncode}). {_explain_mount_failure(host, done.stderr)}"
        )
    if not is_live(point):
        unmount(point)
        raise SshfsMountError(
            f"sshfs reported success but {point} is not readable. "
            f"{_explain_mount_failure(host, done.stderr)}"
        )
    log.ok(f"sshfs mounted: {user}@{host}:{root} -> {point}")
    return SshfsMount(host=host, user=user, remote_root=root, mountpoint=point, owned=True)


_REGISTRY_LOCK = threading.RLock()
_ACTIVE: dict[tuple[str, str, str], SshfsMount] = {}
_REFCOUNTS: dict[tuple[str, str, str], int] = {}
#: When each unheld mount stopped being used, for the idle sweep.
_IDLE_SINCE: dict[tuple[str, str, str], float] = {}


def active_mounts() -> list[SshfsMount]:
    """Mounts this process is holding, including idle ones not yet swept."""
    with _REGISTRY_LOCK:
        return list(_ACTIVE.values())


def _sweep_idle_mounts(*, now: float | None = None) -> None:
    """Unmount owned mounts nobody has held for longer than the idle TTL.

    Called when a mount is acquired rather than from a timer: the only process that
    accumulates mounts is one that keeps asking for them, so that is exactly when the sweep
    needs to run, and it costs nothing to a process that has finished.
    """
    ttl = sge_json.sshfs_idle_ttl_seconds()
    moment = time.monotonic() if now is None else now
    with _REGISTRY_LOCK:
        for key, idle_since in list(_IDLE_SINCE.items()):
            if _REFCOUNTS.get(key):
                _IDLE_SINCE.pop(key, None)  # picked up again since it went idle
                continue
            if moment - idle_since < ttl:
                continue
            _IDLE_SINCE.pop(key, None)
            held = _ACTIVE.pop(key, None)
            if held is not None and held.owned:
                log.info("Unmounting idle sshfs mount %s", held.mountpoint)
                unmount(held.mountpoint)


def unmount_all() -> None:
    """Unmount everything this process mounted, held or not."""
    with _REGISTRY_LOCK:
        held = [handle for handle in _ACTIVE.values() if handle.owned]
        _ACTIVE.clear()
        _REFCOUNTS.clear()
        _IDLE_SINCE.clear()
    for handle in held:
        unmount(handle.mountpoint)


@contextmanager
def cluster_mount(
    *,
    host: str,
    user: str,
    password: str,
    remote_root: str | Path,
    options: Sequence[str] | None = None,
) -> Iterator[SshfsMount]:
    """Hold an sshfs mount of *remote_root* for the duration of the ``with`` block.

    Reference counted per ``(user, host, remote_root)``, and kept for
    :func:`~nvitk.cluster.sge_json.sshfs_idle_ttl_seconds` after the last holder leaves. Both
    halves matter: the refcount stops a per-subject loop remounting per iteration, and the
    idle grace stops a *sequence* of short blocks -- the GUI polling ``output/.done`` every
    five seconds -- from paying a full SSH handshake and FUSE setup on every tick. Only mounts
    this process created are ever unmounted.
    """
    root = _normalise_remote(remote_root)
    key = (user, host, root)
    _sweep_idle_mounts()
    with _REGISTRY_LOCK:
        existing = _ACTIVE.get(key)
        if existing is not None and not is_live(existing.mountpoint):
            log.warning("Held sshfs mount %s went stale; remounting.", existing.mountpoint)
            unmount(existing.mountpoint)
            _ACTIVE.pop(key, None)
            existing = None
        if existing is None:
            existing = mount(
                host=host,
                user=user,
                password=password,
                remote_root=root,
                options=options,
            )
            _ACTIVE[key] = existing
            _REFCOUNTS[key] = 0
        _IDLE_SINCE.pop(key, None)
        _REFCOUNTS[key] = _REFCOUNTS.get(key, 0) + 1
        handle = existing
    try:
        yield handle
    finally:
        with _REGISTRY_LOCK:
            remaining = _REFCOUNTS.get(key, 1) - 1
            if remaining > 0:
                _REFCOUNTS[key] = remaining
                return
            _REFCOUNTS.pop(key, None)
            # Left mounted on purpose: the next caller within the idle TTL reuses it instead
            # of paying another handshake. _sweep_idle_mounts and the atexit hook clean up.
            _IDLE_SINCE[key] = time.monotonic()


def resolve_mount_root(remote_path: str | Path) -> str:
    """The narrowest configured cluster root containing *remote_path*.

    Raises
    ------
    SshfsRootNotAllowed
        When no configured root contains the path. This is deliberate: mounting an arbitrary
        parent directory (or ``/``) to satisfy a stray path would expose far more of the
        cluster than the caller asked for. The fix is to add the root to ``sge.json``.
    """
    target = _normalise_remote(remote_path)
    roots = sge_json.cluster_working_roots()
    for root in roots:
        if target == root or target.startswith(root + "/"):
            return root
    raise SshfsRootNotAllowed(
        f"{target!r} is not under any configured cluster working root, so nvitk will not "
        f"mount it. Add the root to sge.json -- as a pipeline 'cluster_*' path or in "
        f"'paths.sshfs_extra_roots'. Currently configured roots: "
        f"{', '.join(roots) if roots else '(none)'}"
    )


@contextmanager
def mount_for_path(
    *,
    host: str,
    user: str,
    password: str,
    remote_path: str | Path,
) -> Iterator[SshfsMount]:
    """Hold a mount of whichever configured root contains *remote_path*."""
    with cluster_mount(
        host=host,
        user=user,
        password=password,
        remote_root=resolve_mount_root(remote_path),
    ) as handle:
        yield handle


def assert_no_mountpoint_leak(text: str, *, what: str = "emitted script") -> None:
    """Raise when *text* contains a local sshfs path, which the cluster cannot resolve.

    Emitted driver scripts, ``qsub`` arguments and ``singularity -B`` binds all run on the
    cluster. A mountpoint path in any of them names a directory that exists only on this
    workstation, and the job fails on the compute node with a misleading "no such file".
    Checking the rendered text catches that here, where the message can say why.
    """
    body = str(text)
    candidates = [str(sge_json.sshfs_mount_root())]
    candidates.extend(str(handle.mountpoint) for handle in active_mounts())
    for candidate in candidates:
        if candidate and candidate in body:
            raise SshfsError(
                f"Local sshfs mountpoint {candidate!r} leaked into {what}. "
                "Cluster-side commands must use real cluster paths; use "
                "SshfsMount.local() only for local reads and writes."
            )


@atexit.register
def _unmount_owned_at_exit() -> None:
    """Unmount anything this process still holds, so mountpoints do not accumulate."""
    unmount_all()


__all__ = [
    "DEFAULT_MOUNT_TIMEOUT",
    "DEFAULT_PROBE_TIMEOUT",
    "ExternalMount",
    "SshfsError",
    "SshfsMount",
    "SshfsMountError",
    "SshfsRootNotAllowed",
    "active_mounts",
    "assert_no_mountpoint_leak",
    "cluster_mount",
    "find_external_mount",
    "is_live",
    "iter_external_sshfs_mounts",
    "is_mounted",
    "mount",
    "mount_for_path",
    "mountpoint_for",
    "require_sshfs",
    "resolve_mount_root",
    "unmount",
    "unmount_all",
]
