"""Stage SGE driver scripts locally and publish them on the cluster via SFTP."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from nvitk.core.logger import Logger

log = Logger()


def is_path_writable_locally(path: Path) -> bool:
    """Return True when *path* (or its parent) can be created/written on this host."""
    target = path if path.suffix else path
    parent = target.parent
    if not str(parent):
        return False
    try:
        parent.mkdir(parents=True, exist_ok=True)
        probe = parent / f".nvitk_write_probe_{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def local_sge_staging_dir() -> Path:
    """Writable directory for emitted bash scripts on the workstation."""
    root = Path(tempfile.gettempdir()) / "nvitk-sge-scripts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_sge_script_paths(
    emit_script: Path | None,
    *,
    remote_scripts_dir: Path,
    default_basename: str | None = None,
) -> tuple[Path, str]:
    """Return ``(local_write_path, remote_cluster_path)`` for an SGE driver script.

    *local_write_path* is always on the current machine. *remote_cluster_path* is the
    path passed to ``bash`` on the cluster login node.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = (
        emit_script.name
        if emit_script is not None
        else (default_basename or f"submit_nvitk_{ts}.sh")
    )
    remote = str((remote_scripts_dir / basename).expanduser()).replace("\\", "/")

    if emit_script is not None:
        candidate = emit_script.expanduser()
        if is_path_writable_locally(candidate if candidate.suffix else candidate / "x"):
            local = candidate.resolve() if candidate.is_absolute() else candidate
            local = Path(local)
            if local.suffix != ".sh":
                local = local.with_suffix(".sh") if local.suffix else Path(f"{local}.sh")
        else:
            log.warning(
                f"--emit-script {emit_script} is not writable locally; "
                f"staging under {local_sge_staging_dir()}"
            )
            local = local_sge_staging_dir() / basename
    else:
        local = local_sge_staging_dir() / basename

    local.parent.mkdir(parents=True, exist_ok=True)
    return local, remote


def publish_sge_driver_script(
    local_path: Path,
    remote_path: str,
    *,
    host: str,
    user: str,
    password: str,
    no_remote: bool = False,
    port: int = 22,
) -> str:
    """Upload *local_path* to *remote_path* when needed; return cluster ``bash`` target."""
    local_path = local_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Local SGE script not found: {local_path}")

    os.chmod(local_path, 0o755)

    if no_remote:
        log.info(f"SGE script written locally: {local_path}")
        log.info(f"On the cluster login node: bash {remote_path}")
        return str(local_path)

    from nvitk.cluster.remote_transfer import resolve_cluster_host, sftp_session, upload_file

    host_resolved = resolve_cluster_host(host)
    log.info(f"Uploading SGE script via SFTP -> {user}@{host_resolved}:{remote_path}")
    with sftp_session(host=host, user=user, password=password, port=port) as (_ssh, sftp):
        upload_file(sftp, local_path, remote_path)
    log.info(f"SGE script uploaded: {remote_path}")
    return remote_path


__all__ = [
    "is_path_writable_locally",
    "local_sge_staging_dir",
    "publish_sge_driver_script",
    "resolve_sge_script_paths",
]
