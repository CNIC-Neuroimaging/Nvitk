"""SFTP upload of PESA-Fat DICOM trees to cluster storage."""

from __future__ import annotations

from pathlib import Path

from nvitk.cluster.remote_transfer import (
    remote_path_exists,
    resolve_cluster_host,
    sftp_session,
    upload_directory,
)
from nvitk.core.logger import Logger

log = Logger()


def remote_subject_dicom_dir(
    remote_dicom_root: Path,
    batch: str,
    subject: str,
) -> str:
    """POSIX path ``<remote_dicom_root>/<batch>/<subject>`` on the cluster."""
    return f"{str(remote_dicom_root).rstrip('/')}/{batch}/{subject}"


def local_subject_has_dicoms(local_subject_dir: Path) -> bool:
    """True if *local_subject_dir* exists and contains at least one entry."""
    if not local_subject_dir.is_dir():
        return False
    try:
        return any(local_subject_dir.iterdir())
    except OSError:
        return False


def remote_subject_has_dicoms(sftp, remote_subject_dir: str) -> bool:
    """True if *remote_subject_dir* exists on the SFTP connection and is non-empty."""
    if not remote_path_exists(sftp, remote_subject_dir):
        return False
    try:
        return bool(sftp.listdir(remote_subject_dir))
    except OSError:
        return False


def upload_subject_dicoms(
    local_subject_dir: Path,
    remote_dicom_root: Path,
    batch: str,
    subject: str,
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    skip_if_remote_nonempty: bool = True,
) -> bool:
    """Upload one subject's DICOM tree to the cluster PESA-Fat layout.

    Returns ``True`` if data is present on the cluster (uploaded or skipped).
    """
    local_subject_dir = Path(local_subject_dir)
    if not local_subject_has_dicoms(local_subject_dir):
        raise FileNotFoundError(f"No local DICOM data under {local_subject_dir}")

    remote_subj = remote_subject_dicom_dir(remote_dicom_root, batch, subject)
    host_resolved = resolve_cluster_host(host)

    if skip_if_remote_nonempty:
        with sftp_session(host=host, user=user, password=password, port=port) as (_ssh, sftp):
            if remote_subject_has_dicoms(sftp, remote_subj):
                log.info(
                    "[%s] cluster DICOM already present at %s — skip upload",
                    subject,
                    remote_subj,
                )
                return True

    log.info(
        "[%s] uploading DICOM %s -> %s@%s:%s",
        subject,
        local_subject_dir,
        user,
        host_resolved,
        remote_subj,
    )
    upload_directory(
        host=host,
        user=user,
        password=password,
        local_root=local_subject_dir,
        remote_root=remote_subj,
        port=port,
    )
    return True


def upload_batch_dicoms(
    local_lay,
    cluster_lay,
    subjects: list[str],
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    skip_if_remote_nonempty: bool = True,
) -> None:
    """Upload DICOM trees for all *subjects* in a batch."""
    for subject in subjects:
        local_dir = local_lay.subject_dicom_dir(subject)
        upload_subject_dicoms(
            local_dir,
            cluster_lay.dicom_root,
            cluster_lay.batch,
            subject,
            host=host,
            user=user,
            password=password,
            port=port,
            skip_if_remote_nonempty=skip_if_remote_nonempty,
        )


__all__ = [
    "local_subject_has_dicoms",
    "remote_subject_dicom_dir",
    "upload_batch_dicoms",
    "upload_subject_dicoms",
]
