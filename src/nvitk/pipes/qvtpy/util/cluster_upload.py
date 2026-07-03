"""SFTP upload of qvtpy DICOM trees from workstation to cluster storage."""

from __future__ import annotations

from pathlib import Path

from nvitk.cluster.remote_transfer import (
    remote_path_exists,
    resolve_cluster_host,
    sftp_session,
    upload_directory,
)
from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy.util.paths import QvtpyPaths

log = Logger()


def remote_subject_dicom_dir(remote_dicom_root: Path, subject: str) -> str:
    """POSIX path ``<remote_dicom_root>/<subject>`` on the cluster."""
    return f"{str(remote_dicom_root).rstrip('/')}/{subject}"


def local_subject_has_dicoms(local_subject_dir: Path) -> bool:
    if not local_subject_dir.is_dir():
        return False
    try:
        return any(local_subject_dir.iterdir())
    except OSError:
        return False


def remote_subject_has_dicoms(sftp, remote_subject_dir: str) -> bool:
    if not remote_path_exists(sftp, remote_subject_dir):
        return False
    try:
        return bool(sftp.listdir(remote_subject_dir))
    except OSError:
        return False


def upload_subject_dicoms(
    local_subject_dir: Path,
    remote_dicom_root: Path,
    subject: str,
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    skip_if_remote_nonempty: bool = True,
) -> bool:
    """Upload one subject DICOM tree to the cluster qvtpy layout."""
    local_subject_dir = Path(local_subject_dir)
    if not local_subject_has_dicoms(local_subject_dir):
        raise FileNotFoundError(f"No local DICOM data under {local_subject_dir}")

    remote_subj = remote_subject_dicom_dir(remote_dicom_root, subject)
    host_resolved = resolve_cluster_host(host)

    if skip_if_remote_nonempty:
        with sftp_session(host=host, user=user, password=password, port=port) as (_ssh, sftp):
            if remote_subject_has_dicoms(sftp, remote_subj):
                log.info(
                    f"[{subject}] cluster DICOM already present at {remote_subj} — skip upload"
                )
                return True

    log.info(
        f"[{subject}] uploading DICOM {local_subject_dir} -> "
        f"{user}@{host_resolved}:{remote_subj}"
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


def upload_subjects_dicoms(
    local_paths: QvtpyPaths,
    cluster_paths: QvtpyPaths,
    subjects: list[str],
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
    skip_if_remote_nonempty: bool = True,
) -> None:
    """Upload DICOM trees for all *subjects* to the cluster layout."""
    for subject in subjects:
        upload_subject_dicoms(
            local_paths.subject_dicom_dir(subject),
            cluster_paths.dicom_root,
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
    "upload_subject_dicoms",
    "upload_subjects_dicoms",
]
