"""SFTP upload of qvtpy DICOM trees from workstation to cluster storage."""

from __future__ import annotations

import getpass
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import click

from nvitk.cluster.remote_transfer import (
    ensure_remote_dir,
    remote_path_exists,
    resolve_cluster_host,
    sftp_session,
    upload_directory,
)
from nvitk.core.logger import Logger
from nvitk.db.xnat_config import XnatConnectionConfig
from nvitk.pipes.qvtpy.util.io.paths import QvtpyPaths

log = Logger()


def prompt_ssh_credentials(
    *,
    remote_host: str | None = None,
    remote_user: str | None = None,
    host_aliases: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    """Prompt for cluster SSH credentials; return ``(host, user, password)``."""
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            "SGE + XNAT cluster sync requires Paramiko (pip install paramiko)."
        ) from exc

    aliases = host_aliases or {}
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host = aliases.get(host_key, host_key)
    user = remote_user or click.prompt("SSH user")
    password = getpass.getpass("SSH password: ")
    return host, user, password


def verify_ssh_connection(
    *,
    host: str,
    user: str,
    password: str,
    port: int = 22,
) -> None:
    """Open and close an SSH/SFTP session to confirm cluster access."""
    host_resolved = resolve_cluster_host(host)
    log.info(f"Verifying SSH connection to {user}@{host_resolved} ...")
    with sftp_session(host=host, user=user, password=password, port=port):
        log.info(f"SSH connection OK ({host_resolved})")


def remote_subject_dicom_dir(remote_dicom_root: Path, subject: str) -> str:
    """POSIX path ``<remote_dicom_root>/<subject>`` on the cluster."""
    return f"{str(remote_dicom_root).rstrip('/')}/{subject}"


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


def remove_local_subject_dicoms(local_subject_dir: Path) -> None:
    """Delete a subject DICOM tree from local staging storage."""
    local_subject_dir = Path(local_subject_dir)
    if not local_subject_dir.exists():
        return
    log.info(f"Removing local DICOM staging: {local_subject_dir}")
    shutil.rmtree(local_subject_dir)


def upload_directory_sftp(
    sftp: Any,
    local_root: Path,
    remote_root: str,
) -> None:
    """Recursively upload *local_root* to *remote_root* using an open SFTP handle."""
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


def upload_subject_dicoms_sftp(
    sftp: Any,
    local_subject_dir: Path,
    remote_dicom_root: Path,
    subject: str,
) -> None:
    """Upload one subject DICOM tree via an existing SFTP session."""
    local_subject_dir = Path(local_subject_dir)
    if not local_subject_has_dicoms(local_subject_dir):
        raise FileNotFoundError(f"No local DICOM data under {local_subject_dir}")
    remote_subj = remote_subject_dicom_dir(remote_dicom_root, subject)
    log.info(f"[{subject}] uploading DICOM -> {remote_subj}")
    upload_directory_sftp(sftp, local_subject_dir, remote_subj)


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


def stream_subjects_xnat_to_cluster(
    subjects: list[str],
    *,
    local_paths: QvtpyPaths,
    cluster_paths: QvtpyPaths,
    xnat_config: XnatConnectionConfig,
    sequences: Iterable[str],
    host: str,
    user: str,
    password: str,
    skip_existing: bool = False,
    skip_existing_downloads: bool = False,
    delete_local_after_upload: bool = True,
    port: int = 22,
) -> dict[str, dict[str, list[Path]]]:
    """Per subject: XNAT download → cluster SFTP upload → delete local DICOM staging.

    SSH is verified before any XNAT download. A single SFTP session is reused for
    the full subject loop.
    """
    from nvitk.db.xnat import connect_xnat
    from nvitk.pipes.qvtpy.stage0_download import (
        DEFAULT_SEQUENCES,
        collect_local_subject_dicoms,
        download_subject,
        local_subject_dicoms_complete,
    )

    seq_set = set(sequences) if sequences else set(DEFAULT_SEQUENCES)
    local_root = local_paths.dicom_root
    local_root.mkdir(parents=True, exist_ok=True)

    verify_ssh_connection(host=host, user=user, password=password, port=port)

    log.info(
        f"Streaming XNAT -> cluster | subjects={len(subjects)} | "
        f"local staging={local_root} -> cluster={cluster_paths.dicom_root}"
    )

    results: dict[str, dict[str, list[Path]]] = {}
    uploaded = 0
    skipped_remote = 0
    failed: list[str] = []

    with sftp_session(host=host, user=user, password=password, port=port) as (_ssh, sftp):
        with connect_xnat(xnat_config) as xnat_session:
            for subject in subjects:
                local_dir = local_paths.subject_dicom_dir(subject)
                remote_subj = remote_subject_dicom_dir(cluster_paths.dicom_root, subject)

                try:
                    if skip_existing and remote_subject_has_dicoms(sftp, remote_subj):
                        log.info(
                            f"[{subject}] cluster DICOM present — skip download/upload"
                        )
                        skipped_remote += 1
                        results[subject] = {seq: [] for seq in seq_set}
                        continue

                    log.info(f"[{subject}] XNAT download -> {local_dir}")
                    if skip_existing_downloads and local_subject_dicoms_complete(
                        local_root, subject, seq_set
                    ):
                        log.info(
                            f"[{subject}] all requested sequences present locally "
                            "— skip XNAT download"
                        )
                        results[subject] = collect_local_subject_dicoms(
                            local_root, subject, seq_set
                        )
                    else:
                        results[subject] = download_subject(
                            xnat_session,
                            xnat_config.project,
                            subject,
                            dicom_root=local_root,
                            sequences=seq_set,
                            skip_existing=skip_existing_downloads,
                        )

                    if not local_subject_has_dicoms(local_dir):
                        log.warning(f"[{subject}] no DICOM files downloaded — skip upload")
                        failed.append(subject)
                        continue

                    upload_subject_dicoms_sftp(
                        sftp,
                        local_dir,
                        cluster_paths.dicom_root,
                        subject,
                    )
                    uploaded += 1

                    if delete_local_after_upload:
                        remove_local_subject_dicoms(local_dir)

                except LookupError as exc:
                    log.warning(f"[{subject}] {exc}")
                    results[subject] = {seq: [] for seq in seq_set}
                    failed.append(subject)
                except Exception as exc:
                    log.warning(f"[{subject}] stream failed: {exc}")
                    results[subject] = {seq: [] for seq in seq_set}
                    failed.append(subject)
                    if local_dir.is_dir():
                        try:
                            remove_local_subject_dicoms(local_dir)
                        except OSError:
                            pass

    log.info(
        f"Stream complete: uploaded={uploaded} skipped_remote={skipped_remote} "
        f"failed={len(failed)} total={len(subjects)}"
    )
    if failed:
        preview = ", ".join(failed[:8])
        suffix = f" ... (+{len(failed) - 8} more)" if len(failed) > 8 else ""
        log.info(f"  subjects with errors or empty download: {preview}{suffix}")

    return results


def remote_subject_results_dir(remote_results_root: Path | str, subject: str) -> str:
    """POSIX path ``<remote_results_root>/<subject>`` on the cluster."""
    return f"{str(remote_results_root).rstrip('/')}/{subject}"


def fetch_subject_results_sftp(
    sftp: Any,
    *,
    remote_results_root: Path | str,
    local_subject_root: Path,
    subject: str,
) -> dict[str, int]:
    """Download remote ``eicab/`` and ``qvtpy/`` into *local_subject_root*.

    Returns ``{resource_label: n_files_downloaded}``.
    """
    from nvitk.cluster.remote_transfer import download_directory_sftp, remote_path_exists
    from nvitk.pipes.qvtpy import config as qcfg

    remote_subj = remote_subject_results_dir(remote_results_root, subject)
    local_subject_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for label in (qcfg.STAGE1_EICAB_DIR, qcfg.QVT_SUBDIR):
        remote_dir = f"{remote_subj}/{label}"
        local_dir = local_subject_root / label
        if not remote_path_exists(sftp, remote_dir):
            counts[label] = 0
            continue
        log.info(f"[{subject}] SFTP fetch {remote_dir} -> {local_dir}")
        counts[label] = download_directory_sftp(sftp, remote_dir, local_dir)
    return counts


def remove_local_subject_results(local_subject_root: Path) -> None:
    """Delete a staged subject results tree from local storage."""
    local_subject_root = Path(local_subject_root)
    if not local_subject_root.exists():
        return
    log.info(f"Removing local results staging: {local_subject_root}")
    shutil.rmtree(local_subject_root)


__all__ = [
    "fetch_subject_results_sftp",
    "local_subject_has_dicoms",
    "prompt_ssh_credentials",
    "remote_subject_dicom_dir",
    "remote_subject_results_dir",
    "remove_local_subject_dicoms",
    "remove_local_subject_results",
    "stream_subjects_xnat_to_cluster",
    "upload_subject_dicoms",
    "upload_subjects_dicoms",
    "verify_ssh_connection",
]
