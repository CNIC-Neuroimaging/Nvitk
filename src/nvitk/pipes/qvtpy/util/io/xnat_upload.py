"""QVTpy XNAT upload: completion checks and per-subject orchestration."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from nvitk.cluster.remote_transfer import remote_path_exists, sftp_session
from nvitk.core.logger import Logger
from nvitk.db.xnat import connect_xnat
from nvitk.db.xnat_config import XnatConnectionConfig
from nvitk.db.xnat_upload import (
    iter_upload_files,
    resolve_subject_experiment,
    upload_directory_to_xnat_resource,
    xnat_resource_has_files,
)
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.stages import (
    STAGE_CENTERLINE,
    STAGE_EICAB,
    STAGE_LOC,
    STAGE_MEASURE,
    STAGE_MORPHOMETRICS,
    STAGE_REG,
    STAGE_SEG,
    STAGE_SEG_T,
)
from nvitk.pipes.qvtpy.stage1_eicab import _output_has_segmentation
from nvitk.pipes.qvtpy.util.io.qc_report import check_subject_stages, parse_stages

from nvitk.pipes.qvtpy.util.io.cluster_upload import (
    fetch_subject_results_sftp,
    remote_subject_results_dir,
)

log = Logger()

ResultsSource = Literal["local", "cluster"]

XNAT_RESOURCE_EICAB = cfg.STAGE1_EICAB_DIR
XNAT_RESOURCE_QVTPY = cfg.QVT_SUBDIR

DEFAULT_XNAT_UPLOAD_STAGES: tuple[str, ...] = (
    STAGE_REG,
    STAGE_CENTERLINE,
    STAGE_SEG,
    STAGE_LOC,
    STAGE_MEASURE,
    STAGE_MORPHOMETRICS,
)

_ALLOWED_REQUIRE_STAGES: frozenset[str] = frozenset(
    {
        STAGE_REG,
        STAGE_CENTERLINE,
        STAGE_SEG,
        STAGE_SEG_T,
        STAGE_LOC,
        STAGE_MEASURE,
        STAGE_MORPHOMETRICS,
    }
)


class UploadStatus(str, Enum):
    UPLOADED = "uploaded"
    SKIPPED = "skipped"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    DRY_RUN = "dry_run"


@dataclass
class ResourceUploadOutcome:
    resource: str
    status: UploadStatus
    detail: str = ""
    n_files: int = 0


@dataclass
class UploadResult:
    subject: str
    experiment_label: str = ""
    eicab: ResourceUploadOutcome | None = None
    qvtpy: ResourceUploadOutcome | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        if self.error:
            return False
        outcomes = [o for o in (self.eicab, self.qvtpy) if o is not None]
        return all(o.status != UploadStatus.ERROR for o in outcomes)


def parse_require_stages(spec: str) -> list[str]:
    """Parse ``--require-stages`` into canonical qvtpy stage ids."""
    stages = parse_stages(spec)
    invalid = [s for s in stages if s not in _ALLOWED_REQUIRE_STAGES]
    if invalid:
        allowed = ", ".join(sorted(_ALLOWED_REQUIRE_STAGES))
        raise ValueError(
            f"Invalid --require-stages for XNAT upload: {', '.join(invalid)}. "
            f"Allowed: {allowed}"
        )
    return stages


def eicab_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.STAGE1_EICAB_DIR


def qvtpy_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR


def eicab_complete(output_root: Path, subject: str) -> bool:
    return _output_has_segmentation(eicab_dir(output_root, subject))


def qvtpy_stage_complete(output_root: Path, subject: str, stage_id: str) -> bool:
    checks = check_subject_stages(subject, [stage_id], results_root=output_root)
    return bool(checks) and checks[0].complete


def qvtpy_all_required_complete(
    output_root: Path,
    subject: str,
    required_stages: list[str],
) -> tuple[bool, list[str]]:
    checks = check_subject_stages(subject, required_stages, results_root=output_root)
    missing = [c.stage for c in checks if not c.complete]
    return not missing, missing


def upload_subject_to_xnat(
    subject: str,
    *,
    output_root: Path,
    xnat_session: Any,
    project_id: str,
    required_stages: list[str],
    upload_eicab: bool = True,
    upload_qvtpy: bool = True,
    overwrite: bool = False,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> UploadResult:
    """Upload eicab/qvtpy resources for one subject."""
    result = UploadResult(subject=subject)
    try:
        project = xnat_session.projects[project_id]
        experiment, exp_label = resolve_subject_experiment(project, subject)
        result.experiment_label = exp_label
        log.info(f"[{subject}] experiment={exp_label}")

        if upload_eicab:
            result.eicab = _upload_resource(
                experiment,
                XNAT_RESOURCE_EICAB,
                eicab_dir(output_root, subject),
                complete=eicab_complete(output_root, subject),
                incomplete_detail="stage1 eICAB outputs incomplete",
                overwrite=overwrite,
                skip_existing=skip_existing,
                dry_run=dry_run,
            )

        if upload_qvtpy:
            complete, missing = qvtpy_all_required_complete(
                output_root, subject, required_stages
            )
            detail = (
                f"incomplete stages: {', '.join(missing)}"
                if missing
                else ""
            )
            result.qvtpy = _upload_resource(
                experiment,
                XNAT_RESOURCE_QVTPY,
                qvtpy_dir(output_root, subject),
                complete=complete,
                incomplete_detail=detail or "qvtpy outputs incomplete",
                overwrite=overwrite,
                skip_existing=skip_existing,
                dry_run=dry_run,
            )
    except Exception as exc:
        result.error = str(exc)
        log.exception(f"[{subject}] XNAT upload failed: {exc}")
    return result


def _upload_resource(
    experiment: Any,
    resource_label: str,
    local_dir: Path,
    *,
    complete: bool,
    incomplete_detail: str,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool,
) -> ResourceUploadOutcome:
    if not complete:
        log.info(f"skip {resource_label}: {incomplete_detail}")
        return ResourceUploadOutcome(
            resource=resource_label,
            status=UploadStatus.INCOMPLETE,
            detail=incomplete_detail,
        )

    if not local_dir.is_dir():
        detail = f"local directory missing: {local_dir}"
        log.warning(f"skip {resource_label}: {detail}")
        return ResourceUploadOutcome(
            resource=resource_label,
            status=UploadStatus.INCOMPLETE,
            detail=detail,
        )

    n_files = len(iter_upload_files(local_dir))
    if n_files == 0:
        detail = f"no files under {local_dir}"
        log.warning(f"skip {resource_label}: {detail}")
        return ResourceUploadOutcome(
            resource=resource_label,
            status=UploadStatus.INCOMPLETE,
            detail=detail,
        )

    if skip_existing and not overwrite and xnat_resource_has_files(experiment, resource_label):
        detail = "XNAT resource already has files (--skip-existing)"
        log.info(f"skip {resource_label}: {detail}")
        return ResourceUploadOutcome(
            resource=resource_label,
            status=UploadStatus.SKIPPED,
            detail=detail,
            n_files=n_files,
        )

    try:
        uploaded = upload_directory_to_xnat_resource(
            experiment,
            resource_label,
            local_dir,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        status = UploadStatus.DRY_RUN if dry_run else UploadStatus.UPLOADED
        return ResourceUploadOutcome(
            resource=resource_label,
            status=status,
            detail="ok",
            n_files=uploaded,
        )
    except Exception as exc:
        log.exception(f"resource {resource_label} upload failed: {exc}")
        return ResourceUploadOutcome(
            resource=resource_label,
            status=UploadStatus.ERROR,
            detail=str(exc),
            n_files=n_files,
        )


@dataclass
class BatchUploadSummary:
    subjects: int = 0
    eicab_uploaded: int = 0
    eicab_skipped: int = 0
    eicab_incomplete: int = 0
    qvtpy_uploaded: int = 0
    qvtpy_skipped: int = 0
    qvtpy_incomplete: int = 0
    errors: int = 0
    results: list[UploadResult] = field(default_factory=list)


def _tally_resource(summary: BatchUploadSummary, outcome: ResourceUploadOutcome | None, kind: str) -> None:
    if outcome is None:
        return
    if outcome.status in (UploadStatus.UPLOADED, UploadStatus.DRY_RUN):
        if kind == "eicab":
            summary.eicab_uploaded += 1
        else:
            summary.qvtpy_uploaded += 1
    elif outcome.status == UploadStatus.SKIPPED:
        if kind == "eicab":
            summary.eicab_skipped += 1
        else:
            summary.qvtpy_skipped += 1
    elif outcome.status == UploadStatus.INCOMPLETE:
        if kind == "eicab":
            summary.eicab_incomplete += 1
        else:
            summary.qvtpy_incomplete += 1
    elif outcome.status == UploadStatus.ERROR:
        summary.errors += 1


def _upload_subject_from_staging(
    subject: str,
    *,
    staging_root: Path,
    xnat_session: Any,
    project_id: str,
    required_stages: list[str],
    upload_eicab: bool,
    upload_qvtpy: bool,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool,
) -> UploadResult:
    return upload_subject_to_xnat(
        subject,
        output_root=staging_root,
        xnat_session=xnat_session,
        project_id=project_id,
        required_stages=required_stages,
        upload_eicab=upload_eicab,
        upload_qvtpy=upload_qvtpy,
        overwrite=overwrite,
        skip_existing=skip_existing,
        dry_run=dry_run,
    )


def _cluster_dry_run_subject(
    subject: str,
    *,
    sftp: Any,
    remote_results_root: Path,
    upload_eicab: bool,
    upload_qvtpy: bool,
) -> None:
    remote_subj = remote_subject_results_dir(remote_results_root, subject)
    if upload_eicab:
        remote_eicab = f"{remote_subj}/{XNAT_RESOURCE_EICAB}"
        exists = remote_path_exists(sftp, remote_eicab)
        log.info(
            f"[{subject}] [dry-run] would fetch eicab from {remote_eicab} "
            f"(exists={exists})"
        )
    if upload_qvtpy:
        remote_qvtpy = f"{remote_subj}/{XNAT_RESOURCE_QVTPY}"
        exists = remote_path_exists(sftp, remote_qvtpy)
        log.info(
            f"[{subject}] [dry-run] would fetch qvtpy from {remote_qvtpy} "
            f"(exists={exists})"
        )


def run_xnat_upload(
    subjects: list[str],
    *,
    output_root: Path,
    xnat_config: XnatConnectionConfig,
    required_stages: list[str] | None = None,
    upload_eicab: bool = True,
    upload_qvtpy: bool = True,
    overwrite: bool = False,
    skip_existing: bool = True,
    dry_run: bool = False,
    results_source: ResultsSource = "local",
    ssh_host: str | None = None,
    ssh_user: str | None = None,
    ssh_password: str | None = None,
    remote_results_root: Path | None = None,
) -> BatchUploadSummary:
    """Upload qvtpy/eicab results for *subjects* to XNAT."""
    stages = list(required_stages or DEFAULT_XNAT_UPLOAD_STAGES)
    summary = BatchUploadSummary()
    summary.subjects = len(subjects)
    cluster_mode = results_source == "cluster"

    if cluster_mode:
        if remote_results_root is None:
            raise ValueError("remote_results_root is required when results_source='cluster'")
        if not ssh_host or not ssh_user or not ssh_password:
            raise ValueError(
                "SSH host, user, and password are required when results_source='cluster'"
            )

    log.info(f"qvtpy XNAT upload | subjects={len(subjects)} project={xnat_config.project}")
    log.info(f"  results_source : {results_source}")
    if cluster_mode:
        log.info(f"  remote_results : {remote_results_root}")
        log.info(f"  ssh            : {ssh_user}@{ssh_host}")
    else:
        log.info(f"  output_root    : {output_root}")
    log.info(f"  server         : {xnat_config.server}")
    log.info(f"  require        : {', '.join(stages)}")
    log.info(f"  upload         : eicab={upload_eicab} qvtpy={upload_qvtpy}")
    log.info(
        f"  mode           : dry_run={dry_run} skip_existing={skip_existing} overwrite={overwrite}"
    )

    if cluster_mode and dry_run:
        for subject in subjects:
            with sftp_session(
                host=ssh_host,
                user=ssh_user,
                password=ssh_password,
            ) as (_ssh, sftp):
                _cluster_dry_run_subject(
                    subject,
                    sftp=sftp,
                    remote_results_root=remote_results_root,
                    upload_eicab=upload_eicab,
                    upload_qvtpy=upload_qvtpy,
                )
        return summary

    with connect_xnat(xnat_config) as session:
        for subject in subjects:
            if cluster_mode:
                with tempfile.TemporaryDirectory(
                    prefix=f"nvitk-xnat-{subject}-"
                ) as tmp:
                    staging_root = Path(tmp)
                    local_subject_root = staging_root / subject
                    with sftp_session(
                        host=ssh_host,
                        user=ssh_user,
                        password=ssh_password,
                    ) as (_ssh, sftp):
                        fetch_subject_results_sftp(
                            sftp,
                            remote_results_root=remote_results_root,
                            local_subject_root=local_subject_root,
                            subject=subject,
                        )
                    result = _upload_subject_from_staging(
                        subject,
                        staging_root=staging_root,
                        xnat_session=session,
                        project_id=xnat_config.project,
                        required_stages=stages,
                        upload_eicab=upload_eicab,
                        upload_qvtpy=upload_qvtpy,
                        overwrite=overwrite,
                        skip_existing=skip_existing,
                        dry_run=dry_run,
                    )
            else:
                result = upload_subject_to_xnat(
                    subject,
                    output_root=output_root,
                    xnat_session=session,
                    project_id=xnat_config.project,
                    required_stages=stages,
                    upload_eicab=upload_eicab,
                    upload_qvtpy=upload_qvtpy,
                    overwrite=overwrite,
                    skip_existing=skip_existing,
                    dry_run=dry_run,
                )

            summary.results.append(result)
            if result.error:
                summary.errors += 1
            _tally_resource(summary, result.eicab, "eicab")
            _tally_resource(summary, result.qvtpy, "qvtpy")

    log.info(
        "XNAT upload summary: "
        f"eicab uploaded={summary.eicab_uploaded} skipped={summary.eicab_skipped} "
        f"incomplete={summary.eicab_incomplete}; "
        f"qvtpy uploaded={summary.qvtpy_uploaded} skipped={summary.qvtpy_skipped} "
        f"incomplete={summary.qvtpy_incomplete}; errors={summary.errors}"
    )
    return summary


__all__ = [
    "BatchUploadSummary",
    "DEFAULT_XNAT_UPLOAD_STAGES",
    "ResourceUploadOutcome",
    "UploadResult",
    "UploadStatus",
    "eicab_complete",
    "eicab_dir",
    "parse_require_stages",
    "qvtpy_all_required_complete",
    "qvtpy_dir",
    "qvtpy_stage_complete",
    "run_xnat_upload",
    "upload_subject_to_xnat",
    "ResultsSource",
]
