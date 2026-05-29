"""Poll cluster jobs via ``output/.done`` marker (SFTP)."""

from __future__ import annotations

import json
from typing import Any, Callable

from nvitk.cluster.remote_transfer import read_remote_text, remote_path_exists, sftp_session
from nvitk.gui.sge_models import SgeDoneMarker, SgePendingJob, SgeConnection, remote_done_path

try:
    from qtpy.QtCore import QObject, QTimer, Signal
except Exception:
    QObject = object
    QTimer = None

    class Signal:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def emit(self, *args: Any, **kwargs: Any) -> None:
            pass

        def connect(self, *args: Any, **kwargs: Any) -> None:
            pass


def read_done_marker(
    *,
    host: str,
    user: str,
    password: str,
    remote_job_root: str,
    port = 22,
) -> SgeDoneMarker | None:
    """Return parsed ``output/.done`` when present, else ``None``."""
    done_path = remote_done_path(remote_job_root)
    with sftp_session(host=host, user=user, password=password, port=port) as (_c, sftp):
        if not remote_path_exists(sftp, done_path):
            return None
        raw = read_remote_text(sftp, done_path)
    data = json.loads(raw)
    if not isinstance(data, dict):
        return None
    return SgeDoneMarker.from_dict(data)


class SgeJobMonitor(QObject):
    """Background timer that polls ``output/.done`` for tracked GUI jobs."""

    job_finished = Signal(str, object)
    job_failed = Signal(str, object)

    def __init__(self, parent: QObject | None = None, interval_ms: int = 15000) -> None:
        super().__init__(parent)
        self._jobs: dict[str, SgePendingJob] = {}
        self._timer = QTimer(self) if QTimer is not None else None
        if self._timer is not None:
            self._timer.setInterval(max(5000, int(interval_ms)))
            self._timer.timeout.connect(self._poll)

    def track(self, job: SgePendingJob) -> None:
        self._jobs[job.job_id] = job
        if self._timer is not None and not self._timer.isActive():
            self._timer.start()

    def pending_jobs(self) -> list[SgePendingJob]:
        return list(self._jobs.values())

    def _poll(self) -> None:
        if not self._jobs:
            if self._timer is not None:
                self._timer.stop()
            return
        finished = []
        for job_id, job in list(self._jobs.items()):
            try:
                done = read_done_marker(
                    host=job.connection.host,
                    user=job.connection.user,
                    password=job.connection.password,
                    remote_job_root=job.remote_job_root,
                )
            except Exception:
                continue
            if done is None:
                continue
            job.done_payload = done.to_dict()
            job.status = "done" if done.exit_code == 0 else "failed"
            finished.append(job_id)
            if done.exit_code == 0:
                self.job_finished.emit(job_id, done)
            else:
                self.job_failed.emit(job_id, done)
        for job_id in finished:
            self._jobs.pop(job_id, None)
        if not self._jobs and self._timer is not None:
            self._timer.stop()


def register_sge_monitor(
    app_state: dict[str, Any],
    *,
    on_finished = None,
    on_failed = None,
) -> SgeJobMonitor:
    """Create or return the shared :class:`SgeJobMonitor` stored in *app_state*."""
    mon = app_state.get("_sge_monitor")
    if isinstance(mon, SgeJobMonitor):
        return mon
    mon = SgeJobMonitor()
    if on_finished is not None:
        mon.job_finished.connect(lambda jid, done: on_finished(jid, done))
    if on_failed is not None:
        mon.job_failed.connect(lambda jid, done: on_failed(jid, done))
    app_state["_sge_monitor"] = mon
    return mon


def store_pending_job(app_state: dict[str, Any], job: SgePendingJob) -> None:
    bucket = app_state.setdefault("sge_pending_jobs", [])
    if not isinstance(bucket, list):
        bucket = []
        app_state["sge_pending_jobs"] = bucket
    serial = {
        "job_id": job.job_id,
        "tool_id": job.tool_id,
        "remote_job_root": job.remote_job_root,
        "output_name": job.output_name,
        "status": job.status,
        "done_payload": job.done_payload,
        "local_download_dir": job.local_download_dir,
        "host": job.connection.host,
        "user": job.connection.user,
    }
    bucket[:] = [x for x in bucket if x.get("job_id") != job.job_id]
    bucket.append(serial)
    app_state["sge_last_connection"] = {
        "host": job.connection.host,
        "user": job.connection.user,
        "password": job.connection.password,
        "remote_job_root": job.remote_job_root,
    }


def update_pending_job_status(
    app_state: dict[str, Any],
    job_id: str,
    *,
    status: str,
    done_payload = None,
    local_download_dir = None,
) -> None:
    bucket = app_state.get("sge_pending_jobs") or []
    if not isinstance(bucket, list):
        return
    for row in bucket:
        if row.get("job_id") == job_id:
            row["status"] = status
            if done_payload is not None:
                row["done_payload"] = done_payload
            if local_download_dir is not None:
                row["local_download_dir"] = local_download_dir
            break


def resolve_session_import(
    app_state: dict[str, Any],
    *,
    job_id = None,
) -> tuple[SgeConnection, str, str | None]:
    """Return ``(connection, remote_job_root, job_id)`` from the current GUI session."""
    last = app_state.get("sge_last_connection") or {}
    if not isinstance(last, dict):
        last = {}
    conn = SgeConnection(
        host=str(last.get("host") or "").strip(),
        user=str(last.get("user") or "").strip(),
        password=str(last.get("password") or ""),
    )
    if not conn.host or not conn.user:
        raise ValueError(
            "No SGE session credentials in this Napari session. "
            "Submit a job with Run SGE first."
        )

    bucket = app_state.get("sge_pending_jobs") or []
    if not isinstance(bucket, list):
        bucket = []

    remote_root = ""
    resolved_job_id = job_id

    if job_id:
        for row in bucket:
            if row.get("job_id") == job_id:
                remote_root = str(row.get("remote_job_root") or "").strip().rstrip("/")
                break
    else:
        for row in reversed(bucket):
            if row.get("local_download_dir"):
                continue
            st = str(row.get("status") or "")
            if st in ("done", "submitted", "failed"):
                remote_root = str(row.get("remote_job_root") or "").strip().rstrip("/")
                resolved_job_id = str(row.get("job_id") or "") or None
                if st == "done":
                    break

    if not remote_root:
        remote_root = str(last.get("remote_job_root") or "").strip().rstrip("/")

    if not remote_root:
        raise ValueError("No remote job directory found for import.")

    return conn, remote_root, resolved_job_id


__all__ = [
    "SgeJobMonitor",
    "read_done_marker",
    "register_sge_monitor",
    "resolve_session_import",
    "store_pending_job",
    "update_pending_job_status",
]
