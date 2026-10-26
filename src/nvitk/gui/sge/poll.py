"""Follow the cluster jobs the GUI sent: ``output/.done`` markers (read over SSH) and
``qstat``, with each job shown in the activity indicator while it waits."""

from __future__ import annotations

import json
from typing import Any, Callable

from nvitk.cluster.remote_transfer import cluster_session, read_remote_text, remote_path_exists
from nvitk.gui.sge.models import SgeDoneMarker, SgePendingJob, SgeConnection, remote_done_path

try:
    from qtpy.QtCore import QObject, QTimer, Signal
except Exception:
    QObject = object
    QTimer = None

    class Signal:
        """No-op stand-in for ``qtpy.QtCore.Signal`` when Qt isn't installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept and discard any Signal type arguments."""
            pass

        def emit(self, *args: Any, **kwargs: Any) -> None:
            """No-op emit."""
            pass

        def connect(self, *args: Any, **kwargs: Any) -> None:
            """No-op connect."""
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
    with cluster_session(host=host, user=user, password=password, port=port) as session:
        if not remote_path_exists(session, done_path):
            return None
        raw = read_remote_text(session, done_path)
    data = json.loads(raw)
    if not isinstance(data, dict):
        return None
    return SgeDoneMarker.from_dict(data)


#: Seconds between two ``qstat`` queries (the ``.done`` markers are polled every 5 s).
_QSTAT_EVERY_S = 15.0


class _PollRelay(QObject):
    """Hands a background poll's results to the GUI thread (signals queue across threads)."""

    polled = Signal(object)


def _connection_key(conn: SgeConnection) -> tuple[str, str]:
    return (str(conn.host), str(conn.user))


class SgeJobMonitor(QObject):
    """Background timer that follows the cluster jobs the GUI sent.

    Two kinds: the Imaging tab's jobs, finished when their ``output/.done`` appears
    (:meth:`track`; their results are then imported), and jobs only known by their
    scheduler ids (a voxelwise analysis) finished when those leave ``qstat``
    (:meth:`watch_queue`). Each can carry an activity-indicator task, kept up to date
    with the job's state (submitted, queued, running...) until it ends.

    The checks open SSH sessions, so they run on a worker thread; the results come
    back here, on the GUI thread, where the signals are emitted.
    """

    job_finished = Signal(str, object)
    job_failed = Signal(str, object)
    queue_finished = Signal(str)

    def __init__(self, parent: QObject | None = None, interval_ms: int = 5000) -> None:
        """Set up the (initially stopped) polling timer at *interval_ms* (min 5000ms)."""
        super().__init__(parent)
        self._jobs: dict[str, SgePendingJob] = {}
        self._tasks: dict[str, Any] = {}
        #: key -> {"connection", "ids", "label", "task", "on_done"}: qstat-only jobs.
        self._watches: dict[str, dict[str, Any]] = {}
        #: Scheduler job id -> its last ``qstat`` state ("" once it left the queue).
        self._states: dict[str, str] = {}
        self._polling = False
        self._last_qstat = 0.0
        self._relay = _PollRelay(self) if QTimer is not None else None
        if self._relay is not None:
            self._relay.polled.connect(self._on_polled)
        self._timer = QTimer(self) if QTimer is not None else None
        if self._timer is not None:
            self._timer.setInterval(max(5000, int(interval_ms)))
            self._timer.timeout.connect(self._poll)

    # what is followed

    def track(self, job: SgePendingJob, task: Any = None) -> None:
        """Follow *job* until its ``.done`` marker appears; *task* (activity) shows it meanwhile."""
        self._jobs[job.job_id] = job
        if task is not None:
            self._tasks[job.job_id] = task
            self._describe_job(job)
        self._start()

    def watch_queue(self, key: str, connection: SgeConnection, sge_job_ids: list[str], *,
                    label: str, task: Any = None, on_done: Callable[[], None] | None = None) -> None:
        """Follow scheduler jobs *sge_job_ids* until none is left in ``qstat``."""
        ids = [str(i) for i in sge_job_ids if str(i).isdigit()]
        if not ids:
            return
        self._watches[key] = {"connection": connection, "ids": ids, "label": label, "task": task,
                              "on_done": on_done}
        self._last_qstat = 0.0          # ask the queue on the next poll
        self._describe_watch(self._watches[key])
        self._start()

    def pending_jobs(self) -> list[SgePendingJob]:
        """Currently tracked, not-yet-finished jobs."""
        return list(self._jobs.values())

    def _start(self) -> None:
        if self._timer is not None and not self._timer.isActive():
            self._timer.start()

    # messages

    def _state_of(self, ids: list[str]) -> str:
        from nvitk.cluster.sge_chunk import describe_sge_state

        if not ids:
            return "waiting for the cluster"
        states = [self._states.get(i) for i in ids]
        if all(st is None for st in states):
            return "submitted"
        live = [st for st in states if st]
        if not live:
            return "finishing"
        words = [describe_sge_state(st) for st in live]
        for word in ("in error", "running", "queued", "held", "suspended"):
            hit = next((w for w in words if w.startswith(word)), None)
            if hit:
                return hit
        return words[0]

    @staticmethod
    def _ids_text(ids: list[str]) -> str:
        if not ids:
            return ""
        return f" (job {ids[0]})" if len(ids) == 1 else f" ({len(ids)} jobs)"

    def _describe_job(self, job: SgePendingJob) -> None:
        task = self._tasks.get(job.job_id)
        if task is not None:
            label = job.label or job.tool_id
            task.update(message=f"Cluster: {label} — {self._state_of(job.sge_job_ids)}"
                                f"{self._ids_text(job.sge_job_ids)}")

    def _describe_watch(self, watch: dict[str, Any]) -> None:
        task = watch.get("task")
        if task is not None:
            task.update(message=f"Cluster: {watch['label']} — {self._state_of(watch['ids'])}"
                                f"{self._ids_text(watch['ids'])}")

    # polling

    def _poll(self) -> None:
        """Start one round of checks on a worker thread (one at a time)."""
        import threading
        import time

        if not self._jobs and not self._watches:
            if self._timer is not None:
                self._timer.stop()
            return
        if self._polling:
            return
        jobs = list(self._jobs.values())
        connections: dict[tuple[str, str], SgeConnection] = {}
        if time.monotonic() - self._last_qstat >= _QSTAT_EVERY_S:
            for job in jobs:
                if job.sge_job_ids:
                    connections[_connection_key(job.connection)] = job.connection
            for watch in self._watches.values():
                connections[_connection_key(watch["connection"])] = watch["connection"]
        self._polling = True

        def work() -> None:
            results: dict[str, Any] = {"done": {}, "states": {}, "asked": list(connections)}
            for job in jobs:
                try:
                    results["done"][job.job_id] = read_done_marker(
                        host=job.connection.host,
                        user=job.connection.user,
                        password=job.connection.password,
                        remote_job_root=job.remote_job_root,
                    )
                except Exception:  # noqa: BLE001 (unreachable this time: ask again next poll)
                    pass
            for key, conn in connections.items():
                try:
                    from nvitk.cluster.sge_chunk import query_sge_job_states

                    results["states"][key] = query_sge_job_states(conn.host, conn.user, conn.password)
                except Exception:  # noqa: BLE001
                    pass
            try:
                if self._relay is not None:
                    self._relay.polled.emit(results)
            except RuntimeError:  # the monitor went with the window
                pass

        if self._relay is None:
            work()
            return
        threading.Thread(target=work, name="nvitk-sge-poll", daemon=True).start()

    def _on_polled(self, results: dict[str, Any]) -> None:
        """Act on a poll's results (GUI thread)."""
        import time

        self._polling = False
        answered = results.get("states") or {}
        if results.get("asked"):
            self._last_qstat = time.monotonic()
        for key, states in answered.items():
            followed = [i for job in self._jobs.values() if _connection_key(job.connection) == key
                        for i in job.sge_job_ids]
            followed += [i for w in self._watches.values() if _connection_key(w["connection"]) == key
                         for i in w["ids"]]
            for sge_id in followed:
                self._states[sge_id] = str(states.get(sge_id, ""))

        for job_id, done in (results.get("done") or {}).items():
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if done is None:
                self._describe_job(job)
                continue
            job.done_payload = done.to_dict()
            job.status = "done" if done.exit_code == 0 else "failed"
            self._jobs.pop(job_id, None)
            task = self._tasks.pop(job_id, None)
            label = job.label or job.tool_id
            if done.exit_code == 0:
                if task is not None:
                    task.update(message=f"Cluster: {label} — done, importing the results")
                self.job_finished.emit(job_id, done)
            else:
                if task is not None:
                    task.update(message=f"Cluster: {label} — failed (exit {done.exit_code})")
                self.job_failed.emit(job_id, done)
            if task is not None:
                task.finish(ok=done.exit_code == 0)

        for key, watch in list(self._watches.items()):
            ckey = _connection_key(watch["connection"])
            if ckey in answered and not any(answered[ckey].get(i) for i in watch["ids"]):
                self._watches.pop(key, None)
                task = watch.get("task")
                if task is not None:
                    task.update(message=f"Cluster: {watch['label']} — finished")
                    task.finish(ok=True)
                if watch.get("on_done") is not None:
                    try:
                        watch["on_done"]()
                    except Exception:  # noqa: BLE001 (a callback must not stop the monitor)
                        pass
                self.queue_finished.emit(key)
            else:
                self._describe_watch(watch)

        if not self._jobs and not self._watches and self._timer is not None:
            self._timer.stop()

    def stop(self) -> None:
        """Stop following everything (the window is closing)."""
        if self._timer is not None:
            self._timer.stop()
        for task in list(self._tasks.values()) + [w.get("task") for w in self._watches.values()]:
            if task is not None:
                try:
                    task.finish(ok=True)
                except Exception:  # noqa: BLE001
                    pass
        self._jobs.clear()
        self._tasks.clear()
        self._watches.clear()


def register_sge_monitor(
    app_state: dict[str, Any],
    *,
    on_finished = None,
    on_failed = None,
    parent: QObject | None = None,
) -> SgeJobMonitor:
    """Create or return the shared :class:`SgeJobMonitor` stored in *app_state*."""
    mon = app_state.get("_sge_monitor")
    if isinstance(mon, SgeJobMonitor):
        return mon
    mon = SgeJobMonitor(parent=parent)
    if on_finished is not None:
        mon.job_finished.connect(lambda jid, done: on_finished(jid, done))
    if on_failed is not None:
        mon.job_failed.connect(lambda jid, done: on_failed(jid, done))
    app_state["_sge_monitor"] = mon
    return mon


def shutdown_sge_monitor(app_state: dict[str, Any]) -> None:
    """Stop polling and block auto-import after the Napari window closes."""
    app_state.pop("viewer", None)
    mon = app_state.get("_sge_monitor")
    if isinstance(mon, SgeJobMonitor):
        mon.stop()


def store_pending_job(app_state: dict[str, Any], job: SgePendingJob) -> None:
    """Persist a serialized snapshot of *job* into ``app_state["sge_pending_jobs"]`` (replacing any
    prior entry with the same job id) and remember the connection used, for session recovery."""
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
    """Update the stored status (and optionally done payload / local download dir) for the pending job
    matching *job_id* in ``app_state["sge_pending_jobs"]``."""
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
    "shutdown_sge_monitor",
    "store_pending_job",
    "update_pending_job_status",
]
