"""Download GUI SGE results from the cluster and import into Napari."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from nvitk.cluster.remote_transfer import (
    download_remote_files,
    is_safe_gui_job_root,
    remove_remote_job_tree,
)
from nvitk.gui.io.napari_io import open_paths_with_nvitk
from nvitk.gui.sge.dialog import _default_host, _default_remote_job_root
from nvitk.gui.sge.models import (
    SgeConnection,
    SgeDoneMarker,
    remote_done_path,
    remote_output_dir,
    verify_local_downloads,
)
from nvitk.gui.sge.poll import read_done_marker, resolve_session_import, update_pending_job_status
from nvitk.gui.tools.runner import log_tool_failure, notify


@dataclass(frozen=True)
class SgeRetrieveResult:
    remote_job_root: str
    local_dir: Path
    downloaded_files: list[Path]
    done: SgeDoneMarker
    imported_layers: int
    remote_deleted: bool


class SgeManualImportDialog(QDialog):
    """Fallback when session credentials are missing (e.g. after GUI restart)."""

    def __init__(
        self,
        parent=None,
        *,
        host = "",
        user = "",
        password = "",
        remote_job_root = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import SGE results (manual)")
        self.setMinimumWidth(420)

        intro = QLabel(
            "No active SGE session was found. Enter SSH credentials and the remote job path."
        )
        intro.setWordWrap(True)

        self.host = QLineEdit(host or _default_host())
        self.user = QLineEdit(user)
        self.password = QLineEdit(password)
        self.password.setEchoMode(QLineEdit.Password)
        self.remote_job_root = QLineEdit(remote_job_root or _default_remote_job_root())

        form = QFormLayout()
        form.addRow("SSH host", self.host)
        form.addRow("Username", self.user)
        form.addRow("Password", self.password)
        form.addRow("Remote job directory", self.remote_job_root)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _try_accept(self) -> None:
        if not self.host.text().strip() or not self.user.text().strip():
            notify("Host and username are required.", error=True)
            return
        if not self.remote_job_root.text().strip():
            notify("Remote job directory is required.", error=True)
            return
        self.accept()

    def connection(self) -> SgeConnection:
        return SgeConnection(
            host=self.host.text().strip(),
            user=self.user.text().strip(),
            password=self.password.text(),
        )

    def remote_root(self) -> str:
        return self.remote_job_root.text().strip().rstrip("/")


def retrieve_sge_results(
    *,
    connection: SgeConnection,
    remote_job_root: str,
) -> SgeRetrieveResult:
    """Download verified outputs from *remote_job_root* (must have ``output/.done``)."""
    root = str(remote_job_root or "").strip().rstrip("/")
    if not root:
        raise ValueError("Remote job directory is required.")

    done = read_done_marker(
        host=connection.host,
        user=connection.user,
        password=connection.password,
        remote_job_root=root,
    )
    if done is None:
        raise FileNotFoundError(
            f"No completion marker at {remote_done_path(root)}. "
            "The job may still be running, or it failed before writing .done."
        )
    if done.exit_code != 0:
        raise RuntimeError(
            f"Remote job {done.job_id!r} failed (exit {done.exit_code}): {done.error or 'unknown error'}"
        )

    local_dir = Path(tempfile.mkdtemp(prefix=f"nvitk_sge_import_{done.job_id}_"))
    remote_out = remote_output_dir(root)
    pairs = [(f"{remote_out}/{name}", local_dir / name) for name in done.output_files]
    download_remote_files(
        host=connection.host,
        user=connection.user,
        password=connection.password,
        remote_files=pairs,
    )
    downloaded = verify_local_downloads(local_dir, done)

    return SgeRetrieveResult(
        remote_job_root=root,
        local_dir=local_dir,
        downloaded_files=downloaded,
        done=done,
        imported_layers=0,
        remote_deleted=False,
    )


def import_sge_results_to_viewer(
    viewer: Any,
    result: SgeRetrieveResult,
    *,
    on_layers_changed = None,
) -> SgeRetrieveResult:
    count = 0
    for path in result.downloaded_files:
        layers = open_paths_with_nvitk(viewer, path)
        count += len(layers or [])
    if on_layers_changed is not None:
        on_layers_changed()
    return SgeRetrieveResult(
        remote_job_root=result.remote_job_root,
        local_dir=result.local_dir,
        downloaded_files=result.downloaded_files,
        done=result.done,
        imported_layers=count,
        remote_deleted=result.remote_deleted,
    )


def _apply_remote_cleanup(
    *,
    connection: SgeConnection,
    remote_job_root: str,
    result: SgeRetrieveResult,
) -> SgeRetrieveResult:
    root = remote_job_root.rstrip("/")
    if not is_safe_gui_job_root(root):
        notify(
            f"Refusing to delete {root!r}: outside configured gui_sge_job_root.",
            error=True,
        )
        return result
    code, out, err = remove_remote_job_tree(
        host=connection.host,
        user=connection.user,
        password=connection.password,
        remote_job_root=root,
    )
    if code != 0:
        notify(
            f"Import succeeded but remote cleanup failed (exit {code}). "
            f"Data remains at {root}. stderr: {err or out}",
            error=True,
        )
        return result
    return SgeRetrieveResult(
        remote_job_root=result.remote_job_root,
        local_dir=result.local_dir,
        downloaded_files=result.downloaded_files,
        done=result.done,
        imported_layers=result.imported_layers,
        remote_deleted=True,
    )


def import_sge_job(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    job_id = None,
    parent = None,
    on_layers_changed = None,
    auto_delete_remote: bool = False,
    manual_fallback: bool = False,
) -> bool:
    """Download + import using session credentials from the current GUI session."""
    if app_state.get("_sge_import_in_progress"):
        return False

    bucket = app_state.get("sge_pending_jobs") or []
    if job_id and isinstance(bucket, list):
        for row in bucket:
            if row.get("job_id") == job_id and row.get("local_download_dir"):
                notify(f"Job {job_id} was already imported.", error=True)
                return False

    app_state["_sge_import_in_progress"] = True

    conn = None
    root = ""
    resolved_job_id = job_id
    result = None

    try:
        try:
            conn, root, resolved_job_id = resolve_session_import(app_state, job_id=job_id)
        except ValueError as exc:
            if not manual_fallback:
                notify(str(exc), error=True)
                return False
            last = app_state.get("sge_last_connection") or {}
            dlg = SgeManualImportDialog(
                parent=parent,
                host=str(last.get("host") or ""),
                user=str(last.get("user") or ""),
                remote_job_root=str(last.get("remote_job_root") or ""),
            )
            if dlg.exec() != dlg.Accepted:
                return False
            conn = dlg.connection()
            root = dlg.remote_root()
            resolved_job_id = None

        assert conn is not None
        notify(f"Downloading SGE results from {root} …")
        result = retrieve_sge_results(connection=conn, remote_job_root=root)
        result = import_sge_results_to_viewer(
            viewer,
            result,
            on_layers_changed=on_layers_changed,
        )

        update_pending_job_status(
            app_state,
            result.done.job_id,
            status="retrieved",
            local_download_dir=str(result.local_dir),
        )

        if auto_delete_remote:
            result = _apply_remote_cleanup(
                connection=conn,
                remote_job_root=root,
                result=result,
            )

        msg = (
            f"Imported {result.imported_layers} layer(s) from {root}.\n"
            f"Local copy: {result.local_dir}"
        )
        if result.remote_deleted:
            msg += f"\nRemote directory deleted: {root}"
        notify(msg)
        return True
    except Exception as exc:
        log_tool_failure(exc)
        notify(
            f"Import failed: {exc}\nRemote data remains at: {root or '(unknown)'}",
            error=True,
        )
        if result is not None and result.local_dir.exists():
            notify(f"Partial local download kept at: {result.local_dir}")
        return False
    finally:
        app_state["_sge_import_in_progress"] = False


__all__ = [
    "SgeManualImportDialog",
    "SgeRetrieveResult",
    "import_sge_job",
    "import_sge_results_to_viewer",
    "retrieve_sge_results",
]
