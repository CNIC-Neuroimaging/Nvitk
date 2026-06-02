"""Orchestrate GUI SGE export, upload, and remote submission."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from nvitk.cluster import sge_json
from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.remote_transfer import resolve_cluster_host, upload_staged_job
from nvitk.gui.core.backend import gpu_enabled
from nvitk.gui.sge.dialog import SgeSubmitDialog
from nvitk.gui.sge.job import emit_gui_sge_script, stage_job_locally
from nvitk.gui.sge.models import SgeConnection, SgePendingJob
from nvitk.gui.sge.poll import register_sge_monitor, store_pending_job
from nvitk.gui.tools.panel import _collect_params, _update_reference_layers
from nvitk.gui.tools.presets import apply_preset_to_panel, preset_key_from_title
from nvitk.gui.tools.runner import log_tool_failure, notify, parse_label_ids
from nvitk.gui.tools.registry import is_sge_capable, sge_block_reason, tool_by_id, tool_id_from_label


def _resolve_remote_job_root(user_root: str, job_id: str) -> str:
    """Use ``{gui_sge_job_root}/{job_id}`` when the dialog still has the configured base."""
    root = str(user_root or "").strip().rstrip("/")
    base = sge_json.gui_sge_job_root()
    if base and root == base.rstrip("/"):
        return f"{base.rstrip('/')}/{job_id}"
    return root


def _require_paramiko() -> None:
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Remote SGE requires Paramiko. Install with: pip install 'nvitk[cluster]'"
        ) from exc


def _ensure_sge_monitor(app_state: dict[str, Any]) -> None:
    if app_state.get("_sge_monitor_registered"):
        return

    def _on_finished(job_id: str, done: Any) -> None:
        from nvitk.gui.sge.poll import update_pending_job_status
        from nvitk.gui.sge.retrieve import import_sge_job

        payload = done.to_dict() if hasattr(done, "to_dict") else done
        update_pending_job_status(app_state, job_id, status="done", done_payload=payload)
        viewer = app_state.get("viewer")
        if viewer is None:
            return
        notify(f"SGE job finished: {job_id}. Downloading results …")
        import_sge_job(
            viewer,
            app_state,
            job_id=job_id,
            auto_delete_remote=True,
            manual_fallback=False,
        )

    def _on_failed(job_id: str, done: Any) -> None:
        from nvitk.gui.sge.poll import update_pending_job_status

        payload = done.to_dict() if hasattr(done, "to_dict") else done
        update_pending_job_status(app_state, job_id, status="failed", done_payload=payload)
        err = getattr(done, "error", None) or "unknown error"
        notify(
            f"SGE job failed: {job_id}\n{err}\n"
            f"Remote data remains on the cluster.",
            error=True,
        )

    parent = None
    viewer = app_state.get("viewer")
    if viewer is not None:
        try:
            parent = viewer.window._qt_viewer
        except Exception:
            try:
                parent = viewer.window._qt_window
            except Exception:
                parent = None
    register_sge_monitor(
        app_state,
        on_finished=_on_finished,
        on_failed=_on_failed,
        parent=parent,
    )
    app_state["_sge_monitor_registered"] = True


def submit_gui_sge(
    viewer: Any,
    tool_panel: Any,
    app_state: dict[str, Any],
    *,
    get_label_ids = None,
    get_totalseg_roi = None,
    parent = None,
) -> bool:
    """Validate tool/layer, stage locally, upload, and SSH-run ``submit.sh``."""
    if not viewer.layers:
        notify("No layers loaded. Open an image first.", error=True)
        return False

    category = tool_panel.category.value
    operation = tool_panel.operation.value
    tool_id = tool_id_from_label(category, operation)
    if not tool_id:
        notify("Select a valid operation.", error=True)
        return False
    if not is_sge_capable(tool_id):
        notify(sge_block_reason(tool_id), error=True)
        return False

    from nvitk.gui.labels.visibility import infer_target_mode

    spec = tool_by_id(tool_id)
    layer = viewer.layers.selection.active or viewer.layers[-1]

    ids = []
    if get_label_ids is not None:
        ids = list(get_label_ids())
    if not ids:
        ids = parse_label_ids(tool_panel.label_ids.value)

    target_mode = infer_target_mode(layer, label_ids=ids or None)

    if spec and spec.run_mode == "layer" and target_mode == "label" and not ids:
        notify("Select at least one label (checkbox list or id field).", error=True)
        return False

    try:
        _require_paramiko()
    except ImportError as exc:
        notify(str(exc), error=True)
        return False

    dlg = SgeSubmitDialog(parent=parent)
    if dlg.exec() != dlg.Accepted:
        return False
    conn = dlg.settings()

    _update_reference_layers(tool_panel, viewer)
    if tool_id == "seg_region_grow":
        preset = getattr(tool_panel, "pipeline_preset", None)
        if preset is not None:
            key = preset_key_from_title(tool_id, preset.value)
            apply_preset_to_panel(tool_panel, tool_id, key)

    params = _collect_params(tool_panel, tool_id)
    params["selected_label_ids"] = ids
    if ids and tool_id in ("seg_combine_labels", "seg_remove_labels"):
        params["label_ids"] = ",".join(str(i) for i in ids)
    if tool_id == "seg_totalsegmentator" and get_totalseg_roi is not None:
        params["roi_subset"] = get_totalseg_roi()

    staging = None
    try:
        staging, job = stage_job_locally(
            viewer,
            layer,
            tool_id=tool_id,
            params=params,
            target_mode=target_mode,
            label_ids=ids or None,
            gpu=gpu_enabled(),
        )
        remote_job_root = _resolve_remote_job_root(conn.remote_job_root, job.job_id)
        emit_gui_sge_script(
            job,
            local_staging=staging,
            remote_job_root=remote_job_root,
        )
        host = resolve_cluster_host(conn.host)
        notify(f"Uploading job {job.job_id} to {host}:{remote_job_root} …")
        upload_staged_job(
            host=conn.host,
            user=conn.user,
            password=conn.password,
            local_staging=staging,
            remote_job_root=remote_job_root,
        )
        remote_script = Path(f"{remote_job_root.rstrip('/')}/submit.sh")
        notify(f"Submitting {remote_script} on {host} …")
        ok = run_sge_script_ssh(
            host,
            conn.user,
            conn.password,
            script_path=remote_script,
        )
        if ok:
            pending = SgePendingJob(
                job_id=job.job_id,
                tool_id=tool_id,
                remote_job_root=remote_job_root,
                connection=SgeConnection(
                    host=conn.host,
                    user=conn.user,
                    password=conn.password,
                ),
                output_name=job.output_name,
            )
            store_pending_job(app_state, pending)
            _ensure_sge_monitor(app_state)
            monitor = app_state.get("_sge_monitor")
            if monitor is not None:
                monitor.track(pending)
            notify(
                f"SGE job submitted.\n"
                f"  job_id: {job.job_id}\n"
                f"  remote: {remote_job_root}\n"
                f"  tool: {tool_id}\n"
                f"Results will auto-import when output/.done appears (~5s poll)."
            )
        else:
            notify(
                f"Remote submission may have failed. Check SSH output and run manually:\n"
                f"  bash {remote_script}\n"
                f"If the job was uploaded, data remains at: {remote_job_root}",
                error=True,
            )
        return ok
    except Exception as exc:
        log_tool_failure(exc)
        notify(f"SGE submit failed: {exc}", error=True)
        return False
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = ["submit_gui_sge"]
