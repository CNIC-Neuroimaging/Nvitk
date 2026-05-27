"""Orchestrate GUI SGE export, upload, and remote submission."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from nvitk.cluster.remote_submit import run_sge_script_ssh
from nvitk.cluster.remote_transfer import resolve_cluster_host, upload_staged_job
from nvitk.gui.gui_backend import gpu_enabled
from nvitk.gui.sge_dialog import SgeSubmitDialog
from nvitk.gui.sge_job import emit_gui_sge_script, stage_job_locally
from nvitk.gui.tool_panel import _collect_params, _update_reference_layers
from nvitk.gui.tool_presets import apply_preset_to_panel, preset_key_from_title
from nvitk.gui.tool_runner import log_tool_failure, notify, parse_label_ids
from nvitk.gui.tools_registry import is_sge_capable, sge_block_reason, tool_by_id, tool_id_from_label


def _require_paramiko() -> None:
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Remote SGE requires Paramiko. Install with: pip install 'nvitk[cluster]'"
        ) from exc


def submit_gui_sge(
    viewer: Any,
    tool_panel: Any,
    *,
    get_label_ids: Callable[[], list[int]] | None = None,
    get_totalseg_roi: Callable[[], list[str] | None] | None = None,
    parent: Any = None,
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

    spec = tool_by_id(tool_id)
    target_mode = tool_panel.target_mode.value
    layer = viewer.layers.selection.active or viewer.layers[-1]

    ids: list[int] = []
    if get_label_ids is not None:
        ids = list(get_label_ids())
    if not ids:
        ids = parse_label_ids(tool_panel.label_ids.value)

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
        emit_gui_sge_script(
            job,
            local_staging=staging,
            remote_job_root=conn.remote_job_root,
        )
        host = resolve_cluster_host(conn.host)
        notify(f"Uploading job {job.job_id} to {host}:{conn.remote_job_root} …")
        upload_staged_job(
            host=conn.host,
            user=conn.user,
            password=conn.password,
            local_staging=staging,
            remote_job_root=conn.remote_job_root,
        )
        remote_script = Path(f"{conn.remote_job_root.rstrip('/')}/submit.sh")
        notify(f"Submitting {remote_script} on {host} …")
        ok = run_sge_script_ssh(
            host,
            conn.user,
            conn.password,
            script_path=remote_script,
        )
        if ok:
            notify(
                f"SGE job submitted.\n"
                f"  job_id: {job.job_id}\n"
                f"  remote: {conn.remote_job_root}\n"
                f"  tool: {tool_id}\n"
                f"Results remain on the cluster (v1)."
            )
        else:
            notify(
                f"Remote submission may have failed. Check SSH output and run manually:\n"
                f"  bash {remote_script}",
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
