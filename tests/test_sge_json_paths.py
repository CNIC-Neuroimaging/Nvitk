"""Tests for sge.json path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from nvitk.cluster import sge_json


def test_resolve_nvitk_src_dir_from_repo_sge_json():
    path = sge_json.sge_json_path()
    if path is None:
        return
    src = sge_json.resolve_nvitk_src_dir()
    assert str(src) == "/data3/BIOIT_IMAGE/nvitk/src"


def test_gui_sge_job_root_from_repo_sge_json():
    path = sge_json.sge_json_path()
    if path is None:
        return
    root = sge_json.gui_sge_job_root()
    assert root == "/data3/BIOIT_IMAGE/nvitk-sge/gui"


def test_emit_submit_script_uses_cluster_src(tmp_path: Path, monkeypatch):
    """Emitted singularity bind must use paths.nvitk_src_dir, not laptop src/."""
    monkeypatch.setattr(
        sge_json,
        "paths_section",
        lambda: {
            "nvitk_src_dir": "/data3/BIOIT_IMAGE/nvitk/src",
            "gui_sge_job_root": "/data3/BIOIT_IMAGE/nvitk-sge/gui/",
        },
    )
    from nvitk.gui.sge_job import GuiSgeJob, emit_gui_sge_script

    staging = tmp_path / "staging"
    (staging / "data").mkdir(parents=True)
    (staging / "output").mkdir()
    job = GuiSgeJob(
        job_id="j1",
        tool_id="dilate",
        params={},
        target_mode="raw",
        label_ids=[],
    )
    script = emit_gui_sge_script(
        job,
        local_staging=staging,
        remote_job_root="/data3/BIOIT_IMAGE/nvitk-sge/gui/j1",
    )
    text = script.read_text(encoding="utf-8")
    assert "/data3/BIOIT_IMAGE/nvitk/src" in text
    assert "/home/" not in text or "/data3/BIOIT_IMAGE/nvitk/src" in text
