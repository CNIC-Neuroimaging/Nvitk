"""Tests for GUI SGE staging, capabilities, and headless worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nvitk.gui.sge_job import (
    GuiSgeJob,
    build_remote_paths,
    emit_gui_sge_script,
    stage_job_locally,
)
from nvitk.gui.tools_registry import is_sge_capable, sge_block_reason
from nvitk.io import imread, imsave
from nvitk.types import Image


def test_build_remote_paths():
    data, output, script = build_remote_paths("/home/user/jobs/run1")
    assert data == "/home/user/jobs/run1/data"
    assert output == "/home/user/jobs/run1/output"
    assert script == "/home/user/jobs/run1/submit.sh"


def test_is_sge_capable_morphology():
    assert is_sge_capable("dilate")
    assert is_sge_capable("erode")
    assert not is_sge_capable("viz_flowshow")
    assert not is_sge_capable("measure_generate_suv")
    assert not is_sge_capable(None)
    assert "Napari" in sge_block_reason("viz_flowshow") or "not supported" in sge_block_reason(
        "viz_flowshow"
    ).lower()


def test_gui_sge_job_json_roundtrip():
    job = GuiSgeJob(
        job_id="test_job",
        tool_id="dilate",
        params={"footprint": 1, "iterations": 2},
        target_mode="binary_mask",
        label_ids=[1, 2],
        aux_layers=[],
        gpu=False,
    )
    raw = job.to_dict()
    restored = GuiSgeJob.from_dict(raw)
    assert restored.job_id == "test_job"
    assert restored.tool_id == "dilate"
    assert restored.params["iterations"] == 2
    assert restored.label_ids == [1, 2]


def test_emit_gui_sge_script_writes_submit_sh(tmp_path: Path):
    job = GuiSgeJob(
        job_id="j1",
        tool_id="dilate",
        params={},
        target_mode="raw",
        label_ids=[],
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "data").mkdir()
    (staging / "output").mkdir()
    script = emit_gui_sge_script(
        job,
        local_staging=staging,
        remote_job_root="/cluster/jobs/j1",
    )
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "sge_worker" in text or "nvitk.gui.sge_worker" in text
    assert "/cluster/jobs/j1/data" in text


def test_stage_job_locally_exports_input(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NVITK_SGE_STAGING", str(tmp_path))
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1
    layer = MagicMock()
    layer.name = "mask"
    layer.data = data
    layer.metadata = {"affine": np.eye(4)}
    layer.scale = (1.0, 1.0, 1.0)
    layer.affine = np.eye(4)

    viewer = MagicMock()
    viewer.layers = [layer]

    staging, job = stage_job_locally(
        viewer,
        layer,
        tool_id="dilate",
        params={"footprint": 1, "iterations": 1, "mode": "binary"},
        target_mode="binary_mask",
        label_ids=None,
    )
    try:
        assert (staging / "data" / "input.nii.gz").is_file()
        job_json = json.loads((staging / "data" / "job.json").read_text(encoding="utf-8"))
        assert job_json["tool_id"] == "dilate"
        assert job.job_id == job_json["job_id"]
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)


def test_run_gui_tool_headless_dilate(tmp_path: Path):
    pytest.importorskip("qtpy")
    from nvitk.gui.tool_runner import run_gui_tool_headless

    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[2, 2, 2] = 1
    img = Image(data=data, metadata={"spacing": (1.0, 1.0, 1.0)}, name="input")
    out = run_gui_tool_headless(
        "dilate",
        primary=img,
        aux={},
        target_mode="binary_mask",
        label_ids=None,
        params={"footprint": 1, "iterations": 1, "mode": "binary"},
    )
    assert out is not None
    assert out.sum() >= 1


def test_sge_worker_smoke(tmp_path: Path, monkeypatch):
    pytest.importorskip("qtpy")
    from click.testing import CliRunner

    from nvitk.gui import sge_worker

    data = np.zeros((4, 4, 4), dtype=np.uint8)
    data[1:3, 1:3, 1:3] = 1
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"
    data_dir.mkdir()
    out_dir.mkdir()
    imsave(data_dir / "input.nii.gz", Image(data=data, metadata={"spacing": (1.0, 1.0, 1.0)}))
    job = {
        "job_id": "smoke",
        "tool_id": "dilate",
        "params": {"footprint": 1, "iterations": 1, "mode": "binary"},
        "target_mode": "binary_mask",
        "label_ids": [],
        "input_name": "input.nii.gz",
        "output_name": "output.nii.gz",
        "aux_layers": {},
        "gpu": False,
    }
    (data_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")

    _orig_path = Path

    def _path(arg=None):
        if arg is not None and str(arg) == "/nvitk/output":
            return out_dir
        if arg is None:
            return _orig_path()
        return _orig_path(arg)

    monkeypatch.setattr(sge_worker, "Path", _path)
    runner = CliRunner()
    result = runner.invoke(sge_worker.main, ["--job", str(data_dir / "job.json")])
    assert result.exit_code == 0, result.output
    assert (out_dir / "output.nii.gz").is_file()
    out_img = imread(out_dir / "output.nii.gz")
    assert int(np.asarray(out_img.data).sum()) >= 1


def test_upload_staged_job_calls_paramiko(tmp_path: Path):
    from nvitk.cluster.remote_transfer import upload_staged_job

    local = tmp_path / "job"
    local.mkdir()
    (local / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    mock_sftp = MagicMock()
    mock_client = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    mock_paramiko = MagicMock()
    mock_paramiko.SSHClient.return_value = mock_client
    mock_paramiko.AutoAddPolicy = MagicMock()

    with patch.dict(sys.modules, {"paramiko": mock_paramiko}):
        upload_staged_job(
            host="samwise",
            user="u",
            password="p",
            local_staging=local,
            remote_job_root="/remote/job",
        )

    mock_client.connect.assert_called_once()
    mock_sftp.put.assert_called()
