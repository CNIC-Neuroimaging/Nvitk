"""Local staging, job.json, and SGE driver script emission for GUI remote jobs."""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nvitk.cli._sge import emit_submit_script
from nvitk.cluster.sge import gui_sge_worker_argv
from nvitk.gui.core.spatial import layer_to_image
from nvitk.gui.tools.registry import params_for_tool
from nvitk.io import imsave

_LAYER_NONE = "(none)"
INPUT_NAME = "input.nii.gz"
OUTPUT_NAME = "output.nii.gz"
JOB_JSON = "job.json"


def _resolve_layer(viewer: Any, name: str) -> Any:
    name = str(name or "").strip()
    if not name:
        raise ValueError("Select a reference layer.")
    for lyr in viewer.layers:
        if lyr.name == name:
            return lyr
    raise ValueError(f"Layer not found: {name}")


def _export_layer(layer: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, layer_to_image(layer))


@dataclass
class AuxLayerSpec:
    param: str
    file: str
    layer_name: str


@dataclass
class GuiSgeJob:
    job_id: str
    tool_id: str
    params: dict[str, Any]
    target_mode: str
    label_ids: list[int]
    input_name: str = INPUT_NAME
    output_name: str = OUTPUT_NAME
    aux_layers: list[AuxLayerSpec] = field(default_factory=list)
    gpu: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aux_layers"] = {a.param: {"file": a.file, "layer_name": a.layer_name} for a in self.aux_layers}
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GuiSgeJob:
        aux_raw = data.get("aux_layers") or {}
        aux = []
        if isinstance(aux_raw, dict):
            for param, info in aux_raw.items():
                if isinstance(info, dict):
                    aux.append(
                        AuxLayerSpec(
                            param=str(param),
                            file=str(info.get("file") or ""),
                            layer_name=str(info.get("layer_name") or ""),
                        )
                    )
        return cls(
            job_id=str(data.get("job_id") or ""),
            tool_id=str(data.get("tool_id") or ""),
            params=dict(data.get("params") or {}),
            target_mode=str(data.get("target_mode") or "raw"),
            label_ids=[int(x) for x in (data.get("label_ids") or [])],
            input_name=str(data.get("input_name") or INPUT_NAME),
            output_name=str(data.get("output_name") or OUTPUT_NAME),
            aux_layers=aux,
            gpu=bool(data.get("gpu")),
        )


def build_remote_paths(remote_job_root: str) -> tuple[str, str, str]:
    """Return ``(data_root, output_root, submit_script_path)`` on the cluster."""
    root = str(remote_job_root or "").strip().rstrip("/")
    if not root:
        raise ValueError("Remote job directory is required.")
    return f"{root}/data", f"{root}/output", f"{root}/submit.sh"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _new_job_id(tool_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tool_id}_{uuid.uuid4().hex[:8]}"


def _staging_root() -> Path:
    base = os.environ.get("NVITK_SGE_STAGING", "").strip()
    if base:
        root = Path(base)
        root.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="nvitk_sge_", dir=root))
    else:
        path = Path(tempfile.mkdtemp(prefix="nvitk_sge_"))
    return path


def stage_job_locally(
    viewer: Any,
    layer: Any,
    *,
    tool_id: str,
    params: dict[str, Any],
    target_mode: str,
    label_ids: list[int] | None,
    gpu = False,
) -> tuple[Path, GuiSgeJob]:
    """Export inputs + ``job.json`` under a local staging directory."""
    staging = _staging_root()
    data_dir = staging / "data"
    output_dir = staging / "output"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    _export_layer(layer, data_dir / INPUT_NAME)

    aux = []
    for pspec in params_for_tool(tool_id):
        if pspec.kind != "layer":
            continue
        layer_name = str(params.get(pspec.name) or "").strip()
        if not layer_name or layer_name == _LAYER_NONE:
            continue
        aux_layer = _resolve_layer(viewer, layer_name)
        fname = f"{pspec.name}.nii.gz"
        _export_layer(aux_layer, data_dir / fname)
        aux.append(AuxLayerSpec(param=pspec.name, file=fname, layer_name=layer_name))

    job = GuiSgeJob(
        job_id=_new_job_id(tool_id),
        tool_id=tool_id,
        params=_json_safe(dict(params)),
        target_mode=str(target_mode or "raw"),
        label_ids=[int(x) for x in (label_ids or [])],
        aux_layers=aux,
        gpu=gpu,
    )
    (data_dir / JOB_JSON).write_text(
        json.dumps(job.to_dict(), indent=2),
        encoding="utf-8",
    )
    return staging, job


def emit_gui_sge_script(
    job: GuiSgeJob,
    *,
    local_staging: Path,
    remote_job_root: str,
) -> Path:
    """Write ``submit.sh`` into *local_staging* with cluster-side bind paths."""
    data_root, output_root, _remote_script = build_remote_paths(remote_job_root)
    job_arg = shlex.quote("/nvitk/data/job.json")
    python_cmd = " ".join([*gui_sge_worker_argv(), "--job", job_arg])
    job_name = f"gui_{job.tool_id}"[:200]
    script_path = local_staging / "submit.sh"
    emit_submit_script(
        script_path=script_path,
        stages=[(job_name, python_cmd)],
        data_root=Path(data_root),
        output_root=Path(output_root),
        gpu=job.gpu,
    )
    return script_path


__all__ = [
    "AuxLayerSpec",
    "GuiSgeJob",
    "INPUT_NAME",
    "JOB_JSON",
    "OUTPUT_NAME",
    "build_remote_paths",
    "emit_gui_sge_script",
    "stage_job_locally",
]
