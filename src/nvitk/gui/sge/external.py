"""FastSurfer and FSL tools on the cluster (the Tools panel's *Run SGE*).

The image tools send one layer and get one array back. These tools take several inputs
— layers chosen in the form, layers named in a command, bvecs/bvals, acquisition files,
a BEDPOSTX folder, the FreeSurfer license — and produce a set of outputs. So a job is
staged as:

* ``data/input.nii.gz`` — the active layer (when the tool has one);
* ``data/<param>.nii.gz`` — each layer the form names, and ``data/named_<k>.nii.gz`` for
  layers named in a command (``{layer:NAME}``) or the longitudinal time points;
* ``data/files/<param>/…`` — every file or folder a path parameter names, the parameter
  rewritten to the container path it lands on;
* ``data/job.json`` — the usual job, its params rewritten.

On the node the worker (:mod:`nvitk.gui.sge.worker`) rebuilds the tool's inputs from
those files (:class:`~nvitk.gui.tools.external_io.StagedInputs`), runs the same tool
definition a local run uses — headless, in the foreground — and writes the import
manifest into ``.done``. The GUI downloads the files the manifest lists and opens them
exactly as a local run would (:func:`~nvitk.gui.tools.external_io.load_manifest`).

The nvitk container carries FastSurfer, FSL and a pruned FreeSurfer for this
(``registry/containers/projects/nvitk``); the FreeSurfer license travels with each job.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from nvitk.gui.sge.job import (
    INPUT_NAME,
    JOB_JSON,
    AuxLayerSpec,
    GuiSgeJob,
    _json_safe,
    _new_job_id,
    _resolve_layer,
    _staging_root,
)

#: Where the job's ``data`` folder is inside the container.
CONTAINER_DATA = "/nvitk/data"

#: Tool families sent to the cluster this way.
_PREFIXES = ("fastsurfer_", "fsl_")

#: Tools that read local folders or install software: never sent to the cluster.
LOCAL_ONLY: frozenset[str] = frozenset({"fastsurfer_load_subject", "fastsurfer_install_freesurfer", "fsl_install"})

#: Tools that ask for a GPU node (the networks; eddy / bedpostx / probtrackx GPU builds).
GPU_TOOLS: frozenset[str] = frozenset({
    "fastsurfer_seg", "fastsurfer_cereb", "fastsurfer_hypothal", "fastsurfer_cc", "fastsurfer_full",
    "fastsurfer_long", "fsl_eddy", "fsl_bedpostx", "fsl_probtrackx2", "fsl_xtract",
})

#: Parameters naming one file or folder, uploaded with the job.
FILE_PARAMS: frozenset[str] = frozenset({
    "fsl_bvecs", "fsl_bvals", "fsl_acqp", "fsl_index", "fsl_datain", "fsl_affine", "fsl_warp", "fsl_premat",
    "fsl_matrix", "fsl_matrix2", "fsl_bedpostx_dir", "fsl_std2diff", "fsl_diff2std", "fsl_timings", "fs_license",
    "fsl_asl_fslanat", "fsl_asl_asl2struct",
})
#: Parameters naming a basename: every file starting with it is uploaded (TOPUP's outputs).
PREFIX_PARAMS: frozenset[str] = frozenset({"fsl_topup_base"})
#: Parameters naming a comma-separated list of files and folders.
LIST_PARAMS: frozenset[str] = frozenset({"fsl_fa_list"})
#: Parameters that mean nothing on the node (its output folder and FreeSurfer are fixed).
DROP_PARAMS: frozenset[str] = frozenset({"output_dir", "fs_freesurfer_home", "fsl_output_matrix"})

#: DWI tools whose bvecs / bvals default to the files next to the layer's source.
_DWI_TOOLS = frozenset({"fsl_eddy", "fsl_dtifit", "fsl_bedpostx"})
_LAYER_TOKEN = re.compile(r"\{layer:([^}]+)\}")


def is_external_tool(tool_id: str | None) -> bool:
    return bool(tool_id) and str(tool_id).startswith(_PREFIXES)


def external_sge_capable(tool_id: str | None) -> bool:
    """Whether *tool_id* is a FastSurfer / FSL tool that can run on the cluster."""
    return is_external_tool(tool_id) and str(tool_id) not in LOCAL_ONLY


def wants_gpu(tool_id: str) -> bool:
    return str(tool_id) in GPU_TOOLS


# ── staging (GUI side) ───────────────────────────────────────────────────────


def _export(layer: Any, path: Path) -> None:
    from nvitk.gui.io.export import export_layer

    path.parent.mkdir(parents=True, exist_ok=True)
    export_layer(layer, path)


def _upload_copy(source: Path, files_dir: Path, key: str) -> str:
    """Copy *source* (file or folder) under ``files/<key>/`` and return its container path."""
    target = files_dir / key / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copyfile(source, target)
    return f"{CONTAINER_DATA}/files/{key}/{source.name}"


def _local_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    return Path(raw).expanduser() if raw else None


def stage_external_job(
    viewer: Any,
    layer: Any,
    *,
    tool_id: str,
    params: dict[str, Any],
    gpu: bool | None = None,
) -> tuple[Path, GuiSgeJob]:
    """Write a FastSurfer / FSL job's inputs and ``job.json`` under a local staging folder."""
    from nvitk.gui.labels.visibility import is_label_like_layer
    from nvitk.gui.tools.external_io import sidecar_file
    from nvitk.gui.tools.registry import params_for_tool

    if not external_sge_capable(tool_id):
        raise ValueError(f"{tool_id} cannot run on the cluster.")
    staging = _staging_root()
    data_dir = staging / "data"
    files_dir = data_dir / "files"
    (staging / "output").mkdir(parents=True)
    data_dir.mkdir(parents=True)
    params = dict(params)
    aux: list[AuxLayerSpec] = []

    input_name = ""
    if layer is not None:
        _export(layer, data_dir / INPUT_NAME)
        input_name = INPUT_NAME
        params["_input_layer_name"] = str(layer.name)
        params["_input_is_labels"] = bool(is_label_like_layer(layer))

    for pspec in params_for_tool(tool_id):
        if pspec.kind != "layer":
            continue
        name = str(params.get(pspec.name) or "").strip()
        if not name or name == "(none)":
            continue
        fname = f"{pspec.name}.nii.gz"
        _export(_resolve_layer(viewer, name), data_dir / fname)
        aux.append(AuxLayerSpec(param=pspec.name, file=fname, layer_name=name))

    named: set[str] = set()
    for key in ("fsl_command", "fsl_ops"):
        named |= {n.strip() for n in _LAYER_TOKEN.findall(str(params.get(key) or ""))}
    if tool_id == "fastsurfer_long":
        named |= {n.strip() for n in str(params.get("fs_long_layers") or "").split(",") if n.strip()}
    for k, name in enumerate(sorted(named)):
        fname = f"named_{k}.nii.gz"
        _export(_resolve_layer(viewer, name), data_dir / fname)
        aux.append(AuxLayerSpec(param="@layer", file=fname, layer_name=name))

    if tool_id in _DWI_TOOLS and layer is not None:
        for key, suffixes, fallbacks in (("fsl_bvecs", (".bvec", ".bvecs"), ("bvecs",)),
                                         ("fsl_bvals", (".bval", ".bvals"), ("bvals",))):
            if not str(params.get(key) or "").strip():
                found = sidecar_file(layer, suffixes, fallbacks)
                if found is None:
                    raise ValueError(f"Set the {key.split('_')[1]} file (none found next to the layer's source).")
                params[key] = str(found)

    from nvitk.gui.tools.fastsurfer_tools import SURFACE_TOOL_IDS

    if tool_id in SURFACE_TOOL_IDS or params.get("fs_tal_reg"):
        if not str(params.get("fs_license") or "").strip():
            from nvitk.segmentation.fastsurfer.config import fs_license

            lic = fs_license()
            if lic is None:
                raise ValueError(
                    "The surface pipeline needs a FreeSurfer license: set the license file field "
                    "(it is sent with the job)."
                )
            params["fs_license"] = str(lic)

    for key in FILE_PARAMS:
        source = _local_path(params.get(key))
        if source is None:
            continue
        if not source.exists():
            raise ValueError(f"{key}: {source} does not exist.")
        params[key] = _upload_copy(source, files_dir, key)
    for key in PREFIX_PARAMS:
        prefix = _local_path(params.get(key))
        if prefix is None:
            continue
        matches = sorted(prefix.parent.glob(prefix.name + "*"))
        if not matches:
            raise ValueError(f"{key}: no files start with {prefix}.")
        for match in matches:
            _upload_copy(match, files_dir, key)
        params[key] = f"{CONTAINER_DATA}/files/{key}/{prefix.name}"
    for key in LIST_PARAMS:
        raw = str(params.get(key) or "").strip()
        if not raw:
            continue
        sent = []
        for k, item in enumerate(p.strip() for p in raw.split(",") if p.strip()):
            source = Path(item).expanduser()
            if not source.exists():
                raise ValueError(f"{key}: {source} does not exist.")
            sent.append(_upload_copy(source, files_dir, f"{key}_{k}"))
        params[key] = ",".join(sent)
    for key in DROP_PARAMS:
        params.pop(key, None)

    job = GuiSgeJob(
        job_id=_new_job_id(tool_id),
        tool_id=tool_id,
        params=_json_safe(params),
        target_mode="raw",
        label_ids=[],
        input_name=input_name,
        aux_layers=aux,
        gpu=wants_gpu(tool_id) if gpu is None else bool(gpu),
    )
    (data_dir / JOB_JSON).write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    return staging, job


# ── running (node side) ──────────────────────────────────────────────────────


def run_external_job(job: GuiSgeJob, data_dir: Path, output_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Run a staged FastSurfer / FSL job; returns ``(output files, import manifest)``.

    Output files are relative to *output_dir*, which is where the tool writes.
    """
    from nvitk.gui.tools.external_io import StagedInputs

    params = dict(job.params or {})
    inputs = StagedInputs(
        job.tool_id,
        params,
        output_dir,
        input_file=(data_dir / job.input_name) if job.input_name else None,
        input_name=str(params.get("_input_layer_name") or job.tool_id),
        layer_files={a.param: data_dir / a.file for a in job.aux_layers if a.param != "@layer"},
        named_files={a.layer_name: data_dir / a.file for a in job.aux_layers if a.param == "@layer"},
        input_is_labels=bool(params.get("_input_is_labels")),
    )
    if job.tool_id.startswith("fastsurfer_"):
        from nvitk.gui.tools.fastsurfer_tools import run_fastsurfer_headless

        manifest = run_fastsurfer_headless(job.tool_id, inputs, gpu=job.gpu)
        if job.tool_id != "fastsurfer_long":
            manifest["prefix"] = inputs.display_name
    else:
        from nvitk.gui.tools.fsl_tools import run_fsl_headless

        manifest = run_fsl_headless(job.tool_id, inputs)
        manifest["prefix"] = inputs.display_name
    files = [f for f in manifest.get("files", []) if (output_dir / f).is_file()]
    return files, manifest


__all__ = [
    "CONTAINER_DATA",
    "GPU_TOOLS",
    "LOCAL_ONLY",
    "external_sge_capable",
    "is_external_tool",
    "run_external_job",
    "stage_external_job",
    "wants_gpu",
]
