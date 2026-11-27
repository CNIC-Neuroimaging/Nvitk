"""The rest of the Tools panel on the cluster: tools that work through the viewer.

The image tools send one layer and get one array back; FSL and FastSurfer have their own staging
(:mod:`nvitk.gui.sge.external`). The tools here (registration with FLIRT / ANTsPy / FireANTs, the
FreeSurfer topology programs, the ANTsPyNet networks, the 3D+t, resampling and k-space tools...)
read layers from the viewer, add layers to it, write transforms, show result tables. So a job runs
the very same tool code against a viewer rebuilt on the node:

* **staging (GUI)**: the active layer, every layer the form picks and every layer a text field
  names (*Also warp*, a list of layers to stack) are written to ``data/`` (volumes as NIfTI,
  meshes as FreeSurfer surfaces, scanner-RAS mm); every file a field names (a FLIRT matrix, ANTs
  transforms, the FreeSurfer license) is uploaded to ``data/files/`` and the field rewritten;
* **running (node)**: a headless Napari ``ViewerModel`` gets those layers back under their own
  names; the output folder fields point into ``output/work``; external programs a tool would
  start as a background job run to completion there; the tool runs (:func:`run_gui_tool`) and
  every layer it adds (or array it returns, or layer it edits) is written to ``output/``, with an
  import manifest: volumes, surfaces, result tables, messages, and the files left in
  ``output/work`` (transforms, matrices);
* **import (GUI)**: the manifest is opened like a FastSurfer / FSL job's
  (:func:`~nvitk.gui.tools.external_io.load_manifest`): the same layers a local run adds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from nvitk.gui.sge.job import JOB_JSON, AuxLayerSpec, GuiSgeJob, _json_safe, _new_job_id, _staging_root

#: Where the job's ``data`` folder is inside the container.
CONTAINER_DATA = "/nvitk/data"

#: Tools run this way (see the module docstring). Each was checked to give the same layers on
#: the cluster path as in the foreground (scratchpad sweep), or needs a program the GUI host may
#: not have (FreeSurfer, FireANTs on a GPU).
REGISTRATION_TOOLS: frozenset[str] = frozenset({
    "reg_flirt_rigid", "reg_flirt_apply", "reg_ants_register", "reg_ants_apply",
    "reg_fireants_register", "reg_fireants_apply",
})
FREESURFER_TOOLS: frozenset[str] = frozenset({
    "freesurfer_pretess", "freesurfer_mc", "freesurfer_main_component", "freesurfer_euler",
    "freesurfer_smooth", "freesurfer_inflate", "freesurfer_qsphere", "freesurfer_fix_topology",
    "freesurfer_topo_fixer", "fastsurfer_spectral_sphere", "fastsurfer_topology_chain",
})
#: ANTsPyNet networks and other deep-learning segmentations (need the image's TensorFlow).
NETWORK_TOOLS: frozenset[str] = frozenset({
    "antsxnet_brain_age", "seg_brain_extraction", "seg_dkt", "mri_super_resolution", "seg_mouse_brain",
    "seg_mra_vessel",
})
OTHER_TOOLS: frozenset[str] = frozenset({
    "recon_background_correction", "mask_genus", "vmtk_vessel_enhancement", "seg_split_lr_cc",
    "seg_split_lr_midline", "interp_resample_axes", "interp_block_reduce", "interp_mask_slices",
    "interp_fill_slices", "time_extract_frame", "time_projection", "time_interpolate_frames",
    "time_stack_layers", "kspace_fft", "kspace_filter", "masked_stats", "suv_stats",
})
RECORDED_TOOLS: frozenset[str] = REGISTRATION_TOOLS | FREESURFER_TOOLS | NETWORK_TOOLS | OTHER_TOOLS

#: Tools that ask for a GPU node.
GPU_TOOLS: frozenset[str] = frozenset({"reg_fireants_register", "reg_fireants_apply"}) | NETWORK_TOOLS

#: Fields naming the folder (or file) a tool writes to: pointed into ``output/work`` on the node.
OUTPUT_DIR_PARAMS: frozenset[str] = frozenset({"output_dir"})
OUTPUT_FILE_PARAMS: frozenset[str] = frozenset({"out_path"})
#: Fields never treated as an input file to upload (folders on this machine, display names...).
NOT_UPLOADED: frozenset[str] = OUTPUT_DIR_PARAMS | OUTPUT_FILE_PARAMS | frozenset({
    "fs_freesurfer_home", "warped_name", "matrix_name",
})
#: Mesh files the node reads back (FreeSurfer geometry, scanner-RAS mm).
SURFACE_SUFFIX = ".surf"
#: The node's offscreen Qt application, once made.
_QT_APP: list[Any] = []
#: Upload ceiling for files left in ``output/work`` that are sent back (transforms, matrices...).
_WORK_RETURN_BYTES = 4 * 1024 ** 3


def is_recorded_tool(tool_id: str | None) -> bool:
    return bool(tool_id) and str(tool_id) in RECORDED_TOOLS


def wants_gpu(tool_id: str) -> bool:
    return str(tool_id) in GPU_TOOLS


# staging (GUI side)


def _layer_kind(layer: Any) -> str:
    from nvitk.gui.labels.visibility import is_label_like_layer

    kind = type(layer).__name__
    if kind == "Surface":
        return "surface"
    if kind == "Labels" or is_label_like_layer(layer):
        return "labels"
    return "image"


def _export(layer: Any, path_stem: Path) -> str:
    """Write *layer* next to *path_stem*; returns the file name used."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if type(layer).__name__ == "Surface":
        from nvitk.freesurfer.topology import write_surface

        vertices = np.asarray(layer.data[0], dtype=np.float64)
        try:
            vertices = np.asarray(layer._data_to_world(vertices), dtype=np.float64)
        except Exception:  # noqa: BLE001 (no transform: data coordinates are world)
            pass
        out = path_stem.with_name(path_stem.name + SURFACE_SUFFIX)
        write_surface(out, vertices[:, -3:], np.asarray(layer.data[1]), orient=False)
        return out.name
    from nvitk.gui.io.export import export_layer

    out = path_stem.with_name(path_stem.name + ".nii.gz")
    export_layer(layer, out)
    return out.name


def _named_layers(viewer: Any, params: dict[str, Any]) -> list[str]:
    """Layer names written in text fields (comma-separated lists, ``{layer:NAME}`` tokens)."""
    names = {str(layer.name) for layer in viewer.layers}
    found: list[str] = []
    for key, value in params.items():
        if key.startswith("_") or not isinstance(value, str) or not value.strip():
            continue
        candidates = [value.strip()] + [p.strip() for p in value.split(",")]
        candidates += [m.strip() for m in re.findall(r"\{layer:([^}]+)\}", value)]
        for cand in candidates:
            if cand in names and cand not in found:
                found.append(cand)
    return found


def _upload(source: Path, files_dir: Path, key: str) -> str:
    target = files_dir / key / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copyfile(source, target)
    return f"{CONTAINER_DATA}/files/{key}/{source.name}"


def _upload_file_params(params: dict[str, Any], files_dir: Path) -> None:
    """Every field naming existing files on this machine (one, or a comma-separated list): the
    files uploaded and the field rewritten to where they land in the container, in order."""
    for key, value in list(params.items()):
        if key in NOT_UPLOADED or key.startswith("_") or not isinstance(value, str) or not value.strip():
            continue
        parts = [p.strip() for p in value.split(",") if p.strip()]
        paths = [Path(p).expanduser() for p in parts]
        if not paths or not all(p.is_absolute() and p.is_file() for p in paths):
            continue
        sent = [_upload(p, files_dir, f"{key}_{k}" if len(paths) > 1 else key) for k, p in enumerate(paths)]
        params[key] = ",".join(sent)


def stage_recorded_job(
    viewer: Any,
    layer: Any,
    *,
    tool_id: str,
    params: dict[str, Any],
    target_mode: str = "raw",
    label_ids: list[int] | None = None,
    gpu: bool | None = None,
) -> tuple[Path, GuiSgeJob]:
    """Write a job's inputs and ``job.json`` under a local staging folder (see the module)."""
    from nvitk.gui.sge.job import _resolve_layer
    from nvitk.gui.tools.registry import params_for_tool

    if not is_recorded_tool(tool_id):
        raise ValueError(f"{tool_id} cannot run on the cluster this way.")
    staging = _staging_root()
    data_dir = staging / "data"
    files_dir = data_dir / "files"
    (staging / "output").mkdir(parents=True)
    data_dir.mkdir(parents=True)
    params = dict(params)
    aux: list[AuxLayerSpec] = []
    kinds: dict[str, str] = {}

    input_name = ""
    if layer is not None:
        input_name = _export(layer, data_dir / "input")
        params["_input_layer_name"] = str(layer.name)
        kinds[str(layer.name)] = _layer_kind(layer)

    staged = {str(layer.name)} if layer is not None else set()
    picked = []
    for pspec in params_for_tool(tool_id):
        if pspec.kind != "layer":
            continue
        name = str(params.get(pspec.name) or "").strip()
        if name and name != "(none)":
            picked.append((pspec.name, name))
    for k, (param, name) in enumerate(picked + [("@layer", n) for n in _named_layers(viewer, params)]):
        if name in staged:
            continue
        lyr = _resolve_layer(viewer, name)
        fname = _export(lyr, data_dir / f"layer_{k}")
        key = param if param != "@layer" else f"@layer:{k}"
        aux.append(AuxLayerSpec(param=key, file=fname, layer_name=name))
        kinds[name] = _layer_kind(lyr)
        staged.add(name)

    if tool_id in FREESURFER_TOOLS and not str(params.get("freesurfer_license") or "").strip():
        # FreeSurfer 7 in the image refuses to run without one: send this machine's.
        try:
            from nvitk.gui.tools.freesurfer_tools import default_license

            lic = default_license()
        except Exception:  # noqa: BLE001
            lic = ""
        if lic:
            params["freesurfer_license"] = lic
    _upload_file_params(params, files_dir)
    params["_layer_kinds"] = kinds
    # Where the viewer stands (the time point of a 3D+t layer, the slice): tools read it.
    try:
        params["_dims_point"] = [float(x) for x in viewer.dims.point]
    except Exception:  # noqa: BLE001
        pass

    job = GuiSgeJob(
        job_id=_new_job_id(tool_id),
        tool_id=tool_id,
        params=_json_safe(params),
        target_mode=str(target_mode or "raw"),
        label_ids=[int(i) for i in (label_ids or [])],
        input_name=input_name,
        aux_layers=aux,
        gpu=wants_gpu(tool_id) if gpu is None else bool(gpu),
    )
    (data_dir / JOB_JSON).write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
    return staging, job


# running (node side)


def _add_staged(viewer: Any, path: Path, name: str, kind: str) -> Any:
    """Put a staged file back in *viewer* as the layer it was exported from."""
    if kind == "surface":
        import nibabel as nib

        vertices, faces = nib.freesurfer.read_geometry(str(path))
        return viewer.add_surface((np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)), name=name)
    from nvitk.gui.tools.runner import _add_volume_file

    return _add_volume_file(viewer, path, name, labels=kind == "labels")


def _fingerprint(layer: Any) -> str:
    try:
        data = layer.data
        if isinstance(data, (list, tuple)):
            parts = [np.ascontiguousarray(np.asarray(d)) for d in data[:2]]
        else:
            parts = [np.ascontiguousarray(np.asarray(data))]
        h = hashlib.sha1()
        for p in parts:
            h.update(str(p.shape).encode())
            h.update(p.tobytes()[: 64 * 1024 * 1024])
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return ""


@contextlib.contextmanager
def _capture(results: list[dict[str, Any]], notes: list[str]) -> Iterator[None]:
    """Collect what the tool would show (result tables, messages) and run the background jobs it
    would start to completion, here and now (no event loop on the node)."""
    import subprocess

    from nvitk.gui.core import external_jobs
    from nvitk.gui.tools import runner
    from nvitk.gui.viz import results_window

    def show_results(viewer: Any, title: str, payload: Any, *, subtitle: str = "", row_header: str = "",
                     note: str = "", **_kw: Any) -> str:
        results.append({"title": str(title), "payload": _json_safe(payload), "subtitle": str(subtitle or ""),
                        "row_header": str(row_header or "")})
        if note:
            notes.append(str(note))
        return f"{title}: {len(payload) if hasattr(payload, '__len__') else ''} row(s)"

    def gui_log(message: str, *, error: bool = False, **_kw: Any) -> None:
        print(("ERROR: " if error else "") + str(message), flush=True)
        if not error:
            notes.append(str(message))

    class _Done:
        def __init__(self, code: int) -> None:
            self.returncode = code

    def start_external_job(argv: Any, *, title: str, env: Any = None, cwd: Any = None, on_finished: Any = None,
                           log_file: Any = None, **_kw: Any) -> Any:
        print(f"[{title}] {' '.join(str(a) for a in argv)}", flush=True)
        log = open(log_file, "w") if log_file else None  # noqa: SIM115
        try:
            proc = subprocess.run([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=env,
                                  stdout=log or None, stderr=subprocess.STDOUT if log else None, check=False)
        finally:
            if log is not None:
                log.close()
        if on_finished is not None:
            on_finished(int(proc.returncode))
        return _Done(int(proc.returncode))

    saved = (results_window.show_results, runner.gui_log, external_jobs.start_external_job)
    results_window.show_results = show_results
    runner.gui_log = gui_log
    external_jobs.start_external_job = start_external_job
    try:
        yield
    finally:
        results_window.show_results, runner.gui_log, external_jobs.start_external_job = saved


def _write_output(layer: Any, output_dir: Path, k: int) -> dict[str, Any] | None:
    """One added (or edited) layer as an output file and its manifest entry."""
    from nvitk.gui.tools.external_io import file_stem, manifest_layer

    kind = type(layer).__name__
    stem = f"{k:02d}_{file_stem(str(layer.name))}"
    if kind == "Surface":
        fname = _export(layer, output_dir / stem)
        return {"surface": {"file": fname, "title": str(layer.name)}}
    if kind not in ("Image", "Labels"):
        return None
    fname = _export(layer, output_dir / stem)
    entry = manifest_layer(output_dir / fname, str(layer.name), base=output_dir, labels=kind == "Labels")
    if kind == "Image":
        entry["image"] = True  # an integer mask the tool made an Image stays one
    try:
        cmap = getattr(getattr(layer, "colormap", None), "name", None)
        if kind == "Image" and cmap and cmap != "gray":
            entry["colormap"] = str(cmap)
    except Exception:  # noqa: BLE001
        pass
    return {"layer": entry}


def run_recorded_job(job: GuiSgeJob, data_dir: Path, output_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Run a staged job (see the module); returns ``(output files, import manifest)``, the files
    relative to *output_dir*."""
    import os

    from napari.components import ViewerModel

    # Some tools build a small widget on the way (a progress dialog, a plot they also log): give
    # them an application, offscreen, rather than let Qt abort the job.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from qtpy.QtWidgets import QApplication

        _QT_APP.append(QApplication.instance() or QApplication([]))  # kept alive for the run
    except Exception:  # noqa: BLE001 (no Qt on the node: the tools that need it fail on their own)
        pass

    from nvitk.gui.tools.external_io import manifest_files
    from nvitk.gui.tools.runner import run_gui_tool

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    params = dict(job.params or {})
    kinds = dict(params.pop("_layer_kinds", {}) or {})
    if Path(data_dir).resolve() != Path(CONTAINER_DATA):
        # Run outside the container (a staged job tried on this machine): uploaded files are here.
        params = {k: v.replace(CONTAINER_DATA, str(Path(data_dir).resolve())) if isinstance(v, str) else v
                  for k, v in params.items()}
    input_layer_name = str(params.get("_input_layer_name") or "input")

    viewer = ViewerModel()
    active = None
    if job.input_name:
        active = _add_staged(viewer, data_dir / job.input_name, input_layer_name, kinds.get(input_layer_name, "image"))
    for spec in job.aux_layers:
        _add_staged(viewer, data_dir / spec.file, spec.layer_name, kinds.get(spec.layer_name, "image"))
    if active is not None:
        viewer.layers.selection.active = active
    point = params.pop("_dims_point", None)
    if point is not None and len(point) == viewer.dims.ndim:
        try:
            viewer.dims.point = tuple(float(x) for x in point)
        except Exception:  # noqa: BLE001 (an older Napari: by step instead)
            try:
                viewer.dims.set_point(list(range(len(point))), [float(x) for x in point])
            except Exception:  # noqa: BLE001
                pass

    for key in OUTPUT_DIR_PARAMS:
        params[key] = str(work)
    for key in OUTPUT_FILE_PARAMS:
        raw = str(params.get(key) or "").strip()
        if raw:
            params[key] = str(work / Path(raw).name)

    if job.tool_id in FREESURFER_TOOLS:
        # Say which program the image lacks, rather than let a command fail half-way through.
        from nvitk.freesurfer import env
        from nvitk.gui.tools.freesurfer_tools import TOOL_PROGRAMS

        missing = [p for p in TOOL_PROGRAMS.get(job.tool_id, ()) if env.which(p) is None]
        if missing:
            raise RuntimeError(f"The cluster image's FreeSurfer has no {', '.join(missing)}: "
                               f"{job.tool_id} cannot run there.")
    before = {id(layer): _fingerprint(layer) for layer in viewer.layers}
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    with _capture(results, notes):
        result = run_gui_tool(job.tool_id, active, viewer, target_mode=job.target_mode,
                              label_ids=job.label_ids or None, params=params)
    if result is not None and active is not None:
        from nvitk.core.array import to_numpy
        from nvitk.gui.core.spatial import layer_spatial_kwargs
        from nvitk.gui.labels.visibility import copy_layer_metadata_for_output

        arr = np.asarray(to_numpy(result))
        # Placed like the input, as the Tools panel places a result (the app's display kwargs).
        kwargs: dict[str, Any] = {"name": f"{active.name}_{job.tool_id}"}
        meta = copy_layer_metadata_for_output(getattr(active, "metadata", None))
        same = int(arr.ndim) == int(np.ndim(active.data))
        if meta and same:
            kwargs["metadata"] = meta
        kwargs.update(layer_spatial_kwargs(active) if same else layer_spatial_kwargs(active, ndim=int(arr.ndim)))
        labels = np.issubdtype(arr.dtype, np.integer) and int(arr.max(initial=0)) > 1
        if labels:
            viewer.add_labels(arr.astype(np.int32, copy=False), **kwargs)
        else:
            viewer.add_image(arr, **kwargs)

    manifest: dict[str, Any] = {"layers": [], "surfaces": [], "tables": results, "notes": []}
    k = 0
    for layer in list(viewer.layers):
        new = id(layer) not in before
        edited = not new and _fingerprint(layer) != before[id(layer)]
        if not (new or edited):
            continue
        if edited:
            layer.name = f"{layer.name} (edited)"
        written = _write_output(layer, output_dir, k)
        k += 1
        if written is None:
            notes.append(f"“{layer.name}” ({type(layer).__name__}) is not sent back.")
        elif "surface" in written:
            manifest["surfaces"].append(written["surface"])
        else:
            manifest["layers"].append(written["layer"])

    kept, total = [], 0
    for path in sorted(p for p in work.rglob("*") if p.is_file()):
        total += path.stat().st_size
        if total > _WORK_RETURN_BYTES:
            notes.append("Some files in the output folder were left on the cluster (over 4 GB).")
            break
        kept.append(str(path.relative_to(output_dir)))
    # Messages naming the node's output folder point at the downloaded copy on import.
    manifest["notes"] = [n.replace(str(output_dir), "{output}") for n in notes][-20:]
    manifest["work_files"] = kept
    manifest["files"] = manifest_files(manifest) + kept
    files = [f for f in manifest["files"] if (output_dir / f).is_file()]
    return files, manifest


__all__ = [
    "GPU_TOOLS",
    "RECORDED_TOOLS",
    "is_recorded_tool",
    "run_recorded_job",
    "stage_recorded_job",
    "wants_gpu",
]
