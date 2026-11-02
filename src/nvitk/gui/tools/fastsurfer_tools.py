"""FastSurfer tools of the Tools panel (category *FastSurfer*).

Each tool becomes a :class:`FastSurferPlan`, the command, and how to describe what it
produced as an import manifest (:mod:`nvitk.gui.tools.external_io`), built by
:func:`build_fastsurfer` from a :class:`~nvitk.gui.tools.external_io.ToolInputs`:

* **locally**, the inputs are the viewer's layers, written to the run folder; the
  segmentation tools run as :class:`~nvitk.gui.core.external_jobs.ExternalJob`\\ s (the GUI
  stays usable (the surface pipeline takes about an hour) with progress and Cancel in
  the Jobs window) and their results open when they finish;
* **on the cluster** (*Run SGE*), the inputs are the files staged with the job, and the
  SGE worker runs the same plan headless (:func:`run_fastsurfer_headless`) and sends the
  manifest back, so the results open exactly as a local run's.

Label maps arrive as Labels layers named and coloured from FastSurfer's lookup tables,
placed by their own affine, on the input's grid with *native image* on (the default),
else on FastSurfer's conformed 1 mm grid; ``.stats`` tables go to the results window.
The surface and longitudinal tools also need FreeSurfer 7.4.1 and its license.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nvitk.gui.tools.external_io import (
    StagedInputs,
    ToolInputs,
    ViewerInputs,
    file_stem,
    load_manifest,
    manifest_files,
    manifest_layer,
    path_param,
)

#: Tool ids handled here.
FASTSURFER_TOOL_IDS: frozenset[str] = frozenset({
    "fastsurfer_seg",
    "fastsurfer_cereb",
    "fastsurfer_hypothal",
    "fastsurfer_cc",
    "fastsurfer_surf",
    "fastsurfer_full",
    "fastsurfer_long",
    "fastsurfer_load_subject",
    "fastsurfer_segstats",
    "fastsurfer_reduce_aseg",
    "fastsurfer_conform",
    "fastsurfer_install_freesurfer",
    "fastsurfer_spectral_sphere",
    "fastsurfer_topology_chain",
})

#: Tools that only make sense in this GUI session (they read local folders or install
#: software), never sent to the cluster.
LOCAL_ONLY_TOOL_IDS: frozenset[str] = frozenset({"fastsurfer_load_subject", "fastsurfer_install_freesurfer"})

#: The sub-module each module-only tool runs.
_MODULE_OF = {"fastsurfer_cereb": "cereb", "fastsurfer_hypothal": "hypothal", "fastsurfer_cc": "cc"}

#: Tools that need FreeSurfer (surfaces / longitudinal).
SURFACE_TOOL_IDS: frozenset[str] = frozenset({"fastsurfer_surf", "fastsurfer_full", "fastsurfer_long"})


def _notify(message: str, *, error: bool = False) -> None:
    from nvitk.gui.tools.runner import notify

    notify(message, error=error)


def _gpu_device() -> str:
    from nvitk.gui.core.backend import gpu_enabled

    return "cuda" if gpu_enabled() else "cpu"


def _sid(inputs: ToolInputs) -> str:
    raw = str(inputs.params.get("fs_sid") or "").strip() or inputs.display_name
    return file_stem(raw, "subject")


def _require_fastsurfer() -> bool:
    """True when FastSurfer is installed; otherwise offers to install it (in the background)."""
    from nvitk.segmentation.fastsurfer import config

    if config.fastsurfer_home() is not None:
        return True
    from qtpy.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        None,
        "FastSurfer is not installed",
        f"FastSurfer {config.FASTSURFER_VERSION} is not installed in this environment.\n\n"
        "Install it now (≈20 MB of code and ≈320 MB of network weights)? It runs in the "
        "background; run the tool again when it is done.",
    )
    if answer == QMessageBox.StandardButton.Yes:
        from nvitk.gui.core.external_jobs import start_external_job

        start_external_job(
            [sys.executable, "-m", "nvitk.segmentation.fastsurfer.cli", "install"],
            title="Install FastSurfer",
            on_finished=lambda code: _notify(
                "FastSurfer installed." if code == 0 else "FastSurfer installation failed (see the log).",
                error=code != 0,
            ),
        )
    return False


def _surface_settings(inputs: ToolInputs) -> tuple[Path | None, Path | None]:
    """FreeSurfer home and license for the surface tools.

    Locally: the tool's fields, remembered for next time. On the cluster the container
    brings its own FreeSurfer (``$FREESURFER_HOME``) and the license is a staged file.
    """
    lic = path_param(inputs.params, "fs_license")
    if isinstance(inputs, StagedInputs):
        return None, lic
    from nvitk.core import external_tools

    fs_home = path_param(inputs.params, "fs_freesurfer_home")
    remember: dict[str, str] = {}
    if fs_home is not None and (fs_home / "SetUpFreeSurfer.sh").is_file():
        remember["freesurfer_home"] = str(fs_home)
    if lic is not None and lic.is_file():
        remember["fs_license"] = str(lic)
    if remember:
        try:
            external_tools.remember(**remember)
        except Exception:  # noqa: BLE001 (remembering is a convenience)
            pass
    return fs_home, lic


# plans


@dataclass
class FastSurferPlan:
    """One FastSurfer command and how to describe its results."""

    argv: list[str]
    env: dict[str, str]
    title: str
    #: Folder the manifest's paths are relative to (the subjects dir, or the run folder).
    work: Path
    manifest: Callable[[], dict[str, Any]]
    #: Minutes to hours (a background job locally).
    long: bool = True
    log_file: Path | None = None
    #: Files that must exist afterwards, else the run failed.
    expects: list[Path] = field(default_factory=list)


def _subject_plan(tool_id: str, inputs: ToolInputs, device: str) -> FastSurferPlan:
    from nvitk.segmentation.fastsurfer import runner
    from nvitk.segmentation.fastsurfer.outputs import MODULE_VOLUMES, SubjectOutputs, subject_manifest

    sd = inputs.work
    sid = _sid(inputs)
    subject = SubjectOutputs(sd / sid)
    existing_seg = subject.has_segmentation()
    module = _MODULE_OF.get(tool_id)
    surfaces = tool_id in ("fastsurfer_surf", "fastsurfer_full")
    segmentation = tool_id != "fastsurfer_surf"
    t1 = None
    if module and existing_seg:
        # The module reuses the whole-brain segmentation; FastSurfer still wants the T1.
        t1 = subject.input_t1()
    elif segmentation:
        t1 = inputs.input()
    t2 = inputs.layer_file("fs_t2_layer", required=False)
    fs_home, lic = _surface_settings(inputs) if (surfaces or inputs.b("fs_tal_reg")) else (None, None)
    if module:
        flags = {m: m == module for m in ("cereb", "hypothal", "cc")}
        load = MODULE_VOLUMES[module] + (() if existing_seg else MODULE_VOLUMES["asegdkt"])
    else:
        flags = {m: inputs.b(f"fs_{m}", True) for m in ("cereb", "hypothal", "cc")}
        load = MODULE_VOLUMES["asegdkt"] + tuple(
            k for m, k in (("cereb", "cereb"), ("hypothal", "hypothal"), ("cc", "callosum")) if flags[m]
        )
        if surfaces:
            load = MODULE_VOLUMES["surfaces"] + load
    req = runner.FastSurferRequest(
        sd=sd, sid=sid, t1=t1, t2=t2, segmentation=segmentation, surfaces=surfaces,
        asegdkt=not (module and existing_seg),
        biasfield=inputs.b("fs_biasfield", True), tal_reg=inputs.b("fs_tal_reg"), qc_snap=inputs.b("fs_qc_snap"),
        reg_mode=inputs.s("fs_reg_mode", "coreg"), vox_size=inputs.s("fs_vox_size", "min"),
        native_image=inputs.b("fs_native_image", True) and not surfaces,
        device=device, viewagg_device=inputs.s("fs_viewagg_device", "auto"), batch=inputs.i("fs_batch", 1),
        threads=inputs.i("fs_threads", 0) or None,
        three_tesla=inputs.b("fs_3t"), parallel=inputs.b("fs_parallel"), no_surfreg=inputs.b("fs_no_surfreg"),
        fsaparc=inputs.b("fs_fsaparc"), ignore_fs_version=inputs.b("fs_ignore_fs_version"),
        freesurfer_home=fs_home, fs_license=lic, **flags,
    )
    argv, env = runner.fastsurfer_argv(req)
    from nvitk.gui.tools.registry import tool_by_id

    spec = tool_by_id(tool_id)
    return FastSurferPlan(
        argv, env, f"FastSurfer {spec.label if spec else tool_id} · {sid}", sd,
        manifest=lambda: subject_manifest(sd / sid, keys=load, surfaces=surfaces, base=sd),
        log_file=sd / sid / "scripts" / "nvitk-job.log",
    )


def _long_plan(inputs: ToolInputs) -> FastSurferPlan:
    from nvitk.segmentation.fastsurfer.outputs import subject_manifest
    from nvitk.segmentation.fastsurfer.runner import long_argv

    names = [n.strip() for n in inputs.s("fs_long_layers").split(",") if n.strip()]
    if len(names) < 2:
        raise ValueError("Name at least two time-point layers (comma-separated).")
    sd = inputs.work
    tid = file_stem(inputs.s("fs_template_id", "template"), "template")
    t1s, tpids = [], []
    for name in names:
        t1s.append(inputs.named_layer_file(name))
        tpids.append(file_stem(name, f"tp{len(tpids) + 1}"))
    fs_home, lic = _surface_settings(inputs)
    argv, env = long_argv(sd, tid, t1s, tpids, parallel=inputs.i("fs_long_parallel", 0) or None,
                          freesurfer_home=fs_home, fs_license=lic)

    def manifest() -> dict[str, Any]:
        merged: dict[str, Any] = {"layers": [], "surfaces": [], "tables": [], "files": []}
        for tp in tpids:
            # Time points are written as <tpid> after a longitudinal run.
            if (sd / tp).is_dir():
                part = subject_manifest(sd / tp, keys=("aparc_mapped", "aparc_aseg"), surfaces=False, base=sd)
                for key in merged:
                    merged[key] += part.get(key, [])
        return merged

    return FastSurferPlan(argv, env, f"FastSurfer longitudinal · {tid}", sd, manifest=manifest,
                          log_file=sd / f"{tid}_nvitk-job.log")


def _quick_plan(tool_id: str, inputs: ToolInputs) -> FastSurferPlan:
    """Module tools that take seconds."""
    from nvitk.segmentation.fastsurfer import runner
    from nvitk.segmentation.fastsurfer.outputs import lut_for, stats_table

    work, name = inputs.work, inputs.name
    if tool_id == "fastsurfer_segstats":
        out = work / f"{name}.stats"
        argv, env = runner.segstats_argv(inputs.input(), out, normfile=inputs.layer_file("fs_norm_layer", required=False))
        manifest = lambda: {"tables": [{"title": "segstats", "payload": stats_table(out), "row_header": "Structure",
                                        "subtitle": out.name}], "files": [out.name]}  # noqa: E731
        return FastSurferPlan(argv, env, "FastSurfer segstats", work, manifest, long=False, expects=[out])
    if tool_id == "fastsurfer_reduce_aseg":
        aseg, mask = work / f"{name}_aseg.nii.gz", work / f"{name}_mask.nii.gz"
        argv, env = runner.reduce_to_aseg_argv(inputs.input(), aseg, out_mask=mask, fixwm=inputs.b("fs_fixwm", True))
        manifest = lambda: {"layers": [  # noqa: E731
            manifest_layer(aseg, "aseg", base=work, labels=True, lut=lut_for("freesurfer"), schema="fastsurfer_aseg"),
            *([manifest_layer(mask, "brain mask", base=work, labels=True)] if mask.is_file() else []),
        ]}
        return FastSurferPlan(argv, env, "FastSurfer reduce to aseg", work, manifest, long=False, expects=[aseg])
    # fastsurfer_conform
    seg_input = inputs.input_is_labels()
    out = work / f"{name}_conformed.mgz"
    argv, env = runner.conform_argv(inputs.input(), out, vox_size=inputs.s("fs_conform_vox", "1"), seg_input=seg_input)
    manifest = lambda: {"layers": [manifest_layer(out, "conformed", base=work, labels=seg_input)]}  # noqa: E731
    return FastSurferPlan(argv, env, "FastSurfer conform", work, manifest, long=False, expects=[out])


def build_fastsurfer(tool_id: str, inputs: ToolInputs, *, device: str) -> FastSurferPlan:
    """The :class:`FastSurferPlan` for *tool_id* on *inputs* (nothing runs)."""
    if tool_id in ("fastsurfer_segstats", "fastsurfer_reduce_aseg", "fastsurfer_conform"):
        return _quick_plan(tool_id, inputs)
    if tool_id == "fastsurfer_long":
        return _long_plan(inputs)
    if tool_id in LOCAL_ONLY_TOOL_IDS or tool_id not in FASTSURFER_TOOL_IDS:
        raise ValueError(f"{tool_id} does not run FastSurfer.")
    return _subject_plan(tool_id, inputs, device)


def run_fastsurfer_headless(tool_id: str, inputs: StagedInputs, *, gpu: bool) -> dict[str, Any]:
    """Run a FastSurfer tool on staged inputs in the foreground (the SGE worker); returns the
    import manifest of what it produced, relative to ``inputs.work``."""
    from nvitk.segmentation.fastsurfer import runner

    plan = build_fastsurfer(tool_id, inputs, device="cuda" if gpu else "cpu")
    code = runner.run(plan.argv, plan.env)
    if code != 0:
        raise RuntimeError(f"{plan.title} failed (exit code {code}).")
    missing = [p.name for p in plan.expects if not p.is_file()]
    if missing:
        raise RuntimeError(f"{plan.title} finished but wrote no {', '.join(missing)}.")
    manifest = plan.manifest()
    manifest["files"] = manifest_files(manifest)
    return manifest


# results


def load_subject(viewer: Any, subject_dir: Path, *, keys: tuple[str, ...] | None = None, surfaces: bool = True,
                 prefix: str = "") -> int:
    """Open a FastSurfer / FreeSurfer subject's volumes (*keys*, else all), surfaces and stats."""
    from nvitk.segmentation.fastsurfer.outputs import subject_manifest

    subject_dir = Path(subject_dir)
    manifest = subject_manifest(subject_dir, keys=keys, surfaces=surfaces)
    return load_manifest(viewer, manifest, subject_dir, prefix=prefix or subject_dir.name)


# tools


def run_fastsurfer_tool(tool_id: str, viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Run one FastSurfer tool of the Tools panel; outputs are added to *viewer*."""
    from nvitk.gui.tools.freesurfer_tools import FASTSURFER_TOPOLOGY_IDS

    if tool_id in FASTSURFER_TOPOLOGY_IDS:
        from nvitk.gui.tools.freesurfer_tools import run_freesurfer_tool

        run_freesurfer_tool(tool_id, viewer, layer, params)
        return
    if tool_id == "fastsurfer_install_freesurfer":
        _install_freesurfer(params)
        return
    if tool_id == "fastsurfer_load_subject":
        subject = path_param(params, "fs_subject_dir")
        if subject is None or not subject.is_dir():
            raise ValueError("Set the subject directory (…/<subjects>/<id>).")
        count = load_subject(viewer, subject, surfaces=bool(params.get("fs_load_surfaces", True)))
        _notify(f"Loaded {count} layer(s) from {subject}.")
        return
    if not _require_fastsurfer():
        return

    inputs = ViewerInputs(tool_id, viewer, layer, params)
    plan = build_fastsurfer(tool_id, inputs, device=_gpu_device())
    prefix = "" if tool_id in ("fastsurfer_long",) else inputs.display_name

    def finished(code: int) -> None:
        if code != 0:
            _notify(f"{plan.title} failed (exit code {code}); see the log in {plan.work}.", error=True)
            return
        missing = [p.name for p in plan.expects if not p.is_file()]
        if missing:
            _notify(f"{plan.title} wrote no {', '.join(missing)}; see the log.", error=True)
            return
        count = load_manifest(viewer, plan.manifest(), plan.work, prefix=prefix)
        _notify(f"{plan.title} finished: {count} layer(s) from {plan.work}.")

    if not plan.long:
        from nvitk.gui.core.log_panel import run_subprocess_logged

        finished(run_subprocess_logged(plan.argv, env=plan.env))
        return

    from nvitk.gui.core.external_jobs import start_external_job
    from nvitk.segmentation.fastsurfer.runner import stage_of

    start_external_job(plan.argv, title=plan.title, env=plan.env, on_finished=finished, stage_of=stage_of,
                       log_file=plan.log_file)
    _notify(f"{plan.title} started in the background (Jobs window); results go to {plan.work}.")


def _install_freesurfer(params: dict[str, Any]) -> None:
    from qtpy.QtWidgets import QMessageBox

    from nvitk.segmentation.fastsurfer import config
    from nvitk.segmentation.fastsurfer.install import free_gb, freesurfer_url

    dest = path_param(params, "fs_install_dest") or config.default_freesurfer_dest()
    lic = path_param(params, "fs_license")
    url = freesurfer_url()
    answer = QMessageBox.question(
        None,
        "Install FreeSurfer",
        f"Download FreeSurfer {config.FREESURFER_VERSION} (≈9.5 GB) from\n{url}\n\nand unpack it in {dest} "
        f"(≈20 GB; {free_gb(dest):.0f} GB free there)?\n\nIt runs in the background and resumes if "
        "interrupted. FreeSurfer is free, but needs a license file (register at "
        "surfer.nmr.mgh.harvard.edu).",
    )
    if answer != QMessageBox.StandardButton.Yes:
        return
    argv = [sys.executable, "-m", "nvitk.segmentation.fastsurfer.cli", "install-freesurfer", "--yes",
            "--dest", str(dest)]
    if lic is not None and lic.is_file():
        argv += ["--license", str(lic)]
    from nvitk.gui.core.external_jobs import start_external_job

    start_external_job(
        argv,
        title="Install FreeSurfer",
        on_finished=lambda code: _notify(
            "FreeSurfer installed." if code == 0 else "FreeSurfer installation stopped (see the log; run again to resume).",
            error=code != 0,
        ),
    )


__all__ = [
    "FASTSURFER_TOOL_IDS",
    "FastSurferPlan",
    "LOCAL_ONLY_TOOL_IDS",
    "SURFACE_TOOL_IDS",
    "build_fastsurfer",
    "load_subject",
    "run_fastsurfer_headless",
    "run_fastsurfer_tool",
]
