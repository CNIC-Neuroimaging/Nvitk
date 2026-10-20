"""FSL tools of the Tools panel (category *FSL*).

Each tool writes the layers it needs to a run folder, runs the :mod:`nvitk.fsl` wrapper — the
same one ``nvitk-fsl`` runs — and opens the outputs as layers placed by their own affine (FSL
reorients, crops and resamples, so nothing is assumed to share the input's grid). Label maps
(masks, FAST/FIRST segmentations, cluster indices) open as Labels layers; reports and tables
(SIENA/SIENAX, fslstats, cluster, smoothest) go to the results window.

Short commands run in the foreground (seconds); long ones — FAST, FIRST, fsl_anat, FNIRT,
EDDY, BEDPOSTX, TBSS … — run as background jobs (Jobs window, cancellable) and open their
outputs when they finish. With no FSL installed the tool offers to install nvitk's FSL core.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nvitk.gui.tools.external_io import (
    StagedInputs,
    ToolInputs,
    ViewerInputs,
    file_stem,
    manifest_files,
    manifest_layer,
    path_param,
)


def _notify(message: str, *, error: bool = False) -> None:
    from nvitk.gui.tools.runner import notify

    notify(message, error=error)


@dataclass
class _Load:
    """One output to open: its role in :attr:`FslCommand.outputs` and how to show it."""

    role: str
    title: str = ""
    labels: bool = False
    visible: bool = True
    lut: dict[int, Any] | None = None


@dataclass
class _Plan:
    loads: list[_Load] = field(default_factory=list)
    #: ``(cmd) -> [(title, payload)]`` tables for the results window.
    tables: Callable[[Any], list[tuple[str, dict]]] | None = None
    #: Also open every NIfTI under this output role's directory (hidden).
    load_dir: str | None = None
    note: Callable[[Any], str] | None = None


# ── builders: tool id → (FslCommand, _Plan) ──────────────────────────────────

_FAST_NAMES = {3: {1: "CSF", 2: "GM", 3: "WM"}, 2: {1: "non-brain/CSF", 2: "tissue"}}


def _bet(c: ToolInputs):
    from nvitk.fsl import structural

    cmd = structural.bet(c.input(), c.out("brain"), f=c.f("fsl_f", 0.5), g=c.f("fsl_g", 0.0),
                         mode=c.s("fsl_bet_mode", "standard"), mask=c.b("fsl_mask", True), skull=c.b("fsl_skull"))
    return cmd, _Plan([_Load("brain", "brain"), _Load("mask", "brain mask", labels=True), _Load("skull", "skull", visible=False)])


def _fast(c: ToolInputs):
    from nvitk.fsl import structural

    classes = c.i("fsl_classes", 3)
    cmd = structural.fast(c.input(), c.out("fast"), classes=classes, img_type=c.s("fsl_img_type", "T1"),
                          no_bias=not c.b("fsl_bias", True))
    names = _FAST_NAMES.get(classes, {})
    loads = [_Load("seg", "FAST tissues", labels=True, lut=names or None), _Load("restore", "bias-corrected")]
    loads += [_Load(f"pve_{k}", f"PVE {names.get(k + 1, k)}", visible=False) for k in range(classes)]
    return cmd, _Plan(loads)


def _first(c: ToolInputs):
    from nvitk.fsl import structural

    names = [s.strip() for s in c.s("fsl_structures").split(",") if s.strip()]
    cmd = structural.first(c.input(), c.out("first"), structures=names, brain_extracted=c.b("fsl_brain_extracted"),
                           method=c.s("fsl_first_method", "auto"))
    return cmd, _Plan([_Load("seg", "FIRST subcortical", labels=True, lut=structural.FIRST_LABELS)])


def _anat(c: ToolInputs):
    from nvitk.fsl import structural

    cmd = structural.fsl_anat(c.input(), c.out("anat"), img_type=c.s("fsl_img_type", "T1"),
                              strong_bias=c.b("fsl_strong_bias"), no_crop=c.b("fsl_no_crop"),
                              no_nonlinreg=c.b("fsl_no_nonlinreg"), no_subcortseg=c.b("fsl_no_subcortseg"))
    fast_names = _FAST_NAMES[3]
    return cmd, _Plan([
        _Load("biascorr", "bias-corrected"), _Load("brain", "brain", visible=False),
        _Load("brain_mask", "brain mask", labels=True, visible=False),
        _Load("fast_seg", "FAST tissues", labels=True, lut=fast_names),
        _Load("subcort_seg", "FIRST subcortical", labels=True, lut=structural.FIRST_LABELS, visible=False),
        _Load("to_mni_nonlin", "in MNI (non-linear)", visible=False),
    ])


def _siena(c: ToolInputs):
    from nvitk.fsl import structural

    second = c.layer_file("fsl_layer2", what="second time point layer")
    cmd = structural.siena(c.input(), second, c.work / "siena", bet_opts=c.s("fsl_bet_opts"),
                           ventricles=c.b("fsl_ventricles"))

    def tables(cmd):
        report = cmd.existing_outputs().get("report")
        return [("SIENA", structural.parse_siena_report(report))] if report else []

    return cmd, _Plan(tables=tables)


def _sienax(c: ToolInputs):
    from nvitk.fsl import structural

    cmd = structural.sienax(c.input(), c.work / "sienax", bet_opts=c.s("fsl_bet_opts"), regional=c.b("fsl_regional"))

    def tables(cmd):
        report = cmd.existing_outputs().get("report")
        return [("SIENAX", structural.parse_sienax_report(report))] if report else []

    return cmd, _Plan(tables=tables, note=lambda cmd: f"SIENAX report: {cmd.outputs['html']}")


def _reorient(c: ToolInputs):
    from nvitk.fsl import structural

    return structural.reorient2std(c.input(), c.out("std")), _Plan([_Load("image", "reoriented to standard")])


def _robustfov(c: ToolInputs):
    from nvitk.fsl import structural

    return (structural.robustfov(c.input(), c.out("fov"), brain_size_mm=c.f("fsl_brain_size", 170.0)),
            _Plan([_Load("image", "cropped FOV")]))


def _deface(c: ToolInputs):
    from nvitk.fsl import structural

    cmd = structural.deface(c.input(), c.out("defaced"), f=c.f("fsl_f", 0.5), bias_correct=c.b("fsl_bias"))
    return cmd, _Plan([_Load("image", "defaced")])


def _lesion_filling(c: ToolInputs):
    from nvitk.fsl import structural

    cmd = structural.lesion_filling(c.input(), c.layer_file("fsl_lesion_layer", what="lesion mask layer"),
                                    c.out("filled"), wm_mask=c.layer_file("fsl_wm_layer", required=False))
    return cmd, _Plan([_Load("image", "lesion-filled")])


def _fnirt(c: ToolInputs):
    from nvitk.fsl import registration

    cmd = registration.fnirt(c.input(), c.layer_file("fsl_ref_layer", what="reference layer"), c.out("fnirt"),
                             affine=c.path("fsl_affine"), config=c.s("fsl_fnirt_config", "T1_2_MNI152_2mm"))
    return cmd, _Plan([_Load("warped", "FNIRT warped"), _Load("field", "warp field", visible=False)],
                      note=lambda cmd: f"Warp coefficients: {cmd.outputs['warpcoef']}")


def _applywarp(c: ToolInputs):
    from nvitk.fsl import registration

    labels = c.input_is_labels()
    interp = "nn" if labels else c.s("fsl_interp", "trilinear")
    cmd = registration.applywarp(c.input(), c.layer_file("fsl_ref_layer", what="reference layer"), c.out("warped"),
                                 warp=c.path("fsl_warp"), premat=c.path("fsl_premat"), interp=interp)
    if cmd.argv and not (c.path("fsl_warp") or c.path("fsl_premat")):
        raise ValueError("Set a warp field and/or an affine (premat).")
    return cmd, _Plan([_Load("image", "warped", labels=labels)])


def _invwarp(c: ToolInputs):
    from nvitk.fsl import registration

    cmd = registration.invwarp(c.input(), c.layer_file("fsl_ref_layer", what="reference layer (original input space)"),
                               c.out("invwarp"))
    return cmd, _Plan([_Load("warp", "inverse warp", visible=False)], note=lambda cmd: f"Inverse warp: {cmd.outputs['warp']}")


def _convert_xfm(c: ToolInputs):
    from nvitk.fsl import registration

    first = c.path("fsl_matrix", required=True, what="matrix")
    out = Path(c.s("fsl_output_matrix") or str(c.work / "converted.mat"))
    if c.s("fsl_xfm_mode", "inverse") == "inverse":
        cmd = registration.convert_xfm(out, inverse=first)
    else:
        cmd = registration.convert_xfm(out, concat=(first, c.path("fsl_matrix2", required=True, what="second matrix")))
    return cmd, _Plan(note=lambda cmd: f"Matrix written: {out}")


def _mcflirt(c: ToolInputs):
    from nvitk.fsl import registration

    ref = c.i("fsl_ref_vol", -1)
    cmd = registration.mcflirt(c.input(), c.out("mcf"), ref_vol=None if ref < 0 else ref,
                               mean_vol=c.b("fsl_mean_vol"), cost=c.s("fsl_cost", "normcorr"))

    def tables(cmd):
        par = cmd.existing_outputs().get("par")
        if not par:
            return []
        import numpy as np

        values = np.loadtxt(par, ndmin=2)
        cols = ("rot x rad", "rot y rad", "rot z rad", "trans x mm", "trans y mm", "trans z mm")
        return [("MCFLIRT motion (max |value|)", {col: float(np.abs(values[:, k]).max()) for k, col in enumerate(cols)})]

    return cmd, _Plan([_Load("image", "motion-corrected")], tables=tables)


def _epi_reg(c: ToolInputs):
    from nvitk.fsl import registration

    cmd = registration.epi_reg(c.input(), c.layer_file("fsl_t1_layer", what="T1 layer"),
                               c.layer_file("fsl_t1brain_layer", what="brain-extracted T1 layer"), c.out("epi2t1"),
                               wmseg=c.layer_file("fsl_wmseg_layer", required=False))
    return cmd, _Plan([_Load("image", "EPI in T1 space")], note=lambda cmd: f"Matrix: {cmd.outputs['matrix']}")


def _topup(c: ToolInputs):
    from nvitk.fsl import diffusion

    cmd = diffusion.topup(c.input(), c.path("fsl_datain", required=True, what="acquisition parameters (datain)"),
                          c.out("topup"), config=c.s("fsl_topup_config", "b02b0.cnf"))
    return cmd, _Plan([_Load("corrected", "TOPUP corrected b0"), _Load("field", "TOPUP field", visible=False)],
                      note=lambda cmd: f"TOPUP basename for EDDY / applytopup: {c.out('topup')}")


def _eddy(c: ToolInputs):
    from nvitk.fsl import diffusion

    cmd = diffusion.eddy(c.input(), c.layer_file("fsl_mask_layer", what="brain mask layer"),
                         c.path("fsl_acqp", required=True, what="acqp"), c.path("fsl_index", required=True, what="index"),
                         c.bvecs(), c.bvals(), c.out("eddy"), topup_base=path_param(c.params, "fsl_topup_base"),
                         repol=c.b("fsl_repol", True))
    return cmd, _Plan([_Load("image", "EDDY corrected")],
                      note=lambda cmd: f"Rotated bvecs: {cmd.outputs['rotated_bvecs']}")


def _dtifit(c: ToolInputs):
    from nvitk.fsl import diffusion

    cmd = diffusion.dtifit(c.input(), c.layer_file("fsl_mask_layer", what="brain mask layer"), c.bvecs(), c.bvals(),
                           c.out("dti"), wls=c.b("fsl_wls"))
    return cmd, _Plan([_Load("FA", "FA"), _Load("MD", "MD", visible=False), _Load("L1", "L1 (AD)", visible=False),
                       _Load("V1", "V1", visible=False), _Load("S0", "S0", visible=False)])


def _bedpostx(c: ToolInputs):
    from nvitk.fsl import diffusion

    folder = diffusion.prepare_bedpostx_dir(c.work / "bedpostx", c.input(),
                                            c.layer_file("fsl_mask_layer", what="brain mask layer"), c.bvecs(), c.bvals())
    cmd = diffusion.bedpostx(folder, fibres=c.i("fsl_fibres", 3))
    return cmd, _Plan([_Load("mean_f1", "mean f1"), _Load("dyads1", "dyads1", visible=False)],
                      note=lambda cmd: f"BEDPOSTX result: {cmd.outputs['dir']} (use it for PROBTRACKX2 / XTRACT)")


def _probtrackx2(c: ToolInputs):
    from nvitk.fsl import diffusion

    bpx = c.path("fsl_bedpostx_dir", required=True, what="BEDPOSTX folder")
    cmd = diffusion.probtrackx2(bpx, c.input(), c.work / "probtrackx", samples=c.i("fsl_samples", 5000),
                                waypoints=c.layer_file("fsl_waypoint_layer", required=False),
                                exclusion=c.layer_file("fsl_exclusion_layer", required=False))
    return cmd, _Plan([_Load("paths", "probtrackx paths")])


def _tbss(c: ToolInputs):
    from nvitk.fsl import diffusion

    raw = c.s("fsl_fa_list")
    if not raw:
        raise ValueError("List the FA maps (comma-separated paths, or one folder holding them).")
    entries = [Path(p.strip()).expanduser() for p in raw.split(",") if p.strip()]
    images: list[Path] = []
    for entry in entries:
        images += sorted(entry.glob("*.nii*")) if entry.is_dir() else [entry]
    if len(images) < 2:
        raise ValueError("TBSS needs at least two FA maps.")
    folder = diffusion.prepare_tbss_dir(c.work / "tbss", images)
    cmd = diffusion.tbss(folder, target=c.s("fsl_tbss_target", "FMRIB58"), threshold=c.f("fsl_threshold", 0.2))
    return cmd, _Plan([_Load("mean_FA", "TBSS mean FA"), _Load("skeleton", "TBSS mean FA skeleton"),
                       _Load("skeleton_mask", "TBSS skeleton mask", labels=True, visible=False),
                       _Load("all_FA_skeletonised", "TBSS all FA skeletonised (4D)", visible=False)],
                      note=lambda cmd: f"TBSS stats folder: {cmd.outputs['dir']} — all_FA_skeletonised is the 4D input "
                                       "for randomise (Voxelwise tool / nvitk-voxelwise).")


def _xtract(c: ToolInputs):
    from nvitk.fsl import diffusion

    warps = (c.path("fsl_std2diff"), c.path("fsl_diff2std"))
    cmd = diffusion.xtract(c.path("fsl_bedpostx_dir", required=True, what="BEDPOSTX folder"), c.work / "xtract",
                           species=c.s("fsl_species", "HUMAN"), std_warps=warps if all(warps) else None,
                           tracts=[t.strip() for t in c.s("fsl_tracts").split(",") if t.strip()])
    return cmd, _Plan(load_dir="tracts", note=lambda cmd: f"XTRACT output: {cmd.outputs['dir']}")


def _melodic(c: ToolInputs):
    from nvitk.fsl import functional

    tr = c.f("fsl_tr", 0.0)
    cmd = functional.melodic(c.input(), c.work / "melodic", dim=c.i("fsl_dim", 0),
                             approach=c.s("fsl_approach", "symm"), tr=tr or None)
    return cmd, _Plan([_Load("ics", "MELODIC ICs"), _Load("mean", "mean", visible=False)],
                      note=lambda cmd: f"MELODIC report: {cmd.outputs['report']}")


def _susan(c: ToolInputs):
    from nvitk.fsl import functional

    cmd = functional.susan(c.input(), c.out("susan"), sigma_mm=c.f("fsl_sigma", 2.0),
                           brightness_threshold=c.f("fsl_bt", -1.0))
    return cmd, _Plan([_Load("image", "SUSAN")])


def _slicetimer(c: ToolInputs):
    from nvitk.fsl import functional

    cmd = functional.slicetimer(c.input(), c.out("st"), tr=c.f("fsl_tr", 2.0), order=c.s("fsl_slice_order", "bottom-up"),
                                timings=c.path("fsl_timings"))
    return cmd, _Plan([_Load("image", "slice-time corrected")])


def _fugue(c: ToolInputs):
    from nvitk.fsl import functional

    cmd = functional.fugue(c.input(), c.out("unwarped"), c.layer_file("fsl_fmap_layer", what="field map layer (rad/s)"),
                           dwell_time=c.f("fsl_dwell", 0.0005), unwarp_dir=c.s("fsl_unwarpdir", "y"))
    return cmd, _Plan([_Load("image", "unwarped")])


def _prepare_fieldmap(c: ToolInputs):
    from nvitk.fsl import functional

    cmd = functional.prepare_fieldmap(c.input(), c.layer_file("fsl_mag_layer", what="brain-extracted magnitude layer"),
                                      c.out("fmap_rads"), delta_te_ms=c.f("fsl_delta_te", 2.46),
                                      scanner=c.s("fsl_scanner", "SIEMENS"))
    return cmd, _Plan([_Load("fieldmap", "field map (rad/s)")])


def _maths(c: ToolInputs):
    from nvitk.fsl import utils

    ops = c.s("fsl_ops")
    if not ops:
        raise ValueError('Give the fslmaths operations, e.g. "-thr 100 -bin" or "-s 2".')
    binary = any(tok in ops.split() for tok in ("-bin", "-binv"))
    cmd = utils.maths(c.input(), _expand_layers(c, ops), c.out("maths"), odt=c.s("fsl_odt") or None)
    return cmd, _Plan([_Load("image", f"fslmaths {ops}", labels=binary)])


def _stats(c: ToolInputs):
    from nvitk.fsl import utils

    options = c.s("fsl_stats_options", "-R -r -M -S -V").split()
    cmd = utils.stats(c.input(), options, mask=c.layer_file("fsl_mask_layer", required=False))
    return cmd, _Plan(tables=lambda cmd: [("fslstats", utils.parse_stats(cmd.stdout, options))])


def _meants(c: ToolInputs):
    from nvitk.fsl import utils

    out = c.work / f"{c.name}_meants.txt"
    cmd = utils.meants(c.input(), out, mask=c.layer_file("fsl_mask_layer", what="mask layer"))

    def tables(cmd):
        import numpy as np

        values = np.loadtxt(out, ndmin=1) if out.is_file() else []
        return [("fslmeants", {f"t{k}": float(v) for k, v in enumerate(np.ravel(values))})]

    return cmd, _Plan(tables=tables)


def _cluster(c: ToolInputs):
    from nvitk.fsl import utils

    cmd = utils.cluster(c.input(), c.f("fsl_threshold", 3.1), c.out("cluster"),
                        connectivity=int(c.s("fsl_connectivity", "26")), min_extent=c.i("fsl_min_extent", 0))
    return cmd, _Plan([_Load("index", "clusters", labels=True)],
                      tables=lambda cmd: [("cluster", utils.parse_cluster_table(cmd.stdout))])


def _smoothest(c: ToolInputs):
    from nvitk.fsl import utils

    cmd = utils.smoothest(zstat=c.input(), mask=c.layer_file("fsl_mask_layer", what="mask layer"))
    return cmd, _Plan(tables=lambda cmd: [("smoothest", utils.parse_smoothest(cmd.stdout))])


_LAYER_TOKEN = re.compile(r"\{layer:([^}]+)\}")
_OUT_TOKEN = re.compile(r"\{out:([^}]+)\}")


def _expand_layers(c: ToolInputs, text: str) -> str:
    """``{layer:NAME}`` → that layer's file (quoted)."""
    return _LAYER_TOKEN.sub(lambda m: shlex.quote(str(c.named_layer_file(m.group(1)))), text)


def _command(c: ToolInputs):
    from nvitk.fsl import utils

    text = c.s("fsl_command")
    if not text:
        raise ValueError("Write the FSL command, e.g. fslmaths {layer:T1} -s 2 {out:smooth}")
    outputs: dict[str, Path] = {}

    def out_path(m: re.Match) -> str:
        name = file_stem(m.group(1), "out")
        outputs[name] = c.work / f"{name}.nii.gz"
        return shlex.quote(str(c.work / name))

    text = _OUT_TOKEN.sub(out_path, _expand_layers(c, text))
    cmd = utils.command(text, title=f"FSL: {text.split()[0]}", long=c.b("fsl_long"), outputs=outputs)
    return cmd, _Plan([_Load(role, role) for role in outputs], tables=lambda cmd: (
        [("FSL command output", {f"line {k + 1}": line for k, line in enumerate(cmd.stdout.splitlines()[-40:])})]
        if cmd.stdout.strip() and not outputs else []))


_BUILDERS: dict[str, Callable[[ToolInputs], tuple[Any, _Plan]]] = {
    "fsl_bet": _bet, "fsl_fast": _fast, "fsl_first": _first, "fsl_anat": _anat, "fsl_siena": _siena,
    "fsl_sienax": _sienax, "fsl_reorient2std": _reorient, "fsl_robustfov": _robustfov, "fsl_deface": _deface,
    "fsl_lesion_filling": _lesion_filling, "fsl_fnirt": _fnirt, "fsl_applywarp": _applywarp, "fsl_invwarp": _invwarp,
    "fsl_convert_xfm": _convert_xfm, "fsl_mcflirt": _mcflirt, "fsl_epi_reg": _epi_reg, "fsl_topup": _topup,
    "fsl_eddy": _eddy, "fsl_dtifit": _dtifit, "fsl_bedpostx": _bedpostx, "fsl_probtrackx2": _probtrackx2,
    "fsl_tbss": _tbss, "fsl_xtract": _xtract, "fsl_melodic": _melodic, "fsl_susan": _susan,
    "fsl_slicetimer": _slicetimer, "fsl_fugue": _fugue, "fsl_prepare_fieldmap": _prepare_fieldmap,
    "fsl_maths": _maths, "fsl_stats": _stats, "fsl_meants": _meants, "fsl_cluster": _cluster,
    "fsl_smoothest": _smoothest, "fsl_command": _command,
}

#: Tool ids handled here.
FSL_TOOL_IDS: frozenset[str] = frozenset(_BUILDERS) | {"fsl_install"}


# ── running ──────────────────────────────────────────────────────────────────


def plan_manifest(cmd: Any, plan: _Plan, base: Path) -> dict[str, Any]:
    """The import manifest of a finished *cmd* (see :mod:`nvitk.gui.tools.external_io`), with
    paths relative to *base*. Built without Qt: a cluster job sends it back in its ``.done``."""
    found = cmd.existing_outputs()
    layers = []
    for load in plan.loads:
        path = found.get(load.role)
        if path is None or path.is_dir() or not path.name.endswith((".nii", ".nii.gz")):
            continue
        layers.append(manifest_layer(path, load.title or load.role, base=base, labels=load.labels,
                                     visible=load.visible, lut=load.lut))
    if plan.load_dir:
        folder = found.get(plan.load_dir)
        if folder is not None and folder.is_dir():
            for path in sorted(folder.rglob("densityNorm.nii.gz")) or sorted(folder.rglob("*.nii.gz")):
                layers.append(manifest_layer(path, path.parent.name, base=base, visible=False, colormap="hot"))
    tables = []
    if plan.tables is not None:
        for title, payload in plan.tables(cmd):
            if payload:
                tables.append({"title": title, "payload": _jsonable(payload)})
    notes = [plan.note(cmd)] if plan.note is not None else []
    # Small text outputs (matrices, motion parameters, reports) come back too.
    extra = [
        str(Path(p).resolve().relative_to(Path(base).resolve()))
        for role, p in found.items()
        if Path(p).is_file() and not Path(p).name.endswith((".nii", ".nii.gz")) and Path(p).stat().st_size < 5_000_000
        and Path(p).resolve().is_relative_to(Path(base).resolve())
    ]
    return {"layers": layers, "tables": tables, "notes": notes, "files": extra}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _finish(viewer: Any, cmd: Any, plan: _Plan, inputs: ToolInputs) -> int:
    """Open *cmd*'s outputs per *plan*; returns how many layers were added."""
    from nvitk.gui.tools.external_io import load_manifest

    return load_manifest(viewer, plan_manifest(cmd, plan, inputs.work), inputs.work, prefix=inputs.display_name)


def build(tool_id: str, inputs: ToolInputs) -> tuple[Any, _Plan]:
    """``(FslCommand, plan)`` for *tool_id* on *inputs* (nothing runs)."""
    builder = _BUILDERS.get(tool_id)
    if builder is None:
        raise NotImplementedError(f"FSL tool {tool_id!r} is not implemented.")
    return builder(inputs)


def run_fsl_headless(tool_id: str, inputs: StagedInputs) -> dict[str, Any]:
    """Run an FSL tool on staged inputs in the foreground (the SGE worker) and return the
    import manifest of what it produced, relative to ``inputs.work``."""
    cmd, plan = build(tool_id, inputs)
    cmd.run(on_line=lambda line: print(line, flush=True))
    manifest = plan_manifest(cmd, plan, inputs.work)
    manifest["files"] = manifest_files(manifest)
    return manifest


def _offer_install(reason: str) -> None:
    from qtpy.QtWidgets import QMessageBox

    answer = QMessageBox.question(
        None,
        "FSL is not available",
        f"{reason}\n\nInstall nvitk's own FSL core now (FSL's conda packages, a few GB, into this "
        "environment)? It runs in the background; run the tool again when it is done.",
    )
    if answer == QMessageBox.StandardButton.Yes:
        _start_install({})


def _start_install(params: dict[str, Any]) -> None:
    from nvitk.gui.core.external_jobs import start_external_job

    components = [c for c in ("structural", "registration", "diffusion", "functional")
                  if params.get(f"fsl_install_{c}", True)]
    argv = [sys.executable, "-m", "nvitk.fsl.cli", "install", "-y", "--components", ",".join(components) or "structural"]
    if "fsl_install_cuda" in params:
        argv.append("--cuda" if params.get("fsl_install_cuda") else "--no-cuda")
    start_external_job(
        argv, title="Install FSL core",
        on_finished=lambda code: _notify("FSL core installed." if code == 0 else "FSL installation failed (see the log).",
                                         error=code != 0),
    )
    _notify("Installing nvitk's FSL core in the background (Jobs window).")


def run_fsl_tool(tool_id: str, viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Run one FSL tool of the Tools panel; outputs are added to *viewer*."""
    from nvitk.fsl.env import FslNotFound

    if tool_id == "fsl_install":
        from qtpy.QtWidgets import QMessageBox

        if QMessageBox.question(None, "Install FSL core", "Download FSL's conda packages (a few GB) and create "
                                "nvitk's FSL core in this environment?") == QMessageBox.StandardButton.Yes:
            _start_install(params)
        return
    ctx = ViewerInputs(tool_id, viewer, layer, params)
    cmd, plan = build(tool_id, ctx)
    try:
        environment = cmd.environment()
    except FslNotFound as exc:
        _offer_install(str(exc))
        return
    prefix = ctx.display_name

    if cmd.long:
        from nvitk.gui.core.external_jobs import start_external_job

        def finished(code: int) -> None:
            if code != 0:
                _notify(f"{cmd.title} failed (exit code {code}); see the log in {ctx.work}.", error=True)
                return
            n = _finish(viewer, cmd, plan, ctx)
            _notify(f"{cmd.title} finished: {n} layer(s) from {ctx.work}.")

        # The job's output is also wanted as ``cmd.stdout`` by the tables.
        lines: list[str] = []

        def stage(line: str) -> None:
            lines.append(line)
            cmd.stdout = "\n".join(lines[-2000:])
            return None

        start_external_job(cmd.full_argv(), title=f"{cmd.title} · {prefix}", env=environment, cwd=cmd.cwd,
                           on_finished=finished, stage_of=stage, log_file=ctx.work / "nvitk-job.log")
        _notify(f"{cmd.title} started in the background (Jobs window); outputs go to {ctx.work}.")
        return

    from nvitk.gui.core.log_panel import run_subprocess_logged

    lines: list[str] = []
    code = run_subprocess_logged(cmd.full_argv(), cwd=str(cmd.cwd) if cmd.cwd else None, env=environment,
                                 on_line=lines.append)
    cmd.stdout = "\n".join(lines)
    if code != 0:
        raise RuntimeError(f"{cmd.title} failed (exit code {code}); see the log.")
    n = _finish(viewer, cmd, plan, ctx)
    _notify(f"{cmd.title}: {n} layer(s) from {ctx.work}." if n else f"{cmd.title} done ({ctx.work}).")


__all__ = ["FSL_TOOL_IDS", "build", "plan_manifest", "run_fsl_headless", "run_fsl_tool"]
