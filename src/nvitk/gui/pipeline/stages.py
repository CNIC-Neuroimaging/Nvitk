"""Pipeline stage metadata for the Napari GUI pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageInputSpec:
    """One layer or on-disk input required by a pipeline stage."""

    slot: str
    label: str
    optional: bool = False
    from_stage: str | None = None


@dataclass(frozen=True)
class PipelineStageSpec:
    id: str
    label: str
    description: str = ""
    default_enabled: bool = True
    inputs: tuple[StageInputSpec, ...] = ()


@dataclass(frozen=True)
class PipelineStageDef:
    cli_command: str
    title: str
    stages: tuple[PipelineStageSpec, ...]
    hidden_params: frozenset[str] = frozenset({"stages", "help"})


@dataclass
class StageInputBinding:
    stage_id: str
    input_name: str
    source: str
    layer_name: str | None = None


# ---------------------------------------------------------------------------
# Stage registry helpers
# ---------------------------------------------------------------------------

DOWNLOAD_STAGE_IDS: frozenset[str] = frozenset({"stage0_d", "download"})
CONVERT_STAGE_IDS: frozenset[str] = frozenset(
    {"stage0_c", "convert", "stage0", "stage0_convert"}
)


def _default_enabled(stage_id: str, defaults: str) -> bool:
    default_ids = {s.strip() for s in defaults.split(",") if s.strip()}
    return stage_id in default_ids


def _exclude_setup_stages(
    specs: tuple[PipelineStageSpec, ...],
) -> tuple[PipelineStageSpec, ...]:
    skip = DOWNLOAD_STAGE_IDS | CONVERT_STAGE_IDS
    return tuple(s for s in specs if s.id not in skip)


def _qvtpy_stages() -> tuple[PipelineStageSpec, ...]:
    from nvitk.pipes.qvtpy.run import DEFAULT_STAGES, _STAGE_LABELS, _STAGES_ORDERED

    return _exclude_setup_stages(
        tuple(
            PipelineStageSpec(
                sid,
                _STAGE_LABELS.get(sid, sid),
                _QVT_DESCRIPTIONS.get(sid, ""),
                default_enabled=_default_enabled(sid, DEFAULT_STAGES),
                inputs=_QVT_INPUTS.get(sid, ()),
            )
            for sid in _STAGES_ORDERED
        )
    )


def _bbtpy_stages() -> tuple[PipelineStageSpec, ...]:
    from nvitk.pipes.bbtpy.run import _DEFAULT_STAGES, _STAGE_LABELS

    order = ("stage1", "stage2")
    return tuple(
        PipelineStageSpec(
            sid,
            _STAGE_LABELS[sid],
            _BBT_DESCRIPTIONS.get(sid, ""),
            default_enabled=_default_enabled(sid, _DEFAULT_STAGES),
            inputs=_BBT_INPUTS.get(sid, ()),
        )
        for sid in order
    )


def _gpetpy_stages() -> tuple[PipelineStageSpec, ...]:
    defaults = "stage1"
    order = ("stage1",)
    labels = {"stage1": "PET brain crop"}
    return tuple(
        PipelineStageSpec(
            sid,
            labels[sid],
            _GPET_DESCRIPTIONS.get(sid, ""),
            default_enabled=_default_enabled(sid, defaults),
            inputs=_GPET_INPUTS.get(sid, ()),
        )
        for sid in order
    )


def _pesa_ctpet_stages() -> tuple[PipelineStageSpec, ...]:
    labels = {
        "stage1": "Segment (TotalSegmentator)",
        "stage2": "Post-process masks",
        "stage3": "Measure & export",
    }
    defaults = "stage1,stage2,stage3"
    order = ("stage1", "stage2", "stage3")
    return tuple(
        PipelineStageSpec(
            sid,
            labels[sid],
            _PESA_CTPET_DESCRIPTIONS.get(sid, ""),
            default_enabled=_default_enabled(sid, defaults),
            inputs=_PESA_CTPET_INPUTS.get(sid, ()),
        )
        for sid in order
    )


def _pesa_dixon_stages() -> tuple[PipelineStageSpec, ...]:
    labels = {
        "stage1": "Segment (Dixon fat)",
        "stage2": "Post-process masks",
        "stage3": "Measure & export",
    }
    defaults = "stage1,stage2,stage3"
    order = ("stage1", "stage2", "stage3")
    return tuple(
        PipelineStageSpec(
            sid,
            labels[sid],
            _PESA_DIXON_DESCRIPTIONS.get(sid, ""),
            default_enabled=_default_enabled(sid, defaults),
            inputs=_PESA_DIXON_INPUTS.get(sid, ()),
        )
        for sid in order
    )


# ---------------------------------------------------------------------------
# Per-pipeline stage inputs and descriptions
# ---------------------------------------------------------------------------

_QVT_DESCRIPTIONS: dict[str, str] = {
    "stage1": "Run eICAB on the TOF angiography volume.",
    "stage2": "Rigid FLIRT: eICAB TOF_resampled → 4D flow angiography or complex difference.",
    "stage3": "Warp eICAB to 4D flow space and extract arterial / venous centerlines.",
    "stage4": "Per-vessel CD segmentation using centerline ROIs.",
    "stage4t": "Per-phase CD segmentation on ComplexDifference_4D.",
    "stage5": "Generate arterial and venous LOC CSV from centerlines.",
    "stage6": "Per-LOC flow, PI, and RI from phase volumes.",
}

_QVT_INPUTS: dict[str, tuple[StageInputSpec, ...]] = {
    "stage1": (
        StageInputSpec("tof", "TOF angiography (TOF.nii.gz)"),
    ),
    "stage2": (
        StageInputSpec("eicab", "eICAB TOF_resampled", from_stage="stage1"),
        StageInputSpec("fixed_angio", "4D flow angiography or complex difference"),
    ),
    "stage3": (
        StageInputSpec("eicab_4d", "eICAB in 4D flow space", from_stage="stage2"),
        StageInputSpec("cd", "Complex difference (ComplexDifference_3D)"),
    ),
    "stage4": (
        StageInputSpec("cd", "Complex difference (ComplexDifference_3D)"),
        StageInputSpec("centerlines", "Arterial / venous centerlines", from_stage="stage3"),
    ),
    "stage4t": (
        StageInputSpec("cd_4d", "Complex difference 4D (ComplexDifference_4D)"),
        StageInputSpec("centerlines", "Centerlines", from_stage="stage3"),
    ),
    "stage5": (
        StageInputSpec("centerlines", "Centerlines", from_stage="stage3"),
        StageInputSpec("cd", "Complex difference", optional=True),
    ),
    "stage6": (
        StageInputSpec("locs", "LOC CSV", from_stage="stage5"),
        StageInputSpec("ap", "AP phase volume"),
        StageInputSpec("rl", "RL phase volume"),
        StageInputSpec("fh", "FH phase volume"),
        StageInputSpec("seg", "4D flow segmentation", from_stage="stage4", optional=True),
    ),
}

_BBT_DESCRIPTIONS: dict[str, str] = {
    "stage1": "Rigid FLIRT: eICAB TOF_resampled → native black-blood VWI.",
    "stage2": "Centerline QC and hypointense BB vessel segmentation.",
}

_BBT_INPUTS: dict[str, tuple[StageInputSpec, ...]] = {
    "stage1": (
        StageInputSpec("vwi_bb", "Black-blood VWI (vwi_bb.nii.gz)"),
        StageInputSpec("eicab", "eICAB TOF_resampled (from QVTpy)"),
    ),
    "stage2": (
        StageInputSpec("vwi_bb", "Black-blood VWI"),
        StageInputSpec("eicab_warped", "eICAB warped to VWI", from_stage="stage1"),
    ),
}

_GPET_DESCRIPTIONS: dict[str, str] = {
    "stage1": "Crop PET to brain using CT TotalSegmentator mask.",
}

_GPET_INPUTS: dict[str, tuple[StageInputSpec, ...]] = {
    "stage1": (
        StageInputSpec("pet", "PET volume (PT.nii.gz)"),
        StageInputSpec("ct", "CT volume (optional, for brain mask)", optional=True),
    ),
}

_PESA_CTPET_DESCRIPTIONS: dict[str, str] = {
    "stage1": "TotalSegmentator on CT; writes organ / fat masks.",
    "stage2": "Post-process masks; uses PET for bladder / fat cleanup.",
    "stage3": "Measure volumes and SUV; export Excel.",
}

_PESA_CTPET_INPUTS: dict[str, tuple[StageInputSpec, ...]] = {
    "stage1": (
        StageInputSpec("ct", "CT volume"),
        StageInputSpec("pet", "PET volume (PT.nii.gz)"),
    ),
    "stage2": (
        StageInputSpec("seg_ct", "Stage-1 CT segmentation", from_stage="stage1"),
        StageInputSpec("pet", "PET volume", from_stage="stage1"),
    ),
    "stage3": (
        StageInputSpec("masks", "Post-processed masks", from_stage="stage2"),
        StageInputSpec("pet", "PET volume", from_stage="stage1"),
    ),
}

_PESA_DIXON_DESCRIPTIONS: dict[str, str] = {
    "stage1": "Segment Dixon fat regions per body part.",
    "stage2": "Post-process Dixon masks.",
    "stage3": "Measure fat fraction / T2* and export Excel.",
}

_PESA_DIXON_INPUTS: dict[str, tuple[StageInputSpec, ...]] = {
    "stage1": (
        StageInputSpec("ff", "Fat fraction map"),
        StageInputSpec("water", "Water map", optional=True),
    ),
    "stage2": (
        StageInputSpec("seg_dixon", "Stage-1 Dixon segmentation", from_stage="stage1"),
    ),
    "stage3": (
        StageInputSpec("masks", "Post-processed masks", from_stage="stage2"),
        StageInputSpec("ff", "Fat fraction maps", from_stage="stage1"),
    ),
}

PIPELINE_STAGE_DEFS: dict[str, PipelineStageDef] = {
    "nvitk-pesa-fat-ctpet": PipelineStageDef(
        "nvitk-pesa-fat-ctpet",
        "PESA-Fat CT-PET",
        _pesa_ctpet_stages(),
    ),
    "nvitk-pesa-fat-dixon": PipelineStageDef(
        "nvitk-pesa-fat-dixon",
        "PESA-Fat DIXON",
        _pesa_dixon_stages(),
    ),
    "nvitk-qvtpy": PipelineStageDef(
        "nvitk-qvtpy",
        "QVTPy (4DFlows)",
        _qvtpy_stages(),
    ),
    "nvitk-bbtpy": PipelineStageDef(
        "nvitk-bbtpy",
        "BBTPy",
        _bbtpy_stages(),
    ),
    "nvitk-gpetpy": PipelineStageDef(
        "nvitk-gpetpy",
        "GPETPy",
        _gpetpy_stages(),
    ),
}


def pipeline_def_for_script(script_name: str) -> PipelineStageDef | None:
    return PIPELINE_STAGE_DEFS.get(script_name)


def _input_key(stage_id: str, slot: str) -> str:
    return f"{stage_id}:{slot}"


def visible_stage_inputs(
    stages: tuple[PipelineStageSpec, ...],
    enabled: dict[str, bool],
    stage_index: int,
) -> tuple[str, ...]:
    """Return UI keys for inputs that still need a Napari layer selection."""
    stage = stages[stage_index]
    if not enabled.get(stage.id, False):
        return ()

    checked_indices = [i for i, st in enumerate(stages) if enabled.get(st.id, False)]
    if stage_index not in checked_indices:
        return ()

    first_checked = checked_indices[0]
    visible = []

    if not stage.inputs:
        if stage_index == first_checked:
            return (_input_key(stage.id, "active"),)
        return (_input_key(stage.id, "primary"),)

    for i, inp in enumerate(stage.inputs):
        if inp.from_stage and enabled.get(inp.from_stage, False):
            continue
        if stage_index == first_checked and i == 0:
            visible.append(_input_key(stage.id, "active"))
        else:
            visible.append(_input_key(stage.id, inp.slot))

    return tuple(visible)


def resolve_stage_input_bindings(
    stages: tuple[PipelineStageSpec, ...],
    enabled: dict[str, bool],
    *,
    active_layer_name: str | None,
    layer_selections: dict[tuple[str, str], str | None],
) -> list[StageInputBinding]:
    """Resolve how each checked stage obtains its layer inputs."""
    checked_indices = [i for i, st in enumerate(stages) if enabled.get(st.id, False)]
    if not checked_indices:
        return []

    first_checked = checked_indices[0]
    bindings = []

    for idx in checked_indices:
        stage = stages[idx]

        if not stage.inputs:
            if idx == first_checked:
                bindings.append(
                    StageInputBinding(
                        stage.id,
                        "primary",
                        "active",
                        layer_name=active_layer_name,
                    )
                )
            else:
                layer_name = layer_selections.get((stage.id, "primary"))
                bindings.append(
                    StageInputBinding(
                        stage.id,
                        "primary",
                        "layer",
                        layer_name=layer_name,
                    )
                )
            continue

        for inp in stage.inputs:
            if inp.from_stage and enabled.get(inp.from_stage, False):
                bindings.append(
                    StageInputBinding(stage.id, inp.slot, "chained")
                )
                continue

            if idx == first_checked and inp is stage.inputs[0]:
                bindings.append(
                    StageInputBinding(
                        stage.id,
                        inp.slot,
                        "active",
                        layer_name=active_layer_name,
                    )
                )
                continue

            layer_name = layer_selections.get((stage.id, inp.slot))
            bindings.append(
                StageInputBinding(
                    stage.id,
                    inp.slot,
                    "layer",
                    layer_name=layer_name,
                )
            )

    return bindings


def binding_requires_layer(
    stages: tuple[PipelineStageSpec, ...],
    binding: StageInputBinding,
) -> bool:
    """True when a layer dropdown must be filled before running."""
    if binding.source != "layer":
        return False
    stage = next(s for s in stages if s.id == binding.stage_id)
    for inp in stage.inputs:
        if inp.slot == binding.input_name:
            return not inp.optional
    return binding.input_name in ("primary", "active")


def input_label_for_key(
    stages: tuple[PipelineStageSpec, ...],
    key: str,
) -> str:
    """Human-readable label for a stage input UI key."""
    if key.endswith(":active") or key.endswith(":primary"):
        stage_id = key.split(":", 1)[0]
        stage = next(s for s in stages if s.id == stage_id)
        if stage.inputs:
            return stage.inputs[0].label
        return "Primary input"
    stage_id, slot = key.split(":", 1)
    stage = next(s for s in stages if s.id == stage_id)
    for inp in stage.inputs:
        if inp.slot == slot:
            suffix = " (optional)" if inp.optional else ""
            return f"{inp.label}{suffix}"
    return slot.replace("_", " ").title()


__all__ = [
    "CONVERT_STAGE_IDS",
    "DOWNLOAD_STAGE_IDS",
    "PIPELINE_STAGE_DEFS",
    "PipelineStageDef",
    "PipelineStageSpec",
    "StageInputBinding",
    "StageInputSpec",
    "binding_requires_layer",
    "input_label_for_key",
    "pipeline_def_for_script",
    "resolve_stage_input_bindings",
    "visible_stage_inputs",
]
