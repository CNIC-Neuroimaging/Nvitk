"""ANTsPy / ANTsPyNet tools in the GUI: registry entries and their dispatcher.

The library wrappers live in :mod:`nvitk.segmentation.antsxnet_tools` and
:mod:`nvitk.registration.ants_extra`; here they become tool-browser entries (Segmentation,
Preprocess, Measure... with ``ANTs`` / ``ANTsPyNet`` badges) and run on the active layer, with a
second input picked from the layer list where a tool needs one. Multi-atlas fusion, template
building and the registration presets stay on the command line (``nvitk-ants jlf|template|
register --preset``), where their many inputs are files.
"""

from __future__ import annotations

from typing import Any

from nvitk.gui.tools.registry import GuiToolSpec, ParamSpec
from nvitk.gui.tools.taxonomy import T

_L2 = "second_layer"


def _spec(tool_id: str, category: str, label: str, meta: Any, params: tuple[ParamSpec, ...] = (), *,
          run_mode: str = "layer", description: str = "") -> GuiToolSpec:
    return GuiToolSpec(tool_id, category, label, params, needs_3d=True, run_mode=run_mode,  # type: ignore[arg-type]
                       cli_command="nvitk-ants", description=description, meta=meta)


ANTS_TOOLS: tuple[GuiToolSpec, ...] = (
    _spec("antsxnet_deep_atropos", "Segmentation", "Deep Atropos tissues (ANTsPyNet)",
          T("segmentation", "Tissue", "smri", "neuro", suite="ANTsPyNet"),
          description="Six tissues from a T1: CSF, gray matter, white matter, deep gray matter, brain stem, cerebellum."),
    _spec("antsxnet_cortical_thickness", "Measure", "Cortical thickness (ANTsPyNet)",
          T("measure", "Morphometry", "smri", "neuro", suite="ANTsPyNet"),
          description="DiReCT cortical thickness map from a T1 (via Deep Atropos)."),
    _spec("antsxnet_hippocampus", "Segmentation", "Hippocampus (HippMapp3r)",
          T("segmentation", "Subcortical", "smri", "neuro", suite="ANTsPyNet"),
          description="Hippocampal segmentation from a T1."),
    _spec("antsxnet_mtl", "Segmentation", "Medial temporal lobe (DeepFLASH)",
          T("segmentation", "Subcortical", "smri", "neuro", suite="ANTsPyNet"),
          (ParamSpec(_L2, "T2 layer (optional)", "layer", ""),),
          description="Hippocampal subfields and MTL cortices from a T1 (and an optional T2)."),
    _spec("antsxnet_wmh_sysu", "Segmentation", "White-matter hyperintensities (SysU)",
          T("segmentation", "Lesions", "smri", "neuro", suite="ANTsPyNet"),
          (ParamSpec(_L2, "T1 layer (optional)", "layer", ""),),
          description="WMH probability from the active FLAIR (and an optional T1)."),
    _spec("antsxnet_wmh_hypermapp3r", "Segmentation", "White-matter hyperintensities (HyperMapp3r)",
          T("segmentation", "Lesions", "smri", "neuro", suite="ANTsPyNet"),
          (ParamSpec(_L2, "FLAIR layer", "layer", ""),),
          description="WMH from the active T1 and a FLAIR layer."),
    _spec("antsxnet_pvs", "Segmentation", "Perivascular spaces (SHIVA)",
          T("segmentation", "Lesions", "smri", "neuro", suite="ANTsPyNet"),
          (ParamSpec(_L2, "FLAIR layer (optional)", "layer", ""),),
          description="Enlarged perivascular space probability from a T1 (and an optional FLAIR)."),
    _spec("antsxnet_lung", "Segmentation", "Lung extraction (ANTsPyNet)",
          T("segmentation", "Whole-body", "ct", "whole-body", suite="ANTsPyNet"),
          (ParamSpec("lung_modality", "Modality", "choice", "ct", choices=("ct", "proton", "protonLobes", "maskLobes")),),
          description="Left / right lung masks from CT or proton MRI."),
    _spec("antsxnet_brain_age", "Measure", "Brain age (DeepBrainNet)",
          T("measure", "Morphometry", "smri", "neuro", suite="ANTsPyNet"), run_mode="notify",
          description="Predicted brain age in years from a T1."),
    _spec("antsxnet_claustrum", "Segmentation", "Claustrum (ANTsPyNet)",
          T("segmentation", "Subcortical", "smri", "neuro", suite="ANTsPyNet")),
    _spec("antsxnet_hypothalamus", "Segmentation", "Hypothalamus (ANTsPyNet)",
          T("segmentation", "Subcortical", "smri", "neuro", suite="ANTsPyNet")),
    _spec("antsxnet_harvard_oxford", "Segmentation", "Harvard-Oxford subcortical labels (ANTsPyNet)",
          T("segmentation", "Parcellation", "smri", "neuro", suite="ANTsPyNet")),
    _spec("antsxnet_cerebellum", "Segmentation", "Cerebellum parcellation (ANTsPyNet)",
          T("segmentation", "Subcortical", "smri", "neuro", suite="ANTsPyNet")),
    _spec("ants_denoise", "Restoration", "Non-local-means denoising (ANTs)",
          T("preprocess", "Denoising", suite="ANTs"),
          (ParamSpec("noise_model", "Noise model", "choice", "Rician", choices=("Rician", "Gaussian")),
           ParamSpec("shrink_factor", "Shrink factor", "int", 1, min=1, max=4)),
          description="Rician (MRI) or Gaussian non-local-means denoising."),
    _spec("ants_histogram_match", "Restoration", "Histogram matching (ANTs)",
          T("preprocess", "Intensity normalisation", suite="ANTs"),
          (ParamSpec(_L2, "Reference layer", "layer", ""),),
          description="Match the active image's intensity histogram to a reference layer's."),
    _spec("ants_atropos", "Segmentation", "Atropos tissue classes (ANTs)",
          T("segmentation", "Tissue", "smri", suite="ANTs"),
          (ParamSpec(_L2, "Mask layer", "layer", ""),
           ParamSpec("n_classes", "Classes", "int", 3, min=2, max=10)),
          description="K-means / MRF tissue classes inside a mask (3 on a T1: CSF, GM, WM)."),
)

ANTS_TOOL_IDS: frozenset[str] = frozenset(t.id for t in ANTS_TOOLS)


def run_ants_tool(tool_id: str, img: Any, viewer: Any, params: dict[str, Any]) -> Any:
    """Run one of :data:`ANTS_TOOLS` on *img* (the active layer as an nvitk image)."""
    from nvitk.gui.tools.runner import _resolve_layer, coerce_tool_output, gui_log, layer_to_image, notify

    def second() -> Any:
        name = str(params.get(_L2) or "").strip()
        if not name or name == "(none)":
            return None
        return layer_to_image(_resolve_layer(viewer, name))

    if tool_id.startswith("antsxnet_"):
        from nvitk.segmentation.antsxnet_tools import TOOLS

        fn = TOOLS[tool_id][0]
        if tool_id == "antsxnet_wmh_hypermapp3r":
            other = second()
            if other is None:
                raise ValueError("HyperMapp3r needs a FLAIR layer.")
            result = fn(img, other)
        elif tool_id == "antsxnet_lung":
            result = fn(img, modality=str(params.get("lung_modality") or "ct"))
        elif tool_id in ("antsxnet_mtl", "antsxnet_wmh_sysu", "antsxnet_pvs"):
            result = fn(img, second())
        else:
            result = fn(img)
        if result.scalars:
            text = ", ".join(f"{k} = {v:.2f}" for k, v in result.scalars.items())
            gui_log(f"{tool_id}: {text}")
            notify(text)
        if result.primary is None:
            return None
        return coerce_tool_output(result.primary)

    from nvitk.registration import ants_extra as ax

    if tool_id == "ants_denoise":
        return coerce_tool_output(ax.denoise(img, noise_model=str(params.get("noise_model") or "Rician"),
                                             shrink_factor=int(params.get("shrink_factor") or 1)))
    if tool_id == "ants_histogram_match":
        reference = second()
        if reference is None:
            raise ValueError("Choose a reference layer.")
        return coerce_tool_output(ax.histogram_match(img, reference))
    if tool_id == "ants_atropos":
        mask = second()
        if mask is None:
            raise ValueError("Choose a mask layer.")
        return coerce_tool_output(ax.atropos(img, mask, n_classes=int(params.get("n_classes") or 3))["segmentation"])
    raise KeyError(tool_id)


__all__ = ["ANTS_TOOLS", "ANTS_TOOL_IDS", "run_ants_tool"]
