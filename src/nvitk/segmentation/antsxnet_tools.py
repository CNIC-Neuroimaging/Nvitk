"""ANTsPyNet deep-learning tools beyond brain extraction / DKT: tissue, cortical thickness,
hippocampus and medial temporal lobe, white-matter hyperintensities, perivascular spaces, lung,
brain age...

Every wrapper takes nvitk images (or arrays carrying the same geometry), converts them through
:mod:`nvitk.io.ants_bridge` (which feeds ANTs LPS from nvitk's RAS correctly; see that module),
calls the ANTsPyNet function, and returns an :class:`AntsXNetResult` whose arrays lie on the
*first input's* grid. Models download their weights into ANTsPyNet's cache
(``~/.keras/ANTsXNet``) on first use, which needs network access once; the log says so.

ANTsPyNet imports TensorFlow, so it is imported only when a wrapper runs. :data:`TOOLS` describes
every wrapper (inputs, modality, domain, speed) for menus and the GUI's tool browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import from_ants_image, require_antspynet, to_ants_image

log = Logger()


@dataclass(frozen=True)
class ToolInfo:
    """What a tool needs and does, for listings and the GUI tool browser."""

    id: str
    label: str
    description: str
    inputs: tuple[str, ...]
    modalities: tuple[str, ...]
    domain: str = "neuro"
    species: str = "human"
    suite: str = "ANTsPyNet"
    slow: bool = True
    outputs: tuple[str, ...] = ("segmentation",)


@dataclass
class AntsXNetResult:
    """Arrays (on the first input's grid) and scalars a tool produced, plus label names."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    labels: dict[int, str] | None = None

    @property
    def primary(self) -> np.ndarray | None:
        """The main output: the segmentation, else the first array."""
        if "segmentation" in self.arrays:
            return self.arrays["segmentation"]
        # A nested result (cerebellum: ``parcellation_segmentation``): its labels, not a probability.
        nested = next((k for k in self.arrays if k.endswith("_segmentation")), None)
        if nested is not None:
            return self.arrays[nested]
        return next(iter(self.arrays.values()), None)


#: Deep Atropos's six tissue classes.
DEEP_ATROPOS_LABELS: dict[int, str] = {
    0: "Background",
    1: "CSF",
    2: "Gray matter",
    3: "White matter",
    4: "Deep gray matter",
    5: "Brain stem",
    6: "Cerebellum",
}


def _is_ants_image(value: Any) -> bool:
    return hasattr(value, "numpy") and hasattr(value, "spacing")


def _collect(result: Any, name: str = "segmentation") -> AntsXNetResult:
    """Normalise whatever an ANTsPyNet function returned (image, list, dict, number)."""
    out = AntsXNetResult()
    if _is_ants_image(result):
        out.arrays[name] = from_ants_image(result)
    elif isinstance(result, (int, float, np.floating)):
        out.scalars[name] = float(result)
    elif isinstance(result, (list, tuple)):
        for i, item in enumerate(result):
            if _is_ants_image(item):
                out.arrays[name if i == 0 else f"{name}_{i}"] = from_ants_image(item)
    elif isinstance(result, dict):
        for key, value in result.items():
            short = str(key).replace("_images", "").replace("_image", "")
            short = "segmentation" if short in ("segmentation", "labels", "label") else short
            if _is_ants_image(value):
                out.arrays[short] = from_ants_image(value)
            elif isinstance(value, (list, tuple)) and value and all(_is_ants_image(v) for v in value):
                for i, item in enumerate(value):
                    out.arrays[f"{short}_{i}"] = from_ants_image(item)
            elif isinstance(value, (int, float, np.floating, np.integer)):
                out.scalars[short] = float(value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                out.scalars[short] = float(value.ravel()[0])
    return out


def _run(function: str, *images: Any, output: str = "segmentation", **kwargs: Any) -> AntsXNetResult:
    antspynet = require_antspynet()
    fn: Callable[..., Any] = getattr(antspynet, function)
    converted = [to_ants_image(image) if image is not None else None for image in images]
    log.info(f"ANTsPyNet {function}: inputs={[tuple(c.shape) if c is not None else None for c in converted]}")
    log.info("  (model weights download to ~/.keras/ANTsXNet on first use)")
    return _collect(fn(*converted, **kwargs), output)


# Wrappers
def deep_atropos(t1: Any, *, do_preprocessing: bool = True, verbose: bool = False) -> AntsXNetResult:
    """Six-tissue segmentation of a T1 (CSF, GM, WM, deep GM, brain stem, cerebellum)."""
    result = _run("deep_atropos", t1, do_preprocessing=do_preprocessing, verbose=verbose)
    result.labels = dict(DEEP_ATROPOS_LABELS)
    return result


def cortical_thickness(t1: Any, *, verbose: bool = False) -> AntsXNetResult:
    """KellyKapowski cortical thickness from a T1 via Deep Atropos (thickness image + tissues)."""
    result = _run("cortical_thickness", t1, output="thickness", verbose=verbose)
    if "segmentation" in result.arrays:
        result.labels = dict(DEEP_ATROPOS_LABELS)
    return result


def hippocampus(t1: Any, *, do_preprocessing: bool = True, verbose: bool = False) -> AntsXNetResult:
    """Hippocampal segmentation (HippMapp3r) of a T1."""
    result = _run("hippmapp3r_segmentation", t1, do_preprocessing=do_preprocessing, verbose=verbose)
    # Checked on a RAS T1 (2026-11-25): label 1 lies at x < 0 (left), label 2 at x > 0 (right).
    result.labels = {0: "Background", 1: "Left hippocampus", 2: "Right hippocampus"}
    return result


def medial_temporal_lobe(
    t1: Any, t2: Any | None = None, *, parcellation: str = "yassa", do_preprocessing: bool = True, verbose: bool = False
) -> AntsXNetResult:
    """Hippocampal / medial temporal lobe parcellation (DeepFLASH), T1 with an optional T2."""
    return _run("deep_flash", t1, t2, which_parcellation=parcellation, do_preprocessing=do_preprocessing, verbose=verbose)


def wmh_sysu(flair: Any, t1: Any | None = None, *, use_ensemble: bool = True, verbose: bool = False) -> AntsXNetResult:
    """White-matter hyperintensity probability (SysU MICCAI 2017 winner) from FLAIR (+ T1)."""
    return _run("sysu_media_wmh_segmentation", flair, t1, output="probability", use_ensemble=use_ensemble, verbose=verbose)


def wmh_hypermapp3r(t1: Any, flair: Any, *, monte_carlo: int = 30, do_preprocessing: bool = True, verbose: bool = False) -> AntsXNetResult:
    """White-matter hyperintensity probability (HyperMapp3r, Monte-Carlo averaged) from T1 + FLAIR."""
    return _run(
        "hypermapp3r_segmentation", t1, flair, output="probability",
        number_of_monte_carlo_iterations=int(monte_carlo), do_preprocessing=do_preprocessing, verbose=verbose,
    )


def perivascular_spaces(t1: Any, flair: Any | None = None, *, model: str = "all", verbose: bool = False) -> AntsXNetResult:
    """Perivascular space probability (SHIVA) from T1 (+ FLAIR)."""
    return _run("shiva_pvs_segmentation", t1, flair, output="probability", which_model=model, verbose=verbose)


def lung_extraction(image: Any, *, modality: str = "ct", verbose: bool = False) -> AntsXNetResult:
    """Lung masks from CT (``ct``), proton MRI (``proton``) or ventilation MRI."""
    return _run("lung_extraction", image, modality=modality, verbose=verbose)


def brain_age(t1: Any, *, do_preprocessing: bool = True, verbose: bool = False) -> AntsXNetResult:
    """Predicted brain age (years) from a T1 (DeepBrainNet).

    DeepBrainNet ships as a legacy Keras ``.h5`` model that Keras 3 cannot rebuild (ANTsPyNet
    0.3.2 + Keras 3: ``Layer "dense_3" expects 1 input(s), but it received 2``). It loads with
    the legacy Keras 2 API: install ``tf_keras`` and set ``TF_USE_LEGACY_KERAS=1`` before
    TensorFlow is first imported.
    """
    try:
        return _run("brain_age", t1, output="predicted_age", do_preprocessing=do_preprocessing, verbose=verbose)
    except ValueError as exc:
        if "expects 1 input(s), but it received 2" not in str(exc):
            raise
        raise RuntimeError(
            "Brain age (DeepBrainNet) cannot load under Keras 3. Install tf_keras in this environment "
            "and start nvitk with TF_USE_LEGACY_KERAS=1 to use it."
        ) from exc


def claustrum(t1: Any, *, use_ensemble: bool = True, verbose: bool = False) -> AntsXNetResult:
    """Claustrum probability from a T1."""
    return _run("claustrum_segmentation", t1, output="probability", use_ensemble=use_ensemble, verbose=verbose)


def hypothalamus(t1: Any, *, verbose: bool = False) -> AntsXNetResult:
    """Hypothalamic sub-region segmentation of a T1."""
    return _run("hypothalamus_segmentation", t1, verbose=verbose)


def harvard_oxford_labels(t1: Any, *, do_preprocessing: bool = True, verbose: bool = False) -> AntsXNetResult:
    """Harvard-Oxford subcortical labelling of a T1 (deep-learning, subject space)."""
    return _run("harvard_oxford_atlas_labeling", t1, do_preprocessing=do_preprocessing, verbose=verbose)


def cerebellum(t1: Any, *, thickness: bool = False, verbose: bool = False) -> AntsXNetResult:
    """Cerebellum parcellation (and optionally its cortical thickness) of a T1."""
    return _run("cerebellum_morphology", t1, compute_thickness_image=bool(thickness), verbose=verbose)


#: Wrapper id -> (function, info).
TOOLS: dict[str, tuple[Callable[..., AntsXNetResult], ToolInfo]] = {
    "antsxnet_deep_atropos": (deep_atropos, ToolInfo(
        "antsxnet_deep_atropos", "Deep Atropos tissue segmentation",
        "Six tissues from a T1: CSF, gray matter, white matter, deep gray matter, brain stem, cerebellum.",
        ("t1",), ("smri",), outputs=("segmentation", "probability"))),
    "antsxnet_cortical_thickness": (cortical_thickness, ToolInfo(
        "antsxnet_cortical_thickness", "Cortical thickness (DiReCT / KellyKapowski)",
        "Cortical thickness map from a T1, via Deep Atropos tissues.", ("t1",), ("smri",), outputs=("thickness", "segmentation"))),
    "antsxnet_hippocampus": (hippocampus, ToolInfo(
        "antsxnet_hippocampus", "Hippocampus (HippMapp3r)", "Hippocampal segmentation from a T1.", ("t1",), ("smri",))),
    "antsxnet_mtl": (medial_temporal_lobe, ToolInfo(
        "antsxnet_mtl", "Medial temporal lobe (DeepFLASH)",
        "Hippocampal subfields and MTL cortices from a T1 (and optional T2).", ("t1", "t2?"), ("smri",))),
    "antsxnet_wmh_sysu": (wmh_sysu, ToolInfo(
        "antsxnet_wmh_sysu", "WMH probability (SysU)",
        "White-matter hyperintensity probability from FLAIR (+ optional T1).", ("flair", "t1?"), ("smri",), outputs=("probability",))),
    "antsxnet_wmh_hypermapp3r": (wmh_hypermapp3r, ToolInfo(
        "antsxnet_wmh_hypermapp3r", "WMH segmentation (HyperMapp3r)",
        "White-matter hyperintensities from T1 + FLAIR.", ("t1", "flair"), ("smri",))),
    "antsxnet_pvs": (perivascular_spaces, ToolInfo(
        "antsxnet_pvs", "Perivascular spaces (SHIVA)",
        "Enlarged perivascular space probability from T1 (+ optional FLAIR).", ("t1", "flair?"), ("smri",), outputs=("probability",))),
    "antsxnet_lung": (lung_extraction, ToolInfo(
        "antsxnet_lung", "Lung extraction",
        "Left/right lung masks from CT or proton MRI.", ("image",), ("ct", "mri"), domain="whole-body")),
    "antsxnet_brain_age": (brain_age, ToolInfo(
        "antsxnet_brain_age", "Brain age (DeepBrainNet)",
        "Predicted brain age in years from a T1.", ("t1",), ("smri",), outputs=("predicted_age",))),
    "antsxnet_claustrum": (claustrum, ToolInfo(
        "antsxnet_claustrum", "Claustrum", "Claustrum probability from a T1.", ("t1",), ("smri",), outputs=("probability",))),
    "antsxnet_hypothalamus": (hypothalamus, ToolInfo(
        "antsxnet_hypothalamus", "Hypothalamus", "Hypothalamic sub-regions from a T1.", ("t1",), ("smri",))),
    "antsxnet_harvard_oxford": (harvard_oxford_labels, ToolInfo(
        "antsxnet_harvard_oxford", "Harvard-Oxford subcortical labels",
        "Deep-learning Harvard-Oxford subcortical labelling in subject space.", ("t1",), ("smri",))),
    "antsxnet_cerebellum": (cerebellum, ToolInfo(
        "antsxnet_cerebellum", "Cerebellum morphology", "Cerebellar parcellation (and thickness) from a T1.", ("t1",), ("smri",))),
}

#: ANTsPyNet function each wrapper calls (for checking an installation offers it).
ANTSPYNET_FUNCTIONS: dict[str, str] = {
    "antsxnet_deep_atropos": "deep_atropos",
    "antsxnet_cortical_thickness": "cortical_thickness",
    "antsxnet_hippocampus": "hippmapp3r_segmentation",
    "antsxnet_mtl": "deep_flash",
    "antsxnet_wmh_sysu": "sysu_media_wmh_segmentation",
    "antsxnet_wmh_hypermapp3r": "hypermapp3r_segmentation",
    "antsxnet_pvs": "shiva_pvs_segmentation",
    "antsxnet_lung": "lung_extraction",
    "antsxnet_brain_age": "brain_age",
    "antsxnet_claustrum": "claustrum_segmentation",
    "antsxnet_hypothalamus": "hypothalamus_segmentation",
    "antsxnet_harvard_oxford": "harvard_oxford_atlas_labeling",
    "antsxnet_cerebellum": "cerebellum_morphology",
}


__all__ = [
    "ANTSPYNET_FUNCTIONS",
    "AntsXNetResult",
    "DEEP_ATROPOS_LABELS",
    "TOOLS",
    "ToolInfo",
    "brain_age",
    "cerebellum",
    "claustrum",
    "cortical_thickness",
    "deep_atropos",
    "harvard_oxford_labels",
    "hippocampus",
    "hypothalamus",
    "lung_extraction",
    "medial_temporal_lobe",
    "perivascular_spaces",
    "wmh_hypermapp3r",
    "wmh_sysu",
]
