"""ANTsPy tools beyond pairwise registration: denoising, histogram matching, Atropos,
DiReCT cortical thickness, joint label fusion, template building, and named SyN presets.

All take nvitk images (or arrays) and convert through :mod:`nvitk.io.ants_bridge`. Results are
arrays on the first input's grid, except :func:`build_template`, whose output lies on the
template grid it reports (:class:`TemplateResult`). Unlike the ANTsPyNet models these need no
downloads and run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import from_ants_image, require_ants, to_ants_image
from nvitk.segmentation.antsxnet_tools import ToolInfo

log = Logger()

#: Named registration presets -> ``ants.registration`` ``type_of_transform``.
SYN_PRESETS: dict[str, tuple[str, str]] = {
    "rigid": ("Rigid", "Rigid only (6 dof)."),
    "affine": ("Affine", "Rigid + affine (12 dof)."),
    "syn": ("SyN", "Affine + symmetric normalisation, mutual information (the ANTsPy default)."),
    "syn_cc": ("SyNCC", "Affine + SyN with cross-correlation: slower, better within a modality."),
    "syn_ra": ("SyNRA", "Rigid + affine + SyN."),
    "syn_only": ("SyNOnly", "SyN only, for inputs already affinely aligned."),
    "elastic_syn": ("ElasticSyN", "Affine + SyN with an elastic regulariser."),
    "syn_quick": ("antsRegistrationSyNQuick[s]", "antsRegistrationSyNQuick.sh: rigid + affine + SyN, fast."),
    "syn_full": ("antsRegistrationSyN[s]", "antsRegistrationSyN.sh: rigid + affine + SyN, thorough."),
    "bspline_syn_quick": ("antsRegistrationSyNQuick[b]", "Rigid + affine + B-spline SyN, fast."),
    "syn_aggro": ("SyNAggro", "Affine + aggressive SyN (more deformation)."),
}


def _as_mask(mask: Any | None) -> Any | None:
    return to_ants_image(mask) if mask is not None else None


def denoise(
    image: Any,
    *,
    mask: Any | None = None,
    noise_model: str = "Rician",
    shrink_factor: int = 1,
    patch_radius: int = 1,
    search_radius: int = 2,
) -> np.ndarray:
    """Non-local-means denoising (Manjón et al.); ``noise_model`` ``"Rician"`` (MRI) or ``"Gaussian"``."""
    ants = require_ants()
    out = ants.denoise_image(
        to_ants_image(image), mask=_as_mask(mask), shrink_factor=int(shrink_factor),
        p=int(patch_radius), r=int(search_radius), noise_model=str(noise_model),
    )
    return from_ants_image(out)


def histogram_match(
    source: Any,
    reference: Any,
    *,
    bins: int = 255,
    match_points: int = 64,
    threshold_at_mean: bool = False,
) -> np.ndarray:
    """*source*'s intensities remapped so its histogram matches *reference*'s (on *source*'s grid)."""
    ants = require_ants()
    out = ants.histogram_match_image(
        to_ants_image(source), to_ants_image(reference), number_of_histogram_bins=int(bins),
        number_of_match_points=int(match_points), use_threshold_at_mean_intensity=bool(threshold_at_mean),
    )
    return from_ants_image(out)


def atropos(
    image: Any,
    mask: Any,
    *,
    n_classes: int = 3,
    initialization: str = "Kmeans",
    mrf: str = "[0.2,1x1x1]",
    convergence: str = "[5,0]",
    prior_weight: float = 0.25,
) -> dict[str, np.ndarray]:
    """Atropos n-class tissue segmentation inside *mask*: ``segmentation`` plus ``probability_k``.

    Classes are ordered by mean intensity (for a T1 with 3 classes: CSF, GM, WM).
    """
    ants = require_ants()
    img = to_ants_image(image)
    msk = to_ants_image(mask)
    if img.dimension == 2 and mrf == "[0.2,1x1x1]":
        mrf = "[0.2,1x1]"
    result = ants.atropos(a=img, x=msk, i=f"{initialization}[{int(n_classes)}]", m=mrf, c=convergence, priorweight=float(prior_weight))
    out = {"segmentation": from_ants_image(result["segmentation"])}
    for k, prob in enumerate(result.get("probabilityimages") or []):
        out[f"probability_{k + 1}"] = from_ants_image(prob)
    return out


def cortical_thickness_direct(
    segmentation: Any,
    gm_probability: Any,
    wm_probability: Any,
    *,
    iterations: int = 45,
    gradient_step: float = 0.025,
    smoothing: float = 1.5,
    gm_label: int = 2,
    wm_label: int = 3,
) -> np.ndarray:
    """DiReCT (KellyKapowski) cortical thickness in mm from a tissue segmentation and GM/WM probabilities."""
    ants = require_ants()
    seg = to_ants_image(segmentation)
    out = ants.kelly_kapowski(
        s=seg, g=to_ants_image(gm_probability), w=to_ants_image(wm_probability), its=int(iterations),
        r=float(gradient_step), m=float(smoothing), gm_label=int(gm_label), wm_label=int(wm_label),
    )
    return from_ants_image(out)


def joint_label_fusion(
    target: Any,
    target_mask: Any,
    atlas_images: Sequence[Any],
    atlas_labels: Sequence[Any],
    *,
    beta: float = 4.0,
    patch_radius: int = 2,
    search_radius: int = 3,
    rho: float = 0.01,
) -> dict[str, np.ndarray]:
    """Joint label fusion (Wang et al.) of atlases already registered to *target*.

    Returns ``segmentation`` (fused labels on the target grid) and ``intensity`` (fused atlas
    intensities). Atlas images and label maps must be on the target grid.
    """
    if len(atlas_images) != len(atlas_labels) or not atlas_images:
        raise ValueError("joint_label_fusion needs one label map per atlas image, and at least one atlas")
    ants = require_ants()
    result = ants.joint_label_fusion(
        target_image=to_ants_image(target),
        target_image_mask=to_ants_image(target_mask),
        atlas_list=[to_ants_image(a) for a in atlas_images],
        label_list=[to_ants_image(l) for l in atlas_labels],
        beta=float(beta), rad=int(patch_radius), r_search=int(search_radius), rho=float(rho),
    )
    out = {"segmentation": from_ants_image(result["segmentation"])}
    if result.get("intensity") is not None:
        out["intensity"] = from_ants_image(result["intensity"])
    return out


@dataclass
class TemplateResult:
    """A built template and its grid (ANTs LPS physical space, as ANTs reports it)."""

    data: np.ndarray
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: np.ndarray

    def affine_ras(self) -> np.ndarray:
        """4x4 voxel -> RAS affine (ANTs' LPS flipped on x and y)."""
        n = len(self.spacing)
        aff = np.eye(4)
        aff[:n, :n] = np.asarray(self.direction)[:n, :n] @ np.diag(self.spacing)
        aff[:n, 3] = self.origin
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        return flip @ aff


def build_template(
    images: Sequence[Any],
    *,
    initial_template: Any | None = None,
    iterations: int = 3,
    gradient_step: float = 0.2,
    blending_weight: float = 0.75,
    type_of_transform: str = "SyN",
) -> TemplateResult:
    """Unbiased population template from *images* (ANTs ``build_template``)."""
    if len(images) < 2:
        raise ValueError("build_template needs at least two images")
    ants = require_ants()
    template = ants.build_template(
        initial_template=to_ants_image(initial_template) if initial_template is not None else None,
        image_list=[to_ants_image(i) for i in images],
        iterations=int(iterations), gradient_step=float(gradient_step),
        blending_weight=float(blending_weight), type_of_transform=str(type_of_transform),
    )
    return TemplateResult(
        data=from_ants_image(template),
        spacing=tuple(float(s) for s in template.spacing),
        origin=tuple(float(o) for o in template.origin),
        direction=np.asarray(template.direction),
    )


def register_preset(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    preset: str = "syn_quick",
    **options: Any,
):
    """:func:`~nvitk.registration.ants.ants_register` with a named :data:`SYN_PRESETS` entry."""
    from nvitk.registration.ants import ants_register

    if preset not in SYN_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; one of {', '.join(SYN_PRESETS)}")
    return ants_register(
        fixed_path=fixed_path, moving_path=moving_path, out_dir=out_dir,
        type_of_transform=SYN_PRESETS[preset][0], **options,
    )


TOOLS: dict[str, ToolInfo] = {
    "ants_denoise": ToolInfo("ants_denoise", "Non-local-means denoising",
                             "Rician (MRI) or Gaussian non-local-means denoising.", ("image", "mask?"),
                             ("smri", "mra", "ct", "pet"), domain="general", suite="ANTs", slow=False, outputs=("image",)),
    "ants_histogram_match": ToolInfo("ants_histogram_match", "Histogram matching",
                                     "Match an image's intensity histogram to a reference.", ("source", "reference"),
                                     ("smri", "mra", "ct"), domain="general", suite="ANTs", slow=False, outputs=("image",)),
    "ants_atropos": ToolInfo("ants_atropos", "Atropos tissue classes",
                             "K-means / MRF n-class segmentation inside a mask.", ("image", "mask"),
                             ("smri",), domain="general", suite="ANTs", slow=False, outputs=("segmentation", "probability")),
    "ants_direct_thickness": ToolInfo("ants_direct_thickness", "DiReCT cortical thickness",
                                      "KellyKapowski thickness from tissue segmentation + GM/WM probabilities.",
                                      ("segmentation", "gm_probability", "wm_probability"), ("smri",), suite="ANTs",
                                      outputs=("thickness",)),
    "ants_jlf": ToolInfo("ants_jlf", "Joint label fusion",
                         "Fuse registered atlases' labels onto a target (multi-atlas segmentation).",
                         ("target", "mask", "atlas images", "atlas labels"), ("smri", "ct"), domain="general", suite="ANTs"),
    "ants_build_template": ToolInfo("ants_build_template", "Build population template",
                                    "Unbiased group-wise template from several images.", ("images",),
                                    ("smri", "mra", "ct"), domain="general", suite="ANTs", outputs=("template",)),
    "ants_register_preset": ToolInfo("ants_register_preset", "Registration (named presets)",
                                     "Rigid / affine / SyN variants by name (antsRegistrationSyN[Quick] included).",
                                     ("fixed", "moving"), ("smri", "mra", "ct", "pet"), domain="general", suite="ANTs",
                                     outputs=("warped", "transforms")),
}


__all__ = [
    "SYN_PRESETS",
    "TOOLS",
    "TemplateResult",
    "atropos",
    "build_template",
    "cortical_thickness_direct",
    "denoise",
    "histogram_match",
    "joint_label_fusion",
    "register_preset",
]
