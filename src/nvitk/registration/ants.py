"""ANTsPy registration backend.

This module wraps `ants.registration` / `ants.apply_transforms` behind a small,
file-path oriented API for use in pipelines and CLIs.

Supported `type_of_transform` values (from ANTsPy docs):

- Translation
- Rigid
- Similarity
- QuickRigid
- DenseRigid
- BOLDRigid
- Affine
- AffineFast
- BOLDAffine
- TRSAA
- Elastic
- ElasticSyN
- SyN
- SyNRA
- SyNOnly
- SyNCC
- SyNabp
- SyNBold
- SyNBoldAff
- SyNAggro
- SyNLessAggro
- TV[n]
- TVMSQ
- TVMSQC
- antsRegistrationSyN[x]
- antsRegistrationSyNQuick[x]
- antsRegistrationSyNRepro[x]
- antsRegistrationSyNQuickRepro[x]

See the upstream documentation for full details and parameter meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvitk.core.exceptions import BackendUnavailableError


ANTSPY_TYPE_OF_TRANSFORM: tuple[str, ...] = (
    "Translation",
    "Rigid",
    "Similarity",
    "QuickRigid",
    "DenseRigid",
    "BOLDRigid",
    "Affine",
    "AffineFast",
    "BOLDAffine",
    "TRSAA",
    "Elastic",
    "ElasticSyN",
    "SyN",
    "SyNRA",
    "SyNOnly",
    "SyNCC",
    "SyNabp",
    "SyNBold",
    "SyNBoldAff",
    "SyNAggro",
    "SyNLessAggro",
    "TV[n]",
    "TVMSQ",
    "TVMSQC",
    "antsRegistrationSyN[x]",
    "antsRegistrationSyNQuick[x]",
    "antsRegistrationSyNRepro[x]",
    "antsRegistrationSyNQuickRepro[x]",
)


#: ``type_of_transform`` values the GUI offers: :data:`ANTSPY_TYPE_OF_TRANSFORM` with the
#: templated entries (``TV[n]``, ``antsRegistrationSyN[x]``…) spelled out, since ANTsPy
#: needs the concrete value — ``[r]`` rigid, ``[a]`` affine, ``[s]`` rigid+affine+SyN,
#: ``[b]`` rigid+affine+B-spline SyN.
ANTSPY_TRANSFORM_CHOICES: tuple[str, ...] = (
    *(t for t in ANTSPY_TYPE_OF_TRANSFORM if "[" not in t),
    "TV[2]",
    "TV[4]",
    "antsRegistrationSyN[r]",
    "antsRegistrationSyN[a]",
    "antsRegistrationSyN[s]",
    "antsRegistrationSyN[b]",
    "antsRegistrationSyNQuick[r]",
    "antsRegistrationSyNQuick[a]",
    "antsRegistrationSyNQuick[s]",
    "antsRegistrationSyNQuick[b]",
    "antsRegistrationSyNRepro[s]",
    "antsRegistrationSyNQuickRepro[s]",
)

#: Similarity metrics of the linear (``aff_metric``) and deformable (``syn_metric``) stages.
ANTSPY_AFF_METRICS: tuple[str, ...] = ("mattes", "GC", "meansquares")
ANTSPY_SYN_METRICS: tuple[str, ...] = ("mattes", "CC", "meansquares", "demons")

#: Interpolators ``ants.apply_transforms`` accepts (``genericLabel`` / ``nearestNeighbor``
#: for label maps).
ANTSPY_INTERPOLATORS: tuple[str, ...] = (
    "linear",
    "nearestNeighbor",
    "genericLabel",
    "multiLabel",
    "gaussian",
    "bSpline",
    "cosineWindowedSinc",
    "welchWindowedSinc",
    "hammingWindowedSinc",
    "lanczosWindowedSinc",
)

#: Optional ``ants.registration`` keywords :func:`ants_register` passes through.
_REGISTRATION_OPTIONS: frozenset[str] = frozenset({
    "grad_step",
    "flow_sigma",
    "total_sigma",
    "aff_metric",
    "aff_sampling",
    "aff_random_sampling_rate",
    "syn_metric",
    "syn_sampling",
    "reg_iterations",
    "aff_iterations",
    "aff_shrink_factors",
    "aff_smoothing_sigmas",
    "random_seed",
    "smoothing_in_mm",
    "restrict_transformation",
    "mask_all_stages",
})


def parse_int_tuple(text: Any) -> tuple[int, ...] | None:
    """``"40,20,0"`` / ``"40x20x0"`` / ``[40, 20, 0]`` → ``(40, 20, 0)``; empty → ``None``."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return tuple(int(v) for v in text) or None
    parts = [p for p in str(text).replace("x", ",").split(",") if p.strip()]
    return tuple(int(float(p)) for p in parts) or None


@dataclass(frozen=True)
class AntsRegistrationResult:
    """Paths produced by an ANTs registration: warped moving image and forward/inverse transforms."""

    warped_moving_path: Path
    fwd_transforms: tuple[Path, ...]
    inv_transforms: tuple[Path, ...]
    out_prefix: str


def _require_ants() -> Any:
    """Import and return the ``ants`` module, or raise a clear install hint."""
    try:
        import ants
    except Exception as exc:
        raise BackendUnavailableError(
            "ANTsPy is not installed. Install with: pip install antspyx"
        ) from exc
    return ants


def ants_register(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    type_of_transform: str = "SyN",
    write_composite_transform: bool = False,
    verbose: bool = False,
    initial_transform: str | Path | None = None,
    fixed_mask_path: Path | None = None,
    moving_mask_path: Path | None = None,
    **options: Any,
) -> AntsRegistrationResult:
    """Register MOVING to FIXED and write outputs under *out_dir*.

    *options* are passed to ``ants.registration`` — the stage metrics
    (``aff_metric``, ``syn_metric``), sampling, ``grad_step`` / ``flow_sigma`` /
    ``total_sigma``, iteration schedules (``reg_iterations``, ``aff_iterations``…;
    tuples or ``"40,20,0"`` strings) and ``random_seed``. Unknown keys raise.
    *initial_transform* is a transform file (or ``"Identity"``) to start from; the
    masks restrict the metric to a region of the fixed / moving image.
    """
    ants = _require_ants()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "ants_")

    unknown = sorted(set(options) - _REGISTRATION_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown ants.registration option(s): {unknown}")
    kwargs: dict[str, Any] = {}
    for key, value in options.items():
        if value is None or value == "":
            continue
        if key in ("reg_iterations", "aff_iterations", "aff_shrink_factors", "aff_smoothing_sigmas"):
            value = parse_int_tuple(value)
            if value is None:
                continue
        kwargs[key] = value

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    if fixed_mask_path is not None:
        kwargs["mask"] = ants.image_read(str(fixed_mask_path))
    if moving_mask_path is not None:
        kwargs["moving_mask"] = ants.image_read(str(moving_mask_path))
    if initial_transform:
        init = str(initial_transform).strip()
        kwargs["initial_transform"] = init if init.lower() == "identity" else [init]
    tx = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=str(type_of_transform),
        outprefix=out_prefix,
        write_composite_transform=bool(write_composite_transform),
        verbose=bool(verbose),
        **kwargs,
    )
    warped = tx.get("warpedmovout")
    warped_path = out_dir / "moving_warped.nii.gz"
    if warped is not None:
        ants.image_write(warped, str(warped_path))
    else:
        raise RuntimeError("ANTsPy registration did not return warpedmovout.")

    fwd = tuple(Path(p) for p in tx.get("fwdtransforms", []) if p)
    inv = tuple(Path(p) for p in tx.get("invtransforms", []) if p)
    return AntsRegistrationResult(
        warped_moving_path=warped_path,
        fwd_transforms=fwd,
        inv_transforms=inv,
        out_prefix=out_prefix,
    )


def ants_apply(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_path: Path,
    transforms: list[Path],
    interpolator: str = "linear",
    whichtoinvert: list[bool] | None = None,
    verbose: bool = False,
) -> Path:
    """Apply transforms to map MOVING into FIXED space."""
    ants = _require_ants()
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    out = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(p) for p in transforms],
        interpolator=str(interpolator),
        whichtoinvert=whichtoinvert,
        verbose=bool(verbose),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(out, str(out_path))
    return out_path


__all__ = [
    "ANTSPY_AFF_METRICS",
    "ANTSPY_INTERPOLATORS",
    "ANTSPY_SYN_METRICS",
    "ANTSPY_TRANSFORM_CHOICES",
    "ANTSPY_TYPE_OF_TRANSFORM",
    "parse_int_tuple",
    "AntsRegistrationResult",
    "ants_register",
    "ants_apply",
]

