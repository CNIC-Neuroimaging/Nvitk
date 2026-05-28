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


@dataclass(frozen=True)
class AntsRegistrationResult:
    warped_moving_path: Path
    fwd_transforms: tuple[Path, ...]
    inv_transforms: tuple[Path, ...]
    out_prefix: str


def _require_ants() -> Any:
    try:
        import ants  # type: ignore[import-not-found]
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
) -> AntsRegistrationResult:
    """Register MOVING to FIXED and write outputs under *out_dir*."""
    ants = _require_ants()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / "ants_")

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    tx = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=str(type_of_transform),
        outprefix=out_prefix,
        write_composite_transform=bool(write_composite_transform),
        verbose=bool(verbose),
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
    "ANTSPY_TYPE_OF_TRANSFORM",
    "AntsRegistrationResult",
    "ants_register",
    "ants_apply",
]

