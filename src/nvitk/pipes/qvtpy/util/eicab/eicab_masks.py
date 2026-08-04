"""Resolve eICAB Circle-of-Willis (CW) or whole-brain (WB) multilabel masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nvitk.core.logger import Logger

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

EicabMaskKind = Literal["cw", "wb"]

CENTERLINES_MASK_PP_NIFTI = "centerlines_mask_pp.nii.gz"

# ---- Filename patterns -------------------------------------------------------

_CW_PATTERNS = ("*_eICAB_CW.nii.gz", "*_eICAB_CW.nii")
_WB_PATTERNS = ("*_eICAB_WB.nii.gz", "*_eICAB_WB.nii")
_CW_PP_PATTERNS = ("*_eICAB_CW_pp.nii.gz", "*_eICAB_CW_pp.nii")
_WB_PP_PATTERNS = ("*_eICAB_WB_pp.nii.gz", "*_eICAB_WB_pp.nii")


@dataclass(frozen=True)
class EicabMaskResolution:
    """Result of :func:`resolve_eicab_mask`."""

    path: Path
    requested: EicabMaskKind
    used: EicabMaskKind
    fallback: bool
    fallback_reason: str | None
    postprocessed: bool = False
    original_path: Path | None = None


# ---------------------------------------------------------------------------
# Resolve CW / WB multilabel path
# ---------------------------------------------------------------------------


def _stem_without_suffix(p: Path) -> str:
    """*p*'s filename without a trailing ``.nii``/``.nii.gz`` extension."""
    if p.name.lower().endswith(".nii.gz"):
        return p.name[: -len(".nii.gz")]
    if p.name.lower().endswith(".nii"):
        return p.name[: -len(".nii")]
    return p.stem


def eicab_pp_path(original: Path) -> Path:
    """Post-processed mask path: ``foo_eICAB_CW.nii.gz`` → ``foo_eICAB_CW_pp.nii.gz``."""
    p = Path(original)
    stem = _stem_without_suffix(p)
    if stem.endswith("_pp"):
        return p
    if p.name.lower().endswith(".nii.gz"):
        return p.with_name(f"{stem}_pp.nii.gz")
    if p.suffix.lower() == ".nii":
        return p.with_name(f"{stem}_pp.nii")
    return p.parent / f"{stem}_pp{p.suffix}"


def find_tof_resampled_volume(eicab_output_dir: Path) -> Path | None:
    """Return path to eICAB ``TOF_resampled`` NIfTI under *eicab_output_dir*, or None."""
    if not eicab_output_dir.is_dir():
        return None
    for name in ("TOF_resampled.nii.gz", "TOF_resampled.nii"):
        p = eicab_output_dir / name
        if p.is_file():
            return p
    hits: list[Path] = []
    for p in sorted(eicab_output_dir.rglob("*")):
        if not p.is_file():
            continue
        if not (p.suffix == ".nii" or p.name.endswith(".nii.gz")):
            continue
        stem = _stem_without_suffix(p).lower()
        if not stem.endswith("_resampled"):
            continue
        if "tof" in stem:
            hits.append(p)
    return hits[0] if hits else None


def _glob_first(eicab_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    """First file under *eicab_dir* matching any of *patterns* (in order), or ``None``."""
    for pat in patterns:
        hits = sorted(eicab_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _resolve_base_mask(
    eicab_dir: Path,
    preference: EicabMaskKind,
) -> EicabMaskResolution:
    """Resolve raw (non-pp) CW/WB mask with warn-and-fallback."""
    pref = preference
    cw = _glob_first(eicab_dir, _CW_PATTERNS)
    wb = _glob_first(eicab_dir, _WB_PATTERNS)

    if pref == "cw":
        if cw is not None:
            return EicabMaskResolution(
                path=cw,
                requested="cw",
                used="cw",
                fallback=False,
                fallback_reason=None,
            )
        if wb is not None:
            msg = (
                f"Requested eICAB CW mask but none found under {eicab_dir}; "
                f"continuing with WB mask {wb.name}"
            )
            log.warning(msg)
            return EicabMaskResolution(
                path=wb,
                requested="cw",
                used="wb",
                fallback=True,
                fallback_reason=msg,
            )
    else:
        if wb is not None:
            return EicabMaskResolution(
                path=wb,
                requested="wb",
                used="wb",
                fallback=False,
                fallback_reason=None,
            )
        if cw is not None:
            msg = (
                f"Requested eICAB WB mask but none found under {eicab_dir}; "
                f"continuing with CW mask {cw.name}"
            )
            log.warning(msg)
            return EicabMaskResolution(
                path=cw,
                requested="wb",
                used="cw",
                fallback=True,
                fallback_reason=msg,
            )

    raise FileNotFoundError(f"No eICAB CW/WB NIfTI under {eicab_dir}")


def resolve_eicab_mask(
    eicab_dir: Path,
    preference: EicabMaskKind = "cw",
    *,
    prefer_postprocessed: bool = True,
) -> EicabMaskResolution:
    """Return the eICAB label NIfTI path for *preference*, with warn-and-fallback.

    When *prefer_postprocessed* is True and a ``*_pp`` sibling of the base mask exists,
    that post-processed file is returned (``postprocessed=True``).
    """
    pref = preference.strip().lower()
    if pref not in ("cw", "wb"):
        raise ValueError(f"eicab_mask preference must be 'cw' or 'wb', got {preference!r}")

    if prefer_postprocessed:
        pp_cw = _glob_first(eicab_dir, _CW_PP_PATTERNS)
        pp_wb = _glob_first(eicab_dir, _WB_PP_PATTERNS)
        if pref == "cw" and pp_cw is not None:
            return EicabMaskResolution(
                path=pp_cw,
                requested="cw",
                used="cw",
                fallback=False,
                fallback_reason=None,
                postprocessed=True,
                original_path=_glob_first(eicab_dir, _CW_PATTERNS),
            )
        if pref == "wb" and pp_wb is not None:
            return EicabMaskResolution(
                path=pp_wb,
                requested="wb",
                used="wb",
                fallback=False,
                fallback_reason=None,
                postprocessed=True,
                original_path=_glob_first(eicab_dir, _WB_PATTERNS),
            )
        if pref == "cw" and pp_wb is not None and _glob_first(eicab_dir, _CW_PATTERNS) is None:
            msg = (
                f"Requested eICAB CW pp mask but none found; "
                f"continuing with WB pp mask {pp_wb.name}"
            )
            log.warning(msg)
            return EicabMaskResolution(
                path=pp_wb,
                requested="cw",
                used="wb",
                fallback=True,
                fallback_reason=msg,
                postprocessed=True,
                original_path=_glob_first(eicab_dir, _WB_PATTERNS),
            )
        if pref == "wb" and pp_cw is not None and _glob_first(eicab_dir, _WB_PATTERNS) is None:
            msg = (
                f"Requested eICAB WB pp mask but none found; "
                f"continuing with CW pp mask {pp_cw.name}"
            )
            log.warning(msg)
            return EicabMaskResolution(
                path=pp_cw,
                requested="wb",
                used="cw",
                fallback=True,
                fallback_reason=msg,
                postprocessed=True,
                original_path=_glob_first(eicab_dir, _CW_PATTERNS),
            )

    base = _resolve_base_mask(eicab_dir, pref)
    return base


__all__ = [
    "CENTERLINES_MASK_PP_NIFTI",
    "EicabMaskKind",
    "EicabMaskResolution",
    "eicab_pp_path",
    "find_tof_resampled_volume",
    "resolve_eicab_mask",
]
