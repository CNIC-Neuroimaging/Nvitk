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

# ---- Filename patterns -------------------------------------------------------

_CW_PATTERNS = ("*_eICAB_CW.nii.gz", "*_eICAB_CW.nii")
_WB_PATTERNS = ("*_eICAB_WB.nii.gz", "*_eICAB_WB.nii")


@dataclass(frozen=True)
class EicabMaskResolution:
    """Result of :func:`resolve_eicab_mask`."""

    path: Path
    requested: EicabMaskKind
    used: EicabMaskKind
    fallback: bool
    fallback_reason: str | None


# ---------------------------------------------------------------------------
# Resolve CW / WB multilabel path
# ---------------------------------------------------------------------------


def _glob_first(eicab_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    for pat in patterns:
        hits = sorted(eicab_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def resolve_eicab_mask(
    eicab_dir: Path,
    preference: EicabMaskKind = "cw",
) -> EicabMaskResolution:
    """Return the eICAB label NIfTI path for *preference*, with warn-and-fallback.

    If the requested mask is missing but the alternate exists, logs a warning and
    uses the alternate. Raises :class:`FileNotFoundError` if neither exists.
    """
    pref = preference.strip().lower()  # type: ignore[assignment]
    if pref not in ("cw", "wb"):
        raise ValueError(f"eicab_mask preference must be 'cw' or 'wb', got {preference!r}")

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


__all__ = [
    "EicabMaskKind",
    "EicabMaskResolution",
    "resolve_eicab_mask",
]
