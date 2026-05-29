"""Shared PESA-Fat QC review structure lists and status aggregation."""

from __future__ import annotations

from typing import Iterable

from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

QcStatus = str  # "PENDING" | "OK" | "FAIL"


def filter_review_structures(structures: list[str] | tuple[str, ...]) -> list[str]:
    """Remove unwanted review labels (shared policy for CT-PET and Dixon)."""
    out: list[str] = []
    for s in structures:
        k = str(s).strip()
        if not k:
            continue
        ku = k.upper()
        if ku == "CORP":
            continue
        if ku.endswith("_LR"):
            continue
        if ku == "MO" or ku.startswith("MO_"):
            continue
        out.append(k)
    if any(str(s).strip().upper() == "BN" for s in structures) and not any(
        x.upper() == "BN" for x in out
    ):
        out.append("BN")
    return sorted(set(out))


def is_reviewable_structure(name: str, *, reviewable_structures: Iterable[str] | None = None) -> bool:
    """True when *name* participates in QC review (excludes ``*_LR`` composites)."""
    k = str(name).strip()
    if not k or k.upper().endswith("_LR"):
        return False
    if reviewable_structures is None:
        return True
    return k in {str(s).strip() for s in reviewable_structures if str(s).strip()}


def expected_review_structures(pipeline: str) -> list[str]:
    """All structures that must be reviewed OK for dashboard overall OK."""
    pl = str(pipeline).strip().lower()
    if pl == "ct-pet-v5":
        structs = sorted(
            {spec.column_prefix for spec in ct_cfg.SUV_SPECS}
            | {spec.column.replace("_VOL", "") for spec in ct_cfg.VOL_SPECS}
        )
        return filter_review_structures(structs)
    if pl == "dixon-v5":
        names: list[str] = []
        for spec in dx_cfg.MEASURE_SPECS:
            p = spec.prefix
            name = p.replace("DIXON_", "", 1) if p.startswith("DIXON_") else p
            names.append(name)
        return filter_review_structures(names)
    return []


def overall_status(
    reviews_by_structure: dict[str, str],
    *,
    expected_structures: list[str] | None = None,
) -> str:
    """Legacy aggregate: PENDING / OK / FAIL."""
    label, _tone = portal_display_status(
        reviews_by_structure, expected_structures=expected_structures
    )
    if label == "REVISED":
        if _tone == "fail":
            return "FAIL"
        return "OK"
    return label


def portal_display_status(
    reviews_by_structure: dict[str, str],
    *,
    expected_structures: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(label, tone)`` for dashboard cells.

    - ``PENDING`` / ``pending`` — any expected structure still pending.
    - ``REVISED`` / ``ok`` — all reviewed, not all FAIL (green).
    - ``REVISED`` / ``fail`` — all reviewed and all FAIL (red).
    """
    if expected_structures:
        statuses = [reviews_by_structure.get(s, "PENDING") for s in expected_structures]
        if any(s == "PENDING" for s in statuses):
            return ("PENDING", "pending")
        if all(s == "FAIL" for s in statuses):
            return ("REVISED", "fail")
        return ("REVISED", "ok")

    if not reviews_by_structure:
        return ("PENDING", "pending")
    statuses = list(reviews_by_structure.values())
    if any(s == "PENDING" for s in statuses):
        return ("PENDING", "pending")
    if all(s == "FAIL" for s in statuses):
        return ("REVISED", "fail")
    return ("REVISED", "ok")


__all__ = [
    "filter_review_structures",
    "expected_review_structures",
    "is_reviewable_structure",
    "overall_status",
    "portal_display_status",
]
