"""Shared PESA-Fat QC review structure lists and status aggregation."""

from __future__ import annotations

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
        return filter_review_structures([spec.prefix for spec in dx_cfg.MEASURE_SPECS])
    return []


def overall_status(
    reviews_by_structure: dict[str, str],
    *,
    expected_structures: list[str] | None = None,
) -> str:
    """Aggregate per-structure QC into one status for a subject+pipeline.

    When *expected_structures* is set, ``OK`` requires every expected structure to be ``OK``.
    Any ``FAIL`` → ``FAIL``; otherwise ``PENDING``.
    """
    if expected_structures:
        if any(reviews_by_structure.get(s) == "FAIL" for s in expected_structures):
            return "FAIL"
        if all(reviews_by_structure.get(s) == "OK" for s in expected_structures):
            return "OK"
        return "PENDING"

    if not reviews_by_structure:
        return "PENDING"
    statuses = set(reviews_by_structure.values())
    if "FAIL" in statuses:
        return "FAIL"
    if statuses == {"OK"}:
        return "OK"
    return "PENDING"


__all__ = [
    "filter_review_structures",
    "expected_review_structures",
    "overall_status",
]
