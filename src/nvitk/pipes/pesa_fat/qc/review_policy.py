"""Shared PESA-Fat QC review structure lists and status aggregation."""

from __future__ import annotations

from typing import Iterable

from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

QcStatus = str  # "PENDING" | "OK" | "FAIL"

REVIEW_ASPECTS: tuple[str, ...] = ("SEGMENTATION", "MEASUREMENT")
REVIEW_ASPECT_LABELS: dict[str, str] = {
    "SEGMENTATION": "Segmentation Mask Quality",
    "MEASUREMENT": "Measurement Quality",
}
DEFAULT_REVIEW_ASPECT = "MEASUREMENT"


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


def is_reviewable_aspect(aspect: str) -> bool:
    a = str(aspect).strip().upper()
    return a in REVIEW_ASPECTS


def is_reviewable_entry(
    structure: str,
    aspect: str,
    *,
    pipeline: str,
) -> bool:
    """True when *(structure, aspect)* is a valid QC review target."""
    if not is_reviewable_aspect(aspect):
        return False
    expected = expected_review_structures(pipeline)
    return is_reviewable_structure(structure, reviewable_structures=expected)


def expected_review_structures(pipeline: str) -> list[str]:
    """All structures that must be reviewed for dashboard overall OK."""
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


def expected_review_entries(pipeline: str) -> list[tuple[str, str]]:
    """All (structure, aspect) pairs required for a complete review."""
    return [
        (structure, aspect)
        for structure in expected_review_structures(pipeline)
        for aspect in REVIEW_ASPECTS
    ]


def _entry_status(
    reviews_by_entry: dict[tuple[str, str], str],
    structure: str,
    aspect: str,
) -> str:
    return reviews_by_entry.get((structure, aspect), "PENDING")


def reviews_dict_to_entries(reviews_by_structure: dict) -> dict[tuple[str, str], str]:
    """Normalize nested or flat review state into ``(structure, aspect) -> status``."""
    out: dict[tuple[str, str], str] = {}
    for key, val in reviews_by_structure.items():
        if isinstance(val, dict):
            structure = str(key).strip()
            for aspect, payload in val.items():
                a = str(aspect).strip().upper()
                if not is_reviewable_aspect(a):
                    continue
                if isinstance(payload, dict):
                    status = str(payload.get("qc_status", "PENDING")).strip().upper()
                else:
                    status = str(payload).strip().upper()
                out[(structure, a)] = status if status in {"PENDING", "OK", "FAIL"} else "PENDING"
            continue
        if "::" in str(key):
            structure, aspect = str(key).split("::", 1)
            a = aspect.strip().upper()
            if is_reviewable_aspect(a):
                status = str(val.get("qc_status", val) if isinstance(val, dict) else val).strip().upper()
                out[(structure.strip(), a)] = status if status in {"PENDING", "OK", "FAIL"} else "PENDING"
            continue
        structure = str(key).strip()
        status = str(val.get("qc_status", val) if isinstance(val, dict) else val).strip().upper()
        if status not in {"PENDING", "OK", "FAIL"}:
            status = "PENDING"
        out[(structure, DEFAULT_REVIEW_ASPECT)] = status
    return out


def overall_status(
    reviews_by_structure: dict[str, str],
    *,
    expected_structures: list[str] | None = None,
    pipeline: str | None = None,
) -> str:
    """Legacy aggregate: PENDING / OK / FAIL."""
    label, _tone = portal_display_status(
        reviews_by_structure,
        expected_structures=expected_structures,
        pipeline=pipeline,
    )
    if label == "REVISED":
        if _tone == "fail":
            return "FAIL"
        return "OK"
    return label


def portal_display_status(
    reviews_by_structure: dict,
    *,
    expected_structures: list[str] | None = None,
    pipeline: str | None = None,
) -> tuple[str, str]:
    """Return ``(label, tone)`` for dashboard cells.

    - ``PENDING`` / ``pending`` — any expected (structure, aspect) still pending.
    - ``REVISED`` / ``ok`` — all reviewed, not all FAIL (green).
    - ``REVISED`` / ``fail`` — all reviewed and all FAIL (red).
    """
    if pipeline is not None:
        entries = expected_review_entries(pipeline)
    elif expected_structures:
        entries = [(s, aspect) for s in expected_structures for aspect in REVIEW_ASPECTS]
    else:
        entries = []

    reviews_by_entry = reviews_dict_to_entries(reviews_by_structure)

    if entries:
        statuses = [_entry_status(reviews_by_entry, s, a) for s, a in entries]
        if any(s == "PENDING" for s in statuses):
            return ("PENDING", "pending")
        if all(s == "FAIL" for s in statuses):
            return ("REVISED", "fail")
        return ("REVISED", "ok")

    if not reviews_by_structure:
        return ("PENDING", "pending")
    statuses = list(reviews_by_entry.values()) or list(reviews_by_structure.values())
    if any(s == "PENDING" for s in statuses):
        return ("PENDING", "pending")
    if all(s == "FAIL" for s in statuses):
        return ("REVISED", "fail")
    return ("REVISED", "ok")


__all__ = [
    "DEFAULT_REVIEW_ASPECT",
    "REVIEW_ASPECTS",
    "REVIEW_ASPECT_LABELS",
    "expected_review_entries",
    "expected_review_structures",
    "filter_review_structures",
    "is_reviewable_aspect",
    "is_reviewable_entry",
    "is_reviewable_structure",
    "overall_status",
    "portal_display_status",
    "reviews_dict_to_entries",
]
