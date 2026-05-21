"""Registry of integer label id → human name maps for GUI label selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# ── Schema entry ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelSchema:
    """Named label vocabulary (pipeline output or model taxonomy)."""

    key: str
    title: str
    group: str
    id_to_name: Mapping[int, str]
    description: str = ""

    def name_for(self, label_id: int) -> str | None:
        return self.id_to_name.get(int(label_id))

    def display(self, label_id: int) -> str:
        name = self.name_for(label_id)
        if name:
            return f"{name} ({label_id})"
        return f"Label {label_id}"


def _invert(name_to_id: Mapping[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in name_to_id.items()}


def _build_pipeline_schemas() -> dict[str, LabelSchema]:
    from nvitk.pipes.bbtpy.labels import BB_ARTERIAL_ID_TO_NAME
    from nvitk.pipes.pesa_fat.ct_pet_v5.labels import (
        BODY_LABELS,
        FAT_BATCH_LABELS,
        FAT_LABELS,
        MO_LABELS,
        MUSCLES_LABELS,
        ORGANS_LABELS,
    )
    from nvitk.pipes.pesa_fat.dixon_v5.labels import HEAD_LABELS, LEGS_LABELS, THORAX_LABELS
    from nvitk.pipes.qvtpy.labels import EICAB_ID_TO_NAME, QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID

    return {
        "generic": LabelSchema(
            "generic",
            "Generic (numeric only)",
            "General",
            {},
            "Show label ids without a named vocabulary.",
        ),
        "eicab": LabelSchema(
            "eicab",
            "eICAB (Circle of Willis input)",
            "Segmentation / vessels",
            dict(EICAB_ID_TO_NAME),
            "eICAB multilabel masks (*_eICAB_*.nii).",
        ),
        "qvtpy-4dflow": LabelSchema(
            "qvtpy-4dflow",
            "QVTpy — 4D flow / centerline",
            "Pipelines / QVTpy",
            dict(QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID),
            "seg_4dflow.nii, centerlines_mask.nii, LOCs.",
        ),
        "bbtpy-bb": LabelSchema(
            "bbtpy-bb",
            "BBTpy — black-blood arterial",
            "Pipelines / BBTpy",
            dict(BB_ARTERIAL_ID_TO_NAME),
            "Stage-2 BB segmentation after eICAB→BB relabel.",
        ),
        "pesa-fat-mo": LabelSchema(
            "pesa-fat-mo",
            "PESA-FAT — MO (vertebrae L3/L4)",
            "Pipelines / PESA-FAT",
            _invert(MO_LABELS),
            "MO.nii from ct-pet-v5 postprocess.",
        ),
        "pesa-fat-fat": LabelSchema(
            "pesa-fat-fat",
            "PESA-FAT — FAT",
            "Pipelines / PESA-FAT",
            _invert(FAT_LABELS),
            "FAT.nii (visceral / subcutaneous).",
        ),
        "pesa-fat-fat-batch": LabelSchema(
            "pesa-fat-fat-batch",
            "PESA-FAT — FAT batch",
            "Pipelines / PESA-FAT",
            _invert(FAT_BATCH_LABELS),
            "FAT_BATCH.nii (vertebra-limited).",
        ),
        "pesa-fat-body": LabelSchema(
            "pesa-fat-body",
            "PESA-FAT — BODY",
            "Pipelines / PESA-FAT",
            _invert(BODY_LABELS),
            "BODY.nii merged trunk + extremities.",
        ),
        "pesa-fat-organs": LabelSchema(
            "pesa-fat-organs",
            "PESA-FAT — organs",
            "Pipelines / PESA-FAT",
            _invert(ORGANS_LABELS),
            "ORGANS.nii (liver, spleen, pancreas).",
        ),
        "pesa-fat-muscles": LabelSchema(
            "pesa-fat-muscles",
            "PESA-FAT — muscles",
            "Pipelines / PESA-FAT",
            _invert(MUSCLES_LABELS),
            "MUSCLES.nii (quads, paravertebral, deltoid, trapezius).",
        ),
        "dixon-head": LabelSchema(
            "dixon-head",
            "Dixon — HEAD",
            "Pipelines / PESA-FAT Dixon",
            _invert(HEAD_LABELS),
            "HEAD.nii from dixon-v5.",
        ),
        "dixon-thorax": LabelSchema(
            "dixon-thorax",
            "Dixon — THORAX",
            "Pipelines / PESA-FAT Dixon",
            _invert(THORAX_LABELS),
            "THORAX.nii from dixon-v5.",
        ),
        "dixon-legs": LabelSchema(
            "dixon-legs",
            "Dixon — LEGS",
            "Pipelines / PESA-FAT Dixon",
            _invert(LEGS_LABELS),
            "LEGS.nii from dixon-v5.",
        ),
    }


def _build_totalsegmentator_schemas() -> dict[str, LabelSchema]:
    from nvitk.segmentation.total_segmentator.class_maps import AVAILABLE_TASKS, get_class_map

    out: dict[str, LabelSchema] = {}
    for task in AVAILABLE_TASKS:
        key = f"ts:{task}"
        cmap = get_class_map(task)
        out[key] = LabelSchema(
            key,
            f"TotalSegmentator — {task}",
            "TotalSegmentator",
            dict(cmap),
            f"Raw multi-label output for TS task '{task}'.",
        )
    return out


_SCHEMAS: dict[str, LabelSchema] | None = None


def all_schemas() -> dict[str, LabelSchema]:
    """Lazy-built catalog of every registered label schema."""
    global _SCHEMAS
    if _SCHEMAS is None:
        merged = _build_pipeline_schemas()
        merged.update(_build_totalsegmentator_schemas())
        _SCHEMAS = merged
    return _SCHEMAS


def schema_keys() -> list[str]:
    """Sorted schema keys (generic first, then pipelines, then TotalSegmentator)."""
    schemas = all_schemas()

    def _sort_key(k: str) -> tuple[int, str]:
        s = schemas[k]
        order = {"General": 0, "Segmentation / vessels": 1, "Pipelines / QVTpy": 2,
                 "Pipelines / BBTpy": 3, "Pipelines / PESA-FAT": 4,
                 "Pipelines / PESA-FAT Dixon": 5, "TotalSegmentator": 6}
        return (order.get(s.group, 99), s.title)

    return sorted(schemas.keys(), key=_sort_key)


def get_schema(key: str | None) -> LabelSchema | None:
    if not key or key == "generic":
        return all_schemas().get("generic")
    return all_schemas().get(key)


def schema_for_totalsegmentator_task(task: str) -> str:
    """Schema key matching a TotalSegmentator CLI task name."""
    key = f"ts:{task.strip()}"
    if key in all_schemas():
        return key
    return "generic"


# Filename / layer-name hints → schema key (first match wins).
_GUESS_RULES: tuple[tuple[str, str], ...] = (
    ("eicab", "eicab"),
    ("seg_4dflow", "qvtpy-4dflow"),
    ("centerlines_mask", "qvtpy-4dflow"),
    ("centerline", "qvtpy-4dflow"),
    ("fat_batch", "pesa-fat-fat-batch"),
    ("fat.nii", "pesa-fat-fat"),
    ("fat_", "pesa-fat-fat"),
    ("mo.nii", "pesa-fat-mo"),
    ("organs", "pesa-fat-organs"),
    ("muscles", "pesa-fat-muscles"),
    ("body.nii", "pesa-fat-body"),
    ("head.nii", "dixon-head"),
    ("thorax", "dixon-thorax"),
    ("legs.nii", "dixon-legs"),
)


def guess_schema_from_layer(layer: Any | None) -> str | None:
    """Heuristic schema from layer name or ``nvitk_metadata['source']``."""
    if layer is None:
        return None
    from nvitk.gui.spatial import nvitk_metadata_from_layer

    meta = nvitk_metadata_from_layer(layer)
    blob = " ".join(
        str(x).lower()
        for x in (meta.get("source"), getattr(layer, "name", None), meta.get("series_description"))
        if x
    )
    for needle, key in _GUESS_RULES:
        if needle in blob:
            return key
    return None


__all__ = [
    "LabelSchema",
    "all_schemas",
    "get_schema",
    "guess_schema_from_layer",
    "schema_for_totalsegmentator_task",
    "schema_keys",
]
