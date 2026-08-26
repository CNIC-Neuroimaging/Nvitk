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
        """Human name registered for *label_id*, or ``None`` if this schema doesn't have one."""
        return self.id_to_name.get(int(label_id))

    def display(self, label_id: int) -> str:
        """Display string for *label_id*: ``"<name> (<id>)"`` if named, else ``"Label <id>"``."""
        name = self.name_for(label_id)
        if name:
            return f"{name} ({label_id})"
        return f"Label {label_id}"


def _invert(name_to_id: Mapping[str, int]) -> dict[int, str]:
    """Flip a ``{name: id}`` mapping to ``{id: name}``, for building an :class:`LabelSchema`."""
    return {int(v): str(k) for k, v in name_to_id.items()}


def _build_pipeline_schemas() -> dict[str, LabelSchema]:
    """Build the registry of :class:`LabelSchema` entries for every known pipeline label vocabulary
    (eICAB, QVTpy, BBTpy, PESA-FAT, Dixon)."""
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
    from nvitk.pipes.topbrain.labels import label_map as topbrain_label_map

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
        # ── TopBrain ────────────────────────────────────────────────────────
        # Three vocabularies that share the value range 1-34 and diverge above it. Keeping them
        # as separate schemas is what stops label 35 being shown as "VoG" on a mask where it
        # means "R-ICA-C1-C5"; :func:`guess_schema_from_layer` picks between them.
        "topbrain-ta36": LabelSchema(
            "topbrain-ta36",
            "TopBrain — TA36 (36-class, CTA+MRA)",
            "Pipelines / TopBrain",
            topbrain_label_map("ta36"),
            "labelsTr_topbrain_v2_topaneu36class, Dataset501 masks and predictions. "
            "Modality-agnostic; 35/36 are the infraclinoid ICA segments.",
        ),
        "topbrain-v1-ct": LabelSchema(
            "topbrain-v1-ct",
            "TopBrain — v1 CTA (40-class, with veins)",
            "Pipelines / TopBrain",
            topbrain_label_map("v1_ct"),
            "labelsTr_topbrain_v1_ct, Dataset502. CTA only; 35-40 are the venous classes "
            "(VoG, StS, ICVs, BVR, SSS).",
        ),
        "topbrain-v1-mr": LabelSchema(
            "topbrain-v1-mr",
            "TopBrain — v1 MRA (42-class, with extracranial)",
            "Pipelines / TopBrain",
            topbrain_label_map("v1_mr"),
            "labelsTr_topbrain_v1_mr, Dataset503. MRA only; 35-42 are the extracranial "
            "carotid branches (ECA, STA, MaxA, MMA).",
        ),
    }


def _build_totalsegmentator_schemas() -> dict[str, LabelSchema]:
    """Build one :class:`LabelSchema` per registered TotalSegmentator task, keyed ``ts:<task>``."""
    from nvitk.segmentation.total_segmentator.class_maps import AVAILABLE_TASKS, get_class_map

    out = {}
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


#: The three TopBrain schema keys, in the order the challenge introduced them.
TOPBRAIN_SCHEMA_KEYS: tuple[str, ...] = ("topbrain-ta36", "topbrain-v1-ct", "topbrain-v1-mr")

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
        """Sort key placing schema *k* by its display group order, then title, alphabetically."""
        s = schemas[k]
        order = {"General": 0, "Segmentation / vessels": 1, "Pipelines / QVTpy": 2,
                 "Pipelines / BBTpy": 3, "Pipelines / TopBrain": 4,
                 "Pipelines / PESA-FAT": 5, "Pipelines / PESA-FAT Dixon": 6,
                 "TotalSegmentator": 7}
        return (order.get(s.group, 99), s.title)

    return sorted(schemas.keys(), key=_sort_key)


def get_schema(key: str | None) -> LabelSchema | None:
    """Look up the schema registered under *key* (``"generic"`` or falsy *key* returns the generic
    schema); ``None`` if unregistered."""
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
    # TopBrain first: its release directory names are unambiguous, and the three label sets
    # collide above value 34, so a wrong guess renames real anatomy rather than just failing.
    ("topaneu36class", "topbrain-ta36"),
    ("topbrainta36", "topbrain-ta36"),
    ("dataset501", "topbrain-ta36"),
    ("topbrain_v1_ct", "topbrain-v1-ct"),
    ("topbrainv1ct", "topbrain-v1-ct"),
    ("dataset502", "topbrain-v1-ct"),
    ("topbrain_v1_mr", "topbrain-v1-mr"),
    ("topbrainv1mr", "topbrain-v1-mr"),
    ("dataset503", "topbrain-v1-mr"),
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


#: Substrings marking a mask as TopBrain when the label set itself is not named in the path —
#: predictions, for instance, live under a trainer-named directory.
_TOPBRAIN_HINTS: tuple[str, ...] = ("topbrain", "topcow", "topaneu")


def _max_label_value(layer: Any) -> int | None:
    """Largest label value in *layer*, or ``None`` if it cannot be read cheaply.

    One pass over an integer mask; a few tens of milliseconds even for the ~70 M-voxel
    angiographic volumes this catalog is used with. Any failure (lazy array, unreadable dtype,
    non-integer data) returns ``None`` so the caller falls back to name matching rather than
    raising inside a GUI event handler.
    """
    try:
        from nvitk.core.array import to_numpy

        data = getattr(layer, "data", None)
        if data is None:
            return None
        array = to_numpy(data)
        if array.size == 0 or array.dtype.kind not in "iub":
            return None
        return int(array.max())
    except Exception:
        return None


def _guess_topbrain_schema(layer: Any, blob: str) -> str | None:
    """Disambiguate the three TopBrain label sets when the path does not name one.

    The release stores all three under the *same* filenames (``topcow_ct_001.nii.gz``), and our
    own predictions land in a directory named after the trainer, so the label set often cannot
    be recovered from the path at all. Two independent signals settle it:

    * **modality**, from the ``topcow_{ct|mr}_`` filename convention — ``v1_ct`` masks only
      exist for CTA and ``v1_mr`` only for MRA;
    * **the largest label value present**, since the sets top out at 36, 40 and 42.

    A maximum of 36 or below is reported as TA36: that is this pipeline's own output, and it is
    also the safe answer, because every value it names means the same thing in all three sets
    except 35/36 — which is exactly the range a higher maximum resolves.
    """
    if not any(hint in blob for hint in _TOPBRAIN_HINTS):
        return None

    maximum = _max_label_value(layer)
    if maximum is None:
        return "topbrain-ta36"

    if maximum > 40:
        return "topbrain-v1-mr"  # 41/42 (MMA) exist only here
    if maximum > 36:
        # 37-40 are venous on CTA and extracranial on MRA; the filename says which.
        if "topcow_mr" in blob or "_mr_" in blob:
            return "topbrain-v1-mr"
        if "topcow_ct" in blob or "_ct_" in blob:
            return "topbrain-v1-ct"
        return None  # ambiguous: let the user choose rather than guess wrong
    return "topbrain-ta36"


def guess_schema_from_layer(layer: Any | None) -> str | None:
    """Heuristic schema from layer name or ``nvitk_metadata['source']``."""
    if layer is None:
        return None
    from nvitk.gui.core.spatial import nvitk_metadata_from_layer

    meta = nvitk_metadata_from_layer(layer)
    blob = " ".join(
        str(x).lower()
        for x in (meta.get("source"), getattr(layer, "name", None), meta.get("series_description"))
        if x
    )
    for needle, key in _GUESS_RULES:
        if needle in blob:
            return key
    # Only reached when nothing in the path named a label set; TopBrain masks routinely land
    # here, so fall back to inspecting the data itself.
    return _guess_topbrain_schema(layer, blob)


__all__ = [
    "LabelSchema",
    "TOPBRAIN_SCHEMA_KEYS",
    "all_schemas",
    "get_schema",
    "guess_schema_from_layer",
    "schema_for_totalsegmentator_task",
    "schema_keys",
]
