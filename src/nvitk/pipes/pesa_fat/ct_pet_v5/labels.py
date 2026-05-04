"""Output label definitions for the ct-pet-v5 post-processing stage.

These constants describe the canonical mask IDs written to disk so
downstream stages (stage 3, external consumers of the Excel summary) can
rely on a stable scheme.

Hemisphere-preserving changes w.r.t. v4
---------------------------------------

* Quadriceps left / right are kept separate (IDs 1, 2) instead of merged.
* Autochthon left / right are kept separate (IDs 3, 4) instead of merged.
* Deltoid is split into left / right (IDs 5, 6) via connected-component
  analysis on the bilateral TotalSegmentator label (id 9).
* Trapezius remains a single bilateral mask (ID 7) as TotalSegmentator emits.
"""

from __future__ import annotations

MO_LABELS: dict[str, int] = {"L4": 1, "L3": 2}

FAT_LABELS: dict[str, int] = {"GRASA_V": 1, "GRASA_SC": 2}

FAT_BATCH_LABELS: dict[str, int] = {"GRASA_V_BATCH": 1, "GRASA_SC_BATCH": 2}

BODY_LABELS: dict[str, int] = {"BODY": 1}

ORGANS_LABELS: dict[str, int] = {"HIGADO": 1, "BAZO": 2, "PANCREAS": 3}

MUSCLES_LABELS: dict[str, int] = {
    "CUADRICEPS_L": 1,
    "CUADRICEPS_R": 2,
    "PARAVERTEBRAL_L": 3,
    "PARAVERTEBRAL_R": 4,
    "DELTOIDES_L": 5,
    "DELTOIDES_R": 6,
    "TRAPECIOS": 7,
}

# (task, ts_label) pairs consumed by stage2_postprocess to build each output.
OUTPUT_LABEL_TO_TS: dict[str, list[tuple[str, str]]] = {
    # MO
    "L4": [("total", "vertebrae_L4")],
    "L3": [("total", "vertebrae_L3")],
    # FAT
    "GRASA_V": [("tissue_types", "torso_fat")],
    "GRASA_SC": [("tissue_types", "subcutaneous_fat")],
    "GRASA_V_BATCH": [("tissue_types", "torso_fat")],
    "GRASA_SC_BATCH": [("tissue_types", "subcutaneous_fat")],
    # BODY (merged)
    "BODY_TRUNC": [("body", "body_trunc")],
    "BODY_EXT": [("body", "body_extremities")],
    # ORGANS
    "HIGADO": [("total", "liver")],
    "BAZO": [("total", "spleen")],
    "PANCREAS": [("total", "pancreas")],
    # MUSCLES (hemisphere-aware)
    "CUADRICEPS_L": [("thigh_shoulder_muscles", "quadriceps_femoris_left")],
    "CUADRICEPS_R": [("thigh_shoulder_muscles", "quadriceps_femoris_right")],
    "PARAVERTEBRAL_L": [("total", "autochthon_left")],
    "PARAVERTEBRAL_R": [("total", "autochthon_right")],
    # DELTOIDES is a single bilateral TS label; split_lr_by_cc handles L/R.
    "DELTOIDES": [("thigh_shoulder_muscles", "deltoid")],
    "TRAPECIOS": [("thigh_shoulder_muscles", "trapezius")],
}

__all__ = [
    "MO_LABELS",
    "FAT_LABELS",
    "FAT_BATCH_LABELS",
    "BODY_LABELS",
    "ORGANS_LABELS",
    "MUSCLES_LABELS",
    "OUTPUT_LABEL_TO_TS",
]
