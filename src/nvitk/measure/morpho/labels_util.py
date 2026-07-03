"""Label selection and connected-component helpers."""

from __future__ import annotations

import re
from typing import Any, List

import numpy as np
from scipy import ndimage as ndi

from nvitk.measure.morphometrics_config import LABELS, PROCESS_SELECTED_TAGS_ONLY, SELECTED_TAGS
from .models import VesselInfo

def normalized_vessel_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def is_acoa_vessel(vessel_info: VesselInfo) -> bool:
    tokens = {
        normalized_vessel_token(vessel_info.name),
        normalized_vessel_token(vessel_info.full_name),
        normalized_vessel_token(vessel_info.pair),
    }
    return any(token in {"acoa", "acom", "acoma", "acommunicating", "anteriorcommunicatingartery"} for token in tokens)


def empty_vessel_info(label: int) -> VesselInfo:
    return VesselInfo(
        name=f"label_{label}", full_name="", side="", pair=None, territory="",
        flow_from="", flow_to=[], no_upstream_start=None,
    )


def resolve_labels_to_process(labels_all: List[int]) -> List[int]:
    labels_present = set(int(x) for x in labels_all)
    requested = [int(x) for x in SELECTED_TAGS] if PROCESS_SELECTED_TAGS_ONLY else (labels_all if LABELS == "auto" else [int(x) for x in LABELS])
    missing = [label for label in requested if label not in labels_present]
    if missing:
        print(f"Warning : requested tags not present in segmentation: {missing}")
    labels = [label for label in requested if label in labels_present]
    if not labels:
        raise ValueError("No requested labels/tags were found in the segmentation.")
    return labels


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    labels, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    comps = []
    for i in range(1, n + 1):
        comps.append(labels == i)
    comps.sort(key=lambda x: int(x.sum()), reverse=True)
    return comps


def keep_largest_component_per_label(multilabel: np.ndarray) -> np.ndarray:
    result = np.zeros_like(multilabel)
    for lv in np.unique(multilabel):
        if lv == 0:
            continue
        result[keep_largest_component(multilabel == lv)] = lv
    return result
