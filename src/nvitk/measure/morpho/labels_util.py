"""Label selection and connected-component helpers."""

from __future__ import annotations

import re
from typing import Any, List

import numpy as np
from scipy import ndimage as ndi

from nvitk.measure import morphometrics_config as _cfg
from .models import VesselInfo

def normalized_vessel_token(value: Any) -> str:
    """Lowercase *value* and strip non-alphanumeric characters, for fuzzy vessel-name matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def is_acoa_vessel(vessel_info: VesselInfo) -> bool:
    """True when *vessel_info* names the anterior communicating artery (ACoA), by any known alias."""
    tokens = {
        normalized_vessel_token(vessel_info.name),
        normalized_vessel_token(vessel_info.full_name),
        normalized_vessel_token(vessel_info.pair),
    }
    return any(token in {"acoa", "acom", "acoma", "acommunicating", "anteriorcommunicatingartery"} for token in tokens)


def empty_vessel_info(label: int) -> VesselInfo:
    """Placeholder :class:`VesselInfo` for a label id with no anatomical metadata."""
    return VesselInfo(
        name=f"label_{label}", full_name="", side="", pair=None, territory="",
        flow_from="", flow_to=[], no_upstream_start=None,
    )


def resolve_labels_to_process(labels_all: List[int]) -> List[int]:
    """Intersect the requested label set (config ``SELECTED_TAGS``/``LABELS``) with what's present.

    Warns (prints) about requested labels missing from the segmentation and raises
    if the resulting intersection is empty.
    """
    labels_present = set(int(x) for x in labels_all)
    requested = [int(x) for x in _cfg.SELECTED_TAGS] if _cfg.PROCESS_SELECTED_TAGS_ONLY else (labels_all if _cfg.LABELS == "auto" else [int(x) for x in _cfg.LABELS])
    missing = [label for label in requested if label not in labels_present]
    if missing:
        print(f"Warning : requested tags not present in segmentation: {missing}")
    labels = [label for label in requested if label in labels_present]
    if not labels:
        raise ValueError("No requested labels/tags were found in the segmentation.")
    return labels


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of only the largest 26-connected component of *mask*."""
    labels, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """26-connected components of *mask* as separate boolean masks, largest first."""
    labels, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    comps = []
    for i in range(1, n + 1):
        comps.append(labels == i)
    comps.sort(key=lambda x: int(x.sum()), reverse=True)
    return comps


def keep_largest_component_per_label(multilabel: np.ndarray) -> np.ndarray:
    """Per label id, keep only its largest connected component; drop stray islands."""
    result = np.zeros_like(multilabel)
    for lv in np.unique(multilabel):
        if lv == 0:
            continue
        result[keep_largest_component(multilabel == lv)] = lv
    return result


#: Proximal / distal halves of a split vessel (stage 1's WB_pp mask): suffix -> other half.
SPLIT_PARTNER_SUFFIXES: dict[str, str] = {
    "_A1": "_A2", "_A2": "_A1", "_M1": "_M2", "_M2": "_M1", "_P1": "_P2", "_P2": "_P1",
}


def split_partner_labels(vessel_name: str, mapping: dict) -> list[int]:
    """Label ids of the other half of a split vessel (``LMCA_M1`` -> ``LMCA_M2``'s id); empty for
    an unsplit vessel or when the other half is not in *mapping*."""
    name = str(vessel_name or "")
    for suffix, other in SPLIT_PARTNER_SUFFIXES.items():
        if name.endswith(suffix):
            want = name[: -len(suffix)] + other
            return [int(lid) for lid, info in mapping.items() if str(getattr(info, "name", "")) == want]
    return []


def split_partner_junctions(
    mask_cc: np.ndarray,
    multilabel: np.ndarray,
    partner_ids: list[int],
    spacing,
    *,
    margin_mm: float = 3.0,
    crop_mm: float = 15.0,
) -> np.ndarray:
    """The bifurcations a split vessel's component (*mask_cc*) shares with its other half: branch
    points of the skeleton of the merged mask (component + the *partner_ids* labels, in a crop
    *crop_mm* around the component) within *margin_mm* of the component, in mm (index × spacing,
    the frame of the measured paths). Measured alone, an M1 ends in the flare where its M2s leave,
    with no junction of its own there, and that flare reads as an enlargement."""
    from .skeleton import skeleton_tree_from_mask

    sp = np.asarray(spacing, dtype=float)
    if not partner_ids or not bool(np.asarray(mask_cc).any()):
        return np.empty((0, 3))
    own = np.asarray(mask_cc, dtype=bool)
    idx = np.argwhere(own)
    pad = np.ceil(float(crop_mm) / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.asarray(own.shape))
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    partner = np.isin(np.asarray(multilabel)[box], np.asarray(partner_ids, dtype=int))
    if not partner.any():
        return np.empty((0, 3))
    merged = own[box] | partner
    tree = skeleton_tree_from_mask(merged, spacing=sp)
    if not len(tree.branchpoints):
        return np.empty((0, 3))
    bp = np.asarray([tree.pts_vox[b] for b in tree.branchpoints], dtype=float).reshape(-1, 3)
    dist = ndi.distance_transform_edt(~own[box], sampling=sp)
    ijk = np.clip(np.round(bp).astype(int), 0, np.asarray(merged.shape) - 1)
    near = dist[ijk[:, 0], ijk[:, 1], ijk[:, 2]] <= float(margin_mm)
    return (bp[near] + lo) * sp
