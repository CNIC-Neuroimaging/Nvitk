"""ASL atlas presets: region_id lists normalized like ``importers._region_id`` / ``normalize_variable_id``."""

from __future__ import annotations

from .exceptions import FilterError
from .storage import normalize_variable_id


# ──────────────────────────────────────────────────────────────────────────────
# Presets
# ──────────────────────────────────────────────────────────────────────────────


def _norm_regions(raw: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize each region name in *raw* via :func:`~nvitk.db.storage.normalize_variable_id`."""
    return tuple(normalize_variable_id(r) for r in raw)


# Keys: desikan, vascular-0, vascular-8, vascular-12
ASL_ATLAS_REGIONS: dict[str, tuple[str, ...]] = {
    "desikan": _norm_regions(
        (
            "ctx-whole-brain",
            "ctx-left-hemisphere",
            "ctx-right-hemisphere",
            "ctx-Left-Frontal-Lobe",
            "ctx-Right-Frontal-Lobe",
            "ctx-Left-Parietal-Lobe",
            "ctx-Right-Parietal-Lobe",
            "ctx-Left-Occipital-Lobe",
            "ctx-Right-Occipital-Lobe",
            "ctx-Left-Temporal-Lobe",
            "ctx-Right-Temporal-Lobe",
            "ctx-Left-Anterior-Cingulate",
            "ctx-Right-Anterior-Cingulate",
            "ctx-lh-bankssts",
            "ctx-rh-bankssts",
            "ctx-lh-caudalanteriorcingulate",
            "ctx-rh-caudalanteriorcingulate",
            "ctx-lh-caudalmiddlefrontal",
            "ctx-rh-caudalmiddlefrontal",
            "ctx-lh-cuneus",
            "ctx-rh-cuneus",
            "ctx-lh-entorhinal",
            "ctx-rh-entorhinal",
            "ctx-lh-fusiform",
            "ctx-rh-fusiform",
            "ctx-lh-inferiorparietal",
            "ctx-rh-inferiorparietal",
            "ctx-lh-inferiortemporal",
            "ctx-rh-inferiortemporal",
            "ctx-lh-isthmuscingulate",
            "ctx-rh-isthmuscingulate",
            "ctx-lh-lateraloccipital",
            "ctx-rh-lateraloccipital",
            "ctx-lh-lateralorbitofrontal",
            "ctx-rh-lateralorbitofrontal",
            "ctx-lh-lingual",
            "ctx-rh-lingual",
            "ctx-lh-medialorbitofrontal",
            "ctx-rh-medialorbitofrontal",
            "ctx-lh-middletemporal",
            "ctx-rh-middletemporal",
            "ctx-lh-parahippocampal",
            "ctx-rh-parahippocampal",
            "ctx-lh-paracentral",
            "ctx-rh-paracentral",
            "ctx-lh-parsopercularis",
            "ctx-rh-parsopercularis",
            "ctx-lh-parsorbitalis",
            "ctx-rh-parsorbitalis",
            "ctx-lh-parstriangularis",
            "ctx-rh-parstriangularis",
            "ctx-lh-pericalcarine",
            "ctx-rh-pericalcarine",
            "ctx-lh-postcentral",
            "ctx-rh-postcentral",
            "ctx-lh-posteriorcingulate",
            "ctx-rh-posteriorcingulate",
            "ctx-lh-precentral",
            "ctx-rh-precentral",
            "ctx-lh-precuneus",
            "ctx-rh-precuneus",
            "ctx-lh-rostralanteriorcingulate",
            "ctx-rh-rostralanteriorcingulate",
            "ctx-lh-rostralmiddlefrontal",
            "ctx-rh-rostralmiddlefrontal",
            "ctx-lh-superiorfrontal",
            "ctx-rh-superiorfrontal",
            "ctx-lh-superiorparietal",
            "ctx-rh-superiorparietal",
            "ctx-lh-superiortemporal",
            "ctx-rh-superiortemporal",
            "ctx-lh-supramarginal",
            "ctx-rh-supramarginal",
            "ctx-lh-frontalpole",
            "ctx-rh-frontalpole",
            "ctx-lh-temporalpole",
            "ctx-rh-temporalpole",
            "ctx-lh-transversetemporal",
            "ctx-rh-transversetemporal",
            "ctx-lh-insula",
            "ctx-rh-insula",
        )
    ),
    "vascular-0": _norm_regions(
        (
            "Left_ACA-0",
            "Left_Basilar-0",
            "Left_MCA-0",
            "Left_PCA-0",
            "Right_ACA-0",
            "Right_Basilar-0",
            "Right_MCA-0",
            "Right_PCA-0",
            "Watershed-0",
        )
    ),
    "vascular-8": _norm_regions(
        (
            "Left_ACA-8",
            "Left_Basilar-8",
            "Left_MCA-8",
            "Left_PCA-8",
            "Right_ACA-8",
            "Right_Basilar-8",
            "Right_MCA-8",
            "Right_PCA-8",
            "Watershed-8",
        )
    ),
    "vascular-12": _norm_regions(
        (
            "Left_ACA-12",
            "Left_Basilar-12",
            "Left_MCA-12",
            "Left_PCA-12",
            "Right_ACA-12",
            "Right_Basilar-12",
            "Right_MCA-12",
            "Right_PCA-12",
            "Watershed-12",
        )
    ),
}


def regions_for_atlas(name: str) -> list[str]:
    """Return normalized ``region_id`` values for a named ASL atlas preset."""
    key = str(name).strip().lower()
    if key not in ASL_ATLAS_REGIONS:
        allowed = ", ".join(sorted(ASL_ATLAS_REGIONS))
        raise FilterError(f"Unknown ASL atlas '{name}'. Choose one of: {allowed}")
    return list(ASL_ATLAS_REGIONS[key])
