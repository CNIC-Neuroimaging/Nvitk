"""T1 volumetry atlas presets: region_id lists for ``DataRepo.image(..., modality='t1', atlas=...)``."""

from __future__ import annotations

from typing import Iterable

from .asl_atlases import _norm_regions
from .exceptions import FilterError
from .storage import normalize_variable_id


def _norm_region_token(value: str) -> str:
    return normalize_variable_id(str(value).strip())


# Populated by imports via register_t1_atlas_regions; keys match atlas= argument.
T1_ATLAS_REGIONS: dict[str, tuple[str, ...]] = {
    "cortical": _norm_regions(()),
    "subcortical": _norm_regions(()),
}


def register_t1_atlas_regions(atlas: str, region_ids: Iterable[str]) -> None:
    """Merge normalized region tokens into the named T1 atlas (cortical / subcortical)."""
    key = str(atlas).strip().lower()
    if key not in T1_ATLAS_REGIONS:
        allowed = ", ".join(sorted(T1_ATLAS_REGIONS))
        raise FilterError(f"Unknown T1 atlas {atlas!r}. Choose one of: {allowed}")
    existing = set(T1_ATLAS_REGIONS[key])
    for r in region_ids:
        t = _norm_region_token(r)
        if t:
            existing.add(t)
    T1_ATLAS_REGIONS[key] = tuple(sorted(existing))


def regions_for_t1_atlas(name: str) -> list[str]:
    """Return normalized ``region_id`` values for a named T1 atlas preset."""
    key = str(name).strip().lower()
    if key not in T1_ATLAS_REGIONS:
        allowed = ", ".join(sorted(T1_ATLAS_REGIONS))
        raise FilterError(f"Unknown T1 atlas '{name}'. Choose one of: {allowed}")
    return list(T1_ATLAS_REGIONS[key])
