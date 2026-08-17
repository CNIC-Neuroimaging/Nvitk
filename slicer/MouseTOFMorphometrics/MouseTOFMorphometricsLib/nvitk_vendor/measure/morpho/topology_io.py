# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/measure/morpho/topology_io.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Morphometrics vessel-topology JSON helpers.

Topology files live under ``nvitk/measure/morpho/topology/*.json`` and use the
schema expected by :func:`nvitk_vendor.measure.morpho.io_utils.load_mapping`.

- ``eicab_topology.json`` — eICAB TOF label IDs (1–18). Used by qvtpy stage-7
  morphometrics (input masks are eICAB, not ``seg_4dflow``).
- ``qvtpy_topology.json`` — 4D-flow / qvtpy pipeline label IDs (arterial 1–14,
  venous 31–34). Reference only; not selected by the qvtpy pipeline.
- ``mouse_root_topology.json`` — mouse CoW root vessels.

Beside the integer-keyed vessel entries a topology may carry an optional
``"_meta"`` object describing the *subject* the label ids belong to (species,
axis overrides, length scaling). ``load_mapping`` skips non-integer keys, so
``_meta`` is invisible to every existing consumer; read it with
:func:`load_topology_meta`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nvitk_vendor.measure.morpho.anatomy_axes import SPECIES_HUMAN, normalize_species
from nvitk_vendor.measure.morpho.io_utils import load_mapping
from nvitk_vendor.measure.morpho.models import VesselInfo

TOPOLOGY_NONE = "none"
TOPOLOGY_META_KEY = "_meta"
EICAB_TOPOLOGY_NAME = "eicab_topology.json"
QVTPY_TOPOLOGY_NAME = "qvtpy_topology.json"
MOUSE_ROOT_TOPOLOGY_NAME = "mouse_root_topology.json"


def topology_dir() -> Path:
    """Return the package directory that stores topology JSON files."""
    return Path(__file__).resolve().parent / "topology"


def list_topology_jsons() -> list[str]:
    """Sorted basenames of ``*.json`` topology files (excluding hidden)."""
    d = topology_dir()
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.json") if p.is_file())


def topology_choices() -> tuple[str, ...]:
    """GUI / CLI choices: ``none`` plus available topology JSON basenames."""
    return (TOPOLOGY_NONE, *list_topology_jsons())


def resolve_topology_path(name_or_path: str | Path | None) -> Path | None:
    """Resolve a topology name/path.

    Returns ``None`` for empty / ``none`` (vessel-wise only, no topology).
    """
    if name_or_path is None:
        return None
    token = str(name_or_path).strip()
    if not token or token.lower() == TOPOLOGY_NONE:
        return None
    p = Path(token).expanduser()
    if p.is_file():
        return p.resolve()
    cand = topology_dir() / token
    if cand.is_file():
        return cand.resolve()
    if not token.endswith(".json"):
        cand2 = topology_dir() / f"{token}.json"
        if cand2.is_file():
            return cand2.resolve()
    raise FileNotFoundError(f"Topology JSON not found: {name_or_path!r}")


def load_topology(name_or_path: str | Path | None) -> dict[int, VesselInfo] | None:
    """Load a topology mapping, or ``None`` for vessel-wise-only mode."""
    path = resolve_topology_path(name_or_path)
    if path is None:
        return None
    return load_mapping(str(path))


@dataclass(frozen=True)
class TopologyMeta:
    """Subject-level settings declared by a topology JSON's ``"_meta"`` block.

    ``species`` picks the animal frame used to resolve ``rostral``/``caudal``/
    ``dorsal``/``ventral`` in ``no_upstream_start``. ``axes_override`` replaces
    the header-derived axis codes for volumes whose NIfTI orientation labels are
    known to be wrong. ``length_scale`` rescales the human-calibrated millimetre
    thresholds (a mouse circle of Willis is roughly 0.15× human).
    """

    species: str = SPECIES_HUMAN
    axes_override: str | None = None
    length_scale: float = 1.0
    description: str = ""


def load_topology_meta(name_or_path: str | Path | None) -> TopologyMeta:
    """Read the ``"_meta"`` block of a topology JSON; defaults for none/absent."""
    path = resolve_topology_path(name_or_path)
    if path is None:
        return TopologyMeta()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return TopologyMeta()
    return parse_topology_meta(raw.get(TOPOLOGY_META_KEY) if isinstance(raw, dict) else None)


def parse_topology_meta(meta: dict | None) -> TopologyMeta:
    """Build a :class:`TopologyMeta` from a raw ``"_meta"`` dict (tolerant of junk)."""
    if not isinstance(meta, dict):
        return TopologyMeta()
    try:
        species = normalize_species(meta.get("species"))
    except ValueError:
        species = SPECIES_HUMAN
    override = str(meta.get("axes_override") or "").strip().upper() or None
    try:
        length_scale = float(meta.get("length_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        length_scale = 1.0
    if not (length_scale > 0):
        length_scale = 1.0
    return TopologyMeta(
        species=species,
        axes_override=override,
        length_scale=length_scale,
        description=str(meta.get("description") or ""),
    )


def default_eicab_topology_path() -> Path:
    """Path to the eICAB topology JSON (qvtpy stage-7 morphometrics default)."""
    return topology_dir() / EICAB_TOPOLOGY_NAME


def load_eicab_topology() -> dict[int, VesselInfo]:
    """Load the eICAB topology mapping (TOF / eICAB label IDs)."""
    path = default_eicab_topology_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing eICAB topology JSON: {path}")
    return load_mapping(str(path))


# Back-compat aliases (historically misnamed "qvtpy" while content was eICAB).
default_qvtpy_topology_path = default_eicab_topology_path
load_qvtpy_topology = load_eicab_topology


__all__ = [
    "EICAB_TOPOLOGY_NAME",
    "MOUSE_ROOT_TOPOLOGY_NAME",
    "QVTPY_TOPOLOGY_NAME",
    "TOPOLOGY_META_KEY",
    "TOPOLOGY_NONE",
    "TopologyMeta",
    "default_eicab_topology_path",
    "default_qvtpy_topology_path",
    "list_topology_jsons",
    "load_eicab_topology",
    "load_qvtpy_topology",
    "load_topology",
    "load_topology_meta",
    "parse_topology_meta",
    "resolve_topology_path",
    "topology_choices",
    "topology_dir",
]
