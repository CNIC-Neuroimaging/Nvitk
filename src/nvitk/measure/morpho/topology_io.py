"""Morphometrics vessel-topology JSON helpers.

Topology files live under ``nvitk/measure/morpho/topology/*.json`` and use the
schema expected by :func:`nvitk.measure.morpho.io_utils.load_mapping`.

- ``eicab_topology.json`` — eICAB TOF label IDs (1–18). Used by qvtpy stage-7
  morphometrics (input masks are eICAB, not ``seg_4dflow``).
- ``qvtpy_topology.json`` — 4D-flow / qvtpy pipeline label IDs (arterial 1–14,
  venous 31–34). Reference only; not selected by the qvtpy pipeline.
- ``mouse_root_topology.json`` — mouse CoW root vessels.
"""

from __future__ import annotations

from pathlib import Path

from nvitk.measure.morpho.io_utils import load_mapping
from nvitk.measure.morpho.models import VesselInfo

TOPOLOGY_NONE = "none"
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
    "TOPOLOGY_NONE",
    "default_eicab_topology_path",
    "default_qvtpy_topology_path",
    "list_topology_jsons",
    "load_eicab_topology",
    "load_qvtpy_topology",
    "load_topology",
    "resolve_topology_path",
    "topology_choices",
    "topology_dir",
]
