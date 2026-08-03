"""Morphometrics vessel-topology JSON helpers.

Topology files live under ``nvitk/measure/morpho/topology/*.json`` and use the
schema expected by :func:`nvitk.measure.morpho.io_utils.load_mapping`.
"""

from __future__ import annotations

from pathlib import Path

from nvitk.measure.morpho.io_utils import load_mapping
from nvitk.measure.morpho.models import VesselInfo

TOPOLOGY_NONE = "none"
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


def default_qvtpy_topology_path() -> Path:
    """Path to the qvtpy / eICAB topology JSON (source of truth for stage-7)."""
    return topology_dir() / QVTPY_TOPOLOGY_NAME


def load_qvtpy_topology() -> dict[int, VesselInfo]:
    """Load the default qvtpy topology mapping."""
    path = default_qvtpy_topology_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing qvtpy topology JSON: {path}")
    return load_mapping(str(path))


__all__ = [
    "MOUSE_ROOT_TOPOLOGY_NAME",
    "QVTPY_TOPOLOGY_NAME",
    "TOPOLOGY_NONE",
    "default_qvtpy_topology_path",
    "list_topology_jsons",
    "load_qvtpy_topology",
    "load_topology",
    "resolve_topology_path",
    "topology_choices",
    "topology_dir",
]
