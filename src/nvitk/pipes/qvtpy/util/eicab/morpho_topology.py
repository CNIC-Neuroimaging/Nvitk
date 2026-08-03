"""eICAB / qvtpy label topology for TOF morphometrics.

The vessel graph is defined by
``nvitk/measure/morpho/topology/qvtpy_topology.json``. This module keeps a
compatibility loader used by legacy imports.
"""

from __future__ import annotations

from nvitk.measure.morpho.models import VesselInfo
from nvitk.measure.morpho.topology_io import load_qvtpy_topology


def build_eicab_topology_mapping() -> dict[int, VesselInfo]:
    """Return label-id -> :class:`VesselInfo` from ``qvtpy_topology.json``."""
    return load_qvtpy_topology()


__all__ = ["build_eicab_topology_mapping"]
