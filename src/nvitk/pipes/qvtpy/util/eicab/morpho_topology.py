"""eICAB vessel-topology mapping for qvtpy stage-7 morphometrics.

Stage-7 still runs on eICAB TOF multilabel masks, so the topology source of
truth is ``nvitk/measure/morpho/topology/eicab_topology.json`` (not the
4D-flow ``qvtpy_topology.json`` reference file).
"""

from __future__ import annotations

from nvitk.measure.morpho.models import VesselInfo
from nvitk.measure.morpho.topology_io import load_eicab_topology


def build_eicab_topology_mapping() -> dict[int, VesselInfo]:
    """Return label-id -> :class:`VesselInfo` from ``eicab_topology.json``."""
    return load_eicab_topology()


__all__ = ["build_eicab_topology_mapping"]
