"""Facade over the vendored morphometrics pipeline.

Everything the Slicer widget needs from ``nvitk_vendor``, imported lazily so a
missing pip package surfaces as the module's dependency message rather than an
import error at Slicer startup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import deps


def _api() -> dict[str, Any]:
    """Import the vendored entry points (raises a helpful error if deps are absent)."""
    deps.ensure()
    from nvitk_vendor.measure.morpho.anatomy_axes import SPECIES_AUTO, SPECIES_CHOICES
    from nvitk_vendor.measure.morpho.topology_io import (
        MOUSE_ROOT_TOPOLOGY_NAME,
        TOPOLOGY_NONE,
        load_topology_meta,
        topology_choices,
    )
    from nvitk_vendor.measure.morphometrics import run_morphometrics_case

    return {
        "MOUSE_ROOT_TOPOLOGY_NAME": MOUSE_ROOT_TOPOLOGY_NAME,
        "SPECIES_AUTO": SPECIES_AUTO,
        "SPECIES_CHOICES": tuple(SPECIES_CHOICES),
        "TOPOLOGY_NONE": TOPOLOGY_NONE,
        "load_topology_meta": load_topology_meta,
        "run_morphometrics_case": run_morphometrics_case,
        "topology_choices": topology_choices,
    }


def topology_choices() -> tuple[str, ...]:
    """Topology JSON basenames bundled with the module, plus ``'none'``."""
    return tuple(_api()["topology_choices"]())


def species_choices() -> tuple[str, ...]:
    """``('auto', 'human', 'mouse')``."""
    return tuple(_api()["SPECIES_CHOICES"])


def default_topology() -> str:
    """``mouse_root_topology.json`` when present — this module's primary use case."""
    api = _api()
    return api["MOUSE_ROOT_TOPOLOGY_NAME"] if api["MOUSE_ROOT_TOPOLOGY_NAME"] in topology_choices() else api["TOPOLOGY_NONE"]


def topology_meta(name: str):
    """The topology's ``_meta`` block (species, length_scale, axes_override)."""
    return _api()["load_topology_meta"](name)


def none_topology() -> str:
    """The sentinel meaning "no topology, per-label metrics only"."""
    return _api()["TOPOLOGY_NONE"]


def run_case(seg_path: str, out_dir: str, **kwargs: Any):
    """Run the vendored ``run_morphometrics_case``; returns the workbook path."""
    return _api()["run_morphometrics_case"](seg_path, out_dir, **kwargs)


def bridge_same_label_components(seg: Any, max_gap: int = 24) -> np.ndarray:
    """Reconnect same-label fragments with MST tubes before measuring.

    Returns the input unchanged if the bridge is unavailable, so a partial
    install degrades instead of aborting the run.
    """
    out = np.asarray(seg, dtype=np.int32).copy()
    try:
        from scipy import ndimage as ndi

        from nvitk_vendor.morphology.mst_bridge import bridge_binary_components_mst
    except Exception:  # noqa: BLE001
        return out

    for lab in (int(x) for x in np.unique(out) if int(x) != 0):
        mask = out == lab
        if int(ndi.label(mask)[1]) <= 1:
            continue
        try:
            bridged = bridge_binary_components_mst(mask, max_gap=max_gap, tube_radius=1)
        except Exception:  # noqa: BLE001
            continue
        out[np.asarray(bridged, dtype=bool)] = lab
    return out


__all__ = [
    "bridge_same_label_components",
    "default_topology",
    "none_topology",
    "run_case",
    "species_choices",
    "topology_choices",
    "topology_meta",
]
