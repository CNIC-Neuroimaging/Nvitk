"""Self-contained compute helpers for the Mouse TOF Morphometrics Slicer module.

Like ``MouseTOFCoWLib``, this package needs no nvitk installation: the
morphometrics pipeline is vendored under ``nvitk_vendor/`` (see
``nvitk_vendor/VENDORED.md``). Unlike the CoW module's simplified blood-flood
port, the algorithm modules here are copied **verbatim** from nvitk with only
the root package renamed, so the measurements are exactly those of the upstream
pipeline.

Importing this package puts its own directory on ``sys.path`` so the vendored
tree resolves as the top-level package ``nvitk_vendor``, matching the rewritten
import lines inside it.
"""

from __future__ import annotations

import os
import sys

_lib_dir = os.path.dirname(os.path.abspath(__file__))
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

from . import deps, morphometrics  # noqa: E402
from .deps import PIP_INSTALL_ARGS  # noqa: E402
from .morphometrics import (  # noqa: E402
    bridge_same_label_components,
    default_topology,
    none_topology,
    run_case,
    species_choices,
    topology_choices,
    topology_meta,
)

__all__ = [
    "PIP_INSTALL_ARGS",
    "bridge_same_label_components",
    "default_topology",
    "deps",
    "morphometrics",
    "none_topology",
    "run_case",
    "species_choices",
    "topology_choices",
    "topology_meta",
]
