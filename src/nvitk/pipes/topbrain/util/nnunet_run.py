"""Invoke the in-tree nnU-Net build as a subprocess (topbrain re-export).

The implementation moved to :mod:`nvitk.pipes._engines.nnunet_run` when the vendored engines
were promoted to ``pipes/_engines`` so the CViT pipeline could share them. Every function there
already takes an explicit ``env`` dict, so this module simply re-exports it and existing
``from nvitk.pipes.topbrain.util import nnunet_run`` call sites keep working unchanged.
"""

from __future__ import annotations

from nvitk.pipes._engines.nnunet_run import *  # noqa: F401,F403
from nvitk.pipes._engines.nnunet_run import __all__  # noqa: F401
