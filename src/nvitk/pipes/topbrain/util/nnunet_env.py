"""Point the in-tree nnU-Net build at the topbrain data roots.

Description
-----------
Thin adapter over :mod:`nvitk.pipes._engines.env`, which holds the pipeline-agnostic logic
(locating ``pipes/_engines/nnunet``, building the subprocess ``PYTHONPATH``). This module only
unpacks :class:`~nvitk.pipes.topbrain.util.paths.TopBrainPaths` into the three nnU-Net roots so
existing topbrain call sites keep their signatures.

``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
    Read lazily through ``nnunetv2.paths._EnvPath``, so unlike nnssl (see
    :mod:`~nvitk.pipes.topbrain.util.nnssl_env`) they can be set after import.

``PYTHONPATH``
    The in-tree build is **not** installed, because the rest of nvitk (TotalSegmentator in
    particular) depends on the released ``nnunetv2``. :func:`nnunet_env` prepends it for the
    *training subprocess only*.

Trainer discovery
-----------------
The build resolves trainers only within its own package, so the ToPBrain loss trainers live at
``_engines/nnunet/nnunetv2/training/nnUNetTrainer/topbrain/``. The loss implementations
themselves stay in :mod:`nvitk.segmentation.losses`.
"""

from __future__ import annotations

import os

from nvitk.core.logger import Logger
from nvitk.pipes._engines import env as _engine_env
from nvitk.pipes._engines.env import NNUNET_ENV_KEYS, nnunet_root
from nvitk.pipes.topbrain.util.paths import TopBrainPaths

log = Logger()


def nnunet_env(
    paths: TopBrainPaths,
    *,
    num_processes: int | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment variables an nnU-Net subprocess needs, as a plain dict.

    See :func:`nvitk.pipes._engines.env.nnunet_env`.
    """
    return _engine_env.nnunet_env(
        paths.nnunet_raw,
        paths.nnunet_preprocessed,
        paths.nnunet_results,
        num_processes=num_processes,
        extra=extra,
    )


def apply_nnunet_env(
    paths: TopBrainPaths,
    *,
    num_processes: int | None = None,
    create: bool = True,
) -> None:
    """Export the nnU-Net roots and trainer search path into this process.

    Parameters
    ----------
    create
        ``mkdir -p`` the three roots. nnU-Net's planners assume they exist.
    """
    env = nnunet_env(paths, num_processes=num_processes)
    os.environ.update(env)
    if create:
        paths.ensure_dirs(paths.nnunet_raw, paths.nnunet_preprocessed, paths.nnunet_results)
    log.debug("nnU-Net env: %s", env)


__all__ = [
    "NNUNET_ENV_KEYS",
    "apply_nnunet_env",
    "nnunet_env",
    "nnunet_root",
]
