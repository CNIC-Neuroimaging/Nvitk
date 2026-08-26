"""Point stock nnU-Net v2 at our data roots and at this pipeline's external trainers.

Description
-----------
The pipeline fine-tunes with **unmodified** ``nnunetv2`` — no fork, no vendored copy. Two
mechanisms make that possible:

``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results``
    Read lazily through ``nnunetv2.paths._EnvPath``, so unlike nnssl (see
    :mod:`~nvitk.pipes.topbrain.util.nnssl_env`) they can be set after import.

``PYTHONPATH``
    The pipeline's nnU-Net lives in-tree at ``pipes/topbrain/nnunet`` — a build carrying the
    nnssl fine-tuning support (``nnUNetv2_preprocess_like_nnssl``, ``PretrainedTrainer``) that
    the released ``nnunetv2`` does not have. It is **not** installed, because the rest of nvitk
    (TotalSegmentator in particular) depends on the released ``nnunetv2`` and shadowing it
    globally would break them. Instead :func:`nnunet_env` prepends the in-tree copy to
    ``PYTHONPATH`` for the *training subprocess only*, so the parent process keeps the released
    package and the subprocess gets the in-tree one.

Trainer discovery
-----------------
This build resolves trainers only within its own package, so the ToPBrain loss trainers live at
``nnunet/nnunetv2/training/nnUNetTrainer/topbrain/`` rather than in a separate directory. The
loss implementations themselves stay in :mod:`nvitk.segmentation.losses`.
"""

from __future__ import annotations

import os
from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain.util.paths import TopBrainPaths

log = Logger()

#: Environment variables nnU-Net reads for its three data roots.
NNUNET_ENV_KEYS: tuple[str, ...] = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")

def nnunet_root() -> Path:
    """The in-tree nnU-Net build carrying nnssl fine-tuning support.

    Raises
    ------
    FileNotFoundError
        Naming the expected path. Stage 2 is unusable without it, and a bare
        ``ModuleNotFoundError`` several frames deep is far harder to act on.
    """
    root = Path(__file__).resolve().parents[1] / "nnunet"
    if not (root / "nnunetv2" / "run" / "run_training_from_pretrained.py").is_file():
        raise FileNotFoundError(
            f"In-tree nnU-Net build not found under {root}. It must provide "
            f"nnunetv2/run/run_training_from_pretrained.py (the nnssl fine-tuning entry point)."
        )
    return root


def nnunet_env(
    paths: TopBrainPaths,
    *,
    num_processes: int | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment variables an nnU-Net subprocess needs, as a plain dict.

    Parameters
    ----------
    num_processes
        Caps both nnU-Net's general worker pool (``nnUNet_def_n_proc``) and its data-augmentation
        pool (``nnUNet_n_proc_DA``). Left unset, nnU-Net sizes them from the host's core count,
        which oversubscribes an SGE slot allocation.
    """
    root = str(nnunet_root())
    existing = os.environ.get("PYTHONPATH", "")
    env = {
        # The in-tree build must win over the released nnunetv2 for this subprocess only.
        "PYTHONPATH": os.pathsep.join([root, existing]) if existing else root,
        "nnUNet_raw": str(paths.nnunet_raw),
        "nnUNet_preprocessed": str(paths.nnunet_preprocessed),
        "nnUNet_results": str(paths.nnunet_results),
    }
    if num_processes is not None:
        env["nnUNet_def_n_proc"] = str(int(num_processes))
        env["nnUNet_n_proc_DA"] = str(int(num_processes))
    if extra:
        env.update(extra)
    return env



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
