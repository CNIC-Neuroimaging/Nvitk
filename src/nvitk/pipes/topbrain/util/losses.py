"""Selectable training objectives for the supervised and self-supervised stages.

Description
-----------
Four of the six metrics the challenge scores are topology or detection metrics, and vessels
occupy 0.2-0.5 % of a head volume. Neither fact is served by hard-coding Dice+CE, so the loss
is a first-class, swappable axis on both stages.

A name given to ``--loss`` / ``--ssl-loss`` resolves in three ways, in order:

1. a **built-in registry name** — see :data:`SEGMENTATION_LOSSES` and :data:`SSL_LOSSES`;
2. a **dotted path** ``package.module:Callable`` for a user-supplied loss;
3. nothing — an unknown name is an error listing the valid ones, never a silent fallback.

Torch-free by default
---------------------
The registry tables — now in :mod:`nvitk.segmentation.loss_registry`, shared with the CViT
pipeline and re-exported here — hold only names, descriptions and default kwargs, so validating
a ``--loss`` flag, listing the options, or printing ``--help`` never imports torch.
:func:`build_segmentation_loss` and :func:`build_ssl_loss` import it when a loss is actually
constructed, inside the worker.

How a name reaches nnU-Net
--------------------------
nnU-Net selects a loss by *trainer class*, not by argument. Each registry entry therefore has a
matching trainer in ``nnunet/nnunetv2/training/nnUNetTrainer/topbrain/``, named by
:func:`trainer_for_loss`. One class per loss, so each objective gets its own results folder
rather than overwriting the previous run.

Two families exist, because the encoder can be convolutional or a transformer:
:data:`TRAINER_PREFIX` builds on ``PretrainedTrainer`` (ResEnc-L) and
:data:`TRAINER_PREFIX_PRIMUS` on ``PretrainedTrainer_Primus`` (Primus-M).
:func:`trainer_for_loss` picks between them from the checkpoint's architecture.
"""

from __future__ import annotations

from nvitk.segmentation.loss_registry import (  # noqa: F401  (re-exported)
    SEGMENTATION_LOSSES,
    SSL_LOSSES,
    LossContext,
    LossSpec,
    build_segmentation_loss,
    build_ssl_loss,
    is_dotted_path,
    loss_spec_payload,
    parse_loss_config,
    resolve_dotted,
    validate_loss_name,
)

#: Prefix of the generated trainer classes for convolutional (ResEnc-L) encoders.
TRAINER_PREFIX: str = "nnUNetTrainerTopBrain"

#: Prefix of the generated trainer classes for transformer (Primus-M) encoders.
TRAINER_PREFIX_PRIMUS: str = "nnUNetTrainerTopBrainPrimus"

#: Environment variable through which a custom loss specification reaches the trainer. Only
#: used by the ``_custom`` trainer, which cannot take constructor arguments from nnU-Net.
LOSS_SPEC_ENV: str = "TOPBRAIN_LOSS_SPEC"

#: Environment variable overriding nnU-Net's fixed 1000-epoch schedule. Same mechanism and same
#: reason: nnU-Net exposes the epoch count only by subclassing.
EPOCHS_ENV: str = "TOPBRAIN_NUM_EPOCHS"


def trainer_for_loss(name: str, *, architecture: str = "ResEncL") -> str:
    """Trainer class implementing loss *name* for an *architecture* family.

    Parameters
    ----------
    architecture
        ``ResEncL`` (or any convolutional preset) selects the ``PretrainedTrainer`` family;
        anything beginning with ``Primus`` selects the transformer family. Taken from the
        pretrained checkpoint's adaptation plan, so the trainer always matches the weights.

    A custom dotted path maps to the ``_custom`` trainer of the same family, which reads its
    specification from :data:`LOSS_SPEC_ENV` — nnU-Net cannot pass constructor arguments.
    """
    prefix = TRAINER_PREFIX_PRIMUS if str(architecture).startswith("Primus") else TRAINER_PREFIX
    if is_dotted_path(name):
        return f"{prefix}_custom"
    validate_loss_name(name, registry=SEGMENTATION_LOSSES)
    return f"{prefix}_{name}"


__all__ = [
    "EPOCHS_ENV",
    "LOSS_SPEC_ENV",
    "TRAINER_PREFIX_PRIMUS",
    "SEGMENTATION_LOSSES",
    "SSL_LOSSES",
    "TRAINER_PREFIX",
    "LossContext",
    "LossSpec",
    "build_segmentation_loss",
    "build_ssl_loss",
    "is_dotted_path",
    "loss_spec_payload",
    "parse_loss_config",
    "resolve_dotted",
    "trainer_for_loss",
    "validate_loss_name",
]
