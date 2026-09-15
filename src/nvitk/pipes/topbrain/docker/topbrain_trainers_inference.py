"""Inference-time stand-ins for the ToPBrain trainer classes.

Copied over ``nnunetv2/training/nnUNetTrainer/topbrain/topbrain_trainers.py`` when stage 5
assembles the build context. The real module imports the loss, the rare-class sampler and the
lateral-swap augmentation from ``nvitk``, and it is the last thing that would drag the library
into an image that needs none of it.

Why a stand-in is enough
------------------------
nnU-Net records the *name* of the trainer that produced a checkpoint and, at inference, looks it
up with ``recursive_find_python_class`` so it can call :meth:`build_network_architecture`. That
is the only thing it asks of the class. Everything that made the ToPBrain trainers different —
the Dice+CE+SkeletonRecall loss, rare-class-aware patch sampling, the lateral swap — happens only
during training and is never read back: the architecture comes from ``plans.json``, the weights
from the checkpoint, and even ``inference_allowed_mirroring_axes`` is stored *in* the checkpoint
rather than taken from the class. The trained models therefore load and predict identically.

Names are manufactured on demand
--------------------------------
The real module materialises one class per (encoder family x registered loss). Rather than
freeze that list — and fail to load a checkpoint trained with a loss added later — this
synthesises any ``nnUNetTrainerTopBrain*`` name through a module-level ``__getattr__``.
``recursive_find_python_class`` probes with ``hasattr``, which triggers it.
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import (
    PretrainedTrainer,
    PretrainedTrainer_Primus,
)

#: Class-name prefixes and the base each maps to. Longest first: every Primus name also starts
#: with the convolutional prefix, so testing in this order is what keeps them apart.
_PREFIXES: tuple[tuple[str, type], ...] = (
    ("nnUNetTrainerTopBrainPrimus", PretrainedTrainer_Primus),
    ("nnUNetTrainerTopBrain", PretrainedTrainer),
)

_cache: dict[str, type] = {}


def __getattr__(name: str) -> type:
    """Manufacture a ToPBrain trainer class for *name*, or raise :class:`AttributeError`.

    Cached, because the lookup may happen more than once and nnU-Net compares classes by
    identity in a few places.
    """
    if name in _cache:
        return _cache[name]
    for prefix, base in _PREFIXES:
        if name.startswith(prefix):
            cls = type(name, (base,), {
                "__doc__": (
                    f"Inference stand-in for {name}. Architecture and weights are identical to "
                    f"the trained model; the training-only loss and sampling are absent."
                ),
                "__module__": __name__,
            })
            _cache[name] = cls
            return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = []
