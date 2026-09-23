"""Inference-time stand-ins for the ToPBrain trainer classes.

Copied over ``nnunetv2/training/nnUNetTrainer/topbrain/topbrain_trainers.py`` when stage 5
assembles the build context. The real module imports the loss, the rare-class sampler and the
lateral-swap augmentation from ``nvitk``, and it is the last thing that would drag the library
into an image that needs none of it.
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import (
    PretrainedTrainer,
    PretrainedTrainer_Primus,
)

# Class-name prefixes and the base each maps to. Longest first: every Primus name also starts
# with the convolutional prefix, so testing in this order is what keeps them apart.
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
