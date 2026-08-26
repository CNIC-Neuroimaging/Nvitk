"""ToPBrain trainers: nnssl-pretrained fine-tuning with a selectable loss.

Two families, matching the two encoder types this pipeline can start from:

``nnUNetTrainerTopBrain_<loss>``
    Convolutional (ResEnc-L) encoders, on top of :class:`PretrainedTrainer`.
``nnUNetTrainerTopBrainPrimus_<loss>``
    Transformer (Primus-M) encoders, on top of :class:`PretrainedTrainer_Primus`.

Both inherit the pretrained-weight loading, warm-up schedule and learning rates that the two
base classes configure — this module only replaces the objective and the mirroring policy.

Why one class per loss
----------------------
nnU-Net selects a loss by trainer class and names its output folder
``{trainer}__{plans}__{configuration}``. One class per loss therefore gives each objective its
own results directory instead of silently overwriting the previous run, which is the whole point
of being able to compare them.

Mirroring
---------
Every trainer here disables mirroring, in training **and** at inference. TopBrain labels are
lateralised (``R-ICA`` vs ``L-ICA``, ``R-M1`` vs ``L-M1``, …), so a left-right flip produces an
image whose correct labels are the mirrored *class ids*, not the mirrored mask. The base class
sets ``inference_allowed_mirroring_axes`` separately, so both have to be cleared.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import (
    PretrainedTrainer,
    PretrainedTrainer_Primus,
)

from nvitk.pipes.topbrain.util.losses import (
    EPOCHS_ENV,
    LOSS_SPEC_ENV,
    SEGMENTATION_LOSSES,
    TRAINER_PREFIX,
    TRAINER_PREFIX_PRIMUS,
    LossContext,
    build_segmentation_loss,
)


class _TopBrainLossMixin:
    """Loss selection, epoch override and no-mirroring, shared by both encoder families.

    A mixin rather than a base class so the two families keep their own pretrained base
    (``PretrainedTrainer`` / ``PretrainedTrainer_Primus``) and everything those configure.
    """

    #: Registry name of the loss this trainer optimises.
    loss_name: str = "dice_ce"

    #: Keyword overrides forwarded to the loss factory.
    loss_config: dict = {}

    def _loss_context(self) -> LossContext:
        """Snapshot the trainer state a loss needs in order to configure itself."""
        return LossContext(
            batch_dice=self.configuration_manager.batch_dice,
            has_regions=self.label_manager.has_regions,
            ignore_label=self.label_manager.ignore_label,
            is_ddp=self.is_ddp,
            num_classes=self.label_manager.num_segmentation_heads,
        )

    def _resolve_loss(self) -> tuple[str, dict]:
        """The loss name and kwargs for this trainer. Overridden by the custom trainers."""
        return self.loss_name, dict(self.loss_config)

    def initialize(self) -> None:
        """Apply the optional epoch-count override, then initialise as usual.

        Hooks ``initialize`` rather than ``__init__``: ``nnUNetTrainer.__init__`` rebuilds its
        ``my_init_kwargs`` by introspecting its own signature and looking each parameter up in
        ``locals()``, so a subclass taking ``*args, **kwargs`` makes it raise ``KeyError``.
        ``initialize`` still runs before the optimiser and its schedule are built, which is the
        only ordering ``num_epochs`` requires.
        """
        override = os.environ.get(EPOCHS_ENV)
        if override:
            try:
                self.num_epochs = int(override)
            except ValueError:
                raise ValueError(f"{EPOCHS_ENV}={override!r} is not an integer.") from None
            self.print_to_log_file(f"{EPOCHS_ENV} override: training for {self.num_epochs} epochs")
        super().initialize()

    def _build_loss(self):
        """Build the configured loss, with deep supervision only if the base class enabled it.

        The pretrained trainers set ``enable_deep_supervision = False``, so in practice this
        returns the bare loss; the wrapping branch is kept because a from-scratch variant of the
        same trainer re-enables it.
        """
        name, config = self._resolve_loss()
        self.print_to_log_file(f"Building loss {name!r} with config {config}")
        loss = build_segmentation_loss(name, self._loss_context(), config)

        if self._do_i_compile() and hasattr(loss, "dc"):
            loss.dc = torch.compile(loss.dc)

        if not self.enable_deep_supervision:
            return loss

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
        if self.is_ddp and not self._do_i_compile():
            # DDP trips over a weight of exactly 0 (unused parameters); a tiny value is the same
            # fix nnU-Net itself applies.
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()
        return DeepSupervisionWrapper(loss, weights)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Disable mirroring; keep the rotation and patch-size logic untouched.

        Clearing ``inference_allowed_mirroring_axes`` matters as much as clearing the training
        axes — the base implementation sets it from ``mirror_axes`` just before returning, and
        leaving it populated would keep mirroring in test-time augmentation, producing
        side-confused predictions from a model never trained to be flip-invariant.
        """
        rotation, dummy_2d, initial_patch_size, _ = super(
        ).configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy_2d, initial_patch_size, None


class _TopBrainCustomLossMixin(_TopBrainLossMixin):
    """Reads its loss specification from :data:`LOSS_SPEC_ENV`.

    nnU-Net's CLI can only pass a trainer *name*, so an arbitrary ``--custom-loss`` reference
    plus its kwargs has to travel out of band.
    """

    def _resolve_loss(self) -> tuple[str, dict]:
        """Read the loss specification from the environment.

        Raises
        ------
        RuntimeError
            If the variable is unset or malformed. Falling back to a default would train a model
            the user did not ask for and label the run as if they had.
        """
        raw = os.environ.get(LOSS_SPEC_ENV)
        if not raw:
            raise RuntimeError(
                f"{type(self).__name__} requires {LOSS_SPEC_ENV} to be set to a JSON loss "
                f"specification. Run through 'nvitk-topbrain' rather than invoking "
                f"nnUNetv2_train_pretrained directly."
            )
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{LOSS_SPEC_ENV} is not valid JSON: {exc}") from None
        name = spec.get("loss")
        if not name:
            raise RuntimeError(f"{LOSS_SPEC_ENV} has no 'loss' key: {raw}")
        return str(name), dict(spec.get("config") or {})


def _make(prefix: str, base: type, mixin: type, loss_name: str | None) -> type:
    """Build one trainer class. ``loss_name=None`` produces the custom-loss variant."""
    suffix = loss_name if loss_name is not None else "custom"
    doc = (
        f"ToPBrain {'transformer' if 'Primus' in prefix else 'convolutional'} trainer, "
        f"no mirroring, loss {suffix!r}."
    )
    if loss_name is not None:
        doc += f"\n\n{SEGMENTATION_LOSSES[loss_name].description}"
    attributes: dict = {"__doc__": doc}
    if loss_name is not None:
        attributes.update({"loss_name": loss_name, "loss_config": {}})
    return type(f"{prefix}_{suffix}", (mixin, base), attributes)


# One trainer per registered loss, for each encoder family, materialised into the module
# namespace where nnU-Net's getattr-based lookup finds them.
for _prefix, _base in ((TRAINER_PREFIX, PretrainedTrainer),
                       (TRAINER_PREFIX_PRIMUS, PretrainedTrainer_Primus)):
    for _loss in SEGMENTATION_LOSSES:
        _cls = _make(_prefix, _base, _TopBrainLossMixin, _loss)
        globals()[_cls.__name__] = _cls
    _cls = _make(_prefix, _base, _TopBrainCustomLossMixin, None)
    globals()[_cls.__name__] = _cls
del _prefix, _base, _loss, _cls


__all__ = [
    *(f"{TRAINER_PREFIX}_{name}" for name in SEGMENTATION_LOSSES),
    f"{TRAINER_PREFIX}_custom",
    *(f"{TRAINER_PREFIX_PRIMUS}_{name}" for name in SEGMENTATION_LOSSES),
    f"{TRAINER_PREFIX_PRIMUS}_custom",
]
