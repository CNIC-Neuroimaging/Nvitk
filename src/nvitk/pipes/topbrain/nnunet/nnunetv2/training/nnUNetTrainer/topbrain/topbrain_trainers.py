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

Sampling
--------
:data:`~nvitk.pipes.topbrain.util.sampling.SAMPLING_SPEC_ENV` selects between nnU-Net's own
patch sampling and the rare-class-aware variant, which oversamples both the *cases* carrying
rare classes and, inside them, the rare class itself. Off by default: it is a change worth
measuring against the default, not assuming.

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
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as nnunet_trainer_module
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import (
    PretrainedTrainer,
    PretrainedTrainer_Primus,
)

from nvitk.pipes.topbrain.util import sampling as sampling_util
from nvitk.pipes.topbrain.util.losses import (
    EPOCHS_ENV,
    LOSS_SPEC_ENV,
    SEGMENTATION_LOSSES,
    TRAINER_PREFIX,
    TRAINER_PREFIX_PRIMUS,
    LossContext,
    build_segmentation_loss,
)


class RareClassAwareDataLoader(nnUNetDataLoader):
    """nnU-Net's loader, with the foreground class drawn by rarity instead of uniformly.

    nnU-Net picks the class a forced-foreground patch centres on **uniformly among the classes
    present in that case**. That is fair within a case but blind across the cohort: a class in
    three volumes out of forty is still seen a thousand times less often than the ICA.

    Overriding :meth:`get_bbox` is all it takes — the base implementation already accepts an
    ``overwrite_class``, so this only has to decide which class that should be and hand the
    rest of the (fiddly, well-tested) bounding-box logic straight back to it.

    Parameters
    ----------
    class_weights
        Label → relative weight, mean-normalised to 1. Labels absent from a given case are
        never candidates, so the weights are renormalised over what that case actually has.
    """

    def __init__(self, *args, class_weights: dict = None, rng_seed: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = dict(class_weights or {})
        self._rng = np.random.default_rng(rng_seed)

    def _weighted_class(self, class_locations):
        """Pick a foreground class by weight, or ``None`` to leave the base class alone."""
        eligible = [
            key for key, locations in class_locations.items()
            if len(locations) > 0 and not isinstance(key, tuple)
        ]
        if len(eligible) < 2:
            return None  # nothing to choose between; let the base class handle it
        weights = np.array(
            [float(self.class_weights.get(int(key), 1.0)) for key in eligible], dtype=np.float64
        )
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        return eligible[int(self._rng.choice(len(eligible), p=weights / total))]

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None,
                 verbose=False):
        """Choose the class by weight, then defer to nnU-Net's bounding-box logic."""
        if force_fg and class_locations and overwrite_class is None:
            overwrite_class = self._weighted_class(class_locations)
        return super().get_bbox(
            data_shape, force_fg, class_locations, overwrite_class, verbose
        )


class _TopBrainSamplingMixin:
    """Optional rare-class-aware sampling, selected by an environment variable.

    Two levers, applied together because either alone is half a fix:

    - **case level** — ``sampling_probabilities`` on the training loader, so volumes carrying
      rare classes come up more often. nnU-Net already supports this and simply passes ``None``;
    - **class level** — :class:`RareClassAwareDataLoader`, so inside such a volume the patch
      centres on the rare class rather than on whichever of its ~28 classes came up.

    The foreground-oversampling fraction is raised at the same time (0.33 -> 0.5 by default):
    with 0.3 % foreground, the unforced patches are almost pure background and are what the
    rare classes are competing against for the batch.

    Validation is deliberately left on nnU-Net's default sampling. Weighting it too would make
    the validation loss incomparable with every other run and with the pseudo-dice the "best"
    checkpoint is selected on.
    """

    @staticmethod
    def _nnunet_raw() -> str:
        """``nnUNet_raw``, read when it is needed.

        This build's ``nnunetv2.paths`` captures the environment variables at import rather
        than lazily, so binding the name at module scope would freeze whatever was set when
        this module first loaded — ``None`` if it was imported before the environment was.
        """
        import nnunetv2.paths

        return nnunetv2.paths.nnUNet_raw or os.environ.get("nnUNet_raw", "")

    def _sampling_spec(self) -> dict:
        """The parsed sampling specification for this run."""
        if not hasattr(self, "_topbrain_sampling_spec"):
            self._topbrain_sampling_spec = sampling_util.parse_sampling_spec(
                os.environ.get(sampling_util.SAMPLING_SPEC_ENV)
            )
        return self._topbrain_sampling_spec

    def _rare_class_weights(self, train_identifiers):
        """``(class_weights, frequencies)`` for *train_identifiers*, or ``(None, None)``.

        Frequencies are counted over the **training** identifiers only: taking them over the
        whole dataset would let the validation half's class distribution steer training.
        """
        spec = self._sampling_spec()
        dataset_dir = join(self._nnunet_raw(), self.plans_manager.dataset_name)
        case_classes = sampling_util.read_case_classes(dataset_dir)
        if not case_classes:
            self.print_to_log_file(
                f"WARNING: rare-aware sampling requested but no "
                f"{sampling_util.CASE_CLASSES_FILE} under {dataset_dir}. Re-run stage 0 to "
                f"produce it. Falling back to nnU-Net's default sampling."
            )
            return None, None

        frequencies = sampling_util.class_frequencies(
            case_classes, identifiers=list(train_identifiers)
        )
        weights = sampling_util.class_weights(
            frequencies,
            temperature=float(spec["temperature"]),
            max_weight=float(spec["max_weight"]),
        )
        self.print_to_log_file(
            "Rare-class-aware sampling (temperature=%.2f, max_weight=%.1f, oversample=%.2f):\n%s"
            % (spec["temperature"], spec["max_weight"], spec["oversample_percent"],
               sampling_util.describe_weights(frequencies, weights))
        )
        return weights, frequencies

    def get_dataloaders(self):
        """nnU-Net's dataloaders, with the training one built rare-class-aware.

        Implemented by swapping the ``nnUNetDataLoader`` name that
        ``nnUNetTrainer.get_dataloaders`` resolves at call time, for the duration of that call.
        The alternative — calling ``super()`` and replacing the training loader afterwards —
        does not work: the base method ends by pulling a batch from each loader, which has
        already spawned a ``NonDetMultiThreadedAugmenter`` worker pool, and the discarded pool
        would be left running for the length of the training.

        Only the *training* loader is swapped; the factory identifies it by its dataset's
        identifiers. Validation keeps nnU-Net's sampling, so its loss and pseudo-dice stay
        comparable with every other run.
        """
        if self._sampling_spec()["mode"] != "rare_aware":
            return super().get_dataloaders()

        train_identifiers, _ = self.do_split()
        weights, _ = self._rare_class_weights(train_identifiers)
        if weights is None:
            return super().get_dataloaders()

        spec = self._sampling_spec()
        train_set = set(train_identifiers)
        dataset_dir = join(self._nnunet_raw(), self.plans_manager.dataset_name)
        case_classes = sampling_util.read_case_classes(dataset_dir)

        def factory(data, batch_size, patch_size, final_patch_size, label_manager, **kwargs):
            """Build the training loader rare-aware, the validation loader as usual."""
            if set(data.identifiers) != train_set:
                return nnUNetDataLoader(
                    data, batch_size, patch_size, final_patch_size, label_manager, **kwargs
                )
            # Probabilities must line up with the loader's own identifier order, so they are
            # computed here from the dataset it was actually handed.
            probabilities = sampling_util.case_weights(
                case_classes, weights,
                identifiers=list(data.identifiers),
                max_weight=float(spec["max_weight"]),
            )
            kwargs["oversample_foreground_percent"] = float(spec["oversample_percent"])
            kwargs["sampling_probabilities"] = np.asarray(probabilities, dtype=np.float64)
            return RareClassAwareDataLoader(
                data, batch_size, patch_size, final_patch_size, label_manager,
                class_weights=weights, **kwargs
            )

        original = nnunet_trainer_module.nnUNetDataLoader
        nnunet_trainer_module.nnUNetDataLoader = factory
        try:
            return super().get_dataloaders()
        finally:
            nnunet_trainer_module.nnUNetDataLoader = original


class _TopBrainLossMixin(_TopBrainSamplingMixin):
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
    "RareClassAwareDataLoader",
    *(f"{TRAINER_PREFIX}_{name}" for name in SEGMENTATION_LOSSES),
    f"{TRAINER_PREFIX}_custom",
    *(f"{TRAINER_PREFIX_PRIMUS}_{name}" for name in SEGMENTATION_LOSSES),
    f"{TRAINER_PREFIX_PRIMUS}_custom",
]
