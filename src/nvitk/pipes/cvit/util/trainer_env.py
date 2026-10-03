"""
Out-of-band settings for the CViT nnU-Net trainer.

Description
-----------
nnU-Net constructs trainers from a class *name* and fixed constructor arguments, so anything
else a run needs (epochs, learning rate, pretrained encoder, probe cadence, a custom loss) has to
reach the trainer some other way. As in the topbrain pipeline, it travels through environment
variables that the pipeline sets on the training subprocess and the trainer reads once in its
constructor.

Torch-free: imported by both the pipeline (to build the environment) and the trainer (to read
it), and the pipeline must stay importable without torch.

Architecture-level choices (tokenizer, skips, gates, warm-up schedule …) are **not** here — they
live in the plans file's ``arch_kwargs`` so a checkpoint always rebuilds the same network.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping

# ──────────────────────────────────────────────────────────────────────────────
# Variable names
# ──────────────────────────────────────────────────────────────────────────────

ENV_EPOCHS = "NVITK_CVIT_NUM_EPOCHS"
ENV_LR = "NVITK_CVIT_LR"
ENV_WEIGHT_DECAY = "NVITK_CVIT_WEIGHT_DECAY"
ENV_WARMUP = "NVITK_CVIT_WARMUP_EPOCHS"
ENV_PRETRAINED = "NVITK_CVIT_PRETRAINED"
ENV_LLRD = "NVITK_CVIT_LLRD"
ENV_PROBE_EVERY = "NVITK_CVIT_PROBE_EVERY"
ENV_NO_MIRROR = "NVITK_CVIT_NO_MIRROR"
ENV_LOSS_SPEC = "NVITK_CVIT_LOSS_SPEC"
ENV_ITERS = "NVITK_CVIT_ITERATIONS_PER_EPOCH"

#: Trainer class-name prefix; one class per registered loss (``nnUNetTrainerCViT_<loss>``).
TRAINER_PREFIX = "nnUNetTrainerCViT"


# ──────────────────────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainerSettings:
    """What the pipeline tells the trainer beyond the plans file.

    Parameters
    ----------
    num_epochs
        ``None`` keeps nnU-Net's 1000.
    lr, weight_decay
        AdamW base learning rate and decoupled weight decay.
    warmup_epochs
        Linear LR warm-up before poly decay; clamped by the trainer to a tenth of short runs.
    pretrained
        Checkpoint whose ``tokenizer.*`` / ``encoder.*`` weights initialise the network.
    llrd
        Layer-wise LR decay factor (``1.0`` = off). Only meaningful with ``pretrained``.
    probe_every
        Log attention statistics every N epochs (``0`` = never).
    no_mirror
        Disable mirroring in training and test-time augmentation (lateralised labels).
    loss_spec
        ``{"loss": name, "config": {...}}`` for the ``_custom`` trainer.
    iterations_per_epoch
        Override nnU-Net's 250 training iterations per epoch (smoke tests).
    """

    num_epochs: int | None = None
    lr: float = 3e-4
    weight_decay: float = 5e-2
    warmup_epochs: int = 50
    pretrained: str | None = None
    llrd: float = 1.0
    probe_every: int = 25
    no_mirror: bool = False
    loss_spec: dict[str, Any] | None = None
    iterations_per_epoch: int | None = None

    def to_env(self) -> dict[str, str]:
        """Environment-variable form (unset fields are omitted)."""
        env: dict[str, str] = {
            ENV_LR: repr(float(self.lr)),
            ENV_WEIGHT_DECAY: repr(float(self.weight_decay)),
            ENV_WARMUP: str(int(self.warmup_epochs)),
            ENV_LLRD: repr(float(self.llrd)),
            ENV_PROBE_EVERY: str(int(self.probe_every)),
            ENV_NO_MIRROR: "1" if self.no_mirror else "0",
        }
        if self.num_epochs is not None:
            env[ENV_EPOCHS] = str(int(self.num_epochs))
        if self.pretrained:
            env[ENV_PRETRAINED] = str(self.pretrained)
        if self.loss_spec is not None:
            env[ENV_LOSS_SPEC] = json.dumps(self.loss_spec, sort_keys=True)
        if self.iterations_per_epoch is not None:
            env[ENV_ITERS] = str(int(self.iterations_per_epoch))
        return env

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_settings(environ: Mapping[str, str] | None = None) -> TrainerSettings:
    """Parse :class:`TrainerSettings` from the environment.

    Raises
    ------
    ValueError
        On a malformed value, naming the variable — a typo must not silently fall back to a
        default and train something nobody asked for.
    """
    env = os.environ if environ is None else environ

    def _get(name: str, cast, default):
        raw = env.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name}={raw!r} is not a valid {cast.__name__}.") from None

    loss_spec = None
    if env.get(ENV_LOSS_SPEC):
        try:
            loss_spec = json.loads(env[ENV_LOSS_SPEC])
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ENV_LOSS_SPEC} is not valid JSON: {exc}") from None

    return TrainerSettings(
        num_epochs=_get(ENV_EPOCHS, int, None),
        lr=_get(ENV_LR, float, 3e-4),
        weight_decay=_get(ENV_WEIGHT_DECAY, float, 5e-2),
        warmup_epochs=_get(ENV_WARMUP, int, 50),
        pretrained=env.get(ENV_PRETRAINED) or None,
        llrd=_get(ENV_LLRD, float, 1.0),
        probe_every=_get(ENV_PROBE_EVERY, int, 25),
        no_mirror=str(env.get(ENV_NO_MIRROR, "0")).strip().lower() in ("1", "true", "yes"),
        loss_spec=loss_spec,
        iterations_per_epoch=_get(ENV_ITERS, int, None),
    )


def trainer_name(loss: str) -> str:
    """nnU-Net trainer class for a loss registry name (``_custom`` for a dotted path)."""
    return f"{TRAINER_PREFIX}_{'custom' if ':' in loss else loss}"


__all__ = [
    "ENV_EPOCHS",
    "ENV_ITERS",
    "ENV_LLRD",
    "ENV_LOSS_SPEC",
    "ENV_LR",
    "ENV_NO_MIRROR",
    "ENV_PRETRAINED",
    "ENV_PROBE_EVERY",
    "ENV_WARMUP",
    "ENV_WEIGHT_DECAY",
    "TRAINER_PREFIX",
    "TrainerSettings",
    "read_settings",
    "trainer_name",
]
