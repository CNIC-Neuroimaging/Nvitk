"""Selectable self-supervised trainers for stage 1.

nnssl ships ~270 trainer classes, but almost all are hyper-parameter variants of a handful of
methods (``SparkMAETrainer_BS8_1000ep``, ``VoCoTrainer_lr_1e3``, …). This registry names the
**method families** — the choices that actually differ — so ``--list-trainers`` is a menu rather
than a dump. Any nnssl class name remains valid for ``--ssl-trainer``; the registry documents,
it does not gate.

The constraint that decides the choice
--------------------------------------
**A trainer determines the architecture, and it must match the seed checkpoint.** Seeding a
ResEnc-L checkpoint (``--init-checkpoint-name resencl-*``) into an ``*Eva*`` trainer builds a
Primus network whose parameter names share nothing with the weights, and the seeding finds no
keys. There is deliberately no cross-family adapter.

Loss compatibility
------------------
nnssl's losses are called differently per family — ``SparkTrainer`` uses
``(prediction, groundtruth, mask)`` and the MAE trainers ``(model_output, target, mask)`` — so
:attr:`SSLTrainerSpec.loss_family` records which ``--ssl-loss`` values apply. Stage 1 checks the
actual signatures before training either way.

Deliberately torch-free, like the loss registry: listing the options must not import a deep
learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass

#: nnssl loss registry names usable with each trainer family.
LOSS_FAMILIES: dict[str, tuple[str, ...]] = {
    "mae": ("mse", "mse_masked", "l1", "ssim", "ms_ssim"),
    "spark": ("spark",),
    #: Methods whose loss is bound to their own pretext task and is not meaningfully swappable.
    "fixed": (),
}


@dataclass(frozen=True)
class SSLTrainerSpec:
    """One self-supervised method available to ``--ssl-trainer``."""

    name: str
    architecture: str
    method: str
    loss_family: str
    description: str
    unusable_reason: str | None = None

    @property
    def usable(self) -> bool:
        """Whether this trainer can actually run."""
        return self.unusable_reason is None

    @property
    def losses(self) -> tuple[str, ...]:
        """``--ssl-loss`` values compatible with this trainer."""
        return LOSS_FAMILIES.get(self.loss_family, ())


def _pair(method: str, cnn: str, transformer: str | None, loss_family: str, description: str
          ) -> list[SSLTrainerSpec]:
    """Build the ResEnc-L entry and, where nnssl provides one, its Primus counterpart.

    Both carry the same description: the pretext task is the method's, not the encoder's, and
    the listing already groups them under an architecture heading.
    """
    out = [SSLTrainerSpec(cnn, "ResEncL", method, loss_family, description)]
    if transformer:
        out.append(SSLTrainerSpec(transformer, "PrimusM", method, loss_family, description))
    return out


#: The method families, convolutional entry first within each pair.
SSL_TRAINERS: dict[str, SSLTrainerSpec] = {
    spec.name: spec
    for spec in [
        *_pair("SparK", "SparkMAETrainer", None, "spark",
               "Masked modelling inside the encoder (sparse convolutions)."),
        *_pair("MAE", "BaseMAETrainer", "BaseEvaMAETrainer", "mae",
               "Masked autoencoding: reconstruct masked blocks. Best-studied baseline."),
        *_pair("VolumeFusion", "VolumeFusionTrainer", "VolumeFusionEvaTrainer", "fixed",
               "Dense pseudo-segmentation of fused volumes; pretext closest to our task."),
        *_pair("VoCo", "VoCoTrainer", "VoCoEvaTrainer", "fixed",
               "Volume contrastive: global position and context, not local texture."),
        *_pair("ModelsGenesis", "ModelGenesisTrainer", "ModelGenesisEvaTrainer", "fixed",
               "Restoration from corruptions (shuffle, paint, deform)."),
        *_pair("SwinUNETR", "SwinUNETRTrainer", "SwinUNETREvaTrainer", "fixed",
               "Multi-task pretext: rotation, contrastive and reconstruction."),
        *_pair("SimCLR", "SimCLRTrainer", "SimCLREvaTrainer", "fixed",
               "Instance-level contrastive learning on crop pairs."),
        SSLTrainerSpec("SimMIMEvaTrainer", "PrimusM", "SimMIM", "mae",
                       "Simple masked image modelling (transformer only)."),
        SSLTrainerSpec("PCRLv2Trainer", "NoSkipResEncL", "PCRLv2", "fixed",
                       "Preservational contrastive representation learning.",
                       unusable_reason="raises NotImplementedError('Missing adaptation plan')"),
        SSLTrainerSpec("GVSLTrainer", "ResEncL", "GVSL", "fixed",
                       "Geometric visual self-supervised learning.",
                       unusable_reason="raises NotImplementedError('Missing adaptation plan')"),
    ]
}

#: Suggested when nothing is specified. Matches ``config.DEFAULT_SSL_TRAINER``.
DEFAULT_TRAINER: str = "SparkMAETrainer"


def trainer_names(*, architecture: str | None = None, usable_only: bool = True) -> list[str]:
    """Registered trainer names, optionally filtered by encoder family."""
    return [
        name for name, spec in SSL_TRAINERS.items()
        if (not usable_only or spec.usable)
        and (architecture is None or spec.architecture == architecture)
    ]


def get_trainer(name: str) -> SSLTrainerSpec | None:
    """Registry entry for *name*, or ``None`` for an unregistered variant.

    Returns ``None`` rather than raising: nnssl has ~270 valid class names and the registry
    only documents the method families, so an unknown name is usually a legitimate variant
    such as ``SparkMAETrainer_BS8_1000ep``.
    """
    return SSL_TRAINERS.get(name)


def architecture_for_trainer(name: str) -> str | None:
    """Encoder family a trainer builds, or ``None`` if it cannot be determined by name.

    Falls back to nnssl's own naming convention — an ``Eva`` in the class name means a Primus
    encoder — so hyper-parameter variants outside the registry still resolve.
    """
    spec = SSL_TRAINERS.get(name)
    if spec is not None:
        return spec.architecture
    if "Eva" in name or "Primus" in name:
        return "PrimusM"
    if name.endswith("Trainer") or "Trainer_" in name:
        return "ResEncL"
    return None


def describe_ssl_trainers() -> str:
    """Operator-facing listing for ``--list-trainers``."""
    lines = ["Self-supervised trainers (--ssl-trainer):", ""]
    for label, architecture in (("ResEnc-L (convolutional)", "ResEncL"),
                                ("Primus-M (transformer)", "PrimusM")):
        lines.append(f"  {label}:")
        for name in trainer_names(architecture=architecture):
            spec = SSL_TRAINERS[name]
            marker = "*" if name == DEFAULT_TRAINER else " "
            lines.append(f"  {marker} {name:24s} {spec.description}")
        lines.append("")

    unusable = [s for s in SSL_TRAINERS.values() if not s.usable]
    if unusable:
        lines.append("  Present in nnssl but unusable:")
        for spec in unusable:
            lines.append(f"    {spec.name:24s} {spec.unusable_reason}")
        lines.append("")

    lines += [
        "  * = default (config.DEFAULT_SSL_TRAINER).",
        "",
        "  The trainer sets the architecture, and it must match the seed checkpoint:",
        "  a resencl-* checkpoint needs a ResEnc-L trainer, a primusm-* one an Eva trainer.",
        "",
        "  Compatible --ssl-loss values:",
    ]
    for family, losses in LOSS_FAMILIES.items():
        members = [n for n, spec in SSL_TRAINERS.items()
                   if spec.loss_family == family and spec.usable]
        value = ", ".join(losses) if losses else "(each trainer's own; not swappable)"
        lines.append(f"    {value}")
        lines.append(f"      -> {', '.join(members)}")
    lines += [
        "",
        "  Any other nnssl class name also works — the ~270 hyper-parameter variants",
        "  (e.g. SparkMAETrainer_BS8_1000ep) are valid but not listed here.",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_TRAINER",
    "LOSS_FAMILIES",
    "SSL_TRAINERS",
    "SSLTrainerSpec",
    "architecture_for_trainer",
    "describe_ssl_trainers",
    "get_trainer",
    "trainer_names",
]
