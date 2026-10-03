"""
CViT pipeline stage identifiers, aliases, and parsing.

Stages, in the order they run::

    stage0   data preparation    labelled data -> nnU-Net raw; optional nnssl corpus
    stage1   pre-training        SimMIM / MAE on the corpus -> encoder bundle
    stage2   training            CViT plans, preprocessing, per-fold training
    stage3   evaluation          Dice / HD95 / clDice on cross-validation predictions
    stage3b  attention probe     interventions + attention statistics on validation cases
    stage4   inference           predict new cases
    stage5   export              portable model bundle

:data:`DEFAULT_STAGES` is data preparation + training; everything else is opt-in.
"""

from __future__ import annotations

import click

STAGE_DATAPREP = "stage0"
STAGE_PRETRAIN = "stage1"
STAGE_TRAIN = "stage2"
STAGE_EVALUATE = "stage3"
STAGE_PROBE = "stage3b"
STAGE_INFER = "stage4"
STAGE_EXPORT = "stage5"

STAGE_ALIASES: dict[str, str] = {
    "stage0": STAGE_DATAPREP, "stage0_dataprep": STAGE_DATAPREP, "dataprep": STAGE_DATAPREP,
    "data": STAGE_DATAPREP, "convert": STAGE_DATAPREP,
    "stage1": STAGE_PRETRAIN, "stage1_pretrain": STAGE_PRETRAIN, "pretrain": STAGE_PRETRAIN,
    "ssl": STAGE_PRETRAIN,
    "stage2": STAGE_TRAIN, "stage2_train": STAGE_TRAIN, "train": STAGE_TRAIN,
    "finetune": STAGE_TRAIN,
    "stage3": STAGE_EVALUATE, "stage3_evaluate": STAGE_EVALUATE, "evaluate": STAGE_EVALUATE,
    "eval": STAGE_EVALUATE,
    "stage3b": STAGE_PROBE, "stage3b_probe": STAGE_PROBE, "probe": STAGE_PROBE,
    "attention": STAGE_PROBE,
    "stage4": STAGE_INFER, "stage4_infer": STAGE_INFER, "infer": STAGE_INFER,
    "predict": STAGE_INFER,
    "stage5": STAGE_EXPORT, "stage5_export": STAGE_EXPORT, "export": STAGE_EXPORT,
    "package": STAGE_EXPORT,
}

STAGES_ORDERED: tuple[str, ...] = (
    STAGE_DATAPREP, STAGE_PRETRAIN, STAGE_TRAIN, STAGE_EVALUATE, STAGE_PROBE, STAGE_INFER,
    STAGE_EXPORT,
)

DEFAULT_STAGES: str = f"{STAGE_DATAPREP},{STAGE_TRAIN}"

STAGE_LABELS: dict[str, str] = {
    STAGE_DATAPREP: "data preparation",
    STAGE_PRETRAIN: "self-supervised pre-training",
    STAGE_TRAIN: "training",
    STAGE_EVALUATE: "evaluation",
    STAGE_PROBE: "attention-usage probe",
    STAGE_INFER: "inference",
    STAGE_EXPORT: "export",
}


def parse_stages(spec: str) -> list[str]:
    """Parse a ``--stages`` comma list into canonical ids, always in pipeline order.

    Raises
    ------
    click.ClickException
        On an empty spec or an unknown stage, listing the valid names.
    """
    tokens = [t.strip().lower() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        raise click.ClickException("--stages cannot be empty.")
    canonical: set[str] = set()
    for token in tokens:
        key = token.replace("-", "_")
        if key not in STAGE_ALIASES:
            raise click.ClickException(
                f"Unknown stage {token!r}. Valid: {', '.join(sorted(set(STAGE_ALIASES)))}."
            )
        canonical.add(STAGE_ALIASES[key])
    return [s for s in STAGES_ORDERED if s in canonical]


__all__ = [
    "DEFAULT_STAGES",
    "STAGES_ORDERED",
    "STAGE_ALIASES",
    "STAGE_DATAPREP",
    "STAGE_EVALUATE",
    "STAGE_EXPORT",
    "STAGE_INFER",
    "STAGE_LABELS",
    "STAGE_PRETRAIN",
    "STAGE_PROBE",
    "STAGE_TRAIN",
    "parse_stages",
]
