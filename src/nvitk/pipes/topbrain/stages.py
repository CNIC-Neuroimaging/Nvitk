"""ToPBrain pipeline stage identifiers, aliases, and parsing.

Six stages, in the order they run::

    stage0  data preparation      release (+ your cohorts) -> nnU-Net dataset; optional corpus
    stage1  pre-training          published checkpoint, or nnssl from scratch -> a bundle
    stage2  transfer training     fine-tune the bundle on the stage-0 dataset
    stage3  evaluation            the six challenge metrics on cross-validation predictions
    stage4  inference             predict on new cases with a selected model
    stage5  packaging             Grand Challenge submission container

:data:`DEFAULT_STAGES` covers preparation through training; evaluation, inference and packaging
are opt-in because they need a finished model.
"""

from __future__ import annotations

import click

STAGE_DATAPREP = "stage0"
STAGE_PRETRAIN = "stage1"
STAGE_BINARY = "stage2a"
STAGE_TRAIN = "stage2"
STAGE_EVALUATE = "stage3"
STAGE_INFER = "stage4"
STAGE_PACKAGE = "stage5"
STAGE_SELFTRAIN = "stage6"

STAGE_ALIASES: dict[str, str] = {
    "stage0": STAGE_DATAPREP, "stage0_dataprep": STAGE_DATAPREP,
    "dataprep": STAGE_DATAPREP, "data": STAGE_DATAPREP, "convert": STAGE_DATAPREP,
    "stage1": STAGE_PRETRAIN, "stage1_pretrain": STAGE_PRETRAIN,
    "pretrain": STAGE_PRETRAIN, "ssl": STAGE_PRETRAIN,
    "stage2a": STAGE_BINARY, "stage2a_binary": STAGE_BINARY,
    "binary": STAGE_BINARY, "binary_pretrain": STAGE_BINARY, "silver": STAGE_BINARY,
    "stage2": STAGE_TRAIN, "stage2_train": STAGE_TRAIN,
    "train": STAGE_TRAIN, "finetune": STAGE_TRAIN, "transfer": STAGE_TRAIN,
    "stage3": STAGE_EVALUATE, "stage3_evaluate": STAGE_EVALUATE,
    "evaluate": STAGE_EVALUATE, "eval": STAGE_EVALUATE, "validate": STAGE_EVALUATE,
    "stage4": STAGE_INFER, "stage4_infer": STAGE_INFER,
    "infer": STAGE_INFER, "predict": STAGE_INFER,
    "stage5": STAGE_PACKAGE, "stage5_package": STAGE_PACKAGE,
    "package": STAGE_PACKAGE, "docker": STAGE_PACKAGE,
    "stage6": STAGE_SELFTRAIN, "stage6_selftrain": STAGE_SELFTRAIN,
    "selftrain": STAGE_SELFTRAIN, "pseudo": STAGE_SELFTRAIN,
    "pseudolabel": STAGE_SELFTRAIN,
}

STAGES_ORDERED: tuple[str, ...] = (
    STAGE_DATAPREP, STAGE_PRETRAIN, STAGE_BINARY, STAGE_TRAIN, STAGE_EVALUATE, STAGE_INFER,
    STAGE_PACKAGE, STAGE_SELFTRAIN,
)

ALL_STAGES: tuple[str, ...] = STAGES_ORDERED

DEFAULT_STAGES: str = f"{STAGE_DATAPREP},{STAGE_PRETRAIN},{STAGE_TRAIN}"

STAGE_LABELS: dict[str, str] = {
    STAGE_DATAPREP: "data preparation",
    STAGE_PRETRAIN: "pre-training",
    STAGE_BINARY: "binary vessel fine-tuning (silver)",
    STAGE_TRAIN: "transfer training",
    STAGE_EVALUATE: "evaluation",
    STAGE_INFER: "inference",
    STAGE_PACKAGE: "packaging",
    STAGE_SELFTRAIN: "self-training (pseudo-labelling)",
}

#: Every stage runs once for the whole cohort: nnU-Net and nnssl own their own per-case
#: parallelism, so splitting a run across SGE array tasks would fight them rather than help.
COHORT_STAGES: frozenset[str] = frozenset(STAGES_ORDERED)


def parse_stages(spec: str) -> list[str]:
    """Parse a ``--stages`` comma list into canonical stage ids in pipeline order.

    Order in *spec* is ignored — the result is always sorted into :data:`STAGES_ORDERED`, so
    ``--stages stage2,stage0`` still prepares data before it trains.
    """
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
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
    "ALL_STAGES", "COHORT_STAGES", "DEFAULT_STAGES", "STAGES_ORDERED",
    "STAGE_ALIASES", "STAGE_DATAPREP", "STAGE_EVALUATE", "STAGE_INFER", "STAGE_LABELS",
    "STAGE_BINARY", "STAGE_PACKAGE", "STAGE_PRETRAIN", "STAGE_SELFTRAIN", "STAGE_TRAIN",
    "parse_stages",
]
