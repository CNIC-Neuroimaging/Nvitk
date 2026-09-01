"""
Registry of the models stage 2 has trained, read back from their provenance.

Description
-----------
Every stage 2 run leaves a marker at ``<results_root>/stage2_train/<dataset>/topbrain_stage2.json``
recording what it trained: the label set, the loss (hence the trainer class), the plans
identifier, the configuration and the folds. That is exactly the set of facts inference needs,
and **none of it can be reconstructed from flags** — the plans identifier embeds a spacing
``preprocess_like_nnssl`` only settles at run time, and the trainer family follows from the
pre-trained checkpoint's architecture.

So instead of asking an operator to remember that the binary model was trained with
``dice_ce`` while the multi-class one used ``dice_ce_skelrec``, this module lets them say
``--model binary`` and reads the rest.

Why the run directory is recomposed
-----------------------------------
The marker's ``results_dir`` is written by whichever process trained the model. On the cluster
that is a *container* path (``/nnunet/results/...``) which does not exist on a workstation. Only
the directory's **name** is portable, so :meth:`TrainedModel.run_dir` rebuilds the path against
whatever ``nnUNet_results`` the caller is using now.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.logger import Logger
from nvitk.pipes.topbrain.util.paths import DATASET_IDS, STAGE2_TRAIN_DIR

log = Logger()

#: Marker stage 2 writes for every dataset it trains.
MARKER_NAME: str = "topbrain_stage2.json"


@dataclass(frozen=True)
class TrainedModel:
    """One finished stage 2 run, as described by its provenance."""

    label_set: str
    dataset: str
    loss: str
    trainer: str
    architecture: str
    plans_identifier: str
    configuration: str
    folds: tuple[str, ...]
    num_output_channels: int
    run_name: str
    """``<trainer>__<plans>__<configuration>`` — the only portable part of ``results_dir``."""

    marker: Path
    created: str
    pretrained_from: str | None = None
    """Label set this model was itself initialised from, when it was a second fine-tuning."""

    @property
    def dataset_id(self) -> int:
        """nnU-Net dataset id, needed by the predictor."""
        return DATASET_IDS[self.label_set]

    def run_dir(self, nnunet_results: Path) -> Path:
        """The run directory under *nnunet_results* — recomposed, never taken from the marker."""
        return Path(nnunet_results) / self.dataset / self.run_name

    def available_folds(
        self, nnunet_results: Path, *, checkpoint: str = "checkpoint_final.pth"
    ) -> list[str]:
        """Folds whose training actually finished, in the order they were trained.

        A fold that was queued but never completed has no ``checkpoint_final.pth``, and asking
        the predictor for it fails several layers down with a missing-file error that says
        nothing about which fold.
        """
        run = self.run_dir(nnunet_results)
        return [f for f in self.folds if (run / f"fold_{f}" / checkpoint).is_file()]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "label_set": self.label_set, "dataset": self.dataset, "loss": self.loss,
            "trainer": self.trainer, "architecture": self.architecture,
            "plans_identifier": self.plans_identifier, "configuration": self.configuration,
            "folds": list(self.folds), "num_output_channels": self.num_output_channels,
            "run_name": self.run_name, "created": self.created,
            "pretrained_from": self.pretrained_from,
        }


def _from_marker(path: Path) -> TrainedModel | None:
    """Parse one marker, or ``None`` when it is unreadable or predates ``results_dir``."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Skipping unreadable stage 2 marker %s (%s).", path, exc)
        return None
    results_dir = record.get("results_dir")
    if not results_dir:
        log.warning("Stage 2 marker %s records no results_dir; skipping.", path)
        return None
    initialised_from = record.get("pretrained_weights")
    return TrainedModel(
        label_set=str(record.get("label_set", "")),
        dataset=str(record.get("dataset", "")),
        loss=str(record.get("loss", "")),
        trainer=str(record.get("trainer", "")),
        architecture=str(record.get("architecture", "")),
        plans_identifier=str(record.get("plans_identifier", "")),
        configuration=str(record.get("configuration", "3d_fullres")),
        folds=tuple(str(f) for f in record.get("folds", ())),
        num_output_channels=int(record.get("num_output_channels", 0)),
        run_name=Path(results_dir).name,
        marker=Path(path),
        created=str(record.get("created", "")),
        pretrained_from=str(initialised_from) if initialised_from else None,
    )


def discover_models(results_root: Path) -> list[TrainedModel]:
    """Every stage 2 run under *results_root*, newest first."""
    root = Path(results_root) / STAGE2_TRAIN_DIR
    if not root.is_dir():
        return []
    found = [m for m in (_from_marker(p) for p in sorted(root.glob(f"*/{MARKER_NAME}"))) if m]
    return sorted(found, key=lambda m: m.created, reverse=True)


def resolve_model(results_root: Path, selector: str | None) -> TrainedModel:
    """Find the model *selector* names.

    Accepted forms, in the order they are tried:

    - a **label set** (``ta36``, ``binary``, …) — the usual case;
    - a **dataset folder name** (``Dataset504_TopBrainVesselBinary``);
    - a **path** to a marker file or to the directory holding one.

    Raises
    ------
    FileNotFoundError
        Listing what *is* available, since the commonest cause is that the run has not finished
        or was trained under a different results root.
    """
    models = discover_models(results_root)
    if not selector:
        if len(models) == 1:
            return models[0]
        raise FileNotFoundError(
            "No --model given and "
            + (
                "no trained model was found under "
                f"{Path(results_root) / STAGE2_TRAIN_DIR}. Run stage 2 first."
                if not models else
                f"{len(models)} are available: {', '.join(m.label_set for m in models)}."
            )
        )

    text = str(selector).strip()
    for model in models:
        if text in (model.label_set, model.dataset):
            return model

    candidate = Path(text).expanduser()
    if candidate.is_dir():
        candidate = candidate / MARKER_NAME
    if candidate.is_file():
        model = _from_marker(candidate)
        if model is not None:
            return model

    raise FileNotFoundError(
        f"No trained model matches {selector!r}. Available: "
        f"{', '.join(f'{m.label_set} ({m.loss})' for m in models) or 'none'}. "
        f"Run with --list-models to see the details."
    )


def describe_models(results_root: Path, nnunet_results: Path | None = None) -> str:
    """Operator-facing listing of every trained model, for ``--list-models``."""
    models = discover_models(results_root)
    if not models:
        return (
            f"No trained models under {Path(results_root) / STAGE2_TRAIN_DIR}. "
            f"Run stage 2 (or stage 2a) first."
        )
    lines = [f"Trained models under {Path(results_root) / STAGE2_TRAIN_DIR}:", ""]
    for model in models:
        ready = (
            model.available_folds(nnunet_results) if nnunet_results is not None else None
        )
        lines.append(f"  --model {model.label_set}")
        lines.append(f"    dataset   : {model.dataset}  ({model.num_output_channels} classes)")
        lines.append(f"    loss      : {model.loss}   trainer: {model.trainer}")
        lines.append(f"    plans     : {model.plans_identifier}")
        lines.append(f"    trained   : {model.created}   folds: {', '.join(model.folds)}")
        if ready is not None:
            missing = [f for f in model.folds if f not in ready]
            lines.append(
                f"    usable    : fold(s) {', '.join(ready) or 'NONE'}"
                + (f"   (unfinished: {', '.join(missing)})" if missing else "")
            )
        if model.pretrained_from:
            lines.append(f"    seeded by : {model.pretrained_from}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "MARKER_NAME",
    "TrainedModel",
    "describe_models",
    "discover_models",
    "resolve_model",
]
