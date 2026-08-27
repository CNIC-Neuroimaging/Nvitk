"""ToPBrain stage 6 (optional): self-training on the unlabeled cohort.

**Inputs**

- a trained run from stage 2 (resolved from its provenance)
- a folder of unlabeled volumes — the TOF cohort stage 0 assembled with ``--target corpus``,
  or any directory given with ``--input-dir``

**Outputs**

- ``<results_root>/stage6_selftrain/<run>/raw/`` — predictions on the unlabeled volumes
- ``.../accepted/{imagesTr,labelsTr}/`` — a cohort in ``--extra-train-only`` shape
- ``.../rejected.csv``, ``.../topbrain_stage6.json`` — every case with its verdict and reason

Why this rather than more masked pre-training
---------------------------------------------
Masked reconstruction asks the encoder to rebuild bulk brain tissue; vessels are ~0.3 % of the
voxels and contribute almost nothing to that objective. Self-training instead uses the unlabeled
cohort for the task itself: the model labels it, the labels that survive scrutiny become
training data, and the second round trains on 50 real cases plus however many pseudo-cases were
convincing. With 25 subjects that is usually the larger win, and the two are not exclusive — a
pre-trained encoder makes the first round's pseudo-labels better.

The whole method lives or dies on the filter. An accepted wrong label is worse than a discarded
right one, because it is indistinguishable from ground truth in the next round and the error
compounds. Every criterion here is therefore a **reason to reject**, and a case is accepted only
by passing all of them.

Filters
-------
``required classes``
    The main Circle of Willis branches must all be present. A case where the model lost an ICA
    is not a case to learn an ICA from.
``fragmentation``
    Per class, the number of connected components. A vessel that comes out in five pieces is a
    failed segmentation regardless of how confident the model was.
``invalid neighbours``
    Adjacencies the anatomy does not permit — the classic sign of two labels bleeding together.
``foreground volume``
    Total labelled volume must fall inside the range the *annotated* cohort spans. Catches both
    the collapsed prediction and the flood fill.
``fold agreement`` *(optional, ``--agreement-threshold``)*
    Predict once per fold and keep only cases the folds agree on. The strongest filter available
    and the most expensive: it costs one inference pass per fold. Disagreement between models
    trained on different subsets is a far better calibrated uncertainty signal than any single
    model's softmax, which on 3D segmentation is famously overconfident.

Reading the accepted cohort back in
-----------------------------------
Stage 6 prints the exact command. It uses ``--extra-train-only`` rather than ``--extra-train``:
a pseudo-labelled case must never land in a validation split, or the cross-validation starts
measuring agreement with the previous model instead of accuracy.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.measure.segmentation_metrics import count_components, invalid_neighbour_error
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.stage2_train import resolve_trained_run
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import nnunet_run
from nvitk.pipes.topbrain.util.nnunet_env import nnunet_env
from nvitk.pipes.topbrain.util.paths import (
    DATASET_IDS,
    DATASET_SUFFIXES,
    STAGE6_SELFTRAIN_DIR,
    TopBrainPaths,
)
from nvitk.pipes.topbrain.util.sge_backend import sge_backend_cli_args, torch_device_for_backend
from nvitk.pipes.topbrain.util.sge_stage import (
    build_stage_command,
    container_layout,
    quote_path,
    submit_stage_job,
)
from nvitk.segmentation.vessel_postprocess import postprocess_labelmap
from nvitk.segmentation.vessel_topology import repair_topology

setup(globals())

log = Logger()

#: Main Circle of Willis branches, by TA36 name. A prediction missing any of these has failed
#: at something far more basic than the class being checked, and nothing about it is trustworthy.
REQUIRED_CLASS_NAMES: tuple[str, ...] = (
    "BA", "R-ICA-C6-C7", "L-ICA-C6-C7", "R-M1", "L-M1", "R-A1A2", "L-A1A2",
)

#: Default ceiling on connected components per class. One is the anatomical truth; two allows a
#: vessel that genuinely leaves and re-enters the field of view.
DEFAULT_MAX_COMPONENTS: int = 2

#: Default ceiling on how many classes may exceed :data:`DEFAULT_MAX_COMPONENTS`.
DEFAULT_MAX_FRAGMENTED_CLASSES: int = 3

#: Default ceiling on impossible adjacencies. Not zero: the TA36 adjacency table is derived
#: rather than published, so a small count is as likely to be the table's fault as the model's.
DEFAULT_MAX_INVALID_NEIGHBOURS: int = 2

#: Accepted total-foreground volume, as a multiple of the annotated cohort's median.
DEFAULT_VOLUME_RANGE: tuple[float, float] = (0.5, 2.0)


@dataclass
class CaseVerdict:
    """One unlabeled case, its measurements, and why it was kept or dropped."""

    case_id: str
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    num_classes: int = 0
    missing_required: list[str] = field(default_factory=list)
    fragmented_classes: list[int] = field(default_factory=list)
    invalid_neighbours: float = 0.0
    volume_mm3: float = 0.0
    volume_ratio: float = float("nan")
    agreement: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "case_id": self.case_id, "accepted": self.accepted, "reasons": list(self.reasons),
            "num_classes": self.num_classes,
            "missing_required": list(self.missing_required),
            "fragmented_classes": list(self.fragmented_classes),
            "invalid_neighbours": self.invalid_neighbours,
            "volume_mm3": self.volume_mm3, "volume_ratio": self.volume_ratio,
            "agreement": self.agreement,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Reference statistics from the annotated cohort
# ──────────────────────────────────────────────────────────────────────────────


def reference_volume_mm3(labels_dir: Path, *, limit: int | None = None) -> float:
    """Median total foreground volume across the annotated masks, in mm³.

    The scale a pseudo-label is judged against comes from the real masks rather than from a
    constant, so it follows the cohort if the dataset changes.

    Raises
    ------
    FileNotFoundError
        If there are no reference masks — without them the volume filter has no meaning and
        silently skipping it would let a flood-filled prediction through.
    """
    masks = sorted(Path(labels_dir).glob("*.nii.gz"))
    if limit:
        masks = masks[: int(limit)]
    if not masks:
        raise FileNotFoundError(f"No reference masks under {labels_dir}.")
    volumes: list[float] = []
    for path in masks:
        mask = imread(path)
        voxel = 1.0
        for size in (mask.spacing or ()):
            voxel *= float(size)
        volumes.append(float(to_numpy((mask.data > 0).sum())) * voxel)
    volumes.sort()
    median = volumes[len(volumes) // 2]
    log.info(
        "Reference foreground volume: median %.0f mm3 over %d annotated mask(s).",
        median, len(volumes),
    )
    return median


# ──────────────────────────────────────────────────────────────────────────────
# Per-case scrutiny
# ──────────────────────────────────────────────────────────────────────────────


def assess_case(
    prediction_path: Path,
    *,
    label_set: str,
    reference_volume: float,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    max_fragmented_classes: int = DEFAULT_MAX_FRAGMENTED_CLASSES,
    max_invalid_neighbours: int = DEFAULT_MAX_INVALID_NEIGHBOURS,
    volume_range: Sequence[float] = DEFAULT_VOLUME_RANGE,
    required_names: Sequence[str] = REQUIRED_CLASS_NAMES,
    agreement: float = float("nan"),
    agreement_threshold: float | None = None,
) -> CaseVerdict:
    """Measure one pseudo-label and decide whether it is fit to train on.

    Every check is a veto. A case is accepted only when it fails none of them, because in
    self-training the cost of a wrong accept is asymmetric: it enters the next round
    indistinguishable from ground truth, and its error is what the model then learns.
    """
    label_map = lbl.label_map(label_set)
    by_name = {name: value for value, name in label_map.items()}
    verdict = CaseVerdict(case_id=prediction_path.name[: -len(".nii.gz")])

    image = imread(prediction_path)
    data = to_numpy(image.data).astype("int32", copy=False)
    voxel = 1.0
    for size in (image.spacing or ()):
        voxel *= float(size)

    present = {int(v) for v in to_numpy(np.unique(data)) if int(v) != 0}
    verdict.num_classes = len(present)

    # ---- 1. The main branches must be there ---------------------------------
    verdict.missing_required = [
        name for name in required_names
        if name in by_name and by_name[name] not in present
    ]
    if verdict.missing_required:
        verdict.reasons.append(f"missing required class(es): {', '.join(verdict.missing_required)}")

    # ---- 2. Fragmentation ----------------------------------------------------
    verdict.fragmented_classes = [
        value for value in sorted(present)
        if count_components(data == value) > int(max_components)
    ]
    if len(verdict.fragmented_classes) > int(max_fragmented_classes):
        verdict.reasons.append(
            f"{len(verdict.fragmented_classes)} class(es) split into more than "
            f"{max_components} component(s)"
        )

    # ---- 3. Impossible adjacencies -------------------------------------------
    verdict.invalid_neighbours = invalid_neighbour_error(
        data, valid_neighbours=lbl.valid_neighbours(label_set), labels=sorted(label_map)
    )
    if verdict.invalid_neighbours > float(max_invalid_neighbours):
        verdict.reasons.append(
            f"{verdict.invalid_neighbours:.0f} anatomically impossible adjacency(ies)"
        )

    # ---- 4. Plausible total volume -------------------------------------------
    verdict.volume_mm3 = float((data > 0).sum()) * voxel
    verdict.volume_ratio = (
        verdict.volume_mm3 / reference_volume if reference_volume else float("nan")
    )
    low, high = float(volume_range[0]), float(volume_range[1])
    if not (low <= verdict.volume_ratio <= high):
        verdict.reasons.append(
            f"foreground volume {verdict.volume_ratio:.2f}x the annotated median "
            f"(allowed {low}-{high}x)"
        )

    # ---- 5. Fold agreement, when it was measured -----------------------------
    verdict.agreement = float(agreement)
    if agreement_threshold is not None:
        if verdict.agreement != verdict.agreement:  # NaN
            verdict.reasons.append("fold agreement was not measured")
        elif verdict.agreement < float(agreement_threshold):
            verdict.reasons.append(
                f"fold agreement {verdict.agreement:.3f} below {float(agreement_threshold):.3f}"
            )

    verdict.accepted = not verdict.reasons
    return verdict


def fold_agreement(prediction_dirs: Sequence[Path], case_name: str) -> float:
    """Mean pairwise voxel agreement between the folds' predictions for one case.

    Measured over the **union of the foreground** rather than the whole volume: agreeing that
    99.7 % of a head is background is not information, and including it would push every case
    above any useful threshold.

    Returns ``NaN`` when fewer than two folds predicted the case, or when they all predicted
    nothing — neither is agreement.
    """
    maps = [
        to_numpy(imread(directory / case_name).data).astype("int32", copy=False)
        for directory in prediction_dirs
        if (directory / case_name).is_file()
    ]
    if len(maps) < 2:
        return float("nan")

    scores: list[float] = []
    for index, first in enumerate(maps):
        for second in maps[index + 1:]:
            union = (first > 0) | (second > 0)
            total = int(union.sum())
            if total == 0:
                continue
            scores.append(float((first[union] == second[union]).sum()) / total)
    return float(sum(scores) / len(scores)) if scores else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Stage entry point
# ──────────────────────────────────────────────────────────────────────────────


def _dataset_name(label_set: str) -> str:
    """nnU-Net dataset folder name for *label_set*."""
    return f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"


def run_selftrain(
    *,
    input_dir: Path,
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    nnunet_results: Path,
    results_root: Path,
    label_set: str = "ta36",
    loss: str | None = None,
    folds: Sequence[int | str] = (0, 1, 2, 3, 4),
    plans_identifier: str | None = None,
    configuration_name: str | None = None,
    checkpoint_name: str = "checkpoint_final.pth",
    modality: str = "mr",
    max_components: int = DEFAULT_MAX_COMPONENTS,
    max_fragmented_classes: int = DEFAULT_MAX_FRAGMENTED_CLASSES,
    max_invalid_neighbours: int = DEFAULT_MAX_INVALID_NEIGHBOURS,
    volume_range: Sequence[float] = DEFAULT_VOLUME_RANGE,
    agreement_threshold: float | None = None,
    repair_gaps_mm: float | None = None,
    min_volume_mm3: float | None = 5.0,
    max_accepted: int | None = None,
    device: str = "cuda",
    num_processes: int = 3,
    skip_prediction: bool = False,
) -> Path:
    """Pseudo-label the unlabeled cohort and emit the cases worth training on.

    Returns
    -------
    Path
        The ``accepted/`` directory, in ``--extra-train-only`` shape.
    """
    loss = loss or cfg.DEFAULT_LOSS
    resolved = resolve_trained_run(
        results_root, label_set, plans_identifier=plans_identifier,
        configuration_name=configuration_name,
    )
    plans_identifier = resolved["plans_identifier"]
    configuration_name = resolved["configuration"]
    trainer = loss_util.trainer_for_loss(loss, architecture=resolved["architecture"])
    dataset_name = _dataset_name(label_set)
    run_name = f"{trainer}__{plans_identifier}__{configuration_name}"

    base = Path(results_root) / STAGE6_SELFTRAIN_DIR / run_name
    raw_dir = base / "raw"
    accepted_images = base / "accepted" / "imagesTr"
    accepted_labels = base / "accepted" / "labelsTr"

    paths = TopBrainPaths(
        challenge_root=Path(input_dir),
        nnssl_raw=Path(results_root), nnssl_preprocessed=Path(results_root),
        nnssl_results=Path(results_root),
        nnunet_raw=Path(nnunet_raw), nnunet_preprocessed=Path(nnunet_preprocessed),
        nnunet_results=Path(nnunet_results), results_root=Path(results_root),
        model_root=Path(results_root), corpus_root=Path(input_dir),
    )
    env = nnunet_env(paths, num_processes=num_processes)
    env[loss_util.LOSS_SPEC_ENV] = loss_util.loss_spec_payload(loss, {})

    volumes = sorted(Path(input_dir).glob("*.nii.gz"))
    if not volumes:
        raise FileNotFoundError(f"No unlabeled volumes under {input_dir}.")
    log.info(
        "stage6 | %d unlabeled volume(s) | model=%s | agreement=%s",
        len(volumes), run_name,
        "off" if agreement_threshold is None else f">= {agreement_threshold}",
    )

    # ---- 1. Predict ---------------------------------------------------------
    # With an agreement threshold each fold predicts separately, so the folds can be compared;
    # otherwise one ensembled pass over all folds is both cheaper and slightly more accurate.
    fold_dirs: list[Path] = []
    if agreement_threshold is not None:
        for fold in folds:
            fold_dir = base / f"fold_{fold}"
            fold_dirs.append(fold_dir)
            if not skip_prediction:
                fold_dir.mkdir(parents=True, exist_ok=True)
                nnunet_run.predict(
                    Path(input_dir), fold_dir, env=env, dataset_id=DATASET_IDS[label_set],
                    configuration=configuration_name, trainer=trainer,
                    plans_identifier=plans_identifier, folds=(fold,), device=device,
                    checkpoint_name=checkpoint_name, num_processes=num_processes,
                )
        if not skip_prediction:
            raw_dir.mkdir(parents=True, exist_ok=True)
            nnunet_run.predict(
                Path(input_dir), raw_dir, env=env, dataset_id=DATASET_IDS[label_set],
                configuration=configuration_name, trainer=trainer,
                plans_identifier=plans_identifier, folds=folds, device=device,
                checkpoint_name=checkpoint_name, num_processes=num_processes,
            )
    elif not skip_prediction:
        raw_dir.mkdir(parents=True, exist_ok=True)
        nnunet_run.predict(
            Path(input_dir), raw_dir, env=env, dataset_id=DATASET_IDS[label_set],
            configuration=configuration_name, trainer=trainer,
            plans_identifier=plans_identifier, folds=folds, device=device,
            checkpoint_name=checkpoint_name, num_processes=num_processes,
        )

    # ---- 2. Clean up, then judge --------------------------------------------
    reference = reference_volume_mm3(Path(nnunet_raw) / dataset_name / "labelsTr")
    label_values = sorted(lbl.label_map(label_set))
    accepted_images.mkdir(parents=True, exist_ok=True)
    accepted_labels.mkdir(parents=True, exist_ok=True)

    verdicts: list[CaseVerdict] = []
    for prediction_path in sorted(raw_dir.glob("*.nii.gz")):
        # Repair before judging: the filters ask whether the anatomy is plausible, and it is
        # the repaired mask that would be trained on. Judging the unrepaired one would reject
        # cases the pipeline would have fixed anyway.
        cleaned = postprocess_labelmap(
            imread(prediction_path), labels=label_values,
            spacing=imread(prediction_path).spacing, min_volume_mm3=min_volume_mm3,
        )
        if repair_gaps_mm is not None:
            cleaned, _ = repair_topology(
                cleaned, labels=label_values, spacing=cleaned.spacing,
                affine=cleaned.affine, bridge_gaps_mm=repair_gaps_mm,
            )
        imsave(prediction_path, cleaned.astype(np.uint8))

        verdict = assess_case(
            prediction_path, label_set=label_set, reference_volume=reference,
            max_components=max_components, max_fragmented_classes=max_fragmented_classes,
            max_invalid_neighbours=max_invalid_neighbours, volume_range=volume_range,
            agreement=(
                fold_agreement(fold_dirs, prediction_path.name) if fold_dirs else float("nan")
            ),
            agreement_threshold=agreement_threshold,
        )
        verdicts.append(verdict)
        log.step(
            f"{verdict.case_id}: {'ACCEPT' if verdict.accepted else 'reject'} "
            f"classes={verdict.num_classes} vol={verdict.volume_ratio:.2f}x "
            f"inv={verdict.invalid_neighbours:.0f}"
            + (f" agree={verdict.agreement:.3f}" if fold_dirs else "")
            + ("" if verdict.accepted else f" | {'; '.join(verdict.reasons)}")
        )

    # ---- 3. Emit the accepted cohort ----------------------------------------
    # Best first, so --max-accepted keeps the most convincing rather than the alphabetically
    # first. Ranking is by agreement when it exists, and by anatomical cleanliness otherwise.
    ranked = sorted(
        (v for v in verdicts if v.accepted),
        key=lambda v: (
            -(v.agreement if v.agreement == v.agreement else 0.0),
            len(v.fragmented_classes), v.invalid_neighbours,
        ),
    )
    if max_accepted is not None:
        for extra in ranked[int(max_accepted):]:
            extra.accepted = False
            extra.reasons.append(f"beyond --max-accepted {int(max_accepted)}")
        ranked = ranked[: int(max_accepted)]

    for verdict in ranked:
        source_image = Path(input_dir) / f"{verdict.case_id}_0000.nii.gz"
        if not source_image.is_file():
            candidates = sorted(Path(input_dir).glob(f"{verdict.case_id}*.nii.gz"))
            if not candidates:
                verdict.accepted = False
                verdict.reasons.append("source volume not found")
                continue
            source_image = candidates[0]
        shutil.copyfile(source_image, accepted_images / f"{verdict.case_id}_0000.nii.gz")
        shutil.copyfile(raw_dir / f"{verdict.case_id}.nii.gz",
                        accepted_labels / f"{verdict.case_id}.nii.gz")

    accepted = [v for v in verdicts if v.accepted]
    _write_reports(base, verdicts)

    log.ok(
        f"stage6: {len(accepted)}/{len(verdicts)} case(s) accepted -> {base / 'accepted'}"
    )
    log.info(
        "  feed them back with:\n"
        "    nvitk-topbrain --stages stage0,stage2 --extra-train-only "
        "pseudo=%s:%s:%s",
        accepted_images, accepted_labels, modality,
    )
    log.info(
        "  (--extra-train-only, not --extra-train: a pseudo-label must never be validated "
        "against.)"
    )
    return base / "accepted"


def _write_reports(base: Path, verdicts: Sequence[CaseVerdict]) -> None:
    """Write the per-case verdict table and the stage provenance."""
    with (base / "verdicts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_id", "accepted", "num_classes", "volume_mm3", "volume_ratio",
            "invalid_neighbours", "num_fragmented_classes", "agreement", "reasons",
        ])
        for verdict in verdicts:
            writer.writerow([
                verdict.case_id, verdict.accepted, verdict.num_classes, verdict.volume_mm3,
                verdict.volume_ratio, verdict.invalid_neighbours,
                len(verdict.fragmented_classes), verdict.agreement, "; ".join(verdict.reasons),
            ])

    rejected = [v for v in verdicts if not v.accepted]
    tally: dict[str, int] = {}
    for verdict in rejected:
        for reason in verdict.reasons:
            key = reason.split("(")[0].split(":")[0].strip()
            tally[key] = tally.get(key, 0) + 1
    if tally:
        log.info("  rejection reasons: %s", tally)

    (base / "topbrain_stage6.json").write_text(
        json.dumps(
            {
                "stage": "stage6",
                "created": datetime.now().isoformat(timespec="seconds"),
                "num_cases": len(verdicts),
                "num_accepted": sum(1 for v in verdicts if v.accepted),
                "rejection_reasons": tally,
                "cases": [v.as_dict() for v in verdicts],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_MAX_COMPONENTS",
    "DEFAULT_MAX_FRAGMENTED_CLASSES",
    "DEFAULT_MAX_INVALID_NEIGHBOURS",
    "DEFAULT_VOLUME_RANGE",
    "REQUIRED_CLASS_NAMES",
    "CaseVerdict",
    "assess_case",
    "fold_agreement",
    "build_sge_command",
    "main",
    "reference_volume_mm3",
    "run_selftrain",
    "submit_sge",
]


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(
    *, label_set: str = "ta36", loss: str = "dice_ce",
    folds: Sequence[int | str] = (0, 1, 2, 3, 4),
    checkpoint_name: str = "checkpoint_final.pth", modality: str = "mr",
    max_components: int = DEFAULT_MAX_COMPONENTS,
    max_fragmented_classes: int = DEFAULT_MAX_FRAGMENTED_CLASSES,
    max_invalid_neighbours: int = DEFAULT_MAX_INVALID_NEIGHBOURS,
    volume_range: Sequence[float] = DEFAULT_VOLUME_RANGE,
    agreement_threshold: float | None = None,
    repair_gaps_mm: float | None = None,
    min_volume_mm3: float | None = 5.0,
    max_accepted: int | None = None,
    device: str = "cuda", num_processes: int = 3, skip_prediction: bool = False,
    backend: str = "gpu", plans_identifier: str | None = None,
    configuration_name: str | None = None, **_ignored: Any,
) -> list[str]:
    """Worker argv for stage 6, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage6_selftrain"),
        *sge_backend_cli_args(backend),
        # The corpus root is where stage 0 assembled the unlabeled volumes.
        "--input-dir", quote_path(inside.corpus_root),
        "--nnunet-raw", quote_path(inside.nnunet_raw),
        "--nnunet-preprocessed", quote_path(inside.nnunet_preprocessed),
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--results-root", quote_path(inside.results_root),
        "--label-set", label_set,
        "--loss", quote_path(loss),
        "--folds", quote_path(",".join(str(f) for f in folds)),
        "--checkpoint-name", checkpoint_name,
        "--modality", modality,
        "--max-components", str(int(max_components)),
        "--max-fragmented-classes", str(int(max_fragmented_classes)),
        "--max-invalid-neighbours", str(int(max_invalid_neighbours)),
        "--volume-range", str(float(volume_range[0])), str(float(volume_range[1])),
        "--device", device,
        "--num-processes", str(int(num_processes)),
    ]
    # Omitted when unknown: read from stage 2's provenance inside the container.
    if plans_identifier:
        argv.extend(["--plans-identifier", quote_path(plans_identifier)])
    if configuration_name:
        argv.extend(["--configuration", quote_path(configuration_name)])
    if agreement_threshold is not None:
        argv.extend(["--agreement-threshold", str(float(agreement_threshold))])
    if repair_gaps_mm is not None:
        argv.extend(["--repair-gaps-mm", str(float(repair_gaps_mm))])
    if min_volume_mm3 is None:
        argv.append("--no-postprocess")
    else:
        argv.extend(["--min-volume-mm3", str(float(min_volume_mm3))])
    if max_accepted is not None:
        argv.extend(["--max-accepted", str(int(max_accepted))])
    if skip_prediction:
        argv.append("--skip-prediction")
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **options) -> str:
    """Host shell command for the stage 6 SGE task."""
    return build_stage_command(
        "stage6", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "gpu"),
        request_gpu=options.get("device", "cuda") != "cpu",
        job_suffix=options.get("label_set", ""),
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **options,
) -> str:
    """Emit or submit the stage 6 SGE job."""
    return submit_stage_job(
        "stage6", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "gpu"),
        request_gpu=options.get("device", "cuda") != "cpu",
        job_suffix=options.get("label_set", ""),
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage6-selftrain")
@config_dir_click_option()
@backend_click_option(default="gpu")
@click.option("--input-dir", type=click.Path(path_type=Path), required=True,
              help="Folder of unlabeled volumes to pseudo-label.")
@click.option("--nnunet-raw", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-preprocessed", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-results", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr"]), default="ta36",
              show_default=True)
@click.option("--loss", type=str, default=None)
@click.option("--folds", type=str, default="0,1,2,3,4", show_default=True)
@click.option("--plans-identifier", type=str, default=None)
@click.option("--configuration", "configuration_name", type=str, default=None)
@click.option("--checkpoint-name", type=str, default="checkpoint_final.pth", show_default=True)
@click.option("--modality", type=click.Choice(["mr", "ct"]), default="mr", show_default=True,
              help="Modality of the unlabeled cohort; only used to write the follow-up command.")
@click.option("--max-components", type=int, default=DEFAULT_MAX_COMPONENTS, show_default=True,
              help="Connected components a class may have before it counts as fragmented.")
@click.option("--max-fragmented-classes", type=int, default=DEFAULT_MAX_FRAGMENTED_CLASSES,
              show_default=True, help="How many classes may be fragmented before rejection.")
@click.option("--max-invalid-neighbours", type=int, default=DEFAULT_MAX_INVALID_NEIGHBOURS,
              show_default=True, help="Impossible adjacencies tolerated per case.")
@click.option("--volume-range", type=float, nargs=2, default=DEFAULT_VOLUME_RANGE,
              show_default=True,
              help="Accepted total foreground volume, as a multiple of the annotated median.")
@click.option("--agreement-threshold", type=float, default=None,
              help="Require this mean pairwise agreement between the folds' predictions. The "
                   "strongest filter and the most expensive: one inference pass per fold.")
@click.option("--repair-gaps-mm", type=float, default=None,
              help="Bridge same-class gaps in the pseudo-labels before judging them.")
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True)
@click.option("--no-postprocess", is_flag=True, default=False)
@click.option("--max-accepted", type=int, default=None,
              help="Keep at most this many, best first. Useful to stop the pseudo-cohort from "
                   "outnumbering the 50 real cases.")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-processes", type=int, default=3, show_default=True)
@click.option("--skip-prediction", is_flag=True, default=False,
              help="Re-judge existing predictions without re-running inference.")
def main(
    input_dir: Path, nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path,
    results_root: Path, label_set: str, loss: str | None, folds: str,
    plans_identifier: str | None, configuration_name: str | None, checkpoint_name: str,
    modality: str, max_components: int, max_fragmented_classes: int,
    max_invalid_neighbours: int, volume_range: tuple[float, float],
    agreement_threshold: float | None, repair_gaps_mm: float | None,
    min_volume_mm3: float, no_postprocess: bool, max_accepted: int | None,
    device: str | None, num_processes: int, skip_prediction: bool, backend: str = "gpu",
) -> None:
    """CLI entry point: pseudo-label an unlabeled cohort and filter it."""
    from nvitk.pipes.topbrain.stage2_train import parse_folds

    Logger()
    run_selftrain(
        input_dir=input_dir, nnunet_raw=nnunet_raw, nnunet_preprocessed=nnunet_preprocessed,
        nnunet_results=nnunet_results, results_root=results_root, label_set=label_set,
        loss=loss, folds=parse_folds(folds), plans_identifier=plans_identifier,
        configuration_name=configuration_name, checkpoint_name=checkpoint_name,
        modality=modality, max_components=max_components,
        max_fragmented_classes=max_fragmented_classes,
        max_invalid_neighbours=max_invalid_neighbours, volume_range=volume_range,
        agreement_threshold=agreement_threshold, repair_gaps_mm=repair_gaps_mm,
        min_volume_mm3=None if no_postprocess else min_volume_mm3,
        max_accepted=max_accepted, device=device or torch_device_for_backend(backend),
        num_processes=num_processes, skip_prediction=skip_prediction,
    )


if __name__ == "__main__":
    main()
