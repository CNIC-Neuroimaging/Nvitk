"""ToPBrain stage 3: score predictions with the challenge's six metrics.

**Inputs**

- Cross-validated predictions — by default the *held-out* validation output nnU-Net writes per
  fold, gathered across folds (``--predictions-from cv``); or an explicit folder
  (``--predictions-from folder``, e.g. stage 4's ``postprocessed/``)
- The reference masks, i.e. stage 0's ``labelsTr``

Which predictions to score
--------------------------
``cv`` is the default and the only setting that yields an honest number. nnU-Net writes each
fold's predictions for the cases that fold **held out**, so gathering them across folds gives one
prediction per case, each from a model that never saw it. Pointing this stage at predictions made
over the whole training set instead — which is what inference on ``imagesTr`` produces — scores
the model partly on its own training data and inflates every metric, silently.

**Outputs**

- ``<results_root>/stage3_evaluate/<run>/metrics.json`` — per-case and cohort summary
- ``.../metrics_per_case.csv`` and ``.../metrics_per_class.csv``
- ``.../topbrain_stage3.json`` — provenance

Reported metrics
----------------
The six the challenge weights equally — class-average Dice, centerline Dice, connected-component
(β0) error, HD95, invalid-neighbour error, and side-road detection F1 — plus per-modality
breakdowns, because a single modality-agnostic model can hide a large CT/MRA gap behind a
respectable average.

These are our own implementations (:mod:`nvitk.measure.segmentation_metrics`), suitable for
ranking runs against each other. They are not the organisers' scorer: use
``CoWBenchmark/TopBrain_Eval_Metrics`` for numbers meant to be comparable with a leaderboard,
and note that the TA36 adjacency table is derived rather than published (see
:func:`nvitk.pipes.topbrain.labels.valid_neighbours`).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import click

from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.measure.segmentation_metrics import CaseMetrics, aggregate_cases, evaluate_case
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.util.paths import STAGE3_EVAL_DIR, parse_case_id
from nvitk.pipes.topbrain.util.sge_backend import sge_backend_cli_args
from nvitk.pipes.topbrain.util.sge_stage import (
    build_stage_command,
    container_layout,
    quote_path,
    submit_stage_job,
)

log = Logger()


def collect_cv_predictions(
    nnunet_results: Path,
    dataset_name: str,
    run_name: str,
    destination: Path,
    *,
    folds: Sequence[int | str] = (0, 1, 2, 3, 4),
) -> tuple[Path, list[str]]:
    """Gather each fold's held-out validation predictions into one directory.

    nnU-Net writes ``fold_N/validation/<case>.nii.gz`` for the cases fold *N* held out, so the
    union over folds is a proper cross-validated prediction set: one prediction per case, each
    from a model that never trained on it.

    Returns
    -------
    tuple
        ``(destination, missing_folds)`` — folds whose validation directory was absent, so a
        partial cross-validation is visible rather than silently scored as if complete.

    Raises
    ------
    FileNotFoundError
        If no fold produced any prediction.
    """
    import shutil

    run_dir = Path(nnunet_results) / dataset_name / run_name
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied = 0
    duplicates: list[str] = []
    for fold in folds:
        validation = run_dir / f"fold_{fold}" / "validation"
        if not validation.is_dir():
            missing.append(str(fold))
            continue
        for prediction in sorted(validation.glob("*.nii.gz")):
            target = destination / prediction.name
            if target.exists():
                # A case predicted by two folds means the splits overlap — that is a broken
                # cross-validation, not a mergeable duplicate.
                duplicates.append(prediction.name)
                continue
            shutil.copyfile(prediction, target)
            copied += 1

    if duplicates:
        raise ValueError(
            f"{len(duplicates)} case(s) appear in more than one fold's validation output, e.g. "
            f"{duplicates[:3]}. The folds overlap; the cross-validation is invalid."
        )
    if copied == 0:
        raise FileNotFoundError(
            f"No validation predictions under {run_dir}/fold_*/validation. Training must finish "
            f"(it writes them at the end of a run) before this stage can score a fold."
        )
    if missing:
        log.warning(
            "No validation output for fold(s) %s; scoring a partial cross-validation of %d case(s).",
            ", ".join(missing), copied,
        )
    log.info("Gathered %d cross-validated prediction(s) -> %s", copied, destination)
    return destination, missing


def _match_cases(prediction_dir: Path, reference_dir: Path) -> list[tuple[str, Path, Path]]:
    """Pair predictions with references by case id.

    Raises
    ------
    FileNotFoundError
        If nothing matches, naming both directories — the usual cause is pointing at the wrong
        stage-5 subdirectory.
    """
    pairs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for prediction in sorted(prediction_dir.glob("*.nii.gz")):
        case_id = prediction.name[: -len(".nii.gz")]
        reference = reference_dir / prediction.name
        if reference.is_file():
            pairs.append((case_id, reference, prediction))
        else:
            missing.append(case_id)
    if missing:
        log.warning("%d prediction(s) have no reference, e.g. %s", len(missing), missing[:3])
    if not pairs:
        raise FileNotFoundError(
            f"No prediction/reference pairs between {prediction_dir} and {reference_dir}."
        )
    return pairs


def _write_csvs(cases: Sequence[CaseMetrics], output_dir: Path, label_map: dict[int, str]) -> None:
    """Write the per-case and per-class CSV views of *cases*."""
    per_case = output_dir / "metrics_per_case.csv"
    keys = sorted({k for case in cases for k in case.aggregate})
    with per_case.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "modality", *keys])
        for case in cases:
            modality = parse_case_id(case.case_id)[0] if case.case_id.startswith("topcow_") else ""
            writer.writerow([case.case_id, modality, *(case.aggregate.get(k, "") for k in keys)])

    per_class = output_dir / "metrics_per_class.csv"
    metric_names = sorted({m for case in cases for entry in case.per_class.values() for m in entry})
    with per_class.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "label", "name", *metric_names])
        for case in cases:
            for value, entry in sorted(case.per_class.items()):
                writer.writerow([
                    case.case_id, value, label_map.get(value, "?"),
                    *(entry.get(m, "") for m in metric_names),
                ])
    log.info("Wrote %s and %s", per_case.name, per_class.name)


def run_evaluate(
    *,
    prediction_dir: Path,
    partial_folds: Sequence[str] = (),
    reference_dir: Path,
    results_root: Path,
    label_set: str = "ta36",
    run_name: str | None = None,
    iou_threshold: float | None = None,
    skip_neighbours: bool = False,
) -> Path:
    """Score every prediction and write the reports; returns the output directory."""
    label_map = lbl.label_map(label_set)
    labels = sorted(label_map)
    sideroad = lbl.sideroad_labels(label_set)
    neighbours = None if skip_neighbours else lbl.valid_neighbours(label_set)
    iou_threshold = lbl.IOU_THRESHOLD if iou_threshold is None else float(iou_threshold)

    pairs = _match_cases(Path(prediction_dir), Path(reference_dir))
    output_dir = Path(results_root) / STAGE3_EVAL_DIR / (run_name or Path(prediction_dir).name)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "topbrain stage3 | %d case(s) label_set=%s classes=%d sideroad=%d",
        len(pairs), label_set, len(labels), len(sideroad),
    )

    cases: list[CaseMetrics] = []
    for case_id, reference_path, prediction_path in pairs:
        reference = imread(reference_path)
        prediction = imread(prediction_path)
        metrics = evaluate_case(
            reference, prediction,
            case_id=case_id,
            labels=labels,
            spacing=reference.spacing,
            sideroad_labels=sideroad,
            valid_neighbours=neighbours,
            iou_threshold=iou_threshold,
        )
        cases.append(metrics)
        log.step(
            f"{case_id}: dice={metrics.aggregate.get('class_avg_dice', float('nan')):.4f} "
            f"clDice={metrics.aggregate.get('class_avg_cl_dice', float('nan')):.4f} "
            f"b0err={metrics.aggregate.get('class_avg_b0_error', float('nan')):.2f}"
        )

    summary = aggregate_cases(cases)

    # Per-modality: a modality-agnostic model can hide a large CT/MRA gap behind the average.
    by_modality: dict[str, dict[str, float]] = {}
    for modality in ("ct", "mr"):
        subset = [
            c for c in cases
            if c.case_id.startswith("topcow_") and parse_case_id(c.case_id)[0] == modality
        ]
        if subset:
            by_modality[modality] = aggregate_cases(subset)

    report = {
        "stage": "stage3",
        "created": datetime.now().isoformat(timespec="seconds"),
        "label_set": label_set,
        "num_cases": len(cases),
        "prediction_dir": str(prediction_dir),
        "reference_dir": str(reference_dir),
        "iou_threshold": iou_threshold,
        "invalid_neighbours_scored": neighbours is not None,
        "partial_cross_validation_missing_folds": list(partial_folds),
        "cohort": summary,
        "by_modality": by_modality,
        "per_case": [c.as_dict() for c in cases],
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csvs(cases, output_dir, label_map)
    (output_dir / "topbrain_stage3.json").write_text(
        json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2) + "\n",
        encoding="utf-8",
    )

    log.ok(f"stage3 complete: {len(cases)} case(s) -> {output_dir}")
    for key in sorted(summary):
        log.info("  %-28s %.4f", key, summary[key])
    for modality, values in by_modality.items():
        log.info(
            "  [%s] dice=%.4f clDice=%.4f b0err=%.2f hd95=%.2f",
            modality,
            values.get("class_avg_dice", float("nan")),
            values.get("class_avg_cl_dice", float("nan")),
            values.get("class_avg_b0_error", float("nan")),
            values.get("class_avg_hd95", float("nan")),
        )
    return output_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(
    *, label_set: str, prediction_subdir: str | None = None, run_name: str | None = None,
    folds_spec: str = "0,1,2,3,4",
    iou_threshold: float | None = None, skip_neighbours: bool = False, backend: str = "cpu",
) -> list[str]:
    """Worker argv for stage 3, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv
    from nvitk.pipes.topbrain.util.paths import DATASET_IDS, DATASET_SUFFIXES, STAGE4_INFER_DIR

    inside = container_layout()
    dataset = f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage3_evaluate"),
        *sge_backend_cli_args(backend),
        "--predictions-from", "cv",
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--folds", quote_path(folds_spec),
        "--reference-dir", quote_path(inside.nnunet_raw / dataset / "labelsTr"),
        "--results-root", quote_path(inside.results_root),
        "--label-set", label_set,
    ]
    # Omitted when unknown: the worker reads it from stage 2's provenance, which on the cluster
    # is written by the stage-2 job this one holds on.
    if prediction_subdir:
        argv.extend(["--run-name", quote_path(prediction_subdir)])
    if iou_threshold is not None:
        argv.extend(["--iou-threshold", str(float(iou_threshold))])
    if skip_neighbours:
        argv.append("--skip-neighbours")
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **options) -> str:
    """Host shell command for the stage 6 SGE task."""
    return build_stage_command(
        "stage3", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
        job_suffix=options.get("label_set", ""),
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **options,
) -> str:
    """Emit or submit the stage 6 SGE job."""
    return submit_stage_job(
        "stage3", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
        job_suffix=options.get("label_set", ""),
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage3-evaluate")
@click.option("--predictions-from", type=click.Choice(["cv", "folder"]), default="cv",
              show_default=True,
              help="'cv' gathers each fold's held-out validation output (the honest number); "
                   "'folder' scores --prediction-dir as given.")
@click.option("--prediction-dir", type=click.Path(path_type=Path), default=None,
              help="Required by --predictions-from folder.")
@click.option("--nnunet-results", type=click.Path(path_type=Path), default=None,
              help="Required by --predictions-from cv.")
@click.option("--run-name", "train_run_name", type=str, default=None,
              help="Training run directory (<trainer>__<plans>__3d_fullres). Required by cv.")
@click.option("--folds", type=str, default="0,1,2,3,4", show_default=True,
              help="Folds to gather in cv mode.")
@click.option("--reference-dir", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr"]), default="ta36",
              show_default=True)
@click.option("--run-name", type=str, default=None)
@click.option("--iou-threshold", type=float, default=None,
              help="Side-road detection IoU threshold (default: the challenge's 0.25).")
@click.option("--skip-neighbours", is_flag=True, default=False,
              help="Skip the invalid-neighbour metric.")
def main(
    predictions_from: str, prediction_dir: Path | None, nnunet_results: Path | None,
    train_run_name: str | None, folds: str, reference_dir: Path, results_root: Path,
    label_set: str, run_name: str | None, iou_threshold: float | None, skip_neighbours: bool,
) -> None:
    """CLI entry point: score predictions against reference masks."""
    from nvitk.pipes.topbrain.stage2_train import dataset_name_for, parse_folds

    Logger()
    partial: list[str] = []
    if predictions_from == "cv":
        if nnunet_results is None:
            raise click.UsageError("--predictions-from cv needs --nnunet-results.")
        if train_run_name is None:
            # The run directory name embeds the plans identifier, which stage 2 only settles at
            # run time; read it back rather than making the caller reconstruct it.
            from nvitk.pipes.topbrain.stage2_train import trained_run_name

            train_run_name = trained_run_name(results_root, label_set)
        prediction_dir, partial = collect_cv_predictions(
            nnunet_results, dataset_name_for(label_set), train_run_name,
            Path(results_root) / STAGE3_EVAL_DIR / (run_name or train_run_name) / "cv_predictions",
            folds=parse_folds(folds),
        )
    elif prediction_dir is None:
        raise click.UsageError("--predictions-from folder needs --prediction-dir.")

    run_evaluate(
        prediction_dir=prediction_dir, partial_folds=partial, reference_dir=reference_dir,
        results_root=results_root, label_set=label_set, run_name=run_name or train_run_name,
        iou_threshold=iou_threshold, skip_neighbours=skip_neighbours,
    )


__all__ = [
    "build_sge_command", "collect_cv_predictions", "main", "run_evaluate", "submit_sge",
]


if __name__ == "__main__":
    main()
