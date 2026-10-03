"""
CViT stage 3 — cross-validation evaluation (Dice, clDice, β0 error, HD95 per label).

Description
-----------
nnU-Net writes each fold's held-out predictions to ``fold_N/validation/``; their union is a
proper cross-validated prediction set (every case predicted by a model that never saw it). Each
case is scored against ``labelsTr`` with :func:`nvitk.measure.segmentation_metrics.evaluate_case`
— HD95 in **millimetres** from the reference image's spacing.

Outputs (``<results_root>/stage3_evaluate/<dataset>/<run>/``)
--------------------------------------------------------------
``metrics.csv``    one row per case × label (fold, dice, cl_dice, b0_error, hd95)
``metrics.json``   cohort means, per-label means, per-fold means and the per-case aggregates

Absent-class convention: a label absent from both masks is skipped (not scored 1.0); present in
exactly one, it scores worst-case — see :mod:`nvitk.measure.segmentation_metrics`.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Sequence, TextIO

import click

from nvitk.core.backend import map_in_thread_pool
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.measure.segmentation_metrics import aggregate_cases, evaluate_case
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.stage2_train import resolve_run
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import paths_from_options, pop_roots, root_options
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.cvit.util.sge_stage import (
    container_layout,
    python_module_argv,
    quote_path,
    root_args,
    sge_backend_cli_args,
    submit_stage_job,
)

log = Logger()

METRIC_KEYS: tuple[str, ...] = ("dice", "cl_dice", "b0_error", "hd95")


def foreground_labels(dataset_json: dict[str, Any]) -> dict[int, str]:
    """``{value: name}`` of foreground labels.

    Raises
    ------
    ValueError
        For region-based datasets (tuple labels) — not supported by this evaluation.
    """
    out: dict[int, str] = {}
    for name, value in dataset_json["labels"].items():
        if isinstance(value, (list, tuple)):
            raise ValueError("Region-based labels are not supported by stage 3.")
        if int(value) != 0 and name != "ignore":
            out[int(value)] = name
    return out


def collect_predictions(run_dir: Path, folds: Sequence[str]) -> tuple[dict[str, tuple[str, Path]], list[str]]:
    """``{case: (fold, prediction_path)}`` from each fold's ``validation/``; plus missing folds."""
    found: dict[str, tuple[str, Path]] = {}
    missing: list[str] = []
    for fold in folds:
        validation = run_dir / f"fold_{fold}" / "validation"
        files = sorted(validation.glob("*.nii.gz")) if validation.is_dir() else []
        if not files:
            missing.append(str(fold))
        for f in files:
            found[f.name[: -len(".nii.gz")]] = (str(fold), f)
    return found, missing


def evaluate_folder(
    predictions: dict[str, tuple[str, Path]],
    reference_dir: Path,
    labels: dict[int, str],
    *,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Score every prediction against ``<reference_dir>/<case>.nii.gz``."""

    def _one(item: tuple[str, tuple[str, Path]]) -> dict[str, Any]:
        case, (fold, pred_path) = item
        ref_path = reference_dir / f"{case}.nii.gz"
        if not ref_path.is_file():
            raise FileNotFoundError(f"No reference for {case}: {ref_path}")
        ref = imread(ref_path, backend="cpu")
        pred = imread(pred_path, backend="cpu")
        if tuple(ref.data.shape) != tuple(pred.data.shape):
            raise ValueError(f"{case}: prediction shape {pred.data.shape} != reference {ref.data.shape}.")
        m = evaluate_case(ref.data, pred.data, case_id=case, labels=sorted(labels), spacing=ref.spacing)
        return {"case": case, "fold": fold, "metrics": m}

    return map_in_thread_pool(_one, sorted(predictions.items()), max_workers=int(workers))


def summarise(results: list[dict[str, Any]], labels: dict[int, str]) -> dict[str, Any]:
    """Cohort / per-label / per-fold means."""
    cases = [r["metrics"] for r in results]
    per_label: dict[str, dict[str, float]] = {}
    for value, name in labels.items():
        rows = [c.per_class[value] for c in cases if value in c.per_class]
        per_label[name] = {k: float(mean(r[k] for r in rows)) for k in METRIC_KEYS} if rows else {}
        per_label[name]["n_cases"] = len(rows)
    per_fold: dict[str, dict[str, float]] = {}
    for fold in sorted({r["fold"] for r in results}):
        per_fold[fold] = aggregate_cases([r["metrics"] for r in results if r["fold"] == fold])
    return {"cohort": aggregate_cases(cases), "per_label": per_label, "per_fold": per_fold,
            "num_cases": len(cases)}


def write_outputs(out_dir: Path, results: list[dict[str, Any]], labels: dict[int, str],
                  summary: dict[str, Any], extra: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "fold", "label", "label_name", *METRIC_KEYS])
        for r in results:
            for value, entry in sorted(r["metrics"].per_class.items()):
                w.writerow([r["case"], r["fold"], value, labels.get(value, ""),
                            *(f"{entry[k]:.6g}" for k in METRIC_KEYS)])
    payload = {**extra, "summary": summary,
               "cases": [{"fold": r["fold"], **r["metrics"].as_dict()} for r in results]}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_evaluate(
    *,
    paths: CViTPaths,
    dataset_id: int = pth.DEFAULT_DATASET_ID,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    run_name: str | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Score the cross-validation predictions of a stage-2 run; returns the summary."""
    dataset = pth.dataset_folder_name(dataset_id, dataset_name)
    marker = resolve_run(paths, dataset, run_name)
    run_dir = paths.nnunet_results / dataset / marker["run_name"]
    raw = paths.nnunet_raw / dataset
    labels = foreground_labels(json.loads((raw / "dataset.json").read_text(encoding="utf-8")))
    predictions, missing = collect_predictions(run_dir, marker["folds"])
    if not predictions:
        raise FileNotFoundError(f"No validation predictions under {run_dir}/fold_*/validation.")
    if missing:
        log.warning("stage3: fold(s) %s have no validation predictions; scoring a partial CV.", missing)
    log.info("stage3 | %s: %d case(s), %d label(s)", marker["run_name"], len(predictions), len(labels))
    results = evaluate_folder(predictions, raw / "labelsTr", labels, workers=workers)
    summary = summarise(results, labels)
    out_dir = paths.results_root / pth.STAGE3_EVAL_DIR / dataset / marker["run_name"]
    write_outputs(out_dir, results, labels, summary, {
        "stage": "stage3", "created": datetime.now().isoformat(timespec="seconds"),
        "run_name": marker["run_name"], "missing_folds": missing,
    })
    cohort = summary["cohort"]
    log.ok(f"stage3: Dice {cohort.get('class_avg_dice', float('nan')):.4f}  "
           f"clDice {cohort.get('class_avg_cl_dice', float('nan')):.4f}  "
           f"HD95 {cohort.get('class_avg_hd95', float('nan')):.2f} mm -> {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# SGE + CLI
# ---------------------------------------------------------------------------


def _worker_argv(**o: Any) -> list[str]:
    argv = [*python_module_argv("nvitk.pipes.cvit.stage3_evaluate"),
            *sge_backend_cli_args(o.get("backend", "cpu")), *root_args(container_layout()),
            "--dataset-id", str(o.get("dataset_id", pth.DEFAULT_DATASET_ID)),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME)),
            "--workers", str(o.get("workers", 4))]
    if o.get("run_name"):
        argv += ["--run-name", quote_path(o["run_name"])]
    return argv


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    return submit_stage_job("stage3", _worker_argv(**o), paths=paths, container=container,
                            src_dir=src_dir, backend=o.get("backend", "cpu"), request_gpu=False,
                            job_suffix=str(o.get("dataset_id", "")), hold_jid=hold_jid,
                            dry_run=dry_run, emit=emit)


@click.command("cvit-stage3-evaluate")
@config_dir_click_option()
@backend_click_option(default="cpu")
@root_options
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--run-name", type=str, default=None, help="Default: the latest stage-2 run.")
@click.option("--workers", type=int, default=4, show_default=True)
def main(backend: str, **options: Any) -> None:
    """Score cross-validation predictions of a trained CViT."""
    run_evaluate(paths=paths_from_options(pop_roots(options)), **options)


if __name__ == "__main__":
    main()
