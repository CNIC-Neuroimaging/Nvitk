"""
CViT stage 3b — attention-usage probe: does the trained model actually use its transformer?

Description
-----------
Hybrid CNN-transformer segmenters can route everything through their conv skips and leave the
transformer idle (Wald et al., 2025, *Primus*). This stage measures that on the cross-validation
held-out cases of a stage-2 run, by predicting each case under a set of **interventions**
(:mod:`nvitk.nn.cvit.probe`) and comparing the scores with the unmodified model:

=====================  ===========================================================================
``full``               reference
``attn_off``           attention residual branches removed (MLPs kept)
``attn_local@Rmm``     tokens attend only within R mm (physical, from the plan's spacing)
``attn_uniform``       attention weights replaced by a uniform average
``transformer_off``    the token grid bypasses the encoder
``tokens_off``         the decoder sees the conv skips only
``skips_off``          the decoder sees the tokens only (U-Net decoder only)
``attn_off@Lnn``       one layer at a time (``--layer-sweep``)
=====================  ===========================================================================

Indices (``usage.json``)
------------------------
Per case, on the class-averaged Dice (and per label):

``AR``  attention reliance ``(D_full − D_tokens_off) / D_full``
``SR``  skip reliance ``(D_full − D_skips_off) / D_full``
``TR``  transformer reliance ``(D_full − D_transformer_off) / D_full``
``LR``  long-range gain ``D_full − D_attn_local@R`` (one per radius)
``PR``  pattern reliance ``D_full − D_attn_uniform``

reported as median and inter-quartile range across cases, so one outlier cannot drive them.
High AR with low LR means attention is used, but only locally; high SR with AR ≈ 0 is the
"CNN with an idle transformer" failure mode.

Outputs (``<results_root>/stage3b_probe/<dataset>/<run>/``)
------------------------------------------------------------
``probe.csv``    case × intervention × label scores (Dice, clDice, β0 error, HD95 mm)
``usage.json``   indices above + passive attention statistics per fold
``layers.png``   ΔDice per layer (``--layer-sweep``) and mean attention distance per layer
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Sequence, TextIO

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.pipes._engines import nnunet_run
from nvitk.pipes._engines.env import nnunet_env
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.stage2_train import resolve_run
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import parse_float_list, parse_str_list, paths_from_options, pop_roots, root_options
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.cvit.util.sge_stage import (
    container_layout,
    python_module_argv,
    quote_path,
    root_args,
    sge_backend_cli_args,
    submit_stage_job,
    torch_device_for_backend,
)

log = Logger()

METRICS: tuple[str, ...] = ("dice", "cl_dice", "b0_error", "hd95")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    """Median and IQR (linear interpolation), ignoring NaNs."""
    v = sorted(x for x in values if x == x)
    if not v:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "n": 0}

    def q(p: float) -> float:
        pos = p * (len(v) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"median": float(median(v)), "q25": q(0.25), "q75": q(0.75), "n": len(v)}


def _rel_drop(full: float, other: float) -> float:
    return (full - other) / full if full and full == full and other == other else float("nan")


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reliance indices from probe rows (see module docstring)."""
    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        by_key[(r["case"], str(r["label"]), r["intervention"])] = r
    cases = sorted({r["case"] for r in rows})
    labels = sorted({str(r["label"]) for r in rows})
    interventions = sorted({r["intervention"] for r in rows})

    def dice(case: str, label: str, intervention: str) -> float:
        row = by_key.get((case, label, intervention))
        return float(row["dice"]) if row and row.get("dice") is not None else float("nan")

    def indices_for(label: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        pairs = {"AR": "tokens_off", "SR": "skips_off", "TR": "transformer_off"}
        for name, inter in pairs.items():
            if inter in interventions:
                out[name] = _quantiles([_rel_drop(dice(c, label, "full"), dice(c, label, inter)) for c in cases])
        if "attn_uniform" in interventions:
            out["PR"] = _quantiles([dice(c, label, "full") - dice(c, label, "attn_uniform") for c in cases])
        for inter in interventions:
            if inter.startswith("attn_local@"):
                out[f"LR@{inter.split('@')[1]}"] = _quantiles(
                    [dice(c, label, "full") - dice(c, label, inter) for c in cases])
        out["delta_dice"] = {
            inter: _quantiles([dice(c, label, "full") - dice(c, label, inter) for c in cases])
            for inter in interventions if inter != "full"
        }
        out["full_dice"] = _quantiles([dice(c, label, "full") for c in cases])
        return out

    return {
        "num_cases": len(cases),
        "interventions": interventions,
        "overall": indices_for("all"),
        "per_label": {lab: indices_for(lab) for lab in labels if lab != "all"},
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "fold", "intervention", "label", *METRICS])
        for r in rows:
            w.writerow([r["case"], r["fold"], r["intervention"], r["label"],
                        *(f"{r[m]:.6g}" if isinstance(r.get(m), (int, float)) else "" for m in METRICS)])


def plot_layers(summary: dict[str, Any], stats: dict[str, Any], path: Path) -> bool:
    """ΔDice per layer (layer sweep) and attention distance per layer; returns whether drawn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing; layers.png not written.")
        return False
    deltas = {k: v["median"] for k, v in summary["overall"]["delta_dice"].items() if k.startswith("attn_off@L")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    if deltas:
        keys = sorted(deltas)
        axes[0].bar(range(len(keys)), [deltas[k] for k in keys], color="#3b6fb6")
        axes[0].set_xticks(range(len(keys)), [k.split("@L")[1] for k in keys])
    else:
        axes[0].text(0.5, 0.5, "run with --layer-sweep", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_xlabel("transformer layer removed")
    axes[0].set_ylabel("median ΔDice (full − ablated)")
    axes[0].set_title("Per-layer attention reliance")
    for fold, st in sorted(stats.items()):
        layers = st.get("layers") or []
        axes[1].plot([l["layer"] for l in layers], [l["mean_distance_mm"] for l in layers], marker="o",
                     label=f"fold {fold}")
    axes[1].set_xlabel("transformer layer")
    axes[1].set_ylabel("mean attention distance (mm)")
    axes[1].set_title("Attention range")
    if stats:
        axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Library entry point
# ---------------------------------------------------------------------------


def run_probe(
    *,
    paths: CViTPaths,
    dataset_id: int = pth.DEFAULT_DATASET_ID,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    run_name: str | None = None,
    folds: Sequence[str] | None = None,
    interventions: Sequence[str] = cfg.DEFAULT_PROBE_INTERVENTIONS,
    local_radii_mm: Sequence[float] = cfg.DEFAULT_LOCAL_RADII_MM,
    layer_sweep: bool = False,
    max_cases: int = 0,
    checkpoint: str = "checkpoint_final.pth",
    no_mirror: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the probe on a stage-2 run's held-out cases; returns the usage summary."""
    dataset = pth.dataset_folder_name(dataset_id, dataset_name)
    marker = resolve_run(paths, dataset, run_name)
    run_dir = paths.nnunet_results / dataset / marker["run_name"]
    out_dir = paths.results_root / pth.STAGE3B_PROBE_DIR / dataset / marker["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = list(folds or marker["folds"])
    splits = paths.nnunet_preprocessed / dataset / "splits_final.json"
    if not splits.is_file():
        raise FileNotFoundError(f"{splits} not found; the probe uses each fold's held-out cases.")
    bad = sorted(set(interventions) - set(cfg.DEFAULT_PROBE_INTERVENTIONS))
    if bad:
        raise ValueError(f"Unknown intervention(s) {bad}; choose from {cfg.DEFAULT_PROBE_INTERVENTIONS}.")

    env = nnunet_env(paths.nnunet_raw, paths.nnunet_preprocessed, paths.nnunet_results)
    args = ["--run-dir", str(run_dir), "--raw-dir", str(paths.nnunet_raw / dataset),
            "--splits", str(splits), "--out-dir", str(out_dir / "worker"),
            "--folds", ",".join(map(str, folds)), "--interventions", ",".join(interventions),
            "--local-radii-mm", ",".join(f"{r:g}" for r in local_radii_mm),
            "--max-cases", str(int(max_cases)), "--checkpoint", checkpoint, "--device", device]
    if layer_sweep:
        args.append("--layer-sweep")
    if no_mirror:
        args.append("--no-mirror")
    log.info("stage3b | %s: folds %s, interventions %s", marker["run_name"], folds, list(interventions))
    nnunet_run.run_module("nvitk.pipes.cvit.probe_worker", args, env=env)

    rows = [json.loads(line) for line in (out_dir / "worker" / "rows.jsonl").read_text().splitlines() if line.strip()]
    stats = json.loads((out_dir / "worker" / "stats.json").read_text())
    if not rows:
        raise RuntimeError("The probe produced no rows (no held-out cases?).")
    write_csv(rows, out_dir / "probe.csv")
    summary = summarise(rows)
    usage = {"stage": "stage3b", "created": datetime.now().isoformat(timespec="seconds"),
             "run_name": marker["run_name"], "folds": folds, "cvit_config": marker.get("cvit_config"),
             "summary": summary, "attention_stats": stats}
    (out_dir / "usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    plot_layers(summary, stats, out_dir / "layers.png")

    ov = summary["overall"]
    fmt = lambda k: f"{ov[k]['median']:.3f}" if k in ov else "n/a"  # noqa: E731
    log.ok(f"stage3b: AR={fmt('AR')} SR={fmt('SR')} TR={fmt('TR')} PR={fmt('PR')} "
           + " ".join(f"{k}={ov[k]['median']:.3f}" for k in ov if k.startswith("LR@"))
           + f" -> {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# SGE + CLI
# ---------------------------------------------------------------------------


def _worker_argv(**o: Any) -> list[str]:
    argv = [*python_module_argv("nvitk.pipes.cvit.stage3b_probe"),
            *sge_backend_cli_args(o.get("backend", "gpu")), *root_args(container_layout()),
            "--dataset-id", str(o.get("dataset_id", pth.DEFAULT_DATASET_ID)),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME)),
            "--interventions", ",".join(o.get("interventions", cfg.DEFAULT_PROBE_INTERVENTIONS)),
            "--local-radius-mm", ",".join(f"{r:g}" for r in o.get("local_radii_mm", cfg.DEFAULT_LOCAL_RADII_MM)),
            "--max-cases", str(o.get("max_cases", 0)),
            "--device", torch_device_for_backend(o.get("backend", "gpu"), device=o.get("device"), remote=True)]
    if o.get("run_name"):
        argv += ["--run-name", quote_path(o["run_name"])]
    if o.get("folds"):
        argv += ["--folds", ",".join(str(f) for f in o["folds"])]
    for key, flag in (("layer_sweep", "--layer-sweep"), ("no_mirror", "--no-mirror")):
        if o.get(key):
            argv.append(flag)
    return argv


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    return submit_stage_job("stage3b", _worker_argv(**o), paths=paths, container=container,
                            src_dir=src_dir, backend=o.get("backend", "gpu"),
                            job_suffix=str(o.get("dataset_id", "")), hold_jid=hold_jid,
                            dry_run=dry_run, emit=emit)


@click.command("cvit-stage3b-probe")
@config_dir_click_option()
@backend_click_option(default="gpu")
@root_options
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--run-name", type=str, default=None, help="Default: the latest stage-2 run.")
@click.option("--folds", type=str, default=None, help="Default: the run's trained folds.")
@click.option("--interventions", type=str, default=",".join(cfg.DEFAULT_PROBE_INTERVENTIONS), show_default=True)
@click.option("--local-radius-mm", "local_radii", type=str,
              default=",".join(f"{r:g}" for r in cfg.DEFAULT_LOCAL_RADII_MM), show_default=True)
@click.option("--layer-sweep", is_flag=True, default=False, help="Also remove attention one layer at a time.")
@click.option("--max-cases", type=int, default=0, show_default=True, help="Per fold; 0 = all held-out cases.")
@click.option("--checkpoint", type=str, default="checkpoint_final.pth", show_default=True)
@click.option("--no-mirror", is_flag=True, default=False)
@click.option("--device", type=str, default=None)
def main(backend: str, folds: str | None, interventions: str, local_radii: str, device: str | None,
         **options: Any) -> None:
    """Measure how much a trained CViT relies on attention, long-range context and skips."""
    run_probe(paths=paths_from_options(pop_roots(options)), folds=parse_str_list(folds) or None,
              interventions=parse_str_list(interventions), local_radii_mm=parse_float_list(local_radii),
              device=torch_device_for_backend(backend, device=device), **options)


if __name__ == "__main__":
    main()
