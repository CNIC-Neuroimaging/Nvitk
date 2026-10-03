"""
CViT stage 2 — CViT plans, preprocessing and per-fold training on the in-tree nnU-Net.

Description
-----------
1. **Baseline plan.** nnU-Net fingerprints the dataset and a residual-encoder planner
   (``--baseline-planner``, default ``nnUNetPlannerResEncM``) chooses spacing, normalisation,
   patch size and an anisotropy-aware pooling schedule.
2. **CViT plans** (:func:`nvitk.pipes.cvit.util.plans.derive_cvit_plans`): the same
   preprocessing, with the architecture swapped for ``nvitk.nn.cvit.CViT``. Every variant
   (tokenizer, decoder, skip controls, size) gets its own plans identifier, hence its own results
   folder ``<trainer>__<plans>__<configuration>``.
3. **Preprocessing** against the CViT plans. All variants share the baseline
   ``data_identifier``, so it runs once per dataset/configuration.
4. **Training** each fold with ``nnUNetTrainerCViT_<loss>`` in a subprocess (process isolation:
   an OOM-killed fold cannot take the orchestrator down). Run-time settings (epochs, LR, warm-up,
   pretrained encoder, probe cadence) travel through
   :mod:`nvitk.pipes.cvit.util.trainer_env`.

Pre-trained encoders
--------------------
``--from-bundle`` points at a stage-1 bundle. Its tokenizer, transformer size and stem schedule
are adopted (they must match for the weights to load); decoder and skip controls stay free. The
trainer loads ``encoder.pth`` (positional embedding resampled to the new patch, input channels
adapted), optionally with layer-wise LR decay (``--llrd``).

Provenance
----------
``<results_root>/stage2_train/<dataset>/<run>/cvit_stage2.json`` records dataset, plans, trainer,
configuration, folds and the CViT config; ``latest.json`` beside it points at the most recent run,
which is what stages 3, 3b, 4 and 5 use by default.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.pipes._engines import nnunet_run
from nvitk.pipes._engines.env import nnunet_env
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util import plans as plans_util
from nvitk.pipes.cvit.util.cli import parse_int_list, paths_from_options, pop_roots, root_options
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.cvit.util.sge_stage import (
    container_layout,
    python_module_argv,
    quote_path,
    root_args,
    sge_backend_cli_args,
    submit_stage_job,
    to_container_path,
    torch_device_for_backend,
)
from nvitk.pipes.cvit.util.trainer_env import TRAINER_PREFIX, TrainerSettings, trainer_name
from nvitk.segmentation.loss_registry import SEGMENTATION_LOSSES, validate_loss_name

log = Logger()

STAGE2_MARKER = "cvit_stage2.json"
LATEST_POINTER = "latest.json"

#: Plans file written by each baseline planner.
BASELINE_PLANS_FILES: dict[str, str] = {
    "ExperimentPlanner": "nnUNetPlans.json",
    "nnUNetPlannerResEncM": "nnUNetResEncUNetMPlans.json",
    "nnUNetPlannerResEncL": "nnUNetResEncUNetLPlans.json",
    "nnUNetPlannerResEncXL": "nnUNetResEncUNetXLPlans.json",
}

#: Fields adopted from a pre-training bundle (they decide parameter shapes / names).
BUNDLE_ARCH_FIELDS: tuple[str, ...] = (
    "tokenizer", "stem_channels", "stem_blocks", "stem_strides", "embed_dim", "depth",
    "num_heads", "mlp_ratio", "use_rope", "use_abs_pos_embed", "num_registers",
    "layer_scale_init", "preset",
)

# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------


def run_name_for(trainer: str, plans_identifier: str, configuration: str) -> str:
    """nnU-Net's results folder name."""
    return f"{trainer}__{plans_identifier}__{configuration}"


def stage2_dir(paths: CViTPaths, dataset: str) -> Path:
    return paths.results_root / pth.STAGE2_TRAIN_DIR / dataset


def resolve_run(paths: CViTPaths, dataset: str, run_name: str | None = None) -> dict[str, Any]:
    """The stage-2 marker of *run_name* (default: the latest run of *dataset*).

    Raises
    ------
    FileNotFoundError
        If no stage-2 run is recorded — later stages need a trained model.
    """
    base = stage2_dir(paths, dataset)
    if run_name is None:
        pointer = base / LATEST_POINTER
        if not pointer.is_file():
            raise FileNotFoundError(f"No trained CViT run recorded under {base}. Run stage 2 first.")
        run_name = json.loads(pointer.read_text(encoding="utf-8"))["run_name"]
    marker = base / run_name / STAGE2_MARKER
    if not marker.is_file():
        raise FileNotFoundError(f"No stage-2 marker {marker}.")
    return json.loads(marker.read_text(encoding="utf-8"))


def _preprocessed_complete(paths: CViTPaths, dataset: str, data_identifier: str, n_cases: int) -> bool:
    folder = paths.nnunet_preprocessed / dataset / data_identifier
    if not folder.is_dir():
        return False
    found = {p.name.split(".")[0] for p in folder.iterdir() if p.suffix in (".b2nd", ".npz", ".pkl")}
    return len(found) >= n_cases


# ---------------------------------------------------------------------------
# Library entry point
# ---------------------------------------------------------------------------


def run_train(
    *,
    paths: CViTPaths,
    dataset_id: int = pth.DEFAULT_DATASET_ID,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    configuration: str = cfg.DEFAULT_CONFIGURATION,
    baseline_planner: str = cfg.DEFAULT_BASELINE_PLANNER,
    arch: str | None = cfg.DEFAULT_ARCH,
    arch_overrides: dict[str, Any] | None = None,
    token_stride: int = 8,
    patch_size: Sequence[int] | None = None,
    batch_size: int | None = None,
    from_bundle: Path | None = None,
    llrd: float = 1.0,
    loss: str = cfg.DEFAULT_LOSS,
    loss_config: dict[str, Any] | None = None,
    folds: Sequence[int | str] = (0,),
    num_epochs: int | None = None,
    iterations_per_epoch: int | None = None,
    lr: float = cfg.DEFAULT_LR,
    warmup_epochs: int = cfg.DEFAULT_WARMUP_EPOCHS,
    probe_every: int = cfg.DEFAULT_PROBE_EVERY,
    no_mirror: bool = False,
    tag: str | None = None,
    device: str = "cuda",
    num_processes: int = 8,
    continue_training: bool = False,
    plan_only: bool = False,
    skip_planning: bool = False,
    skip_preprocessing: bool = False,
) -> dict[str, Any]:
    """Plan, preprocess and train a CViT on the stage-0 dataset; returns the stage-2 marker.

    Parameters
    ----------
    arch, arch_overrides
        Preset (``CViTS/B/M/L``) and :class:`~nvitk.nn.cvit.CViTConfig` fields (``tokenizer``,
        ``decoder``, ``skips``, ``skip_drop_prob``, ``skip_schedule``, ``skip_gate`` …).
    token_stride
        Power-of-two token stride; per-axis strides follow the baseline pooling schedule.
    from_bundle
        Stage-1 bundle whose encoder initialises the network (architecture adopted).
    plan_only
        Stop after planning + preprocessing (the SGE ``--parallel-folds`` preparation job).
    """
    validate_loss_name(loss, registry=SEGMENTATION_LOSSES)
    dataset = pth.dataset_folder_name(dataset_id, dataset_name)
    raw_dir = paths.nnunet_raw / dataset
    if not (raw_dir / "dataset.json").is_file():
        raise FileNotFoundError(f"{raw_dir}/dataset.json not found. Run stage 0 first.")
    n_cases = int(json.loads((raw_dir / "dataset.json").read_text())["numTraining"])
    pre_dir = paths.nnunet_preprocessed / dataset

    # ---- 0. Bundle → adopted architecture ----------------------------------------------
    overrides = dict(arch_overrides or {})
    bundle_info = None
    if from_bundle is not None:
        from nvitk.pipes.cvit.stage1_pretrain import read_bundle

        bundle_info = read_bundle(from_bundle)
        adopted = {k: v for k, v in bundle_info["cvit_config"].items() if k in BUNDLE_ARCH_FIELDS}
        clashes = {k for k in adopted if k in overrides and overrides[k] != adopted[k] and k != "preset"}
        if clashes:
            raise ValueError(f"--from-bundle fixes {sorted(clashes)}; drop those overrides.")
        arch = adopted.pop("preset", None)
        overrides.update(adopted)
        log.info("stage2: adopting encoder architecture from %s (%s)", bundle_info["dir"],
                 arch or f"E{adopted.get('embed_dim')}")

    settings = TrainerSettings(
        num_epochs=num_epochs, lr=lr, warmup_epochs=warmup_epochs,
        pretrained=str(bundle_info["encoder"]) if bundle_info else None, llrd=llrd,
        probe_every=probe_every, no_mirror=no_mirror,
        loss_spec={"loss": loss, "config": dict(loss_config or {})} if (":" in loss or loss_config) else None,
        iterations_per_epoch=iterations_per_epoch,
    )
    # A dotted-path loss or any loss kwargs go through the _custom trainer (nnU-Net passes no
    # constructor arguments, so the spec travels in NVITK_CVIT_LOSS_SPEC).
    trainer = f"{TRAINER_PREFIX}_custom" if loss_config else trainer_name(loss)
    env = nnunet_env(paths.nnunet_raw, paths.nnunet_preprocessed, paths.nnunet_results,
                     num_processes=num_processes, extra=settings.to_env())

    # ---- 1. Baseline plan ----------------------------------------------------------------
    baseline_file = BASELINE_PLANS_FILES.get(baseline_planner)
    if baseline_file is None:
        raise ValueError(f"Unknown baseline planner {baseline_planner!r}; choose from {sorted(BASELINE_PLANS_FILES)}.")
    if not skip_planning and not (pre_dir / baseline_file).is_file():
        log.info("stage2: fingerprint + baseline plan (%s)", baseline_planner)
        nnunet_run.plan_baseline(dataset_id, env=env, planner=baseline_planner, num_processes=num_processes)
    if not (pre_dir / baseline_file).is_file():
        raise FileNotFoundError(f"Baseline plans {pre_dir / baseline_file} missing.")
    baseline = json.loads((pre_dir / baseline_file).read_text(encoding="utf-8"))

    # ---- 2. CViT plans ---------------------------------------------------------------------
    # A bundle's stride schedule wins over the baseline's: the linear tokenizer's kernel shape
    # depends on it, and the encoder was pre-trained at that token granularity.
    stem = overrides.pop("stem_strides", None)
    plans, cvit_cfg = plans_util.derive_cvit_plans(
        baseline, configuration=configuration, arch=arch, overrides=overrides,
        token_stride=token_stride, patch_size=patch_size, batch_size=batch_size, tag=tag,
        stem_strides=stem,
    )
    plans_path = pre_dir / f"{plans['plans_name']}.json"
    if not (skip_planning and plans_path.is_file()):
        # Fold jobs of a --parallel-folds run skip this, so they never rewrite a plans file a
        # neighbour is reading.
        plans_util.write_plans(plans, pre_dir)
    plans_id = plans["plans_name"]
    conf = plans["configurations"][configuration]
    log.info("stage2: %s | patch %s, token grid %s (%d tokens), batch %d", plans_id,
             conf["patch_size"], list(cvit_cfg.grid_shape), plans["cvit"]["num_tokens"], conf["batch_size"])

    # ---- 3. Preprocess (shared by every variant of this baseline) ----------------------
    if not skip_preprocessing:
        if _preprocessed_complete(paths, dataset, conf["data_identifier"], n_cases):
            log.info("stage2: preprocessed data %s already complete; reusing it.", conf["data_identifier"])
        else:
            nnunet_run.preprocess_with_plans(dataset_id, env=env, plans_identifier=plans_id,
                                             configuration=configuration, num_processes=num_processes)
        nnunet_run.ensure_gt_segmentations(dataset_id, env=env)

    run_name = run_name_for(trainer, plans_id, configuration)
    marker_dir = stage2_dir(paths, dataset) / run_name
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "stage": "stage2", "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset, "dataset_id": int(dataset_id), "configuration": configuration,
        "plans": plans_id, "plans_path": str(plans_path), "trainer": trainer, "run_name": run_name,
        "results_dir": str(paths.nnunet_results / dataset / run_name),
        "folds": [str(f) for f in folds], "loss": loss, "loss_config": loss_config or {},
        "cvit_config": cvit_cfg.to_dict(), "baseline_planner": baseline_planner,
        "trainer_settings": settings.as_dict(),
        "bundle": str(bundle_info["dir"]) if bundle_info else None,
    }
    if plan_only:
        log.ok(f"stage2: planned + preprocessed ({plans_id}); training left to the fold jobs.")
        (marker_dir / STAGE2_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        return marker

    # ---- 4. Train each fold ------------------------------------------------------------
    for fold in folds:
        log.info("stage2: training fold %s (%s)", fold, run_name)
        nnunet_run.train(dataset_id, configuration, fold, env=env, trainer=trainer,
                         plans_identifier=plans_id, device=device,
                         continue_training=continue_training)
    (marker_dir / STAGE2_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    (stage2_dir(paths, dataset) / LATEST_POINTER).write_text(
        json.dumps({"run_name": run_name}, indent=2) + "\n", encoding="utf-8")
    log.ok(f"stage2: trained folds {list(folds)} -> {marker['results_dir']}")
    return marker


# ---------------------------------------------------------------------------
# SGE
# ---------------------------------------------------------------------------


def _worker_argv(paths: CViTPaths, **o: Any) -> list[str]:
    inside = container_layout()
    argv = [*python_module_argv("nvitk.pipes.cvit.stage2_train"),
            *sge_backend_cli_args(o.get("backend", "gpu")), *root_args(inside),
            "--dataset-id", str(o.get("dataset_id", pth.DEFAULT_DATASET_ID)),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME)),
            "--configuration", o.get("configuration", cfg.DEFAULT_CONFIGURATION),
            "--baseline-planner", o.get("baseline_planner", cfg.DEFAULT_BASELINE_PLANNER),
            "--token-stride", str(o.get("token_stride", 8)),
            "--loss", quote_path(o.get("loss", cfg.DEFAULT_LOSS)),
            "--folds", ",".join(str(f) for f in o.get("folds", (0,))),
            "--lr", repr(float(o.get("lr", cfg.DEFAULT_LR))),
            "--warmup-epochs", str(o.get("warmup_epochs", cfg.DEFAULT_WARMUP_EPOCHS)),
            "--probe-every", str(o.get("probe_every", cfg.DEFAULT_PROBE_EVERY)),
            "--llrd", repr(float(o.get("llrd", 1.0))),
            "--device", torch_device_for_backend(o.get("backend", "gpu"), device=o.get("device"), remote=True),
            "--num-processes", str(o.get("num_processes", 8))]
    if o.get("arch"):
        argv += ["--arch", o["arch"]]
    if o.get("arch_overrides"):
        argv += ["--arch-json", quote_path(json.dumps(o["arch_overrides"]))]
    if o.get("loss_config"):
        argv += ["--loss-config", quote_path(json.dumps(o["loss_config"]))]
    if o.get("patch_size"):
        argv += ["--patch-size", ",".join(str(p) for p in o["patch_size"])]
    if o.get("from_bundle"):
        argv += ["--from-bundle", quote_path(to_container_path(paths, o["from_bundle"]) or o["from_bundle"])]
    for key, flag in (("batch_size", "--batch-size"), ("num_epochs", "--epochs"),
                      ("iterations_per_epoch", "--iterations-per-epoch"), ("tag", "--tag")):
        if o.get(key) is not None:
            argv += [flag, quote_path(str(o[key]))]
    for key, flag in (("no_mirror", "--no-mirror"), ("continue_training", "--continue-training"),
                      ("plan_only", "--plan-only"), ("skip_planning", "--skip-planning"),
                      ("skip_preprocessing", "--skip-preprocessing")):
        if o.get(key):
            argv.append(flag)
    return argv


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    """Emit or submit one stage-2 job (planning only, all folds, or the folds given)."""
    folds = "-".join(str(f) for f in o.get("folds", (0,)))
    suffix = "prep" if o.get("plan_only") else f"f{folds}"
    bundle = [Path(o["from_bundle"])] if o.get("from_bundle") and to_container_path(paths, o["from_bundle"]) is None else []
    return submit_stage_job(
        "stage2", _worker_argv(paths, **o), paths=paths, container=container, src_dir=src_dir,
        backend=o.get("backend", "gpu"), request_gpu=not o.get("plan_only"),
        pe_smp=o.get("num_processes"), job_suffix=suffix, data_paths=bundle,
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("cvit-stage2-train")
@config_dir_click_option()
@backend_click_option(default="gpu")
@root_options
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--configuration", type=click.Choice(["3d_fullres", "2d", "3d_lowres"]),
              default=cfg.DEFAULT_CONFIGURATION, show_default=True)
@click.option("--baseline-planner", type=click.Choice(sorted(BASELINE_PLANS_FILES)),
              default=cfg.DEFAULT_BASELINE_PLANNER, show_default=True)
@click.option("--arch", type=str, default=cfg.DEFAULT_ARCH, show_default=True)
@click.option("--arch-json", type=str, default=None, help="CViTConfig fields as JSON.")
@click.option("--token-stride", type=int, default=8, show_default=True)
@click.option("--patch-size", type=str, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--from-bundle", type=click.Path(path_type=Path), default=None)
@click.option("--llrd", type=float, default=1.0, show_default=True)
@click.option("--loss", type=str, default=cfg.DEFAULT_LOSS, show_default=True)
@click.option("--loss-config", "loss_config_json", type=str, default=None)
@click.option("--folds", type=str, default="0", show_default=True)
@click.option("--epochs", "num_epochs", type=int, default=None)
@click.option("--iterations-per-epoch", type=int, default=None)
@click.option("--lr", type=float, default=cfg.DEFAULT_LR, show_default=True)
@click.option("--warmup-epochs", type=int, default=cfg.DEFAULT_WARMUP_EPOCHS, show_default=True)
@click.option("--probe-every", type=int, default=cfg.DEFAULT_PROBE_EVERY, show_default=True)
@click.option("--no-mirror", is_flag=True, default=False)
@click.option("--tag", type=str, default=None)
@click.option("--device", type=str, default=None)
@click.option("--num-processes", type=int, default=8, show_default=True)
@click.option("--continue-training", is_flag=True, default=False)
@click.option("--plan-only", is_flag=True, default=False)
@click.option("--skip-planning", is_flag=True, default=False)
@click.option("--skip-preprocessing", is_flag=True, default=False)
def main(backend: str, arch_json: str | None, patch_size: str | None, folds: str,
         loss_config_json: str | None, device: str | None, **options: Any) -> None:
    """Plan, preprocess and train a CViT with nnU-Net."""
    paths = paths_from_options(pop_roots(options))
    run_train(
        paths=paths, arch_overrides=json.loads(arch_json) if arch_json else None,
        patch_size=parse_int_list(patch_size) or None, folds=[f.strip() for f in folds.split(",") if f.strip()],
        loss_config=json.loads(loss_config_json) if loss_config_json else None,
        device=torch_device_for_backend(backend, device=device), **options,
    )


if __name__ == "__main__":
    main()
