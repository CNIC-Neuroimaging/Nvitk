"""
CViT stage 1 — self-supervised pre-training (SimMIM / MAE) on the nnssl engine → encoder bundle.

Description
-----------
1. nnssl fingerprint + plan of the stage-0 corpus (reused when already on disk);
2. nnssl preprocessing for the chosen spacing style (reused when complete);
3. :class:`~nvitk.pipes.cvit.ssl_trainers.CViTSimMIMTrainer` / ``CViTMAETrainer`` training;
4. a **bundle** under ``<results_root>/stage1_pretrain/<name>/``::

       checkpoint_final.pth   nnssl checkpoint (network_weights + cvit_config + adaptation plan)
       encoder.pth            just the tokenizer.* / encoder.* tensors (what stage 2 loads)
       cvit_config.json       the encoder's CViTConfig
       adaptation_plan.json   nnssl adaptation plan (CViT key mapping)
       bundle.json            provenance

The bundle's **architecture is authoritative**: stage 2 builds its CViT with the same tokenizer,
transformer size and stem schedule (patch size, decoder and skips may differ), otherwise the
encoder could not load. Pass ``--from-bundle`` to stage 2 / ``nvitk-cvit``.

Import order: nnssl binds its roots at import time, so :func:`apply_nnssl_env` runs before
anything from nnssl (including :mod:`nvitk.pipes.cvit.ssl_trainers`) is imported.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.stage0_dataprep import corpus_folder_name
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import parse_int_list, paths_from_options, pop_roots, root_options
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

BUNDLE_CHECKPOINT = "checkpoint_final.pth"
BUNDLE_ENCODER = "encoder.pth"
BUNDLE_META = "bundle.json"
PREPROCESS_DONE_MARKER = "valid_imgs.json"
SPACING_STYLES: tuple[str, ...] = ("median", "noresample", "onemmiso")


def bundle_dir(results_root: Path, name: str) -> Path:
    return Path(results_root) / pth.STAGE1_PRETRAIN_DIR / name


def read_bundle(path: Path | str) -> dict[str, Any]:
    """Load a bundle's metadata: ``{"dir", "encoder", "cvit_config", "meta"}``.

    Raises
    ------
    FileNotFoundError
        If *path* is not a bundle directory (or a ``bundle.json`` inside one).
    """
    p = Path(path)
    directory = p.parent if p.is_file() else p
    meta_path = directory / BUNDLE_META
    if not meta_path.is_file():
        raise FileNotFoundError(f"{directory} is not a CViT stage-1 bundle (no {BUNDLE_META}).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "dir": directory,
        "encoder": directory / BUNDLE_ENCODER,
        "cvit_config": json.loads((directory / "cvit_config.json").read_text(encoding="utf-8")),
        "meta": meta,
    }


def _preprocessing_complete(dataset_dir: Path, configuration: str) -> bool:
    plans = dataset_dir / "nnsslPlans.json"
    marker = dataset_dir / PREPROCESS_DONE_MARKER
    collection = dataset_dir / f"pretrain_data__{configuration}.json"
    if not (plans.is_file() and marker.is_file() and collection.is_file()):
        return False
    return marker.stat().st_mtime >= collection.stat().st_mtime


def run_pretrain(
    *,
    paths: CViTPaths,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    corpus_id: int = pth.DEFAULT_CORPUS_ID,
    name: str | None = None,
    method: str = "simmim",
    arch: str | None = cfg.DEFAULT_ARCH,
    arch_overrides: dict[str, Any] | None = None,
    configuration: str = cfg.DEFAULT_SSL_CONFIG,
    patch_size: Sequence[int] = cfg.DEFAULT_SSL_PATCH,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    iterations_per_epoch: int | None = None,
    lr: float = cfg.DEFAULT_LR,
    warmup_epochs: int = cfg.DEFAULT_WARMUP_EPOCHS,
    mask_ratio: float = cfg.DEFAULT_MASK_RATIO,
    device: str = "cuda",
    num_processes: int = 8,
    continue_training: bool = False,
    overwrite: bool = False,
) -> Path:
    """Pre-train a CViT encoder on the stage-0 corpus; returns the bundle directory.

    Parameters
    ----------
    method
        ``simmim`` (works with every tokenizer) or ``mae`` (token dropping).
    arch, arch_overrides
        Preset and :class:`~nvitk.nn.cvit.CViTConfig` fields of the encoder (tokenizer, stem,
        transformer size). Decoder/skip fields are irrelevant here.
    configuration
        nnssl spacing style; ``onemmiso`` is discouraged for fine structures.
    """
    from nvitk.pipes._engines.env import apply_nnssl_env

    if method not in ("simmim", "mae"):
        raise ValueError(f"Unknown SSL method {method!r}; expected 'simmim' or 'mae'.")
    if configuration not in SPACING_STYLES:
        raise ValueError(f"Unknown nnssl configuration {configuration!r}; expected {SPACING_STYLES}.")
    apply_nnssl_env(paths.nnssl_raw, paths.nnssl_preprocessed, paths.nnssl_results)

    corpus = corpus_folder_name(corpus_id, dataset_name)
    pretrain_json = paths.nnssl_raw / corpus / "pretrain_data.json"
    if not pretrain_json.is_file():
        raise FileNotFoundError(f"{pretrain_json} not found. Run stage 0 with --corpus-source / --corpus-from-train.")
    dataset_dir = paths.nnssl_preprocessed / corpus

    from nnssl.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess,
    )

    # ---- 1. Plan + preprocess (reused unless --overwrite) -----------------------------
    if overwrite or not (dataset_dir / "nnsslPlans.json").is_file():
        log.info("stage1: fingerprinting and planning %s", corpus)
        extract_fingerprints([corpus_id], num_processes=num_processes, clean=True)
        plan_experiments([corpus_id])
    if overwrite or not _preprocessing_complete(dataset_dir, configuration):
        log.info("stage1: preprocessing %s for %r", corpus, configuration)
        preprocess([corpus_id], plans_identifier="nnsslPlans", configurations=(configuration,),
                   num_processes=num_processes)
    # nnssl reports a failed case with a print and carries on; a corpus with nothing usable must
    # stop here, not as a FileNotFoundError deep inside the first training batch.
    n_volumes = sum(1 for _ in dataset_dir.rglob("*.b2nd"))
    if n_volumes == 0:
        raise RuntimeError(
            f"nnssl preprocessing produced no volumes under {dataset_dir}; see the 'Error processing' "
            f"lines above. Re-run with --overwrite after fixing the corpus."
        )
    log.info("stage1: %d preprocessed volume(s) available.", n_volumes)

    # ---- 2. Trainer ---------------------------------------------------------------------
    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnssl.experiment_planning.experiment_planners.plan import Plan

    from nvitk.nn.cvit.weights import encoder_state_dict
    from nvitk.pipes.cvit.ssl_trainers import CVIT_CONFIG_FILE, SSL_TRAINERS

    plan = Plan.load_from_file(str(dataset_dir / "nnsslPlans.json"))
    collection = load_json(str(dataset_dir / f"pretrain_data__{configuration}.json"))
    trainer = SSL_TRAINERS[method](plan=plan, configuration_name=configuration, fold="all",
                                   pretrain_json=collection, device=torch.device(device))
    trainer.config_plan.patch_size = [int(p) for p in patch_size]
    trainer.cvit_arch = {**({"preset": arch} if arch else {}), **dict(arch_overrides or {})}
    trainer.mask_percentage = float(mask_ratio)
    trainer.initial_lr = float(lr)
    trainer.warmup_epochs = int(warmup_epochs)
    if batch_size is not None:
        trainer.total_batch_size = int(batch_size)
    if num_epochs is not None:
        trainer.num_epochs = int(num_epochs)
    if iterations_per_epoch is not None:
        trainer.num_iterations_per_epoch = int(iterations_per_epoch)
        trainer.num_val_iterations_per_epoch = max(1, int(iterations_per_epoch) // 5)
    log.info("stage1 | %s %s | patch=%s batch=%s epochs=%s mask=%.2f device=%s", method,
             trainer.cvit_arch, list(patch_size), trainer.total_batch_size, trainer.num_epochs,
             mask_ratio, device)

    resumed = None
    latest = Path(trainer.output_folder) / "checkpoint_latest.pth"
    if continue_training and latest.is_file():
        trainer.load_checkpoint(str(latest))
        resumed = str(latest)
        log.ok(f"stage1: resuming at epoch {trainer.current_epoch} from {latest}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    trainer.run_training()

    # ---- 3. Bundle ------------------------------------------------------------------------
    produced = Path(trainer.output_folder) / BUNDLE_CHECKPOINT
    if not produced.is_file():
        raise FileNotFoundError(f"nnssl finished without writing {produced}.")
    target = bundle_dir(paths.results_root, name or f"{method}_{trainer.cvit_arch.get('preset', 'custom')}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, target / BUNDLE_CHECKPOINT)
    for fname in ("adaptation_plan.json", CVIT_CONFIG_FILE):
        src = Path(trainer.output_folder_base) / fname
        if src.is_file():
            shutil.copyfile(src, target / fname)
    net = trainer.network.module if trainer.is_ddp else trainer.network
    torch.save(encoder_state_dict(getattr(net, "_orig_mod", net)), target / BUNDLE_ENCODER)
    meta = {
        "stage": "stage1", "created": datetime.now().isoformat(timespec="seconds"),
        "method": method, "corpus": corpus, "configuration": configuration,
        "patch_size": list(patch_size), "mask_ratio": mask_ratio, "lr": lr,
        "num_epochs": trainer.num_epochs, "batch_size": trainer.total_batch_size,
        "source_checkpoint": str(produced), "resumed_from": resumed,
        "nnssl_output_folder": str(trainer.output_folder_base),
    }
    (target / BUNDLE_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.ok(f"stage1: bundle -> {target}")
    return target


# ---------------------------------------------------------------------------
# SGE
# ---------------------------------------------------------------------------


def _worker_argv(**o: Any) -> list[str]:
    inside = container_layout()
    argv = [*python_module_argv("nvitk.pipes.cvit.stage1_pretrain"),
            *sge_backend_cli_args(o.get("backend", "gpu")), *root_args(inside),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME)),
            "--corpus-id", str(o.get("corpus_id", pth.DEFAULT_CORPUS_ID)),
            "--method", o.get("method", "simmim"),
            "--configuration", o.get("configuration", cfg.DEFAULT_SSL_CONFIG),
            "--patch-size", ",".join(str(p) for p in o.get("patch_size", cfg.DEFAULT_SSL_PATCH)),
            "--mask-ratio", repr(float(o.get("mask_ratio", cfg.DEFAULT_MASK_RATIO))),
            "--lr", repr(float(o.get("lr", cfg.DEFAULT_LR))),
            "--warmup-epochs", str(o.get("warmup_epochs", cfg.DEFAULT_WARMUP_EPOCHS)),
            "--device", torch_device_for_backend(o.get("backend", "gpu"), device=o.get("device"), remote=True),
            "--num-processes", str(o.get("num_processes", 8))]
    if o.get("arch"):
        argv += ["--arch", o["arch"]]
    if o.get("arch_overrides"):
        argv += ["--arch-json", quote_path(json.dumps(o["arch_overrides"]))]
    for key, flag in (("name", "--name"), ("batch_size", "--batch-size"), ("num_epochs", "--epochs"),
                      ("iterations_per_epoch", "--iterations-per-epoch")):
        if o.get(key) is not None:
            argv += [flag, quote_path(str(o[key]))]
    for key, flag in (("continue_training", "--continue-training"), ("overwrite", "--overwrite")):
        if o.get(key):
            argv.append(flag)
    return argv


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    """Emit or submit the stage 1 job (GPU)."""
    return submit_stage_job(
        "stage1", _worker_argv(**o), paths=paths, container=container, src_dir=src_dir,
        backend=o.get("backend", "gpu"), pe_smp=o.get("num_processes"),
        job_suffix=o.get("method", "simmim"), hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("cvit-stage1-pretrain")
@config_dir_click_option()
@backend_click_option(default="gpu")
@root_options
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--corpus-id", type=int, default=pth.DEFAULT_CORPUS_ID, show_default=True)
@click.option("--name", type=str, default=None, help="Bundle name (default: <method>_<preset>).")
@click.option("--method", type=click.Choice(["simmim", "mae"]), default="simmim", show_default=True)
@click.option("--arch", type=str, default=cfg.DEFAULT_ARCH, show_default=True)
@click.option("--arch-json", type=str, default=None, help="Extra CViTConfig fields as JSON.")
@click.option("--configuration", type=click.Choice(SPACING_STYLES), default=cfg.DEFAULT_SSL_CONFIG,
              show_default=True)
@click.option("--patch-size", type=str, default=",".join(map(str, cfg.DEFAULT_SSL_PATCH)), show_default=True)
@click.option("--batch-size", type=int, default=None)
@click.option("--epochs", "num_epochs", type=int, default=None)
@click.option("--iterations-per-epoch", type=int, default=None)
@click.option("--lr", type=float, default=cfg.DEFAULT_LR, show_default=True)
@click.option("--warmup-epochs", type=int, default=cfg.DEFAULT_WARMUP_EPOCHS, show_default=True)
@click.option("--mask-ratio", type=float, default=cfg.DEFAULT_MASK_RATIO, show_default=True)
@click.option("--device", type=str, default=None, help="Torch device (default from --backend).")
@click.option("--num-processes", type=int, default=8, show_default=True)
@click.option("--continue-training", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
def main(backend: str, arch_json: str | None, patch_size: str, device: str | None, **options: Any) -> None:
    """Self-supervised CViT pre-training on the stage-0 corpus."""
    paths = paths_from_options(pop_roots(options))
    run_pretrain(paths=paths, arch_overrides=json.loads(arch_json) if arch_json else None,
                 patch_size=parse_int_list(patch_size),
                 device=torch_device_for_backend(backend, device=device), **options)


if __name__ == "__main__":
    main()
