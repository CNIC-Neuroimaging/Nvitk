"""ToPBrain stage 1: obtain the pre-trained model.

Produces a **pre-trained bundle** the training stage can consume, from one of two sources. It
deliberately never touches the labelled training data prepared by stage 0 — pre-training on the
same volumes you fine-tune on buys little and muddies the evaluation.

``--source openmind`` *(default)*
    Download a published checkpoint trained on the OpenMind corpus (~114 k brain MR volumes).
    Both encoder families are available: ResEnc-L (convolutional) and Primus-M (transformer).
``--source scratch``
    Pre-train from random initialisation with nnssl, on the unlabeled corpus stage 0 built with
    ``--target corpus``. Only sensible with a corpus far larger than the challenge's 50 volumes.
``--source scratch --init-checkpoint-name <name>``
    Domain-adaptive pre-training: start from a published checkpoint and continue on the
    in-domain corpus. Use a low ``--initial-lr`` and few ``--num-epochs``, or the continuation
    overwrites what the checkpoint was worth.

The bundle
----------
``<results_root>/stage1_pretrain/<name>/``

``checkpoint_final.pth``
    The nnssl-format checkpoint. This is what stage 2 feeds to ``nnUNetv2_preprocess_like_nnssl``.
``adaptation_plan.json``
    Architecture and the encoder/stem key layout. Several nnssl trainers drop the copy embedded
    in the checkpoint, so the sidecar is authoritative.
``segmentation_model.pth``
    A **materialised end-to-end segmentation network** — the pre-trained encoder with a
    segmentation decoder and head attached, at the plan's recommended patch size. Portable and
    inspectable on its own, and it proves the weights instantiate before any training is queued.
    Stage 2 does not read it: the training build regenerates the head for the actual class count
    and target spacing of the downstream dataset, which stage 1 cannot know.
``bundle.json``
    Provenance: source, architecture, trainer, transferred parameter counts.
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
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain.util import checkpoints as ckpt_util
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import tensorboard as tb
from nvitk.pipes.topbrain.util.nnssl_env import apply_nnssl_env
from nvitk.pipes.topbrain.util.paths import (
    CORPUS_DATASET_ID,
    STAGE1_PRETRAIN_DIR,
    TopBrainPaths,
)
from nvitk.pipes.topbrain.util.sge_backend import sge_backend_cli_args, torch_device_for_backend
from nvitk.pipes.topbrain.util.sge_stage import (
    build_stage_command,
    container_layout,
    quote_path,
    submit_stage_job,
)

log = Logger()

#: nnssl spacing styles. ``onemmiso`` is accepted but discouraged — see :func:`run_pretrain`.
SPACING_STYLES: tuple[str, ...] = ("median", "noresample", "onemmiso")

#: Class count used for the materialised generic segmentation model. A placeholder: stage 2
#: rebuilds the head for the downstream dataset's real class count.
GENERIC_NUM_CLASSES: int = 2

#: Files that make up a bundle.
BUNDLE_CHECKPOINT: str = "checkpoint_final.pth"
BUNDLE_PLAN: str = "adaptation_plan.json"
BUNDLE_MODEL: str = "segmentation_model.pth"
BUNDLE_META: str = "bundle.json"


def bundle_dir(results_root: Path, name: str) -> Path:
    """Directory holding the pre-trained bundle *name*."""
    return Path(results_root) / STAGE1_PRETRAIN_DIR / name


def _import_trainer(trainer_name: str):
    """Import an nnssl trainer class by name, validating its base class."""
    import nnssl
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
    from nnssl.utilities.find_class_by_name import recursive_find_python_class

    trainer_cls = recursive_find_python_class(
        join(nnssl.__path__[0], "training", "nnsslTrainer"), trainer_name,
        "nnssl.training.nnsslTrainer",
    )

    if trainer_cls is None:
        from nvitk.pipes.topbrain.util.trainers import get_trainer, trainer_names

        known = get_trainer(trainer_name)
        if known is not None and not known.usable:
            raise ValueError(
                f"nnssl trainer {trainer_name!r} exists but is unusable: "
                f"{known.unusable_reason}."
            )
        # A module that failed to import is skipped rather than aborting the scan, so a
        # trainer can be genuinely present yet unreachable. Distinguish the two: "not found"
        # sends you looking for a typo, which is the wrong place when the real cause is a
        # broken optional dependency.
        from nnssl.utilities.find_class_by_name import SKIPPED_MODULES

        related = {
            name: exc for name, exc in SKIPPED_MODULES.items()
            if trainer_name.lower().replace("trainer", "") in name.lower()
        }
        if related:
            name, exc = next(iter(related.items()))
            raise ValueError(
                f"nnssl trainer {trainer_name!r} could not be imported: {name} raised "
                f"{type(exc).__name__}: {exc}. Install that dependency, or pick another "
                f"trainer (see 'nvitk-topbrain --list-trainers')."
            )
        message = (
            f"nnssl trainer {trainer_name!r} not found. Registered choices: "
            f"{', '.join(trainer_names())}. See 'nvitk-topbrain --list-trainers'."
        )
        if SKIPPED_MODULES:
            message += (
                f" Note that {len(SKIPPED_MODULES)} module(s) were skipped because they could "
                f"not be imported: {', '.join(sorted(SKIPPED_MODULES))}."
            )
        raise ValueError(message)
    if not issubclass(trainer_cls, AbstractBaseTrainer):
        raise ValueError(f"{trainer_name!r} does not inherit from AbstractBaseTrainer.")
    return trainer_cls


def _check_ssl_loss_compatible(trainer: Any, candidate: Any, name: str) -> None:
    """Refuse an ``--ssl-loss`` the trainer cannot call.

    nnssl's losses are trainer-family specific: ``SparkTrainer`` calls its loss with
    ``(prediction, groundtruth, mask)`` while the MAE trainers use ``(model_output, target,
    mask)``. Mismatched, the ``TypeError`` surfaces only as "background workers are no longer
    alive", minutes in. Comparing against the trainer's *own* default loss catches it here,
    without a hard-coded table, so it keeps working for custom losses.
    """
    import inspect

    def _params(module: Any) -> list[str]:
        """Positional parameter names of ``module.forward``, excluding ``self``."""
        return [
            p.name for p in inspect.signature(type(module).forward).parameters.values()
            if p.name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
        ]

    try:
        expected, actual = _params(trainer.build_loss()), _params(candidate)
    except (TypeError, ValueError) as exc:
        log.warning("Could not verify --ssl-loss %r against the trainer (%s).", name, exc)
        return
    if expected != actual:
        raise ValueError(
            f"--ssl-loss {name!r} is not compatible with trainer {type(trainer).__name__}: it "
            f"calls its loss with {tuple(expected)} but {type(candidate).__name__}.forward takes "
            f"{tuple(actual)}. Use 'spark' with SparK trainers and the MAE losses "
            f"(mse, mse_masked, l1, ssim, ms_ssim) with the MAE trainers."
        )


def export_segmentation_model(
    checkpoint: Path, plan: dict[str, Any], destination: Path, *, num_classes: int
) -> dict[str, Any]:
    """Materialise an end-to-end segmentation network carrying the pre-trained encoder.

    Builds the architecture the adaptation plan names — ResEnc-L or Primus — attaches a
    segmentation decoder and head, loads the encoder and stem, and saves the whole network.

    Returns
    -------
    dict
        Architecture, parameter counts and how many were seeded.
    """
    import torch
    from nnssl.architectures.get_network_by_name import get_network_by_name
    from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan

    from nvitk.pipes.topbrain.util.weight_adapter import load_encoder_into

    architecture = str((plan.get("architecture_plans") or {}).get("arch_class_name") or "ResEncL")
    patch = tuple(int(p) for p in plan.get("recommended_downstream_patchsize") or (160, 160, 160))
    channels = int(plan.get("pretrain_num_input_channels", 1))

    # Primus needs the patch size to size its positional embedding, so a configuration plan has
    # to be supplied even though nothing else here reads it.
    config_plan = ConfigurationPlan(
        data_identifier="topbrain_generic", preprocessor_name="DefaultPreprocessor",
        spacing_style="median", normalization_schemes=["ZScoreNormalization"],
        use_mask_for_norm=[False], resampling_fn_data="resample_data_or_seg_to_shape",
        resampling_fn_data_kwargs={}, resampling_fn_mask="resample_data_or_seg_to_shape",
        resampling_fn_mask_kwargs={}, spacing=None, patch_size=list(patch),
    )
    network = get_network_by_name(config_plan, architecture, channels, int(num_classes))
    report = load_encoder_into(network, checkpoint)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "network_weights": network.state_dict(),
            "architecture": architecture,
            "num_input_channels": channels,
            "num_output_channels": int(num_classes),
            "patch_size": list(patch),
            "note": (
                "Generic segmentation network: pre-trained encoder plus an initialised decoder "
                "and head. The class count is a placeholder; stage 2 rebuilds the head for the "
                "downstream dataset."
            ),
        },
        destination,
    )
    total = sum(p.numel() for p in network.parameters())
    log.ok(
        f"exported {architecture} segmentation model ({total / 1e6:.1f}M params, "
        f"{report['matched']}/{report['target_encoder_keys']} encoder tensors seeded) "
        f"-> {destination}"
    )
    return {
        "architecture": architecture, "patch_size": list(patch),
        "num_input_channels": channels, "num_output_channels": int(num_classes),
        "total_parameters": int(total), **report,
    }


def _write_bundle(
    destination: Path, checkpoint: Path, plan: dict[str, Any], *, provenance: dict[str, Any],
    num_classes: int, export_model: bool,
) -> Path:
    """Assemble the bundle directory; returns its path."""
    destination.mkdir(parents=True, exist_ok=True)
    if Path(checkpoint).resolve() != (destination / BUNDLE_CHECKPOINT).resolve():
        shutil.copyfile(checkpoint, destination / BUNDLE_CHECKPOINT)
    (destination / BUNDLE_PLAN).write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    if export_model:
        provenance["segmentation_model"] = export_segmentation_model(
            destination / BUNDLE_CHECKPOINT, plan, destination / BUNDLE_MODEL,
            num_classes=num_classes,
        )
    (destination / BUNDLE_META).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    log.ok(f"stage1 bundle ready -> {destination}")
    return destination


def _plan_for(checkpoint: Path) -> dict[str, Any]:
    """Adaptation plan for *checkpoint*: embedded copy, else the sibling sidecar."""
    from nvitk.pipes.topbrain.util.weight_adapter import load_nnssl_checkpoint

    _, plan, _ = load_nnssl_checkpoint(Path(checkpoint))
    if plan is None:
        raise ValueError(
            f"{checkpoint} carries no nnssl adaptation plan and none was found beside it. "
            f"Stage 2 needs it to rebuild the architecture."
        )
    return plan


def run_pretrain(
    *,
    paths: TopBrainPaths,
    source: str = "openmind",
    name: str | None = None,
    checkpoint_name: str | None = None,
    checkpoint: Path | None = None,
    checkpoint_url: str | None = None,
    trainer_name: str | None = None,
    configuration_name: str | None = None,
    ssl_loss: str | None = None,
    ssl_loss_config: dict | None = None,
    patch_size: Sequence[int] | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    initial_lr: float | None = None,
    init_checkpoint_name: str | None = None,
    device: str = "cuda",
    num_processes: int = 8,
    num_classes: int = GENERIC_NUM_CLASSES,
    export_model: bool = True,
    skip_planning: bool = False,
    skip_preprocessing: bool = False,
    overwrite: bool = False,
    tensorboard: bool = False,
    tensorboard_dir: Path | None = None,
    tensorboard_interval: float = tb.DEFAULT_INTERVAL,
) -> Path:
    """Obtain a pre-trained bundle; returns its directory.

    Parameters
    ----------
    tensorboard
        Mirror the nnssl trainer's per-epoch log into TensorBoard events under
        ``<results_root>/tensorboard/stage1/`` while it trains. Only meaningful for
        ``--source scratch``: the openmind route downloads a checkpoint and trains nothing.
    tensorboard_dir
        Overrides that location. Left unset the pipeline's own layout is used, which is what
        keeps stage 1 and stage 2 under a single server.
    """
    provenance: dict[str, Any] = {
        "stage": "stage1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }

    # Both routes need nnssl importable: the openmind route to read an adaptation plan and
    # build the exported segmentation model, the scratch route to train. nnssl binds its roots
    # at import time, so this must precede any nnssl import.
    apply_nnssl_env(paths, create=True)

    # ---- Route A: a published checkpoint ------------------------------------
    if source == "openmind":
        resolved = ckpt_util.resolve_checkpoint(
            source="checkpoint", checkpoint=checkpoint, checkpoint_name=checkpoint_name,
            checkpoint_url=checkpoint_url, model_root=paths.model_root,
            nnssl_results=paths.nnssl_results,
        )
        if resolved is None:
            raise ValueError("--source openmind needs --checkpoint-name, --checkpoint or --url.")
        plan = _plan_for(resolved)
        provenance.update({
            "checkpoint_name": checkpoint_name,
            "source_checkpoint": str(resolved),
            "architecture": (plan.get("architecture_plans") or {}).get("arch_class_name"),
        })
        target = bundle_dir(paths.results_root, name or (checkpoint_name or "openmind"))
        if (target / BUNDLE_CHECKPOINT).is_file() and not overwrite:
            log.info("stage1: bundle already present, skipping -> %s", target)
            return target
        return _write_bundle(
            target, resolved, plan, provenance=provenance,
            num_classes=num_classes, export_model=export_model,
        )

    if source != "scratch":
        raise ValueError(f"Unknown --source {source!r}; expected 'openmind' or 'scratch'.")

    # ---- Route B: pre-train with nnssl --------------------------------------
    trainer_name = trainer_name or cfg.DEFAULT_SSL_TRAINER
    configuration_name = configuration_name or cfg.DEFAULT_SSL_CONFIG
    ssl_loss_config = dict(ssl_loss_config or {})
    if configuration_name not in SPACING_STYLES:
        raise ValueError(
            f"Unknown configuration {configuration_name!r}; expected {', '.join(SPACING_STYLES)}."
        )
    if ssl_loss:
        loss_util.validate_loss_name(ssl_loss, registry=loss_util.SSL_LOSSES)

    pretrain_json = paths.nnssl_raw_dir / "pretrain_data.json"
    if not pretrain_json.is_file():
        raise FileNotFoundError(
            f"Corpus descriptor not found: {pretrain_json}. Run stage 0 with --target corpus."
        )

    from nnssl.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess,
    )

    log.info(
        "stage1 scratch | trainer=%s config=%s loss=%s device=%s seed=%s",
        trainer_name, configuration_name, ssl_loss or "(trainer default)", device,
        init_checkpoint_name or "random",
    )
    if configuration_name == "onemmiso":
        log.warning(
            "Configuration 'onemmiso' resamples to 1 mm isotropic. Intracranial vessels are "
            "0.3-0.6 mm across; sub-millimetre structures will not survive it."
        )

    if not skip_planning:
        extract_fingerprints([CORPUS_DATASET_ID], num_processes=num_processes, clean=True)
        plan_experiments([CORPUS_DATASET_ID])
    if not skip_preprocessing:
        preprocess(
            [CORPUS_DATASET_ID], plans_identifier="nnsslPlans",
            configurations=(configuration_name,), num_processes=num_processes,
        )

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnssl.experiment_planning.experiment_planners.plan import Plan
    from torch.backends import cudnn

    dataset_dir = paths.nnssl_preprocessed / paths.corpus_dataset_name
    plan_obj = Plan.load_from_file(str(dataset_dir / "nnsslPlans.json"))
    collection_json = load_json(join(str(dataset_dir), f"pretrain_data__{configuration_name}.json"))

    trainer = _import_trainer(trainer_name)(
        plan=plan_obj, configuration_name=configuration_name,
        # nnssl always trains on a single deterministic split; 'all' is the only valid fold.
        fold="all", pretrain_json=collection_json, device=torch.device(device),
    )

    applied: dict[str, Any] = {}
    if patch_size is not None:
        trainer.config_plan.patch_size = [int(p) for p in patch_size]
        applied["patch_size"] = list(trainer.config_plan.patch_size)
    if batch_size is not None:
        trainer.total_batch_size = int(batch_size)
        applied["batch_size"] = int(batch_size)
    if num_epochs is not None:
        trainer.num_epochs = int(num_epochs)
        applied["num_epochs"] = int(num_epochs)
    if initial_lr is not None:
        trainer.initial_lr = float(initial_lr)
        applied["initial_lr"] = float(initial_lr)
    if ssl_loss is not None:
        built = loss_util.build_ssl_loss(ssl_loss, ssl_loss_config)
        _check_ssl_loss_compatible(trainer, built, ssl_loss)
        # build_loss() is called from initialize(); overriding the bound method keeps nnssl's
        # own call sequence intact.
        trainer.build_loss = lambda _b=built: _b
        applied["ssl_loss"] = ssl_loss
    if applied:
        log.info("Trainer overrides: %s", applied)

    seed_report = None
    if init_checkpoint_name:
        from nvitk.pipes.topbrain.util.weight_adapter import load_encoder_into

        seed = ckpt_util.resolve_checkpoint(
            source="checkpoint", checkpoint=None, checkpoint_name=init_checkpoint_name,
            checkpoint_url=None, model_root=paths.model_root,
            nnssl_results=paths.nnssl_results,
        )
        # The network only exists after initialize(); run_training() would call it itself, but
        # the first optimiser step would already have run on random weights.
        trainer.initialize()
        seed_report = load_encoder_into(trainer.network, seed)
        provenance["init_checkpoint"] = str(seed)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    # Monitoring only ever *reads* the trainer's log files, so it cannot perturb the run; a
    # failure inside it is warned about and swallowed rather than aborting hours of training.
    with tb.monitoring(
        paths, stages=("stage1",), enabled=tensorboard, event_root=tensorboard_dir,
        interval=tensorboard_interval,
    ):
        trainer.run_training()

    produced = Path(trainer.output_folder) / "checkpoint_final.pth"
    plan = _plan_for(produced)
    provenance.update({
        "trainer": trainer_name, "configuration": configuration_name,
        "ssl_loss": ssl_loss, "ssl_loss_config": ssl_loss_config, "overrides": applied,
        "seeded_encoder": seed_report, "device": device,
        "source_checkpoint": str(produced),
        "architecture": (plan.get("architecture_plans") or {}).get("arch_class_name"),
        "nnssl_output_folder": str(trainer.output_folder_base),
    })
    return _write_bundle(
        bundle_dir(paths.results_root, name or trainer_name), produced, plan,
        provenance=provenance, num_classes=num_classes, export_model=export_model,
    )


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(**o) -> list[str]:
    """Worker argv for stage 1, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage1_pretrain"),
        *sge_backend_cli_args(o.get("backend", "gpu")),
        "--nnssl-raw", quote_path(inside.nnssl_raw),
        "--nnssl-preprocessed", quote_path(inside.nnssl_preprocessed),
        "--nnssl-results", quote_path(inside.nnssl_results),
        "--corpus-root", quote_path(inside.corpus_root),
        "--results-root", quote_path(inside.results_root),
        "--model-root", quote_path(inside.model_root),
        "--source", o.get("source", "openmind"),
        "--device", o.get("device", "cuda"),
        "--num-processes", str(int(o.get("num_processes", 8))),
    ]
    for flag, key in (
        ("--name", "name"), ("--checkpoint-name", "checkpoint_name"),
        ("--trainer", "trainer_name"), ("--configuration", "configuration_name"),
        ("--ssl-loss", "ssl_loss"), ("--init-checkpoint-name", "init_checkpoint_name"),
    ):
        if o.get(key):
            argv.extend([flag, quote_path(str(o[key]))])
    for flag, key in (("--batch-size", "batch_size"), ("--num-epochs", "num_epochs")):
        if o.get(key):
            argv.extend([flag, str(int(o[key]))])
    if o.get("initial_lr"):
        argv.extend(["--initial-lr", str(float(o["initial_lr"]))])
    if o.get("patch_size"):
        argv.extend(["--patch-size", *[str(int(p)) for p in o["patch_size"]]])
    if not o.get("export_model", True):
        argv.append("--no-export-model")
    if o.get("overwrite"):
        argv.append("--overwrite")
    if o.get("tensorboard"):
        # No --tensorboard-dir: the worker's results_root is already the container-side mount,
        # so the default layout resolves to the same shared tree the workstation sees.
        argv.append("--tensorboard")
        argv.extend(["--tensorboard-interval", str(float(
            o.get("tensorboard_interval") or tb.DEFAULT_INTERVAL
        ))])
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **o) -> str:
    """Host shell command for the stage 1 SGE task."""
    return build_stage_command(
        "stage1", _worker_argv(**o), paths=paths, container=container, src_dir=src_dir,
        backend=o.get("backend", "gpu"),
        # Downloading a checkpoint needs no GPU; pre-training does.
        request_gpu=o.get("source") == "scratch" and o.get("device", "cuda") != "cpu",
        pe_smp=o.get("num_processes"), job_suffix=str(o.get("source", ""))[:16],
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **o,
) -> str:
    """Emit or submit the stage 1 SGE job."""
    return submit_stage_job(
        "stage1", _worker_argv(**o), paths=paths, container=container, src_dir=src_dir,
        backend=o.get("backend", "gpu"),
        request_gpu=o.get("source") == "scratch" and o.get("device", "cuda") != "cpu",
        pe_smp=o.get("num_processes"), job_suffix=str(o.get("source", ""))[:16],
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage1-pretrain")
@config_dir_click_option()
@backend_click_option(default="gpu")
@click.option("--nnssl-raw", type=click.Path(path_type=Path), required=True)
@click.option("--nnssl-preprocessed", type=click.Path(path_type=Path), required=True)
@click.option("--nnssl-results", type=click.Path(path_type=Path), required=True)
@click.option("--corpus-root", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--model-root", type=click.Path(path_type=Path), required=True)
@click.option("--source", type=click.Choice(["openmind", "scratch"]), default="openmind",
              show_default=True)
@click.option("--name", type=str, default=None, help="Bundle directory name.")
@click.option("--checkpoint-name", type=str, default=None,
              help="Published checkpoint (see --list-checkpoints on the master CLI).")
@click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
@click.option("--checkpoint-url", type=str, default=None)
@click.option("--trainer", "trainer_name", type=str, default=None)
@click.option("--configuration", "configuration_name",
              type=click.Choice(list(SPACING_STYLES)), default=None)
@click.option("--ssl-loss", type=str, default=None)
@click.option("--ssl-loss-config", type=str, default=None)
@click.option("--patch-size", type=int, nargs=3, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--num-epochs", type=int, default=None)
@click.option("--initial-lr", type=float, default=None)
@click.option("--init-checkpoint-name", type=str, default=None,
              help="Seed --source scratch from a published checkpoint (domain adaptation).")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-processes", type=int, default=8, show_default=True)
@click.option("--num-classes", type=int, default=GENERIC_NUM_CLASSES, show_default=True,
              help="Placeholder class count for the exported generic segmentation model.")
@click.option("--no-export-model", is_flag=True, default=False,
              help="Skip materialising segmentation_model.pth.")
@click.option("--skip-planning", is_flag=True, default=False)
@click.option("--skip-preprocessing", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--tensorboard", is_flag=True, default=False,
              help="Mirror the training log into TensorBoard events under "
                   "<results-root>/tensorboard/stage1/.")
@click.option("--tensorboard-dir", type=click.Path(path_type=Path), default=None,
              help="Event directory (default: <results-root>/tensorboard).")
@click.option("--tensorboard-interval", type=float, default=tb.DEFAULT_INTERVAL,
              show_default=True, help="Seconds between mirror passes.")
def main(
    nnssl_raw: Path, nnssl_preprocessed: Path, nnssl_results: Path, corpus_root: Path,
    results_root: Path, model_root: Path, source: str, name: str | None,
    checkpoint_name: str | None, checkpoint: Path | None, checkpoint_url: str | None,
    trainer_name: str | None, configuration_name: str | None, ssl_loss: str | None,
    ssl_loss_config: str | None, patch_size: tuple[int, int, int] | None, batch_size: int | None,
    num_epochs: int | None, initial_lr: float | None, init_checkpoint_name: str | None,
    device: str | None, num_processes: int, num_classes: int, no_export_model: bool,
    skip_planning: bool, skip_preprocessing: bool, overwrite: bool,
    tensorboard: bool, tensorboard_dir: Path | None, tensorboard_interval: float,
    backend: str = "gpu",
) -> None:
    """CLI entry point: obtain the pre-trained bundle."""
    Logger()
    paths = TopBrainPaths(
        challenge_root=corpus_root, nnssl_raw=nnssl_raw,
        nnssl_preprocessed=nnssl_preprocessed, nnssl_results=nnssl_results,
        nnunet_raw=results_root, nnunet_preprocessed=results_root, nnunet_results=results_root,
        results_root=results_root, model_root=model_root, corpus_root=corpus_root,
    )
    run_pretrain(
        paths=paths, source=source, name=name, checkpoint_name=checkpoint_name,
        checkpoint=checkpoint, checkpoint_url=checkpoint_url, trainer_name=trainer_name,
        configuration_name=configuration_name, ssl_loss=ssl_loss,
        ssl_loss_config=loss_util.parse_loss_config(ssl_loss_config),
        patch_size=patch_size or None, batch_size=batch_size, num_epochs=num_epochs,
        initial_lr=initial_lr, init_checkpoint_name=init_checkpoint_name,
        device=device or torch_device_for_backend(backend), num_processes=num_processes,
        num_classes=num_classes, export_model=not no_export_model,
        skip_planning=skip_planning, skip_preprocessing=skip_preprocessing, overwrite=overwrite,
        tensorboard=tensorboard, tensorboard_dir=tensorboard_dir,
        tensorboard_interval=tensorboard_interval,
    )


__all__ = [
    "BUNDLE_CHECKPOINT", "BUNDLE_META", "BUNDLE_MODEL", "BUNDLE_PLAN", "SPACING_STYLES",
    "bundle_dir", "build_sge_command", "export_segmentation_model", "main", "run_pretrain",
    "submit_sge",
]


if __name__ == "__main__":
    main()
