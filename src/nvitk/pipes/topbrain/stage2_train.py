"""ToPBrain stage 2: transfer-learning training on the prepared dataset.

**Inputs**

- ``<nnunet_raw>/DatasetXXX_…/`` from stage 0
- a pre-trained bundle from stage 1

**Outputs**

- ``<nnunet_preprocessed>/DatasetXXX_…/ptPlans__<name>____Spacing…json`` and its preprocessed data
- ``<nnunet_results>/DatasetXXX_…/<trainer>__<plans>__3d_fullres/fold_N/``
- ``<results_root>/stage2_train/<dataset>/topbrain_stage2.json`` — provenance

Why the in-tree nnU-Net build
-----------------------------
Released ``nnunetv2`` cannot consume an nnssl checkpoint. The in-tree build at
``pipes/topbrain/nnunet`` adds the two pieces that make it possible:

``nnUNetv2_preprocess_like_nnssl``
    Reads the checkpoint's adaptation plan, derives target spacing and normalisation from it,
    and writes a ``ptPlans__…`` plans file that records where the weights are and how to load
    them. It *adapts* an existing plan rather than creating one, so a **baseline plan must
    exist first** — this stage runs fingerprinting and planning to produce it.
``PretrainedTrainer`` / ``PretrainedTrainer_Primus``
    Build the network from those plans, load the pre-trained encoder into it, and fine-tune with
    a warm-up schedule. The Primus variant is what makes the transformer checkpoints usable.

This is also where the encoder finally becomes a task-specific segmentation model: the head is
sized for *this* dataset's class count and spacing, which stage 1 could not know.

The build is invoked as a subprocess with ``PYTHONPATH`` pointing at it, so the released
``nnunetv2`` the rest of nvitk depends on stays untouched — see
:mod:`nvitk.pipes.topbrain.util.nnunet_run`.
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
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import nnunet_run
from nvitk.pipes.topbrain.util import sampling as sampling_util
from nvitk.pipes.topbrain.util import tensorboard as tb
from nvitk.pipes.topbrain.util.nnunet_env import nnunet_env
from nvitk.pipes.topbrain.util.paths import (
    DATASET_IDS,
    DATASET_SUFFIXES,
    STAGE2_TRAIN_DIR,
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

#: How the downstream spacing and normalisation are derived from the pre-training plan.
ADAPTATION_MODES: tuple[str, ...] = ("default_nnunet", "like_pretrained", "no_resample", "fixed")

#: nnU-Net configuration. Vessels are thin and near-isotropic here, so the cascade buys nothing.
CONFIGURATION: str = "3d_fullres"


def training_marker_path(results_root: Path, label_set: str) -> Path:
    """Where stage 2 records what it trained."""
    return Path(results_root) / STAGE2_TRAIN_DIR / dataset_name_for(label_set) / "topbrain_stage2.json"


def read_training_provenance(results_root: Path, label_set: str) -> dict[str, Any]:
    """What stage 2 trained: run directory, trainer, plans identifier, architecture.

    Stages 3-5 all need these, and none can be derived up front: the plans identifier embeds a
    spacing that ``preprocess_like_nnssl`` only decides at run time, and the trainer family
    depends on the pre-trained checkpoint's architecture. Reading the marker is what keeps
    inference, evaluation and packaging pointed at the model that was actually trained rather
    than at a name assembled from defaults.

    Raises
    ------
    FileNotFoundError
        If stage 2 has not run for this label set.
    """
    marker = training_marker_path(results_root, label_set)
    if not marker.is_file():
        raise FileNotFoundError(
            f"No stage 2 provenance at {marker}. Run stage 2 first, or pass the run name, "
            f"plans identifier and architecture explicitly."
        )
    return json.loads(marker.read_text(encoding="utf-8"))


def resolve_trained_run(
    results_root: Path,
    label_set: str,
    *,
    plans_identifier: str | None = None,
    architecture: str | None = None,
    configuration_name: str | None = None,
) -> dict[str, str]:
    """Fill in the identity of the trained run from stage 2's provenance.

    Stages 4 and 5 need three facts about the model they are about to use, and **none of them
    can be assembled from defaults**:

    ``plans_identifier``
        ``ptPlans__<pretrain_name>____Spacing__<x>_<y>_<z>___Norm__Z`` — the spacing is decided
        by ``preprocess_like_nnssl`` at run time, so the name is only knowable after the fact.
    ``architecture``
        Decides the trainer family (``PretrainedTrainer`` vs ``PretrainedTrainer_Primus``), and
        comes from the pre-trained checkpoint rather than from any flag.
    ``configuration``
        Recorded for completeness; currently always ``3d_fullres``.

    Anything passed explicitly wins; the rest is read from the marker. Guessing instead would
    build a run directory name that does not exist on disk, and the failure would surface as a
    confusing "plans not found" several layers down.
    """
    provided = {
        "plans_identifier": plans_identifier,
        "architecture": architecture,
        "configuration": configuration_name,
    }
    if all(v is not None for v in provided.values()):
        return {k: str(v) for k, v in provided.items()}

    provenance = read_training_provenance(results_root, label_set)
    resolved = {
        key: provided[key] if provided[key] is not None else provenance.get(key)
        for key in provided
    }
    missing = [k for k, v in resolved.items() if not v]
    if missing:
        raise KeyError(
            f"Stage 2 provenance at {training_marker_path(results_root, label_set)} is missing "
            f"{missing}. Pass them explicitly, or re-run stage 2."
        )
    log.debug("Resolved trained run from provenance: %s", resolved)
    return {k: str(v) for k, v in resolved.items()}


def trained_run_name(results_root: Path, label_set: str) -> str:
    """Name of the directory nnU-Net trained into, e.g. ``<trainer>__<plans>__3d_fullres``.

    Read from stage 2's provenance rather than composed, for the same reason as
    :func:`resolve_trained_run`: the plans identifier embeds a spacing chosen at run time, so
    the name cannot be assembled from flags. Falls back to composing it from the recorded parts
    if the marker predates ``results_dir``.
    """
    provenance = read_training_provenance(results_root, label_set)
    recorded = provenance.get("results_dir")
    if recorded:
        return Path(recorded).name
    return "__".join((
        provenance["trainer"], provenance["plans_identifier"], provenance["configuration"],
    ))


def run_checkpoint(
    results_root: Path,
    label_set: str,
    *,
    fold: int | str = "all",
    checkpoint: str = "checkpoint_final.pth",
) -> Path:
    """Path to a finished stage 2 run's checkpoint, read from its provenance.

    Used to chain fine-tunings: the multi-class run initialises from whatever the binary run
    produced, without anyone having to reconstruct a directory name whose plans identifier
    embeds a spacing chosen at run time.

    Raises
    ------
    FileNotFoundError
        If that stage 2 run has not finished, naming the file that is missing.
    """
    provenance = read_training_provenance(results_root, label_set)
    results_dir = provenance.get("results_dir")
    if not results_dir:
        raise FileNotFoundError(
            f"Stage 2 provenance for label set {label_set!r} records no results_dir; re-run it."
        )
    path = Path(results_dir) / f"fold_{fold}" / checkpoint
    if not path.is_file():
        raise FileNotFoundError(
            f"No {checkpoint} for label set {label_set!r} at {path}. The {label_set} training "
            f"must finish before a run can initialise from it."
        )
    return path


def dataset_name_for(label_set: str) -> str:
    """nnU-Net dataset folder name for *label_set*."""
    return f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"


def read_bundle(bundle: Path) -> tuple[Path, dict[str, Any], str]:
    """Read a stage 1 bundle; returns ``(checkpoint, plan, architecture)``.

    Raises
    ------
    FileNotFoundError
        If the bundle is incomplete, naming what is missing.
    """
    from nvitk.pipes.topbrain.stage1_pretrain import BUNDLE_CHECKPOINT, BUNDLE_PLAN

    bundle = Path(bundle)
    checkpoint, plan_path = bundle / BUNDLE_CHECKPOINT, bundle / BUNDLE_PLAN
    for path in (checkpoint, plan_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete stage 1 bundle: {path} is missing.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    architecture = str(
        (plan.get("architecture_plans") or {}).get("arch_class_name") or "ResEncL"
    )
    return checkpoint, plan, architecture


def patch_plans_patch_size(
    nnunet_preprocessed: Path,
    dataset_name: str,
    plans_identifier: str,
    *,
    patch_size: Sequence[int],
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Override the patch (and optionally batch) size in a generated ``ptPlans`` file.

    ``preprocess_like_nnssl`` copies ``recommended_downstream_patchsize`` from the pre-training
    plan — 160³ for every OpenMind checkpoint. That is sized for the A100s the checkpoints were
    trained on; a Primus-M transformer at 160³ with 37 output channels does not fit a 16 GB
    card. Patch size is a training-time crop, so lowering it changes what fits in memory without
    invalidating the preprocessed data.

    Edited **in place** rather than written to a new plans name: the file is already a derived
    artefact keyed to this pre-training name and is regenerated on every non-skipped run, and
    keeping one name means ``-p`` and the results directory stay predictable. The change is
    recorded in the stage's provenance.

    Primus interpolates its positional embedding to the configured patch size when loading, so
    a smaller patch stays compatible with the pre-trained weights.

    Raises
    ------
    ValueError
        If the patch size is not divisible by the network's total downsampling factor.
    """
    plans_path = Path(nnunet_preprocessed) / dataset_name / f"{plans_identifier}.json"
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    config = plans["configurations"][CONFIGURATION]

    divisor = 32  # 6 encoder stages / Primus patch embedding both require multiples of 32
    bad = [int(p) for p in patch_size if int(p) % divisor]
    if bad:
        raise ValueError(
            f"Patch size {tuple(patch_size)} is not divisible by {divisor} in {bad}."
        )

    previous = list(config.get("patch_size", []))
    if previous == [int(p) for p in patch_size] and (
        batch_size is None or int(config.get("batch_size", -1)) == int(batch_size)
    ):
        # Already patched -- by this run's preparation job, or by an earlier invocation.
        # Rewriting is not free: the file is truncated for an instant, and a concurrent fold job
        # reading it to build its network gets either a parse error or, worse, the unpatched
        # 160^3 geometry. Nothing to change, so nothing is written.
        log.info("%s already at patch_size %s, batch_size=%s; left untouched.",
                 plans_path.name, previous, config.get("batch_size"))
        return {"previous_patch_size": previous, "patch_size": previous,
                "batch_size": config.get("batch_size")}
    config["patch_size"] = [int(p) for p in patch_size]
    if batch_size is not None:
        config["batch_size"] = int(batch_size)
    plans_path.write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")
    log.info(
        "Patched %s: patch_size %s -> %s, batch_size=%s",
        plans_path.name, previous, config["patch_size"], config.get("batch_size"),
    )
    return {"previous_patch_size": previous, "patch_size": config["patch_size"],
            "batch_size": config.get("batch_size")}


def install_splits(nnunet_raw: Path, nnunet_preprocessed: Path, label_set: str) -> Path:
    """Copy stage 0's patient-grouped folds into ``nnUNet_preprocessed``.

    nnU-Net generates its own random split on first use if it finds none. That split is by
    *case*, which would put a patient's CT in training and their MR in validation and leak the
    subject across the fold.
    """
    import shutil

    dataset_name = dataset_name_for(label_set)
    source = Path(nnunet_raw) / dataset_name / "splits_final.json"
    destination = Path(nnunet_preprocessed) / dataset_name / "splits_final.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Patient-grouped splits not found at {source}. Run stage 0 first — without them "
            f"nnU-Net would invent a case-level split that leaks patients across folds."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.read_bytes() == source.read_bytes():
        # Same content already there. Recopying would blank the file for an instant, and a
        # concurrent fold job reading its split at start-up would see an empty one.
        log.info("Patient-grouped splits already installed at %s", destination)
        return destination
    shutil.copyfile(source, destination)
    log.info("Installed patient-grouped splits -> %s", destination)
    return destination


def run_train(
    *,
    paths: TopBrainPaths,
    bundle: Path,
    label_set: str = "ta36",
    pretrain_name: str | None = None,
    loss: str | None = None,
    loss_config: dict | None = None,
    folds: Sequence[int | str] = (0,),
    adaptation_mode: str = "default_nnunet",
    baseline_planner: str = "nnUNetPlannerResEncL",
    skip_planning: bool = False,
    verify_integrity: bool = False,
    target_spacing: Sequence[float] | None = None,
    patch_size: Sequence[int] | None = None,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    device: str = "cuda",
    num_gpus: int = 1,
    num_processes: int = 8,
    skip_preprocessing: bool = False,
    plan_only: bool = False,
    continue_training: bool = False,
    from_scratch: bool = False,
    tensorboard: bool = False,
    tensorboard_dir: Path | None = None,
    tensorboard_interval: float = tb.DEFAULT_INTERVAL,
    plans_from_label_set: str | None = None,
    pretrained_weights: Path | None = None,
    init_from_label_set: str | None = None,
    sampling: str = "default",
    sampling_temperature: float = sampling_util.DEFAULT_TEMPERATURE,
    sampling_oversample: float | None = None,
) -> Path:
    """Preprocess against the pre-training plan and fine-tune; returns the results directory.

    Parameters
    ----------
    tensorboard
        Mirror each fold's per-epoch log into TensorBoard events under
        ``<results_root>/tensorboard/stage2/`` as it trains. The per-class pseudo-dice series
        are named from *label_set*, so a stalled side-road vessel is identifiable at a glance
        rather than hidden inside a 36-class mean.
    tensorboard_dir
        Overrides that location; left unset the pipeline layout is used, which is what puts
        stage 1 and stage 2 under one server.
    plans_from_label_set
        Borrow another label set's ``ptPlans`` instead of deriving one from this dataset's own
        fingerprint. Two datasets must share a plan for a model trained on one to initialise a
        model on the other — see :func:`~nvitk.pipes.topbrain.util.nnunet_run.move_plans`. Used
        by the binary stage so its network is identical to the multi-class one bar the head.
    pretrained_weights
        Checkpoint of a previous stage 2 run to start from. Everything except the segmentation
        heads is copied, so a binary vessel model can seed the 37-class model.
    sampling
        ``default`` leaves nnU-Net's patch sampling alone. ``rare_aware`` oversamples the cases
        carrying rare classes and, inside them, the rare class itself — see
        :mod:`nvitk.pipes.topbrain.util.sampling`. It changes what the network sees, not what it
        optimises, so it composes with any ``--loss`` and should be measured against the same one.
    """
    loss = loss or cfg.DEFAULT_LOSS
    loss_config = dict(loss_config or {})
    if adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(
            f"Unknown --adaptation-mode {adaptation_mode!r}; expected "
            f"{', '.join(ADAPTATION_MODES)}."
        )

    if init_from_label_set and pretrained_weights is None:
        pretrained_weights = run_checkpoint(paths.results_root, init_from_label_set)
        log.info("Initialising from the %r run: %s", init_from_label_set, pretrained_weights)


    checkpoint, _, architecture = read_bundle(bundle)
    pretrain_name = pretrain_name or Path(bundle).name
    loss_util.validate_loss_name(loss, registry=loss_util.SEGMENTATION_LOSSES)
    trainer = loss_util.trainer_for_loss(loss, architecture=architecture)

    dataset_id = DATASET_IDS[label_set]
    dataset_name = dataset_name_for(label_set)

    env = nnunet_env(paths, num_processes=num_processes)
    # A custom loss cannot be passed as a trainer argument; the trainer reads it from here.
    env[loss_util.LOSS_SPEC_ENV] = loss_util.loss_spec_payload(loss, loss_config)
    # Same out-of-band route as the loss: nnU-Net's CLI can only pass a trainer name.
    env[sampling_util.SAMPLING_SPEC_ENV] = sampling_util.sampling_spec_payload(
        sampling, temperature=sampling_temperature, oversample_percent=sampling_oversample
    )
    if num_epochs is not None:
        env[loss_util.EPOCHS_ENV] = str(int(num_epochs))

    log.info(
        "stage2 | dataset=%s bundle=%s arch=%s trainer=%s loss=%s mode=%s folds=%s sampling=%s",
        dataset_name, pretrain_name, architecture, trainer, loss, adaptation_mode, list(folds),
        sampling,
    )
    if sampling == "rare_aware" and not (
        paths.nnunet_raw / dataset_name / sampling_util.CASE_CLASSES_FILE
    ).is_file():
        log.warning(
            "--sampling rare_aware needs %s in %s, which stage 0 writes. Without it the "
            "trainer falls back to nnU-Net's default sampling — re-run stage 0.",
            sampling_util.CASE_CLASSES_FILE, paths.nnunet_raw / dataset_name,
        )

    # ---- 1. Baseline plan ---------------------------------------------------
    # preprocess_like_nnssl adapts an existing plan; without one it asserts. The baseline also
    # fixes the target spacing that --adaptation-mode default_nnunet inherits.
    existing = nnunet_run.has_baseline_plans(paths.nnunet_preprocessed, dataset_name)
    if plans_from_label_set:
        log.info(
            "Plans are borrowed from label set %r, so this dataset needs no baseline planning.",
            plans_from_label_set,
        )
    elif skip_planning:
        if existing is None:
            raise FileNotFoundError(
                f"--skip-planning but no baseline plans under "
                f"{paths.nnunet_preprocessed / dataset_name}. Expected one of "
                f"{', '.join(nnunet_run.BASELINE_PLANS)}."
            )
        log.info("Reusing baseline plans: %s", existing)
    elif existing is not None:
        log.info("Baseline plans already present (%s); skipping planning.", existing)
    else:
        nnunet_run.plan_baseline(
            dataset_id, env=env, planner=baseline_planner,
            num_processes=num_processes, verify_integrity=verify_integrity,
        )

    # ---- 2. Plans + preprocessing derived from the pre-training plan --------
    if plans_from_label_set:
        # Borrowed geometry. The source dataset must already have been planned; its plans are
        # copied here unchanged so both networks come out identical, and only then is this
        # dataset's data resampled to them.
        source_dataset = dataset_name_for(plans_from_label_set)
        source_id = DATASET_IDS[plans_from_label_set]
        try:
            plans_identifier = nnunet_run.find_generated_plans(
                paths.nnunet_preprocessed, source_dataset, pretrain_name
            )
        except FileNotFoundError:
            # The borrowing stage usually runs *before* the stage that owns the plans, so
            # generating them here is the normal path, not a fallback. The source dataset is
            # planned exactly as it would plan itself, so the run that owns it later finds
            # its own plans already in place.
            log.info(
                "Label set %r has not been planned yet; planning it now so its geometry can "
                "be shared.", plans_from_label_set,
            )
            if nnunet_run.has_baseline_plans(paths.nnunet_preprocessed, source_dataset) is None:
                nnunet_run.plan_baseline(
                    source_id, env=env, planner=baseline_planner,
                    num_processes=num_processes, verify_integrity=verify_integrity,
                )
            nnunet_run.preprocess_like_nnssl(
                source_id, env=env, pretrain_name=pretrain_name, checkpoint=checkpoint,
                adaptation_mode=adaptation_mode, spacing=target_spacing,
                num_processes=num_processes,
            )
            plans_identifier = nnunet_run.find_generated_plans(
                paths.nnunet_preprocessed, source_dataset, pretrain_name
            )
        borrowed = paths.nnunet_preprocessed / dataset_name / f"{plans_identifier}.json"
        if skip_preprocessing and borrowed.is_file():
            log.info("Plans %s already borrowed into %s; not copying them again.",
                     plans_identifier, dataset_name)
        else:
            log.info("Borrowing plans %s from %s", plans_identifier, source_dataset)
            # The target has never been planned, so nothing has created its preprocessed
            # directory, copied its dataset.json there, or left a fingerprint for the trainer.
            nnunet_run.prepare_borrowed_plans_dataset(source_id, dataset_id, env=env)
            nnunet_run.move_plans(source_id, dataset_id, plans_identifier, env=env)
        if not skip_preprocessing:
            nnunet_run.preprocess_with_plans(
                dataset_id, env=env, plans_identifier=plans_identifier,
                configuration=CONFIGURATION, num_processes=num_processes,
            )
    else:
        if not skip_preprocessing:
            nnunet_run.preprocess_like_nnssl(
                dataset_id, env=env, pretrain_name=pretrain_name, checkpoint=checkpoint,
                adaptation_mode=adaptation_mode, spacing=target_spacing,
                num_processes=num_processes,
            )
        plans_identifier = nnunet_run.find_generated_plans(
            paths.nnunet_preprocessed, dataset_name, pretrain_name
        )
    log.info("Using generated plans: %s", plans_identifier)

    patched = None
    if patch_size is not None or batch_size is not None:
        patched = patch_plans_patch_size(
            paths.nnunet_preprocessed, dataset_name, plans_identifier,
            patch_size=patch_size or json.loads(
                (paths.nnunet_preprocessed / dataset_name / f"{plans_identifier}.json")
                .read_text(encoding="utf-8")
            )["configurations"][CONFIGURATION]["patch_size"],
            batch_size=batch_size,
        )

    # Splits go in after preprocessing created the dataset directory, and are read when a
    # training run starts.
    install_splits(paths.nnunet_raw, paths.nnunet_preprocessed, label_set)

    # nnU-Net validates each finished fold against this folder. preprocess_like_nnssl does not
    # create it, and the failure only surfaces after a fold has fully trained.
    #
    # It belongs to preprocessing -- nnU-Net's own preprocess_dataset writes it as its last
    # step -- so a job told to skip preprocessing skips it too: under --parallel-folds the
    # preparation job has already done it, and four fold jobs refreshing the same directory
    # over NFS is how one of them trips over a file another is replacing. Completeness is still
    # checked, so a genuinely missing tree is caught here rather than eight hours later.
    gt_dir = paths.nnunet_preprocessed / dataset_name / "gt_segmentations"
    expected = len(list((paths.nnunet_raw / dataset_name / "labelsTr").glob("*.nii.gz")))
    present = len(list(gt_dir.glob("*.nii.gz"))) if gt_dir.is_dir() else 0
    if skip_preprocessing and expected and present >= expected:
        log.info("gt_segmentations already complete (%d labels); leaving it untouched.", present)
    else:
        nnunet_run.ensure_gt_segmentations(dataset_id, env=env)

    results_dir = (
        paths.nnunet_results / dataset_name / f"{trainer}__{plans_identifier}__{CONFIGURATION}"
    )
    # Provenance is written before training, not after. A run whose folds are trained by
    # separate jobs needs the complete fold list recorded once, by whoever gets here first,
    # rather than narrowed to the last writer; and a run that dies mid-fold stays discoverable.
    marker_dir = paths.results_root / STAGE2_TRAIN_DIR / dataset_name
    marker = {
        "stage": "stage2",
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name, "label_set": label_set,
        "num_output_channels": lbl.num_foreground(label_set) + 1,
        "bundle": str(bundle), "pretrain_name": pretrain_name,
        "architecture": architecture, "trainer": trainer,
        "loss": loss, "loss_config": loss_config,
        "adaptation_mode": adaptation_mode, "baseline_planner": baseline_planner,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "plans_identifier": plans_identifier, "configuration": CONFIGURATION,
        "patched_plans": patched,
        "plans_from_label_set": plans_from_label_set,
        "pretrained_weights": str(pretrained_weights) if pretrained_weights else None,
        "sampling": sampling, "sampling_temperature": sampling_temperature,
        "sampling_oversample": sampling_oversample,
        "folds": [str(f) for f in folds], "num_epochs": num_epochs,
        "from_scratch": from_scratch, "device": device,
        "results_dir": str(results_dir),
    }
    _write_marker(marker_dir, marker)

    if plan_only:
        log.ok(f"stage2: planning and preprocessing complete (--plan-only) -> {results_dir}")
        return results_dir

    # ---- 3. Fine-tune -------------------------------------------------------
    # The mirror watches the results tree rather than the subprocess, so it picks each fold's
    # directory up as nnU-Net creates it and keeps working across the fold boundary. Wrapping
    # the whole loop (not each fold) means one watcher for the run.
    with tb.monitoring(
        paths, stages=("stage2",), enabled=tensorboard, label_set=label_set,
        event_root=tensorboard_dir, interval=tensorboard_interval,
    ):
        for fold in folds:
            # Resume is decided per fold, not once for the run. A multi-fold job that died
            # part-way has finished folds and untouched ones, and they need opposite treatment:
            # nnU-Net refuses to both continue *and* load initial weights, but a fold that has
            # never trained still needs those weights or it would start from the bundle alone
            # and silently drop out of the curriculum.
            fold_dir = results_dir / f"fold_{fold}"
            resumable = continue_training and any(
                (fold_dir / name).is_file()
                for name in ("checkpoint_final.pth", "checkpoint_latest.pth")
            )
            fold_weights = None if resumable else pretrained_weights
            log.info(
                "--- fold %s --- %s", fold,
                "resuming from its checkpoint" if resumable
                else ("starting fresh" + (
                    f" from {Path(fold_weights).parent.name}" if fold_weights else ""
                )),
            )
            nnunet_run.train_pretrained(
                dataset_id, CONFIGURATION, fold, env=env, trainer=trainer,
                plans_identifier=plans_identifier, device=device, num_gpus=num_gpus,
                continue_training=resumable, from_scratch=from_scratch,
                pretrained_weights=fold_weights,
            )

    # Refresh so the timestamp reflects the finished training rather than its planning.
    marker["created"] = datetime.now().isoformat(timespec="seconds")
    _write_marker(marker_dir, marker)
    log.ok(f"stage2 complete: {len(list(folds))} fold(s) -> {results_dir}")
    return results_dir


def _write_marker(marker_dir: Path, payload: dict[str, Any]) -> None:
    """Write stage 2 provenance, unioning the fold list with any run already recorded there.

    A run's folds may be trained by separate jobs -- ``--parallel-folds``, or five hand-launched
    ``--folds N`` submissions. Each of those writes this file, and a plain overwrite would leave
    the marker naming only whichever job finished last, so stage 4 would quietly predict with a
    single model where five were trained. What is recorded here is what was *requested*;
    ``TrainedModel.available_folds`` narrows it to the folds with a finished checkpoint.
    """
    marker_dir.mkdir(parents=True, exist_ok=True)
    path = marker_dir / "topbrain_stage2.json"
    recorded: list[str] = []
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        # Only a marker for the same run directory has comparable folds. A different
        # trainer/plans combination is a different model that happens to share a dataset, and
        # its folds say nothing about this one.
        if previous.get("results_dir") == payload.get("results_dir"):
            recorded = [str(f) for f in previous.get("folds") or []]

    # A fold with a directory in the run tree belongs to this run whoever launched it. Trusting
    # only the requested list would drop folds trained before this marker existed -- relaunching
    # the remaining folds of a half-finished run would then register fewer folds than are on
    # disk, and stage 4 would ensemble a subset of the models actually trained.
    run_dir = Path(payload.get("results_dir") or "")
    trained = sorted(
        d.name[len("fold_"):] for d in run_dir.glob("fold_*")
        if d.is_dir() and any(d.glob("checkpoint_*.pth"))
    ) if run_dir.is_dir() else []

    folds: list[str] = []
    for fold in [*recorded, *trained, *payload["folds"]]:
        if fold not in folds:
            folds.append(fold)
    payload["folds"] = folds
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_folds(spec: str) -> list[int | str]:
    """Parse ``--folds``: a comma list of fold indices, or ``all``."""
    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    if not tokens:
        raise click.BadParameter("--folds cannot be empty.")
    parsed: list[int | str] = []
    for token in tokens:
        if token.lower() == "all":
            parsed.append("all")
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            raise click.BadParameter(f"Invalid fold {token!r}; expected an integer or 'all'.")
    return parsed


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(**o) -> list[str]:
    """Worker argv for stage 2, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv
    from nvitk.pipes.topbrain.util.paths import STAGE1_PRETRAIN_DIR

    inside = container_layout()
    bundle_name = Path(str(o.get("bundle", ""))).name
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage2_train"),
        *sge_backend_cli_args(o.get("backend", "gpu")),
        "--nnunet-raw", quote_path(inside.nnunet_raw),
        "--nnunet-preprocessed", quote_path(inside.nnunet_preprocessed),
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--results-root", quote_path(inside.results_root),
        # Stage 1 writes the bundle under results_root, which is bind-mounted.
        "--bundle", quote_path(inside.results_root / STAGE1_PRETRAIN_DIR / bundle_name),
        "--label-set", o.get("label_set", "ta36"),
        "--loss", quote_path(o.get("loss", cfg.DEFAULT_LOSS)),
        "--folds", quote_path(",".join(str(f) for f in o.get("folds", (0,)))),
        "--adaptation-mode", o.get("adaptation_mode", "default_nnunet"),
        "--baseline-planner", o.get("baseline_planner", "nnUNetPlannerResEncL"),
        "--device", o.get("device", "cuda"),
        "--num-gpus", str(int(o.get("num_gpus", 1))),
        "--num-processes", str(int(o.get("num_processes", 8))),
    ]
    if o.get("loss_config"):
        argv.extend(["--loss-config", quote_path(str(o["loss_config"]))])
    if o.get("pretrain_name"):
        argv.extend(["--pretrain-name", quote_path(str(o["pretrain_name"]))])
    if o.get("target_spacing"):
        argv.extend(["--target-spacing", *[str(float(s)) for s in o["target_spacing"]]])
    if o.get("patch_size"):
        argv.extend(["--patch-size", *[str(int(p)) for p in o["patch_size"]]])
    if o.get("batch_size"):
        argv.extend(["--batch-size", str(int(o["batch_size"]))])
    if o.get("num_epochs"):
        argv.extend(["--num-epochs", str(int(o["num_epochs"]))])
    for flag, key in (("--skip-planning", "skip_planning"),
                      ("--skip-preprocessing", "skip_preprocessing"),
                      ("--plan-only", "plan_only"),
                      ("--continue-training", "continue_training"),
                      ("--from-scratch", "from_scratch")):
        if o.get(key):
            argv.append(flag)
    for flag, key in (("--plans-from-label-set", "plans_from_label_set"),
                      ("--init-from-label-set", "init_from_label_set")):
        if o.get(key):
            argv.extend([flag, str(o[key])])
    if o.get("pretrained_weights"):
        argv.extend(["--pretrained-weights", quote_path(str(o["pretrained_weights"]))])
    if o.get("sampling") and o["sampling"] != "default":
        argv.extend(["--sampling", str(o["sampling"])])
        if o.get("sampling_temperature") is not None:
            argv.extend(["--sampling-temperature", str(float(o["sampling_temperature"]))])
        if o.get("sampling_oversample") is not None:
            argv.extend(["--sampling-oversample", str(float(o["sampling_oversample"]))])
    if o.get("tensorboard"):
        # No --tensorboard-dir: the worker's results_root is already the container-side mount,
        # so the default layout resolves to the same shared tree the workstation sees.
        argv.append("--tensorboard")
        argv.extend(["--tensorboard-interval", str(float(
            o.get("tensorboard_interval") or tb.DEFAULT_INTERVAL
        ))])
    return argv


def _user_data_paths(**o) -> list[Path]:
    """``--loss-config`` when it names a file rather than carrying inline JSON."""
    raw = str(o.get("loss_config") or "").strip()
    return [Path(raw)] if raw.startswith("/") else []


def _job_suffix(**o) -> str:
    """Job-name suffix. Distinct per fold, so parallel fold jobs get their own log and err file.

    Every job of a stage writes to ``logs/<job name>.log``; with the fold missing from the name,
    concurrent folds interleave into one file and none of them stays readable. The fold is
    appended after truncation so it always survives.
    """
    base = f"{o.get('label_set', '')}_{o.get('loss', '')}".strip("_")[:24]
    folds = list(o.get("folds") or [])
    if o.get("plan_only"):
        return f"{base}_prep".strip("_")
    if len(folds) == 1:
        return f"{base}_f{folds[0]}".strip("_")
    return base


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **o) -> str:
    """Host shell command for the stage 2 SGE task."""
    return build_stage_command(
        o.get("stage_id", "stage2"), _worker_argv(**o), paths=paths, container=container,
        src_dir=src_dir,
        data_paths=_user_data_paths(**o),
        backend=o.get("backend", "gpu"), request_gpu=o.get("device", "cuda") != "cpu",
        pe_smp=o.get("num_processes"),
        job_suffix=_job_suffix(**o),
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **o,
) -> str:
    """Emit or submit the stage 2 SGE job."""
    return submit_stage_job(
        o.get("stage_id", "stage2"), _worker_argv(**o), paths=paths, container=container,
        src_dir=src_dir,
        data_paths=_user_data_paths(**o),
        backend=o.get("backend", "gpu"), request_gpu=o.get("device", "cuda") != "cpu",
        pe_smp=o.get("num_processes"),
        job_suffix=_job_suffix(**o),
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage2-train")
@config_dir_click_option()
@backend_click_option(default="gpu")
@click.option("--nnunet-raw", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-preprocessed", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-results", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--bundle", type=click.Path(path_type=Path), required=True,
              help="Stage 1 bundle directory.")
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr", "binary", "binary_ct", "binary_mr"]), default="ta36",
              show_default=True)
@click.option("--pretrain-name", type=str, default=None,
              help="Name embedded in the generated plans (default: the bundle directory name).")
@click.option("--loss", type=str, default=None,
              help="Registry name or 'package.module:Callable'.")
@click.option("--loss-config", type=str, default=None,
              help="JSON object, or path to a JSON file, of loss keyword arguments.")
@click.option("--folds", type=str, default=None,
              help="Comma list of folds, or 'all'. Defaults to every fold (see --num-folds in "
                   "stage 0): a single fold holds out too few patients to rank anything.")
@click.option("--adaptation-mode", type=click.Choice(list(ADAPTATION_MODES)),
              default="default_nnunet", show_default=True,
              help="How spacing/normalisation are derived from the pre-training plan.")
@click.option("--target-spacing", type=float, nargs=3, default=None,
              help="Required by --adaptation-mode fixed.")
@click.option("--patch-size", type=int, nargs=3, default=None,
              help="Override the plan's 160^3 patch (must be divisible by 32). Needed on cards "
                   "smaller than the A100s the checkpoints were trained on.")
@click.option("--batch-size", type=int, default=None)
@click.option("--num-epochs", type=int, default=None)
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-gpus", type=int, default=1, show_default=True)
@click.option("--num-processes", type=int, default=8, show_default=True)
@click.option("--baseline-planner", type=str, default="nnUNetPlannerResEncL", show_default=True,
              help="Planner for the baseline plan that preprocess_like_nnssl adapts. Its target "
                   "spacing is what --adaptation-mode default_nnunet inherits.")
@click.option("--skip-planning", is_flag=True, default=False,
              help="Reuse an existing baseline plan instead of re-fingerprinting.")
@click.option("--verify-integrity", is_flag=True, default=False,
              help="Run nnU-Net's dataset integrity check during fingerprinting.")
@click.option("--skip-preprocessing", is_flag=True, default=False)
@click.option("--plan-only", is_flag=True, default=False)
@click.option("--continue-training", is_flag=True, default=False)
@click.option("--from-scratch", is_flag=True, default=False,
              help="Same architecture, random initialisation — the control run.")
@click.option("--plans-from-label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr", "binary", "binary_ct", "binary_mr"]),
              default=None,
              help="Borrow this label set's ptPlans instead of planning from this dataset's own "
                   "fingerprint. Required for two runs to be weight-compatible.")
@click.option("--init-from-label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr", "binary", "binary_ct", "binary_mr"]),
              default=None,
              help="Initialise from that label set's finished stage 2 run (everything but the "
                   "segmentation heads). Resolved from its provenance.")
@click.option("--pretrained-weights", type=click.Path(path_type=Path), default=None,
              help="Explicit checkpoint to initialise from; overrides --init-from-label-set.")
@click.option("--sampling", type=click.Choice(list(sampling_util.SAMPLING_MODES)),
              default="default", show_default=True,
              help="Patch sampling. 'rare_aware' oversamples the cases carrying rare classes "
                   "and centres their patches on the rare class; 'default' is nnU-Net's own.")
@click.option("--sampling-temperature", type=float,
              default=sampling_util.DEFAULT_TEMPERATURE, show_default=True,
              help="Inverse-frequency exponent for --sampling rare_aware. 0 is uniform, 1 is "
                   "full inverse-frequency.")
@click.option("--sampling-oversample", type=float, default=None,
              help="Fraction of patches forced to contain foreground under rare_aware "
                   f"(default {sampling_util.DEFAULT_OVERSAMPLE_PERCENT}; nnU-Net uses 0.33).")
@click.option("--tensorboard", is_flag=True, default=False,
              help="Mirror each fold's training log into TensorBoard events under "
                   "<results-root>/tensorboard/stage2/.")
@click.option("--tensorboard-dir", type=click.Path(path_type=Path), default=None,
              help="Event directory (default: <results-root>/tensorboard).")
@click.option("--tensorboard-interval", type=float, default=tb.DEFAULT_INTERVAL,
              show_default=True, help="Seconds between mirror passes.")
def main(
    nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path, results_root: Path,
    bundle: Path, label_set: str, pretrain_name: str | None, loss: str | None,
    loss_config: str | None, folds: str | None, adaptation_mode: str,
    baseline_planner: str, skip_planning: bool, verify_integrity: bool,
    target_spacing: tuple[float, float, float] | None,
    patch_size: tuple[int, int, int] | None, batch_size: int | None, num_epochs: int | None,
    device: str | None, num_gpus: int, num_processes: int, skip_preprocessing: bool,
    plan_only: bool, continue_training: bool, from_scratch: bool,
    plans_from_label_set: str | None, init_from_label_set: str | None,
    pretrained_weights: Path | None,
    sampling: str, sampling_temperature: float, sampling_oversample: float | None,
    tensorboard: bool, tensorboard_dir: Path | None, tensorboard_interval: float,
    backend: str = "gpu",
) -> None:
    """CLI entry point: transfer-learning training with the in-tree nnU-Net build."""
    Logger()
    paths = TopBrainPaths(
        challenge_root=nnunet_raw, nnssl_raw=results_root, nnssl_preprocessed=results_root,
        nnssl_results=results_root, nnunet_raw=nnunet_raw,
        nnunet_preprocessed=nnunet_preprocessed, nnunet_results=nnunet_results,
        results_root=results_root, model_root=results_root, corpus_root=results_root,
    )
    run_train(
        paths=paths, bundle=bundle, label_set=label_set, pretrain_name=pretrain_name,
        loss=loss, loss_config=loss_util.parse_loss_config(loss_config),
        folds=parse_folds(folds or ",".join(str(i) for i in range(int(cfg.DEFAULT_NUM_FOLDS)))),
        adaptation_mode=adaptation_mode,
        baseline_planner=baseline_planner, skip_planning=skip_planning,
        verify_integrity=verify_integrity,
        target_spacing=target_spacing or None, patch_size=patch_size or None,
        batch_size=batch_size, num_epochs=num_epochs,
        device=device or torch_device_for_backend(backend), num_gpus=num_gpus,
        num_processes=num_processes, skip_preprocessing=skip_preprocessing,
        plan_only=plan_only, continue_training=continue_training, from_scratch=from_scratch,
        tensorboard=tensorboard, tensorboard_dir=tensorboard_dir,
        tensorboard_interval=tensorboard_interval,
        sampling=sampling, sampling_temperature=sampling_temperature,
        sampling_oversample=sampling_oversample,
        plans_from_label_set=plans_from_label_set, init_from_label_set=init_from_label_set,
        pretrained_weights=pretrained_weights,
    )


__all__ = [
    "ADAPTATION_MODES", "CONFIGURATION", "build_sge_command", "dataset_name_for",
    "install_splits", "main", "parse_folds", "patch_plans_patch_size", "read_bundle",
    "run_checkpoint", "run_train", "submit_sge",
]


if __name__ == "__main__":
    main()
