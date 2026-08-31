"""Invoke the in-tree nnU-Net build as a subprocess.

The pipeline's nnU-Net lives at ``pipes/topbrain/nnunet`` — a build carrying the nnssl
fine-tuning support (``nnUNetv2_preprocess_like_nnssl``, ``PretrainedTrainer``,
``PretrainedTrainer_Primus``) that the released ``nnunetv2`` does not have.

It is **not installed**, and must not be: the rest of nvitk (TotalSegmentator in particular)
depends on the released ``nnunetv2``, and shadowing that globally would break it. Every call
here therefore runs in a subprocess whose ``PYTHONPATH`` puts the in-tree build first, so the
parent process keeps the released package and only the child sees the in-tree one. That also
gives a long training run process isolation — an OOM-killed fold cannot take the orchestrator
down with it.

Its console scripts are not on ``PATH`` either, so entry points are invoked as ``python -m``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from nvitk.core.logger import Logger

log = Logger()

#: Entry-point modules of the in-tree build, by role.
MODULE_PREPROCESS_LIKE_NNSSL: str = "nnunetv2.experiment_planning.like_nnssl"
MODULE_TRAIN_PRETRAINED: str = "nnunetv2.run.run_training_from_pretrained"
MODULE_PLAN_AND_PREPROCESS: str = "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints"

#: Baseline plans files ``like_nnssl`` will accept, in the order it probes for them. It uses the
#: *first* match, and reads only its target spacing — the architecture comes from the pre-trained
#: checkpoint. So the baseline planner effectively chooses the downstream spacing.
BASELINE_PLANS: tuple[str, ...] = (
    "nnUNetPlans.json",
    "nnUNetResEncUNetPlans.json",
    "nnUNetResEncUNetMPlans.json",
    "nnUNetResEncUNetLPlans.json",
    "nnUNetResEncUNetXLPlans.json",
)
MODULE_PREDICT: str = "nnunetv2.inference.predict_from_raw_data"


def run_module(
    module: str, args: Sequence[str], *, env: dict[str, str], check: bool = True
) -> subprocess.CompletedProcess:
    """Run one in-tree entry-point module with *env* overlaid on the current environment.

    Output streams to this process's stdout/stderr rather than being captured: these commands
    run for minutes to hours, and a progress bar nobody can see is worse than noisy logs.
    """
    argv = [sys.executable, "-m", module, *[str(a) for a in args]]
    log.info("$ %s", " ".join(argv))
    completed = subprocess.run(argv, env={**os.environ, **env}, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"{module} failed with exit code {completed.returncode}.")
    return completed


def has_baseline_plans(nnunet_preprocessed: Path, dataset_name: str) -> str | None:
    """The baseline plans file already present for *dataset_name*, or ``None``."""
    directory = Path(nnunet_preprocessed) / dataset_name
    for name in BASELINE_PLANS:
        if (directory / name).is_file():
            return name
    return None


def plan_baseline(
    dataset_id: int,
    *,
    env: dict[str, str],
    planner: str = "nnUNetPlannerResEncL",
    num_processes: int = 8,
    verify_integrity: bool = False,
) -> None:
    """Fingerprint the dataset and write a baseline plans file.

    ``nnUNetv2_preprocess_like_nnssl`` **adapts** an existing plan rather than creating one — it
    asserts that one of :data:`BASELINE_PLANS` is present, reads its ``3d_fullres`` target
    spacing, and then overwrites the architecture from the pre-trained checkpoint. So this step
    is what actually decides the spacing the model trains at under
    ``--adaptation-mode default_nnunet``.

    Preprocessing is deliberately **not** run here: ``preprocess_like_nnssl`` does its own, keyed
    to the adapted plans' data identifier, and running the baseline preprocessing too would
    write a second copy of the whole dataset that nothing reads.

    Invoked through ``python -c`` rather than ``-m`` because this build's
    ``plan_and_preprocess_entrypoints`` module has no ``__main__`` guard.
    """
    snippet = (
        "from nnunetv2.experiment_planning.plan_and_preprocess_api import "
        "extract_fingerprints, plan_experiments; "
        f"extract_fingerprints([{int(dataset_id)}], num_processes={int(num_processes)}, "
        f"check_dataset_integrity={bool(verify_integrity)}); "
        f"plan_experiments([{int(dataset_id)}], experiment_planner_class_name={planner!r})"
    )
    argv = [sys.executable, "-c", snippet]
    log.info("$ %s -c '<fingerprint + plan, planner=%s>'", sys.executable, planner)
    completed = subprocess.run(argv, env={**os.environ, **env}, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Baseline planning failed with exit code {completed.returncode}.")


def preprocess_like_nnssl(
    dataset_id: int,
    *,
    env: dict[str, str],
    pretrain_name: str,
    checkpoint: Path,
    adaptation_mode: str = "default_nnunet",
    spacing: Sequence[float] | None = None,
    num_processes: int = 4,
) -> None:
    """Generate nnssl-aware plans and preprocess the dataset for them.

    Reads the pre-trained checkpoint's adaptation plan, derives a target spacing and
    normalisation from it according to *adaptation_mode*, writes a ``ptPlans__…`` plans file and
    preprocesses ``3d_fullres`` against it.

    Parameters
    ----------
    adaptation_mode
        ``default_nnunet`` keeps nnU-Net's own planning (recommended here: it picks a
        sub-millimetre spacing suited to the vessels, rather than inheriting the pre-training
        one). ``like_pretrained`` copies the pre-training spacing, ``no_resample`` keeps native
        spacing, ``fixed`` uses *spacing*.
    """
    args: list[str] = [
        "-d", str(dataset_id),
        "-n", pretrain_name,
        "-pc", str(checkpoint),
        "-am", adaptation_mode,
        "-np", str(int(num_processes)),
    ]
    if adaptation_mode == "fixed":
        if spacing is None:
            raise ValueError("--adaptation-mode fixed requires --target-spacing.")
        args.extend(["-spacing", *[str(float(s)) for s in spacing]])
    run_module(MODULE_PREPROCESS_LIKE_NNSSL, args, env=env)


def find_generated_plans(
    nnunet_preprocessed: Path, dataset_name: str, pretrain_name: str
) -> str:
    """Discover the plans identifier ``preprocess_like_nnssl`` just wrote.

    The name is ``ptPlans__<pretrain_name>____<data_identifier>``, and the data identifier
    encodes the *derived* target spacing — so it cannot be constructed in advance and has to be
    found on disk.

    Raises
    ------
    FileNotFoundError
        If nothing matches, naming the directory searched.
    """
    directory = Path(nnunet_preprocessed) / dataset_name
    matches = sorted(directory.glob(f"ptPlans__{pretrain_name}____*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No plans matching 'ptPlans__{pretrain_name}____*.json' under {directory}. "
            f"Did preprocess_like_nnssl run?"
        )
    if len(matches) > 1:
        # Different adaptation modes yield different spacings, hence different names.
        log.warning(
            "%d plans match pre-training name %r; using the newest: %s",
            len(matches), pretrain_name, matches[-1].name,
        )
        matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1].stem


def prepare_borrowed_plans_dataset(
    source_dataset_id: int,
    target_dataset_id: int,
    *,
    env: dict[str, str],
) -> None:
    """Create what a dataset needs in ``nnUNet_preprocessed`` before it can receive plans.

    A dataset that is never planned is missing three things the rest of nnU-Net assumes the
    planner left behind:

    ``nnUNet_preprocessed/<dataset>/``
        ``move_plans_between_datasets`` writes straight into it and does not create it.
    ``dataset.json``
        ``DefaultPreprocessor.run`` reads it from the *preprocessed* folder, not from
        ``nnUNet_raw``. Normally the experiment planner copies it across.
    ``dataset_fingerprint.json``
        ``nnUNetTrainer.on_train_start`` copies it into the results folder, so training aborts
        without it.

    The fingerprint is taken from the **source** dataset rather than extracted afresh. Nothing
    reads the target's own: preprocessing normalises from the intensity statistics baked into
    the borrowed plans, and the trainer only files the fingerprint alongside its outputs. The
    source's is therefore both the cheap answer and the accurate one — it is the fingerprint
    that produced the plans actually in use. Extracting the target's would re-read every volume
    to produce a file no code consults.
    """
    snippet = (
        "import shutil; "
        "from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p; "
        "from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed; "
        "from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name; "
        f"src = convert_id_to_dataset_name({int(source_dataset_id)}); "
        f"tgt = convert_id_to_dataset_name({int(target_dataset_id)}); "
        "out = join(nnUNet_preprocessed, tgt); "
        "maybe_mkdir_p(out); "
        "shutil.copy(join(nnUNet_raw, tgt, 'dataset.json'), join(out, 'dataset.json')); "
        "shutil.copy(join(nnUNet_preprocessed, src, 'dataset_fingerprint.json'), "
        "            join(out, 'dataset_fingerprint.json'))"
    )
    log.info("$ prepare dataset %d to receive plans from %d",
             int(target_dataset_id), int(source_dataset_id))
    completed = subprocess.run([sys.executable, "-c", snippet], env={**os.environ, **env})
    if completed.returncode != 0:
        raise RuntimeError(
            f"Preparing dataset {target_dataset_id} for borrowed plans failed with exit code "
            f"{completed.returncode}."
        )


def move_plans(
    source_dataset_id: int,
    target_dataset_id: int,
    plans_identifier: str,
    *,
    env: dict[str, str],
) -> None:
    """Copy a plans file from one dataset to another, unchanged except for its dataset fields.

    This is what makes two datasets **weight-compatible**. ``load_pretrained_weights`` copies
    parameters by name and shape and asserts they match, so a model trained on dataset A can
    only seed a model on dataset B if both networks were built from the same configuration —
    same target spacing, same patch size, same architecture. Planning each dataset separately
    would derive both from its own fingerprint and produce two subtly different networks.

    The moved plans keep pointing at the same nnssl checkpoint, so the pre-trained encoder is
    still loaded when training the target dataset.

    Invoked through ``python -c`` rather than ``-m``: this build's helper has no ``__main__``.
    """
    snippet = (
        "from nnunetv2.experiment_planning.plans_for_pretraining."
        "move_plans_between_datasets import move_plans_between_datasets; "
        f"move_plans_between_datasets({int(source_dataset_id)}, {int(target_dataset_id)}, "
        f"{plans_identifier!r})"
    )
    log.info("$ move plans %s: dataset %d -> %d",
             plans_identifier, int(source_dataset_id), int(target_dataset_id))
    completed = subprocess.run([sys.executable, "-c", snippet], env={**os.environ, **env})
    if completed.returncode != 0:
        raise RuntimeError(f"Moving plans failed with exit code {completed.returncode}.")


def preprocess_with_plans(
    dataset_id: int,
    *,
    env: dict[str, str],
    plans_identifier: str,
    configuration: str,
    num_processes: int = 8,
) -> None:
    """Preprocess a dataset against an existing plans file.

    The counterpart to :func:`move_plans`: once the target dataset has the source's plans, its
    data has to be resampled and normalised to them. ``preprocess_like_nnssl`` cannot be used
    here — it would derive a *new* plan from the checkpoint and defeat the point.
    """
    snippet = (
        "from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess; "
        f"preprocess([{int(dataset_id)}], plans_identifier={plans_identifier!r}, "
        f"configurations=({configuration!r},), num_processes=({int(num_processes)},))"
    )
    log.info("$ preprocess dataset %d against %s", int(dataset_id), plans_identifier)
    completed = subprocess.run([sys.executable, "-c", snippet], env={**os.environ, **env})
    if completed.returncode != 0:
        raise RuntimeError(f"Preprocessing failed with exit code {completed.returncode}.")


def train_pretrained(
    dataset_id: int,
    configuration: str,
    fold: int | str,
    *,
    env: dict[str, str],
    trainer: str,
    plans_identifier: str,
    device: str = "cuda",
    num_gpus: int = 1,
    continue_training: bool = False,
    from_scratch: bool = False,
    pretrained_weights: Path | str | None = None,
) -> None:
    """Fine-tune from the pre-trained weights the plans file points at.

    Parameters
    ----------
    from_scratch
        Train the same architecture with random initialisation. The control run every
        pre-trained result should be compared against.
    pretrained_weights
        Checkpoint of a **previous nnU-Net run** to initialise from, applied after the plans'
        own nnssl weights so it wins. Every parameter except the ``.seg_layers.`` heads is
        copied, which is what lets a binary vessel model seed a 37-class one.
    """
    args: list[str] = [
        str(dataset_id), configuration, str(fold),
        "-tr", trainer,
        "-p", plans_identifier,
        "-device", device,
        "-num_gpus", str(int(num_gpus)),
    ]
    if continue_training:
        args.append("--c")
    if from_scratch:
        args.append("--from_scratch")
    if pretrained_weights is not None:
        args.extend(["-pretrained_weights", str(pretrained_weights)])
    run_module(MODULE_TRAIN_PRETRAINED, args, env=env)


def predict(
    input_dir: Path,
    output_dir: Path,
    *,
    env: dict[str, str],
    dataset_id: int,
    configuration: str,
    trainer: str,
    plans_identifier: str,
    folds: Sequence[int | str] = (0,),
    device: str = "cuda",
    checkpoint_name: str = "checkpoint_final.pth",
    num_processes: int = 3,
    save_probabilities: bool = False,
) -> None:
    """Run inference over a folder of cases with the in-tree build."""
    args: list[str] = [
        "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(dataset_id), "-c", configuration,
        "-tr", trainer, "-p", plans_identifier,
        "-f", *[str(f) for f in folds],
        "-device", device, "-chk", checkpoint_name,
        "-npp", str(num_processes), "-nps", str(num_processes),
    ]
    if save_probabilities:
        args.append("--save_probabilities")
    run_module(MODULE_PREDICT, args, env=env)


__all__ = [
    "BASELINE_PLANS",
    "MODULE_PLAN_AND_PREPROCESS",
    "MODULE_PREDICT",
    "MODULE_PREPROCESS_LIKE_NNSSL",
    "MODULE_TRAIN_PRETRAINED",
    "find_generated_plans",
    "has_baseline_plans",
    "move_plans",
    "prepare_borrowed_plans_dataset",
    "preprocess_with_plans",
    "plan_baseline",
    "predict",
    "preprocess_like_nnssl",
    "run_module",
    "train_pretrained",
]
