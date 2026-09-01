"""ToPBrain stage 4: inference with topology-aware post-processing.

**Inputs**

- Any trained stage 2 run, selected by name with ``--model`` (see ``--list-models``)
- One or more images: single files, directories, or a mixture

Choosing a model
----------------
``--model ta36`` / ``--model binary`` resolves a run through
:mod:`~nvitk.pipes.topbrain.util.models`, which reads stage 2's provenance for the loss,
trainer, plans identifier, configuration and finished folds. None of those can be reconstructed
from flags — the plans name embeds a spacing chosen at run time — so naming the model is both
shorter and the only reliable way to point at one.

Giving it images
----------------
``--input`` takes files and directories interchangeably and is repeatable, so testing one
volume and predicting a whole cohort are the same command. Anything that is not already a
directory of ``<case>_0000.nii.gz`` is staged into that shape first.

``--modality`` harmonises the inputs exactly as stage 0 did. Leave it off only when the inputs
are *already* harmonised, as ``nnUNet_raw/.../imagesTr`` is — predicting on raw scanner
intensities is silently wrong rather than an error, so the assumption is logged either way.

**Outputs**

- ``<results_root>/stage4_infer/<run>/raw/`` — nnU-Net's argmax predictions
- ``<results_root>/stage4_infer/<run>/postprocessed/`` — after island removal
- ``.../topbrain_stage4.json`` — provenance

Post-processing is applied as a separate pass over a *retained* copy of the raw predictions, so
the effect of a threshold change can be measured (stage 3 can score either directory) without
re-running inference, which is by far the expensive half.

Two tiers of post-processing
----------------------------
``--min-volume-mm3`` / ``--largest-only``
    Island removal — see :mod:`nvitk.segmentation.vessel_postprocess`. Always applied.
``--repair-gaps-mm`` / ``--repair-adjacency`` / ``--repair-lateral``
    Topology repair — see :mod:`nvitk.segmentation.vessel_topology`. Off by default, because
    each is a hypothesis about the failure mode, and three of the six challenge metrics (β0
    error, clDice, invalid neighbours) are sensitive enough to them that they must be **tuned
    on cross-validation** and never on leaderboard feedback. Run stage 3 against the raw and
    postprocessed directories to measure what a setting actually bought.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core.backend import map_in_thread_pool, setup
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.normalization import harmonize_modality
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.stage2_train import resolve_trained_run
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import models
from nvitk.pipes.topbrain.util import nnunet_run
from nvitk.pipes.topbrain.util import paths as pth
from nvitk.pipes.topbrain.util.nnunet_env import nnunet_env
from nvitk.pipes.topbrain.util.paths import DATASET_IDS, DATASET_SUFFIXES, STAGE4_INFER_DIR, TopBrainPaths
from nvitk.pipes.topbrain.util.sge_backend import sge_backend_cli_args, torch_device_for_backend
from nvitk.pipes.topbrain.util.sge_stage import (
    build_stage_command,
    container_layout,
    quote_path,
    submit_stage_job,
)
from nvitk.segmentation.vessel_postprocess import postprocess_labelmap
from nvitk.segmentation.vessel_topology import RepairReport, repair_topology

setup(globals())

log = Logger()


def _dataset_name(label_set: str) -> str:
    """nnU-Net dataset folder name for *label_set*."""
    return f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"


#: ``..._ct_...`` / ``..._mr_...`` — how every cohort this pipeline reads names its modality.
_MODALITY_TOKEN = re.compile(r"(?:^|_)(ct|mr)(?:_|$)")


def infer_modality(name: str) -> str:
    """Read ``ct``/``mr`` out of a filename.

    Only used for ``--modality auto``. Deliberately strict: harmonisation applies a Hounsfield
    window to CT and robust percentiles to MR, and getting that backwards produces a
    plausible-looking volume that is completely wrong, so an unrecognisable name raises rather
    than defaulting.

    Raises
    ------
    ValueError
        Naming the file and asking for an explicit ``--modality``.
    """
    match = _MODALITY_TOKEN.search(Path(name).name.lower())
    if match is None:
        raise ValueError(
            f"Cannot tell the modality of {name!r} from its name. Pass --modality ct or "
            f"--modality mr explicitly; guessing would risk applying an HU window to MR."
        )
    return match.group(1)


def expand_inputs(inputs: Sequence[Path]) -> list[Path]:
    """Flatten files and directories into a sorted list of volumes.

    A directory contributes its ``*.nii.gz`` (non-recursively); a file contributes itself. This
    is what lets one image and a whole cohort go through the same code path.

    Raises
    ------
    FileNotFoundError
        If nothing matched, naming what was searched.
    """
    found: list[Path] = []
    for entry in inputs:
        path = Path(entry).expanduser()
        if path.is_dir():
            found.extend(sorted(path.glob("*.nii.gz")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    if not found:
        raise FileNotFoundError(
            f"No .nii.gz volumes found in: {', '.join(str(Path(i)) for i in inputs)}"
        )
    return found


def case_id_for(path: Path) -> str:
    """Case id of a volume: its name without ``.nii.gz`` and without an ``_0000`` channel tag."""
    stem = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem[: -len("_0000")] if stem.endswith("_0000") else stem


def stage_inputs(
    inputs: Sequence[Path],
    workspace: Path,
    *,
    modality: str | None = None,
    expected_channels: int = 1,
    ct_window: Sequence[float] | None = None,
    mr_percentiles: Sequence[float] | None = None,
) -> tuple[Path, list[str]]:
    """Normalise arbitrary inputs into a folder nnU-Net's predictor can read.

    nnU-Net insists on a directory of ``<case>_0000.nii.gz``. Anything else — one file, a
    folder of plainly-named volumes, a mixture — is staged into *workspace* under that
    convention, so a single image and a whole cohort are the same operation.

    Parameters
    ----------
    modality
        ``ct``/``mr`` to harmonise the inputs exactly as stage 0 did, or ``auto`` to read it
        per file from the name. ``None`` means the inputs are **already harmonised** — which is
        true of ``nnUNet_raw/.../imagesTr`` and false of anything straight off a scanner.
        Predicting on unharmonised intensities is silently wrong rather than an error, so the
        choice is logged either way.
    expected_channels
        Input channels the model wants. Staging can only ever produce channel 0, so a
        multi-channel model with unstaged inputs is refused rather than fed a missing channel.

    Returns
    -------
    tuple
        ``(directory, case_ids)`` — the directory to hand the predictor, and the cases in it.
    """
    volumes = expand_inputs(inputs)

    # ---- Fast path: already exactly what the predictor wants -----------------
    already_named = all(v.name.endswith("_0000.nii.gz") for v in volumes)
    single_dir = len(inputs) == 1 and Path(inputs[0]).is_dir()
    if modality is None and already_named and single_dir:
        log.info(
            "Predicting directly on %d volume(s) in %s (assumed already harmonised).",
            len(volumes), inputs[0],
        )
        return Path(inputs[0]), [case_id_for(v) for v in volumes]

    if expected_channels > 1:
        raise ValueError(
            f"This model expects {expected_channels} input channels, which staging cannot "
            f"build from single files — channel 1 is a second intensity window produced by "
            f"stage 0. Point --input at a prepared {expected_channels}-channel dataset "
            f"directory instead."
        )

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    ct_window = tuple(ct_window) if ct_window else cfg.DEFAULT_CT_WINDOW
    mr_percentiles = tuple(mr_percentiles) if mr_percentiles else cfg.DEFAULT_MR_PERCENTILES

    seen: dict[str, Path] = {}
    for volume in volumes:
        case = case_id_for(volume)
        if case in seen:
            raise ValueError(
                f"Two inputs map to the same case id {case!r}: {seen[case]} and {volume}. "
                f"Predictions would overwrite each other."
            )
        seen[case] = volume
        destination = workspace / f"{case}_0000.nii.gz"
        if modality is None:
            shutil.copyfile(volume, destination)
            continue
        resolved = infer_modality(volume.name) if modality == "auto" else modality
        image = imread(volume)
        harmonised = harmonize_modality(
            image, resolved, ct_window=ct_window, mr_percentiles=mr_percentiles
        )
        imsave(destination, harmonised.astype(np.float32))
        log.step(f"{case}: harmonised as {resolved}")

    if modality is None:
        log.warning(
            "Staged %d volume(s) without harmonisation. The model was trained on stage 0's "
            "harmonised intensities; pass --modality ct|mr|auto if these came off a scanner.",
            len(volumes),
        )
    log.ok(f"staged {len(seen)} case(s) -> {workspace}")
    return workspace, sorted(seen)


def postprocess_folder(
    source_dir: Path,
    destination_dir: Path,
    *,
    label_set: str,
    min_volume_mm3: float | None,
    largest_only: bool,
    workers: int = 1,
    repair_gaps_mm: float | None = None,
    repair_adjacency: bool = False,
    repair_lateral: bool = False,
    repair_close_radius: int = 0,
    repair_fragment_fraction: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """Post-process every prediction in *source_dir*; returns ``(count, repair_summary)``.

    Island removal runs first and unconditionally: the topology steps reason about connected
    components, and speckle would give them spurious ones to reason about. The repair steps
    then run in the order :func:`~nvitk.segmentation.vessel_topology.repair_topology` fixes.

    Parameters
    ----------
    repair_gaps_mm
        Bridge same-class gaps up to this many millimetres. ``None`` disables.
    repair_adjacency
        Reassign fragments touching anatomically impossible labels, using this label set's
        adjacency table. Note that TA36's table is *derived*, not published — see
        :func:`nvitk.pipes.topbrain.labels.valid_neighbours` — so this repairs against our own
        reading of the anatomy.
    repair_lateral
        Mirror fragments of a lateralised class found on the wrong side of the midline.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(lbl.label_map(label_set))
    cases = sorted(source_dir.glob("*.nii.gz"))
    if not cases:
        log.warning("No predictions to post-process under %s", source_dir)
        return 0, {}

    neighbours = lbl.valid_neighbours(label_set) if repair_adjacency else None
    pairs = lbl.lateral_pairs(label_set) if repair_lateral else None
    repairing = repair_gaps_mm is not None or repair_adjacency or repair_lateral
    if repairing:
        log.info(
            "topology repair | gaps=%s adjacency=%s lateral=%s (%d mirrored pair(s))",
            f"{repair_gaps_mm} mm" if repair_gaps_mm is not None else "off",
            repair_adjacency, repair_lateral, len(pairs or {}),
        )

    extra = (
        {"max_fragment_fraction": float(repair_fragment_fraction)}
        if repair_fragment_fraction is not None else {}
    )

    def _one(path: Path) -> tuple[Path, RepairReport | None]:
        """Post-process one prediction, preserving its geometry."""
        image = imread(path)
        cleaned = postprocess_labelmap(
            image,
            labels=labels,
            spacing=image.spacing,
            min_volume_mm3=min_volume_mm3,
            largest_only=largest_only,
        )
        report = None
        if repairing:
            cleaned, report = repair_topology(
                cleaned,
                labels=labels,
                spacing=image.spacing,
                affine=image.affine,
                valid_neighbours=neighbours,
                lateral_pairs=pairs,
                bridge_gaps_mm=repair_gaps_mm,
                close_radius=int(repair_close_radius),
                **extra,
            )
        out = destination_dir / path.name
        imsave(out, cleaned.astype(np.uint8))
        return out, report

    results = map_in_thread_pool(_one, cases, max_workers=int(workers))
    summary: dict[str, Any] = {}
    if repairing:
        reports = [r for _, r in results if r is not None]
        summary = {
            "cases": {
                path.name[: -len(".nii.gz")]: report.as_dict()
                for path, report in results if report is not None
            },
            "totals": {
                key: sum(int(r.as_dict()[key]) for r in reports)
                for key in ("bridged_voxels", "reassigned_components", "reassigned_voxels",
                            "mirrored_components", "mirrored_voxels")
            },
        }
        log.ok("topology repair totals: %s", summary["totals"])
    log.ok(f"post-processed {len(results)} prediction(s) -> {destination_dir}")
    return len(results), summary


def model_input_channels(model: models.TrainedModel, nnunet_results: Path) -> int:
    """Input channels the trained model expects, from the ``dataset.json`` it saved.

    The trainer copies ``dataset.json`` into its results folder precisely so inference can
    recover this without the raw dataset being present.
    """
    path = model.run_dir(nnunet_results) / "dataset.json"
    try:
        return len(json.loads(path.read_text(encoding="utf-8"))["channel_names"])
    except (OSError, ValueError, KeyError):
        log.debug("No channel_names in %s; assuming a single input channel.", path)
        return 1


def run_infer(
    *,
    input_dir: Path | None = None,
    inputs: Sequence[Path] = (),
    model: str | None = None,
    modality: str | None = None,
    output: Path | None = None,
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    nnunet_results: Path,
    results_root: Path,
    label_set: str = "ta36",
    loss: str | None = None,
    architecture: str | None = None,
    folds: Sequence[int | str] = (),
    plans_identifier: str | None = None,
    configuration_name: str | None = None,
    checkpoint_name: str = "checkpoint_final.pth",
    output_name: str | None = None,
    min_volume_mm3: float | None = 5.0,
    largest_only: bool = False,
    repair_gaps_mm: float | None = None,
    repair_adjacency: bool = False,
    repair_lateral: bool = False,
    repair_close_radius: int = 0,
    device: str = "cuda",
    num_processes: int = 3,
    workers: int = 1,
    skip_prediction: bool = False,
) -> Path:
    """Predict and post-process; returns the post-processed output directory.

    Parameters
    ----------
    inputs
        Volumes and/or directories to predict on. One file and a whole cohort are the same
        operation — see :func:`stage_inputs`.
    model
        Which trained model to use: a label set (``ta36``, ``binary``), a dataset folder name,
        or a path to a stage 2 marker. Resolved through
        :mod:`~nvitk.pipes.topbrain.util.models`, which recovers the loss, trainer, plans and
        folds from provenance so none of them has to be repeated on the command line.
    modality
        Harmonise the inputs before predicting — see :func:`stage_inputs`.
    output
        Write the single prediction here. Only valid for exactly one input case.
    """
    selected: models.TrainedModel | None = None
    if model is not None:
        selected = models.resolve_model(results_root, model)
        # Everything about the run comes from its provenance; anything passed explicitly still
        # wins, so a one-off override is possible without editing the marker.
        label_set = selected.label_set
        loss = loss or selected.loss
        plans_identifier = plans_identifier or selected.plans_identifier
        configuration_name = configuration_name or selected.configuration
        architecture = architecture or selected.architecture
        log.info(
            "model %r | dataset=%s loss=%s classes=%d trained=%s",
            selected.label_set, selected.dataset, selected.loss,
            selected.num_output_channels, selected.created,
        )

    loss = loss or cfg.DEFAULT_LOSS
    # Plans name, architecture and configuration all come from what stage 2 actually trained.
    # None of them has a usable default: the plans name embeds a spacing chosen at run time,
    # and the architecture decides the trainer family.
    resolved = resolve_trained_run(
        results_root, label_set, plans_identifier=plans_identifier,
        architecture=architecture, configuration_name=configuration_name,
    )
    plans_identifier = resolved["plans_identifier"]
    architecture = resolved["architecture"]
    configuration_name = resolved["configuration"]
    trainer = loss_util.trainer_for_loss(loss, architecture=architecture)
    dataset_name = _dataset_name(label_set)
    run_name = output_name or f"{trainer}__{plans_identifier}__{configuration_name}"

    base = Path(results_root) / STAGE4_INFER_DIR / run_name
    raw_dir = base / "raw"
    post_dir = base / "postprocessed"

    # ---- Folds: default to the ones that actually finished --------------------
    if selected is not None and not folds:
        folds = selected.available_folds(nnunet_results) or list(selected.folds)
        log.info("Using fold(s) %s from the model's provenance.", ", ".join(str(f) for f in folds))
    folds = list(folds) or [0]

    # ---- Inputs: one file, several files, folders, or all three ---------------
    all_inputs = [*(inputs or ()), *([input_dir] if input_dir is not None else [])]
    if not all_inputs:
        raise ValueError("No input given: pass --input with one or more files or directories.")
    channels = (
        model_input_channels(selected, nnunet_results) if selected is not None else 1
    )
    predict_dir, case_ids = stage_inputs(
        all_inputs, base / "input", modality=modality, expected_channels=channels,
    )
    if output is not None and len(case_ids) != 1:
        raise ValueError(
            f"--output names a single file but {len(case_ids)} case(s) were given. Drop it and "
            f"collect the predictions from the output directory instead."
        )

    paths = TopBrainPaths(
        # Only the nnU-Net roots matter to nnunet_env(); challenge_root just has to be a real
        # path, and the staged input directory is the honest one to name here.
        challenge_root=Path(predict_dir),
        nnssl_raw=Path(results_root), nnssl_preprocessed=Path(results_root),
        nnssl_results=Path(results_root),
        nnunet_raw=Path(nnunet_raw),
        nnunet_preprocessed=Path(nnunet_preprocessed),
        nnunet_results=Path(nnunet_results),
        results_root=Path(results_root), model_root=Path(results_root),
        corpus_root=Path(results_root),
    )
    env = nnunet_env(paths, num_processes=num_processes)
    # Inference rebuilds the trainer class to recover the architecture, so the loss trainers
    # must be discoverable here exactly as they were during training.
    env[loss_util.LOSS_SPEC_ENV] = loss_util.loss_spec_payload(loss, {})

    log.info(
        "topbrain stage5 | dataset=%s trainer=%s folds=%s -> %s",
        dataset_name, trainer, list(folds), base,
    )

    if not skip_prediction:
        raw_dir.mkdir(parents=True, exist_ok=True)
        nnunet_run.predict(
            predict_dir, raw_dir,
            env=env,
            dataset_id=DATASET_IDS[label_set],
            configuration=configuration_name,
            trainer=trainer,
            plans_identifier=plans_identifier,
            folds=folds,
            device=device,
            checkpoint_name=checkpoint_name,
            num_processes=num_processes,
        )

    count, repair_summary = postprocess_folder(
        raw_dir, post_dir,
        label_set=label_set,
        min_volume_mm3=min_volume_mm3,
        largest_only=largest_only,
        workers=workers,
        repair_gaps_mm=repair_gaps_mm,
        repair_adjacency=repair_adjacency,
        repair_lateral=repair_lateral,
        repair_close_radius=repair_close_radius,
    )

    if output is not None:
        produced = post_dir / f"{case_ids[0]}.nii.gz"
        if not produced.is_file():
            raise FileNotFoundError(
                f"No prediction at {produced} to copy to --output. "
                + ("--skip-prediction was set, so nothing was predicted."
                   if skip_prediction else "The predictor produced nothing for this case.")
            )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, output)
        log.ok(f"prediction -> {output}")

    (base / "topbrain_stage4.json").write_text(
        json.dumps(
            {
                "stage": "stage4",
                "created": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_name, "label_set": label_set,
                "trainer": trainer, "loss": loss,
                "plans_identifier": plans_identifier, "configuration": configuration_name,
                "folds": [str(f) for f in folds], "checkpoint": checkpoint_name,
                "inputs": [str(i) for i in all_inputs],
                "predict_dir": str(predict_dir), "cases": case_ids,
                "modality": modality, "model": selected.label_set if selected else None,
                "num_input_channels": channels,
                "raw_dir": str(raw_dir), "postprocessed_dir": str(post_dir),
                "min_volume_mm3": min_volume_mm3, "largest_only": largest_only,
                "repair_gaps_mm": repair_gaps_mm, "repair_adjacency": repair_adjacency,
                "repair_lateral": repair_lateral,
                "repair_close_radius": repair_close_radius,
                "topology_repair": repair_summary,
                "num_cases": count, "device": device,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return post_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(
    *, label_set: str, loss: str, folds: Sequence[int | str], plans_identifier: str | None,
    configuration_name: str | None, checkpoint_name: str, min_volume_mm3: float | None,
    largest_only: bool, device: str, num_processes: int, workers: int,
    skip_prediction: bool, backend: str, input_subdir: str = "imagesTr_topbrain",
    repair_gaps_mm: float | None = None, repair_adjacency: bool = False,
    repair_lateral: bool = False, repair_close_radius: int = 0,
) -> list[str]:
    """Worker argv for stage 5, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage4_infer"),
        *sge_backend_cli_args(backend),
        "--input-dir", quote_path(inside.challenge_root / input_subdir),
        "--nnunet-raw", quote_path(inside.nnunet_raw),
        "--nnunet-preprocessed", quote_path(inside.nnunet_preprocessed),
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--results-root", quote_path(inside.results_root),
        "--label-set", label_set,
        "--loss", quote_path(loss),
        "--checkpoint-name", checkpoint_name,
        "--device", device,
        "--num-processes", str(int(num_processes)),
        "--workers", str(int(workers)),
    ]
    # Omitted when unknown: the worker reads them from stage 2's provenance, which is the only
    # place that knows the spacing preprocess_like_nnssl settled on at run time. Emitting a
    # literal "None" would build a run directory name that does not exist.
    if folds:
        argv.extend(["--folds", quote_path(",".join(str(f) for f in folds))])
    if plans_identifier:
        argv.extend(["--plans-identifier", quote_path(plans_identifier)])
    if configuration_name:
        argv.extend(["--configuration", quote_path(configuration_name)])
    if min_volume_mm3 is None:
        argv.append("--no-postprocess")
    else:
        argv.extend(["--min-volume-mm3", str(float(min_volume_mm3))])
    if largest_only:
        argv.append("--largest-only")
    if repair_gaps_mm is not None:
        argv.extend(["--repair-gaps-mm", str(float(repair_gaps_mm))])
    if repair_adjacency:
        argv.append("--repair-adjacency")
    if repair_lateral:
        argv.append("--repair-lateral")
    if repair_close_radius:
        argv.extend(["--repair-close-radius", str(int(repair_close_radius))])
    if skip_prediction:
        argv.append("--skip-prediction")
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **options) -> str:
    """Host shell command for the stage 5 SGE task."""
    return build_stage_command(
        "stage4", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "gpu"),
        request_gpu=options.get("device", "cuda") != "cpu",
        job_suffix=options.get("label_set", ""),
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **options,
) -> str:
    """Emit or submit the stage 5 SGE job."""
    return submit_stage_job(
        "stage4", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "gpu"),
        request_gpu=options.get("device", "cuda") != "cpu",
        job_suffix=options.get("label_set", ""),
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage4-infer")
@config_dir_click_option()
@backend_click_option(default="gpu")
@click.option("-i", "--input", "inputs", multiple=True, type=click.Path(path_type=Path),
              help="Image file or directory to predict on. Repeatable, and files and folders "
                   "can be mixed — one volume and a whole cohort work the same way.")
@click.option("--input-dir", type=click.Path(path_type=Path), default=None,
              help="Folder of <case>_0000.nii.gz images (equivalent to --input on a folder).")
@click.option("--model", type=str, default=None,
              help="Which trained model to use: a label set ('ta36', 'binary'), a dataset "
                   "folder name, or a path to a stage 2 marker. Its loss, trainer, plans and "
                   "folds are read from provenance. See --list-models.")
@click.option("--list-models", is_flag=True, default=False,
              help="Print the trained models found under --results-root and exit.")
@click.option("--modality", type=click.Choice(["ct", "mr", "auto"]), default=None,
              help="Harmonise the inputs as stage 0 did before predicting. Omit only when the "
                   "inputs are already harmonised (e.g. nnUNet_raw/.../imagesTr).")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="Write the prediction to this file. Single input only.")
@click.option("--nnunet-raw", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths (see --config-dir).")
@click.option("--nnunet-preprocessed", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths.")
@click.option("--nnunet-results", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths.")
@click.option("--results-root", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths.")
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr", "binary", "binary_ct", "binary_mr"]),
              default="ta36", show_default=True,
              help="Ignored when --model is given; the model records its own.")
@click.option("--loss", type=str, default=None, help="Loss the model was trained with.")
@click.option("--architecture", type=str, default=None,
              help="Encoder family (ResEncL / PrimusM). Read from stage 2 when omitted.")
@click.option("--folds", type=str, default=None,
              help="Folds to ensemble. Defaults to the finished folds recorded by --model.")
@click.option("--plans-identifier", type=str, default=None)
@click.option("--configuration", "configuration_name", type=str, default=None)
@click.option("--checkpoint-name", type=str, default="checkpoint_final.pth", show_default=True)
@click.option("--output-name", type=str, default=None)
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True,
              help="Drop connected components smaller than this, per class.")
@click.option("--no-postprocess", is_flag=True, default=False)
@click.option("--largest-only", is_flag=True, default=False,
              help="Also reduce each class to its single largest component.")
@click.option("--repair-gaps-mm", type=float, default=None,
              help="Bridge same-class gaps up to this many mm. Targets the beta0 and "
                   "centerline metrics; tune it on cross-validation, not on the leaderboard.")
@click.option("--repair-adjacency", is_flag=True, default=False,
              help="Reassign fragments touching anatomically impossible labels. TA36's "
                   "adjacency table is derived rather than published — see labels.py.")
@click.option("--repair-lateral", is_flag=True, default=False,
              help="Mirror fragments of a lateralised class found on the wrong side of the "
                   "midline. Declines to act when the left/right convention is unreadable.")
@click.option("--repair-close-radius", type=int, default=0, show_default=True,
              help="Per-class morphological closing before bridging. Helps beta0, costs "
                   "Dice/clDice precision.")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-processes", type=int, default=3, show_default=True)
@click.option("--workers", type=int, default=1, show_default=True)
@click.option("--skip-prediction", is_flag=True, default=False,
              help="Only re-run post-processing over existing raw predictions.")
def main(
    inputs: tuple[Path, ...], input_dir: Path | None, model: str | None, list_models: bool,
    modality: str | None, output: Path | None,
    nnunet_raw: Path | None, nnunet_preprocessed: Path | None, nnunet_results: Path | None,
    results_root: Path | None, label_set: str, loss: str | None, architecture: str | None,
    folds: str | None,
    plans_identifier: str | None, configuration_name: str | None, checkpoint_name: str,
    output_name: str | None, min_volume_mm3: float, no_postprocess: bool, largest_only: bool,
    repair_gaps_mm: float | None, repair_adjacency: bool, repair_lateral: bool,
    repair_close_radius: int,
    device: str | None, num_processes: int, workers: int, skip_prediction: bool,
    backend: str = "gpu",
) -> None:
    """CLI entry point: predict on one or more images with a selected model."""
    from nvitk.pipes.topbrain.stage2_train import parse_folds

    Logger()
    # Interactive use should not have to repeat four roots that sge.json already knows. The
    # cluster roots win when they are mounted here, because that is where trained models live —
    # the local_* roots are a separate working copy. Anything passed explicitly still wins, and
    # the SGE worker passes all four (container paths), so its behaviour is unchanged.
    given = {
        "nnunet_raw": nnunet_raw, "nnunet_preprocessed": nnunet_preprocessed,
        "nnunet_results": nnunet_results, "results_root": results_root,
    }
    if any(v is None for v in given.values()):
        layout, origin = pth.layout_auto()
        filled = [k for k, v in given.items() if v is None]
        nnunet_raw = nnunet_raw or layout.nnunet_raw
        nnunet_preprocessed = nnunet_preprocessed or layout.nnunet_preprocessed
        nnunet_results = nnunet_results or layout.nnunet_results
        results_root = results_root or layout.results_root
        log.info(
            "Took %s from sge.json's %s roots; results_root=%s",
            ", ".join(filled), origin, results_root,
        )

    if list_models:
        click.echo(models.describe_models(results_root, nnunet_results))
        return
    if not inputs and input_dir is None:
        raise click.UsageError("Give at least one --input file or directory (or --input-dir).")
    run_infer(
        inputs=list(inputs), input_dir=input_dir, model=model, modality=modality,
        output=output,
        nnunet_raw=nnunet_raw, nnunet_preprocessed=nnunet_preprocessed,
        nnunet_results=nnunet_results, results_root=results_root, label_set=label_set,
        # Left to stage 2's provenance unless overridden: it is the only place that records
        # which checkpoint family was actually fine-tuned.
        loss=loss, architecture=architecture, folds=parse_folds(folds) if folds else (),
        plans_identifier=plans_identifier,
        configuration_name=configuration_name, checkpoint_name=checkpoint_name,
        output_name=output_name,
        min_volume_mm3=None if no_postprocess else min_volume_mm3,
        largest_only=largest_only,
        repair_gaps_mm=repair_gaps_mm, repair_adjacency=repair_adjacency,
        repair_lateral=repair_lateral, repair_close_radius=repair_close_radius,
        device=device or torch_device_for_backend(backend),
        num_processes=num_processes, workers=workers, skip_prediction=skip_prediction,
    )


__all__ = [
    "build_sge_command", "case_id_for", "expand_inputs", "infer_modality", "main",
    "model_input_channels", "postprocess_folder", "run_infer", "stage_inputs", "submit_sge",
]


if __name__ == "__main__":
    main()
