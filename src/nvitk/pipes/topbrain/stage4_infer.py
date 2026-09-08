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
from nvitk.core.array import to_numpy
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
import posixpath

from nvitk.pipes.topbrain.util import postproc
from nvitk.pipes.topbrain.util import staging
from nvitk.pipes.topbrain.util.staging import StagedTransfer
from nvitk.segmentation.vessel_postprocess import postprocess_labelmap
from nvitk.segmentation.vessel_topology import RepairReport, repair_topology

setup(globals())

log = Logger()


def _dataset_name(label_set: str) -> str:
    """nnU-Net dataset folder name for *label_set*."""
    return f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"


#: ``..._ct_...`` / ``..._mr_...`` — how every cohort this pipeline reads names its modality.
_MODALITY_TOKEN = re.compile(r"(?:^|_)(ct|mr)(?:_|$)")


#: A CT volume in Hounsfield units puts air near -1000. Nothing else this pipeline sees goes
#: that low: MR and TOF are arbitrary non-negative units, and even a z-scored volume rarely
#: reaches -5. The gap between -1000 and -5 is what makes the test safe rather than clever.
CT_AIR_HU: float = -500.0

#: Fraction of voxels that must sit below :data:`CT_AIR_HU` before a volume is called CT. A head
#: CT is mostly surrounding air, so this is comfortably exceeded; a stray negative outlier in an
#: MR volume is not.
CT_AIR_FRACTION: float = 0.05


def verify_modality(volumes: Sequence[Path], required: str) -> list[tuple[str, str]]:
    """Volumes whose measured modality is not *required*, as ``(name, found)`` pairs.

    Volumes whose modality cannot be measured are **not** reported: absence of evidence is not
    evidence of a mismatch, and refusing them would block every already-normalised cohort.
    """
    wrong: list[tuple[str, str]] = []
    for volume in volumes:
        try:
            found = detect_modality(imread(volume))
        except Exception:  # noqa: BLE001 - unreadable here means "no evidence"
            continue
        if found is not None and found != required:
            wrong.append((Path(volume).name, found))
    return wrong


def detect_modality(volume: Any) -> str | None:
    """Read the modality off the intensities; ``None`` when they do not say.

    CT is calibrated: air is about -1000 HU and a head scan is mostly air, so a large negative
    population is decisive. MR and TOF carry arbitrary positive units with no such population.
    This is physical evidence, unlike a filename, which is a convention that a cohort like
    ``pesa_tof/<subject>/TOF.nii.gz`` does not follow at all.

    Returns ``None`` rather than guessing when the volume is neither clearly calibrated nor
    clearly non-negative -- an already-normalised volume, say -- so the caller can fall back to
    the name or ask.
    """
    data = to_numpy(volume.data if hasattr(volume, "data") else volume)
    if data.size == 0:
        return None
    sample = data.ravel()
    if sample.size > 2_000_000:  # a few million voxels decide this as well as 200 million
        sample = sample[:: sample.size // 2_000_000]
    below = float((sample < CT_AIR_HU).mean())
    if below >= CT_AIR_FRACTION and float(sample.min()) < -900.0:
        return "ct"
    if float(sample.min()) >= -1.0:
        return "mr"
    return None


def resolve_volume_modality(path: Path, *, declared: str | None = None) -> str:
    """The modality of one volume: measured first, named second, declared last.

    Intensities outrank the filename because they cannot be renamed by accident. When both are
    readable and they disagree, the mismatch is reported rather than silently resolved -- it
    means either the file is misnamed or it is not the modality anyone thinks it is.
    """
    measured = None
    try:
        measured = detect_modality(imread(path))
    except Exception:  # noqa: BLE001 - unreadable here means "no evidence", not a failure
        measured = None
    named = None
    try:
        named = infer_modality(path.name)
    except ValueError:
        named = None

    if measured and named and measured != named:
        log.warning(
            "%s: the intensities look like %s but the name says %s. Trusting the intensities.",
            path.name, measured.upper(), named.upper(),
        )
    resolved = measured or named or declared
    if resolved is None:
        raise ValueError(
            f"Cannot tell the modality of {path.name!r}: its intensities are neither "
            f"Hounsfield-calibrated nor plainly non-negative, and its name carries no ct/mr "
            f"token. Pass --modality ct or --modality mr explicitly."
        )
    return resolved


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
        resolved = (
            resolve_volume_modality(volume) if modality == "auto" else modality
        )
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
    spec: Any = None,
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

    if spec is not None:
        # A selection given explicitly wins over the individual flags: it is what stage 3
        # measured and what stage 5 will bake into the container, and having two ways to say
        # the same thing is how those three drift apart.
        repair_gaps_mm = spec.bridge_gaps_mm if spec.has("bridge") else None
        repair_adjacency = spec.has("adjacency")
        repair_lateral = spec.has("lateral")
        repair_close_radius = spec.close_radius
        min_volume_mm3 = spec.min_volume_mm3 if spec.has("islands") else None
        largest_only = spec.has("largest")
        log.info("%s", spec.describe())
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
    postprocess: str | None = None,
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

    # ---- Modality: the model's label set outranks the flag --------------------
    # A modality-specific model harmonising the other modality is silently wrong, not an error:
    # the numbers come out, they are just computed from intensities normalised the wrong way.
    # Leaving --modality unset used to mean "already harmonised", which is false for anything
    # straight off a scanner, so a single-modality model now fills it in.
    declared = lbl.LABEL_SET_MODALITIES.get(label_set, ("ct", "mr"))
    if len(declared) == 1:
        required = declared[0]
        if modality not in (None, "auto", required):
            raise ValueError(
                f"Model {label_set!r} is {required.upper()}-only, but --modality {modality!r} "
                f"was given. Harmonising {modality.upper()} data as {required.upper()} would "
                f"produce plausible nonsense; use a model that covers it."
            )
        # Measured, not assumed: a mono-modality model fed the other modality is the failure
        # that produces plausible nonsense rather than an error, so the inputs are checked
        # against what they actually contain.
        wrong = verify_modality(expand_inputs(all_inputs), required)
        if wrong:
            raise ValueError(
                f"Model {label_set!r} is {required.upper()}-only, but "
                f"{len(wrong)} input(s) are not: "
                + ", ".join(f"{name} ({found})" for name, found in wrong[:5])
                + (" ..." if len(wrong) > 5 else "")
                + ". Use a model that covers them, or drop them from the input."
            )
        if modality is None:
            log.info("Model %r is %s-only and the inputs agree; harmonising as %s.",
                     label_set, required.upper(), required.upper())
        modality = required

    channels = (
        model_input_channels(selected, nnunet_results) if selected is not None else 1
    )
    predict_dir, case_ids = stage_inputs(
        all_inputs, base / "input", modality=modality, expected_channels=channels,
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
        spec=postproc.spec_from_options(
            postprocess=postprocess, min_volume_mm3=min_volume_mm3,
            repair_gaps_mm=repair_gaps_mm, repair_close_radius=repair_close_radius,
        ) if postprocess is not None else None,
    )

    if output is not None:
        # Always a directory, never a file. One case and fifty are then the same call, and no
        # command has to change shape because the cohort grew -- which is what the old
        # "--output names a single file but N cases were given" refusal amounted to.
        destination = Path(output)
        if destination.is_file() or destination.suffix in (".gz", ".nii", ".mha"):
            raise ValueError(
                f"--output must be a directory, not a file ({destination}). The predictions "
                f"are written inside it, one per case."
            )
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for case_id in case_ids:
            produced = post_dir / f"{case_id}.nii.gz"
            if not produced.is_file():
                log.warning("No prediction for %s at %s.", case_id, produced)
                continue
            shutil.copyfile(produced, destination / produced.name)
            copied += 1
        if not copied:
            raise FileNotFoundError(
                f"Nothing to copy to {destination}. "
                + ("--skip-prediction was set, so nothing was predicted."
                   if skip_prediction else "The predictor produced no output.")
            )
        log.ok(f"{copied} prediction(s) -> {destination}")

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


def _cluster_path(path: Path | str) -> Path:
    """A user-supplied path as the job will see it.

    Under a fixed mount it is translated (``/corpus/...``); otherwise it keeps its host path and
    :func:`_user_data_paths` asks for an identity bind.
    """
    from nvitk.pipes.topbrain.util.sge_stage import to_container_path

    cluster, _ = pth.layout_auto()
    inside = to_container_path(cluster, path)
    return inside if inside is not None else Path(path)


def _user_data_paths(**options: Any) -> list[Path]:
    """Images and output directory that need their own bind mount."""
    from nvitk.pipes.topbrain.util.sge_stage import to_container_path

    cluster, _ = pth.layout_auto()
    candidates = [*(options.get("inputs") or ()), options.get("input_dir")]
    if options.get("output"):
        # The output directory is what the job will *create*, so it usually does not exist yet
        # and binding it would fail. Bind the nearest existing ancestor instead: the job then
        # makes the leaf inside a directory the container can already see.
        target = Path(options["output"])
        while not target.exists() and target != target.parent:
            target = target.parent
        candidates.append(target)
    return [
        Path(c) for c in candidates
        if c is not None and to_container_path(cluster, c) is None
    ]


def _worker_argv(
    *, label_set: str = "ta36", loss: str | None = None,
    folds: Sequence[int | str] | None = None, plans_identifier: str | None = None,
    configuration_name: str | None = None, checkpoint_name: str = "checkpoint_final.pth",
    min_volume_mm3: float | None = 5.0,
    largest_only: bool = False, device: str = "cuda", num_processes: int = 3,
    workers: int = 1, skip_prediction: bool = False, backend: str = "gpu",
    input_subdir: str = "imagesTr_topbrain",
    repair_gaps_mm: float | None = None, repair_adjacency: bool = False,
    repair_lateral: bool = False, repair_close_radius: int = 0,
    inputs: Sequence[Path] = (), input_dir: Path | None = None,
    model: str | None = None, modality: str | None = None,
    output: Path | None = None, output_name: str | None = None,
    no_postprocess: bool = False, postprocess: str | None = None,
    **_ignored: Any,
) -> list[str]:
    """Worker argv for stage 5, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage4_infer"),
        *sge_backend_cli_args(backend),
        "--label-set", label_set,
        "--nnunet-raw", quote_path(inside.nnunet_raw),
        "--nnunet-preprocessed", quote_path(inside.nnunet_preprocessed),
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--results-root", quote_path(inside.results_root),
        "--label-set", label_set,
        "--checkpoint-name", checkpoint_name,
        "--device", device or torch_device_for_backend(backend, remote=True),
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
    # A None here becomes a cryptic join error three frames away; say which flag it was.
    missing = [argv[i - 1] for i, a in enumerate(argv) if a is None]
    if missing:
        raise ValueError(f"stage4 worker argv: no value for {', '.join(missing)}.")
    # Only one source of images. Emitting the release folder as a default *alongside* an
    # explicit -i is how a single-image job ended up predicting 51 cases: the one asked for
    # plus the whole training set.
    if input_dir:
        argv.extend(["--input-dir", quote_path(_cluster_path(input_dir))])
    elif not inputs:
        argv.extend(["--input-dir", quote_path(inside.challenge_root / input_subdir)])

    if loss:
        # Optional interactively: --model resolves the trainer from stage 2's provenance, and a
        # literal "None" here would build a run directory name that does not exist.
        argv.extend(["--loss", quote_path(str(loss))])
    if postprocess:
        argv.extend(["--postprocess", quote_path(str(postprocess))])
    if no_postprocess:
        argv.append("--no-postprocess")
    for path in inputs:
        argv.extend(["-i", quote_path(_cluster_path(path))])
    if model:
        argv.extend(["--model", quote_path(str(model))])
    if modality:
        argv.extend(["--modality", quote_path(str(modality))])
    if output:
        argv.extend(["--output", quote_path(_cluster_path(output))])
    if output_name:
        argv.extend(["--output-name", quote_path(str(output_name))])
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
        data_paths=_user_data_paths(**options),
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
        data_paths=_user_data_paths(**options),
        backend=options.get("backend", "gpu"),
        request_gpu=options.get("device", "cuda") != "cpu",
        job_suffix=options.get("label_set", ""),
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


def resolve_from_source(from_source: str | None, submit: str) -> str:
    """Where the data lives, resolved against where the compute runs.

    ``--submit`` says where inference runs; ``--from-source`` says where the images and the
    output directory are. They are usually the same, so the default follows the submit target.

    Raises
    ------
    click.UsageError
        For ``--submit local --from-source sge``: pulling the cluster's data down to run here
        would be a data transfer with none of the reasons to want one. Ask for it explicitly
        with a cluster path instead.
    """
    submit = str(submit).lower()
    source = str(from_source).lower() if from_source else submit
    if submit == "local" and source == "sge":
        raise click.UsageError(
            "--submit local --from-source sge is not supported: to run here on cluster data, "
            "point --input-dir at the cluster path directly (it is mounted, or it is not "
            "reachable at all)."
        )
    return source


def _local_volumes(options: dict) -> list[Path]:
    """The volumes named on the command line, as local files.

    Accepts both forms stage 4 takes -- repeated ``-i`` and a ``--input-dir`` -- because a
    staged run is exactly when someone points at a folder they have in front of them.
    """
    volumes: list[Path] = [Path(v) for v in (options.get("inputs") or ())]
    directory = options.get("input_dir")
    if directory:
        volumes.extend(sorted(Path(directory).glob("*.nii.gz")))
    return [v for v in volumes if v.is_file()]


def _staged_submit(
    *,
    volumes: Sequence[Path],
    local_output: Path,
    credentials: tuple[str, str, str],
    submit_job: Any,
    poll_seconds: float,
) -> StagedTransfer:
    """Upload, run, retrieve, clean up. Returns what moved in each direction.

    Cleanup is deliberately skipped when anything goes wrong: the uploaded inputs and whatever
    the job did produce are exactly what is needed to work out why, and they are cheap to
    delete by hand once they have been looked at.
    """
    host, user, password = credentials
    remote_root = staging.new_remote_root()
    transfer = StagedTransfer(
        remote_root=remote_root,
        remote_input=posixpath.join(remote_root, "input"),
        remote_output=posixpath.join(remote_root, "output"),
    )
    log.info("Staging %d volume(s) through %s", len(volumes), remote_root)
    transfer.uploaded = staging.upload_inputs(
        volumes, transfer.remote_input, host=host, user=user, password=password
    )

    job_ids = submit_job(transfer.remote_input, transfer.remote_output)
    if not job_ids:
        transfer.notes.append("nothing was queued")
        log.warning("No job id came back; the staged data is left at %s", remote_root)
        return transfer

    finished = staging.wait_for_jobs(
        job_ids, host=host, user=user, password=password, poll_seconds=poll_seconds
    )
    if not finished:
        transfer.notes.append("timed out waiting for the job")
        return transfer

    transfer.retrieved = staging.retrieve_outputs(
        transfer.remote_output, Path(local_output), host=host, user=user, password=password
    )
    if not transfer.retrieved:
        # An empty output directory means the job failed. Keeping the tree is what makes that
        # diagnosable; the .err file names the reason.
        transfer.notes.append("the job produced no output; staged data kept for inspection")
        log.warning("Nothing came back from %s. The job likely failed — check its .err file. "
                    "Staged data left at %s", transfer.remote_output, remote_root)
        return transfer

    staging.remove_remote_root(remote_root, host=host, user=user, password=password)
    transfer.notes.append("staged data removed")
    return transfer


def _submit_to_cluster(
    *,
    container: Path | None,
    src_dir: Path | None,
    sge_project: str | None,
    sge_h_vmem: str | None,
    remote_host: str | None,
    remote_user: str | None,
    emit_script: Path | None,
    dry_run: bool,
    no_remote: bool,
    options: dict,
    from_source: str = "sge",
    poll_seconds: float = 30.0,
) -> list[str]:
    """Send one inference job to the cluster, the same way the master pipeline does.

    The images and the output directory are the *cluster's* paths, and both are bind-mounted
    when they fall outside the fixed roots -- an inference run is usually pointed at data the
    pipeline did not choose.
    """
    from nvitk.pipes.topbrain import config as cfg
    from nvitk.pipes.topbrain.util.sge_backend import (
        set_sge_h_vmem_override, set_sge_project_override,
    )
    from nvitk.pipes.topbrain.util.sge_stage import run_driver_script

    set_sge_project_override(sge_project)
    set_sge_h_vmem_override(sge_h_vmem)

    image = container or cfg.CONTAINER_PATH
    if image is None:
        raise click.UsageError(
            "--submit sge needs a container: pass --container, or set "
            "pipelines.topbrain.default_sge_container_root in sge.json."
        )
    cluster, _origin = pth.layout_auto()
    label_set = options.get("label_set") or "ta36"

    def _run(job_options: dict) -> list[str]:
        """Emit and run the driver for one set of stage 4 options."""

        def _emit(handle: TextIO) -> None:
            submit_sge(paths=cluster, container=Path(image), src_dir=src_dir,
                       dry_run=True, emit=handle, **job_options)

        try:
            return run_driver_script(
                _emit,
                title=f"topbrain inference label_set={label_set}",
                basename=f"submit_topbrain_infer_{label_set}",
                emit_script=emit_script, dry_run=dry_run, no_remote=no_remote,
                remote_host=remote_host, remote_user=remote_user,
                credentials=credentials,
            )
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc

    if from_source != "local":
        credentials = None
        return _run(options)

    # ---- Staged: the data is here, the GPU is there --------------------------
    from nvitk.cluster.remote_submit import prompt_ssh_credentials

    volumes = _local_volumes(options)
    if not volumes:
        raise click.UsageError(
            "--from-source local found no volumes to send. Pass -i FILE (repeatable) or "
            "--input-dir DIR containing .nii.gz files."
        )
    local_output = Path(options.get("output") or Path.cwd() / "topbrain_predictions")
    if dry_run or no_remote:
        log.info("--from-source local would upload %d volume(s) to %s and return the results "
                 "to %s", len(volumes), staging.staging_root(), local_output)

    # One prompt for the whole round trip: upload, submit, poll, download, clean up.
    credentials = prompt_ssh_credentials(
        remote_host=remote_host, remote_user=remote_user,
        host_aliases=pth.CLUSTER_HOST_ALIASES,
    )

    def _submit_job(remote_input: str, remote_output: str) -> list[str]:
        """Queue the job against the staged paths rather than the local ones."""
        return _run({
            **options, "inputs": (), "input_dir": remote_input, "output": remote_output,
        })

    transfer = _staged_submit(
        volumes=volumes, local_output=local_output, credentials=credentials,
        submit_job=_submit_job, poll_seconds=poll_seconds,
    )
    log.info("staged transfer: %s", transfer.as_dict())
    return []


@click.option("--submit", type=click.Choice(["local", "sge"], case_sensitive=False),
              default="local", show_default=True,
              help="Where inference runs. 'sge' builds a driver script and executes it on a "
                   "login node, exactly as the master pipeline does — useful when the images "
                   "or the model live where this host cannot reach them.")
@click.option("--from-source", type=click.Choice(["local", "sge"], case_sensitive=False),
              default=None,
              help="Where the images and the output directory live. Defaults to --submit. "
                   "With '--submit sge --from-source local' the inputs are uploaded to a "
                   "scratch directory on the cluster, inference runs there, the results come "
                   "back and the scratch directory is removed.")
@click.option("--poll-seconds", type=float, default=30.0, show_default=True,
              help="--from-source local: how often to ask qstat whether the job has finished.")
@click.option("--container", type=click.Path(path_type=Path), default=None,
              help="--submit sge: Singularity image (default: sge.json).")
@click.option("--src-dir", type=click.Path(path_type=Path), default=None,
              help="--submit sge: nvitk checkout to bind into the job.")
@click.option("--sge-project", type=str, default=None,
              help="--submit sge: SGE project (-P), overriding sge.json.")
@click.option("--sge-h-vmem", type=str, default=None,
              help="--submit sge: memory limit (-l h_vmem), overriding sge.json.")
@click.option("--remote-host", type=str, default=None)
@click.option("--remote-user", type=str, default=None)
@click.option("--emit-script", "emit_script", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, default=False,
              help="--submit sge: write the submission script and stop.")
@click.option("--no-remote", is_flag=True, default=False,
              help="--submit sge: write the script but do not run it.")
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
              help="Directory to copy the predictions into, one file per case. Always a directory — a single image is written inside it like any other.")
@click.option("--nnunet-raw", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths (see --config-dir).")
@click.option("--nnunet-preprocessed", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths.")
@click.option("--nnunet-results", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths.")
@click.option("--results-root", type=click.Path(path_type=Path), default=None,
              help="Defaults to sge.json's topbrain_paths.")
@click.option("--label-set", type=click.Choice(list(lbl.ALL_LABEL_SETS)),
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
@click.option("--postprocess", type=str, default=None,
              help="Which post-processing steps to apply: a comma list of "
                   "islands,largest,bridge,adjacency,lateral — or 'none' for the raw argmax, "
                   "or 'all'. Supersedes the individual --repair-* switches. Default: islands.")
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True,
              help="Drop connected components smaller than this, per class.")
@click.option("--no-postprocess", is_flag=True, default=False,
              help="Shorthand for --postprocess none.")
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
    submit: str = "local", from_source: str | None = None, poll_seconds: float = 30.0,
    container: Path | None = None, src_dir: Path | None = None,
    sge_project: str | None = None, sge_h_vmem: str | None = None,
    remote_host: str | None = None, remote_user: str | None = None,
    emit_script: Path | None = None, dry_run: bool = False, no_remote: bool = False,
    postprocess: str | None = None, backend: str = "gpu",
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
    source = resolve_from_source(from_source, submit)
    if str(submit).lower() == "sge":
        _submit_to_cluster(
            from_source=source, poll_seconds=poll_seconds,
            container=container, src_dir=src_dir, sge_project=sge_project,
            sge_h_vmem=sge_h_vmem, remote_host=remote_host, remote_user=remote_user,
            emit_script=emit_script, dry_run=dry_run, no_remote=no_remote,
            options=dict(
                inputs=list(inputs), input_dir=input_dir, model=model, modality=modality,
                output=output, output_name=output_name, label_set=label_set, loss=loss,
                architecture=architecture, folds=parse_folds(folds) if folds else None,
                plans_identifier=plans_identifier, configuration_name=configuration_name,
                checkpoint_name=checkpoint_name, min_volume_mm3=min_volume_mm3,
                largest_only=largest_only, no_postprocess=no_postprocess,
                postprocess=postprocess, repair_gaps_mm=repair_gaps_mm,
                repair_adjacency=repair_adjacency, repair_lateral=repair_lateral,
                repair_close_radius=repair_close_radius, device=device,
                num_processes=num_processes, workers=workers,
                skip_prediction=skip_prediction, backend=backend,
            ),
        )
        return

    run_infer(
        # --no-postprocess is the shorthand for the same thing, kept so existing commands work.
        postprocess="none" if no_postprocess else postprocess,
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
