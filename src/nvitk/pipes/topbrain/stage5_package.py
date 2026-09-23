"""ToPBrain stage 5: build the Grand Challenge algorithm container.

**Inputs**

- One or two trained nnU-Net runs under
  ``<nnunet_results>/DatasetXXX_.../<trainer>__<plans>__<config>/``

**Outputs**

- ``<results_root>/stage5_package/<name>/`` — the assembled build context
- ``.../<name>.tar.gz`` when ``--save`` is given, ready to upload
- ``.../topbrain_stage5.json`` — provenance

The build context is assembled and left on disk whether or not Docker is available, so the
image can be built on a different machine — the analysis host and the machine with a Docker
daemon are frequently not the same one.

One image, both tracks
----------------------
Grand Challenge exposes a CT **input** socket and an MR one, and a submission may be sent to
either portal. (Both write to the *same* output socket in the 2026 TA36 edition -- see
``docker/inference.py``.) This packs whichever models it is given into a single image and lets
the entry point pick at run time, so one build covers both tracks:

* two modality-specific models (``ta36_ct`` + ``ta36_mr``) — each socket gets its own;
* one modality-agnostic model (``ta36``) — registered for both sockets;
* one modality-specific model — that socket works, the other fails with a clear message rather
  than silently predicting with the wrong network.

What the entry point is told
----------------------------
``models.json`` carries, per modality, the model directory, the checkpoint name, how many input
channels the network expects and **the exact intensity windows stage 0 applied**. None of that
is recoverable from the weights, and a container that guesses a window produces a confident
wrong segmentation rather than an error — so the channel count is read back from each model's
own ``dataset.json`` and cross-checked against the windows given here.

Submission constraints, from the 2026 template (``CoWBenchmark/TopBrain_Algo_Submission``):
container ≤10 GB, ≤31 GiB DRAM, one ``.mha`` in and one ``.mha`` out with identical shape, no
network at run time. The entry point also reads NIfTI and NRRD, which the platform never sends
but local testing often has.

The template also allows the weights to be uploaded **separately** as a ``.tar.gz`` that the
platform extracts to ``/opt/ml/model/`` at run time. This stage bakes them in instead, which is
the template's other documented option and keeps the image self-contained -- with ``--layout
split`` a full five-fold ensemble lands around 9 GiB, inside the ceiling. The separate tarball
is what to reach for if a future model no longer fits.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

import nvitk
from nvitk.core.logger import Logger
from nvitk.normalization import CTA_WINDOW, MR_PERCENTILES
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.stage2_train import resolve_trained_run
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import postproc
from nvitk.pipes.topbrain.util.paths import DATASET_IDS, DATASET_SUFFIXES, STAGE5_PACKAGE_DIR
from nvitk.pipes.topbrain.util.sge_stage import build_stage_command, submit_stage_job

log = Logger()

#: Files nnU-Net needs in a model folder for ``initialize_from_trained_model_folder``.
MODEL_FILES: tuple[str, ...] = ("dataset.json", "plans.json")

#: Checkpoint copied per fold, when one is named explicitly.
CHECKPOINT_NAME: str = "checkpoint_final.pth"

#: Tried in order when no checkpoint is named. ``checkpoint_final.pth`` is the end of training
#: and is what a finished run ships; ``checkpoint_best.pth`` is the highest-validation-Dice
#: epoch and is what exists while a fold is still training. Resolved per model, because a CT run
#: still training and a finished MR run legitimately have different ones. ``checkpoint_latest``
#: is never chosen implicitly: it is whatever epoch the job stopped at, which is not a model
#: anyone should submit by accident.
CHECKPOINT_ORDER: tuple[str, ...] = ("checkpoint_final.pth", "checkpoint_best.pth")

#: Grand Challenge's hard image-size ceiling, in gibibytes.
MAX_IMAGE_GIB: float = 10.0

#: Roughly what the CUDA base image plus the Python dependencies weigh, in gibibytes. Used only
#: to warn early: the context check alone is misleading, because the base image is most of the
#: final size and never appears in the context.
BASE_IMAGE_GIB: float = 7.0

#: Checkpoint keys inference reads. Everything else in a training checkpoint — above all the
#: optimizer momentum buffer, which is half the file — is dead weight in a submission image.
#: Stripping is what makes a 5-fold ensemble fit under the 10 GiB ceiling.
CHECKPOINT_KEEP_KEYS: tuple[str, ...] = (
    "network_weights", "trainer_name", "inference_allowed_mirroring_axes",
)

#: Name of the per-modality model map written into the build context.
MODELS_CONFIG_NAME: str = "models.json"

#: Where the weights ride.
#:
#: ``baked``    inside the image, under ``/opt/algorithm/model``. Self-contained, and what keeps
#:              a submission reproducible from one artefact -- but it counts against the 10 GiB
#:              image ceiling.
#: ``tarball``  a separate ``.tar.gz`` uploaded alongside, which Grand Challenge extracts to
#:              ``/opt/ml/model/`` at run time. The image then stays around 7 GiB whatever the
#:              ensemble weighs, which is what makes ``--layout mixed`` viable with two full
#:              five-fold ensembles.
WEIGHT_MODES: tuple[str, ...] = ("baked", "tarball")


@dataclass
class ModelSpec:
    """One trained run destined for the image, and how to feed it."""

    modality: str
    """``ct`` or ``mr`` — which socket this model serves."""

    run_dir: Path
    """Trained nnU-Net run directory."""

    label_set: str
    """Label set it predicts; the post-processing needs it for the laterality pairs."""

    folds: tuple[int | str, ...]
    checkpoint: str
    serves: tuple[str, ...] = ()
    """Sockets this model is registered for. A modality-agnostic model serves both."""

    channels: int = 1
    """Input channels, read back from the run's own ``dataset.json``."""

    copied_folds: list[str] = field(default_factory=list)

    @property
    def directory(self) -> str:
        """Path of this model **relative to whichever weight root holds it**.

        Just the modality, so the same ``models.json`` works whether the weights were baked in
        (``/opt/algorithm/model/ct``) or arrived as a tarball (``/opt/ml/model/ct``). The entry
        point probes both roots.
        """
        return self.modality


def _docker_assets_dir() -> Path:
    """Directory holding the Dockerfile, entry point and requirements."""
    return Path(__file__).resolve().parent / "docker"


def discover_folds(run_dir: Path) -> list[str]:
    """Every fold on disk in *run_dir* that carries a checkpoint, in order.

    ``--folds`` naming an explicit subset is for fitting under the image ceiling; the default is
    the whole ensemble, because that is what the reported accuracy was measured on and quietly
    shipping one fold of five is a silent downgrade. ``fold_all`` -- nnU-Net's single
    all-the-data fold -- is picked up here too when a run has one.
    """
    found: list[str] = []
    for entry in sorted(Path(run_dir).glob("fold_*")):
        if entry.is_dir() and any(entry.glob("checkpoint_*.pth")):
            found.append(entry.name[len("fold_"):])
    if not found:
        raise FileNotFoundError(f"No fold_*/checkpoint_*.pth under {run_dir}.")
    return found


def resolve_checkpoint(run_dir: Path, folds: Sequence[int | str],
                       requested: str | None) -> str:
    """Pick the checkpoint to package for one run.

    An explicit name is honoured and fails loudly if absent. Otherwise the first of
    :data:`CHECKPOINT_ORDER` present in **every** requested fold is used -- a mixed ensemble,
    some folds final and some best, would be a different model from the one that was measured.
    """
    run_dir = Path(run_dir)
    names = [f"fold_{f}" for f in folds]
    if requested:
        missing = [n for n in names if not (run_dir / n / requested).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{requested} is missing from {', '.join(missing)} of {run_dir}."
            )
        return requested
    for candidate in CHECKPOINT_ORDER:
        if all((run_dir / n / candidate).is_file() for n in names):
            return candidate
    present = sorted({p.name for n in names for p in (run_dir / n).glob("*.pth")})
    raise FileNotFoundError(
        f"No checkpoint from {CHECKPOINT_ORDER} is in every fold of {run_dir}. "
        f"Found: {', '.join(present) or 'nothing'}. Name one with --checkpoint."
    )


def read_channels(run_dir: Path) -> int:
    """Input channels the trained run expects, from its ``dataset.json``.

    Read rather than assumed: the CT model takes two (a wide anatomical window and a narrow
    lumen window) and the MR model one, and a mismatch between what the image feeds and what the
    network wants surfaces as a shape error deep inside nnU-Net.
    """
    path = Path(run_dir) / "dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; it is needed to know the channel count.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get("channel_names") or {"0": None})


def strip_checkpoint(source: Path, destination: Path) -> tuple[int, int]:
    """Write an inference-only copy of *source*; returns ``(bytes_before, bytes_after)``.

    A training checkpoint is about 820 MB, half of which is the SGD momentum buffer. Inference
    reads four things — the weights, the trainer name, the allowed mirroring axes and
    ``init_args['configuration']`` — so everything else goes. ``init_args`` is rebuilt with only
    ``configuration``, because its ``plans`` entry embeds absolute paths from the training
    cluster that have no business in a submitted image.

    The weights are deliberately **not** cast to fp16: the state dict aliases each tensor several
    times over a smaller set of storages, ``torch.save`` de-duplicates them, and ``.half()``
    materialises independent copies that make the file larger rather than smaller.
    """
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    slim: dict[str, Any] = {k: checkpoint[k] for k in CHECKPOINT_KEEP_KEYS if k in checkpoint}
    configuration = (checkpoint.get("init_args") or {}).get("configuration")
    if configuration is None:
        raise KeyError(
            f"{source} has no init_args['configuration']; inference reads it to pick the "
            f"configuration, so a stripped copy would not load."
        )
    slim["init_args"] = {"configuration": configuration}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, destination)
    return source.stat().st_size, destination.stat().st_size


def collect_model(
    run_dir: Path, destination: Path, *, folds: Sequence[int | str], checkpoint: str,
    slim: bool = True,
) -> list[str]:
    """Copy a trained nnU-Net run into an inference-ready model folder.

    nnU-Net's predictor expects ``plans.json`` and ``dataset.json`` beside ``fold_N/`` — note
    it wants them named exactly that, not ``nnUNetResEncUNetLPlans.json``, so the plans file is
    renamed on the way in.

    With *slim* the checkpoints are reduced to what inference reads; the predictions are
    unchanged. Pass ``slim=False`` only to ship a byte-identical copy of the training artefact.

    Returns
    -------
    list of str
        The fold directory names copied.
    """
    run_dir, destination = Path(run_dir), Path(destination)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Trained run not found: {run_dir}")
    destination.mkdir(parents=True, exist_ok=True)

    plans_candidates = [
        p for p in run_dir.glob("*.json")
        if p.name not in ("dataset.json", "dataset_fingerprint.json")
    ]
    plans_source = run_dir / "plans.json"
    if not plans_source.is_file():
        if not plans_candidates:
            raise FileNotFoundError(f"No plans JSON in {run_dir}.")
        plans_source = plans_candidates[0]
    shutil.copyfile(plans_source, destination / "plans.json")

    dataset_json = run_dir / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(f"dataset.json not found in {run_dir}.")
    shutil.copyfile(dataset_json, destination / "dataset.json")

    copied: list[str] = []
    before = after = 0
    for fold in folds:
        name = f"fold_{fold}"
        source = run_dir / name / checkpoint
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source}")
        target = destination / name / checkpoint
        if slim:
            was, now = strip_checkpoint(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            was = now = source.stat().st_size
        before += was
        after += now
        copied.append(name)

    if slim and before:
        log.info("Checkpoints stripped: %.2f -> %.2f GiB (inference reads %s)",
                 before / 2**30, after / 2**30, ", ".join(CHECKPOINT_KEEP_KEYS))
    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    log.info("Model folder: %s folds, %.2f GiB -> %s", copied, total / 2**30, destination)
    return copied


def resolve_run_dir(
    nnunet_results: Path, results_root: Path, label_set: str, *, loss: str,
    architecture: str | None = None, plans_identifier: str | None = None,
    configuration_name: str | None = None,
) -> tuple[Path, dict[str, str]]:
    """Locate the trained run for *label_set*; returns ``(run_dir, identity)``.

    Plans name, architecture and configuration all come from what stage 2 actually trained.
    None of them has a usable default: the plans name embeds a spacing chosen at run time, and
    the architecture decides the trainer family.
    """
    resolved = resolve_trained_run(
        results_root, label_set, plans_identifier=plans_identifier,
        architecture=architecture, configuration_name=configuration_name,
    )
    trainer = loss_util.trainer_for_loss(loss, architecture=resolved["architecture"])
    dataset_name = f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"
    run_dir = (
        Path(nnunet_results) / dataset_name
        / f"{trainer}__{resolved['plans_identifier']}__{resolved['configuration']}"
    )
    identity = {
        "dataset": dataset_name, "label_set": label_set, "trainer": trainer, "loss": loss,
        "architecture": resolved["architecture"],
        "plans_identifier": resolved["plans_identifier"],
        "configuration": resolved["configuration"],
    }
    return run_dir, identity


def resolve_models(
    *, nnunet_results: Path, results_root: Path, ct_model: Path | None, mr_model: Path | None,
    label_set: str, loss: str, architecture: str | None, plans_identifier: str | None,
    configuration_name: str | None, folds: Sequence[int | str] | None,
    checkpoint: str | None,
) -> tuple[list[ModelSpec], list[dict[str, str]]]:
    """Decide which models go into the image and which sockets each one serves.

    Explicit ``--ct-model`` / ``--mr-model`` run directories win. With neither, the run is
    resolved from *label_set* the way earlier versions did, and a modality-agnostic label set
    produces one model registered for both sockets.

    Raises
    ------
    ValueError
        When nothing resolves. An image with no model builds fine and fails on the platform.
    """
    specs: list[ModelSpec] = []
    identities: list[dict[str, str]] = []

    explicit = {"ct": ct_model, "mr": mr_model}
    if any(explicit.values()):
        for modality, path in explicit.items():
            if path is None:
                continue
            run_dir = Path(path)
            declared = _label_set_for_run(run_dir, fallback=f"ta36_{modality}")
            wanted = list(folds) if folds else discover_folds(run_dir)
            chosen = resolve_checkpoint(run_dir, wanted, checkpoint)
            specs.append(ModelSpec(
                modality=modality, run_dir=run_dir, label_set=declared,
                folds=tuple(wanted), checkpoint=chosen, serves=(modality,),
            ))
            identities.append({"modality": modality, "run_dir": str(run_dir),
                               "label_set": declared, "checkpoint": chosen,
                               "source": "--%s-model" % modality})
        missing = [m for m, p in explicit.items() if p is None]
        if missing:
            log.warning(
                "No %s model given: the container will accept only the %s socket and fail "
                "clearly on the other. Pass --%s-model to cover both tracks.",
                ", ".join(m.upper() for m in missing),
                ", ".join(m.upper() for m, p in explicit.items() if p),
                missing[0],
            )
        return specs, identities

    run_dir, identity = resolve_run_dir(
        nnunet_results, results_root, label_set, loss=loss, architecture=architecture,
        plans_identifier=plans_identifier, configuration_name=configuration_name,
    )
    covers = lbl.LABEL_SET_MODALITIES.get(label_set, ("ct", "mr"))
    # A modality-agnostic label set is one network for both sockets; the harmonisation still
    # differs per socket, which is why it is registered twice rather than once.
    primary = covers[0]
    wanted = list(folds) if folds else discover_folds(run_dir)
    chosen = resolve_checkpoint(run_dir, wanted, checkpoint)
    specs.append(ModelSpec(
        modality=primary, run_dir=run_dir, label_set=label_set, folds=tuple(wanted),
        checkpoint=chosen, serves=tuple(covers),
    ))
    identities.append({**identity, "modality": primary, "serves": ",".join(covers),
                       "run_dir": str(run_dir), "checkpoint": chosen,
                       "source": "--label-set"})
    return specs, identities


def _label_set_for_run(run_dir: Path, *, fallback: str) -> str:
    """Infer the label set of an explicitly given run from its dataset directory name.

    The post-processing needs it for the laterality pairs and the valid-neighbour table, and the
    directory name is the only place a bare run directory records which dataset it belongs to.
    """
    dataset = Path(run_dir).parent.name
    for name, index in DATASET_IDS.items():
        if dataset.startswith(f"Dataset{index:03d}_"):
            return name
    log.warning("Cannot tell the label set of %s from its path; assuming %r.", run_dir, fallback)
    return fallback


def harmonisation_for(
    modality: str, channels: int, *, ct_window: Sequence[float] | None,
    ct_context_window: Sequence[float] | None, mr_percentiles: Sequence[float] | None,
    mr_context_percentiles: Sequence[float] | None,
) -> dict[str, Any]:
    """The intensity transform the container must apply, one entry per input channel.

    The list is returned under ``channel_transforms``; the plain channel *count* is a separate
    key, so neither can quietly overwrite the other in the registry.

    Raises
    ------
    ValueError
        When the channel count and the windows disagree. A two-channel network fed one channel
        fails loudly inside nnU-Net, but a two-channel network fed a *wrong* second channel does
        not fail at all — it just predicts worse. Refusing here is the only place that mismatch
        is cheap to catch.
    """
    if modality == "ct":
        main = list(ct_window or CTA_WINDOW)
        windows = [{"kind": "ct_window", "window": main}]
        if channels == 2:
            if not ct_context_window:
                raise ValueError(
                    "The CT model expects 2 input channels, so it was trained with a context "
                    "window, but --ct-context-window was not given. Pass the exact values "
                    "stage 0 used (they are in its provenance as 'ct_context_window_hu'); "
                    "guessing would ship a container that harmonises the second channel "
                    "differently from training and degrades silently."
                )
            windows.append({"kind": "ct_window", "window": list(ct_context_window)})
        elif ct_context_window:
            raise ValueError(
                f"--ct-context-window was given but the CT model expects {channels} input "
                f"channel(s). Either the wrong run was selected, or the window belongs to a "
                f"different training run."
            )
    else:
        main = list(mr_percentiles or MR_PERCENTILES)
        windows = [{"kind": "mr_percentiles", "percentiles": main}]
        if channels == 2:
            if not mr_context_percentiles:
                raise ValueError(
                    "The MR model expects 2 input channels but --mr-context-percentiles was "
                    "not given. Pass the values stage 0 used."
                )
            windows.append({"kind": "mr_percentiles",
                            "percentiles": list(mr_context_percentiles)})
        elif mr_context_percentiles:
            raise ValueError(
                f"--mr-context-percentiles was given but the MR model expects {channels} input "
                f"channel(s)."
            )
    if len(windows) != channels:
        raise ValueError(
            f"Built {len(windows)} channel transform(s) for a {channels}-channel "
            f"{modality.upper()} model."
        )
    return {"channel_transforms": windows}


#: How the models are distributed over images.
#:
#: ``split``   one image per modality. The default, because a full fold ensemble of both models
#:             does not fit under the 10 GiB ceiling, and Grand Challenge has a separate portal
#:             per track anyway — so nothing is lost by shipping two.
#: ``mixed``   one image serving both sockets. Convenient, and the only option when a single
#:             modality-agnostic model covers both, but with two full ensembles it will be
#:             rejected for size.
LAYOUTS: tuple[str, ...] = ("split", "mixed")


def _assemble_context(
    specs: Sequence[ModelSpec], identities: Sequence[dict[str, str]],
    *,
    results_root: Path,
    slim: bool = True,
    weights: str = "baked",
    base_image: str | None = None,
    ct_window: Sequence[float] | None = None,
    ct_context_window: Sequence[float] | None = None,
    mr_percentiles: Sequence[float] | None = None,
    mr_context_percentiles: Sequence[float] | None = None,
    name: str = "topbrain-ta36",
    tag: str = "latest",
    build: bool = False,
    save: bool = False,
    postprocess: str | None = None,
    min_volume_mm3: float = 5.0,
    repair_gaps_mm: float | None = None,
    repair_close_radius: int = 0,
) -> Path:
    """Build one image's context from *specs*; returns the context directory."""
    context = Path(results_root) / STAGE5_PACKAGE_DIR / name
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True, exist_ok=True)

    log.info("topbrain stage5 | %d model(s) -> %s", len(specs), context)

    # ---- 1. Docker assets ----------------------------------------------------
    assets = _docker_assets_dir()
    for filename in ("Dockerfile", "inference.py", "requirements.txt"):
        shutil.copyfile(assets / filename, context / filename)

    # ---- 2. The in-tree nnU-Net build ---------------------------------------
    # Inference must rebuild the network from the trainer class that trained it, and the
    # ToPBrain loss trainers live inside this build — the released nnunetv2 cannot resolve
    # them. Copying it in (and *not* pip-installing nnunetv2) is what makes the container able
    # to load its own model.
    from nvitk.pipes.topbrain.util.nnunet_env import nnunet_root

    shutil.copytree(
        nnunet_root() / "nnunetv2",
        context / "nnunetv2",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
    )

    # The ToPBrain trainers are the last thing in that tree importing nvitk -- for the loss, the
    # rare-class sampler and the lateral swap, all of which exist only during training. nnU-Net
    # only ever asks the class for build_network_architecture, so an inference-only stand-in
    # loads the same weights into the same network and removes the dependency outright.
    trainers = (
        context / "nnunetv2" / "training" / "nnUNetTrainer" / "topbrain" / "topbrain_trainers.py"
    )
    if not trainers.is_file():
        raise FileNotFoundError(
            f"{trainers} is missing from the vendored nnU-Net; the container could not resolve "
            f"the trainer class its checkpoints name."
        )
    shutil.copyfile(assets / "topbrain_trainers_inference.py", trainers)

    # Statements, not mentions: several files name nvitk in a comment or docstring, and only an
    # actual import would fail inside the image.
    importers = re.compile(r"^\s*(?:from|import)\s+nvitk\b", re.M)
    leftover = sorted(
        path.relative_to(context).as_posix()
        for path in (context / "nnunetv2").rglob("*.py")
        if importers.search(path.read_text(encoding="utf-8", errors="ignore"))
    )
    if leftover:
        raise RuntimeError(
            f"The vendored nnU-Net still imports nvitk in: {', '.join(leftover)}. The image does "
            f"not ship nvitk, so these would fail at run time on the platform."
        )

    # ---- 3. The slim algorithm package -------------------------------------
    # Not nvitk. The image needs the harmonisation, the component clean-up and nothing else;
    # copying the library in to get them dragged 151 MB of unrelated pipelines into an image
    # with a 10 GiB ceiling. topbrain_algo is numpy/scipy only and is kept in step by hand.
    shutil.copytree(
        assets / "topbrain_algo",
        context / "topbrain_algo",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    # ---- 3b. The post-processing the container will apply ---------------------
    # Written as data rather than hard-coded in inference.py: the point is that the selection
    # stage 3 measured is provably the one that ships.
    spec = postproc.spec_from_options(
        postprocess=postprocess, min_volume_mm3=min_volume_mm3,
        repair_gaps_mm=repair_gaps_mm, repair_close_radius=repair_close_radius,
    )
    _check_postprocess_supported(spec)
    postproc.write_container_config(context, spec)

    # ---- 4. Trained weights, one folder per model ----------------------------
    # Baked: straight into the context's model/, which the Dockerfile COPYs. Tarball: into a
    # sibling staging tree that becomes the .tar.gz, leaving model/ empty in the image so the
    # weights do not count against the 10 GiB ceiling.
    weights_root = (
        context / "model" if weights == "baked"
        else Path(results_root) / STAGE5_PACKAGE_DIR / f"{name}_model"
    )
    if weights == "tarball":
        shutil.rmtree(weights_root, ignore_errors=True)
        # Still created, and still COPYed: one Dockerfile has to serve both modes, and COPY of a
        # missing directory is a build error.
        (context / "model").mkdir(parents=True, exist_ok=True)
        (context / "model" / ".keep").write_text(
            "Weights ship as a separate tarball; see topbrain_stage5.json.\n", encoding="utf-8"
        )
    weights_root.mkdir(parents=True, exist_ok=True)

    registry: dict[str, dict[str, Any]] = {}
    for model in specs:
        model.channels = read_channels(model.run_dir)
        model.copied_folds = collect_model(
            model.run_dir, weights_root / model.directory,
            folds=model.folds, checkpoint=model.checkpoint, slim=slim,
        )
        for socket_modality in (model.serves or (model.modality,)):
            registry[socket_modality] = {
                "dir": model.directory,
                "checkpoint": model.checkpoint,
                "channels": model.channels,
                "label_set": model.label_set,
                "folds": model.copied_folds,
                **harmonisation_for(
                    socket_modality, model.channels,
                    ct_window=ct_window, ct_context_window=ct_context_window,
                    mr_percentiles=mr_percentiles,
                    mr_context_percentiles=mr_context_percentiles,
                ),
            }
        log.ok(f"{model.modality.upper()} model: {model.channels} channel(s), "
               f"{len(model.copied_folds)} fold(s), serves "
               f"{', '.join(s.upper() for s in (model.serves or (model.modality,)))}")

    (context / MODELS_CONFIG_NAME).write_text(
        # The image name travels with the registry so the entry point can say which build it
        # is. With --layout split there are two tarballs that look alike, and "this image has no
        # CT model" is far more useful when it also names the image you are running.
        json.dumps({"image": name, "models": registry}, indent=2) + "\n", encoding="utf-8"
    )

    _check_requirements_cover_imports(context)

    tarball: Path | None = None
    if weights == "tarball":
        tarball = weights_root.with_suffix(".tar.gz")
        log.info("Packing the weights into %s ...", tarball.name)
        # Members are stored relative to weights_root, so the archive expands to ct/ and mr/ at
        # the root of /opt/ml/model -- which is where the entry point looks.
        with tarfile.open(tarball, "w:gz") as archive:
            for entry in sorted(weights_root.iterdir()):
                archive.add(entry, arcname=entry.name)
        # The staging tree is a full second copy of the weights; the archive is the deliverable.
        shutil.rmtree(weights_root, ignore_errors=True)
        log.ok(f"weights tarball: {tarball} ({tarball.stat().st_size / 2**30:.2f} GiB) — "
               f"upload it under the algorithm's Models section; Grand Challenge extracts it "
               f"to /opt/ml/model/")

    context_size = sum(p.stat().st_size for p in context.rglob("*") if p.is_file())
    estimate = BASE_IMAGE_GIB + context_size / 2**30
    log.info("Build context: %.2f GiB (image ~%.2f GiB with the %.0f GiB base)",
             context_size / 2**30, estimate, BASE_IMAGE_GIB)
    if estimate > MAX_IMAGE_GIB:
        log.warning(
            "The image is likely to exceed the %.0f GiB challenge limit (estimate %.2f GiB). "
            "Drop folds with --folds, or package one modality per image.",
            MAX_IMAGE_GIB, estimate,
        )

    image = f"{name}:{tag}"
    archive: Path | None = None
    if build or save:
        _require_docker()
        log.info("Building %s ...", image)
        command = ["docker", "build", "-t", image]
        if base_image:
            command += ["--build-arg", f"BASE_IMAGE={base_image}"]
        subprocess.run([*command, str(context)], check=True)
        if save:
            archive = context.parent / f"{name}_{tag}.tar.gz"
            log.info("Saving %s ...", archive)
            with archive.open("wb") as handle:
                saver = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
                gzip = subprocess.Popen(["gzip", "-c"], stdin=saver.stdout, stdout=handle)
                saver.stdout.close()
                if gzip.wait() != 0 or saver.wait() != 0:
                    raise RuntimeError("docker save | gzip failed.")
            log.ok(f"wrote {archive} ({archive.stat().st_size / 2**30:.2f} GiB)")

    (context / "topbrain_stage5.json").write_text(
        json.dumps(
            {
                "stage": "stage5",
                "created": datetime.now().isoformat(timespec="seconds"),
                "models": list(identities),
                "registry": registry,
                "checkpoints_stripped": slim,
                "weights": weights,
                "weights_tarball": str(tarball) if tarball else None,
                "base_image": base_image,
                "postprocess": spec.as_dict(),
                "bundled_nnunet": "in-tree build (released nnunetv2 not installed)",
                "image": image, "context": str(context),
                "context_bytes": context_size,
                "estimated_image_gib": round(estimate, 2),
                "archive": str(archive) if archive else None,
                "built": bool(build or save),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    log.ok(f"stage5 complete: build context ready -> {context}")
    if not (build or save):
        log.info("Build it with:  docker build -t %s %s", image, context)
    return context


def run_package(
    *,
    nnunet_results: Path,
    results_root: Path,
    ct_model: Path | None = None,
    mr_model: Path | None = None,
    label_set: str = "ta36",
    loss: str | None = None,
    architecture: str | None = None,
    plans_identifier: str | None = None,
    configuration_name: str | None = None,
    folds: Sequence[int | str] | None = None,
    checkpoint: str | None = None,
    layout: str = "split",
    slim: bool = True,
    weights: str = "baked",
    base_image: str | None = None,
    ct_window: Sequence[float] | None = None,
    ct_context_window: Sequence[float] | None = None,
    mr_percentiles: Sequence[float] | None = None,
    mr_context_percentiles: Sequence[float] | None = None,
    name: str = "topbrain-ta36",
    tag: str = "latest",
    build: bool = False,
    save: bool = False,
    postprocess: str | None = None,
    min_volume_mm3: float = 5.0,
    repair_gaps_mm: float | None = None,
    repair_close_radius: int = 0,
) -> list[Path]:
    """Assemble (and optionally build) the submission container(s).

    Returns one context directory per image: a single one for ``mixed``, and one per modality
    for ``split``. Every fold present is packaged unless *folds* narrows it.

    Raises
    ------
    ValueError
        On an unknown *layout*.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown --layout {layout!r}; expected one of {', '.join(LAYOUTS)}.")
    if weights not in WEIGHT_MODES:
        raise ValueError(
            f"Unknown --weights {weights!r}; expected one of {', '.join(WEIGHT_MODES)}."
        )
    loss = loss or cfg.DEFAULT_LOSS
    specs, identities = resolve_models(
        nnunet_results=nnunet_results, results_root=results_root,
        ct_model=ct_model, mr_model=mr_model, label_set=label_set, loss=loss,
        architecture=architecture, plans_identifier=plans_identifier,
        configuration_name=configuration_name, folds=folds, checkpoint=checkpoint,
    )

    # A modality-agnostic model is one network registered for both sockets; splitting it would
    # write the same weights into two images for nothing.
    if layout == "split" and len(specs) > 1:
        groups = [([spec], [identity]) for spec, identity in zip(specs, identities)]
    else:
        if layout == "split" and len(specs) == 1 and len(specs[0].serves) > 1:
            log.info("One model serves %s; --layout split has nothing to split.",
                     ", ".join(s.upper() for s in specs[0].serves))
        groups = [(list(specs), list(identities))]

    shared = dict(
        results_root=results_root, slim=slim, weights=weights, base_image=base_image,
        ct_window=ct_window,
        ct_context_window=ct_context_window, mr_percentiles=mr_percentiles,
        mr_context_percentiles=mr_context_percentiles, tag=tag, build=build, save=save,
        postprocess=postprocess, min_volume_mm3=min_volume_mm3,
        repair_gaps_mm=repair_gaps_mm, repair_close_radius=repair_close_radius,
    )
    contexts: list[Path] = []
    for group_specs, group_identities in groups:
        suffix = f"-{group_specs[0].modality}" if len(groups) > 1 else ""
        log.info("stage5 | %s image %r: %s", layout, f"{name}{suffix}",
                 ", ".join(f"{m.modality.upper()}({len(m.folds)} folds)" for m in group_specs))
        contexts.append(_assemble_context(
            group_specs, group_identities, name=f"{name}{suffix}", **shared
        ))
    return contexts


def _container_supported_steps() -> tuple[str, ...]:
    """Post-processing steps the shipped ``topbrain_algo`` can apply.

    Read from the module that will be copied into the image, rather than repeated here, so the
    two cannot drift apart.
    """
    import ast

    path = _docker_assets_dir() / "topbrain_algo" / "postprocess.py"
    # Parsed rather than imported: the module is meant to run inside the image, and importing it
    # here would execute a second copy of code that shares no module identity with anything in
    # this process -- which is enough to break ``@dataclass``, since it resolves annotations
    # through ``sys.modules``.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign)
            else node.targets if isinstance(node, ast.Assign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "SUPPORTED_STEPS":
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"{path} does not define SUPPORTED_STEPS.")


def _check_postprocess_supported(spec: postproc.PostProcessSpec) -> None:
    """Refuse a selection the container cannot honour.

    The offline pipeline has more steps than the image does -- the topology repair rests on
    parts of nvitk that are deliberately not shipped. Catching the mismatch here, rather than
    letting the container skip a step at run time, is what keeps "what shipped is what was
    measured" true.

    Raises
    ------
    ValueError
        Naming the steps that would be silently dropped.
    """
    supported = _container_supported_steps()
    missing = [step for step in spec.steps if step not in supported]
    if missing:
        raise ValueError(
            f"--postprocess selects {', '.join(missing)}, which the submission container does "
            f"not implement (it has: {', '.join(supported)}). Those steps rest on parts of "
            f"nvitk that are deliberately not shipped in the image. Either drop them, or "
            f"measure and submit with a selection the container can honour."
        )


#: Import name -> distribution name, where they differ. Only the ones the vendored tree uses.
_IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "skimage": "scikit-image", "sklearn": "scikit-learn", "PIL": "pillow", "yaml": "pyyaml",
    "cv2": "opencv-python",
}

#: Provided by the base image rather than ``requirements.txt``.
_PROVIDED_BY_BASE: frozenset[str] = frozenset({"torch", "torchvision", "torchaudio"})


def _module_level_imports(path: Path) -> list[str]:
    """Modules *path* imports at module level, outside any ``try`` -- the ones that fire on
    import and therefore the ones a missing dependency breaks."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def _check_requirements_cover_imports(context: Path) -> None:
    """Verify ``requirements.txt`` covers everything the image will import.

    nnU-Net resolves the trainer class with ``recursive_find_python_class``, which **imports
    every module** under ``training/nnUNetTrainer/`` looking for it. That pulls in the training
    logger, the resampling helpers and more, none of which run at inference but all of which must
    import -- so the dependency list is much wider than what inference visibly touches, and a gap
    surfaces only once the built image is run.

    Raises
    ------
    RuntimeError
        Naming the missing distribution and the module that imports it, so the fix is a one-line
        addition rather than another build-and-run cycle.
    """
    import re as _re
    import sys

    root = context / "nnunetv2"
    declared = {
        _re.split(r"[><=;\s#]", line.strip())[0].lower().replace("_", "-")
        for line in (context / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    def module_path(dotted: str) -> Path | None:
        relative = Path(*dotted.split("."))
        for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
            resolved = context / candidate
            if resolved.is_file():
                return resolved
        return None

    seeds = ["nnunetv2.inference.predict_from_raw_data"]
    seeds += [
        ".".join(p.relative_to(context).with_suffix("").parts)
        for p in (root / "training" / "nnUNetTrainer").rglob("*.py")
    ]
    seen: set[str] = set()
    missing: dict[str, str] = {}
    stack = list(seeds)
    while stack:
        dotted = stack.pop()
        if dotted in seen:
            continue
        seen.add(dotted)
        path = module_path(dotted)
        if path is None:
            continue
        for imported in _module_level_imports(path):
            top = imported.split(".")[0]
            if top == "nnunetv2":
                stack.append(imported)
                continue
            if top in sys.stdlib_module_names or top in _PROVIDED_BY_BASE:
                continue
            distribution = _IMPORT_TO_DISTRIBUTION.get(top, top).lower().replace("_", "-")
            if distribution not in declared:
                missing.setdefault(distribution, str(path.relative_to(context)))

    if missing:
        detail = "; ".join(f"{dist} (imported by {where})" for dist, where in sorted(missing.items()))
        raise RuntimeError(
            f"requirements.txt does not cover everything the image imports: {detail}. Add them "
            f"to {_docker_assets_dir() / 'requirements.txt'} -- the container would build fine "
            f"and fail on the platform."
        )


def _require_docker() -> None:
    """Fail clearly when the Docker CLI is unavailable."""
    if shutil.which("docker") is None:
        raise FileNotFoundError(
            "docker not found on PATH. The build context has still been assembled — copy it to "
            "a machine with Docker and run 'docker build' there."
        )


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(
    *, label_set: str, loss: str, plans_identifier: str | None, configuration_name: str | None,
    folds: Sequence[int | str] | None, checkpoint: str | None, name: str, tag: str,
    build: bool, save: bool, backend: str, layout: str = "split",
    weights: str = "baked", base_image: str | None = None,
    ct_model: Path | None = None, mr_model: Path | None = None,
    ct_window: Sequence[float] | None = None,
    ct_context_window: Sequence[float] | None = None,
    mr_percentiles: Sequence[float] | None = None,
    mr_context_percentiles: Sequence[float] | None = None,
) -> list[str]:
    """Worker argv for stage 5, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv
    from nvitk.pipes.topbrain.util.sge_stage import container_layout, quote_path

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage5_package"),
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--results-root", quote_path(inside.results_root),
        "--label-set", label_set,
        "--loss", quote_path(loss),
        "--layout", layout,
        "--weights", weights,
        "--name", quote_path(name),
        "--tag", quote_path(tag),
    ]
    # Omitted when unknown: the worker reads them from stage 2's provenance, which is the only
    # place that knows the spacing preprocess_like_nnssl settled on at run time.
    if plans_identifier:
        argv.extend(["--plans-identifier", quote_path(plans_identifier)])
    if configuration_name:
        argv.extend(["--configuration", quote_path(configuration_name)])
    # Omitted so the worker packages every fold each run actually has.
    if folds:
        argv.extend(["--folds", quote_path(",".join(str(f) for f in folds))])
    # Omitted so the worker resolves it per model: a run still training has only
    # checkpoint_best, and naming checkpoint_final for both would fail on that one.
    if checkpoint:
        argv.extend(["--checkpoint", quote_path(checkpoint)])
    for flag, value in (("--ct-model", ct_model), ("--mr-model", mr_model)):
        if value:
            argv.extend([flag, quote_path(str(value))])
    for flag, pair in (
        ("--ct-window", ct_window), ("--ct-context-window", ct_context_window),
        ("--mr-percentiles", mr_percentiles),
        ("--mr-context-percentiles", mr_context_percentiles),
    ):
        if pair:
            argv.extend([flag, *[str(float(v)) for v in pair]])
    # Deliberately never --build on the cluster: Docker is not available inside Singularity.
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **options) -> str:
    """Host shell command for the stage 5 SGE task (context assembly only)."""
    return build_stage_command(
        "stage5", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **options,
) -> str:
    """Emit or submit the stage 5 SGE job."""
    return submit_stage_job(
        "stage5", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage5-package")
@click.option("--nnunet-results", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--ct-model", type=click.Path(path_type=Path), default=None,
              help="Trained CT run directory (the one holding plans.json and fold_*/). Takes "
                   "precedence over --label-set.")
@click.option("--mr-model", type=click.Path(path_type=Path), default=None,
              help="Trained MR run directory. Give both to cover CT and MR in one image.")
@click.option("--label-set", type=click.Choice(list(lbl.MULTICLASS_LABEL_SETS)), default="ta36",
              show_default=True,
              help="Used only when neither --ct-model nor --mr-model is given.")
@click.option("--loss", type=str, default=None)
@click.option("--architecture", type=str, default=None,
              help="Encoder family (ResEncL / PrimusM). Read from stage 2 when omitted.")
@click.option("--plans-identifier", type=str, default=None)
@click.option("--configuration", "configuration_name", type=str, default=None)
@click.option("--folds", type=str, default=None,
              help="Folds to package, e.g. '0,1,2'. Default: every fold present in each run — "
                   "the ensemble the reported accuracy was measured on.")
@click.option("--layout", type=click.Choice(list(LAYOUTS)), default="split", show_default=True,
              help="'split' writes one image per modality; 'mixed' puts both in one image, "
                   "which with two full ensembles will exceed the 10 GiB ceiling.")
@click.option("--checkpoint", type=str, default=None,
              help="Checkpoint name inside each fold. Default: resolved per model — "
                   "checkpoint_final.pth when every fold has one, else checkpoint_best.pth.")
@click.option("--weights", type=click.Choice(list(WEIGHT_MODES)), default="baked",
              show_default=True,
              help="'baked' puts the weights in the image; 'tarball' writes them as a separate "
                   ".tar.gz that Grand Challenge extracts to /opt/ml/model/, keeping the image "
                   "small enough that the ensemble size stops mattering.")
@click.option("--base-image", type=str, default=None,
              help="Override the Dockerfile's BASE_IMAGE. Only used when this command builds; "
                   "a manual 'docker build' takes the Dockerfile default.")
@click.option("--no-slim", is_flag=True, default=False,
              help="Ship the training checkpoints whole. They are about twice the size and "
                   "predict identically; the extra is optimizer state.")
@click.option("--ct-window", type=float, nargs=2, default=None,
              help="CT window in HU for the main channel, as stage 0 applied it "
                   "(default -100 1500).")
@click.option("--ct-context-window", type=float, nargs=2, default=None,
              help="CT window for the second channel. Required when the CT model expects 2 "
                   "input channels.")
@click.option("--mr-percentiles", type=float, nargs=2, default=None,
              help="MR robust percentiles for the main channel (default 0.5 99.5).")
@click.option("--mr-context-percentiles", type=float, nargs=2, default=None,
              help="MR percentiles for a second channel, if the MR model has one.")
@click.option("--name", type=str, default="topbrain-ta36", show_default=True)
@click.option("--tag", type=str, default="latest", show_default=True)
@click.option("--postprocess", type=str, default='none',
              help="Post-processing the container will apply: a comma list of "
                   "islands,largest,bridge,adjacency,lateral — or 'none'/'all'. Baked into the "
                   "image as data, so what shipped is what was measured. Default: islands.")
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True)
@click.option("--repair-gaps-mm", type=float, default=None,
              help="Gap ceiling for the 'bridge' step.")
@click.option("--repair-close-radius", type=int, default=0, show_default=True)
@click.option("--build", is_flag=True, default=False, help="Run 'docker build'.")
@click.option("--save", is_flag=True, default=False,
              help="Build, then write a tar.gz for upload to Grand Challenge.")
def main(
    nnunet_results: Path, results_root: Path, ct_model: Path | None, mr_model: Path | None,
    label_set: str, loss: str | None, architecture: str | None, plans_identifier: str | None,
    postprocess: str | None, min_volume_mm3: float,
    repair_gaps_mm: float | None, repair_close_radius: int, configuration_name: str | None,
    folds: str | None, layout: str, checkpoint: str | None, no_slim: bool,
    weights: str, base_image: str | None,
    ct_window: tuple[float, float] | None, ct_context_window: tuple[float, float] | None,
    mr_percentiles: tuple[float, float] | None,
    mr_context_percentiles: tuple[float, float] | None,
    name: str, tag: str, build: bool, save: bool,
) -> None:
    """CLI entry point: assemble (and optionally build) the submission container."""
    from nvitk.pipes.topbrain.stage2_train import parse_folds

    Logger()
    run_package(
        nnunet_results=nnunet_results, results_root=results_root,
        ct_model=ct_model, mr_model=mr_model, label_set=label_set,
        # Falls back to stage 2's provenance when --architecture is not given.
        loss=loss, architecture=architecture, plans_identifier=plans_identifier,
        configuration_name=configuration_name,
        folds=parse_folds(folds) if folds else None, checkpoint=checkpoint,
        layout=layout, slim=not no_slim, weights=weights, base_image=base_image,
        ct_window=ct_window or None, ct_context_window=ct_context_window or None,
        mr_percentiles=mr_percentiles or None,
        mr_context_percentiles=mr_context_percentiles or None,
        name=name, tag=tag, build=build, save=save,
        postprocess=postprocess, min_volume_mm3=min_volume_mm3,
        repair_gaps_mm=repair_gaps_mm, repair_close_radius=repair_close_radius,
    )


__all__ = [
    "CHECKPOINT_KEEP_KEYS", "CHECKPOINT_NAME", "CHECKPOINT_ORDER", "MODELS_CONFIG_NAME",
    "LAYOUTS", "WEIGHT_MODES", "ModelSpec", "discover_folds", "resolve_checkpoint",
    "build_sge_command", "collect_model", "harmonisation_for", "main", "read_channels",
    "resolve_models", "resolve_run_dir", "run_package", "strip_checkpoint", "submit_sge",
]


if __name__ == "__main__":
    main()
