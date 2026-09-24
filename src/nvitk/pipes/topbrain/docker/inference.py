"""Grand Challenge entry point for the TopBrain algorithm container.

Reads the single volume supplied on the input socket, runs nnU-Net inference plus the
pipeline's post-processing, and writes a mask of **identical shape** to the output socket.
Sockets follow the challenge's published interface::

    /input/images/head-ct-angio/<uuid>.mha   ->  /output/images/extended-head-angio-segmentation/output.mha
    /input/images/head-mr-angio/<uuid>.mha   ->  /output/images/extended-head-angio-segmentation/output.mha

Both tracks share **one** output socket and a **fixed** output filename. That changed between the
2025 and 2026 editions -- 2025 had per-modality ``head-{ct,mr}-angio-segmentation`` sockets and
echoed the input's name -- and writing to the old location leaves the expected one empty, which
the platform reports as a failed job with nothing in the log to explain it.

Which socket was populated is read from ``/input/inputs.json``, the manifest Grand Challenge
generates; the ``<uuid>`` input name is never predictable, so the volume itself is found by
scanning. NIfTI and NRRD are accepted alongside ``.mha`` (see :data:`INPUT_SUFFIXES`) so the same
image can be run over local data during development; the platform only ever supplies ``.mha``.

Local batch mode
----------------
The platform sends exactly one case per run. A hand-assembled local directory often holds
several -- both sockets filled, or several files in one -- and those are all segmented, each
routed to the model for its own modality and written under its own name. Only the single-case
path uses the fixed ``output.mha``, because only that path can ever be a real submission.

One image, both tracks
----------------------
Grand Challenge populates exactly one socket per run. Which one it is already names the
modality, and that is the primary signal — but it is cross-checked against the intensities,
because feeding a TOF volume through the CT harmonisation produces a confident wrong
segmentation rather than an error, and that is the failure worth spending a check on. On a
genuine conflict the run stops.

What this reads from the image
------------------------------
``models.json`` (written by stage 5) maps each socket to a model directory, the checkpoint, how
many input channels the network wants and the exact intensity windows stage 0 applied. Nothing
about harmonisation is hard-coded here: the container applies what its own build recorded.

``postprocess.json`` carries the post-processing selection, for the same reason — so the
pipeline the submission applies is provably the one that was measured.

The only first-party code in the image is :mod:`topbrain_algo` (harmonisation and component
clean-up, numpy/scipy only). ``nvitk`` itself is deliberately **not** copied in: the image needed
three functions from it and carrying the whole library cost 151 MB under a 10 GiB ceiling.

Runs offline with no network, weights baked into the image at ``/opt/algorithm/model``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

ALGORITHM_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path(os.environ.get("TOPBRAIN_INPUT_ROOT", "/input"))
OUTPUT_ROOT = Path(os.environ.get("TOPBRAIN_OUTPUT_ROOT", "/output"))
MODELS_CONFIG = Path(os.environ.get("TOPBRAIN_MODELS_CONFIG", ALGORITHM_ROOT / "models.json"))

# The challenge's input sockets: ``(slug, relative path under /input, modality)``.
#
# The slug is what ``inputs.json`` names; the relative path is where the volume lands. Both come
# from the 2026 submission template (``CoWBenchmark/TopBrain_Algo_Submission``), whose
# ``inputs.json`` fixtures pair ``head-ct-angiography`` with ``images/head-ct-angio``.
SOCKETS: tuple[tuple[str, str, str], ...] = (
    ("head-ct-angiography", "images/head-ct-angio", "ct"),
    ("head-mr-angiography", "images/head-mr-angio", "mr"),
)

# Where the mask goes. **One socket for both tracks** in the 2026 TA36 edition -- the 2025
# per-modality ``head-{ct,mr}-angio-segmentation`` sockets are gone. Writing to the old ones
# leaves the expected location empty and the job is marked failed, with nothing in the log to
# say why, so this is the single most important constant in the file.
OUTPUT_SOCKET: str = "images/extended-head-angio-segmentation"

# The mask's filename, fixed by the template -- *not* the input's name. ``main.py`` in the
# template writes ``location / f"output{suffix}"`` with ``suffix = ".mha"``.
OUTPUT_NAME: str = "output.mha"

# Written by Grand Challenge beside the images, naming the sockets that were populated.
INPUTS_MANIFEST: str = "inputs.json"

# Where the weights may live, most specific first.
#
# The challenge allows them either baked into the image or uploaded separately as a ``.tar.gz``
# that the platform extracts to ``/opt/ml/model/``. Probing both means a single image works in
# either mode: if a tarball is attached it wins, and otherwise the baked copy is used. That also
# makes it possible to attach an updated tarball to an already-uploaded image without rebuilding.
# ``$TOPBRAIN_WEIGHT_ROOTS`` (colon-separated) overrides them, which is how the tarball path is
# exercised off-platform -- ``/opt/ml`` is not writable on a development machine.
WEIGHT_ROOTS: tuple[str, ...] = tuple(
    os.environ["TOPBRAIN_WEIGHT_ROOTS"].split(":")
) if os.environ.get("TOPBRAIN_WEIGHT_ROOTS") else (
    "/opt/ml/model", str(ALGORITHM_ROOT / "model")
)

# Volume extensions accepted on the input socket, longest suffix first so ``.mha.gz`` is not
# mistaken for ``.gz``. Grand Challenge always supplies ``.mha``; the NIfTI forms are here so
# the same image can be run over local data during development without converting it first.
# SimpleITK reads all of them, and the output keeps whatever the input used -- which on the
# platform is always ``.mha``.
INPUT_SUFFIXES: tuple[str, ...] = (".mha.gz", ".mha", ".nii.gz", ".nii", ".nrrd")

# Below this many HU is air, for the purposes of recognising a CT.
CT_AIR_HU: float = -500.0

# A head CT has at least this fraction of its voxels in air.
CT_AIR_FRACTION: float = 0.05

# Scratch directories to try, in order. On Grand Challenge ``/tmp`` is writable and starts empty
# on every run, whatever the Dockerfile put there. The fallbacks matter for local testing, where
# a container run with ``--read-only`` but no ``--tmpfs /tmp`` has nowhere to work.
WORK_CANDIDATES: tuple[str, ...] = ("/tmp", "/var/tmp")


def log(message: str) -> None:
    """Print a progress line; the platform captures stdout."""
    print(f"[topbrain] {message}", flush=True)


def declared_modality() -> str | None:
    """The modality Grand Challenge says it populated, from ``inputs.json``.

    The platform generates this file listing the sockets it filled. Reading it is the documented
    contract, and it is unambiguous where scanning directories is merely usually right. Returns
    ``None`` when the file is absent (local testing) or names nothing recognised, so the caller
    falls back to looking at what is actually on disk.
    """
    manifest = INPUT_ROOT / INPUTS_MANIFEST
    if not manifest.is_file():
        return None
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        slugs = {entry["socket"]["slug"] for entry in entries}
    except (OSError, ValueError, KeyError, TypeError):
        log(f"WARNING: {manifest} is unreadable; falling back to scanning the sockets.")
        return None
    for slug, _relative, modality in SOCKETS:
        if slug in slugs:
            return modality
    log(f"WARNING: {manifest} names {sorted(slugs)}, none of which is a ToPBrain image socket.")
    return None


def find_inputs() -> list[tuple[Path, str]]:
    """Every supplied volume, as ``(path, socket_modality)`` pairs.

    Grand Challenge populates exactly one socket with exactly one image, so in a real run this
    returns a single pair and :func:`main` takes the single-case path. It returns more only when
    a hand-assembled local directory holds several -- several files in one socket, or both
    sockets filled at once, which is a natural way to keep a couple of test cases side by side.

    Returning them all rather than the first is the point. The earlier version stopped at the
    first socket that had anything, so a directory holding a CT *and* an MR silently segmented
    the CT and discarded the MR without a word.

    When ``inputs.json`` names a socket only that one is searched, because on the platform it is
    authoritative and a stray file in the other socket is not something to act on.

    Raises
    ------
    FileNotFoundError
        When no socket holds a readable volume.
    """
    announced = declared_modality()
    candidates = [s for s in SOCKETS if announced is None or s[2] == announced]
    found: list[tuple[Path, str]] = []
    for _slug, relative, modality in candidates:
        directory = INPUT_ROOT / relative
        if not directory.is_dir():
            continue
        for path in sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.name.endswith(INPUT_SUFFIXES)
        ):
            found.append((path, modality))
    if not found:
        searched = ", ".join(str(INPUT_ROOT / s[1]) for s in candidates)
        raise FileNotFoundError(
            f"No volume ({', '.join(INPUT_SUFFIXES)}) found under {searched}."
        )
    if announced is not None:
        log(f"inputs.json announced the {announced.upper()} socket")
    return found


def detect_modality(data: np.ndarray) -> str | None:
    """Read the modality off the intensities; ``None`` when they do not say.

    CT is calibrated: air sits near -1000 HU and a head scan is mostly air, so a large negative
    population is decisive. MR and TOF carry arbitrary non-negative units with no such
    population. Returns ``None`` rather than guessing for a volume that is neither — an
    already-normalised one, say — so an inconclusive reading never overrides the socket.
    """
    if data.size == 0:
        return None
    sample = data.ravel()
    if sample.size > 2_000_000:  # a few million voxels decide this as well as 200 million
        sample = sample[:: sample.size // 2_000_000]
    if float((sample < CT_AIR_HU).mean()) >= CT_AIR_FRACTION and float(sample.min()) < -900.0:
        return "ct"
    if float(sample.min()) >= -1.0:
        return "mr"
    return None


def resolve_modality(data: np.ndarray, socket_modality: str) -> str:
    """Reconcile the socket with what the intensities show.

    The socket is authoritative — the platform routes by track — but a measured contradiction is
    refused rather than obeyed. Harmonising TOF with an HU window does not fail, it just
    produces a worse answer, and a submission is not the place to discover that.

    Raises
    ------
    RuntimeError
        When the intensities unmistakably disagree with the socket.
    """
    measured = detect_modality(data)
    if measured is None:
        log(f"modality: {socket_modality.upper()} (from the socket; intensities inconclusive)")
        return socket_modality
    if measured != socket_modality:
        raise RuntimeError(
            f"The {socket_modality.upper()} socket was populated but the intensities are "
            f"unmistakably {measured.upper()} (min {float(data.min()):.1f}, "
            f"{float((data.ravel() < CT_AIR_HU).mean()):.1%} below {CT_AIR_HU:.0f}). "
            f"Harmonising it as {socket_modality.upper()} would corrupt the input."
        )
    log(f"modality: {socket_modality.upper()} (socket and intensities agree)")
    return socket_modality


def model_file_ending(model_dir: Path) -> str:
    """The extension nnU-Net will look for when it scans the staged input folder.

    ``predict_from_files`` enumerates cases by the ``file_ending`` recorded in the model's own
    ``dataset.json``, not by what it can read. Staging ``.mha`` for a model trained on
    ``.nii.gz`` finds zero cases and produces no output -- and nnU-Net reports that as a
    cheerful "There are 0 cases", not an error. The images themselves are read through
    SimpleITK either way, so matching the declared ending is all that is needed.
    """
    path = Path(model_dir) / "dataset.json"
    if not path.is_file():
        return ".nii.gz"
    return json.loads(path.read_text(encoding="utf-8")).get("file_ending") or ".nii.gz"


def silence_nnunet_path_warnings() -> None:
    """Point nnU-Net's three path variables somewhere harmless.

    They matter only for training and preprocessing. Unset, importing nnU-Net prints three
    paragraphs of setup advice into the submission log that read like failures.

    The paths are never opened -- ``nnunetv2.paths`` only reads the variables and warns when they
    are absent -- so nothing is created here. An earlier version did create them, which turned a
    purely cosmetic step into a crash the moment the filesystem was read-only.
    """
    for variable in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(variable, "/nonexistent/nnunet-unused-at-inference")


def resolve_work_root() -> Path:
    """A writable scratch directory for the staged channels and nnU-Net's output.

    ``$TOPBRAIN_WORK_ROOT`` wins when set; otherwise :data:`WORK_CANDIDATES` is tried in order.
    Writability is *probed*, not assumed: a directory can exist and still be read-only, which is
    exactly what happens under ``docker run --read-only`` without a ``--tmpfs`` for it.

    Raises
    ------
    RuntimeError
        When nothing is writable, naming what was tried and the flag that fixes it.
    """
    explicit = os.environ.get("TOPBRAIN_WORK_ROOT")
    candidates = [Path(explicit)] if explicit else [Path(c) for c in WORK_CANDIDATES]
    failures: list[str] = []
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".topbrain_write_probe"
            probe.touch()
            probe.unlink()
        except OSError as error:
            failures.append(f"{candidate} ({error.strerror or error})")
            continue
        return candidate
    raise RuntimeError(
        "No writable scratch directory: " + "; ".join(failures) + ". Grand Challenge always "
        "makes /tmp writable; locally, a container run with --read-only needs "
        "--tmpfs /tmp:rw,exec,size=30g alongside it, or set $TOPBRAIN_WORK_ROOT."
    )


def check_output_writable() -> Path:
    """Verify the output socket can be written to, before spending minutes on a prediction.

    Same probe as :func:`resolve_work_root`, applied to the socket the mask must land in. Not a
    safety net -- ``OUTPUT_ROOT`` is fixed at ``/output`` on Grand Challenge and the platform
    guarantees it is writable there -- but the commonest local-testing mistake is bind-mounting a
    host directory owned by the host user onto this path: the image runs as the non-root
    ``algorithm`` user (the challenge requires that), whose uid does not match, and the mount
    itself succeeds so nothing surfaces until after prediction, deep inside SimpleITK. Failing
    here instead turns a wasted GPU run into an immediate, actionable message.

    Returns
    -------
    Path
        The directory the mask will be written to, already confirmed writable.

    Raises
    ------
    RuntimeError
        Naming the directory and, for the local-mount case specifically, the fix.
    """
    destination = OUTPUT_ROOT / OUTPUT_SOCKET
    try:
        destination.mkdir(parents=True, exist_ok=True)
        probe = destination / ".topbrain_write_probe"
        probe.touch()
        probe.unlink()
    except OSError as error:
        raise RuntimeError(
            f"Cannot write to {destination} ({error.strerror or error}). On Grand Challenge this "
            f"path is always writable; locally it usually means the host directory bind-mounted "
            f"onto it is not writable by this image's non-root user. Fix the host directory's "
            f"permissions (e.g. 'chmod 777' it for a quick local test) or its ownership, then "
            f"re-run -- nothing has been computed yet."
        ) from error
    return destination


def load_registry() -> dict[str, Any]:
    """Read ``models.json``; returns the whole payload, image name included.

    Raises
    ------
    FileNotFoundError
        When it is absent. Without it there is no way to know the channel count or the windows,
        and every default would be a guess.
    """
    if not MODELS_CONFIG.is_file():
        raise FileNotFoundError(
            f"{MODELS_CONFIG} is missing. stage 5 writes it; an image built without it cannot "
            f"know how many channels the model wants or which windows to apply."
        )
    payload = json.loads(MODELS_CONFIG.read_text(encoding="utf-8"))
    models = payload.get("models") or {}
    if not models:
        raise ValueError(f"{MODELS_CONFIG} lists no models.")
    log(f"image={payload.get('image') or '?'} serves "
        f"{', '.join(sorted(k.upper() for k in models))}")
    return payload


def select_model(payload: dict[str, Any], modality: str) -> dict[str, Any]:
    """The entry for *modality*.

    Raises
    ------
    RuntimeError
        When this image carries no model for that socket. Falling back to the other modality's
        network would produce plausible output from the wrong model, which is worse than a
        failed job. The message names the image, because ``--layout split`` produces two
        tarballs that look alike and the commonest mistake is feeding the wrong one.
    """
    registry = payload.get("models") or {}
    entry = registry.get(modality)
    if entry is None:
        carried = sorted(k.upper() for k in registry)
        raise RuntimeError(
            f"Image {payload.get('image') or '<unnamed>'!r} serves the "
            f"{'/'.join(carried)} track only, but the {modality.upper()} socket was populated. "
            f"Submit this image to the {'/'.join(carried)} portal, run the "
            f"{modality.upper()} image on this case, or rebuild with --{modality}-model to "
            f"cover both."
        )
    return entry


def resolve_model_dir(relative: str) -> Path:
    """Find the model directory named by ``models.json`` under one of :data:`WEIGHT_ROOTS`.

    Raises
    ------
    FileNotFoundError
        Listing what was tried. The two failure modes look identical from inside -- a tarball
        that was never attached, and one whose contents are nested a level deeper than expected
        -- so the message names both.
    """
    for root in WEIGHT_ROOTS:
        candidate = Path(root) / relative
        if (candidate / "plans.json").is_file():
            # Judged by location, not by a path prefix: the roots are overridable for testing,
            # and a label that lies about where the weights came from is worse than none.
            inside_image = candidate.is_relative_to(ALGORITHM_ROOT)
            log(f"weights: {candidate} "
                f"({'baked into the image' if inside_image else 'mounted from outside it'})")
            return candidate
    tried = ", ".join(str(Path(r) / relative) for r in WEIGHT_ROOTS)
    raise FileNotFoundError(
        f"No model directory with a plans.json at any of: {tried}. If the weights were uploaded "
        f"as a separate tarball, check it is attached to this algorithm and that it expands to "
        f"{relative}/ at the root of /opt/ml/model rather than inside another directory."
    )


def harmonise_channels(data: np.ndarray, entry: dict[str, Any]) -> list[np.ndarray]:
    """Build every input channel the model expects, in order.

    The transforms come from the build, not from a default here: stage 5 recorded the windows
    stage 0 applied, and only the clipping they do survives nnU-Net's per-image z-score, so a
    different clip point is a domain shift nothing downstream corrects.
    """
    from topbrain_algo.harmonize import robust_scale, window_ct

    channels: list[np.ndarray] = []
    for index, transform in enumerate(entry["channel_transforms"]):
        kind = transform["kind"]
        if kind == "ct_window":
            window = tuple(float(v) for v in transform["window"])
            values = window_ct(data, window=window)
            detail = f"HU {window[0]:g}..{window[1]:g}"
        elif kind == "mr_percentiles":
            percentiles = tuple(float(v) for v in transform["percentiles"])
            # Non-zero voxels only: TOF stores air as exactly 0 over most of the field of view,
            # which would otherwise pin the low percentile at 0 and waste the output range.
            values = robust_scale(data, percentiles=percentiles)
            detail = f"p{percentiles[0]:g}..p{percentiles[1]:g}"
        else:
            raise ValueError(f"Unknown channel transform {kind!r} in {MODELS_CONFIG}.")
        channels.append(np.asarray(values, dtype=np.float32))
        log(f"channel {index:04d}: {kind} {detail}")

    if len(channels) != int(entry["channels"]):
        raise RuntimeError(
            f"Built {len(channels)} channel(s) but the model expects {entry['channels']}."
        )
    return channels


def build_predictor(model_dir: Path, entry: dict[str, Any]):
    """Load one modality's ensemble. Built once per modality, reused for every case of it.

    Loading a five-fold ResEnc-L ensemble is by far the most expensive setup step, so a batch of
    several CT cases must not pay it once per case.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory
        log(f"device: cuda ({name}, {total / 2**30:.1f} GiB VRAM)")
    else:
        # Loud, because it is not a detail: a 5-fold ResEnc-L ensemble on CPU is orders of
        # magnitude slower than on a T4, and a submission that silently landed here would be
        # killed by the phase time limit rather than fail visibly.
        log("WARNING: no CUDA device visible — running on CPU. This is very slow; on Grand "
            "Challenge it means a GPU instance was not selected, or --gpus was omitted locally.")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # labels are lateralised; see the trainer docstring
        # Explicit rather than left to default to True. nnU-Net prints its "only supported for
        # cuda devices!" line regardless, so this does not silence it -- but it states the
        # intent, and the preceding WARNING is what tells a reader the line is expected.
        perform_everything_on_device=on_cuda,
        device=torch.device("cuda" if on_cuda else "cpu"),
        allow_tqdm=False,
    )
    # initialize_from_trained_model_folder reads the trainer name out of the model folder and
    # resolves the class to rebuild the architecture — which only works because the in-tree
    # build (carrying the ToPBrain trainers) is the nnunetv2 on PYTHONPATH here.
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=None,  # every fold present in the image
        checkpoint_name=entry.get("checkpoint", "checkpoint_final.pth"),
    )
    return predictor


def segment(image, array: np.ndarray, *, predictor, entry: dict[str, Any], model_dir: Path,
            work_root: Path) -> np.ndarray:
    """Harmonise, predict and post-process one volume; returns the ``uint8`` mask."""
    work_in = work_root / "topbrain_in"
    work_out = work_root / "topbrain_out"
    for directory in (work_in, work_out):
        # Cleared, not merely created. Grand Challenge hands each run an empty /tmp, but a
        # locally reused one is not: a CT case leaves case_0001 behind, the next MR case writes
        # only case_0000, and nnU-Net then sees a two-channel case for a one-channel model. It
        # reports that as "Background workers died", which points nowhere near the cause. The
        # same applies between cases of one batch, which is why this is per case, not per run.
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    ending = model_file_ending(model_dir)
    for index, channel in enumerate(harmonise_channels(array, entry)):
        prepared = sitk.GetImageFromArray(channel)
        prepared.CopyInformation(image)
        # nnU-Net identifies channels by the _000N suffix, and finds cases at all only if the
        # extension matches the model's declared file_ending.
        sitk.WriteImage(prepared, str(work_in / f"case_{index:04d}{ending}"))

    staged = sorted(p.name for p in work_in.iterdir() if p.is_file())
    if len(staged) != int(entry["channels"]):
        raise RuntimeError(
            f"Staged {len(staged)} file(s) ({', '.join(staged)}) for a "
            f"{entry['channels']}-channel model. nnU-Net would report this as "
            f"'Background workers died', which says nothing about the cause."
        )

    # Sequential, not predict_from_files: that one hands the preprocessed volume to a worker
    # process through a torch.multiprocessing Queue, which moves it into /dev/shm. Docker gives a
    # container 64 MB of /dev/shm by default, while a head angiogram resampled to the plan's
    # spacing is several hundred MB of float32 -- so the worker dies instantly and nnU-Net
    # reports "Background workers died", blaming RAM. Nothing here can set --shm-size: on Grand
    # Challenge we do not control the run flags at all.
    #
    # The worker costs nothing to give up. It exists to overlap preprocessing of case N+1 with
    # prediction of case N, and the platform sends exactly one case per run, so there is nothing
    # to overlap. This trades a failure mode for no measurable time.
    predictor.predict_from_files_sequential(
        str(work_in), str(work_out), save_probabilities=False, overwrite=True,
    )

    produced = sorted(work_out.glob("case.*"))
    if not produced:
        raise RuntimeError(f"nnU-Net produced no output under {work_out}.")
    mask_image = sitk.ReadImage(str(produced[0]))
    mask = sitk.GetArrayFromImage(mask_image)

    # The selection is read from postprocess.json, which stage 5 wrote beside this file, rather
    # than hard-coded: the whole point is that the pipeline the submission applies is the one
    # that was measured in stage 3, not a second copy of it that can drift.
    from topbrain_algo import postprocess as pp

    steps, min_volume_mm3 = pp.read_config(ALGORITHM_ROOT)
    cleaned, report = pp.apply(
        mask, steps=steps,
        spacing=tuple(reversed(mask_image.GetSpacing())),  # sitk arrays are (z, y, x)
        min_volume_mm3=min_volume_mm3,
    )
    log(report.describe())
    return np.asarray(cleaned, dtype=np.uint8)


def output_name(path: Path, batch: bool) -> str:
    """The filename for one case's mask.

    A real Grand Challenge run has exactly one case and the socket expects exactly
    ``output.mha``, so the single-case path keeps that name unchanged. A local batch has no
    single output slot to reuse, so each mask is named after its input instead -- which can only
    happen off-platform, since the platform never sends more than one case per run.
    """
    if not batch:
        return OUTPUT_NAME
    name = path.name
    for suffix in INPUT_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}.mha"


def main() -> int:
    """Predict for every supplied case and write the masks to the output socket."""
    silence_nnunet_path_warnings()
    # Checked first, before any preprocessing or prediction: a permission problem here is the
    # same after ten minutes of GPU work as it is right now, so there is nothing to gain by
    # discovering it late.
    output_dir = check_output_writable()
    cases = find_inputs()
    batch = len(cases) > 1
    if batch:
        # Never reachable on the platform, so say plainly that this is the local path and that
        # the filenames differ because of it.
        log(f"{len(cases)} volumes supplied — batch mode (Grand Challenge sends one per run); "
            f"each mask is named after its input rather than {OUTPUT_NAME}")

    registry = load_registry()
    work_root = resolve_work_root()

    # Resolved up front so the run stops on a modality this image cannot serve before predicting
    # anything, and so the cases can be grouped.
    resolved: list[tuple[Path, str]] = []
    for path, socket_modality in cases:
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
        resolved.append((path, resolve_modality(array, socket_modality)))
        del array
    for _path, modality in resolved:
        select_model(registry, modality)

    names = [output_name(path, batch) for path, _ in resolved]
    clashing = {n for n in names if names.count(n) > 1}
    if clashing:
        raise RuntimeError(
            f"Two or more inputs would be written to the same mask: "
            f"{', '.join(sorted(clashing))}. Rename them so each case has its own output."
        )

    written = 0
    # Grouped by modality, and the predictor built once per group: loading a five-fold ResEnc-L
    # ensemble dwarfs the per-case work, so alternating CT/MR case by case would reload it every
    # time. Processing a group at a time also keeps only one ensemble resident.
    for modality in ("ct", "mr"):
        group = [(p, m) for p, m in resolved if m == modality]
        if not group:
            continue
        entry = select_model(registry, modality)
        model_dir = resolve_model_dir(entry["dir"])
        log(f"model={entry['dir']} folds={entry.get('folds')} "
            f"channels={entry['channels']} labels={entry.get('label_set')} "
            f"cases={len(group)}")
        predictor = build_predictor(model_dir, entry)

        for path, _ in group:
            log(f"case {path.name} ({modality.upper()})")
            image = sitk.ReadImage(str(path))
            original_size = image.GetSize()
            array = sitk.GetArrayFromImage(image).astype(np.float32)
            cleaned = segment(
                image, array, predictor=predictor, entry=entry, model_dir=model_dir,
                work_root=work_root,
            )

            output = sitk.GetImageFromArray(cleaned)
            output.CopyInformation(image)
            if output.GetSize() != original_size:
                raise RuntimeError(
                    f"Output size {output.GetSize()} != input size {original_size}; the "
                    f"challenge requires an identical grid."
                )

            # One socket for both tracks: see OUTPUT_SOCKET. output_dir was confirmed writable
            # at the very start of main(), so this is the only place that computes the path.
            out_path = output_dir / output_name(path, batch)
            sitk.WriteImage(output, str(out_path), useCompression=True)
            log(f"wrote {out_path} labels={sorted(np.unique(cleaned).tolist())[:8]}...")
            written += 1

        del predictor

    if written != len(resolved):
        raise RuntimeError(f"Wrote {written} mask(s) for {len(resolved)} case(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
