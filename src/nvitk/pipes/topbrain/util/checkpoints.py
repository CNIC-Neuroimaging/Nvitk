"""Locate the pre-trained checkpoint that seeds the segmentation encoder.

Description
-----------
The encoder can come from three places, selected by ``--pretrain-source``:

``checkpoint``
    A published self-supervised checkpoint (an OpenMind ResEnc-L, or any nnssl-format file).
    Resolved from an explicit path, from a name under the configured model root, or by
    download from an explicit URL.
``ssl``
    The output of this pipeline's own stage 2 — the newest ``checkpoint_final.pth`` under
    ``nnssl_results``.
``none``
    No pre-training: nnU-Net trains from random initialisation. Worth keeping as a first-class
    option, because it is the baseline every pre-trained run has to beat to justify itself.

Published OpenMind checkpoints
------------------------------
:data:`OPENMIND_MODELS` lists the DKFZ checkpoints trained on the OpenMind corpus (~114 k brain
MR volumes). **Both families are usable**: the in-tree nnU-Net build fine-tunes ResEnc-L through
``PretrainedTrainer`` and Primus-M through ``PretrainedTrainer_Primus``, and
:func:`~nvitk.pipes.topbrain.util.losses.trainer_for_loss` picks the family from the checkpoint's
architecture. :func:`resolve_openmind` fetches one by name, alongside the
``adaptation_plan.json`` the training stage reads.

On downloading
--------------
Checkpoint download is best-effort and deliberately not the only route: ``huggingface.co`` is
not reachable from every analysis host (it currently returns an auth error from this one), and
a pipeline that can only run where the internet is open is not much use on a cluster. Placing a
file under the model root by hand is always sufficient, and
:func:`describe_checkpoint_sources` prints exactly where to put it.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nvitk.core.logger import Logger

log = Logger()

#: Where ``--pretrain-source`` can take an encoder from.
PretrainSource = Literal["checkpoint", "ssl", "none"]

#: File names nnssl writes, in the order we prefer them.
NNSSL_CHECKPOINT_NAMES: tuple[str, ...] = (
    "checkpoint_final.pth",
    "checkpoint_best.pth",
    "checkpoint_latest.pth",
)

#: Extensions accepted when resolving a checkpoint name against the model root.
_CHECKPOINT_SUFFIXES: tuple[str, ...] = (".pth", ".pt", ".ckpt")

#: Files fetched for a published checkpoint. The adaptation plan is not optional in practice:
#: several nnssl trainers drop the embedded copy from the checkpoint, and stage 3 falls back to
#: this sidecar before it falls back to assumed key names.
_OPENMIND_FILES: tuple[str, ...] = ("checkpoint_final.pth", "adaptation_plan.json")

#: Template for a HuggingFace file download.
_HF_URL: str = "https://huggingface.co/{repo}/resolve/main/{filename}"


@dataclass(frozen=True)
class PublishedCheckpoint:
    """A published self-supervised checkpoint that can seed the segmentation encoder."""

    name: str
    repo: str
    architecture: str
    method: str
    description: str

    @property
    def is_transformer(self) -> bool:
        """Whether this is a Primus (transformer) encoder rather than a convolutional one."""
        return self.architecture.startswith("Primus")


def _openmind(name: str, method: str, description: str) -> PublishedCheckpoint:
    """Build an OpenMind registry entry from its short name."""
    architecture = "ResEncL" if name.startswith("resencl") else "PrimusM"
    repo_arch = "ResEncL" if architecture == "ResEncL" else "PrimusM"
    return PublishedCheckpoint(
        name=name,
        repo=f"MIC-DKFZ/{repo_arch}-OpenMind-{method}",
        architecture=architecture,
        method=method,
        description=description,
    )


#: Published OpenMind checkpoints, pre-trained on ~114 k brain MR volumes.
#:
#: The ``primusm-*`` entries are listed so that asking for one gives a useful error rather than a
#: mysterious adapter failure — they are transformers and cannot seed a convolutional decoder.
OPENMIND_MODELS: dict[str, PublishedCheckpoint] = {
    entry.name: entry
    for entry in (
        _openmind("resencl-mae", "MAE",
                  "Masked autoencoding. The general-purpose default; strong, well-studied."),
        _openmind("resencl-voco", "VoCo",
                  "Volume contrastive. Learns global position/context rather than local texture."),
        _openmind("resencl-vf", "VF",
                  "VolumeFusion. Dense pseudo-segmentation pretext, closest in shape to the "
                  "downstream task."),
        _openmind("resencl-mg", "MG",
                  "Models Genesis. Restoration from several corruptions."),
        _openmind("resencl-s3d", "S3D", "S3D self-distillation."),
        _openmind("resencl-simclr", "SimCLR", "Instance-level contrastive."),
        _openmind("resencl-swinunetr", "SwinUNETR",
                  "SwinUNETR-style pretext, on a ResEnc-L encoder."),
        _openmind("primusm-mae", "MAE",
                  "Transformer masked autoencoding. The strongest Primus variant in the paper."),
        _openmind("primusm-simmim", "SimMIM", "Transformer simple masked image modelling."),
        _openmind("primusm-voco", "VoCo", "Transformer volume contrastive."),
        _openmind("primusm-vf", "VF", "Transformer VolumeFusion."),
        _openmind("primusm-mg", "MG", "Transformer Models Genesis."),
        _openmind("primusm-simclr", "SimCLR", "Transformer instance contrastive."),
        _openmind("primusm-swinunetr", "SwinUNETR", "Transformer SwinUNETR-style pretext."),
    )
}


def describe_openmind_models() -> str:
    """Operator-facing listing of the published checkpoints."""
    lines = ["Published OpenMind checkpoints (--checkpoint-name):", ""]
    for label, transformer in (("ResEnc-L (convolutional)", False), ("Primus-M (transformer)", True)):
        lines.append(f"  {label}:")
        for name, entry in OPENMIND_MODELS.items():
            if entry.is_transformer is transformer:
                lines.append(f"    {name:20s} {entry.description}")
        lines.append("")
    lines.append("  All are trained on OpenMind (~114k brain MR volumes) and fine-tune through")
    lines.append("  PretrainedTrainer / PretrainedTrainer_Primus respectively.")
    return "\n".join(lines)


def resolve_openmind(name: str, model_root: Path, *, overwrite: bool = False) -> Path:
    """Fetch a published OpenMind checkpoint into *model_root*; returns its path.

    Downloads the adaptation plan beside the checkpoint so stage 3 can read the real key layout
    rather than assuming one.

    Raises
    ------
    ValueError
        For an unknown name, or a Primus checkpoint this pipeline cannot use — refused here
        rather than several steps later inside the weight adapter.
    """
    entry = OPENMIND_MODELS.get(name.strip().lower())
    if entry is None:
        raise ValueError(
            f"Unknown published checkpoint {name!r}.\n" + describe_openmind_models()
        )
    destination = Path(model_root) / entry.name
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "checkpoint_final.pth"

    for filename in _OPENMIND_FILES:
        target = destination / filename
        if target.is_file() and not overwrite:
            log.info("Already present, not re-downloading: %s", target)
            continue
        download_checkpoint(
            _HF_URL.format(repo=entry.repo, filename=filename), target, overwrite=overwrite
        )

    log.ok(f"OpenMind {entry.name} ({entry.method}) ready at {checkpoint}")
    return checkpoint


def find_ssl_checkpoint(nnssl_results: Path, *, run_name: str | None = None) -> Path:
    """Newest nnssl checkpoint under *nnssl_results*.

    Parameters
    ----------
    run_name
        Restrict to one training directory (``<Trainer>__<plans>__<configuration>``). Without
        it the most recently modified checkpoint across all runs wins, which is right for the
        common "adapt what I just pre-trained" case and wrong the moment two runs are in
        flight — hence the log line naming the file that was chosen.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists, naming the directory searched.
    """
    root = Path(nnssl_results)
    if not root.is_dir():
        raise FileNotFoundError(
            f"nnssl results directory does not exist: {root}. Run stage 2 first, or pass "
            f"--pretrain-source checkpoint with an explicit --checkpoint."
        )

    # nnssl nests results as <results>/<DatasetXXX_Name>/<run_name>/fold_all/<checkpoint>, so a
    # run name has to be matched at arbitrary depth rather than directly under the root.
    candidates: list[Path] = []
    for name in NNSSL_CHECKPOINT_NAMES:
        pattern = f"**/{run_name}/**/{name}" if run_name else f"**/{name}"
        candidates.extend(root.glob(pattern))
    if not candidates:
        where = f"{root}/{run_name}" if run_name else str(root)
        raise FileNotFoundError(
            f"No nnssl checkpoint ({', '.join(NNSSL_CHECKPOINT_NAMES)}) found under {where}."
        )

    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    if len(candidates) > 1:
        log.info("Found %d nnssl checkpoints; using the newest: %s", len(candidates), chosen)
    return chosen


def resolve_named_checkpoint(name: str, model_root: Path) -> Path:
    """Resolve a checkpoint *name* against the configured model root.

    Accepts a bare name (``openmind_resencl_spark``), a name with an extension, or a path
    relative to the model root.

    Raises
    ------
    FileNotFoundError
        Listing what *is* present, so a near-miss name is obvious.
    """
    root = Path(model_root)
    direct = root / name
    if direct.is_file():
        return direct
    for suffix in _CHECKPOINT_SUFFIXES:
        candidate = root / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    if direct.is_dir():
        for filename in NNSSL_CHECKPOINT_NAMES:
            candidate = direct / filename
            if candidate.is_file():
                return candidate

    available = sorted(
        p.name for p in root.glob("*") if p.is_dir() or p.suffix in _CHECKPOINT_SUFFIXES
    ) if root.is_dir() else []
    raise FileNotFoundError(
        f"Checkpoint {name!r} not found under {root}. "
        + (f"Available: {', '.join(available)}." if available else f"{root} is empty or absent.")
    )


def download_checkpoint(url: str, destination: Path, *, overwrite: bool = False) -> Path:
    """Download a checkpoint to *destination*.

    Raises
    ------
    RuntimeError
        On any network failure, with the manual-placement instruction attached — this host may
        simply have no route to the host serving the weights.
    """
    destination = Path(destination)
    if destination.is_file() and not overwrite:
        log.info("Checkpoint already present, not re-downloading: %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    log.info("Downloading checkpoint: %s", url)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}: {exc}. Download it on a machine with access and place "
            f"it at {destination}, then re-run with --checkpoint pointing there."
        ) from None
    partial.replace(destination)
    log.ok(f"downloaded {destination.name} ({destination.stat().st_size / 2**20:.0f} MB)")
    return destination


def resolve_checkpoint(
    *,
    source: str,
    checkpoint: Path | None = None,
    checkpoint_name: str | None = None,
    checkpoint_url: str | None = None,
    model_root: Path,
    nnssl_results: Path,
    run_name: str | None = None,
) -> Path | None:
    """Resolve ``--pretrain-source`` and its options to a checkpoint path.

    Returns
    -------
    Path or None
        ``None`` for ``source="none"`` — train from scratch.

    Raises
    ------
    ValueError, FileNotFoundError, RuntimeError
        If the request cannot be satisfied. Never falls back to a different source: silently
        training from scratch when pre-trained weights were asked for would misattribute every
        result that followed.
    """
    if source == "none":
        log.info("Pre-training source 'none': the encoder will be randomly initialised.")
        return None

    if source == "ssl":
        found = find_ssl_checkpoint(nnssl_results, run_name=run_name)
        log.info("Using stage 2 checkpoint: %s", found)
        return found

    if source != "checkpoint":
        raise ValueError(
            f"Unknown --pretrain-source {source!r}; expected 'checkpoint', 'ssl' or 'none'."
        )

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"--checkpoint does not exist: {path}")
        return path

    if checkpoint_url is not None:
        filename = checkpoint_name or Path(checkpoint_url).name or "pretrained.pth"
        if not filename.endswith(_CHECKPOINT_SUFFIXES):
            filename = f"{filename}.pth"
        return download_checkpoint(checkpoint_url, Path(model_root) / filename)

    if checkpoint_name is not None:
        # A published-registry name wins over a local file of the same name: the registry is
        # explicit about what it is, and downloads are idempotent.
        if checkpoint_name.strip().lower() in OPENMIND_MODELS:
            return resolve_openmind(checkpoint_name, model_root)
        return resolve_named_checkpoint(checkpoint_name, model_root)

    raise ValueError(
        "--pretrain-source checkpoint needs one of --checkpoint, --checkpoint-name or "
        "--checkpoint-url.\n" + describe_checkpoint_sources(model_root)
    )


def describe_checkpoint_sources(model_root: Path) -> str:
    """Operator-facing text explaining where to obtain and place a checkpoint."""
    return (
        describe_openmind_models()
        + f"\n\nThese are fetched automatically into {model_root}.\n"
        f"Alternatively place any nnssl-format checkpoint there and pass its filename to\n"
        f"--checkpoint-name, or an absolute path to --checkpoint. The adapter reads the\n"
        f"embedded adaptation plan, falls back to a sibling adaptation_plan.json, and finally\n"
        f"to the standard ResEnc-L key layout."
    )


__all__ = [
    "NNSSL_CHECKPOINT_NAMES",
    "OPENMIND_MODELS",
    "PublishedCheckpoint",
    "describe_openmind_models",
    "resolve_openmind",
    "PretrainSource",
    "describe_checkpoint_sources",
    "download_checkpoint",
    "find_ssl_checkpoint",
    "resolve_checkpoint",
    "resolve_named_checkpoint",
]
