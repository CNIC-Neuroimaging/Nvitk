"""ToPBrain stage 5: build the Grand Challenge algorithm container.

**Inputs**

- A trained nnU-Net run under ``<nnunet_results>/DatasetXXX_.../<trainer>__<plans>__<config>/``

**Outputs**

- ``<results_root>/stage5_package/<name>/`` — the assembled build context
- ``.../<name>.tar.gz`` when ``--save`` is given, ready to upload
- ``.../topbrain_stage5.json`` — provenance

The build context is assembled and left on disk whether or not Docker is available, so the
image can be built on a different machine — the analysis host and the machine with a Docker
daemon are frequently not the same one.

Submission constraints (from the challenge's template): container ≤10 GB, ≤31 GiB DRAM, one
``.mha`` in and one ``.mha`` out with identical shape, no network at run time.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import click

import nvitk
from nvitk.core.logger import Logger
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain.stage2_train import resolve_trained_run
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util.paths import DATASET_IDS, DATASET_SUFFIXES, STAGE5_PACKAGE_DIR
from nvitk.pipes.topbrain.util.sge_stage import build_stage_command, submit_stage_job

log = Logger()

#: Files nnU-Net needs in a model folder for ``initialize_from_trained_model_folder``.
MODEL_FILES: tuple[str, ...] = ("dataset.json", "plans.json")

#: Checkpoint copied per fold.
CHECKPOINT_NAME: str = "checkpoint_final.pth"

#: Grand Challenge's hard image-size ceiling, in gibibytes.
MAX_IMAGE_GIB: float = 10.0


def _docker_assets_dir() -> Path:
    """Directory holding the Dockerfile, entry point and requirements."""
    return Path(__file__).resolve().parent / "docker"


def collect_model(
    run_dir: Path, destination: Path, *, folds: Sequence[int | str], checkpoint: str
) -> list[str]:
    """Copy a trained nnU-Net run into an inference-ready model folder.

    nnU-Net's predictor expects ``plans.json`` and ``dataset.json`` beside ``fold_N/`` — note
    it wants them named exactly that, not ``nnUNetResEncUNetLPlans.json``, so the plans file is
    renamed on the way in.

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
        p for p in run_dir.glob("*.json") if p.name not in ("dataset.json", "dataset_fingerprint.json")
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
    for fold in folds:
        name = f"fold_{fold}"
        source = run_dir / name / checkpoint
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source}")
        (destination / name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination / name / checkpoint)
        copied.append(name)

    total = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    log.info("Model folder: %s folds, %.2f GiB -> %s", copied, total / 2**30, destination)
    return copied


def run_package(
    *,
    nnunet_results: Path,
    results_root: Path,
    label_set: str = "ta36",
    loss: str | None = None,
    architecture: str | None = None,
    plans_identifier: str | None = None,
    configuration_name: str | None = None,
    folds: Sequence[int | str] = (0,),
    checkpoint: str = CHECKPOINT_NAME,
    name: str = "topbrain-ta36",
    tag: str = "latest",
    build: bool = False,
    save: bool = False,
) -> Path:
    """Assemble (and optionally build) the submission container; returns the context directory."""
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
    dataset_name = f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"
    run_dir = (
        Path(nnunet_results) / dataset_name / f"{trainer}__{plans_identifier}__{configuration_name}"
    )

    context = Path(results_root) / STAGE5_PACKAGE_DIR / name
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True, exist_ok=True)

    log.info("topbrain stage7 | run=%s -> %s", run_dir.name, context)

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

    # ---- 3. nvitk source (the container imports it for harmonisation + post-processing) ----
    nvitk_source = Path(nvitk.__file__).resolve().parent
    shutil.copytree(
        nvitk_source,
        context / "nvitk",
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "gui", "meshlab", "registry",
            # Pre-training-only, and hundreds of MB an inference image never uses. The nnunet
            # tree is excluded here because it is copied to the context root instead, where it
            # shadows the released package on PYTHONPATH.
            "nnssl", "nnunet",
        ),
    )

    # ---- 4. Trained weights --------------------------------------------------
    copied = collect_model(run_dir, context / "model", folds=folds, checkpoint=checkpoint)

    context_size = sum(p.stat().st_size for p in context.rglob("*") if p.is_file())
    log.info("Build context: %.2f GiB", context_size / 2**30)
    if context_size / 2**30 > MAX_IMAGE_GIB:
        log.warning(
            "Build context is already %.2f GiB; the challenge limit for the *image* is %.0f GiB. "
            "Drop folds with --folds to fit.",
            context_size / 2**30, MAX_IMAGE_GIB,
        )

    image = f"{name}:{tag}"
    archive: Path | None = None
    if build or save:
        _require_docker()
        log.info("Building %s ...", image)
        subprocess.run(["docker", "build", "-t", image, str(context)], check=True)
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
                "dataset": dataset_name, "label_set": label_set,
                "trainer": trainer, "loss": loss,
                "plans_identifier": plans_identifier, "configuration": configuration_name,
                "run_dir": str(run_dir), "folds": copied, "checkpoint": checkpoint,
            "trainer": trainer, "architecture": architecture,
            "bundled_nnunet": "in-tree build (released nnunetv2 not installed)",
                "image": image, "context": str(context),
                "context_bytes": context_size,
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
    folds: Sequence[int | str], checkpoint: str, name: str, tag: str,
    build: bool, save: bool, backend: str,
) -> list[str]:
    """Worker argv for stage 7, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv
    from nvitk.pipes.topbrain.util.sge_stage import container_layout, quote_path

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage5_package"),
        "--nnunet-results", quote_path(inside.nnunet_results),
        "--results-root", quote_path(inside.results_root),
        "--label-set", label_set,
        "--loss", quote_path(loss),
        "--folds", quote_path(",".join(str(f) for f in folds)),
        "--checkpoint", checkpoint,
        "--name", quote_path(name),
        "--tag", quote_path(tag),
    ]
    # Omitted when unknown: the worker reads them from stage 2's provenance, which is the only
    # place that knows the spacing preprocess_like_nnssl settled on at run time.
    if plans_identifier:
        argv.extend(["--plans-identifier", quote_path(plans_identifier)])
    if configuration_name:
        argv.extend(["--configuration", quote_path(configuration_name)])
    # Deliberately never --build on the cluster: Docker is not available inside Singularity.
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **options) -> str:
    """Host shell command for the stage 7 SGE task (context assembly only)."""
    return build_stage_command(
        "stage5", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **options,
) -> str:
    """Emit or submit the stage 7 SGE job."""
    return submit_stage_job(
        "stage5", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
        hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage5-package")
@click.option("--nnunet-results", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr"]), default="ta36",
              show_default=True)
@click.option("--loss", type=str, default=None)
@click.option("--architecture", type=str, default=None,
              help="Encoder family (ResEncL / PrimusM). Read from stage 2 when omitted.")
@click.option("--plans-identifier", type=str, default=None)
@click.option("--configuration", "configuration_name", type=str, default=None)
@click.option("--folds", type=str, default="0", show_default=True)
@click.option("--checkpoint", type=str, default=CHECKPOINT_NAME, show_default=True)
@click.option("--name", type=str, default="topbrain-ta36", show_default=True)
@click.option("--tag", type=str, default="latest", show_default=True)
@click.option("--build", is_flag=True, default=False, help="Run 'docker build'.")
@click.option("--save", is_flag=True, default=False,
              help="Build, then write a tar.gz for upload to Grand Challenge.")
def main(
    nnunet_results: Path, results_root: Path, label_set: str, loss: str | None,
    plans_identifier: str | None, configuration_name: str | None, folds: str,
    checkpoint: str, name: str, tag: str, build: bool, save: bool,
) -> None:
    """CLI entry point: assemble (and optionally build) the submission container."""
    from nvitk.pipes.topbrain.stage2_train import parse_folds

    Logger()
    run_package(
        nnunet_results=nnunet_results, results_root=results_root, label_set=label_set,
        # architecture comes from stage 2's provenance, not from a flag.
        loss=loss, architecture=None, plans_identifier=plans_identifier,
        configuration_name=configuration_name,
        folds=parse_folds(folds), checkpoint=checkpoint, name=name, tag=tag,
        build=build, save=save,
    )


__all__ = ["CHECKPOINT_NAME", "build_sge_command", "collect_model", "main", "run_package",
           "submit_sge"]


if __name__ == "__main__":
    main()
