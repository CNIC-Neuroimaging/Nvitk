"""ToPBrain stage 4: inference with topology-aware post-processing.

**Inputs**

- A trained nnU-Net run under ``<nnunet_results>/DatasetXXX_.../``
- A folder of images named ``<case>_0000.nii.gz``

**Outputs**

- ``<results_root>/stage4_infer/<run>/raw/`` — nnU-Net's argmax predictions
- ``<results_root>/stage4_infer/<run>/postprocessed/`` — after island removal
- ``.../topbrain_stage4.json`` — provenance

Post-processing is applied as a separate pass over a *retained* copy of the raw predictions, so
the effect of a threshold change can be measured (stage 6 can score either directory) without
re-running inference, which is by far the expensive half.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import click

from nvitk.core.backend import map_in_thread_pool, setup
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.stage2_train import resolve_trained_run
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import nnunet_run
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

setup(globals())

log = Logger()


def _dataset_name(label_set: str) -> str:
    """nnU-Net dataset folder name for *label_set*."""
    return f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"


def postprocess_folder(
    source_dir: Path,
    destination_dir: Path,
    *,
    label_set: str,
    min_volume_mm3: float | None,
    largest_only: bool,
    workers: int = 1,
) -> int:
    """Post-process every prediction in *source_dir*; returns the number written."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(lbl.label_map(label_set))
    cases = sorted(source_dir.glob("*.nii.gz"))
    if not cases:
        log.warning("No predictions to post-process under %s", source_dir)
        return 0

    def _one(path: Path) -> Path:
        """Post-process one prediction, preserving its geometry."""
        image = imread(path)
        cleaned = postprocess_labelmap(
            image,
            labels=labels,
            spacing=image.spacing,
            min_volume_mm3=min_volume_mm3,
            largest_only=largest_only,
        )
        out = destination_dir / path.name
        imsave(out, cleaned.astype(np.uint8))
        return out

    written = map_in_thread_pool(_one, cases, max_workers=int(workers))
    log.ok(f"post-processed {len(written)} prediction(s) -> {destination_dir}")
    return len(written)


def run_infer(
    *,
    input_dir: Path,
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    nnunet_results: Path,
    results_root: Path,
    label_set: str = "ta36",
    loss: str | None = None,
    architecture: str | None = None,
    folds: Sequence[int | str] = (0,),
    plans_identifier: str | None = None,
    configuration_name: str | None = None,
    checkpoint_name: str = "checkpoint_final.pth",
    output_name: str | None = None,
    min_volume_mm3: float | None = 5.0,
    largest_only: bool = False,
    device: str = "cuda",
    num_processes: int = 3,
    workers: int = 1,
    skip_prediction: bool = False,
) -> Path:
    """Predict and post-process; returns the post-processed output directory."""
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

    paths = TopBrainPaths(
        challenge_root=Path(input_dir),
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
            Path(input_dir), raw_dir,
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

    count = postprocess_folder(
        raw_dir, post_dir,
        label_set=label_set,
        min_volume_mm3=min_volume_mm3,
        largest_only=largest_only,
        workers=workers,
    )

    (base / "topbrain_stage4.json").write_text(
        json.dumps(
            {
                "stage": "stage4",
                "created": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_name, "label_set": label_set,
                "trainer": trainer, "loss": loss,
                "plans_identifier": plans_identifier, "configuration": configuration_name,
                "folds": [str(f) for f in folds], "checkpoint": checkpoint_name,
                "input_dir": str(input_dir),
                "raw_dir": str(raw_dir), "postprocessed_dir": str(post_dir),
                "min_volume_mm3": min_volume_mm3, "largest_only": largest_only,
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
    *, label_set: str, loss: str, folds: Sequence[int | str], plans_identifier: str,
    configuration_name: str, checkpoint_name: str, min_volume_mm3: float | None,
    largest_only: bool, device: str, num_processes: int, workers: int,
    skip_prediction: bool, backend: str, input_subdir: str = "imagesTr_topbrain",
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
        "--folds", quote_path(",".join(str(f) for f in folds)),
        "--plans-identifier", plans_identifier,
        "--configuration", configuration_name,
        "--checkpoint-name", checkpoint_name,
        "--device", device,
        "--num-processes", str(int(num_processes)),
        "--workers", str(int(workers)),
    ]
    if min_volume_mm3 is None:
        argv.append("--no-postprocess")
    else:
        argv.extend(["--min-volume-mm3", str(float(min_volume_mm3))])
    if largest_only:
        argv.append("--largest-only")
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
@click.option("--input-dir", type=click.Path(path_type=Path), required=True,
              help="Folder of <case>_0000.nii.gz images.")
@click.option("--nnunet-raw", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-preprocessed", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-results", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr"]), default="ta36",
              show_default=True)
@click.option("--loss", type=str, default=None, help="Loss the model was trained with.")
@click.option("--architecture", type=str, default=None,
              help="Encoder family (ResEncL / PrimusM). Read from stage 2 when omitted.")
@click.option("--folds", type=str, default="0", show_default=True)
@click.option("--plans-identifier", type=str, default=None)
@click.option("--configuration", "configuration_name", type=str, default=None)
@click.option("--checkpoint-name", type=str, default="checkpoint_final.pth", show_default=True)
@click.option("--output-name", type=str, default=None)
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True,
              help="Drop connected components smaller than this, per class.")
@click.option("--no-postprocess", is_flag=True, default=False)
@click.option("--largest-only", is_flag=True, default=False,
              help="Also reduce each class to its single largest component.")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-processes", type=int, default=3, show_default=True)
@click.option("--workers", type=int, default=1, show_default=True)
@click.option("--skip-prediction", is_flag=True, default=False,
              help="Only re-run post-processing over existing raw predictions.")
def main(
    input_dir: Path, nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path,
    results_root: Path, label_set: str, loss: str | None, folds: str,
    plans_identifier: str | None, configuration_name: str | None, checkpoint_name: str,
    output_name: str | None, min_volume_mm3: float, no_postprocess: bool, largest_only: bool,
    device: str | None, num_processes: int, workers: int, skip_prediction: bool,
    backend: str = "gpu",
) -> None:
    """CLI entry point: predict on a folder and post-process the result."""
    from nvitk.pipes.topbrain.stage2_train import parse_folds

    Logger()
    run_infer(
        input_dir=input_dir, nnunet_raw=nnunet_raw, nnunet_preprocessed=nnunet_preprocessed,
        nnunet_results=nnunet_results, results_root=results_root, label_set=label_set,
        loss=loss, architecture=architecture, folds=parse_folds(folds), plans_identifier=plans_identifier,
        configuration_name=configuration_name, checkpoint_name=checkpoint_name,
        output_name=output_name,
        min_volume_mm3=None if no_postprocess else min_volume_mm3,
        largest_only=largest_only, device=device or torch_device_for_backend(backend),
        num_processes=num_processes, workers=workers, skip_prediction=skip_prediction,
    )


__all__ = ["build_sge_command", "main", "postprocess_folder", "run_infer", "submit_sge"]


if __name__ == "__main__":
    main()
