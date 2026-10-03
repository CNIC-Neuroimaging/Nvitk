"""
CViT stage 4 — predict new cases with a trained (or exported) CViT.

Description
-----------
Inputs are a folder (or files) of volumes. nnU-Net-named inputs (``<case>_0000.nii.gz`` …) are
used as they are; plain single-channel files (``<case>.nii.gz``) are staged as
``<case>_0000.nii.gz`` symlinks. Prediction is nnU-Net's: Gaussian-weighted sliding window,
mirroring test-time augmentation (unless the model disabled it or ``--disable-tta``), fold
ensembling, and export resampled back onto each input's own grid.

Geometry checks
---------------
Every output is re-read with :func:`nvitk.io.imread` and compared with its input: same shape and
same affine (``allclose``), otherwise the run fails — a mask on the wrong grid is worse than no
mask. Empty predictions are warned about.

Post-processing
---------------
``--largest-component`` keeps, per label, the largest 26-connected component (backend ``ndi``).

Model selection
---------------
``--model-dir`` (an nnU-Net results folder or a stage-5 export), or ``--dataset-id`` + optional
``--run-name`` (default: the latest stage-2 run).
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from math import prod
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core import setup
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.pipes._engines import nnunet_run
from nvitk.pipes._engines.env import nnunet_env
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.stage0_dataprep import IMAGE_SUFFIXES, _CHANNEL_RE, _affine_close, _split_suffix
from nvitk.pipes.cvit.stage2_train import resolve_run
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import parse_str_list, paths_from_options, pop_roots, root_options
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.cvit.util.sge_stage import (
    container_layout,
    python_module_argv,
    quote_path,
    root_args,
    run_driver_script,
    sge_backend_cli_args,
    submit_stage_job,
    to_container_path,
    torch_device_for_backend,
)

setup(globals())
log = Logger()

# ---------------------------------------------------------------------------
# Input staging
# ---------------------------------------------------------------------------


def stage_inputs(inputs: Sequence[Path], staging: Path, n_channels: int,
                 file_ending: str = ".nii.gz") -> dict[str, list[Path]]:
    """Map input files to ``{case: [channel paths]}``, staged as ``<case>_000N<file_ending>``.

    Files already in the model's format are symlinked; others are converted with
    :func:`nvitk.io.imsave` (voxels and geometry) — nnU-Net silently skips any other extension.

    Raises
    ------
    ValueError
        On inconsistent channel counts or a multi-channel model fed un-numbered files.
    """
    files: list[Path] = []
    for item in inputs:
        item = Path(item)
        files += sorted(p for p in item.iterdir() if p.is_file()) if item.is_dir() else [item]
    cases: dict[str, dict[int, Path]] = {}
    for f in files:
        stem, suffix = _split_suffix(f)
        if suffix not in IMAGE_SUFFIXES:
            continue
        m = _CHANNEL_RE.match(stem)
        case, ch = (m["case"], int(m["ch"])) if m else (stem, None)
        if ch is None:
            if n_channels != 1:
                raise ValueError(f"{f.name}: the model takes {n_channels} channels; name inputs <case>_000N.")
            ch = 0
        cases.setdefault(case, {})[ch] = f
    if not cases:
        raise FileNotFoundError(f"No input volumes in {[str(i) for i in inputs]}.")
    staging.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[Path]] = {}
    for case, chans in sorted(cases.items()):
        if sorted(chans) != list(range(n_channels)):
            raise ValueError(f"{case}: channels {sorted(chans)} but the model needs 0..{n_channels - 1}.")
        out[case] = []
        for ch in range(n_channels):
            src = chans[ch]
            dst = staging / f"{case}_{ch:04d}{file_ending}"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            if _split_suffix(src)[1] == file_ending:
                os.symlink(src.resolve(), dst)
            else:
                imsave(dst, imread(src, backend="cpu"))
            out[case].append(src)
    return out


# ---------------------------------------------------------------------------
# Post-processing + checks
# ---------------------------------------------------------------------------


def keep_largest_component(seg) -> Any:
    """Per label, keep the largest 26-connected component (backend arrays in, out)."""
    seg = as_backend_array(seg)
    out = np.zeros_like(seg)
    structure = np.ones((3,) * seg.ndim, dtype=bool)
    for value in to_numpy(np.unique(seg)).tolist():
        if value == 0:
            continue
        comp, n = ndi.label(seg == value, structure=structure)
        if int(n) == 0:
            continue
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        out[comp == int(to_numpy(sizes.argmax()))] = value
    return out


def check_and_postprocess(cases: dict[str, list[Path]], out_dir: Path, *, largest_component: bool) -> list[dict[str, Any]]:
    """Verify each prediction's grid against its input; optional largest-component filter."""
    report = []
    for case, inputs in cases.items():
        pred_path = out_dir / f"{case}.nii.gz"
        if not pred_path.is_file():
            raise FileNotFoundError(f"nnU-Net wrote no prediction for {case} ({pred_path}).")
        ref, pred = imread(inputs[0], backend="cpu"), imread(pred_path, backend="cpu")
        if tuple(ref.data.shape) != tuple(pred.data.shape) or not _affine_close(ref.affine, pred.affine):
            raise ValueError(f"{case}: prediction grid differs from the input's (shape/affine).")
        data = as_backend_array(pred.data)
        if largest_component:
            data = keep_largest_component(data)
            imsave(pred_path, pred.with_data(data.astype(pred.data.dtype)))
        n_fg = int(to_numpy((data != 0).sum()))
        if n_fg == 0:
            log.warning("%s: empty prediction.", case)
        # Physical volume (ml) from the voxel spacing in mm, not a voxel count.
        voxel_ml = prod(float(s) for s in (pred.spacing or (1.0,) * data.ndim)) / 1000.0
        report.append({"case": case, "foreground_voxels": n_fg, "foreground_ml": n_fg * voxel_ml})
    return report


# ---------------------------------------------------------------------------
# Library entry point
# ---------------------------------------------------------------------------


def resolve_model_dir(paths: CViTPaths, *, model_dir: Path | None, dataset_id: int, dataset_name: str,
                      run_name: str | None) -> tuple[Path, list[str]]:
    """``(model_folder, trained_folds)``."""
    if model_dir is not None:
        folds = sorted(p.name.split("_", 1)[1] for p in Path(model_dir).glob("fold_*") if p.is_dir())
        return Path(model_dir), folds
    dataset = pth.dataset_folder_name(dataset_id, dataset_name)
    marker = resolve_run(paths, dataset, run_name)
    return Path(paths.nnunet_results) / dataset / marker["run_name"], list(marker["folds"])


def run_infer(
    *,
    paths: CViTPaths,
    inputs: Sequence[Path],
    output_dir: Path | None = None,
    model_dir: Path | None = None,
    dataset_id: int = pth.DEFAULT_DATASET_ID,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    run_name: str | None = None,
    folds: Sequence[str] | None = None,
    checkpoint: str = "checkpoint_final.pth",
    disable_tta: bool = False,
    largest_component: bool = False,
    device: str = "cuda",
    num_processes: int = 3,
) -> Path:
    """Predict *inputs*; returns the output directory."""
    model, trained = resolve_model_dir(paths, model_dir=model_dir, dataset_id=dataset_id,
                                       dataset_name=dataset_name, run_name=run_name)
    folds = list(folds or trained)
    if not folds:
        raise FileNotFoundError(f"No trained folds under {model}.")
    dataset_json = json.loads((model / "dataset.json").read_text(encoding="utf-8"))
    n_channels = len(dataset_json["channel_names"])
    out = Path(output_dir or paths.results_root / pth.STAGE4_INFER_DIR / model.name)
    out.mkdir(parents=True, exist_ok=True)
    staging = out / ".inputs"
    cases = stage_inputs(inputs, staging, n_channels, dataset_json.get("file_ending", ".nii.gz"))
    log.info("stage4 | %d case(s), model %s, folds %s", len(cases), model.name, folds)

    env = nnunet_env(paths.nnunet_raw, paths.nnunet_preprocessed, paths.nnunet_results)
    nnunet_run.predict_model_folder(staging, out, env=env, model_folder=model, folds=folds, device=device,
                                    checkpoint_name=checkpoint, num_processes=num_processes,
                                    disable_tta=disable_tta)
    report = check_and_postprocess(cases, out, largest_component=largest_component)
    shutil.rmtree(staging, ignore_errors=True)
    (out / "cvit_stage4.json").write_text(json.dumps({
        "stage": "stage4", "created": datetime.now().isoformat(timespec="seconds"),
        "model": str(model), "folds": folds, "checkpoint": checkpoint, "disable_tta": disable_tta,
        "largest_component": largest_component, "cases": report,
    }, indent=2) + "\n", encoding="utf-8")
    log.ok(f"stage4: {len(cases)} prediction(s) -> {out}")
    return out


# ---------------------------------------------------------------------------
# SGE + CLI
# ---------------------------------------------------------------------------


def _worker_argv(paths: CViTPaths, **o: Any) -> list[str]:
    argv = [*python_module_argv("nvitk.pipes.cvit.stage4_infer"),
            *sge_backend_cli_args(o.get("backend", "gpu")), *root_args(container_layout()),
            "--submit", "local",
            "--dataset-id", str(o.get("dataset_id", pth.DEFAULT_DATASET_ID)),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME)),
            "--checkpoint", o.get("checkpoint", "checkpoint_final.pth"),
            "--device", torch_device_for_backend(o.get("backend", "gpu"), device=o.get("device"), remote=True)]
    for item in o.get("inputs") or ():
        argv += ["-i", quote_path(to_container_path(paths, item) or item)]
    for key, flag in (("output_dir", "-o"), ("model_dir", "--model-dir")):
        if o.get(key):
            argv += [flag, quote_path(to_container_path(paths, o[key]) or o[key])]
    if o.get("run_name"):
        argv += ["--run-name", quote_path(o["run_name"])]
    if o.get("folds"):
        argv += ["--folds", ",".join(str(f) for f in o["folds"])]
    for key, flag in (("disable_tta", "--disable-tta"), ("largest_component", "--largest-component")):
        if o.get(key):
            argv.append(flag)
    return argv


def _data_paths(paths: CViTPaths, **o: Any) -> list[Path]:
    items = [*(o.get("inputs") or ()), o.get("output_dir"), o.get("model_dir")]
    return [Path(p) for p in items if p and to_container_path(paths, p) is None]


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    return submit_stage_job("stage4", _worker_argv(paths, **o), paths=paths, container=container,
                            src_dir=src_dir, backend=o.get("backend", "gpu"),
                            data_paths=_data_paths(paths, **o), job_suffix=str(o.get("dataset_id", "")),
                            hold_jid=hold_jid, dry_run=dry_run, emit=emit)


@click.command("nvitk-cvit-infer")
@config_dir_click_option()
@backend_click_option(default="gpu")
@root_options
@click.option("-i", "--input", "inputs", multiple=True, type=click.Path(path_type=Path), required=True,
              help="Input folder or file (repeatable).")
@click.option("-o", "--output", "output_dir", type=click.Path(path_type=Path), default=None,
              help="Default: <results_root>/stage4_infer/<model>.")
@click.option("--model-dir", type=click.Path(path_type=Path), default=None,
              help="nnU-Net results folder or stage-5 export (overrides --dataset-id/--run-name).")
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--run-name", type=str, default=None)
@click.option("--folds", type=str, default=None, help="Default: every trained fold (ensemble).")
@click.option("--checkpoint", type=str, default="checkpoint_final.pth", show_default=True)
@click.option("--disable-tta", is_flag=True, default=False)
@click.option("--largest-component", is_flag=True, default=False)
@click.option("--device", type=str, default=None)
@click.option("--submit", type=click.Choice(["local", "sge"]), default="local", show_default=True)
@click.option("--container", type=click.Path(path_type=Path), default=None)
@click.option("--src-dir", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--no-remote", is_flag=True, default=False)
def main(backend: str, folds: str | None, device: str | None, submit: str, container: Path | None,
         src_dir: Path | None, dry_run: bool, no_remote: bool, **options: Any) -> None:
    """Segment new volumes with a trained CViT."""
    roots = pop_roots(options)
    folds_list = parse_str_list(folds) or None
    if submit == "local":
        run_infer(paths=paths_from_options(roots), folds=folds_list,
                  device=torch_device_for_backend(backend, device=device), **options)
        return
    paths = pth.layout_cluster(**roots)
    image = container or cfg.CONTAINER_PATH
    if image is None:
        raise click.ClickException("No container: pass --container or set pipelines.cvit.default_sge_container_root.")
    run_driver_script(
        lambda fh: submit_sge(paths=paths, container=Path(image), src_dir=src_dir, dry_run=True, emit=fh,
                              backend=backend, folds=folds_list, device=device, **options),
        title="cvit stage4 inference", basename="submit_cvit_infer", dry_run=dry_run, no_remote=no_remote,
    )


if __name__ == "__main__":
    main()
