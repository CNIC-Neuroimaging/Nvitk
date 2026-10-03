"""
CViT stage 5 — export a trained CViT as a portable, self-contained model bundle.

Description
-----------
Writes ``<results_root>/stage5_export/<name>/``::

    dataset.json  plans.json  [dataset_fingerprint.json]
    fold_<k>/checkpoint_final.pth     network weights only (optimiser / logging stripped)
    cvit_config.json                  the CViT architecture
    export.json                       provenance (run, folds, plans, sizes, checksums)
    cvit_nn/                          verbatim copy of nvitk.nn (torch-only, relative imports)
    predict.py                        standalone predictor: released nnunetv2 + torch + cvit_nn
    README.md

Two ways to use it
------------------
- with nvitk: ``nvitk-cvit-infer --model-dir <export> -i <inputs>`` (it is an nnU-Net model folder);
- without nvitk: ``python predict.py -i <inputs> -o <outputs>`` — the network is rebuilt from
  ``cvit_nn`` and handed to nnU-Net's predictor through ``manual_initialization``, so neither
  the CViT trainer classes nor the in-tree nnU-Net are needed. This works because
  :mod:`nvitk.nn` imports nothing from nvitk.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.stage2_train import resolve_run
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import parse_str_list, paths_from_options, pop_roots, root_options
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.cvit.util.sge_stage import (
    container_layout,
    python_module_argv,
    quote_path,
    root_args,
    sge_backend_cli_args,
    submit_stage_job,
)

log = Logger()

#: Checkpoint keys kept in the export (nnU-Net's predictor needs exactly these).
KEEP_KEYS: tuple[str, ...] = ("network_weights", "init_args", "trainer_name",
                              "inference_allowed_mirroring_axes", "current_epoch")

PREDICT_SCRIPT = '''"""Standalone CViT predictor (exported by nvitk-cvit stage5).

Requires: torch, nnunetv2 (the released package is fine). No nvitk needed.

    python predict.py -i <input folder of <case>_0000.nii.gz> -o <output folder> [-f 0 1] [-device cuda]
"""
import argparse
import sys
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cvit_nn.cvit import build_cvit  # noqa: E402


def main():
    meta = load_json(str(ROOT / "export.json"))
    ap = argparse.ArgumentParser(description="CViT inference")
    ap.add_argument("-i", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("-f", nargs="+", default=meta["folds"])
    ap.add_argument("-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--disable_tta", action="store_true")
    ap.add_argument("-step_size", type=float, default=0.5)
    a = ap.parse_args()

    dataset_json = load_json(str(ROOT / "dataset.json"))
    pm = PlansManager(load_json(str(ROOT / "plans.json")))
    cm = pm.get_configuration(meta["configuration"])
    n_in = determine_num_input_channels(pm, cm, dataset_json)
    n_out = pm.get_label_manager(dataset_json).num_segmentation_heads
    net = build_cvit(cm.network_arch_init_kwargs, input_channels=n_in, num_classes=n_out,
                     deep_supervision=False)
    params, mirror = [], None
    for f in a.f:
        ckpt = torch.load(str(ROOT / f"fold_{f}" / "checkpoint_final.pth"), map_location="cpu",
                          weights_only=False)
        params.append(ckpt["network_weights"])
        mirror = ckpt.get("inference_allowed_mirroring_axes")
    net.load_state_dict(params[0])
    predictor = nnUNetPredictor(tile_step_size=a.step_size, use_gaussian=True,
                                use_mirroring=not a.disable_tta, device=torch.device(a.device))
    predictor.manual_initialization(net, pm, cm, params, dataset_json, meta["trainer"], mirror)
    predictor.predict_from_files(a.i, a.o, save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2)


if __name__ == "__main__":
    main()
'''

README = """# CViT model export: {name}

Convolutional Vision Transformer trained with nvitk-cvit (`{run}`), folds {folds}.

* With nvitk: `nvitk-cvit-infer --model-dir . -i <inputs> -o <outputs>`
* Without nvitk (torch + nnunetv2 only): `python predict.py -i <inputs> -o <outputs>`

Inputs follow nnU-Net naming (`<case>_0000.nii.gz`, one file per channel: {channels}).
Architecture: `cvit_config.json`. Provenance and checksums: `export.json`.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_export(
    *,
    paths: CViTPaths,
    dataset_id: int = pth.DEFAULT_DATASET_ID,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    run_name: str | None = None,
    folds: Sequence[str] | None = None,
    name: str | None = None,
    checkpoint: str = "checkpoint_final.pth",
    make_zip: bool = False,
) -> Path:
    """Export a stage-2 run; returns the export directory."""
    import torch

    dataset = pth.dataset_folder_name(dataset_id, dataset_name)
    marker = resolve_run(paths, dataset, run_name)
    src = paths.nnunet_results / dataset / marker["run_name"]
    folds = list(folds or marker["folds"])
    target = paths.results_root / pth.STAGE5_EXPORT_DIR / (name or f"{dataset}__{marker['plans']}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for fname in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        if (src / fname).is_file():
            shutil.copyfile(src / fname, target / fname)
    if not (target / "plans.json").is_file():
        raise FileNotFoundError(f"{src}/plans.json missing — is {marker['run_name']} trained?")

    checksums: dict[str, str] = {}
    for fold in folds:
        ckpt_path = src / f"fold_{fold}" / checkpoint
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"{ckpt_path} not found (fold {fold} not finished?).")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
        out = target / f"fold_{fold}" / "checkpoint_final.pth"
        out.parent.mkdir(parents=True)
        torch.save(slim, out)
        checksums[f"fold_{fold}/checkpoint_final.pth"] = _sha256(out)

    nn_src = Path(__file__).resolve().parents[2] / "nn"
    shutil.copytree(nn_src, target / "cvit_nn", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "predict.py").write_text(PREDICT_SCRIPT, encoding="utf-8")
    (target / "cvit_config.json").write_text(json.dumps(marker["cvit_config"], indent=2) + "\n", encoding="utf-8")
    dataset_json = json.loads((target / "dataset.json").read_text(encoding="utf-8"))
    meta = {
        "stage": "stage5", "created": datetime.now().isoformat(timespec="seconds"),
        "run_name": marker["run_name"], "trainer": marker["trainer"], "plans": marker["plans"],
        "configuration": marker["configuration"], "folds": folds, "source": str(src),
        "checksums": checksums,
    }
    (target / "export.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(README.format(
        name=target.name, run=marker["run_name"], folds=folds,
        channels=", ".join(dataset_json["channel_names"].values())), encoding="utf-8")
    if make_zip:
        archive = shutil.make_archive(str(target), "zip", root_dir=target.parent, base_dir=target.name)
        log.info("stage5: archive %s", archive)
    log.ok(f"stage5: exported {len(folds)} fold(s) -> {target}")
    return target


# ---------------------------------------------------------------------------
# SGE + CLI
# ---------------------------------------------------------------------------


def _worker_argv(**o: Any) -> list[str]:
    argv = [*python_module_argv("nvitk.pipes.cvit.stage5_export"),
            *sge_backend_cli_args("cpu"), *root_args(container_layout()),
            "--dataset-id", str(o.get("dataset_id", pth.DEFAULT_DATASET_ID)),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME))]
    for key, flag in (("run_name", "--run-name"), ("name", "--name")):
        if o.get(key):
            argv += [flag, quote_path(o[key])]
    if o.get("folds"):
        argv += ["--folds", ",".join(str(f) for f in o["folds"])]
    if o.get("make_zip"):
        argv.append("--zip")
    return argv


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    return submit_stage_job("stage5", _worker_argv(**o), paths=paths, container=container,
                            src_dir=src_dir, backend="cpu", request_gpu=False,
                            job_suffix=str(o.get("dataset_id", "")), hold_jid=hold_jid,
                            dry_run=dry_run, emit=emit)


@click.command("cvit-stage5-export")
@config_dir_click_option()
@backend_click_option(default="cpu")
@root_options
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--run-name", type=str, default=None)
@click.option("--folds", type=str, default=None)
@click.option("--name", type=str, default=None)
@click.option("--checkpoint", type=str, default="checkpoint_final.pth", show_default=True)
@click.option("--zip", "make_zip", is_flag=True, default=False)
def main(backend: str, folds: str | None, **options: Any) -> None:
    """Export a trained CViT as a portable bundle."""
    run_export(paths=paths_from_options(pop_roots(options)), folds=parse_str_list(folds) or None, **options)


if __name__ == "__main__":
    main()
