"""
``nvitk-cvit`` — master CLI of the CViT pipeline (all stages, local or SGE).

Description
-----------
One command drives every stage; ``--stages`` selects which (ids or aliases, any order — they are
re-sorted into pipeline order). Locally, stages run in-process under
:class:`~nvitk.core.logger.PipelineRunTracker`. With ``--submit sge`` each stage becomes one job
in a ``-hold_jid`` chain, written into a single driver script that is executed where ``qsub``
exists (locally, or on the login node over SSH); ``--parallel-folds`` turns training into one
preparation job plus one job per fold.

Examples
--------
Validate a dataset and train fold 0 of a CViT-B with the default hierarchical tokenizer::

    nvitk-cvit --stages dataprep,train --data-root /data/MyTask --dataset-name MyTask \\
        --channel-names CT --folds 0

The tokenizer ablation (same everything, three tokenizers)::

    for t in hierarchical intra_patch linear; do
        nvitk-cvit --stages train,evaluate,probe --tokenizer $t --folds 0,1,2,3,4
    done

Self-supervised pre-training on the training images plus an external cohort, then fine-tuning::

    nvitk-cvit --stages dataprep,pretrain,train --corpus-from-train \\
        --corpus-source ext:mr=/data/unlabelled --ssl simmim --ssl-epochs 300 --llrd 0.75
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger, PipelineRunTracker
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit import stages as st
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import (
    parse_float_list,
    parse_int_list,
    parse_str_list,
    paths_from_options,
    pop_roots,
    root_options,
)
from nvitk.pipes.cvit.util.sge_stage import run_driver_script, torch_device_for_backend

log = Logger()

#: CLI flag → CViTConfig field for the architecture options (only set flags override).
_ARCH_FLAGS: dict[str, str] = {
    "tokenizer": "tokenizer", "decoder": "decoder", "skips": "skips",
    "skip_drop": "skip_drop_prob", "skip_schedule": "skip_schedule",
    "skip_warmup_epochs": "skip_warmup_epochs", "skip_gate": "skip_gate",
    "skip_gate_l1": "skip_gate_l1", "num_registers": "num_registers",
    "drop_path": "drop_path_rate",
}


def _arch_overrides(o: dict[str, Any]) -> dict[str, Any]:
    out = {field: o[flag] for flag, field in _ARCH_FLAGS.items() if o.get(flag) is not None}
    if o.get("arch_json"):
        out.update(json.loads(o["arch_json"]))
    return out


def _bundle_name(o: dict[str, Any]) -> str:
    return o.get("ssl_name") or f"{o['ssl']}_{o.get('arch') or 'custom'}"


# ---------------------------------------------------------------------------
# Per-stage options
# ---------------------------------------------------------------------------


def stage_options(o: dict[str, Any], paths: Any) -> dict[str, dict[str, Any]]:
    """Keyword arguments for each stage's ``run_*`` / ``submit_sge``, from the master flags."""
    from nvitk.pipes.cvit.stage1_pretrain import bundle_dir

    common = {"dataset_id": o["dataset_id"], "dataset_name": o["dataset_name"], "backend": o["backend"]}
    arch = _arch_overrides(o)
    from_bundle = o.get("from_bundle")
    if from_bundle is None and o["ssl"] != "none":
        from_bundle = bundle_dir(paths.results_root, _bundle_name(o))
    return {
        st.STAGE_DATAPREP: {
            **common, "images_dir": o.get("images_dir"), "labels_dir": o.get("labels_dir"),
            "channel_names": parse_str_list(o.get("channel_names")) or None, "labels": o.get("labels"),
            "num_folds": o["num_folds"], "seed": o["seed"], "group_regex": o.get("group_regex"),
            "allow_empty": o["allow_empty"], "overwrite": o["overwrite"], "workers": o["workers"],
            "corpus_sources": list(o.get("corpus_sources") or ()), "corpus_from_train": o["corpus_from_train"],
            "corpus_id": o["corpus_id"], "corpus_harmonize": o["corpus_harmonize"],
            "skip_dataset": o["corpus_only"],
        },
        st.STAGE_PRETRAIN: {
            # The corpus, not the labelled dataset, identifies stage 1's data.
            "dataset_name": o["dataset_name"], "backend": o["backend"], "corpus_id": o["corpus_id"], "name": _bundle_name(o) if o["ssl"] != "none" else None,
            "method": o["ssl"], "arch": o.get("arch"), "arch_overrides": arch or None,
            "configuration": o["ssl_configuration"],
            "patch_size": parse_int_list(o["ssl_patch_size"]) or list(cfg.DEFAULT_SSL_PATCH),
            "batch_size": o.get("ssl_batch_size"), "num_epochs": o.get("ssl_epochs"),
            "iterations_per_epoch": o.get("iterations_per_epoch"), "lr": o["lr"],
            "warmup_epochs": o["warmup_epochs"], "mask_ratio": o["mask_ratio"],
            "device": o.get("device"), "num_processes": o["num_processes"],
            "continue_training": o["continue_training"], "overwrite": o["overwrite"],
        },
        st.STAGE_TRAIN: {
            **common, "configuration": o["configuration"], "baseline_planner": o["baseline_planner"],
            "arch": o.get("arch"), "arch_overrides": arch or None, "token_stride": o["token_stride"],
            "patch_size": parse_int_list(o.get("patch_size")) or None, "batch_size": o.get("batch_size"),
            "from_bundle": from_bundle, "llrd": o["llrd"], "loss": o["loss"],
            "loss_config": json.loads(o["loss_config"]) if o.get("loss_config") else None,
            "folds": parse_str_list(o["folds"]), "num_epochs": o.get("epochs"),
            "iterations_per_epoch": o.get("iterations_per_epoch"), "lr": o["lr"],
            "warmup_epochs": o["warmup_epochs"], "probe_every": o["probe_every"],
            "no_mirror": o["no_mirror"], "tag": o.get("tag"), "device": o.get("device"),
            "num_processes": o["num_processes"], "continue_training": o["continue_training"],
        },
        st.STAGE_EVALUATE: {**common, "run_name": o.get("run_name"), "workers": o["workers"]},
        st.STAGE_PROBE: {
            **common, "run_name": o.get("run_name"), "folds": parse_str_list(o.get("probe_folds")) or None,
            "interventions": parse_str_list(o["probe_interventions"]),
            "local_radii_mm": parse_float_list(o["local_radius_mm"]), "layer_sweep": o["layer_sweep"],
            "max_cases": o["probe_max_cases"], "no_mirror": o["no_mirror"], "device": o.get("device"),
        },
        st.STAGE_INFER: {
            **common, "inputs": list(o.get("infer_inputs") or ()), "output_dir": o.get("infer_output"),
            "run_name": o.get("run_name"), "disable_tta": o["disable_tta"],
            "largest_component": o["largest_component"], "device": o.get("device"),
        },
        st.STAGE_EXPORT: {**common, "run_name": o.get("run_name"), "name": o.get("export_name"),
                          "make_zip": o["zip"]},
    }


# ---------------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------------


def _local_runners(paths: Any, opts: dict[str, dict[str, Any]], backend: str) -> dict[str, Callable[[], Any]]:
    from nvitk.pipes.cvit import (
        stage0_dataprep, stage1_pretrain, stage2_train, stage3_evaluate, stage3b_probe,
        stage4_infer, stage5_export,
    )

    def _dev(d: dict[str, Any]) -> dict[str, Any]:
        d = {k: v for k, v in d.items() if k != "backend"}
        if "device" in d:
            d["device"] = torch_device_for_backend(backend, device=d["device"])
        return d

    return {
        st.STAGE_DATAPREP: lambda: stage0_dataprep.run_dataprep(paths=paths, **opts[st.STAGE_DATAPREP]),
        st.STAGE_PRETRAIN: lambda: stage1_pretrain.run_pretrain(paths=paths, **_dev(opts[st.STAGE_PRETRAIN])),
        st.STAGE_TRAIN: lambda: stage2_train.run_train(paths=paths, **_dev(opts[st.STAGE_TRAIN])),
        st.STAGE_EVALUATE: lambda: stage3_evaluate.run_evaluate(paths=paths, **_dev(opts[st.STAGE_EVALUATE])),
        st.STAGE_PROBE: lambda: stage3b_probe.run_probe(paths=paths, **_dev(opts[st.STAGE_PROBE])),
        st.STAGE_INFER: lambda: stage4_infer.run_infer(paths=paths, **_dev(opts[st.STAGE_INFER])),
        st.STAGE_EXPORT: lambda: stage5_export.run_export(paths=paths, **_dev(opts[st.STAGE_EXPORT])),
    }


def run_local(paths: Any, selected: Sequence[str], opts: dict[str, dict[str, Any]], backend: str) -> None:
    runners = _local_runners(paths, opts, backend)
    with PipelineRunTracker(log, "cvit", ["(cohort)"], list(selected), stage_labels=st.STAGE_LABELS) as run:
        for stage in selected:
            run.run_stage("(cohort)", stage, runners[stage], reraise=True)


# ---------------------------------------------------------------------------
# SGE execution
# ---------------------------------------------------------------------------


def run_sge(paths: Any, selected: Sequence[str], opts: dict[str, dict[str, Any]], *, container: Path,
            src_dir: Path | None, base_hold: str | None, parallel_folds: bool, emit: TextIO) -> list[str]:
    """Emit every stage's job into *emit*, each holding on its predecessor's jobs."""
    from nvitk.pipes.cvit import (
        stage0_dataprep, stage1_pretrain, stage2_train, stage3_evaluate, stage3b_probe,
        stage4_infer, stage5_export,
    )

    submitters = {
        st.STAGE_DATAPREP: stage0_dataprep.submit_sge, st.STAGE_PRETRAIN: stage1_pretrain.submit_sge,
        st.STAGE_TRAIN: stage2_train.submit_sge, st.STAGE_EVALUATE: stage3_evaluate.submit_sge,
        st.STAGE_PROBE: stage3b_probe.submit_sge, st.STAGE_INFER: stage4_infer.submit_sge,
        st.STAGE_EXPORT: stage5_export.submit_sge,
    }
    common = dict(paths=paths, container=container, src_dir=src_dir, dry_run=True, emit=emit)
    hold: Any = base_hold
    job_ids: list[str] = []
    for stage in selected:
        o = opts[stage]
        folds = list(o.get("folds") or [])
        if stage == st.STAGE_TRAIN and parallel_folds and len(folds) > 1:
            prep = submitters[stage](hold_jid=hold, **common, **dict(o, plan_only=True))
            ids = [submitters[stage](hold_jid=prep, **common,
                                     **dict(o, folds=[f], skip_planning=True, skip_preprocessing=True))
                   for f in folds]
            submitted = [i for i in ids if i] or ([prep] if prep else [])
        else:
            jid = submitters[stage](hold_jid=hold, **common, **o)
            submitted = [jid] if jid else []
        log.info("emitted %s", stage)
        if submitted:
            job_ids += submitted
            hold = submitted
    return job_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_and_exit(ctx: click.Context, _param, value: str | None) -> None:
    if not value or ctx.resilient_parsing:
        return
    if value == "presets":
        from nvitk.nn.cvit.config import PRESETS

        for name, p in PRESETS.items():
            click.echo(f"{name:6s} embed_dim={p['embed_dim']:5d} depth={p['depth']:3d} heads={p['num_heads']:3d}")
    elif value == "losses":
        from nvitk.segmentation.loss_registry import SEGMENTATION_LOSSES

        for name, spec in SEGMENTATION_LOSSES.items():
            click.echo(f"{name:24s} {spec.description}")
    elif value == "stages":
        for s in st.STAGES_ORDERED:
            click.echo(f"{s:8s} {st.STAGE_LABELS[s]}")
    ctx.exit()


@click.command("nvitk-cvit")
@config_dir_click_option()
@backend_click_option(default="gpu")
@click.option("--list", type=click.Choice(["presets", "losses", "stages"]), callback=_list_and_exit,
              expose_value=False, is_eager=True, help="Print presets / losses / stages and exit.")
@click.option("--stages", type=str, default=st.DEFAULT_STAGES, show_default=True,
              help="Comma list of stage ids or aliases (dataprep, pretrain, train, evaluate, probe, infer, export).")
@root_options
# ---- dataset (stage 0) -----------------------------------------------------------------
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--images-dir", type=click.Path(path_type=Path), default=None)
@click.option("--labels-dir", type=click.Path(path_type=Path), default=None)
@click.option("--channel-names", type=str, default=None, help="e.g. 'CT' or 'T1,T2'.")
@click.option("--labels", "labels_text", type=str, default=None, help="'background=0,tumour=1' or JSON.")
@click.option("--num-folds", type=int, default=cfg.DEFAULT_NUM_FOLDS, show_default=True)
@click.option("--seed", type=int, default=cfg.DEFAULT_FOLD_SEED, show_default=True)
@click.option("--group-regex", type=str, default=None)
@click.option("--allow-empty", is_flag=True, default=False)
@click.option("--workers", type=int, default=4, show_default=True)
# ---- corpus + self-supervised (stages 0/1) -------------------------------------------------
@click.option("--corpus-source", "corpus_sources", multiple=True)
@click.option("--corpus-from-train", is_flag=True, default=False)
@click.option("--corpus-only", is_flag=True, default=False, help="Stage 0 builds only the corpus.")
@click.option("--corpus-id", type=int, default=pth.DEFAULT_CORPUS_ID, show_default=True)
@click.option("--corpus-harmonize", is_flag=True, default=False)
@click.option("--ssl", type=click.Choice(["none", "simmim", "mae"]), default=cfg.DEFAULT_SSL, show_default=True)
@click.option("--ssl-name", type=str, default=None)
@click.option("--ssl-epochs", type=int, default=None)
@click.option("--ssl-batch-size", type=int, default=None)
@click.option("--ssl-patch-size", type=str, default=",".join(map(str, cfg.DEFAULT_SSL_PATCH)), show_default=True)
@click.option("--ssl-configuration", type=click.Choice(["median", "noresample", "onemmiso"]),
              default=cfg.DEFAULT_SSL_CONFIG, show_default=True)
@click.option("--mask-ratio", type=float, default=cfg.DEFAULT_MASK_RATIO, show_default=True)
@click.option("--from-bundle", type=click.Path(path_type=Path), default=None,
              help="Fine-tune from an existing stage-1 bundle.")
@click.option("--llrd", type=float, default=1.0, show_default=True, help="Layer-wise LR decay (pretrained).")
# ---- architecture ------------------------------------------------------------------------
@click.option("--arch", type=str, default=cfg.DEFAULT_ARCH, show_default=True, help="CViTS/B/M/L.")
@click.option("--tokenizer", type=click.Choice(["hierarchical", "intra_patch", "linear"]), default=None)
@click.option("--decoder", type=click.Choice(["unet", "patch"]), default=None)
@click.option("--token-stride", type=int, default=8, show_default=True)
@click.option("--skips", type=str, default=None, help="all | none | bits per level, e.g. 0011.")
@click.option("--skip-drop", type=float, default=None, help="Per-sample skip dropout probability.")
@click.option("--skip-schedule", type=click.Choice(["constant", "warmup"]), default=None)
@click.option("--skip-warmup-epochs", type=int, default=None)
@click.option("--skip-gate", type=click.Choice(["none", "learned"]), default=None)
@click.option("--skip-gate-l1", type=float, default=None)
@click.option("--num-registers", type=int, default=None)
@click.option("--drop-path", type=float, default=None)
@click.option("--arch-json", type=str, default=None, help="Any CViTConfig fields as JSON.")
# ---- training (stage 2) --------------------------------------------------------------------
@click.option("--configuration", type=click.Choice(["3d_fullres", "2d", "3d_lowres"]),
              default=cfg.DEFAULT_CONFIGURATION, show_default=True)
@click.option("--baseline-planner", type=str, default=cfg.DEFAULT_BASELINE_PLANNER, show_default=True)
@click.option("--patch-size", type=str, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--loss", type=str, default=cfg.DEFAULT_LOSS, show_default=True)
@click.option("--loss-config", type=str, default=None)
@click.option("--folds", type=str, default="0", show_default=True)
@click.option("--epochs", type=int, default=None)
@click.option("--iterations-per-epoch", type=int, default=None)
@click.option("--lr", type=float, default=cfg.DEFAULT_LR, show_default=True)
@click.option("--warmup-epochs", type=int, default=cfg.DEFAULT_WARMUP_EPOCHS, show_default=True)
@click.option("--no-mirror", is_flag=True, default=False)
@click.option("--tag", type=str, default=None)
@click.option("--run-name", type=str, default=None, help="Stage-2 run for stages 3-5 (default: latest).")
@click.option("--continue-training", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
# ---- probe (stage 3b) ----------------------------------------------------------------------
@click.option("--probe-every", type=int, default=cfg.DEFAULT_PROBE_EVERY, show_default=True)
@click.option("--probe-interventions", type=str, default=",".join(cfg.DEFAULT_PROBE_INTERVENTIONS), show_default=True)
@click.option("--local-radius-mm", type=str, default=",".join(f"{r:g}" for r in cfg.DEFAULT_LOCAL_RADII_MM),
              show_default=True)
@click.option("--layer-sweep", is_flag=True, default=False)
@click.option("--probe-folds", type=str, default=None)
@click.option("--probe-max-cases", type=int, default=0, show_default=True)
# ---- inference / export (stages 4, 5) --------------------------------------------------------
@click.option("--infer-input", "infer_inputs", multiple=True, type=click.Path(path_type=Path))
@click.option("--infer-output", type=click.Path(path_type=Path), default=None)
@click.option("--disable-tta", is_flag=True, default=False)
@click.option("--largest-component", is_flag=True, default=False)
@click.option("--export-name", type=str, default=None)
@click.option("--zip", is_flag=True, default=False)
# ---- execution -----------------------------------------------------------------------------
@click.option("--device", type=str, default=None, help="Torch device (default derived from --backend).")
@click.option("--num-processes", type=int, default=8, show_default=True)
@click.option("--submit", type=click.Choice(["local", "sge"]), default="local", show_default=True)
@click.option("--parallel-folds", is_flag=True, default=False)
@click.option("--container", type=click.Path(path_type=Path), default=None)
@click.option("--src-dir", type=click.Path(path_type=Path), default=None)
@click.option("--hold-jid", type=str, default=None)
@click.option("--emit-script", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--no-remote", is_flag=True, default=False)
def main(**o: Any) -> None:
    """CViT: Convolutional Vision Transformer segmentation — pre-train, train, evaluate, probe, infer, export."""
    from nvitk.pipes.cvit.stage0_dataprep import parse_labels

    selected = st.parse_stages(o.pop("stages"))
    roots = pop_roots(o)
    o["labels"] = parse_labels(o.pop("labels_text"))
    if st.STAGE_PRETRAIN in selected and o["ssl"] == "none":
        raise click.ClickException("Stage pretrain needs --ssl simmim|mae.")
    if st.STAGE_INFER in selected and not o.get("infer_inputs"):
        raise click.ClickException("Stage infer needs --infer-input.")
    log.info("nvitk-cvit | stages %s | submit=%s", ",".join(selected), o["submit"])

    if o["submit"] == "local":
        paths = paths_from_options(roots)
        run_local(paths, selected, stage_options(o, paths), o["backend"])
        return

    paths = pth.layout_cluster(**roots)
    container = o.get("container") or cfg.CONTAINER_PATH
    if container is None:
        raise click.ClickException("No container: pass --container or set pipelines.cvit.default_sge_container_root.")
    opts = stage_options(o, paths)
    try:
        run_driver_script(
            lambda fh: run_sge(paths, selected, opts, container=Path(container), src_dir=o.get("src_dir"),
                               base_hold=o.get("hold_jid"), parallel_folds=o["parallel_folds"], emit=fh),
            title=f"cvit dataset={o['dataset_id']} stages={','.join(selected)}",
            basename=f"submit_cvit_{o['dataset_id']}", emit_script=o.get("emit_script"),
            dry_run=o["dry_run"], no_remote=o["no_remote"],
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
