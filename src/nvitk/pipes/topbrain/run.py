"""ToPBrain pipeline master: ``nvitk-topbrain``.

Entry point for the whole ToPBrain / ToPAneu TA36 pipeline. Dispatches each stage either
locally (in-process) or on SGE (one job per stage, chained with ``-hold_jid``).

Stages
------
``stage0``  data preparation — challenge release (+ your own annotated cohorts) into an nnU-Net
            dataset; optionally an unlabeled pre-training corpus
``stage1``  pre-training — a published OpenMind checkpoint, or nnssl from scratch, producing a
            bundle (checkpoint + adaptation plan + a materialised segmentation model)
``stage2``  transfer training — fine-tune the bundle on the stage-0 dataset with the in-tree
            nnU-Net build
``stage3``  evaluation — the six challenge metrics
``stage4``  inference — predict on new cases
``stage5``  packaging — Grand Challenge submission container

Data layout
-----------
All roots come from ``sge.json`` ``pipelines.topbrain_paths``; see
:mod:`nvitk.pipes.topbrain.util.paths`. Under ``--submit local`` a CLI flag overrides the
configured root; under ``--submit sge`` the ``cluster_*`` config wins, because a host path is
meaningless on the cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import click

from nvitk.cluster.sge import write_script_header
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger, PipelineRunTracker
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import stages as st
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import paths as pth
from nvitk.pipes.topbrain.util.sge_backend import torch_device_for_backend

log = Logger()

#: Stage id → the module implementing it.
STAGE_MODULES: dict[str, str] = {
    st.STAGE_DATAPREP: "nvitk.pipes.topbrain.stage0_dataprep",
    st.STAGE_PRETRAIN: "nvitk.pipes.topbrain.stage1_pretrain",
    st.STAGE_TRAIN: "nvitk.pipes.topbrain.stage2_train",
    st.STAGE_EVALUATE: "nvitk.pipes.topbrain.stage3_evaluate",
    st.STAGE_INFER: "nvitk.pipes.topbrain.stage4_infer",
    st.STAGE_PACKAGE: "nvitk.pipes.topbrain.stage5_package",
}


def _resolve_layout(submit: str, **overrides: Path | None) -> tuple[Any, Any, Any]:
    """Resolve the local and cluster layouts and pick the active one.

    Both are resolved so the log can show what the other context would have used — the
    commonest cluster failure is a path that only exists on the workstation.
    """
    local = pth.layout_local(**overrides)
    try:
        cluster = pth.layout_cluster(**overrides)
    except Exception as exc:  # cluster roots are optional on a workstation-only install
        if submit == "sge":
            raise
        log.debug("Cluster layout unavailable (%s); local-only run.", exc)
        cluster = local
    return local, cluster, (cluster if submit == "sge" else local)


def _stage_options(**o: Any) -> dict[str, dict[str, Any]]:
    """Per-stage keyword arguments, derived from the master CLI's flags."""
    bundle_name = o["bundle_name"] or (
        o["checkpoint_name"] if o["pretrain_source"] == "openmind" else o["ssl_trainer"]
    )
    return {
        st.STAGE_DATAPREP: dict(
            target=o["dataprep_target"], label_set=o["label_set"], modality=o["modality"],
            extra_train=list(o["extra_train"]), include_challenge=o["include_challenge"],
            corpus_sources=list(o["corpus_sources"]), num_folds=o["num_folds"], seed=o["seed"],
            overwrite=o["overwrite"], workers=o["workers"], backend=o["backend"],
        ),
        st.STAGE_PRETRAIN: dict(
            source=o["pretrain_source"], name=bundle_name,
            checkpoint_name=o["checkpoint_name"], trainer_name=o["ssl_trainer"],
            configuration_name=o["ssl_config"], ssl_loss=o["ssl_loss"],
            ssl_loss_config=o["ssl_loss_config"], patch_size=o["ssl_patch_size"],
            batch_size=o["ssl_batch_size"], num_epochs=o["ssl_epochs"],
            initial_lr=o["ssl_lr"], init_checkpoint_name=o["init_checkpoint_name"],
            device=o["device"], num_processes=o["num_processes"],
            export_model=o["export_model"], overwrite=o["overwrite"], backend=o["backend"],
        ),
        st.STAGE_TRAIN: dict(
            bundle=bundle_name, label_set=o["label_set"], pretrain_name=bundle_name,
            loss=o["loss"], loss_config=o["loss_config"], folds=list(o["folds"]),
            adaptation_mode=o["adaptation_mode"], baseline_planner=o["baseline_planner"],
            target_spacing=o["target_spacing"],
            patch_size=o["patch_size"], batch_size=o["batch_size"], num_epochs=o["num_epochs"], device=o["device"], num_gpus=o["num_gpus"],
            num_processes=o["num_processes"], from_scratch=o["from_scratch"],
            backend=o["backend"],
        ),
        st.STAGE_EVALUATE: dict(
            label_set=o["label_set"],
            # Left unset on purpose: the run directory name contains the plans identifier,
            # whose spacing stage 2 only decides at run time. Resolved from stage 2's
            # provenance when the stage actually runs.
            prediction_subdir=None,
            folds_spec=",".join(str(f) for f in o["folds"]),
            iou_threshold=None, skip_neighbours=False, backend=o["backend"],
        ),
        st.STAGE_INFER: dict(
            label_set=o["label_set"], loss=o["loss"], folds=list(o["folds"]),
            # Left unset on purpose: the real plans name embeds a spacing that stage 2 only
            # decides at run time, so it is read from stage 2's provenance instead.
            plans_identifier=None, configuration_name=None,
            checkpoint_name="checkpoint_final.pth", min_volume_mm3=o["min_volume_mm3"],
            largest_only=o["largest_only"], device=o["device"], num_processes=3,
            workers=o["workers"], skip_prediction=False, backend=o["backend"],
        ),
        st.STAGE_PACKAGE: dict(
            label_set=o["label_set"], loss=o["loss"],
            # As above: resolved from stage 2's provenance, not assembled from the bundle name.
            plans_identifier=None, configuration_name=None, folds=list(o["folds"]),
            checkpoint="checkpoint_final.pth", name="topbrain-ta36", tag="latest",
            build=False, save=False, backend=o["backend"],
        ),
    }


def _local_runners(active: Any, options: dict[str, dict[str, Any]]) -> dict[str, Callable[[], Any]]:
    """Zero-argument callables running each stage in-process against *active*."""
    from nvitk.pipes.topbrain import (
        stage0_dataprep, stage1_pretrain, stage2_train,
        stage3_evaluate, stage4_infer, stage5_package,
    )

    def _strip(stage: str, *drop: str) -> dict[str, Any]:
        """Stage options minus keys the local entry point does not take."""
        opts = dict(options[stage])
        for key in ("backend", *drop):
            opts.pop(key, None)
        return opts

    def _dataprep() -> Any:
        """Stage 0 locally."""
        return stage0_dataprep.run_dataprep(paths=active, **_strip(st.STAGE_DATAPREP))

    def _pretrain() -> Any:
        """Stage 1 locally."""
        opts = _strip(st.STAGE_PRETRAIN)
        opts["ssl_loss_config"] = loss_util.parse_loss_config(opts.get("ssl_loss_config"))
        return stage1_pretrain.run_pretrain(paths=active, **opts)

    def _train() -> Any:
        """Stage 2 locally, resolving the bundle name to its directory."""
        opts = _strip(st.STAGE_TRAIN)
        opts["bundle"] = stage1_pretrain.bundle_dir(active.results_root, str(opts["bundle"]))
        opts["loss_config"] = loss_util.parse_loss_config(opts.get("loss_config"))
        return stage2_train.run_train(paths=active, **opts)

    def _evaluate() -> Any:
        """Stage 3 locally, scoring each fold's held-out validation output.

        Deliberately **not** stage 4's directory: stage 4 predicts over whatever folder it is
        pointed at, which by default is the whole training set. Scoring that would grade the
        model partly on its own training data.
        """
        opts = _strip(st.STAGE_EVALUATE)
        run_name = opts.pop("prediction_subdir", None) or stage2_train.trained_run_name(
            active.results_root, opts["label_set"]
        )
        folds_spec = opts.pop("folds_spec", "0,1,2,3,4")
        dataset = stage2_train.dataset_name_for(opts["label_set"])
        predictions, partial = stage3_evaluate.collect_cv_predictions(
            active.nnunet_results, dataset, run_name,
            active.results_root / pth.STAGE3_EVAL_DIR / run_name / "cv_predictions",
            folds=stage2_train.parse_folds(folds_spec),
        )
        return stage3_evaluate.run_evaluate(
            prediction_dir=predictions, partial_folds=partial,
            reference_dir=active.nnunet_raw / dataset / "labelsTr",
            results_root=active.results_root, run_name=run_name, **opts
        )

    def _infer() -> Any:
        """Stage 4 locally, predicting on the prepared training images."""
        opts = _strip(st.STAGE_INFER)
        dataset = stage2_train.dataset_name_for(opts["label_set"])
        return stage4_infer.run_infer(
            input_dir=active.nnunet_raw / dataset / "imagesTr",
            nnunet_raw=active.nnunet_raw, nnunet_preprocessed=active.nnunet_preprocessed,
            nnunet_results=active.nnunet_results, results_root=active.results_root, **opts
        )

    def _package() -> Any:
        """Stage 5 locally."""
        return stage5_package.run_package(
            nnunet_results=active.nnunet_results, results_root=active.results_root,
            **_strip(st.STAGE_PACKAGE)
        )

    return {
        st.STAGE_DATAPREP: _dataprep, st.STAGE_PRETRAIN: _pretrain, st.STAGE_TRAIN: _train,
        st.STAGE_EVALUATE: _evaluate, st.STAGE_INFER: _infer, st.STAGE_PACKAGE: _package,
    }


def _run_local(active: Any, selected: Sequence[str], options: dict[str, dict[str, Any]]) -> None:
    """Run the selected stages in-process, tracked by :class:`PipelineRunTracker`."""
    runners = _local_runners(active, options)
    with PipelineRunTracker(
        log, "topbrain", ["(cohort)"], list(selected), stage_labels=st.STAGE_LABELS
    ) as run:
        for stage in selected:
            run.run_stage("(cohort)", stage, runners[stage], reraise=True)


def _run_sge(
    active: Any, selected: Sequence[str], options: dict[str, dict[str, Any]], *,
    container: Path, src_dir: Path | None, base_hold: str | None,
    dry_run: bool, emit: TextIO | None,
) -> list[str]:
    """Submit (or emit) one SGE job per stage, each holding on the previous one."""
    from nvitk.pipes.topbrain import (
        stage0_dataprep, stage1_pretrain, stage2_train,
        stage3_evaluate, stage4_infer, stage5_package,
    )

    submitters = {
        st.STAGE_DATAPREP: stage0_dataprep.submit_sge,
        st.STAGE_PRETRAIN: stage1_pretrain.submit_sge,
        st.STAGE_TRAIN: stage2_train.submit_sge,
        st.STAGE_EVALUATE: stage3_evaluate.submit_sge,
        st.STAGE_INFER: stage4_infer.submit_sge,
        st.STAGE_PACKAGE: stage5_package.submit_sge,
    }
    job_ids: list[str] = []
    hold: str | None = base_hold
    for stage in selected:
        job_id = submitters[stage](
            paths=active, container=container, src_dir=src_dir,
            hold_jid=hold, dry_run=dry_run, emit=emit, **options[stage]
        )
        if job_id:
            job_ids.append(job_id)
            hold = job_id  # chain: each stage waits for its predecessor
        log.info("submitted %s -> job %s", stage, job_id or "(emitted)")
    return job_ids


@click.command("nvitk-topbrain")
@config_dir_click_option()
@backend_click_option(default="gpu")
# ---- selection -------------------------------------------------------------
@click.option("--stages", type=str, default=st.DEFAULT_STAGES, show_default=True,
              help="Comma-separated stage ids or aliases.")
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr"]), default=None)
@click.option("--list-losses", is_flag=True, default=False, help="Print the losses and exit.")
@click.option("--list-checkpoints", is_flag=True, default=False,
              help="Print the published checkpoints and exit.")
@click.option("--list-trainers", is_flag=True, default=False,
              help="Print the self-supervised trainers and exit.")
# ---- stage 0: data preparation ---------------------------------------------
@click.option("--dataprep-target", type=click.Choice(["train", "corpus", "both"]),
              default="train", show_default=True)
@click.option("--modality", type=click.Choice(["both", "ct", "mr"]), default="both",
              show_default=True)
@click.option("--extra-train", multiple=True,
              help="Extra annotated cohort: 'name=/images:/labels:modality'. Repeatable.")
@click.option("--no-challenge", is_flag=True, default=False,
              help="Exclude the challenge cases from the training set.")
@click.option("--corpus-source", "corpus_sources", multiple=True,
              help="Unlabeled corpus source: 'name:modality=/path[:glob]'. Repeatable.")
@click.option("--num-folds", type=int, default=None)
@click.option("--seed", type=int, default=None)
# ---- stage 1: pre-training --------------------------------------------------
@click.option("--pretrain-source", type=click.Choice(["openmind", "scratch"]),
              default="openmind", show_default=True)
@click.option("--checkpoint-name", type=str, default=None,
              help="Published checkpoint (see --list-checkpoints).")
@click.option("--bundle-name", type=str, default=None,
              help="Bundle directory name (default: the checkpoint or trainer name).")
@click.option("--ssl-trainer", type=str, default=None)
@click.option("--ssl-config", type=click.Choice(["median", "noresample", "onemmiso"]),
              default=None)
@click.option("--ssl-loss", type=str, default=None)
@click.option("--ssl-loss-config", type=str, default=None)
@click.option("--ssl-patch-size", type=int, nargs=3, default=None)
@click.option("--ssl-batch-size", type=int, default=None)
@click.option("--ssl-epochs", type=int, default=None)
@click.option("--ssl-lr", type=float, default=None)
@click.option("--init-checkpoint-name", type=str, default=None,
              help="Seed --pretrain-source scratch from a published checkpoint.")
@click.option("--no-export-model", is_flag=True, default=False,
              help="Skip materialising the bundle's segmentation_model.pth.")
# ---- stage 2: transfer training ---------------------------------------------
@click.option("--loss", type=str, default=None,
              help="Segmentation loss: a registry name or 'package.module:Callable'.")
@click.option("--loss-config", type=str, default=None)
@click.option("--folds", type=str, default="0", show_default=True)
@click.option("--adaptation-mode",
              type=click.Choice(["default_nnunet", "like_pretrained", "no_resample", "fixed"]),
              default="default_nnunet", show_default=True)
@click.option("--baseline-planner", type=str, default="nnUNetPlannerResEncL", show_default=True,
              help="Planner for the baseline plan stage 2 adapts; sets the target spacing.")
@click.option("--target-spacing", type=float, nargs=3, default=None)
@click.option("--patch-size", type=int, nargs=3, default=None,
              help="Override the plan's 160^3 training patch (multiples of 32).")
@click.option("--batch-size", type=int, default=None)
@click.option("--num-epochs", type=int, default=None)
@click.option("--num-gpus", type=int, default=1, show_default=True)
@click.option("--from-scratch", is_flag=True, default=False,
              help="Same architecture, random initialisation — the control run.")
# ---- stages 4-5 --------------------------------------------------------------
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True)
@click.option("--largest-only", is_flag=True, default=False)
# ---- shared ------------------------------------------------------------------
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-processes", type=int, default=8, show_default=True)
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--overwrite", is_flag=True, default=True)
@click.option("--challenge-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
# ---- submission ---------------------------------------------------------------
@click.option("--submit", type=click.Choice(["local", "sge"], case_sensitive=False),
              default="local", show_default=True)
@click.option("--container", type=click.Path(path_type=Path), default=None)
@click.option("--src-dir", type=click.Path(path_type=Path), default=None)
@click.option("--base-hold", type=str, default=None)
@click.option("--emit-script", "emit_script", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--log-level", type=str, default="INFO", show_default=True)
def main(**kw: Any) -> None:
    """ToPBrain / ToPAneu TA36 vessel segmentation pipeline (local or SGE dispatch)."""
    from nvitk.pipes.topbrain.stage2_train import parse_folds

    Logger(level=kw["log_level"].upper())
    log.set_level(kw["log_level"].upper())

    if kw["list_trainers"]:
        from nvitk.pipes.topbrain.util.trainers import describe_ssl_trainers

        click.echo(describe_ssl_trainers())
        return
    if kw["list_checkpoints"]:
        from nvitk.pipes.topbrain.util import checkpoints as ckpt_util

        click.echo(ckpt_util.describe_openmind_models())
        return
    if kw["list_losses"]:
        click.echo("Segmentation losses (--loss):")
        for name, spec in sorted(loss_util.SEGMENTATION_LOSSES.items()):
            click.echo(f"  {name:24s} {spec.description}")
        click.echo("\nSelf-supervised losses (--ssl-loss):")
        for name, spec in sorted(loss_util.SSL_LOSSES.items()):
            click.echo(f"  {name:24s} {spec.description}")
        click.echo("\nOr any 'package.module:Callable' taking (net_output, target).")
        return

    label_set = kw["label_set"] or cfg.DEFAULT_LABEL_SET
    loss = kw["loss"] or cfg.DEFAULT_LOSS
    backend = kw.get("backend", "gpu")
    device = kw["device"] or torch_device_for_backend(backend)
    submit = kw["submit"].lower()

    loss_util.validate_loss_name(loss, registry=loss_util.SEGMENTATION_LOSSES)
    if kw["ssl_loss"]:
        loss_util.validate_loss_name(kw["ssl_loss"], registry=loss_util.SSL_LOSSES)

    selected = st.parse_stages(kw["stages"])

    if kw["pretrain_source"] == "openmind" and not kw["checkpoint_name"] \
            and st.STAGE_PRETRAIN in selected:
        raise click.UsageError(
            "--pretrain-source openmind needs --checkpoint-name (see --list-checkpoints)."
        )

    local, cluster, active = _resolve_layout(
        submit, challenge_root=kw["challenge_root"], results_root=kw["results_root"]
    )

    options = _stage_options(
        label_set=label_set, loss=loss, backend=backend, device=device,
        folds=parse_folds(kw["folds"]),
        include_challenge=not kw["no_challenge"], export_model=not kw["no_export_model"],
        ssl_trainer=kw["ssl_trainer"] or cfg.DEFAULT_SSL_TRAINER,
        ssl_config=kw["ssl_config"] or cfg.DEFAULT_SSL_CONFIG,
        **{k: kw[k] for k in (
            "dataprep_target", "modality", "extra_train", "corpus_sources", "num_folds", "seed",
            "pretrain_source", "checkpoint_name", "bundle_name", "ssl_loss", "ssl_loss_config",
            "ssl_patch_size", "ssl_batch_size", "ssl_epochs", "ssl_lr", "init_checkpoint_name",
            "loss_config", "adaptation_mode", "baseline_planner", "target_spacing",
            "patch_size", "batch_size",
            "num_epochs", "num_gpus",
            "from_scratch", "min_volume_mm3", "largest_only", "num_processes", "workers",
            "overwrite",
        )},
    )

    log.info("topbrain | submit=%s backend=%s device=%s label_set=%s loss=%s",
             submit, backend, device, label_set, loss)
    log.info("  stages : %s", ", ".join(f"{s} ({st.STAGE_LABELS[s]})" for s in selected))
    log.info("  local  challenge=%s results=%s", local.challenge_root, local.results_root)
    log.info("  active nnunet_raw=%s nnssl_raw=%s", active.nnunet_raw, active.nnssl_raw)

    if submit == "local":
        _run_local(active, selected, options)
        return

    container = kw["container"] or cfg.CONTAINER_PATH
    if container is None:
        raise click.UsageError(
            "--submit sge needs a container: pass --container, or set "
            "pipelines.topbrain.default_sge_container_root in sge.json."
        )
    if kw["emit_script"] is not None:
        emit_script = Path(kw["emit_script"])
        emit_script.parent.mkdir(parents=True, exist_ok=True)
        with open(emit_script, "w", encoding="utf-8") as handle:
            write_script_header(
                handle, log_dir=cfg.SGE_LOG_DIR, err_dir=cfg.SGE_ERR_DIR,
                title=f"topbrain label_set={label_set} loss={loss}",
            )
            _run_sge(active, selected, options, container=container, src_dir=kw["src_dir"],
                     base_hold=kw["base_hold"], dry_run=True, emit=handle)
        log.ok(f"wrote submission script: {emit_script}")
        return
    _run_sge(active, selected, options, container=container, src_dir=kw["src_dir"],
             base_hold=kw["base_hold"], dry_run=kw["dry_run"], emit=None)


__all__ = ["STAGE_MODULES", "main"]


if __name__ == "__main__":
    main()
