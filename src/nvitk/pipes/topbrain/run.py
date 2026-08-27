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
``stage6``  self-training *(optional)* — pseudo-label the unlabeled cohort with the stage-2
            model, keep what survives scrutiny, and print the command that feeds it back

Submitting to SGE
-----------------
A workstation has no ``qsub``. ``--submit sge`` therefore emits the whole run as one bash
driver script — every stage's ``singularity exec`` payload and ``qsub`` invocation, with the
``-hold_jid`` chain resolved through shell variables — and then executes that script where
``qsub`` exists: locally if this *is* a submit host, otherwise over SSH to a login node
(SFTP the script, ``bash`` it, parse the job ids back). ``--remote-host`` / ``--remote-user``
skip the prompts; the password is always prompted for. ``--no-remote`` stops after writing.

Data layout
-----------
All roots come from ``sge.json`` ``pipelines.topbrain_paths``; see
:mod:`nvitk.pipes.topbrain.util.paths`. Under ``--submit local`` a CLI flag overrides the
configured root; under ``--submit sge`` the ``cluster_*`` config wins, because a host path is
meaningless on the cluster.

Live monitoring
---------------
``--tensorboard`` mirrors the stage 1 and stage 2 per-epoch training logs into TensorBoard
events under ``<results_root>/tensorboard/``, while they train. ``--tensorboard-serve`` decides
where the server runs:

``auto`` *(default)*
    ``local`` for ``--submit local``; **nothing** for ``--submit sge`` — the cluster roots are
    mounted on the workstation, so serving them from here costs no cluster slot and needs no
    tunnel. The exact ``nvitk-tensorboard`` command to do that is logged at submission.
``local``
    Serve on this workstation. Under ``--submit sge`` this blocks after submitting, mirroring
    and serving the mounted cluster tree until interrupted; the submitted jobs are unaffected.
``cluster``
    Also submit a small CPU-only job that runs the server on a compute node with ``--bind_all``
    and records the node it landed on, for an ``ssh -L`` tunnel.
``none``
    Write events only.

See :mod:`nvitk.pipes.topbrain.util.tensorboard`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

import click

from nvitk.cluster.remote_submit import prompt_ssh_credentials, run_sge_script_ssh_capture
from nvitk.cluster.sge import write_script_header
from nvitk.cluster.sge_chunk import parse_sge_submission_job_ids
from nvitk.cluster.sge_remote import publish_sge_driver_script, resolve_sge_script_paths
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger, PipelineRunTracker
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import stages as st
from nvitk.pipes.topbrain.util import losses as loss_util
from nvitk.pipes.topbrain.util import paths as pth
from nvitk.pipes.topbrain.util import sampling as sampling_util
from nvitk.pipes.topbrain.util import tensorboard as tb
from nvitk.pipes.topbrain.util.sge_backend import (
    set_sge_project_override,
    torch_device_for_backend,
)

log = Logger()

#: Stage id → the module implementing it.
STAGE_MODULES: dict[str, str] = {
    st.STAGE_DATAPREP: "nvitk.pipes.topbrain.stage0_dataprep",
    st.STAGE_PRETRAIN: "nvitk.pipes.topbrain.stage1_pretrain",
    st.STAGE_TRAIN: "nvitk.pipes.topbrain.stage2_train",
    st.STAGE_EVALUATE: "nvitk.pipes.topbrain.stage3_evaluate",
    st.STAGE_INFER: "nvitk.pipes.topbrain.stage4_infer",
    st.STAGE_PACKAGE: "nvitk.pipes.topbrain.stage5_package",
    st.STAGE_SELFTRAIN: "nvitk.pipes.topbrain.stage6_selftrain",
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
            extra_train=list(o["extra_train"]),
            extra_train_only=list(o["extra_train_only"]),
            include_challenge=o["include_challenge"],
            corpus_sources=list(o["corpus_sources"]), num_folds=o["num_folds"], seed=o["seed"],
            overwrite=o["overwrite"], workers=o["workers"], backend=o["backend"],
            ct_context_window=o["ct_context_window"],
            mr_context_percentiles=o["mr_context_percentiles"],
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
            tensorboard=o["tensorboard"], tensorboard_interval=o["tensorboard_interval"],
        ),
        st.STAGE_TRAIN: dict(
            bundle=bundle_name, label_set=o["label_set"], pretrain_name=bundle_name,
            loss=o["loss"], loss_config=o["loss_config"], folds=list(o["folds"]),
            adaptation_mode=o["adaptation_mode"], baseline_planner=o["baseline_planner"],
            target_spacing=o["target_spacing"],
            patch_size=o["patch_size"], batch_size=o["batch_size"], num_epochs=o["num_epochs"], device=o["device"], num_gpus=o["num_gpus"],
            num_processes=o["num_processes"], from_scratch=o["from_scratch"],
            backend=o["backend"],
            tensorboard=o["tensorboard"], tensorboard_interval=o["tensorboard_interval"],
            sampling=o["sampling"], sampling_temperature=o["sampling_temperature"],
            sampling_oversample=o["sampling_oversample"],
        ),
        st.STAGE_EVALUATE: dict(
            label_set=o["label_set"],
            # Left unset on purpose: the run directory name contains the plans identifier,
            # whose spacing stage 2 only decides at run time. Resolved from stage 2's
            # provenance when the stage actually runs.
            prediction_subdir=None,
            folds_spec=",".join(str(f) for f in o["folds"]),
            iou_threshold=None, skip_neighbours=False, baseline=o["compare_to"],
            backend=o["backend"],
        ),
        st.STAGE_INFER: dict(
            label_set=o["label_set"], loss=o["loss"], folds=list(o["folds"]),
            # Left unset on purpose: the real plans name embeds a spacing that stage 2 only
            # decides at run time, so it is read from stage 2's provenance instead.
            plans_identifier=None, configuration_name=None,
            checkpoint_name="checkpoint_final.pth", min_volume_mm3=o["min_volume_mm3"],
            largest_only=o["largest_only"],
            repair_gaps_mm=o["repair_gaps_mm"], repair_adjacency=o["repair_adjacency"],
            repair_lateral=o["repair_lateral"], repair_close_radius=o["repair_close_radius"],
            device=o["device"], num_processes=3,
            workers=o["workers"], skip_prediction=False, backend=o["backend"],
        ),
        st.STAGE_PACKAGE: dict(
            label_set=o["label_set"], loss=o["loss"],
            # As above: resolved from stage 2's provenance, not assembled from the bundle name.
            plans_identifier=None, configuration_name=None, folds=list(o["folds"]),
            checkpoint="checkpoint_final.pth", name="topbrain-ta36", tag="latest",
            build=False, save=False, backend=o["backend"],
        ),
        st.STAGE_SELFTRAIN: dict(
            label_set=o["label_set"], loss=o["loss"], folds=list(o["folds"]),
            # As for stages 4-5: the plans name is only knowable after stage 2 has run.
            plans_identifier=None, configuration_name=None,
            checkpoint_name="checkpoint_final.pth", modality=o["pseudo_modality"],
            max_components=o["pseudo_max_components"],
            max_fragmented_classes=o["pseudo_max_fragmented"],
            max_invalid_neighbours=o["pseudo_max_invalid"],
            volume_range=o["pseudo_volume_range"],
            agreement_threshold=o["pseudo_agreement"],
            repair_gaps_mm=o["repair_gaps_mm"],
            min_volume_mm3=o["min_volume_mm3"], max_accepted=o["pseudo_max_accepted"],
            device=o["device"], num_processes=3, backend=o["backend"],
        ),
    }


def _local_runners(active: Any, options: dict[str, dict[str, Any]]) -> dict[str, Callable[[], Any]]:
    """Zero-argument callables running each stage in-process against *active*."""
    from nvitk.pipes.topbrain import (
        stage0_dataprep, stage1_pretrain, stage2_train,
        stage3_evaluate, stage4_infer, stage5_package, stage6_selftrain,
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

    def _selftrain() -> Any:
        """Stage 6 locally, pseudo-labelling the unlabeled corpus."""
        return stage6_selftrain.run_selftrain(
            input_dir=active.corpus_root,
            nnunet_raw=active.nnunet_raw, nnunet_preprocessed=active.nnunet_preprocessed,
            nnunet_results=active.nnunet_results, results_root=active.results_root,
            **_strip(st.STAGE_SELFTRAIN)
        )

    return {
        st.STAGE_DATAPREP: _dataprep, st.STAGE_PRETRAIN: _pretrain, st.STAGE_TRAIN: _train,
        st.STAGE_EVALUATE: _evaluate, st.STAGE_INFER: _infer, st.STAGE_PACKAGE: _package,
        st.STAGE_SELFTRAIN: _selftrain,
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
        stage3_evaluate, stage4_infer, stage5_package, stage6_selftrain,
    )

    submitters = {
        st.STAGE_DATAPREP: stage0_dataprep.submit_sge,
        st.STAGE_PRETRAIN: stage1_pretrain.submit_sge,
        st.STAGE_TRAIN: stage2_train.submit_sge,
        st.STAGE_EVALUATE: stage3_evaluate.submit_sge,
        st.STAGE_INFER: stage4_infer.submit_sge,
        st.STAGE_PACKAGE: stage5_package.submit_sge,
        st.STAGE_SELFTRAIN: stage6_selftrain.submit_sge,
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


def _tensorboard_for_sge(
    *,
    local: Any, cluster: Any, container: Path, src_dir: Path | None,
    serve_mode: str, enabled: bool, selected: Sequence[str], label_set: str,
    port: int, interval: float, dry_run: bool, emit: TextIO | None,
) -> None:
    """Arrange TensorBoard around an SGE submission, per the resolved serve mode.

    *cluster* is what the jobs see. Anything that runs *here* must read the same tree as this
    workstation sees it — usually the cluster mount at its own absolute path, occasionally the
    local roots; :func:`~nvitk.pipes.topbrain.util.tensorboard.workstation_view` decides which.
    Getting this backwards yields a TensorBoard that loads cleanly and shows nothing.
    """
    here, _mounted = tb.workstation_view(cluster, local)
    sources = tb.stage_sources(here, selected)
    event_root = tb.tensorboard_root(here.results_root)

    if serve_mode == "cluster":
        tb.submit_server_sge(
            paths=cluster, container=container, src_dir=src_dir, port=port,
            interval=interval, label_set=label_set,
            local_results_root=local.results_root, dry_run=dry_run, emit=emit,
        )
        return
    if serve_mode == "local":
        # The server itself starts after the jobs are submitted (it blocks), so all this does
        # is record the command that will be used.
        log.info("Will serve TensorBoard locally once submitted: %s",
                 tb.local_serve_command(event_root, sources=sources, port=port))
        return
    if enabled:
        # 'none' is the default for SGE: the cluster tree is mounted here, so serving it
        # locally costs no slot and needs no tunnel. Hand over the exact command.
        log.info("TensorBoard events -> %s", event_root)
        log.info("  watch them with: %s",
                 tb.local_serve_command(event_root, sources=sources, port=port))


def _submit_via_login_node(
    *,
    active: Any, local: Any, selected: Sequence[str], options: dict[str, dict[str, Any]],
    container: Path, src_dir: Path | None, base_hold: str | None, label_set: str, loss: str,
    emit_script: Path | None, dry_run: bool, no_remote: bool,
    remote_host: str | None, remote_user: str | None,
    tensorboard: dict[str, Any],
) -> list[str]:
    """Emit the whole submission as one bash driver, then run it where ``qsub`` exists.

    A workstation has no ``qsub``, so submitting from here cannot work directly — which is what
    the ``FileNotFoundError: 'qsub'`` was. Every other cluster pipeline in this repo solves it
    the same way and so does this one now: build a single driver script containing every
    stage's ``singularity exec`` payload and ``qsub`` invocation, then execute that script on a
    login node. The script captures each job id in a shell variable, so the ``-hold_jid`` chain
    is resolved on the cluster rather than here.

    Where it runs
    -------------
    ``qsub`` on this host
        Run it locally — you are already on a submit host, and prompting for a password would
        be pointless.
    otherwise
        SFTP it to ``SGE_SCRIPTS_DIR`` and ``bash`` it over SSH.
    ``--no-remote``
        Write it and stop, printing the command to run by hand.

    Returns
    -------
    list of str
        Submitted SGE job ids, parsed from the driver's output. Empty when nothing was run.
    """
    import shutil
    import subprocess
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_script, remote_script = resolve_sge_script_paths(
        Path(emit_script) if emit_script is not None else None,
        remote_scripts_dir=cfg.SGE_SCRIPTS_DIR,
        default_basename=f"submit_topbrain_{label_set}_{stamp}.sh",
    )

    with open(local_script, "w", encoding="utf-8") as handle:
        write_script_header(
            handle, log_dir=cfg.SGE_LOG_DIR, err_dir=cfg.SGE_ERR_DIR,
            title=f"topbrain label_set={label_set} loss={loss} stages={','.join(selected)}",
        )
        _run_sge(active, selected, options, container=container, src_dir=src_dir,
                 base_hold=base_hold, dry_run=True, emit=handle)
        _tensorboard_for_sge(
            local=local, cluster=active, container=container, src_dir=src_dir,
            dry_run=True, emit=handle, **tensorboard,
        )
    log.info("  local script : %s", local_script)
    log.info("  cluster path : %s", remote_script)

    if dry_run:
        log.ok(f"--dry-run: submission script written, nothing submitted -> {local_script}")
        return []
    if no_remote:
        log.ok(f"--no-remote: run it yourself on the login node:\n    bash {remote_script}")
        return []

    if shutil.which("qsub"):
        log.info("qsub found on this host; running the driver script locally.")
        completed = subprocess.run(
            ["bash", str(local_script)], check=False, capture_output=True, text=True
        )
        exit_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    else:
        host, user, password = prompt_ssh_credentials(
            remote_host=remote_host, remote_user=remote_user,
            host_aliases=pth.CLUSTER_HOST_ALIASES,
        )
        cluster_path = publish_sge_driver_script(
            local_script, remote_script, host=host, user=user, password=password,
        )
        exit_code, stdout, stderr = run_sge_script_ssh_capture(
            host, user, password, cluster_path, local_script_path=local_script,
        )

    if stdout.strip():
        log.info("submission output:\n%s", stdout.strip()[-4000:])
    if stderr.strip():
        log.warning("submission stderr:\n%s", stderr.strip()[-4000:])
    if exit_code != 0:
        raise click.ClickException(
            f"The submission script exited with code {exit_code}. Nothing may have been "
            f"queued; check the output above and {local_script}."
        )

    job_ids = parse_sge_submission_job_ids(stdout, stderr)
    log.ok(f"submitted {len(job_ids)} job(s): {', '.join(job_ids) or '(none parsed)'}")
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
@click.option("--extra-train-only", multiple=True,
              help="Same syntax, but never held out for validation. This is how stage 6's "
                   "pseudo-labelled cohort is fed back in. Repeatable.")
@click.option("--no-challenge", is_flag=True, default=False,
              help="Exclude the challenge cases from the training set.")
@click.option("--corpus-source", "corpus_sources", multiple=True,
              help="Unlabeled corpus source: 'name:modality=/path[:glob]'. Repeatable.")
@click.option("--num-folds", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--ct-context-window", type=float, nargs=2, default=None,
              help="Add a second input channel with a wide CT window (e.g. -100 900) beside "
                   "the narrow vessel window, restoring the anatomical context the narrow one "
                   "clips away. Changes the dataset, so stage 0 must be re-run.")
@click.option("--mr-context-percentiles", type=float, nargs=2, default=None,
              help="Percentiles for the MR half of the context channel (default 0 100).")
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
@click.option("--folds", type=str, default=None,
              help="Folds to train and score: a comma list, or 'all' to train on every case "
                   "with no validation split. Defaults to every fold — one fold holds out 5 "
                   "patients, which cannot separate two configurations from noise.")
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
@click.option("--sampling", type=click.Choice(["default", "rare_aware"]), default="default",
              show_default=True,
              help="Patch sampling. 'rare_aware' oversamples the cases carrying rare classes "
                   "and centres their patches on the rare class rather than on whichever of "
                   "the ~28 classes in that volume came up.")
@click.option("--sampling-temperature", type=float, default=None,
              help="Inverse-frequency exponent for --sampling rare_aware (default 0.5).")
@click.option("--sampling-oversample", type=float, default=None,
              help="Foreground-patch fraction under rare_aware (default 0.5; nnU-Net's is 0.33).")
# ---- stages 4-5 --------------------------------------------------------------
@click.option("--compare-to", type=str, default=None,
              help="Stage 3 run to compare this one against, case by case: a run name under "
                   "<results-root>/stage3_evaluate, or a path. Use the --from-scratch control "
                   "run — a cohort mean cannot tell you whether pre-training helped.")
@click.option("--min-volume-mm3", type=float, default=5.0, show_default=True)
@click.option("--largest-only", is_flag=True, default=False)
@click.option("--repair-gaps-mm", type=float, default=None,
              help="Stage 4: bridge same-class gaps up to this many mm (targets beta0/clDice).")
@click.option("--repair-adjacency", is_flag=True, default=False,
              help="Stage 4: reassign fragments with anatomically impossible neighbours.")
@click.option("--repair-lateral", is_flag=True, default=False,
              help="Stage 4: mirror fragments found on the wrong side of the midline.")
@click.option("--repair-close-radius", type=int, default=0, show_default=True,
              help="Stage 4: per-class closing before gap bridging.")
# ---- stage 6: self-training ---------------------------------------------------
@click.option("--pseudo-modality", type=click.Choice(["mr", "ct"]), default="mr",
              show_default=True, help="Stage 6: modality of the unlabeled cohort.")
@click.option("--pseudo-agreement", type=float, default=None,
              help="Stage 6: require this mean agreement between the folds' predictions. The "
                   "strongest filter; costs one inference pass per fold.")
@click.option("--pseudo-max-components", type=int, default=2, show_default=True,
              help="Stage 6: components a class may have before it counts as fragmented.")
@click.option("--pseudo-max-fragmented", type=int, default=3, show_default=True,
              help="Stage 6: fragmented classes tolerated per case.")
@click.option("--pseudo-max-invalid", type=int, default=2, show_default=True,
              help="Stage 6: impossible adjacencies tolerated per case.")
@click.option("--pseudo-volume-range", type=float, nargs=2, default=(0.5, 2.0),
              show_default=True,
              help="Stage 6: accepted foreground volume, as a multiple of the annotated median.")
@click.option("--pseudo-max-accepted", type=int, default=None,
              help="Stage 6: keep at most this many pseudo-cases, best first.")
# ---- shared ------------------------------------------------------------------
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default=None)
@click.option("--num-processes", type=int, default=1, show_default=True)
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--challenge-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
# ---- submission ---------------------------------------------------------------
@click.option("--submit", type=click.Choice(["local", "sge"], case_sensitive=False),
              default="local", show_default=True)
@click.option("--container", type=click.Path(path_type=Path), default=None)
@click.option("--sge-project", type=str, default=None,
              help="SGE project (-P), overriding sge.json. The project selects the queue and "
                   "how a GPU is requested, so a CPU run usually needs a different one. "
                   "List valid names on the login node with: qconf -sprjl")
@click.option("--src-dir", type=click.Path(path_type=Path), default=None)
@click.option("--base-hold", type=str, default=None)
@click.option("--emit-script", "emit_script", type=click.Path(path_type=Path), default=None,
              help="Where to write the driver script (default: a staged temporary file).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Write the submission script and stop; queue nothing.")
@click.option("--no-remote", is_flag=True, default=False,
              help="Write the script but do not run it — print the command to run by hand on "
                   "the login node.")
@click.option("--remote-host", type=str, default=None,
              help="Cluster login node (short alias or hostname). Prompted for if omitted, "
                   "and skipped entirely when qsub exists on this host.")
@click.option("--remote-user", type=str, default=None,
              help="SSH user for the login node. Prompted for if omitted; the password is "
                   "always prompted for and never taken from an argument.")
# ---- monitoring ---------------------------------------------------------------
@click.option("--tensorboard", is_flag=True, default=False,
              help="Mirror stage 1 / stage 2 training logs into TensorBoard events under "
                   "<results-root>/tensorboard, live.")
@click.option("--tensorboard-serve", type=click.Choice(list(tb.SERVE_MODES)), default="auto",
              show_default=True,
              help="Where the TensorBoard server runs. 'auto': locally for --submit local, "
                   "nowhere for --submit sge (the command to serve the mounted cluster tree "
                   "is logged instead). 'cluster' submits a CPU-only server job.")
@click.option("--tensorboard-port", type=int, default=tb.DEFAULT_PORT, show_default=True)
@click.option("--tensorboard-interval", type=float, default=tb.DEFAULT_INTERVAL,
              show_default=True, help="Seconds between mirror passes.")
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

    # Every fold by default. A single fold is 5 held-out patients; with 36 classes, several of
    # them rare, the fold-to-fold spread is larger than the differences worth measuring, so a
    # one-fold run can only ever be a smoke test.
    num_folds = kw["num_folds"] or cfg.DEFAULT_NUM_FOLDS
    folds_spec = kw["folds"] or ",".join(str(i) for i in range(int(num_folds)))
    if kw["folds"] is None:
        log.info("--folds unset; running all %d folds. Pass --folds 0 for a smoke test.",
                 int(num_folds))

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
        folds=parse_folds(folds_spec),
        include_challenge=not kw["no_challenge"], export_model=not kw["no_export_model"],
        ssl_trainer=kw["ssl_trainer"] or cfg.DEFAULT_SSL_TRAINER,
        ssl_config=kw["ssl_config"] or cfg.DEFAULT_SSL_CONFIG,
        sampling_temperature=(
            kw["sampling_temperature"] if kw["sampling_temperature"] is not None
            else sampling_util.DEFAULT_TEMPERATURE
        ),
        **{k: kw[k] for k in (
            "dataprep_target", "modality", "extra_train", "extra_train_only",
            "corpus_sources", "num_folds", "seed",
            "ct_context_window", "mr_context_percentiles",
            "pretrain_source", "checkpoint_name", "bundle_name", "ssl_loss", "ssl_loss_config",
            "ssl_patch_size", "ssl_batch_size", "ssl_epochs", "ssl_lr", "init_checkpoint_name",
            "loss_config", "adaptation_mode", "baseline_planner", "target_spacing",
            "patch_size", "batch_size",
            "num_epochs", "num_gpus", "tensorboard", "tensorboard_interval", "compare_to",
            "sampling", "sampling_oversample",
            "from_scratch", "min_volume_mm3", "largest_only", "num_processes", "workers",
            "repair_gaps_mm", "repair_adjacency", "repair_lateral", "repair_close_radius",
            "pseudo_modality", "pseudo_agreement", "pseudo_max_components",
            "pseudo_max_fragmented", "pseudo_max_invalid", "pseudo_volume_range",
            "pseudo_max_accepted",
            "overwrite",
        )},
    )

    set_sge_project_override(kw["sge_project"])

    serve_mode = tb.resolve_serve_mode(kw["tensorboard_serve"], submit=submit)
    if serve_mode == "cluster":
        if submit != "sge":
            raise click.UsageError(
                "--tensorboard-serve cluster only applies to --submit sge; for a local run "
                "use --tensorboard-serve local."
            )
        if not kw["tensorboard"]:
            raise click.UsageError(
                "--tensorboard-serve cluster without --tensorboard would submit a server with "
                "nothing writing events. Add --tensorboard."
            )

    log.info("topbrain | submit=%s backend=%s device=%s label_set=%s loss=%s",
             submit, backend, device, label_set, loss)
    log.info("  stages : %s", ", ".join(f"{s} ({st.STAGE_LABELS[s]})" for s in selected))
    log.info("  local  challenge=%s results=%s", local.challenge_root, local.results_root)
    log.info("  active nnunet_raw=%s nnssl_raw=%s", active.nnunet_raw, active.nnssl_raw)
    if kw["tensorboard"]:
        log.info("  tboard events=%s serve=%s",
                 tb.tensorboard_root(active.results_root), serve_mode)

    if submit == "local":
        # The stages do their own mirroring; the server here spans the whole run so stage 1
        # and stage 2 appear side by side under one URL.
        with tb.local_server(
            active.results_root,
            enabled=kw["tensorboard"] and serve_mode == "local",
            port=kw["tensorboard_port"],
        ):
            _run_local(active, selected, options)
        return

    container = kw["container"] or cfg.CONTAINER_PATH
    if container is None:
        raise click.UsageError(
            "--submit sge needs a container: pass --container, or set "
            "pipelines.topbrain.default_sge_container_root in sge.json."
        )
    tensorboard_kwargs = dict(
        serve_mode=serve_mode, enabled=kw["tensorboard"], selected=selected,
        label_set=label_set, port=kw["tensorboard_port"],
        interval=kw["tensorboard_interval"],
    )
    _submit_via_login_node(
        active=active, local=local, selected=selected, options=options,
        container=container, src_dir=kw["src_dir"], base_hold=kw["base_hold"],
        label_set=label_set, loss=loss, emit_script=kw["emit_script"],
        dry_run=kw["dry_run"], no_remote=kw["no_remote"],
        remote_host=kw["remote_host"], remote_user=kw["remote_user"],
        tensorboard=tensorboard_kwargs,
    )

    # Serving blocks, so it has to come after the jobs are queued. The submitted jobs are
    # unaffected by stopping it.
    if serve_mode == "local" and kw["tensorboard"] and not kw["dry_run"] and not kw["no_remote"]:
        here, _mounted = tb.workstation_view(active, local)
        tb.watch_and_serve(
            here, stages=selected, label_set=label_set,
            port=kw["tensorboard_port"], interval=kw["tensorboard_interval"],
        )


__all__ = ["STAGE_MODULES", "main"]


if __name__ == "__main__":
    main()
