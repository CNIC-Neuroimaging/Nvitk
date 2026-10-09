"""``nvitk-voxelwise`` CLI — mass-univariate voxelwise analysis with FSL ``randomise``."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Sequence

import click

from nvitk.core.logger import Logger
from nvitk.core.click_config import config_dir_click_option

log = Logger()

PIPELINE_ID = "voxelwise"

#: Where the image directory is mounted inside the container on the SGE path. ``ClusterPaths``
#: already binds one data root and one output root; the images are a third tree that may live
#: anywhere, so they get their own mount.
CONTAINER_IMAGES = "/nvitk/images/"

#: The two optional single-file inputs travel with the images, so on the ``--from-source local``
#: path they need no mount of their own — they arrive inside the uploaded inputs directory under
#: these fixed names.
CONTAINER_EXCLUDE_NAME = "exclude.csv"
CONTAINER_MASK_NAME = "mask_input.nii.gz"
CONTAINER_EXCLUDE_CSV = f"{CONTAINER_IMAGES}{CONTAINER_EXCLUDE_NAME}"
CONTAINER_MASK = f"{CONTAINER_IMAGES}{CONTAINER_MASK_NAME}"


# ──────────────────────────────────────────────────────────────────────────────
# SGE settings (``.nvitk/sge.json`` → ``pipelines.voxelwise``)
# ──────────────────────────────────────────────────────────────────────────────
def _sge_settings() -> dict[str, object]:
    """The merged ``defaults`` + ``pipelines.voxelwise`` block, read lazily.

    Lazily rather than at import (as :mod:`nvitk.cli.config` does) because almost every invocation
    of this command runs locally and has no business reading a cluster config to print ``--help``.
    """
    from nvitk.cluster import sge_json as sj

    return sj.merged_pipeline_flat(PIPELINE_ID)


#: Where the uploaded cohort lands, relative to the results root. Kept inside the output tree
#: rather than in a shared staging area so one analysis is one self-contained directory on the
#: cluster — it needs no extra config key, and it is removed with the results.
INPUTS_SUBDIR = "inputs"


def _ssh_credentials(remote_host: str | None, remote_user: str | None) -> tuple[str, str, str]:
    """Resolve ``(host, user, password)`` for the cluster.

    Order: ``NVITK_SGE_SSH_HOST`` / ``_USER`` / ``_PASSWORD`` environment variables, then the
    ``--remote-host`` / ``--remote-user`` options, then an interactive prompt.

    The env vars have to be sufficient on their own: the napari GUI runs this command as a
    subprocess with no tty, where a ``getpass`` fallback would hang with no visible reason.
    """
    import getpass
    import os

    from nvitk.cluster.remote_transfer import resolve_cluster_host

    host = (os.environ.get("NVITK_SGE_SSH_HOST", "") or (remote_host or "")).strip()
    user = (os.environ.get("NVITK_SGE_SSH_USER", "") or (remote_user or "")).strip()
    password = os.environ.get("NVITK_SGE_SSH_PASSWORD", "")

    if not host:
        host = click.prompt("SSH host (short name or IP)").strip()
    if not user:
        user = click.prompt("SSH user").strip()
    if not password:
        password = getpass.getpass("SSH password: ")
    if not (host and user and password):
        raise click.ClickException(
            "Incomplete SSH credentials. Set NVITK_SGE_SSH_HOST / _USER / _PASSWORD, or pass "
            "--remote-host and --remote-user and answer the prompt."
        )
    return resolve_cluster_host(host), user, password


def _verify_ssh(host: str, user: str, password: str) -> None:
    """Open and close one session, so a bad password fails now rather than after an upload."""
    from nvitk.cluster.remote_transfer import cluster_session

    try:
        with cluster_session(host=host, user=user, password=password) as session:
            pass
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"Cannot reach {user}@{host}: {exc}") from None
    log.info(f"SSH to {user}@{host} verified")


def _resolve_selection(**kw) -> tuple[list, int, int]:
    """Work out which images the analysis will actually use, locally.

    Returns ``(images, n_found, n_cohort)``. This is the same resolution the worker performs
    inside the container — run here only to decide *what to upload*, so a 4000-file directory
    sends the few hundred volumes the design keeps rather than the whole tree.
    """
    from nvitk.db.repo import get_repo
    from nvitk.measure.voxelwise import (
        DEFAULT_ID_NAMESPACES,
        VoxelwiseDesign,
        align_design_to_images,
        apply_prefilters,
        build_design_frame,
        cohort_id_subjects,
        cohort_subjects,
        parse_contrasts,
        parse_prefilters,
        resolve_cohort_images,
    )

    repo = get_repo()
    images = resolve_cohort_images(
        kw["image_dir"],
        repo=repo,
        include=kw["include"],
        exclude_csv=kw["exclude_csv"],
        id_pattern=kw["id_pattern"],
        namespaces=kw["namespaces"] or DEFAULT_ID_NAMESPACES,
        on_duplicate=kw["on_duplicate"],
    )
    n_found = len(images)

    cohort = kw["cohort"]
    if cohort:
        allowed = cohort_subjects(repo, cohort)
        images = [im for im in images if im.subject_uid in allowed]
    if kw["cohort_id"]:
        members = cohort_id_subjects(repo, str(kw["cohort_id"]))
        if members is not None:
            images = [im for im in images if im.subject_uid in members]
    n_cohort = len(images)

    if not cohort:
        # Without a cohort there is no design frame to intersect against, so the cohort scan is
        # the whole selection. The worker will say so too.
        return images, n_found, n_cohort

    evs = list(kw["evs"])
    rules = parse_prefilters(list(kw["prefilters"]))
    frame, _meta = build_design_frame(
        repo,
        pipeline=cohort,
        pipeline_kind=kw["pipeline_kind"],
        feature=kw["feature"],
        grouping=kw["grouping"],
        atlas=kw["atlas"],
        covariates=evs + [r.column for r in rules],
    )
    frame = apply_prefilters(frame, rules)
    design = VoxelwiseDesign(
        evs=tuple(evs),
        contrasts=parse_contrasts(list(kw["contrasts"]), evs, add_intercept=kw["add_intercept"]),
        add_intercept=kw["add_intercept"],
        demean=kw["demean"],
    )
    problems = design.validate(frame)
    if problems:
        raise click.ClickException(f"Design is not fittable:\n{problems}")
    images, _aligned = align_design_to_images(images, frame, design)
    return images, n_found, n_cohort


def _upload_inputs(
    pairs: Sequence[tuple[Path, str]],
    *,
    refresh: Sequence[tuple[Path, str]] = (),
    remote_inputs: str,
    host: str,
    user: str,
    password: str,
) -> None:
    """Send ``(local, remote)`` *pairs* and *refresh* to ``{out}/inputs/``.

    *pairs* skip a remote file of the same size — what makes re-running a cohort cheap. *refresh*
    is always sent: it holds the small files uploaded under fixed names (``design.mat``,
    ``mask_input…``, ``exclude.csv``), where a different file of the same size is not just
    possible but likely — two designs over the same subjects differ in values, not in length.
    """
    from nvitk.cluster.remote_transfer import cluster_session, upload_files

    every = [*pairs, *refresh]
    total_mb = sum(Path(p).stat().st_size for p, _ in every) / 1e6
    log.info(f"Uploading {len(every)} file(s), {total_mb:.0f} MB → {remote_inputs}")

    def report(done: int, total: int) -> None:
        """Log every 10% so a long transfer visibly advances."""
        step = max(1, total // 10)
        if done % step == 0 or done == total:
            log.info(f"  … {done}/{total}")

    with cluster_session(host=host, user=user, password=password) as session:
        uploaded, skipped = upload_files(session, pairs, on_progress=report)
        if refresh:
            sent, _ = upload_files(session, refresh, skip_existing=False)
            uploaded += sent
    log.info(f"Upload complete: {uploaded} sent, {skipped} already present")


#: Slots requested for ``--parallel --submit sge`` when neither ``--jobs`` nor ``sge.json``
#: (``pipelines.voxelwise.sge_pe_smp``) says otherwise. Inside the job the fragments run on exactly
#: these, through ``NSLOTS``.
DEFAULT_SGE_SLOTS = 4


def _parallel_slots(parallel: bool, jobs: int | None) -> int | None:
    """``-pe smp`` slots for a cluster run: ``None`` unless ``--parallel``."""
    if not parallel:
        return None
    if jobs:
        return int(jobs)
    configured = _sge_settings().get("sge_pe_smp")
    return int(configured) if configured else DEFAULT_SGE_SLOTS


def _submit_sge(
    *,
    argv: list[str],
    out_dir: Path,
    job_name: str,
    emit_script: Path | None,
    dry_run: bool,
    no_remote: bool = False,
    uploads: Sequence[tuple[Path, str]] = (),
    refresh: Sequence[tuple[Path, str]] = (),
    extra_binds: Sequence[tuple[Path, str]] = (),
    slots: int | None = None,
    remote_host: str | None = None,
    remote_user: str | None = None,
) -> None:
    """Upload what the analysis needs, publish a driver script, and run it on the login node.

    The worker re-enters this same CLI on the cluster, so there is exactly one implementation of
    the analysis and the SGE path cannot drift from the local one. What to upload (*uploads*,
    *refresh*, into ``<out>/inputs/``) and what to mount (*extra_binds*) is the caller's: ``run``
    sends images, ``randomise`` a stack and its design.

    *slots* requests ``-pe smp`` for ``--parallel``: the worker's ``randomise_parallel`` runs its
    fragments inside this one job, on the slots SGE granted (``NSLOTS``).

    ``qsub`` does not exist on a workstation, so this never shells out to it directly: the script
    is written locally, published over sshfs, and executed over SSH — the same six steps the qvtpy and
    pesa_fat pipelines take.
    """
    from nvitk.cluster import sge_json as sj
    from nvitk.cluster.remote_submit import run_sge_script_ssh_capture
    from nvitk.cluster.sge import (
        ClusterPaths,
        SgeResources,
        SingularityBinds,
        StageSpec,
        submit_stage,
        write_script_header,
    )
    from nvitk.cluster.sge_chunk import parse_sge_submission_job_ids
    from nvitk.cluster.sge_remote import publish_sge_driver_script, resolve_sge_script_paths
    from nvitk.db.settings_paths import configured_sge_dataset_root

    pipe = _sge_settings()
    paths_section = sj.paths_section()
    binds = SingularityBinds()

    # -o is always a cluster-visible root under --submit sge, so it is used verbatim; uploaded
    # inputs move into it.
    cluster_results = str(out_dir).rstrip("/")
    remote_inputs = f"{cluster_results}/{INPUTS_SUBDIR}"

    extra_env = {"PYTHONPATH": binds.src, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    mounts: list[tuple[Path, str]] = list(extra_binds)
    dataset_root = configured_sge_dataset_root()
    if dataset_root is not None:
        extra_env["NVITK_SGE"] = "1"
        extra_env["NVITK_DATASET_ROOT"] = str(dataset_root)
        mounts.append((dataset_root, str(dataset_root)))

    resources = SgeResources(
        project=str(pipe.get("sge_project") or "MCC"),
        account=str(pipe.get("sge_account") or "MCC"),
        ngpu=int(pipe.get("sge_ngpu") or 0),
        h_vmem=str(pipe.get("sge_h_vmem") or "40G"),
        queue=str(pipe.get("sge_queue")) if pipe.get("sge_queue") else None,
        pe_smp=int(slots) if slots else None,
    )
    log_dir, err_dir = sj.resolve_log_err_dirs(
        paths=paths_section,
        pipe=pipe,
        fallback_log=Path("/tmp/nvitk-sge/logs"),
        fallback_err=Path("/tmp/nvitk-sge/errs"),
    )
    cluster_paths = ClusterPaths(
        src=sj.resolve_nvitk_src_dir(fallback=Path(__file__).resolve().parents[2]),
        container=sj.resolve_nvitk_container(pipe=pipe),
        models=None,
        data_root=Path(cluster_results),
        output_root=Path(cluster_results),
        log_dir=log_dir,
        err_dir=err_dir,
    )
    spec = StageSpec(
        job_name=f"{pipe.get('sge_job_prefix') or 'nvitk-voxelwise'}_{job_name}"[:200],
        python_cmd=" ".join(shlex.quote(a) for a in argv),
        resources=resources,
        binds=binds,
        # randomise is CPU-only; --nv would ask the scheduler for a device it never touches.
        use_nv=False,
        extra_env=extra_env,
        extra_host_binds=tuple(mounts),
    )

    # This host's FSL says nothing about the container the job runs in, so the note is
    # unconditional. The nvitk image installs fsl-base/flirt/avwutils/warpfns: fslmerge is there,
    # randomise is not until fsl-randomise is added and the SIF rebuilt.
    status = _fsl_status()
    detail = "" if status.available else f" (this host: {status.reason})"
    click.echo(
        click.style("note: ", fg="yellow")
        + f"{cluster_paths.container.name} must ship fsl-randomise as well as fsl-avwutils; "
        f"without it the job fails on the queue at the randomise step{detail}.",
        err=True,
    )

    if dry_run:
        submit_stage(spec, cluster_paths, dry_run=True)
        if uploads or refresh:
            click.echo(
                f"(dry run) would upload {len(uploads) + len(refresh)} file(s) → {remote_inputs}"
            )
        click.echo("(dry run — nothing uploaded, nothing submitted)")
        return

    # ---- credentials, then transfer, then submit -----------------------------
    host = user = password = ""
    if not emit_script or not no_remote:
        host, user, password = _ssh_credentials(remote_host, remote_user)
        _verify_ssh(host, user, password)

    if uploads or refresh:
        _upload_inputs(
            uploads, refresh=refresh,
            remote_inputs=remote_inputs, host=host, user=user, password=password,
        )
    else:
        log.info("--from-source sge: inputs used in place on the cluster, uploading nothing")

    local_script, remote_script = resolve_sge_script_paths(
        emit_script,
        remote_scripts_dir=Path(str(pipe.get("default_sge_scripts_dir") or "/tmp/nvitk-sge")),
        default_basename=f"submit_voxelwise_{job_name}.sh",
    )
    with local_script.open("w", encoding="utf-8") as handle:
        write_script_header(handle, log_dir=log_dir, err_dir=err_dir, title="nvitk-voxelwise")
        submit_stage(spec, cluster_paths, emit=handle)
    log.info(f"Wrote SGE script: {local_script}")

    if emit_script is not None and no_remote:
        click.echo(f"Script written, not published: {local_script}")
        return

    cluster_exec = publish_sge_driver_script(
        local_script, remote_script, host=host, user=user, password=password
    )
    if no_remote:
        click.echo(f"Published, not submitted. On the login node:\n  bash {cluster_exec}")
        return

    exit_code, stdout, stderr = run_sge_script_ssh_capture(
        host, user, password, cluster_exec, local_script_path=local_script
    )
    if exit_code != 0:
        raise click.ClickException(
            f"Submission script failed (exit {exit_code}):\n{(stderr or stdout).strip()[-2000:]}"
        )
    job_ids = parse_sge_submission_job_ids(stdout, stderr)
    click.echo(f"Submitted: {', '.join(job_ids) if job_ids else '(no job id parsed)'}")
    click.echo(
        f"Results will land in {cluster_results}. Fetch them with:\n"
        f"  nvitk-voxelwise fetch {cluster_results} --to <local-dir>"
    )


def _fsl_status():
    """The FSL backend probe, imported lazily so ``--help`` costs nothing."""
    from nvitk.measure.voxelwise import fsl_backend_status

    return fsl_backend_status()


# ──────────────────────────────────────────────────────────────────────────────
# Shared options
# ──────────────────────────────────────────────────────────────────────────────
def cohort_options(f):
    """Attach the image-scan and cohort-selection options shared by ``run``, ``merge`` and ``design``."""
    f = click.option(
        "--image-dir",
        type=click.Path(path_type=Path, file_okay=False),
        required=True,
        help="Flat directory of MNI-normalised NIfTI volumes (no recursion).",
    )(f)
    f = click.option(
        "--include", default="*", show_default=True,
        help="Glob on the file name, e.g. '*_s8*' for 8 mm-smoothed volumes. "
             "The default takes every NIfTI in the directory.",
    )(f)
    f = click.option(
        "--exclude-csv",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=None,
        help="File of id-globs to drop, one per line (BMRI12345*).",
    )(f)
    f = click.option(
        "--id-pattern", default=None,
        help="Optional regex whose first group is the session id inside the file name. "
             "By default the id is auto-detected: every letters-then-digits token in the name "
             "(BMRI100102, IA004754, ...) is checked against subject_ids, so a directory mixing "
             "naming conventions resolves without a pattern.",
    )(f)
    f = click.option(
        "--on-duplicate",
        type=click.Choice(["error", "skip", "first", "last"], case_sensitive=False),
        default="error", show_default=True,
        help="When two files resolve to the same subject: error (list every conflict), "
             "skip (drop those subjects), or first/last (keep one, by filename order).",
    )(f)
    f = click.option(
        "--namespace", "namespaces", multiple=True,
        help="subject_ids namespace(s) to resolve the filename token against "
             "[default: mr_id, mri_id, session]. All three name a session/acquisition code, "
             "which is then standardised to subject_uid.",
    )(f)
    f = click.option(
        "--cohort", default=None,
        help="Restrict to subjects with measurements from this pipeline id "
             "(see `nvitk-voxelwise cohorts`). Also the design matrix's measurement source.",
    )(f)
    f = click.option(
        "--cohort-id", default=None,
        help="Restrict to a named cohort from cohort_membership (distinct from --cohort).",
    )(f)
    return f


def design_options(f):
    """Attach the GLM options (``--ev``, ``--contrast``, intercept/demean, measurement source)."""
    f = click.option(
        "--ev", "evs", multiple=True, required=True,
        help="Explanatory variable: a column of the subject-grain design frame "
             "(e.g. flow_mean__LMCA, age_at_mri, sex). Repeatable; order is the design's.",
    )(f)
    f = click.option(
        "--contrast", "contrasts", multiple=True,
        help="'+ev:name', '-ev:name', or explicit weights '0,1,0:name'. "
             "Repeatable. Default: one positive contrast per EV.",
    )(f)
    f = click.option(
        "--prefilter", "prefilters", multiple=True,
        help="Keep only subjects matching a measurement rule, e.g. 'flow_mean__LICA>=15'. "
             "Repeatable; rules combine with AND. Operators: >=, <=, !=, ==, >, <. "
             "The column need not be an EV.",
    )(f)
    f = click.option(
        "--intercept/--no-intercept", "add_intercept", default=True, show_default=True,
        help="Prepend a constant column to the design.",
    )(f)
    f = click.option(
        "--demean/--no-demean", default=True, show_default=True,
        help="Centre the EVs, as randomise expects alongside an intercept.",
    )(f)
    f = click.option(
        "--pipeline-kind", default="qvtpy", show_default=True,
        help="Measurement family the design frame is built from.",
    )(f)
    f = click.option(
        "--feature", default="flow_mean", show_default=True,
        help="Measurement feature spread into per-region columns.",
    )(f)
    f = click.option(
        "--grouping", default="vessel", show_default=True,
        help="How measurement regions are grouped into columns.",
    )(f)
    f = click.option("--atlas", default=None, help="Atlas preset for the measurement grouping.")(f)
    return f


def randomise_options(f):
    """Attach the ``randomise`` options shared by ``run`` and ``randomise``."""
    f = click.option("--n-perm", type=int, default=5000, show_default=True, help="Permutations.")(f)
    f = click.option("--tfce/--no-tfce", default=True, show_default=True,
                     help="Threshold-free cluster enhancement.")(f)
    f = click.option("--vox-corrp/--no-vox-corrp", "voxelwise_corrp", default=True,
                     show_default=True,
                     help="Also write voxelwise FWE-corrected p-maps (randomise -x).")(f)
    f = click.option("--uncorrp/--no-uncorrp", "uncorrected_p", default=True, show_default=True,
                     help="Also write uncorrected p-maps. Paired rather than a bare flag: on by "
                          "default, it needs a way to be switched off.")(f)
    f = click.option("--parallel", is_flag=True, default=False,
                     help="Use randomise_parallel: split the permutations into fragments run "
                          "side by side on this machine (or on the job's slots on the cluster).")(f)
    f = click.option("--jobs", type=click.IntRange(min=0), default=None,
                     help="Fragments run at once with --parallel. Default: $FSLSUB_PARALLEL if "
                          "set, else as many as the cores and free memory allow (each fragment "
                          "holds the whole 4D stack). With --submit sge, the -pe smp slots "
                          "requested [default: sge.json sge_pe_smp, else 4].")(f)
    f = click.option("--seed", type=int, default=None,
                     help="Permutation seed (reproducible runs). randomise_parallel seeds its "
                          "fragments itself.")(f)
    f = click.option("--out-name", default="randomise", show_default=True,
                     help="Output file-root name inside the results folder.")(f)
    return f


def submission_options(source_help: str):
    """Attach the ``--submit`` / cluster options; *source_help* describes ``--from-source``."""

    def decorate(f):
        """Apply the options to *f*."""
        f = click.option("--submit", type=click.Choice(["local", "sge"], case_sensitive=False),
                         default="local", show_default=True)(f)
        f = click.option("--emit-script", type=click.Path(path_type=Path), default=None,
                         help="Write the qsub driver script instead of submitting.")(f)
        f = click.option("--dry-run", is_flag=True, default=False,
                         help="With --submit sge, print the job without uploading or "
                              "submitting it.")(f)
        f = click.option("--from-source", type=click.Choice(["local", "sge"], case_sensitive=False),
                         default="local", show_default=True, help=source_help)(f)
        f = click.option("--no-remote", is_flag=True, default=False,
                         help="Upload and publish the driver script, but do not run it — print "
                              "the bash line to run on the login node yourself.")(f)
        f = click.option("--remote-host", default=None,
                         help="SSH host (or an alias from sge.json). Overridden by "
                              "NVITK_SGE_SSH_HOST.")(f)
        f = click.option("--remote-user", default=None,
                         help="SSH user. Overridden by NVITK_SGE_SSH_USER.")(f)
        return f

    return decorate


def friendly_errors(f):
    """Turn the resolvers' ``ValueError``/``FileNotFoundError`` into a clean Click error.

    Those messages are written for a person — they name the offending id, the missing column, the
    registered alternatives — and a traceback around them buries exactly that.
    """
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        """Run the command, converting expected failures into ``ClickException``."""
        try:
            return f(*args, **kwargs)
        except click.ClickException:
            raise
        except (ValueError, FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
            raise click.ClickException(str(exc)) from None

    return wrapper


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
def main() -> None:
    """Voxelwise analysis (FSL randomise).

    Stack a cohort's spatially normalised images into one 4D volume and fit the same GLM at every
    voxel, with permutation-based family-wise-error correction. The images and the design matrix
    are independent: images come from a directory, the design from database measurements.

    `randomise` skips all of that and runs on a 4D stack and design.mat / .con / .fts built
    elsewhere.
    """


# ──────────────────────────────────────────────────────────────────────────────
# cohorts
# ──────────────────────────────────────────────────────────────────────────────
@main.command("cohorts")
@friendly_errors
def cmd_cohorts() -> None:
    """List the pipeline ids --cohort accepts, with their subject counts."""
    from nvitk.db.repo import get_repo
    from nvitk.measure.voxelwise import available_cohorts

    options = available_cohorts(get_repo())
    selectable = [o for o in options if o.registered]
    other = [o for o in options if not o.registered]

    click.echo(click.style("--cohort accepts:", bold=True))
    for option in selectable:
        aliases = f"  aka {', '.join(option.aliases)}" if option.aliases else ""
        click.echo(f"  {option.pipeline_id:<20} {option.n_subjects:>6} subject(s){aliases}")
    if other:
        click.echo()
        click.echo(
            click.style("present in image_measurements but not registered ", fg="yellow")
            + "(not selectable):"
        )
        for option in other:
            click.echo(f"  {option.pipeline_id:<20} {option.n_subjects:>6} subject(s)")


# ──────────────────────────────────────────────────────────────────────────────
# design
# ──────────────────────────────────────────────────────────────────────────────
@main.command("design")
@cohort_options
@design_options
@click.option(
    "-o", "--output", "out_dir",
    type=click.Path(path_type=Path, file_okay=False), required=True,
    help="Where design.mat / design.con are written.",
)
@friendly_errors
def cmd_design(
    image_dir: Path, include: str, exclude_csv: Path | None, id_pattern: str | None,
    on_duplicate: str,
    namespaces: tuple[str, ...], cohort: str | None, cohort_id: str | None,
    evs: tuple[str, ...], contrasts: tuple[str, ...], prefilters: tuple[str, ...],
    add_intercept: bool, demean: bool,
    pipeline_kind: str, feature: str, grouping: str, atlas: str | None,
    out_dir: Path,
) -> None:
    """Build and write the design matrix and contrasts — no FSL, no permutations.

    Run this first on a new analysis: it reports the intersection and validates the GLM without
    spending an hour on a design that was never going to fit.
    """
    from nvitk.db.repo import get_repo
    from nvitk.measure.voxelwise import (
        DEFAULT_ID_NAMESPACES,
        VoxelwiseDesign,
        align_design_to_images,
        apply_prefilters,
        build_design_frame,
        cohort_id_subjects,
        cohort_subjects,
        parse_contrasts,
        parse_prefilters,
        resolve_cohort_images,
    )

    repo = get_repo()
    images = resolve_cohort_images(
        image_dir, repo=repo, include=include, exclude_csv=exclude_csv,
        id_pattern=id_pattern, namespaces=namespaces or DEFAULT_ID_NAMESPACES,
        on_duplicate=on_duplicate,
    )
    n_found = len(images)
    if cohort:
        allowed = cohort_subjects(repo, cohort)
        images = [im for im in images if im.subject_uid in allowed]
    if cohort_id:
        members = cohort_id_subjects(repo, str(cohort_id))
        if members is not None:
            images = [im for im in images if im.subject_uid in members]
    n_cohort = len(images)

    if not cohort:
        raise click.ClickException(
            "--cohort <pipeline_id> is required to build the design frame. "
            "Run `nvitk-voxelwise cohorts` to see the options."
        )
    rules = parse_prefilters(list(prefilters))
    frame, _meta = build_design_frame(
        repo, pipeline=cohort, pipeline_kind=pipeline_kind, feature=feature,
        grouping=grouping, atlas=atlas, covariates=list(evs) + [r.column for r in rules],
    )
    n_before_prefilter = len(frame)
    frame = apply_prefilters(frame, rules)
    design = VoxelwiseDesign(
        evs=tuple(evs),
        contrasts=parse_contrasts(list(contrasts), list(evs), add_intercept=add_intercept),
        add_intercept=add_intercept,
        demean=demean,
    )
    problems = design.validate(frame)
    if problems:
        raise click.ClickException(f"Design is not fittable:\n{problems}")

    images, aligned = align_design_to_images(images, frame, design)
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = design.write_mat(out_dir / "design.mat", aligned)
    con = design.write_con(out_dir / "design.con")
    (out_dir / "subjects.txt").write_text(
        "\n".join(f"{i}\t{im.subject_uid}\t{im.session_id}\t{im.path.name}"
                  for i, im in enumerate(images)) + "\n",
        encoding="utf-8",
    )
    click.echo(
        f"{n_found} image(s) found · {n_cohort} in cohort {cohort}"
        + (f" · {len(frame)} of {n_before_prefilter} pass the prefilter(s)" if rules else "")
        + f" · {len(images)} complete case(s)"
    )
    click.echo(f"Wrote {mat}\nWrote {con}\nWrote {out_dir / 'subjects.txt'}")


# ──────────────────────────────────────────────────────────────────────────────
# merge
# ──────────────────────────────────────────────────────────────────────────────
@main.command("merge")
@cohort_options
@click.option(
    "-o", "--output", "out_path",
    type=click.Path(path_type=Path, dir_okay=False), required=True,
    help="Output 4D NIfTI.",
)
@click.option("--no-validate", is_flag=True, default=False, help="Skip the common-space check.")
@friendly_errors
def cmd_merge(
    image_dir: Path, include: str, exclude_csv: Path | None, id_pattern: str | None,
    on_duplicate: str,
    namespaces: tuple[str, ...], cohort: str | None, cohort_id: str | None,
    out_path: Path, no_validate: bool,
) -> None:
    """Concatenate the resolved cohort into one 4D volume, in subject order."""
    from nvitk.db.repo import get_repo
    from nvitk.measure.voxelwise import (
        DEFAULT_ID_NAMESPACES,
        cohort_id_subjects,
        cohort_subjects,
        merge_4d,
        resolve_cohort_images,
        validate_common_space,
    )

    repo = get_repo()
    images = resolve_cohort_images(
        image_dir, repo=repo, include=include, exclude_csv=exclude_csv,
        id_pattern=id_pattern, namespaces=namespaces or DEFAULT_ID_NAMESPACES,
        on_duplicate=on_duplicate,
    )
    if cohort:
        allowed = cohort_subjects(repo, cohort)
        images = [im for im in images if im.subject_uid in allowed]
    if cohort_id:
        members = cohort_id_subjects(repo, str(cohort_id))
        if members is not None:
            images = [im for im in images if im.subject_uid in members]
    if not no_validate:
        validate_common_space(images)
    out = merge_4d(images, out_path)
    order = out_path.with_suffix("").with_suffix(".subjects.txt")
    order.write_text(
        "\n".join(f"{i}\t{im.subject_uid}\t{im.session_id}" for i, im in enumerate(images)) + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote {out} ({len(images)} volume(s))\nWrote {order}")


# ──────────────────────────────────────────────────────────────────────────────
# report
# ──────────────────────────────────────────────────────────────────────────────
@main.command("report")
@click.argument("results_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--threshold", type=float, default=0.95, show_default=True,
              help="1−p threshold; 0.95 means p < 0.05.")
@friendly_errors
def cmd_report(results_dir: Path, threshold: float) -> None:
    """Summarise a finished results folder: contrasts, maps, and suprathreshold voxel counts."""
    from nvitk.measure.voxelwise import STAT_KINDS, count_significant, load_voxelwise_result

    result = load_voxelwise_result(results_dir)
    manifest = result.manifest
    click.echo(click.style(f"{result.out_root.name}", bold=True))
    if manifest:
        evs = ", ".join(manifest.get("evs", [])) or f"{len(manifest.get('columns', []))} column(s)"
        click.echo(
            f"  {manifest.get('n_subjects', '?')} subject(s) · "
            f"{manifest.get('n_perm', '?')} permutation(s) · "
            f"EVs {evs} · FSL {manifest.get('fsl_version', '?')}"
        )
        if manifest.get("cohort"):
            click.echo(
                f"  cohort {manifest['cohort']}: {manifest.get('n_images_found', '?')} found → "
                f"{manifest.get('n_in_cohort', '?')} in cohort → {manifest.get('n_subjects', '?')} "
                "complete"
            )
        if manifest.get("inputs"):
            for label, key in (("stack", "stack"), ("design", "design_mat"),
                               ("contrasts", "design_con"), ("F-tests", "design_fts")):
                if manifest["inputs"].get(key):
                    click.echo(f"  {label:<10} {manifest['inputs'][key]}")
    for kind in sorted(result.maps):
        click.echo(f"\n  {kind} — {STAT_KINDS.get(kind, 'unknown kind')}")
        for name in result.names_for(kind):
            try:
                path = result.map_path(kind, name)
            except KeyError:
                continue
            if kind.endswith(("corrp_tstat", "corrp_fstat")):
                n = count_significant(path, threshold=threshold)
                click.echo(f"    {name:<24} {path.name}  ·  {n} voxel(s) p<{1 - threshold:.3g}")
            else:
                click.echo(f"    {name:<24} {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────────────────────
@main.command("run")
@cohort_options
@design_options
@click.option(
    "-o", "--output", "out_dir",
    type=click.Path(path_type=Path, file_okay=False), required=True,
    help="Results folder: design, 4D stack, mask, maps and manifest.json.",
)
@click.option("--mask", "mask_path", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Analysis mask [default: nilearn MNI152 brain mask on the input grid].")
@randomise_options
@submission_options(
    "Where --image-dir and --mask live. 'local' uploads the selected images to "
    "<output>/inputs/ on the cluster; 'sge' treats them as cluster paths already "
    "in place and uploads nothing."
)
@friendly_errors
def cmd_run(
    image_dir: Path, include: str, exclude_csv: Path | None, id_pattern: str | None,
    on_duplicate: str,
    namespaces: tuple[str, ...], cohort: str | None, cohort_id: str | None,
    evs: tuple[str, ...], contrasts: tuple[str, ...], prefilters: tuple[str, ...],
    add_intercept: bool, demean: bool,
    pipeline_kind: str, feature: str, grouping: str, atlas: str | None,
    out_dir: Path, mask_path: Path | None, n_perm: int, tfce: bool, voxelwise_corrp: bool,
    uncorrected_p: bool, parallel: bool, jobs: int | None, seed: int | None, out_name: str,
    submit: str, emit_script: Path | None, dry_run: bool,
    from_source: str, no_remote: bool, remote_host: str | None, remote_user: str | None,
) -> None:
    """Resolve the cohort, build the design, merge, and run ``randomise``."""
    _warn_jobs_without_parallel(jobs, parallel)
    if submit.lower() == "sge":
        remote_inputs = f"{str(out_dir).rstrip('/')}/{INPUTS_SUBDIR}"
        uploads: list[tuple[Path, str]] = []
        refresh: list[tuple[Path, str]] = []
        # The mask keeps its own extension: nibabel reads by name, and an uncompressed .nii
        # renamed .nii.gz is "not a gzip file" to the worker.
        mask_name = f"mask_input{_nifti_suffix(mask_path)}" if mask_path else CONTAINER_MASK_NAME
        container_mask = f"{CONTAINER_IMAGES}{mask_name}"
        if str(from_source).lower() == "local":
            # Which images the analysis keeps is decided here, on the workstation that can see
            # both the directory and the database — so only those files are sent, not the tree.
            selection, n_found, n_cohort = _resolve_selection(
                image_dir=image_dir, include=include, exclude_csv=exclude_csv,
                id_pattern=id_pattern, namespaces=namespaces, on_duplicate=on_duplicate,
                cohort=cohort, cohort_id=cohort_id, evs=evs, contrasts=contrasts,
                prefilters=prefilters, add_intercept=add_intercept, demean=demean,
                pipeline_kind=pipeline_kind, feature=feature, grouping=grouping, atlas=atlas,
            )
            click.echo(
                f"{n_found} image(s) found · {n_cohort} in cohort"
                f"{f' {cohort}' if cohort else ''} · {len(selection)} to upload"
            )
            uploads = [(im.path, f"{remote_inputs}/{im.path.name}") for im in selection]
            if mask_path is not None:
                refresh.append((mask_path.resolve(), f"{remote_inputs}/{mask_name}"))
            if exclude_csv is not None:
                refresh.append((exclude_csv.resolve(), f"{remote_inputs}/{CONTAINER_EXCLUDE_NAME}"))
            binds = [(Path(remote_inputs), CONTAINER_IMAGES)]
        else:
            # The inputs already live on the cluster and are not being copied, so the mask and
            # the exclusion list keep their own paths and need their own mounts.
            binds = [(Path(str(image_dir).rstrip("/")), CONTAINER_IMAGES)]
            if mask_path is not None:
                binds.append((Path(mask_path), container_mask))
            if exclude_csv is not None:
                binds.append((Path(exclude_csv), CONTAINER_EXCLUDE_CSV))
        _submit_sge(
            argv=_worker_argv(
                include=include, exclude_csv=exclude_csv,
                id_pattern=id_pattern, namespaces=namespaces,
                on_duplicate=on_duplicate, cohort=cohort,
                cohort_id=cohort_id, evs=evs, contrasts=contrasts,
                prefilters=prefilters,
                add_intercept=add_intercept, demean=demean, pipeline_kind=pipeline_kind,
                feature=feature, grouping=grouping, atlas=atlas,
                mask_path=mask_path, container_mask=container_mask, n_perm=n_perm, tfce=tfce,
                voxelwise_corrp=voxelwise_corrp, uncorrected_p=uncorrected_p,
                parallel=parallel, seed=seed, out_name=out_name,
            ),
            out_dir=out_dir,
            job_name=out_dir.name or "run",
            emit_script=emit_script,
            dry_run=dry_run,
            no_remote=no_remote,
            uploads=uploads,
            refresh=refresh,
            extra_binds=binds,
            slots=_parallel_slots(parallel, jobs),
            remote_host=remote_host,
            remote_user=remote_user,
        )
        return

    from nvitk.measure.voxelwise import DEFAULT_ID_NAMESPACES, run_voxelwise

    status = _fsl_status()
    if not status.available:
        raise click.ClickException(f"{status.reason}\n{status.install_hint()}")

    result = run_voxelwise(
        image_dir, out_dir,
        evs=list(evs), contrasts=list(contrasts), prefilters=list(prefilters),
        include=include, exclude_csv=exclude_csv, id_pattern=id_pattern,
        namespaces=namespaces or DEFAULT_ID_NAMESPACES, on_duplicate=on_duplicate,
        cohort=cohort, cohort_id=cohort_id,
        pipeline_kind=pipeline_kind, feature=feature, grouping=grouping, atlas=atlas,
        mask=mask_path, n_perm=n_perm, tfce=tfce, voxelwise_corrp=voxelwise_corrp,
        uncorrected_p=uncorrected_p, parallel=parallel, n_jobs=jobs, seed=seed,
        add_intercept=add_intercept, demean=demean, out_name=out_name,
    )

    click.echo(result.summary())
    click.echo(CORRP_NOTE)


#: Printed under every finished run's summary.
CORRP_NOTE = (
    "\nValues in *_corrp_* maps are 1−p (FWE-corrected), so 0.95 means p < 0.05 — "
    "not an effect size."
)


def _worker_argv(**kw) -> list[str]:
    """The in-container command line: the same ``run`` with the container's mount points.

    Everything the local path would do, with ``--submit local`` so the worker actually runs it.
    """
    argv = [
        "nvitk-voxelwise", "run",
        "--image-dir", CONTAINER_IMAGES,
        "--include", str(kw["include"]),
        "-o", "/nvitk/output/",
        "--n-perm", str(int(kw["n_perm"])),
        "--out-name", str(kw["out_name"]),
        "--submit", "local",
    ]
    if kw["id_pattern"]:
        argv += ["--id-pattern", str(kw["id_pattern"])]
    argv += ["--on-duplicate", str(kw["on_duplicate"])]
    for namespace in kw["namespaces"] or ():
        argv += ["--namespace", str(namespace)]
    if kw["exclude_csv"]:
        argv += ["--exclude-csv", CONTAINER_EXCLUDE_CSV]
    if kw["cohort"]:
        argv += ["--cohort", str(kw["cohort"])]
    if kw["cohort_id"]:
        argv += ["--cohort-id", str(kw["cohort_id"])]
    for ev in kw["evs"]:
        argv += ["--ev", str(ev)]
    for contrast in kw["contrasts"]:
        argv += ["--contrast", str(contrast)]
    for rule in kw["prefilters"]:
        argv += ["--prefilter", str(rule)]
    argv.append("--intercept" if kw["add_intercept"] else "--no-intercept")
    argv.append("--demean" if kw["demean"] else "--no-demean")
    argv += ["--pipeline-kind", str(kw["pipeline_kind"])]
    argv += ["--feature", str(kw["feature"])]
    argv += ["--grouping", str(kw["grouping"])]
    if kw["atlas"]:
        argv += ["--atlas", str(kw["atlas"])]
    if kw["mask_path"]:
        argv += ["--mask", kw.get("container_mask") or CONTAINER_MASK]
    argv.append("--tfce" if kw["tfce"] else "--no-tfce")
    argv.append("--vox-corrp" if kw["voxelwise_corrp"] else "--no-vox-corrp")
    argv.append("--uncorrp" if kw["uncorrected_p"] else "--no-uncorrp")
    if kw["parallel"]:
        argv.append("--parallel")
    if kw["seed"] is not None:
        argv += ["--seed", str(int(kw["seed"]))]
    return argv


def _warn_jobs_without_parallel(jobs: int | None, parallel: bool) -> None:
    """``--jobs`` sizes ``randomise_parallel``; on its own it would silently do nothing."""
    if jobs and not parallel:
        log.warning("--jobs only applies with --parallel; ignoring it.")


# ──────────────────────────────────────────────────────────────────────────────
# randomise (prepared inputs)
# ──────────────────────────────────────────────────────────────────────────────
#: Where ``<output>/inputs/`` appears inside the container: the output root is already mounted at
#: ``/nvitk/output/``, so uploaded inputs need no mount of their own.
CONTAINER_UPLOADED = f"/nvitk/output/{INPUTS_SUBDIR}/"


def _nifti_suffix(path: Path) -> str:
    """``.nii.gz`` or ``.nii`` — kept on upload, since nibabel reads by extension."""
    return ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix


@main.command("randomise")
@click.option("-i", "--stack", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="4D NIfTI, one subject per volume (e.g. from `nvitk-voxelwise merge` or "
                   "fslmerge -t).")
@click.option("-d", "--design-mat", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="Design matrix (VEST design.mat): row i is volume i of the stack.")
@click.option("-t", "--design-con", type=click.Path(path_type=Path, dir_okay=False), required=True,
              help="t-contrasts (VEST design.con).")
@click.option("-f", "--design-fts", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Optional F-tests (VEST design.fts), each a 0/1 selection of t-contrasts.")
@click.option("-m", "--mask", "mask_path", type=click.Path(path_type=Path, dir_okay=False),
              default=None,
              help="Analysis mask on the stack's grid [default: none — randomise tests every "
                   "voxel that is non-zero in the data; no MNI mask is assumed].")
@click.option("-o", "--output", "out_dir", type=click.Path(path_type=Path, file_okay=False),
              required=True, help="Results folder: design copies, maps and manifest.json.")
@click.option("--demean-data", is_flag=True, default=False,
              help="Demean the data temporally before fitting (randomise -D).")
@randomise_options
@submission_options(
    "Where the stack, design files and mask live. 'local' checks them here and uploads them "
    "to <output>/inputs/ on the cluster; 'sge' treats them as absolute cluster paths already "
    "in place and uploads nothing."
)
@friendly_errors
def cmd_randomise(
    stack: Path, design_mat: Path, design_con: Path, design_fts: Path | None,
    mask_path: Path | None, out_dir: Path, demean_data: bool,
    n_perm: int, tfce: bool, voxelwise_corrp: bool, uncorrected_p: bool, parallel: bool,
    jobs: int | None, seed: int | None, out_name: str,
    submit: str, emit_script: Path | None, dry_run: bool,
    from_source: str, no_remote: bool, remote_host: str | None, remote_user: str | None,
) -> None:
    """Run randomise on a prepared 4D stack and design files — no cohort, no database.

    For a design built anywhere: FSL's Glm GUI, another tool, or `design` and `merge` here. The
    stack and the files are checked against each other from their headers before anything runs
    or uploads, and every mismatch is reported at once.
    """
    _warn_jobs_without_parallel(jobs, parallel)
    files: dict[str, Path | None] = {
        "stack": stack, "mat": design_mat, "con": design_con, "fts": design_fts, "mask": mask_path,
    }

    if submit.lower() == "sge":
        remote_inputs = f"{str(out_dir).rstrip('/')}/{INPUTS_SUBDIR}"
        uploads: list[tuple[Path, str]] = []
        refresh: list[tuple[Path, str]] = []
        binds: list[tuple[Path, str]] = []
        if str(from_source).lower() == "local":
            from nvitk.measure.voxelwise import validate_randomise_inputs

            inputs = validate_randomise_inputs(
                stack, design_mat, design_con, fts=design_fts, mask=mask_path
            )
            click.echo(inputs.describe())
            # The stack keeps its own name — it is the one large file, uploaded only when the
            # cluster copy differs in size — while the small files go under fixed names and are
            # always re-sent.
            names = {
                "mat": "design.mat",
                "con": "design.con",
                "fts": "design.fts",
                "mask": f"mask_input{_nifti_suffix(mask_path) if mask_path else ''}",
            }
            stack_name = inputs.stack.name
            if stack_name in names.values():
                stack_name = f"stack_{stack_name}"
            names["stack"] = stack_name
            uploads = [(inputs.stack, f"{remote_inputs}/{stack_name}")]
            refresh = [
                (Path(path).resolve(), f"{remote_inputs}/{names[key]}")
                for key, path in files.items()
                if key != "stack" and path is not None
            ]
            in_container = {
                key: f"{CONTAINER_UPLOADED}{names[key]}"
                for key, path in files.items() if path is not None
            }
        else:
            relative = [str(p) for p in files.values() if p is not None and not p.is_absolute()]
            if relative:
                raise click.ClickException(
                    f"--from-source sge needs absolute cluster paths; got {relative}."
                )
            # Each input's folder is mounted at its own path, so the worker sees the paths as given.
            in_container = {key: str(path) for key, path in files.items() if path is not None}
            folders = sorted({str(Path(p).parent) for p in in_container.values()})
            binds = [(Path(folder), folder) for folder in folders]

        _submit_sge(
            argv=_randomise_worker_argv(
                in_container, n_perm=n_perm, tfce=tfce, voxelwise_corrp=voxelwise_corrp,
                uncorrected_p=uncorrected_p, parallel=parallel, seed=seed,
                demean_data=demean_data, out_name=out_name,
            ),
            out_dir=out_dir,
            job_name=out_dir.name or "randomise",
            emit_script=emit_script,
            dry_run=dry_run,
            no_remote=no_remote,
            uploads=uploads,
            refresh=refresh,
            extra_binds=binds,
            slots=_parallel_slots(parallel, jobs),
            remote_host=remote_host,
            remote_user=remote_user,
        )
        return

    from nvitk.measure.voxelwise import run_randomise_direct

    status = _fsl_status()
    if not status.available:
        raise click.ClickException(f"{status.reason}\n{status.install_hint()}")

    result = run_randomise_direct(
        stack, design_mat, design_con, out_dir,
        fts=design_fts, mask=mask_path, n_perm=n_perm, tfce=tfce,
        voxelwise_corrp=voxelwise_corrp, uncorrected_p=uncorrected_p,
        parallel=parallel, n_jobs=jobs, seed=seed, demean_data=demean_data, out_name=out_name,
    )
    click.echo(result.summary())
    click.echo(CORRP_NOTE)


def _randomise_worker_argv(paths: dict[str, str], **kw) -> list[str]:
    """The in-container ``randomise`` command line, reading *paths* as the container sees them."""
    argv = [
        "nvitk-voxelwise", "randomise",
        "-i", paths["stack"],
        "-d", paths["mat"],
        "-t", paths["con"],
        "-o", "/nvitk/output/",
        "--n-perm", str(int(kw["n_perm"])),
        "--out-name", str(kw["out_name"]),
        "--submit", "local",
    ]
    if paths.get("fts"):
        argv += ["-f", paths["fts"]]
    if paths.get("mask"):
        argv += ["-m", paths["mask"]]
    argv.append("--tfce" if kw["tfce"] else "--no-tfce")
    argv.append("--vox-corrp" if kw["voxelwise_corrp"] else "--no-vox-corrp")
    argv.append("--uncorrp" if kw["uncorrected_p"] else "--no-uncorrp")
    if kw["demean_data"]:
        argv.append("--demean-data")
    if kw["parallel"]:
        # No --jobs: inside the job the fragments size themselves to the granted slots (NSLOTS).
        argv.append("--parallel")
    if kw["seed"] is not None:
        argv += ["--seed", str(int(kw["seed"]))]
    return argv


# ──────────────────────────────────────────────────────────────────────────────
# fetch
# ──────────────────────────────────────────────────────────────────────────────
#: Pulled by default. The 4D stack and the uploaded inputs stay on the cluster: for 500 subjects
#: at 2 mm the stack alone is several GB, and nothing local reads it — the maps and the manifest
#: are what a viewer needs.
FETCH_PATTERNS: tuple[str, ...] = (
    "manifest.json", "design.mat", "design.con", "design.fts", "subjects.txt", "mask.nii.gz",
    "*tstat*.nii*", "*fstat*.nii*",
)


@main.command("fetch")
@friendly_errors
@click.argument("remote_dir")
@click.option("-t", "--to", "local_dir", type=click.Path(path_type=Path, file_okay=False),
              required=True, help="Local directory to write the results into.")
@click.option("--all", "fetch_all", is_flag=True, default=False,
              help="Pull the whole folder, including the 4D stack and the uploaded inputs.")
@click.option("--remote-host", default=None, help="SSH host (or an sge.json alias).")
@click.option("--remote-user", default=None, help="SSH user.")
def cmd_fetch(
    remote_dir: str, local_dir: Path, fetch_all: bool,
    remote_host: str | None, remote_user: str | None,
) -> None:
    """Download a finished results folder from the cluster over sshfs.

    ``run --submit sge`` returns as soon as the job is queued, so the maps appear on the cluster
    minutes or hours later. This is the other half of that round trip.
    """
    import fnmatch

    from nvitk.cluster.remote_transfer import (
        cluster_session,
        download_directory_tree,
        download_remote_file,
        remote_listdir,
    )

    host, user, password = _ssh_credentials(remote_host, remote_user)
    _verify_ssh(host, user, password)
    remote = str(remote_dir).rstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)

    if fetch_all:
        with cluster_session(host=host, user=user, password=password) as session:
            count = download_directory_tree(session, remote, local_dir)
        click.echo(f"Downloaded {count} file(s) → {local_dir}")
        return

    with cluster_session(host=host, user=user, password=password) as session:
        names = remote_listdir(session, remote)
        if not names:
            raise click.ClickException(f"Cannot list {remote}, or it is empty.")
        wanted = [
            n for n in sorted(names)
            if any(fnmatch.fnmatch(n, pattern) for pattern in FETCH_PATTERNS)
        ]
        if not wanted:
            raise click.ClickException(
                f"Nothing to fetch in {remote} — no maps or manifest. Has the job finished? "
                f"Use --all to pull the folder as-is."
            )
        for name in wanted:
            download_remote_file(session, f"{remote}/{name}", local_dir / name)
            log.info(f"  {name}")

    click.echo(f"Downloaded {len(wanted)} file(s) → {local_dir}")
    click.echo(
        "The 4D stack and the uploaded inputs were left on the cluster (--all pulls them too)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# status
# ──────────────────────────────────────────────────────────────────────────────
@main.command("status")
def cmd_status() -> None:
    """Report whether FSL is usable here, and which binary is missing if not."""
    import os

    from nvitk.measure.voxelwise import PARALLEL_ENV, available_cores

    status = _fsl_status()
    click.echo(status.summary())
    click.echo(f"  {'FSLDIR':<20} {status.fsldir or '(unresolved)'}")
    for name, path in sorted(status.binaries.items()):
        click.echo(f"  {name:<20} {path}")
    if status.supports_parallel():
        cap = os.environ.get(PARALLEL_ENV, "").strip()
        click.echo(
            "\n--parallel runs fragments through nvitk's local fsl_sub: "
            + (f"{cap} at a time ({PARALLEL_ENV})" if cap.isdigit() and int(cap) > 0
               else f"up to {available_cores()} at a time here, fewer if the stack needs the "
                    f"memory (set {PARALLEL_ENV} or --jobs to fix the number)")
            + "."
        )
    if not status.available:
        click.echo()
        click.echo(status.install_hint())
        sys.exit(1)


if __name__ == "__main__":
    main()
