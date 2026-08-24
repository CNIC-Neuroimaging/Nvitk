"""CLI: upload qvtpy/eICAB results to XNAT session resources.

Requires a complete qvtpy tree through stage-7 morphometrics before uploading
the ``qvtpy`` resource (stage2–stage7 by default; stage7 is always enforced).
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.stage0_download import resolve_subjects_for_xnat_pipeline
from nvitk.pipes.qvtpy.util.io.paths import layout_cluster, layout_local
from nvitk.pipes.qvtpy.util.io.xnat_upload import (
    DEFAULT_XNAT_UPLOAD_STAGES,
    parse_require_stages,
    run_xnat_upload,
)
from nvitk.core.click_config import config_dir_click_option

log = Logger()

_DEFAULT_REQUIRE_STAGES = ",".join(DEFAULT_XNAT_UPLOAD_STAGES)


@click.command("nvitk-qvtpy-xnat-upload")
@config_dir_click_option()
@click.option(
    "--submit",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Read results from local disk or fetch from cluster via SSH (SGE layout).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Local results root (default: local layout; ignored when --submit sge).",
)
@click.option("--subjects", default=None, help="Comma-separated subject ids or cohort alias PESA-Brain.")
@click.option("--subjects-file", type=click.Path(path_type=Path), default=None)
@click.option("--xnat-config", "xnat_config_path", type=click.Path(path_type=Path), default=None)
@click.option("--remote-host", default=None, help="(sge) SSH hostname or alias.")
@click.option("--remote-user", default=None, help="(sge) SSH username.")
@click.option(
    "--require-stages",
    default=_DEFAULT_REQUIRE_STAGES,
    show_default=True,
    help=(
        "QVTpy stages that must be complete before uploading the qvtpy resource. "
        "stage7 (morphometrics) is always required for a complete pipeline upload."
    ),
)
@click.option("--upload-eicab/--no-upload-eicab", default=True, show_default=True)
@click.option("--upload-qvtpy/--no-upload-qvtpy", default=True, show_default=True)
@click.option(
    "--skip-existing/--overwrite-existing",
    default=False,
    show_default=True,
    help="Skip XNAT resources that already have files, or overwrite them.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Log actions without POSTing to XNAT.")
def main(
    submit: str,
    output_root: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    xnat_config_path: Path | None,
    remote_host: str | None,
    remote_user: str | None,
    require_stages: str,
    upload_eicab: bool,
    upload_qvtpy: bool,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Upload complete eicab/ + qvtpy/ trees (through stage-7 morphometrics) to XNAT."""
    Logger()
    if subjects is None and subjects_file is None:
        raise click.ClickException("Provide --subjects or --subjects-file.")

    try:
        required = parse_require_stages(require_stages)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    profile = load_xnat_profile(xnat_config_path)
    xnat_config = resolve_xnat_connection(profile)

    subject_list, xnat_config = resolve_subjects_for_xnat_pipeline(
        subjects=subjects,
        subjects_file=subjects_file,
        xnat_config=xnat_config,
    )
    if not subject_list:
        raise click.ClickException("No subjects to upload.")

    local_paths = layout_local(results_root=output_root)
    cluster_paths = layout_cluster(results_root=output_root)
    cluster_mode = submit.lower() == "sge"

    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_password: str | None = None
    remote_results: Path | None = None
    if cluster_mode:
        remote_results = cluster_paths.results_root
        from nvitk.pipes.qvtpy.util.io.cluster_upload import prompt_ssh_credentials

        ssh_host, ssh_user, ssh_password = prompt_ssh_credentials(
            remote_host=remote_host,
            remote_user=remote_user,
            host_aliases=cfg.CLUSTER_HOST_ALIASES,
        )

    run_xnat_upload(
        subject_list,
        output_root=local_paths.results_root,
        xnat_config=xnat_config,
        required_stages=required,
        upload_eicab=upload_eicab,
        upload_qvtpy=upload_qvtpy,
        overwrite=not skip_existing,
        skip_existing=skip_existing,
        dry_run=dry_run,
        results_source="cluster" if cluster_mode else "local",
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_password=ssh_password,
        remote_results_root=remote_results,
    )


if __name__ == "__main__":
    main()
