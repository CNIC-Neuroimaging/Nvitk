#!/usr/bin/env python3
"""Sync XNAT image metadata into the dataset DB.

This wrapper combines four dataset sync steps for PESA-Brain:

1. XNAT subjects / sessions / scans metadata for TOF + 4D-flow scans
2. Scan-level ``NIFTI`` resource files (phase/magnitude per 4D-flow scan + TOF)
3. Experiment-level ``4dflows`` derivative NIfTIs
4. Experiment-level ``eicab`` and ``qvtpy`` pipeline resources
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from nvitk.db.repo import DataRepo
from nvitk.db.xnat import sync_xnat_project
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_4dflows_assets import sync_xnat_4dflows_assets
from nvitk.db.xnat_pipeline_resources import sync_xnat_pipeline_resources
from nvitk.db.xnat_scan_nifti_assets import sync_xnat_scan_nifti_assets

DEFAULT_SEQUENCES = "TOF,4DFLOW_AP,4DFLOW_RL,4DFLOW_FH"


@click.command()
@click.option("--dataset-root", type=click.Path(path_type=Path), default=Path("dataset/nvitk-dataset"), show_default=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None, help="XNAT config profile.")
@click.option("--server", type=str, default=None, help="XNAT server URL.")
@click.option("--project", type=str, default="PESA_Brain", show_default=True, help="XNAT project id.")
@click.option("--user", type=str, default=None, help="XNAT username.")
@click.option("--password", type=str, default=None, help="XNAT password.")
@click.option("--netrc-file", type=click.Path(path_type=Path), default=None, help="Optional netrc file.")
@click.option("--catalog-path", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--subjects", type=str, default=None, help="Comma/space separated subject ids.")
@click.option("--subjects-file", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--id-type", type=click.Choice(["subject", "mrid"], case_sensitive=False), default="subject", show_default=True)
@click.option("--sequences", type=str, default=DEFAULT_SEQUENCES, show_default=True, help="XNAT scan sequences to sync.")
@click.option("--scan-nifti-resource-label", type=str, default="NIFTI", show_default=True)
@click.option("--pipeline-resources", type=str, default="eicab,qvtpy", show_default=True)
@click.option("--fourdflows-resource-label", type=str, default="4dflows", show_default=True)
@click.option("--download-4dflows", is_flag=True, help="Download experiment-level 4dflows resources while indexing.")
@click.option("--fourdflows-download-root", type=click.Path(path_type=Path), default=None, help="Local root for downloaded 4dflows files.")
@click.option("--download-scan-niftis", is_flag=True, help="Download scan-level NIfTI resources while indexing.")
@click.option("--scan-nifti-download-root", type=click.Path(path_type=Path), default=None, help="Local root for downloaded scan NIfTI files.")
@click.option("--download-pipeline-resources", is_flag=True, help="Download experiment-level qvtpy/eICAB bundles while indexing.")
@click.option("--pipeline-download-root", type=click.Path(path_type=Path), default=None, help="Local root for downloaded qvtpy/eICAB resources.")
@click.option("--overwrite-downloads", is_flag=True, help="Replace existing local download directories.")
@click.option("--build-sqlite-index", is_flag=True, default=True, help="Rebuild the SQLite cache after all sync steps.")
def main(
    dataset_root: Path,
    config_path: Path | None,
    server: str | None,
    project: str,
    user: str | None,
    password: str | None,
    netrc_file: Path | None,
    catalog_path: Path | None,
    subjects: str | None,
    subjects_file: Path | None,
    id_type: str,
    sequences: str,
    scan_nifti_resource_label: str,
    fourdflows_resource_label: str,
    pipeline_resources: str,
    download_4dflows: bool,
    fourdflows_download_root: Path | None,
    download_scan_niftis: bool,
    scan_nifti_download_root: Path | None,
    download_pipeline_resources: bool,
    pipeline_download_root: Path | None,
    overwrite_downloads: bool,
    build_sqlite_index: bool,
) -> None:
    profile = load_xnat_profile(config_path)
    conn = resolve_xnat_connection(
        profile,
        server=server,
        project=project,
        user=user,
        password=password,
        netrc_file=str(netrc_file) if netrc_file else None,
    )
    repo = DataRepo(dataset_root, auto_scaffold=True)

    frames = sync_xnat_project(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        requested_sequences=sequences,
        download_dicoms=False,
        download_niftis=False,
        build_sqlite_index=False,
    )

    scan_df = sync_xnat_scan_nifti_assets(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        requested_sequences=sequences,
        resource_label=scan_nifti_resource_label,
        download_root=scan_nifti_download_root,
        download_files=download_scan_niftis,
        overwrite_downloads=overwrite_downloads,
        skip_existing_downloads=True,
        build_sqlite_index=False,
    )

    fourdflows_df = sync_xnat_4dflows_assets(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        resource_label=fourdflows_resource_label,
        download_root=fourdflows_download_root,
        download_files=download_4dflows,
        overwrite_downloads=overwrite_downloads,
        skip_existing_downloads=True,
        build_sqlite_index=False,
    )

    pipeline_df = sync_xnat_pipeline_resources(
        repo,
        conn,
        catalog_path=catalog_path,
        subjects=subjects,
        subjects_file=subjects_file,
        id_type=id_type,
        resources=[r.strip().lower() for r in pipeline_resources.split(",") if r.strip()],
        download_root=pipeline_download_root,
        download_resources=download_pipeline_resources,
        overwrite_downloads=overwrite_downloads,
        skip_existing_downloads=True,
        build_sqlite_index=False,
    )

    if build_sqlite_index:
        repo.build_sqlite_index()

    synced_tables = [name for name, frame in frames.items() if not frame.empty]
    click.echo(f"Synced scan metadata tables: {', '.join(synced_tables) if synced_tables else 'none'}")
    click.echo(f"Indexed scan NIfTI asset rows: {len(scan_df)}")
    click.echo(f"Indexed 4dflows asset rows: {len(fourdflows_df)}")
    click.echo(f"Indexed pipeline asset rows: {len(pipeline_df)}")
    if build_sqlite_index:
        click.echo("SQLite index rebuilt.")


if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except SystemExit as exc:
        raise SystemExit(exc.code) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
