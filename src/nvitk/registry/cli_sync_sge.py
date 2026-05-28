"""CLI: sync ``.nvitk/sge.json`` nvitk container paths from the container registry."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from nvitk.registry.containers import (
    registry_path,
    resolve_nvitk_cluster_sif,
    sync_default_sge_json,
    sync_sge_nvitk_container,
)


@click.command("nvitk-sync-sge-container")
@click.option(
    "--sge-json",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Path to sge.json (default: repo .nvitk/sge.json).",
)
@click.option("--version", default=None, help="Registry version tag (default: projects.nvitk.latest).")
@click.option("--dry-run", is_flag=True, help="Print resolved path without writing.")
def main(sge_json: Path | None, version: str | None, dry_run: bool) -> None:
    """Update paths.nvitk_container from containers.json latest nvitk entry."""
    reg = registry_path()
    if reg is None:
        raise click.ClickException("Container registry not found (containers.json).")

    if sge_json is None:
        path = sync_default_sge_json(version=version, dry_run=dry_run)
    else:
        path = sync_sge_nvitk_container(sge_json, version=version, dry_run=dry_run)

    resolved = resolve_nvitk_cluster_sif(version=version)
    click.echo(f"registry: {reg}")
    click.echo(f"nvitk cluster SIF: {resolved}")
    if dry_run:
        click.echo(f"dry-run: would set nvitk_container → {path}")
    else:
        click.echo(f"updated nvitk_container → {path}")


if __name__ == "__main__":
    main(standalone_mode=True)
