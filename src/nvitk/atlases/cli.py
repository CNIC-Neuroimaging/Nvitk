"""``nvitk-atlas``: list, inspect and export the atlases installed sources offer.

.. code-block:: bash

    nvitk-atlas list                         # installed atlases, every source
    nvitk-atlas list --all --source nilearn  # include what could be downloaded
    nvitk-atlas info fsl:HarvardOxford-Cortical
    nvitk-atlas export fsl:JHU-labels --variant 1mm --out ~/atlases
    nvitk-atlas export nilearn:aal --download --out ~/atlases
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

from nvitk.atlases.base import SOURCES, get_atlas, list_atlases, load_atlas


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Atlases from FSL, FreeSurfer/FastSurfer, ANTsXNet, nilearn and your own settings."""


@main.command("list")
@click.option("--source", type=click.Choice(sorted(SOURCES)), default=None, help="Only this source.")
@click.option("--all", "include_all", is_flag=True, help="Include atlases that would have to be downloaded.")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def list_cmd(source: str | None, include_all: bool, as_json: bool) -> None:
    """List atlases (installed ones, unless --all)."""
    errors: list[str] = []
    atlases = list_atlases(source, include_downloadable=include_all, on_error=lambda s, e: errors.append(f"{s}: {e}"))
    if as_json:
        click.echo(json.dumps([
            {"id": a.id, "name": a.name, "kind": a.kind, "space": a.space, "species": a.species,
             "status": a.status, "variants": list(a.variants)} for a in atlases
        ], indent=2))
        return
    if not atlases:
        click.echo("No atlases found." + ("" if include_all else " Try --all to see downloadable ones."))
    width = max((len(a.id) for a in atlases), default=10)
    for a in atlases:
        flag = "" if a.available else click.style(" [download]", fg="yellow")
        variants = f"  ({', '.join(a.variants)})" if a.variants else ""
        click.echo(f"{a.id:<{width}}  {a.kind:<13} {a.space:<10} {a.name}{variants}{flag}")
    for line in errors:
        click.echo(click.style(f"skipped {line}", fg="red"), err=True)


@main.command("info")
@click.argument("atlas_id")
@click.option("--variant", default=None)
@click.option("--labels/--no-labels", default=True, help="Print the label table.")
def info_cmd(atlas_id: str, variant: str | None, labels: bool) -> None:
    """Describe one atlas: files and labels (installed atlases only)."""
    info = get_atlas(atlas_id)
    click.echo(f"{info.id}: {info.name}")
    click.echo(f"  kind {info.kind}, space {info.space}, species {info.species}, {info.status}")
    if info.variants:
        click.echo(f"  variants: {', '.join(info.variants)}")
    if info.description:
        click.echo(f"  {info.description}")
    if not info.available:
        return
    data = load_atlas(atlas_id, variant=variant)
    click.echo(f"  image: {data.image}")
    for key, path in data.files.items():
        click.echo(f"  {key}: {path}")
    if labels:
        click.echo(f"  {len(data.labels)} labels")
        for label in data.labels[:500]:
            click.echo(f"    {label.id:>5}  {label.name}")


@main.command("export")
@click.argument("atlas_id")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--variant", default=None)
@click.option("--download", is_flag=True, help="Fetch the atlas first if it is not on this machine.")
def export_cmd(atlas_id: str, out_dir: Path, variant: str | None, download: bool) -> None:
    """Copy an atlas image and write its label table (TSV and JSON) to --out."""
    data = load_atlas(atlas_id, variant=variant, download=download)
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = atlas_id.replace(":", "_").replace("/", "_") + (f"_{data.variant}" if data.variant else "")
    written = []
    if data.image is not None:
        suffix = "".join(data.image.suffixes[-2:]) if data.image.name.endswith(".nii.gz") else data.image.suffix
        target = out_dir / f"{stem}{suffix}"
        shutil.copyfile(data.image, target)
        written.append(target)
    if data.labels:
        tsv = out_dir / f"{stem}_labels.tsv"
        tsv.write_text(data.to_tsv(), encoding="utf-8")
        meta = out_dir / f"{stem}.json"
        meta.write_text(json.dumps({
            "id": data.info.id, "name": data.info.name, "kind": data.info.kind, "space": data.info.space,
            "variant": data.variant, "labels": [{"id": l.id, "name": l.name, "rgb": list(l.rgb)} for l in data.labels],
        }, indent=2), encoding="utf-8")
        written += [tsv, meta]
    for path in written:
        click.echo(f"wrote {path}")


if __name__ == "__main__":
    main()
