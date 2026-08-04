"""``nvitk-measure`` CLI — measurement tools."""

from __future__ import annotations

import json
from pathlib import Path

import click

from nvitk.cli._common import backend_option, io_options, mask_option, submit_options
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.measure import dice, surface_metrics, suv_stats, volume_cc, volume_mm3

log = Logger()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Measurement tools."""


def _write_metrics(output_path: Path, data: dict) -> None:
    """Write *data* to *output_path* as pretty JSON (``.json`` suffix) or plain ``key: value`` lines
    otherwise."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".json",):
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        lines = [f"{k}: {v}" for k, v in data.items()]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"Wrote metrics to {output_path}")


@main.command("volume")
@io_options
@mask_option
@backend_option(False)
def cmd_volume(input_path: Path, output_path: Path, mask_path: Path | None) -> None:
    """Compute mask volume (mm³ and cc)."""
    mask = imread(mask_path or input_path)
    data = {
        "volume_mm3": volume_mm3(mask),
        "volume_cc": volume_cc(mask),
    }
    _write_metrics(output_path, data)


@main.command("suv")
@io_options
@mask_option
@backend_option(False)
def cmd_suv(input_path: Path, output_path: Path, mask_path: Path | None) -> None:
    """SUV statistics inside a mask."""
    pet = imread(input_path)
    mask = imread(mask_path) if mask_path else pet
    stats = suv_stats(pet, mask)
    _write_metrics(output_path, dict(stats))


@main.command("dice")
@io_options
@mask_option
@backend_option(False)
@click.option("--reference", "-r", type=click.Path(path_type=Path, exists=True), required=True)
def cmd_dice(input_path: Path, output_path: Path, mask_path: Path | None, reference: Path) -> None:
    """Dice coefficient between mask and reference."""
    pred = imread(mask_path or input_path)
    ref = imread(reference)
    _write_metrics(output_path, {"dice": float(dice(pred, ref))})


@main.command("surface-metrics")
@io_options
@mask_option
@backend_option(False)
@click.option("--reference", "-r", type=click.Path(path_type=Path, exists=True), required=True)
def cmd_surface(input_path: Path, output_path: Path, mask_path: Path | None, reference: Path) -> None:
    """Surface distance metrics vs reference."""
    pred = imread(mask_path or input_path)
    ref = imread(reference)
    metrics = surface_metrics(pred, ref)
    _write_metrics(output_path, dict(metrics))


if __name__ == "__main__":
    main()
