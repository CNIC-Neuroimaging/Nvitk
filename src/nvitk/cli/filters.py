"""``nvitk-filter`` CLI — image filters."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.cli._common import backend_option, dispatch_tool, io_options, submit_options
from nvitk.filters.sliding_threshold import (
    binary_mask_sliding_threshold_2d,
    binary_mask_sliding_threshold_3d,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Image filtering tools."""


@main.command("sliding-threshold")
@io_options
@backend_option(True)
@submit_options
@click.option("--dim", type=click.Choice(["2d", "3d"]), default="3d")
@click.option("--step", type=float, default=0.001)
@click.option("--up-thresh", type=float, default=0.8)
def cmd_sliding_threshold(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    dim: str,
    step: float,
    up_thresh: float,
) -> None:
    """Sliding-threshold binary segmentation."""
    from nvitk.types import Image

    def runner(image, *, step, up_thresh, dim):
        data = image.data if isinstance(image, Image) else image
        if dim == "2d":
            mask, _ = binary_mask_sliding_threshold_2d(data, step=step, up_thresh=up_thresh)
        else:
            mask, _ = binary_mask_sliding_threshold_3d(data, step=step, up_thresh=up_thresh)
        if isinstance(image, Image):
            return image.with_data(mask.astype(image.data.dtype))
        return mask

    dispatch_tool(
        tool="filter",
        subcommand="sliding-threshold",
        module_file="filters.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=runner,
        runner_kwargs={"step": step, "up_thresh": up_thresh, "dim": dim},
    )


if __name__ == "__main__":
    main()
