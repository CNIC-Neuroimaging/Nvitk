"""``nvitk-transform`` CLI — geometric transforms."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.cli._common import backend_option, dispatch_tool, io_options, submit_options
from nvitk.io import imread
from nvitk.transform import isotropy, oblique_slice, resample_to


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Geometric transform tools."""


@main.command("resample")
@io_options
@backend_option(True)
@submit_options
@click.option("--reference", "-r", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--order", type=int, default=1)
def cmd_resample(
    input_path, output_path, reference, backend, submit, emit_script,
    direct_submit, no_remote, dry_run, order,
):
    def runner(image, *, reference_path, order):
        target = imread(reference_path, backend="numpy")
        return resample_to(image, target, order=order)
    dispatch_tool(
        tool="transform", subcommand="resample", module_file="transform.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=None, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=runner,
        runner_kwargs={"reference_path": str(reference), "order": order},
    )


@main.command("isotropy")
@io_options
@backend_option(True)
@submit_options
def cmd_isotropy(input_path, output_path, backend, submit, emit_script, direct_submit, no_remote, dry_run):
    dispatch_tool(
        tool="transform", subcommand="isotropy", module_file="transform.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=None, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=isotropy,
    )


@main.command("oblique-slice")
@io_options
@backend_option(True)
@submit_options
@click.option("--point", nargs=3, type=float, required=True)
@click.option("--normal", nargs=3, type=float, required=True)
def cmd_oblique(input_path, output_path, backend, submit, emit_script, direct_submit, no_remote, dry_run, point, normal):
    def runner(image, *, point, normal):
        return oblique_slice(image, point=point, normal=normal)
    dispatch_tool(
        tool="transform", subcommand="oblique-slice", module_file="transform.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=None, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=runner,
        runner_kwargs={"point": tuple(point), "normal": tuple(normal)},
    )


if __name__ == "__main__":
    main()
