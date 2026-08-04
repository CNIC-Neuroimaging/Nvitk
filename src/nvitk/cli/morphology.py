"""``nvitk-morph`` CLI — morphological image tools."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.cli._common import (
    backend_option,
    dispatch_tool,
    io_options,
    mask_option,
    submit_options,
)
from nvitk.morphology import (
    close,
    compute_centerlines,
    correct_siphon_centerlines,
    dilate,
    erode,
    fill_holes,
    label_connected,
    open,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Morphology tools (binary ops, centerlines, ICA siphon correction)."""


def _morph_op(op):
    """Wrap a base morphology *op* (e.g. :func:`~nvitk.morphology.dilate`) as a CLI runner that
    operates on the mask if one was supplied, else the input image itself."""

    def runner(image, mask=None, **kw):
        """Apply *op* to *mask* if given, else to *image*, forwarding any extra keyword arguments."""
        target = mask if mask is not None else image
        return op(target, **kw)
    return runner


@main.command("dilate")
@io_options
@mask_option
@backend_option(True)
@submit_options
@click.option("--footprint", type=int, default=1)
def cmd_dilate(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run, footprint):
    """Binary dilation."""
    dispatch_tool(tool="morph", subcommand="dilate", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=_morph_op(dilate),
        runner_kwargs={"footprint": footprint})


@main.command("erode")
@io_options
@mask_option
@backend_option(True)
@submit_options
@click.option("--footprint", type=int, default=1)
def cmd_erode(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run, footprint):
    """Binary erosion."""
    dispatch_tool(tool="morph", subcommand="erode", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=_morph_op(erode),
        runner_kwargs={"footprint": footprint})


@main.command("open")
@io_options
@mask_option
@backend_option(True)
@submit_options
@click.option("--footprint", type=int, default=1)
def cmd_open(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run, footprint):
    """Binary opening (erode then dilate)."""
    dispatch_tool(tool="morph", subcommand="open", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=_morph_op(open),
        runner_kwargs={"footprint": footprint})


@main.command("close")
@io_options
@mask_option
@backend_option(True)
@submit_options
@click.option("--footprint", type=int, default=1)
def cmd_close(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run, footprint):
    """Binary closing (dilate then erode)."""
    dispatch_tool(tool="morph", subcommand="close", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=_morph_op(close),
        runner_kwargs={"footprint": footprint})


@main.command("fill-holes")
@io_options
@mask_option
@backend_option(True)
@submit_options
def cmd_fill_holes(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run):
    """Fill interior holes in a binary mask."""
    dispatch_tool(tool="morph", subcommand="fill-holes", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=_morph_op(fill_holes))


@main.command("label-cc")
@io_options
@mask_option
@backend_option(True)
@submit_options
def cmd_label_cc(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run):
    """Label connected components in a binary mask."""

    def runner(image, mask=None):
        """Label connected components in *mask* if given, else in *image*."""
        target = mask if mask is not None else image
        return label_connected(target)
    dispatch_tool(tool="morph", subcommand="label-cc", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=runner)


@main.command("centerline")
@io_options
@mask_option
@backend_option(False)
@submit_options
def cmd_centerline(input_path, output_path, mask_path, backend, submit, emit_script, direct_submit, no_remote, dry_run):
    """Compute a binary centerline/skeleton from a vessel mask."""

    def runner(image, mask=None):
        """Skeletonize *mask* if given, else *image*, preserving :class:`~nvitk.types.Image` metadata."""
        target = mask if mask is not None else image
        result = compute_centerlines(target)
        import numpy as np
        from nvitk.types import Image
        if isinstance(target, Image):
            return target.with_data(np.asarray(result, dtype=target.data.dtype))
        return result
    dispatch_tool(tool="morph", subcommand="centerline", module_file="morphology.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=mask_path, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=runner)


@main.command("siphon-correct")
@io_options
@mask_option
@backend_option(False)
@submit_options
@click.option("--tof", "tof_path", type=click.Path(path_type=Path, exists=True), required=True, help="TOF MRA volume (same grid as mask).")
def cmd_siphon(input_path, output_path, mask_path, tof_path, backend, submit, emit_script, direct_submit, no_remote, dry_run):
    """Correct ICA siphon centerline geometry using a TOF MRA reference volume."""
    from nvitk.io import imread, imsave

    if submit.lower() != "local":
        raise click.ClickException("siphon-correct supports --submit local only in this version.")
    tof = imread(tof_path)
    vessel = imread(mask_path or input_path)
    result = correct_siphon_centerlines(tof, vessel)
    corrected = result.get("vessel_mask_corrected") or result.get("mask")
    if corrected is None:
        raise click.ClickException("siphon-correct did not return vessel_mask_corrected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imsave(output_path, corrected)


if __name__ == "__main__":
    main()
