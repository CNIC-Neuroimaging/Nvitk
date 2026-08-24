"""``nvitk-transform`` CLI — geometric transforms."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.cli._common import backend_option, dispatch_tool, io_options, submit_options
from nvitk.io import imread
from nvitk.transform import isotropy, oblique_slice, resample_to, rotate_volume
from nvitk.core.click_config import config_dir_click_option


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
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
    """Resample a volume onto the grid of a reference image."""

    def runner(image, *, reference_path, order):
        """Resample *image* onto the grid of the volume at *reference_path*."""
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
    """Resample a volume to isotropic voxel spacing."""
    dispatch_tool(
        tool="transform", subcommand="isotropy", module_file="transform.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=None, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=isotropy,
    )


@main.command("rotate")
@io_options
@backend_option(True)
@submit_options
@click.option("--angle", type=float, required=True, help="Counter-clockwise rotation in degrees.")
@click.option("--axis", type=int, default=2, show_default=True, help="Axis to rotate around (0/1/2).")
@click.option("--order", type=int, default=1, show_default=True, help="Interpolation order (0 for labels).")
@click.option("--reshape/--no-reshape", default=False, show_default=True)
def cmd_rotate(
    input_path, output_path, backend, submit, emit_script, direct_submit, no_remote, dry_run,
    angle, axis, order, reshape,
):
    """Rotate a 2D/3D volume around a spatial axis."""
    def runner(image, *, angle, axis, order, reshape):
        """Rotate *image* by *angle* degrees around *axis* with the given interpolation order."""
        return rotate_volume(
            image, angle, axis=axis, order=order, reshape=reshape
        )
    dispatch_tool(
        tool="transform", subcommand="rotate", module_file="transform.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=None, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=runner,
        runner_kwargs={
            "angle": float(angle),
            "axis": int(axis),
            "order": int(order),
            "reshape": bool(reshape),
        },
    )


@main.command("swap-axes")
@io_options
@backend_option(True)
@submit_options
@click.option("--axis0", type=int, default=0, show_default=True, help="First axis to swap.")
@click.option("--axis1", type=int, default=1, show_default=True, help="Second axis to swap.")
@click.option(
    "--order",
    "perm_order",
    default=None,
    help="Full axis permutation instead of a pairwise swap (e.g. 2,1,0).",
)
def cmd_swap_axes(
    input_path, output_path, backend, submit, emit_script, direct_submit, no_remote, dry_run,
    axis0, axis1, perm_order,
):
    """Swap two axes, or apply a full axis permutation with --order."""
    from nvitk.transform import permute_axes, swap_axes

    def runner(image, *, axis0, axis1, perm_order):
        """Apply a full axis permutation if *perm_order* is given, else swap *axis0*/*axis1* pairwise."""
        if perm_order:
            parts = [int(p.strip()) for p in str(perm_order).replace(";", ",").split(",") if p.strip()]
            return permute_axes(image, parts)
        return swap_axes(image, int(axis0), int(axis1))

    dispatch_tool(
        tool="transform", subcommand="swap-axes", module_file="transform.py",
        input_path=input_path, output_path=output_path, submit=submit, backend=backend,
        mask_path=None, emit_script=emit_script, direct_submit=direct_submit,
        no_remote=no_remote, dry_run=dry_run, runner=runner,
        runner_kwargs={"axis0": int(axis0), "axis1": int(axis1), "perm_order": perm_order},
    )


@main.command("oblique-slice")
@io_options
@backend_option(True)
@submit_options
@click.option("--point", nargs=3, type=float, required=True)
@click.option("--normal", nargs=3, type=float, required=True)
def cmd_oblique(input_path, output_path, backend, submit, emit_script, direct_submit, no_remote, dry_run, point, normal):
    """Sample an oblique 2D slice through a volume defined by a point and normal vector."""

    def runner(image, *, point, normal):
        """Sample the oblique slice of *image* through *point* with orientation *normal*."""
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
