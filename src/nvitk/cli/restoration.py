"""``nvitk-restore`` CLI — restoration tools (bilateral, N4 bias correction)."""

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
from nvitk.restoration import (
    bilateral,
    bilateral_2d,
    bilateral_3d,
    estimate_bilateral_parameters,
    n4_bias_field_correction,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Image restoration tools."""


def _bilateral_runner(image, *, sigma_spatial: float | None, sigma_color: float | None, mode: str):
    if sigma_spatial is None or sigma_color is None:
        ss, sc = estimate_bilateral_parameters(image)
        sigma_spatial = sigma_spatial if sigma_spatial is not None else ss
        sigma_color = sigma_color if sigma_color is not None else sc
    if mode == "2d":
        return bilateral_2d(image, sigma_spatial=sigma_spatial, sigma_color=sigma_color)
    if mode == "3d":
        return bilateral_3d(image, sigma_spatial=sigma_spatial, sigma_color=sigma_color)
    return bilateral(image, sigma_spatial=sigma_spatial, sigma_color=sigma_color)


@main.command("bilateral")
@io_options
@backend_option(True)
@submit_options
@click.option("--mode", type=click.Choice(["auto", "2d", "3d"]), default="auto")
@click.option("--sigma-spatial", type=float, default=None)
@click.option("--sigma-color", type=float, default=None)
def cmd_bilateral(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    mode: str,
    sigma_spatial: float | None,
    sigma_color: float | None,
) -> None:
    """Bilateral denoising (CPU skimage or GPU CUDA kernels)."""
    dispatch_tool(
        tool="restore",
        subcommand="bilateral",
        module_file="restoration.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=_bilateral_runner,
        runner_kwargs={
            "sigma_spatial": sigma_spatial,
            "sigma_color": sigma_color,
            "mode": mode,
        },
    )


def _n4_runner(
    image,
    mask=None,
    *,
    shrink_factor: int,
    spline_param: float | None,
    rescale_intensities: bool,
    verbose: bool,
):
    return n4_bias_field_correction(
        image,
        mask=mask,
        shrink_factor=int(shrink_factor),
        spline_param=spline_param,
        rescale_intensities=bool(rescale_intensities),
        verbose=bool(verbose),
    )


@main.command("n4")
@io_options
@mask_option
@backend_option(False)
@submit_options
@click.option("--shrink-factor", type=int, default=4, show_default=True)
@click.option(
    "--spline-param",
    type=float,
    default=None,
    help="B-spline fitting distance in voxels (ANTs default when omitted).",
)
@click.option("--rescale-intensities/--no-rescale-intensities", default=False, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def cmd_n4(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    shrink_factor: int,
    spline_param: float | None,
    rescale_intensities: bool,
    verbose: bool,
) -> None:
    """N4 bias-field correction (ANTsPy)."""
    dispatch_tool(
        tool="restore",
        subcommand="n4",
        module_file="restoration.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=mask_path,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=_n4_runner,
        runner_kwargs={
            "shrink_factor": shrink_factor,
            "spline_param": spline_param,
            "rescale_intensities": rescale_intensities,
            "verbose": verbose,
        },
    )


if __name__ == "__main__":
    main()
