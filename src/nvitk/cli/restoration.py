"""``nvitk-restore`` CLI — restoration tools (bilateral filter)."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.cli._common import backend_option, dispatch_tool, io_options, submit_options
from nvitk.restoration import bilateral, bilateral_2d, bilateral_3d, estimate_bilateral_parameters


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


if __name__ == "__main__":
    main()
