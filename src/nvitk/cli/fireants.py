"""FireANTs CLI: `nvitk-fireants register|apply`."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger
from nvitk.registration.fireants import fireants_apply, fireants_register
from nvitk.core.click_config import config_dir_click_option

log = Logger()


@click.group("nvitk-fireants", context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
    show_default=True,
    help="Array backend: cpu (NumPy) or gpu (CuPy).",
)
@click.pass_context
def main(ctx: click.Context, backend: str) -> None:
    """FireANTs (GPU) registration helpers."""
    apply_cli_backend(backend)
    ctx.ensure_object(dict)
    ctx.obj["backend"] = backend


@main.command("register")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def register_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_dir: Path,
    device: str,
    verbose: bool,
) -> None:
    """Register MOVING→FIXED using FireANTs (wraps fireantsRegistration)."""
    res = fireants_register(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_dir=out_dir,
        device=device,
        verbose=verbose,
    )
    log.info(f"FireANTs warped: {res.warped_moving_path}")
    log.info(f"FireANTs prefix: {res.output_prefix}")


@main.command("apply")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--transform",
    "transform_paths",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
def apply_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_path: Path,
    transform_paths: tuple[Path, ...],
) -> None:
    """Apply FireANTs transforms (requires fireantsApplyTransforms)."""
    outp = fireants_apply(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_path=out_path,
        transforms=list(transform_paths),
    )
    log.info(f"FireANTs output: {outp}")


__all__ = ["main"]

