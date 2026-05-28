"""ANTsPy CLI: `nvitk-ants register|apply`."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger
from nvitk.registration.ants import ants_apply, ants_register

log = Logger()


@click.group("nvitk-ants", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
    show_default=True,
    help="Array backend: cpu (NumPy) or gpu (CuPy).",
)
@click.pass_context
def main(ctx: click.Context, backend: str) -> None:
    """ANTsPy registration helpers."""
    apply_cli_backend(backend)
    ctx.ensure_object(dict)
    ctx.obj["backend"] = backend


@main.command("register")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--type-of-transform",
    "type_of_transform",
    default="SyN",
    show_default=True,
    help="ANTsPy type_of_transform (e.g. Translation, Rigid, Affine, SyN, SyNCC, SyNRA).",
)
@click.option("--write-composite-transform", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def register_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_dir: Path,
    type_of_transform: str,
    write_composite_transform: bool,
    verbose: bool,
) -> None:
    """Register MOVING→FIXED using ANTsPy."""
    res = ants_register(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_dir=out_dir,
        type_of_transform=type_of_transform,
        write_composite_transform=write_composite_transform,
        verbose=verbose,
    )
    log.info(f"ANTs warped: {res.warped_moving_path}")
    log.info(f"ANTs fwd: {list(map(str, res.fwd_transforms))}")


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
    help="Repeat for each transform file; order matters (ANTs transformlist).",
)
@click.option("--interp", "interpolator", default="linear", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def apply_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_path: Path,
    transform_paths: tuple[Path, ...],
    interpolator: str,
    verbose: bool,
) -> None:
    """Apply ANTs transforms to map MOVING into FIXED space."""
    outp = ants_apply(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_path=out_path,
        transforms=list(transform_paths),
        interpolator=interpolator,
        verbose=verbose,
    )
    log.info(f"ANTs output: {outp}")


__all__ = ["main"]

