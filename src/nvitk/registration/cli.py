"""Command-line entry points for nvitk registration (FSL FLIRT + optional ANTs/FireANTs).

Requires a working FSL installation (``flirt`` on ``PATH``, ``FSLDIR`` set).
Install the optional extra ``pip install nvitk[fsl]`` if NiPype is not already present.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger
from nvitk.registration.fsl.flirt import flirt_apply_rigid, flirt_register_rigid
from nvitk.registration.ants import ANTSPY_TYPE_OF_TRANSFORM, ants_apply, ants_register
from nvitk.registration.fireants import fireants_apply, fireants_register

log = Logger()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
    show_default=True,
    help="Array backend: cpu (NumPy) or gpu (CuPy).",
)
@click.pass_context
def main(ctx: click.Context, backend: str) -> None:
    """Medical image registration helpers (FLIRT + optional ANTs/FireANTs)."""
    apply_cli_backend(backend)
    ctx.ensure_object(dict)
    ctx.obj["backend"] = backend


@main.command("register-rigid")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Output directory (created if missing).")
@click.option("--dof", type=int, default=6, show_default=True, help="FLIRT degrees of freedom (6 = rigid).")
@click.option(
    "--cost",
    type=str,
    default="corratio",
    show_default=True,
    help="FLIRT cost / cost_fun (e.g. corratio, normmi, mutualinfo).",
)
@click.option(
    "--warped-name",
    default="moving_warped.nii.gz",
    show_default=True,
    help="Filename for the warped moving image under --out-dir.",
)
@click.option(
    "--matrix-name",
    default="affine.mat",
    show_default=True,
    help="Filename for the FLIRT transform matrix under --out-dir.",
)
@click.option(
    "--searchr-x",
    type=float,
    default=None,
    help="Optional FLIRT search range in x (degrees); sets searchr_x symmetric if supported.",
)
def register_rigid_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_dir: Path,
    dof: int,
    cost: str,
    warped_name: str,
    matrix_name: str,
    searchr_x: float | None,
) -> None:
    """Run rigid FLIRT: align MOVING to FIXED, write affine matrix and warped image."""
    res = flirt_register_rigid(
        moving_path,
        fixed_path,
        out_dir,
        dof=dof,
        cost=cost,
        warped_name=warped_name,
        matrix_name=matrix_name,
        searchr_x=searchr_x,
    )
    log.info(f"FLIRT matrix: {res.matrix_path}")
    if res.warped_path is not None:
        log.info(f"FLIRT warped: {res.warped_path}")


@main.command("apply-rigid")
@click.option("--in", "in_path", type=click.Path(path_type=Path, exists=True), required=True, help="Input image to resample.")
@click.option("--ref", "ref_path", type=click.Path(path_type=Path, exists=True), required=True, help="Reference space (target grid).")
@click.option("--mat", "mat_path", type=click.Path(path_type=Path, exists=True), required=True, help="FLIRT ``*.mat`` from register-rigid.")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True, help="Output NIfTI path.")
@click.option(
    "--interp",
    type=str,
    default="trilinear",
    show_default=True,
    help="Interpolation (e.g. trilinear, nearestneighbour).",
)
def apply_rigid_cmd(in_path: Path, ref_path: Path, mat_path: Path, out_path: Path, interp: str) -> None:
    """Apply an existing FLIRT rigid transform (``-applyxfm``)."""
    outp = flirt_apply_rigid(in_path, ref_path, mat_path, out_path, interp=interp)
    log.info(f"FLIRT output: {outp}")


@main.command("ants-register")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--type-of-transform",
    "type_of_transform",
    default="SyN",
    show_default=True,
    help="ANTsPy type_of_transform. Common: Translation, Rigid, Affine, SyN, SyNCC, SyNRA.",
)
@click.option("--write-composite-transform", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def ants_register_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_dir: Path,
    type_of_transform: str,
    write_composite_transform: bool,
    verbose: bool,
) -> None:
    """Register MOVING→FIXED using ANTsPy (ants.registration)."""
    _ = ANTSPY_TYPE_OF_TRANSFORM  # surfaced in module docs; CLI accepts any string ANTsPy supports
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


@main.command("ants-apply")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--transform",
    "transform_paths",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Repeat for each transform file; order matters (same as ants.apply_transforms transformlist).",
)
@click.option("--interp", "interpolator", default="linear", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def ants_apply_cmd(
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


@main.command("fireants-register")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def fireants_register_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_dir: Path,
    device: str,
    verbose: bool,
) -> None:
    """Register MOVING→FIXED using FireANTs (GPU; wraps fireantsRegistration)."""
    res = fireants_register(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_dir=out_dir,
        device=device,
        verbose=verbose,
    )
    log.info(f"FireANTs warped: {res.warped_moving_path}")
    log.info(f"FireANTs prefix: {res.output_prefix}")


@main.command("fireants-apply")
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
def fireants_apply_cmd(
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


if __name__ == "__main__":
    main()
