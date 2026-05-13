"""Command-line entry points for nvitk registration (FSL FLIRT via NiPype).

Requires a working FSL installation (``flirt`` on ``PATH``, ``FSLDIR`` set).
Install the optional extra ``pip install nvitk[fsl]`` if NiPype is not already present.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.registration.fsl.flirt import flirt_apply_rigid, flirt_register_rigid

log = Logger()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Medical image registration helpers (currently FSL FLIRT only)."""


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


if __name__ == "__main__":
    main()
