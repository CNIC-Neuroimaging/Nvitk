"""qvtpy 4DFlow viewer CLI.

Loads AP/RL/FH phase NIfTI volumes from the stage-0 NIfTI layout produced by
:mod:`nvitk.pipes.qvtpy.stage0_convert` — namely ``<nifti_root>/<subject>/``
with a ``4DFlow/`` folder containing direction subfolders (AP, RL, FH) — using
:func:`nvitk.io.conversors.phase2volume.discover_phase_inputs`.

It then opens :func:`nvitk.viz.flowshow.flowshow` with a multilabel vessel mask
(3D, same spatial shape as the phase volumes).

Masks
-----
Provide ``--vessel-mask`` as an absolute path or as a path relative to the
subject folder. If omitted, the CLI looks for, in order:
``vessels.nii.gz``, ``vessel_mask.nii.gz``, ``VesselSeg.nii.gz``, ``vessels.nii``
under the subject directory.

Optional ``--batch`` uses ``<nifti_root>/<batch>/<subject>`` (when data are
nested by batch). Otherwise the subject folder is ``<nifti_root>/<subject>``,
matching stage0.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.io.conversors.phase2volume import discover_phase_inputs
from nvitk.types import Image
from nvitk.viz.flowshow import flowshow

from . import config as cfg

log = Logger()

_DEFAULT_VESSEL_REL_NAMES: tuple[str, ...] = (
    "vessels.nii.gz",
    "vessel_mask.nii.gz",
    "VesselSeg.nii.gz",
    "vessels.nii",
)


def _patient_dir(
    *,
    nifti_root: Path,
    subject: str | None,
    batch: str | None,
    patient_dir: Path | None,
) -> Path:
    if patient_dir is not None:
        p = patient_dir.expanduser().resolve()
        if not p.is_dir():
            raise click.ClickException(f"--patient-dir is not a directory: {p}")
        return p
    if not subject:
        raise click.ClickException("Pass --subject or --patient-dir.")
    root = nifti_root.expanduser().resolve()
    if batch:
        p = (root / batch / subject).resolve()
    else:
        p = (root / subject).resolve()
    if not p.is_dir():
        raise click.ClickException(f"Subject NIfTI folder not found: {p}")
    return p


def _resolve_vessel_mask(patient: Path, vessel_mask: Path | None) -> Path:
    if vessel_mask is None:
        for name in _DEFAULT_VESSEL_REL_NAMES:
            cand = patient / name
            if cand.is_file():
                log.info(f"Using default vessel mask: {cand.name}")
                return cand
        raise click.ClickException(
            f"No vessel mask found under {patient}. "
            f"Tried: {', '.join(_DEFAULT_VESSEL_REL_NAMES)}. "
            "Pass --vessel-mask explicitly."
        )
    vm = Path(vessel_mask)
    if not vm.is_absolute():
        vm = (patient / vm).resolve()
    if not vm.is_file():
        raise click.ClickException(f"Vessel mask not found: {vm}")
    return vm


def _resolve_optional_mask(patient: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (patient / p).resolve()
    if not p.is_file():
        raise click.ClickException(f"Centerline mask not found: {p}")
    return p


def _mask_to_xyz_3d(mask_img: Image, ref_spatial: tuple[int, int, int]) -> Image:
    """Ensure 3D label mask with spatial shape *ref_spatial* (from phase data)."""
    arr = to_numpy(mask_img.data)
    if arr.ndim == 4:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValidationError(f"vessel_mask must be 3D (or 4D with time); got shape {arr.shape}.")
    if tuple(arr.shape) != ref_spatial:
        raise ValidationError(
            f"vessel_mask spatial shape {tuple(arr.shape)} does not match phase volumes "
            f"{ref_spatial}. Resample the mask to the phase grid or check axis order."
        )
    return mask_img.with_data(arr)


def _spatial_xyz_from_phase(ap: Image) -> tuple[int, int, int]:
    sh = tuple(int(x) for x in ap.data.shape[:3])
    return sh


@click.command("nvitk-qvtpy-flowshow")
@click.option("--subject", default=None, help="Subject id (NIfTI folder name under --nifti-root).")
@click.option(
    "--batch",
    default=None,
    help="Optional batch folder between nifti-root and subject: <root>/<batch>/<subject>.",
)
@click.option(
    "--patient-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Full path to subject NIfTI folder (overrides --subject / --batch / --nifti-root).",
)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_NIFTI_ROOT)
@click.option(
    "--vessel-mask",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Multilabel vessel mask (NIfTI). Relative paths are resolved under the subject folder.",
)
@click.option(
    "--centerline-mask",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional centerline mask NIfTI (passed through to flowshow; reserved for future use).",
)
@click.option("--stride", type=int, default=4, show_default=True, help="Stride for velocity glyph subsampling.")
@click.option("--timepoint", type=int, default=0, show_default=True, help="Initial time index.")
@click.option(
    "--notebook/--no-notebook",
    default=False,
    show_default=True,
    help="Use Jupyter ipywidgets UI instead of a desktop PyVista window.",
)
@click.option(
    "--no-show",
    is_flag=True,
    default=False,
    help="Build the viewer but do not open a window (smoke test / headless).",
)
@click.option(
    "--list-inputs",
    is_flag=True,
    default=False,
    help="Print discovered phase NIfTI paths and exit.",
)
def main(
    subject: str | None,
    batch: str | None,
    patient_dir: Path | None,
    nifti_root: Path,
    vessel_mask: Path | None,
    centerline_mask: Path | None,
    stride: int,
    timepoint: int,
    notebook: bool,
    no_show: bool,
    list_inputs: bool,
) -> None:
    patient = _patient_dir(
        nifti_root=Path(nifti_root),
        subject=subject,
        batch=batch,
        patient_dir=patient_dir,
    )

    try:
        inputs = discover_phase_inputs(patient)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if list_inputs:
        click.echo(f"patient_dir={patient}")
        click.echo(f"  ap_phase={inputs.ap_phase_path}")
        click.echo(f"  rl_phase={inputs.rl_phase_path}")
        click.echo(f"  fh_phase={inputs.fh_phase_path}")
        click.echo(f"  angio_magnitude={inputs.angio_path}")
        return

    ap = imread(inputs.ap_phase_path)
    rl = imread(inputs.rl_phase_path)
    fh = imread(inputs.fh_phase_path)
    spatial = _spatial_xyz_from_phase(ap)

    vm_path = _resolve_vessel_mask(patient, vessel_mask)
    mask_img = imread(vm_path)
    mask_img = _mask_to_xyz_3d(mask_img, spatial)

    cl_path = _resolve_optional_mask(patient, centerline_mask)
    centerline_img = imread(cl_path) if cl_path is not None else None

    try:
        flowshow(
            ap,
            rl,
            fh,
            mask_img,
            centerline_mask=centerline_img,
            stride=stride,
            timepoint=timepoint,
            notebook=notebook,
            show=not no_show,
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


__all__ = ["main"]
