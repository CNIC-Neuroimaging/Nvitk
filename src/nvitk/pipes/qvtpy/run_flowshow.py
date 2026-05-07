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
from typing import cast

import click

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.io.conversors.phase2volume import discover_phase_inputs
from nvitk.types import Image
from nvitk.viz.flowshow import (
    FlowshowAnimationOptions,
    FlowshowVectorOptions,
    VectorColorMode,
    flowshow,
)

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


def _parse_speed_clim(value: str | None) -> tuple[float, float] | None:
    if not value or not value.strip():
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise click.ClickException("--speed-clim must be two numbers: min,max")
    return (float(parts[0]), float(parts[1]))


def _resolve_nii_optional(folder: Path, stem: str) -> Path | None:
    for name in (f"{stem}.nii.gz", f"{stem}.nii"):
        p = folder / name
        if p.is_file():
            return p
    return None


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
    "--single-label",
    is_flag=True,
    default=False,
    help="Show one vessel label at a time (slider/dropdown) instead of all labels together.",
)
@click.option(
    "--max-glyphs",
    type=int,
    default=50_000,
    show_default=True,
    help="Cap on velocity arrows (shared across labels in multi-label view).",
)
@click.option(
    "--depth-peeling/--no-depth-peeling",
    default=False,
    show_default=True,
    help="Enable VTK depth peeling (nicer transparency; unstable on some GPU drivers).",
)
@click.option(
    "--vector-color",
    type=click.Choice(["label", "speed", "fixed"]),
    default="label",
    show_default=True,
    help="Glyph coloring: by vessel label, by |v| (colormap), or fixed color.",
)
@click.option(
    "--vector-scale-magnitude/--no-vector-scale-magnitude",
    default=True,
    show_default=True,
    help="Scale arrow length using |v| (normalized with --speed-clim or auto percentiles).",
)
@click.option(
    "--vector-scale-factor",
    type=float,
    default=3.0,
    show_default=True,
    help="Glyph length multiplier when --vector-scale-magnitude is set.",
)
@click.option(
    "--vector-fixed-color",
    default="#00A6FB",
    show_default=True,
    help="Arrow color when --vector-color=fixed.",
)
@click.option("--speed-cmap", default="turbo", show_default=True, help="Colormap when --vector-color=speed.")
@click.option(
    "--speed-clim",
    default=None,
    help="Fixed |v| range for speed colors/scale, as min,max (e.g. 0,450). Default: auto from samples.",
)
@click.option(
    "--glyph-opacity",
    type=float,
    default=0.9,
    show_default=True,
)
@click.option(
    "--no-precompute-indices",
    is_flag=True,
    help="Do not cache glyph voxel indices (slower scrubbing; uses less setup memory).",
)
@click.option(
    "--auto-play/--no-auto-play",
    default=False,
    show_default=True,
    help="Start time animation immediately (Space pauses desktop).",
)
@click.option("--animation-fps", type=float, default=8.0, show_default=True)
@click.option(
    "--no-loop-animation",
    is_flag=True,
    help="Stop at last time frame instead of looping (desktop timer).",
)
@click.option("--streamline-radius", type=float, default=None, help="Override streamline tube radius.")
@click.option("--streamline-seeds", type=int, default=None, help="Override number of streamline seeds.")
@click.option(
    "--cross-section/--no-cross-section",
    default=True,
    show_default=True,
    help="Enable oblique cross-section panel (ComplexDifference/Angio/VelMag) on pick.",
)
@click.option("--cross-section-radius-vox", type=float, default=5.0, show_default=True)
@click.option("--cross-section-res", type=int, default=112, show_default=True)
@click.option("--centerline-window", type=click.Choice(["3", "5"]), default="5", show_default=True)
@click.option(
    "--show-gradient/--no-show-gradient",
    default=False,
    show_default=True,
    help="Enable interior velocity field points by default (can also be toggled in-window).",
)
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
    single_label: bool,
    max_glyphs: int,
    depth_peeling: bool,
    vector_color: str,
    vector_scale_magnitude: bool,
    vector_scale_factor: float,
    vector_fixed_color: str,
    speed_cmap: str,
    speed_clim: str | None,
    glyph_opacity: float,
    no_precompute_indices: bool,
    auto_play: bool,
    animation_fps: float,
    no_loop_animation: bool,
    streamline_radius: float | None,
    streamline_seeds: int | None,
    cross_section: bool,
    cross_section_radius_vox: float,
    cross_section_res: int,
    centerline_window: str,
    show_gradient: bool,
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

    cs_vols = None
    if cross_section:
        flow_dir = patient / "4DFlow"
        cs_vols = {}
        for stem in ("ComplexDifference_3D", "Angiography_3D", "VelocityMagnitude_3D"):
            p = _resolve_nii_optional(flow_dir, stem)
            if p is not None:
                cs_vols[stem] = imread(p)

    vec = FlowshowVectorOptions(
        color_mode=cast(VectorColorMode, vector_color),
        fixed_color=vector_fixed_color,
        scale_by_magnitude=vector_scale_magnitude,
        scale_factor=vector_scale_factor,
        speed_cmap=speed_cmap,
        speed_clim=_parse_speed_clim(speed_clim),
        glyph_opacity=glyph_opacity,
    )
    if streamline_radius is not None:
        vec.streamline_radius = streamline_radius
    if streamline_seeds is not None:
        vec.streamline_n_seeds = streamline_seeds

    anim = FlowshowAnimationOptions(
        precompute_glyph_indices=not no_precompute_indices,
        auto_play=auto_play,
        animation_fps=animation_fps,
        loop=not no_loop_animation,
    )

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
            show_all_labels=not single_label,
            max_glyphs=max_glyphs,
            depth_peeling=depth_peeling,
            vector=vec,
            animation=anim,
            cross_section_volumes=cs_vols,
            centerline_window=int(centerline_window),
            cross_section_radius_vox=float(cross_section_radius_vox),
            cross_section_res=int(cross_section_res),
        )
    except ValidationError as exc:
        raise click.ClickException(str(exc)) from exc


__all__ = ["main"]
