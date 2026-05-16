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

If none of those exist, the CLI then looks for QVTPy pipeline outputs under
``<pipeline-output-root>/<subject>/qvtpy/stage4_4dflow_segmentation/seg_4dflow.nii.gz``.
When ``--pipeline-output-root`` is omitted, :data:`nvitk.pipes.qvtpy.config.DEFAULT_RESULTS_ROOT`
is tried (same layout as ``nvitk-qvtpy`` / stage4). From that tree it can also
load ``locs.csv`` and ``centerlines_mask.nii.gz`` for the viewer.

Optional ``--batch`` uses ``<nifti_root>/<batch>/<subject>`` (when data are
nested by batch). Otherwise the subject folder is ``<nifti_root>/<subject>``,
matching stage0.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import click
import numpy as np

from nvitk.core.backend import using
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

# ---------------------------------------------------------------------------
# Default vessel mask search names
# ---------------------------------------------------------------------------

_DEFAULT_VESSEL_REL_NAMES: tuple[str, ...] = (
    "vessels.nii.gz",
    "vessel_mask.nii.gz",
    "VesselSeg.nii.gz",
    "vessels.nii",
)


# ---------------------------------------------------------------------------
# Subject / mask path resolution
# ---------------------------------------------------------------------------


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


def _find_nifti_default_vessel_mask(patient: Path) -> Path | None:
    for name in _DEFAULT_VESSEL_REL_NAMES:
        cand = patient / name
        if cand.is_file():
            return cand
    return None


def _resolve_vessel_mask(patient: Path, vessel_mask: Path | None) -> Path:
    if vessel_mask is None:
        hit = _find_nifti_default_vessel_mask(patient)
        if hit is not None:
            log.info(f"Using default vessel mask: {hit.name}")
            return hit
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


def _qvtpy_subject_dir(results_root: Path, sub_key: str) -> Path:
    return results_root / sub_key / cfg.QVT_SUBDIR


def _seg_4dflow_path(qvt_dir: Path) -> Path:
    return qvt_dir / cfg.STAGE4_SEG_DIR / "seg_4dflow.nii.gz"


def _resolve_vessel_mask_and_results_root(
    patient: Path,
    sub_key: str,
    vessel_mask: Path | None,
    pipeline_output_root: Path | None,
) -> tuple[Path, Path | None]:
    """Return (vessel_mask_path, pipeline_results_root_for_sidecars).

    *pipeline_results_root_for_sidecars* is the directory that contains
    ``<subject>/qvtpy/`` (e.g. ``.../RESULTS/res_QVTPy``). It is set when
    ``--pipeline-output-root`` is passed, or when a segmentation was taken from
    the default results root, or when only ``--pipeline-output-root`` is needed
    to load LOCs while the mask comes from NIfTI.
    """
    explicit_root = (
        Path(pipeline_output_root).expanduser().resolve() if pipeline_output_root is not None else None
    )

    if vessel_mask is not None:
        return _resolve_vessel_mask(patient, vessel_mask), explicit_root

    nifti_hit = _find_nifti_default_vessel_mask(patient)
    if nifti_hit is not None:
        log.info(f"Using default vessel mask: {nifti_hit}")
        return nifti_hit, explicit_root

    if explicit_root is not None:
        seg = _seg_4dflow_path(_qvtpy_subject_dir(explicit_root, sub_key))
        if not seg.is_file():
            raise click.ClickException(
                f"--pipeline-output-root: missing segmentation for subject {sub_key!r}: {seg}"
            )
        log.info(f"Using pipeline segmentation as vessel mask: {seg}")
        return seg, explicit_root

    default_root = Path(cfg.DEFAULT_RESULTS_ROOT).expanduser().resolve()
    seg = _seg_4dflow_path(_qvtpy_subject_dir(default_root, sub_key))
    if seg.is_file():
        log.info(f"Using pipeline segmentation as vessel mask: {seg}")
        return seg, default_root

    tried = ", ".join(_DEFAULT_VESSEL_REL_NAMES)
    raise click.ClickException(
        f"No vessel mask under {patient} (tried: {tried}).\n"
        f"No seg_4dflow.nii.gz for subject {sub_key!r} under default pipeline results:\n  {seg}\n"
        f"(config DEFAULT_RESULTS_ROOT = {default_root}).\n"
        "Pass --vessel-mask or --pipeline-output-root, or run the qvtpy pipeline through stage 4."
    )


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


# ---------------------------------------------------------------------------
# CLI (flowshow viewer)
# ---------------------------------------------------------------------------


@click.command("nvitk-qvtpy-flowshow")
@click.option(
    "--pipeline-output-root",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "QVTPy pipeline *results* root (directory that contains ``<subject>/qvtpy/``), "
        "e.g. the parent of your ``res_QVTPy`` folder. When set, loads stage5 locs.csv "
        "and stage3 centerlines_mask if present; if --vessel-mask is omitted and there is "
        "no mask under the NIfTI subject folder, uses stage4 seg_4dflow.nii.gz from this tree. "
        "When omitted, the same is attempted using nvitk.pipes.qvtpy.config.DEFAULT_RESULTS_ROOT."
    ),
)
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
@click.option("--dt-seconds", type=float, default=None, help="Temporal resolution between frames (seconds) for pathlines.")
@click.option(
    "--cross-section/--no-cross-section",
    default=True,
    show_default=True,
    help="Enable oblique cross-section panel (ComplexDifference/Angio/VelMag) on pick.",
)
@click.option("--cross-section-radius-vox", type=float, default=5.0, show_default=True)
@click.option(
    "--cross-section-res",
    type=int,
    default=0,
    show_default=True,
    help="Optional max oblique slice edge length (pixels); 0 = auto from --cross-section-radius-vox (~1 voxel/sample).",
)
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
    pipeline_output_root: Path | None,
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
    dt_seconds: float | None,
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
    sub_key = subject or patient.name

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

    log.info(f"Loading phase volumes from:")
    log.info(f"  ap_phase={inputs.ap_phase_path}")
    log.info(f"  rl_phase={inputs.rl_phase_path}")
    log.info(f"  fh_phase={inputs.fh_phase_path}")
    log.info(f"  angio_magnitude={inputs.angio_path}")
    ap = imread(inputs.ap_phase_path)
    rl = imread(inputs.rl_phase_path)
    fh = imread(inputs.fh_phase_path)
    spatial = _spatial_xyz_from_phase(ap)

    voxel_sp_tuple: tuple[float, float, float] | None = None
    sp = ap.spacing
    if sp is not None and len(sp) >= 3:
        voxel_sp_tuple = (float(sp[0]), float(sp[1]), float(sp[2]))
    elif ap.affine is not None:
        a = np.asarray(ap.affine, dtype=np.float64)
        voxel_sp_tuple = (
            float(np.linalg.norm(a[:3, 0])),
            float(np.linalg.norm(a[:3, 1])),
            float(np.linalg.norm(a[:3, 2])),
        )

    vm_path, pipeline_sidecar_root = _resolve_vessel_mask_and_results_root(
        patient, sub_key, vessel_mask, pipeline_output_root
    )

    loc_rows: list[dict[str, str]] | None = None
    single_label_eff = single_label
    centerline_mask_eff: Path | None = centerline_mask

    if pipeline_sidecar_root is not None:
        qd = _qvtpy_subject_dir(pipeline_sidecar_root, sub_key)
        loc_p = qd / cfg.STAGE5_LOC_DIR / "locs.csv"
        if loc_p.is_file():
            with loc_p.open(newline="", encoding="utf-8") as file_handler:
                loc_rows = list(csv.DictReader(file_handler))
            log.info(f"Loaded {len(loc_rows)} LOC row(s) from {loc_p}")
        cl_stage3 = qd / cfg.STAGE3_CENTERLINE_DIR / "centerlines_mask.nii.gz"
        if centerline_mask_eff is None and cl_stage3.is_file():
            centerline_mask_eff = cl_stage3
            log.info(f"Using pipeline stage3 centerline mask: {cl_stage3}")

    mask_img = imread(vm_path)
    mask_img = _mask_to_xyz_3d(mask_img, spatial)

    cl_path = _resolve_optional_mask(patient, centerline_mask_eff)
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
        with using("cpu"):
            flowshow(
                to_numpy(ap),
                to_numpy(rl),
                to_numpy(fh),
                to_numpy(mask_img),
                centerline_mask=to_numpy(centerline_img),
                stride=stride,
                timepoint=timepoint,
                notebook=notebook,
                show=not no_show,
                show_all_labels=not single_label_eff,
                max_glyphs=max_glyphs,
                depth_peeling=depth_peeling,
                vector=vec,
                animation=anim,
                dt_seconds=dt_seconds,
                cross_section_volumes=cs_vols,
                centerline_window=int(centerline_window),
                cross_section_radius_vox=float(cross_section_radius_vox),
                cross_section_res=int(cross_section_res),
                show_gradient=show_gradient,
                loc_records=loc_rows,
                voxel_spacing_mm=voxel_sp_tuple,
            )
    except ValidationError as exc:
        import traceback
        log.exception(traceback.format_exc())
        raise click.ClickException(str(exc)) from exc


__all__ = ["main"]
