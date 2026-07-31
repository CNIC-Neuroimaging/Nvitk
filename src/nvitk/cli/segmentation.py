"""``nvitk-seg`` CLI — segmentation tools (ANTsPyNet + blood flood)."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from nvitk.cli._common import (
    backend_option,
    dispatch_tool,
    io_options,
    mask_option,
    submit_options,
)
from nvitk.core.array import to_numpy
from nvitk.core.backend import using
from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.segmentation.blood_flood import (
    FRANGI_SIGMAS_DEFAULT,
    HYST_HIGH_FACTOR_DEFAULT,
    HYST_LOW_FACTOR_DEFAULT,
    MIN_TREE_CC_VOXELS_DEFAULT,
    TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    blood_flood,
    blood_flood_from_scratch,
)
from nvitk.segmentation.brain_extraction import (
    BRAIN_EXTRACTION_MODALITIES,
    brain_extraction,
)
from nvitk.segmentation.dkt import desikan_killiany_tourville_labeling
from nvitk.segmentation.mouse_brain import (
    MOUSE_EXTRACTION_MODALITIES,
    mouse_brain_segmentation,
)
from nvitk.segmentation.mra_vessel import mra_vessel_segmentation

log = Logger()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Segmentation tools."""


def _mouse_brain_runner(
    image,
    mask=None,
    *,
    mode: str,
    modality: str,
    which_parcellation: str,
    do_n4: bool,
    binarize: bool,
    return_isotropic_output: bool,
    fix_spacing: bool,
    verbose: bool,
):
    return mouse_brain_segmentation(
        image,
        mode=mode,  # type: ignore[arg-type]
        modality=modality,
        which_parcellation=which_parcellation,
        mask=mask,
        do_n4=bool(do_n4),
        binarize=bool(binarize),
        return_isotropic_output=bool(return_isotropic_output),
        fix_spacing=bool(fix_spacing),
        verbose=bool(verbose),
    )


@main.command("mouse-brain")
@io_options
@mask_option
@backend_option(False)
@submit_options
@click.option(
    "--mode",
    type=click.Choice(["extraction", "parcellation"], case_sensitive=False),
    default="extraction",
    show_default=True,
    help="Brain extraction (mask) or regional parcellation.",
)
@click.option(
    "--modality",
    type=click.Choice(list(MOUSE_EXTRACTION_MODALITIES), case_sensitive=False),
    default="t2",
    show_default=True,
    help="Imaging contrast for extraction (no T1 model in ANTsPyNet).",
)
@click.option(
    "--which-parcellation",
    type=click.Choice(["nick", "tct", "jay"], case_sensitive=False),
    default="nick",
    show_default=True,
    help="Parcellation scheme (ANTsPyNet).",
)
@click.option("--n4/--no-n4", "do_n4", default=True, show_default=True, help="N4 bias correction first.")
@click.option(
    "--fix-spacing/--no-fix-spacing",
    default=True,
    show_default=True,
    help="If spacing looks like unit voxels, rescale FOV to ~20 mm (mouse template).",
)
@click.option(
    "--binarize/--probabilities",
    default=True,
    show_default=True,
    help="Threshold extraction probability map to a binary mask.",
)
@click.option("--isotropic-output", "return_isotropic_output", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def cmd_mouse_brain(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    mode: str,
    modality: str,
    which_parcellation: str,
    do_n4: bool,
    fix_spacing: bool,
    binarize: bool,
    return_isotropic_output: bool,
    verbose: bool,
) -> None:
    """Mouse brain extraction / parcellation (ANTsPyNet)."""
    dispatch_tool(
        tool="seg",
        subcommand="mouse-brain",
        module_file="segmentation.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=mask_path,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=_mouse_brain_runner,
        runner_kwargs={
            "mode": mode.lower(),
            "modality": modality.lower(),
            "which_parcellation": which_parcellation.lower(),
            "do_n4": do_n4,
            "binarize": binarize,
            "return_isotropic_output": return_isotropic_output,
            "fix_spacing": fix_spacing,
            "verbose": verbose,
        },
    )


def _brain_extraction_runner(
    image,
    mask=None,
    *,
    modality: str,
    image2_path: str | None,
    verbose: bool,
):
    _ = mask
    if image2_path:
        image2 = imread(image2_path)
        return brain_extraction([image, image2], modality=modality, verbose=verbose)
    return brain_extraction(image, modality=modality, verbose=verbose)


@main.command("brain-extraction")
@io_options
@backend_option(False)
@submit_options
@click.option(
    "--modality",
    type=click.Choice(list(BRAIN_EXTRACTION_MODALITIES), case_sensitive=False),
    default="t1",
    show_default=True,
    help="ANTsPyNet brain_extraction modality.",
)
@click.option(
    "--image2",
    "image2_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional second modality volume (e.g. T2 for t1t2infant).",
)
@click.option("--verbose", is_flag=True, default=False)
def cmd_brain_extraction(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    modality: str,
    image2_path: Path | None,
    verbose: bool,
) -> None:
    """Multi-modal brain extraction (ANTsPyNet)."""
    if image2_path is not None and submit.lower() != "local":
        raise click.ClickException("--image2 currently requires --submit local.")
    dispatch_tool(
        tool="seg",
        subcommand="brain-extraction",
        module_file="segmentation.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=_brain_extraction_runner,
        runner_kwargs={
            "modality": modality.lower(),
            "image2_path": str(image2_path) if image2_path else None,
            "verbose": verbose,
        },
    )


def _mra_runner(
    image,
    mask=None,
    *,
    prediction_batch_size: int,
    patch_stride_length: int,
    verbose: bool,
):
    return mra_vessel_segmentation(
        image,
        mask=mask,
        prediction_batch_size=prediction_batch_size,
        patch_stride_length=patch_stride_length,
        verbose=verbose,
    )


@main.command("mra-vessel")
@io_options
@mask_option
@backend_option(False)
@submit_options
@click.option("--prediction-batch-size", type=int, default=2, show_default=True)
@click.option("--patch-stride-length", type=int, default=32, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def cmd_mra_vessel(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    prediction_batch_size: int,
    patch_stride_length: int,
    verbose: bool,
) -> None:
    """MRA-TOF vessel segmentation (ANTsPyNet probability map)."""
    dispatch_tool(
        tool="seg",
        subcommand="mra-vessel",
        module_file="segmentation.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=mask_path,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=_mra_runner,
        runner_kwargs={
            "prediction_batch_size": prediction_batch_size,
            "patch_stride_length": patch_stride_length,
            "verbose": verbose,
        },
    )


def _dkt_runner(
    image,
    mask=None,
    *,
    do_preprocessing: bool,
    do_lobar_parcellation: bool,
    do_denoising: bool,
    version: int,
    verbose: bool,
):
    _ = mask
    return desikan_killiany_tourville_labeling(
        image,
        do_preprocessing=do_preprocessing,
        do_lobar_parcellation=do_lobar_parcellation,
        do_denoising=do_denoising,
        version=version,
        verbose=verbose,
    )


@main.command("dkt")
@io_options
@backend_option(False)
@submit_options
@click.option("--preprocessing/--no-preprocessing", default=True, show_default=True)
@click.option("--lobar/--no-lobar", "do_lobar_parcellation", default=False, show_default=True)
@click.option("--denoising/--no-denoising", default=True, show_default=True)
@click.option("--version", type=int, default=0, show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def cmd_dkt(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    preprocessing: bool,
    do_lobar_parcellation: bool,
    denoising: bool,
    version: int,
    verbose: bool,
) -> None:
    """Desikan-Killiany-Tourville cortical parcellation (ANTsPyNet)."""
    dispatch_tool(
        tool="seg",
        subcommand="dkt",
        module_file="segmentation.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=_dkt_runner,
        runner_kwargs={
            "do_preprocessing": preprocessing,
            "do_lobar_parcellation": do_lobar_parcellation,
            "do_denoising": denoising,
            "version": version,
            "verbose": verbose,
        },
    )


def _parse_sigmas(text: str | None) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        return tuple(FRANGI_SIGMAS_DEFAULT)
    return tuple(float(x) for x in str(text).replace(";", ",").split(",") if x.strip())


@main.command("blood-flood")
@io_options
@backend_option(True)
@submit_options
@click.option(
    "--mode",
    type=click.Choice(["expand", "from-scratch"], case_sensitive=False),
    default="expand",
    show_default=True,
    help="expand: markers + intensity; from-scratch: intensity only → CC labels.",
)
@click.option(
    "--markers",
    "markers_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Seed / marker labels (required for expand; 0 = background).",
)
@click.option(
    "--mask",
    "mask_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional ROI / brain mask (from-scratch).",
)
@click.option(
    "--barrier",
    "barrier_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional hard barrier mask (removed from the vessel tree).",
)
@click.option(
    "--frangi-sigmas",
    default=None,
    help=f"Comma-separated Frangi scales (default: {','.join(str(s) for s in FRANGI_SIGMAS_DEFAULT)}).",
)
@click.option("--hyst-low-factor", type=float, default=HYST_LOW_FACTOR_DEFAULT, show_default=True)
@click.option("--hyst-high-factor", type=float, default=HYST_HIGH_FACTOR_DEFAULT, show_default=True)
@click.option("--thicken-iter", type=int, default=0, show_default=True)
@click.option(
    "--thin-vesselness-percentile",
    type=float,
    default=TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    show_default=True,
    help="Keep tree voxels above this vesselness percentile; <0 disables thinning.",
)
@click.option(
    "--min-cc-voxels",
    type=int,
    default=MIN_TREE_CC_VOXELS_DEFAULT,
    show_default=True,
    help="Drop tree connected components smaller than this (from-scratch).",
)
@click.option("--connectivity", type=int, default=3, show_default=True)
@click.option(
    "--save-tree",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to write the binary vessel tree.",
)
def cmd_blood_flood(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    mode: str,
    markers_path: Path | None,
    mask_path: Path | None,
    barrier_path: Path | None,
    frangi_sigmas: str | None,
    hyst_low_factor: float,
    hyst_high_factor: float,
    thicken_iter: int,
    thin_vesselness_percentile: float,
    min_cc_voxels: int,
    connectivity: int,
    save_tree: Path | None,
) -> None:
    """Frangi → hysteresis vessel tree; expand from markers or segment from scratch."""
    if submit.lower() != "local":
        raise click.ClickException(
            "blood-flood SGE submit is not wired yet; use --submit local."
        )
    mode_key = mode.lower().replace("_", "-")
    if mode_key == "expand" and markers_path is None:
        raise click.ClickException("--markers is required for --mode expand.")
    apply_cli_backend(backend)
    bk = "cupy" if backend.lower() in ("gpu", "cupy") else "numpy"
    thin = (
        None
        if float(thin_vesselness_percentile) < 0
        else float(thin_vesselness_percentile)
    )
    common_kw = {
        "frangi_sigmas": _parse_sigmas(frangi_sigmas),
        "hyst_low_factor": float(hyst_low_factor),
        "hyst_high_factor": float(hyst_high_factor),
        "thicken_iter": int(thicken_iter),
        "thin_vesselness_percentile": thin,
        "connectivity": int(connectivity),
    }
    with using(bk):
        intensity = imread(input_path, backend=bk)
        barrier = imread(barrier_path, backend=bk) if barrier_path else None
        barrier_np = to_numpy(barrier.data) if barrier is not None else None
        if mode_key in ("from-scratch", "fromscratch"):
            roi = imread(mask_path, backend=bk) if mask_path else None
            result = blood_flood_from_scratch(
                to_numpy(intensity.data),
                mask=to_numpy(roi.data) if roi is not None else None,
                barrier=barrier_np,
                min_cc_voxels=int(min_cc_voxels),
                **common_kw,
            )
        else:
            markers = imread(markers_path, backend=bk)
            result = blood_flood(
                to_numpy(intensity.data),
                to_numpy(markers.data),
                barrier=barrier_np,
                **common_kw,
            )
        out = output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        imsave(out, result.labels, metadata=dict(intensity.metadata or {}))
        if save_tree is not None:
            save_tree.parent.mkdir(parents=True, exist_ok=True)
            imsave(
                save_tree,
                result.tree.astype(np.uint8),
                metadata=dict(intensity.metadata or {}),
            )
            log.info(f"Wrote vessel tree {save_tree}")
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
