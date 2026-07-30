"""``nvitk-seg`` CLI — segmentation tools (mouse brain, blood flood)."""

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
    TREE_VESSELNESS_KEEP_PERCENTILE_DEFAULT,
    blood_flood,
)
from nvitk.segmentation.mouse_brain import mouse_brain_segmentation

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
    verbose: bool,
):
    return mouse_brain_segmentation(
        image,
        mode=mode,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
        which_parcellation=which_parcellation,
        mask=mask,
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
    type=click.Choice(["t2", "t1"], case_sensitive=False),
    default="t2",
    show_default=True,
    help="Imaging contrast for extraction.",
)
@click.option(
    "--which-parcellation",
    default="nick",
    show_default=True,
    help="Parcellation scheme (ANTsPyNet).",
)
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
            "which_parcellation": which_parcellation,
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
    "--markers",
    "markers_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Integer seed / marker label volume (0 = background).",
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
    markers_path: Path,
    barrier_path: Path | None,
    frangi_sigmas: str | None,
    hyst_low_factor: float,
    hyst_high_factor: float,
    thicken_iter: int,
    thin_vesselness_percentile: float,
    connectivity: int,
    save_tree: Path | None,
) -> None:
    """Frangi → hysteresis vessel tree → marker watershed (qvtpy distal expand)."""
    if submit.lower() != "local":
        raise click.ClickException(
            "blood-flood SGE submit is not wired yet; use --submit local."
        )
    apply_cli_backend(backend)
    bk = "cupy" if backend.lower() in ("gpu", "cupy") else "numpy"
    thin = (
        None
        if float(thin_vesselness_percentile) < 0
        else float(thin_vesselness_percentile)
    )
    with using(bk):
        intensity = imread(input_path, backend=bk)
        markers = imread(markers_path, backend=bk)
        barrier = imread(barrier_path, backend=bk) if barrier_path else None
        result = blood_flood(
            to_numpy(intensity.data),
            to_numpy(markers.data),
            barrier=to_numpy(barrier.data) if barrier is not None else None,
            frangi_sigmas=_parse_sigmas(frangi_sigmas),
            hyst_low_factor=float(hyst_low_factor),
            hyst_high_factor=float(hyst_high_factor),
            thicken_iter=int(thicken_iter),
            thin_vesselness_percentile=thin,
            connectivity=int(connectivity),
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
