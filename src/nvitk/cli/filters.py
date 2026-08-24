"""``nvitk-filter`` CLI — image filters."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.cli._common import (
    backend_option,
    dispatch_tool,
    io_options,
    mask_option,
    submit_options,
)
from nvitk.filters.hessian import HESSIAN_SIGMAS_DEFAULT, hessian_filter, parse_sigmas
from nvitk.filters.jerman import (
    JERMAN_SIGMAS_DEFAULT,
    JERMAN_TAU_DEFAULT,
    jerman_filter,
)
from nvitk.filters.sliding_threshold import (
    binary_mask_sliding_threshold_2d,
    binary_mask_sliding_threshold_3d,
)
from nvitk.filters.snakes import (
    SNAKES_ALPHA_DEFAULT,
    SNAKES_BETA_DEFAULT,
    SNAKES_GAMMA_DEFAULT,
    SNAKES_MAX_ITER_DEFAULT,
    SNAKES_N_POINTS_DEFAULT,
    SNAKES_SIGMA_DEFAULT,
    SNAKES_W_EDGE_DEFAULT,
    SNAKES_W_LINE_DEFAULT,
    snakes_filter,
)
from nvitk.core.click_config import config_dir_click_option


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
def main() -> None:
    """Image filtering tools."""


@main.command("sliding-threshold")
@io_options
@backend_option(True)
@submit_options
@click.option("--dim", type=click.Choice(["2d", "3d"]), default="3d")
@click.option("--step", type=float, default=0.001)
@click.option("--up-thresh", type=float, default=0.8)
def cmd_sliding_threshold(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    dim: str,
    step: float,
    up_thresh: float,
) -> None:
    """Sliding-threshold binary segmentation."""
    from nvitk.types import Image

    def runner(image, *, step, up_thresh, dim):
        """Apply 2D or 3D sliding-threshold binary segmentation to *image*, wrapping the result back
        into an :class:`~nvitk.types.Image` when the input was one."""
        data = image.data if isinstance(image, Image) else image
        if dim == "2d":
            mask, _ = binary_mask_sliding_threshold_2d(data, step=step, up_thresh=up_thresh)
        else:
            mask, _ = binary_mask_sliding_threshold_3d(data, step=step, up_thresh=up_thresh)
        if isinstance(image, Image):
            return image.with_data(mask.astype(image.data.dtype))
        return mask

    dispatch_tool(
        tool="filter",
        subcommand="sliding-threshold",
        module_file="filters.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=runner,
        runner_kwargs={"step": step, "up_thresh": up_thresh, "dim": dim},
    )


@main.command("hessian")
@io_options
@backend_option(False)
@submit_options
@click.option(
    "--sigmas",
    default=None,
    help=(
        "Comma-separated Gaussian scales "
        f"(default: {','.join(str(s) for s in HESSIAN_SIGMAS_DEFAULT)})."
    ),
)
@click.option(
    "--black-ridges/--bright-ridges",
    default=False,
    show_default=True,
    help="Detect dark ridges (default off → bright ridges, e.g. TOF vessels).",
)
@click.option("--alpha", type=float, default=0.5, show_default=True)
@click.option("--beta", type=float, default=0.5, show_default=True)
@click.option("--gamma", type=float, default=15.0, show_default=True)
@click.option(
    "--mode",
    type=click.Choice(["reflect", "constant", "nearest", "mirror", "wrap"], case_sensitive=False),
    default="reflect",
    show_default=True,
)
def cmd_hessian(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    sigmas: str | None,
    black_ridges: bool,
    alpha: float,
    beta: float,
    gamma: float,
    mode: str,
) -> None:
    """Hybrid Hessian ridge / vessel filter (skimage)."""

    def runner(image, *, sigmas, black_ridges, alpha, beta, gamma, mode):
        """Apply the Hessian ridge/vessel filter to *image* with the parsed CLI parameters."""
        return hessian_filter(
            image,
            sigmas=parse_sigmas(sigmas),
            black_ridges=bool(black_ridges),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            mode=str(mode),
        )

    dispatch_tool(
        tool="filter",
        subcommand="hessian",
        module_file="filters.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=runner,
        runner_kwargs={
            "sigmas": sigmas,
            "black_ridges": black_ridges,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "mode": mode,
        },
    )


@main.command("jerman")
@io_options
@backend_option(False)
@submit_options
@click.option(
    "--sigmas",
    default=None,
    help=(
        "Comma-separated Gaussian scales "
        f"(default: {','.join(str(s) for s in JERMAN_SIGMAS_DEFAULT)})."
    ),
)
@click.option(
    "--tau",
    type=float,
    default=JERMAN_TAU_DEFAULT,
    show_default=True,
    help="Regularization in [0.5, 1]; lower → stronger response.",
)
@click.option(
    "--black-ridges/--bright-ridges",
    default=False,
    show_default=True,
    help="Detect dark ridges (default off → bright ridges, e.g. TOF vessels).",
)
@click.option(
    "--mode",
    type=click.Choice(["reflect", "constant", "nearest", "mirror", "wrap"], case_sensitive=False),
    default="reflect",
    show_default=True,
)
def cmd_jerman(
    input_path: Path,
    output_path: Path,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    sigmas: str | None,
    tau: float,
    black_ridges: bool,
    mode: str,
) -> None:
    """Jerman vesselness / ridge filter (IEEE TMI 2016)."""

    def runner(image, *, sigmas, tau, black_ridges, mode):
        """Apply the Jerman vesselness/ridge filter to *image* with the parsed CLI parameters."""
        return jerman_filter(
            image,
            sigmas=parse_sigmas(sigmas),
            tau=float(tau),
            black_ridges=bool(black_ridges),
            mode=str(mode),
        )

    dispatch_tool(
        tool="filter",
        subcommand="jerman",
        module_file="filters.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=None,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=runner,
        runner_kwargs={
            "sigmas": sigmas,
            "tau": tau,
            "black_ridges": black_ridges,
            "mode": mode,
        },
    )


@main.command("snakes")
@io_options
@mask_option
@backend_option(False)
@submit_options
@click.option("--alpha", type=float, default=SNAKES_ALPHA_DEFAULT, show_default=True, help="Tension (length) weight.")
@click.option("--beta", type=float, default=SNAKES_BETA_DEFAULT, show_default=True, help="Rigidity (smoothness) weight.")
@click.option("--w-line", type=float, default=SNAKES_W_LINE_DEFAULT, show_default=True, help="Line (intensity) attraction.")
@click.option("--w-edge", type=float, default=SNAKES_W_EDGE_DEFAULT, show_default=True, help="Edge attraction.")
@click.option("--gamma", type=float, default=SNAKES_GAMMA_DEFAULT, show_default=True, help="Time-step size.")
@click.option("--max-iter", type=int, default=SNAKES_MAX_ITER_DEFAULT, show_default=True)
@click.option("--sigma", type=float, default=SNAKES_SIGMA_DEFAULT, show_default=True, help="Gaussian pre-smooth sigma (0=off).")
@click.option("--n-points", type=int, default=SNAKES_N_POINTS_DEFAULT, show_default=True, help="Snake control points.")
@click.option(
    "--axis",
    type=click.IntRange(0, 2),
    default=0,
    show_default=True,
    help="Slice axis for 3-D volumes.",
)
@click.option(
    "--boundary",
    type=click.Choice(
        ["periodic", "free", "fixed", "free-fixed", "fixed-free"],
        case_sensitive=False,
    ),
    default="periodic",
    show_default=True,
)
def cmd_snakes(
    input_path: Path,
    output_path: Path,
    mask_path: Path | None,
    backend: str,
    submit: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    alpha: float,
    beta: float,
    w_line: float,
    w_edge: float,
    gamma: float,
    max_iter: int,
    sigma: float,
    n_points: int,
    axis: int,
    boundary: str,
) -> None:
    """Kass snakes / active contours (IJCV 1988); needs --mask init contour."""
    if mask_path is None:
        raise click.ClickException("snakes requires --mask (initial contour / seed mask).")

    def runner(image, mask, *, alpha, beta, w_line, w_edge, gamma, max_iter, sigma, n_points, axis, boundary):
        """Run Kass active-contour (snakes) evolution on *image* seeded from the initial *mask*
        contour, using the parsed CLI parameters."""
        return snakes_filter(
            image,
            mask,
            alpha=float(alpha),
            beta=float(beta),
            w_line=float(w_line),
            w_edge=float(w_edge),
            gamma=float(gamma),
            max_num_iter=int(max_iter),
            gaussian_sigma=float(sigma),
            n_points=int(n_points),
            axis=int(axis),
            boundary_condition=str(boundary),
        )

    dispatch_tool(
        tool="filter",
        subcommand="snakes",
        module_file="filters.py",
        input_path=input_path,
        output_path=output_path,
        submit=submit,
        backend=backend,
        mask_path=mask_path,
        emit_script=emit_script,
        direct_submit=direct_submit,
        no_remote=no_remote,
        dry_run=dry_run,
        runner=runner,
        runner_kwargs={
            "alpha": alpha,
            "beta": beta,
            "w_line": w_line,
            "w_edge": w_edge,
            "gamma": gamma,
            "max_iter": max_iter,
            "sigma": sigma,
            "n_points": n_points,
            "axis": axis,
            "boundary": boundary,
        },
    )


if __name__ == "__main__":
    main()
