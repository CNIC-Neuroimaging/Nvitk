"""Shared Click options and local/SGE execution for module CLIs."""

from __future__ import annotations

import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from nvitk.core.backend import using
from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave

log = Logger()


def backend_option(supports_gpu: bool = True):
    """Add ``--backend cpu|gpu`` (default gpu) and apply it before the command runs."""

    _ = supports_gpu  # legacy arg; backend is always exposed on tool CLIs

    def decorator(f):
        f = click.option(
            "--backend",
            type=click.Choice(["cpu", "gpu"], case_sensitive=False),
            default="gpu",
            show_default=True,
            help="Array backend: cpu (NumPy) or gpu (CuPy).",
        )(f)

        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            apply_cli_backend(kwargs.get("backend", "gpu"))
            return f(*args, **kwargs)

        return wrapper

    return decorator


def io_options(f):
    f = click.option(
        "-i",
        "--input",
        "input_path",
        type=click.Path(path_type=Path, exists=True),
        required=True,
        help="Input image path (NIfTI, DICOM folder, etc.).",
    )(f)
    f = click.option(
        "-o",
        "--output",
        "output_path",
        type=click.Path(path_type=Path),
        required=True,
        help="Output image or directory path.",
    )(f)
    return f


def mask_option(f):
    return click.option(
        "--mask",
        "mask_path",
        type=click.Path(path_type=Path, exists=True),
        default=None,
        help="Optional mask/label volume path.",
    )(f)


def submit_options(f):
    f = click.option(
        "--submit",
        type=click.Choice(["local", "sge"], case_sensitive=False),
        default="local",
        show_default=True,
    )(f)
    f = click.option("--emit-script", "emit_script", type=click.Path(path_type=Path), default=None)
    f = click.option("--direct-submit", is_flag=True, default=False)
    f = click.option("--no-remote", is_flag=True, default=False)
    f = click.option("--dry-run", is_flag=True, default=False)
    return f


def run_local(
    input_path: Path,
    output_path: Path,
    *,
    backend: str = "cpu",
    mask_path: Path | None = None,
    runner: Callable[..., Any],
    runner_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Read input, run tool, save output locally."""
    bk = "cupy" if backend.lower() in ("gpu", "cupy") else "numpy"
    kwargs = dict(runner_kwargs or {})
    with using(bk):
        image = imread(input_path, backend=bk)
        mask = imread(mask_path, backend=bk) if mask_path else None
        if mask is not None:
            result = runner(image, mask, **kwargs)
        else:
            result = runner(image, **kwargs)
        if result is None:
            raise click.ClickException("Tool returned no output.")
        out = output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        imsave(out, result)
    log.info(f"Wrote {out}")
    return out


def run_sge(
    *,
    tool: str,
    subcommand: str,
    module_file: str,
    input_path: Path,
    output_path: Path,
    backend: str = "cpu",
    mask_path: Path | None = None,
    emit_script: Path | None = None,
    direct_submit: bool = False,
    no_remote: bool = False,
    dry_run: bool = False,
    extra_cli_args: list[str] | None = None,
) -> None:
    from nvitk.cli._sge import build_worker_command, default_emit_path, emit_submit_script, submit_tool_job
    from nvitk.cluster.remote_submit import run_sge_script_ssh

    gpu = backend.lower() in ("gpu", "cupy")
    data_root = input_path.parent.resolve()
    output_root = output_path.parent.resolve()
    container_in = f"/nvitk/data/{input_path.name}"
    container_out = f"/nvitk/output/{output_path.name}"
    extra: list[str] = ["--backend", backend]
    if mask_path:
        extra.extend(["--mask", f"/nvitk/data/{mask_path.name}"])
    if extra_cli_args:
        extra.extend(extra_cli_args)

    python_cmd = build_worker_command(
        module_file,
        subcommand,
        container_input=container_in,
        container_output=container_out,
        extra_args=extra,
    )
    job_name = f"{tool}_{subcommand}"[:200]

    if dry_run:
        click.echo(python_cmd)
        return

    script = emit_script or default_emit_path(tool, subcommand)
    if emit_script or not direct_submit:
        emit_submit_script(
            script_path=script,
            stages=[(job_name, python_cmd)],
            data_root=data_root,
            output_root=output_root,
            gpu=gpu,
        )
        log.info(f"Wrote SGE script: {script}")
        if not direct_submit and not no_remote:
            run_sge_script_ssh(script)
        return

    submit_tool_job(
        job_name=job_name,
        python_cmd=python_cmd,
        data_root=data_root,
        output_root=output_root,
        gpu=gpu,
    )


def dispatch_tool(
    *,
    tool: str,
    subcommand: str,
    module_file: str,
    input_path: Path,
    output_path: Path,
    submit: str,
    backend: str = "gpu",
    mask_path: Path | None,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
    runner: Callable[..., Any],
    runner_kwargs: dict[str, Any] | None = None,
) -> None:
    apply_cli_backend(backend)
    if submit.lower() == "local":
        run_local(
            input_path,
            output_path,
            backend=backend,
            mask_path=mask_path,
            runner=runner,
            runner_kwargs=runner_kwargs,
        )
    else:
        run_sge(
            tool=tool,
            subcommand=subcommand,
            module_file=module_file,
            input_path=input_path,
            output_path=output_path,
            backend=backend,
            mask_path=mask_path,
            emit_script=emit_script,
            direct_submit=direct_submit,
            no_remote=no_remote,
            dry_run=dry_run,
        )
