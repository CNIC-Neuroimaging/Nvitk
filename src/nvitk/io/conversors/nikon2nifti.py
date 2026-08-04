"""Convert Nikon ND2 to NIfTI via :func:`~nvitk.io.readers.read_nd2` and :func:`~nvitk.io.writers.write_nifti`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import click
except Exception:
    click = None


def _cli_decorator(*args, **kwargs):
    """No-op stand-in for ``click.command``/``click.option`` when click isn't installed."""
    def decorator(func):
        """Return *func* unmodified (click is unavailable, so no CLI wiring is applied)."""
        return func
    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator

from nvitk.core.exceptions import BackendUnavailableError
from ..readers import read_nd2
from ..writers import write_nifti

__all__ = ["nikon2nifti", "main"]


def nikon2nifti(
    nikon_path: str,
    nifti_path: str,
    *,
    axes: str | None = None,
    **kwargs: Any,
) -> str:
    """Convert a Nikon ND2 file to NIfTI; library entry point behind the ``nikon2nifti`` CLI."""
    data, metadata = read_nd2(nikon_path, axes=axes, **kwargs)
    write_nifti(nifti_path, data, metadata=metadata, axes=axes)
    return nifti_path


@_click_command(context_settings={"help_option_names": ["-h", "--help"]})
@_click_option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Path to Nikon ND2 file.",
)
@_click_option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path) if click is not None else None,
    required=True,
    help="Output NIfTI path (.nii or .nii.gz).",
)
@_click_option("--axes", type=str, default=None, help="Optional axis order for ND2 read.")
def main(input_path: Path, output_path: Path, axes: str | None) -> None:
    """CLI entry point: convert one Nikon ND2 file to NIfTI."""
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nikon2nifti(str(input_path), str(output_path), axes=axes)
