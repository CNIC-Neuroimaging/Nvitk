"""``nvitk-gui`` entry point — Napari workbench for nvitk tools."""

from __future__ import annotations

import sys

import click

from nvitk.core.click_backend import apply_cli_backend, backend_click_option


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@backend_click_option(default="cpu")
def main(backend: str) -> None:
    """Launch the Napari workbench for nvitk image tools."""
    try:
        import napari  
    except ImportError as exc:
        print(
            "nvitk-gui requires napari. Install with: pip install 'nvitk[gui]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from nvitk.gui.app import run_app
    from nvitk.gui.warnings import install_napari_display_warnings

    install_napari_display_warnings()
    apply_cli_backend(backend)
    run_app()


if __name__ == "__main__":
    main()
