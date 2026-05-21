"""``nvitk-gui`` entry point — Napari workbench for nvitk tools."""

from __future__ import annotations

import sys

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Launch the Napari workbench for nvitk image tools."""
    try:
        import napari  # noqa: F401
    except ImportError as exc:
        print(
            "nvitk-gui requires napari. Install with: pip install 'nvitk[gui]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from nvitk.gui.app import run_app
    from nvitk.gui.warnings import install_napari_display_warnings

    install_napari_display_warnings()
    run_app()


if __name__ == "__main__":
    main()
