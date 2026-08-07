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
    from nvitk.gui.core.warnings import install_napari_display_warnings

    # Qt refuses to load its web engine once a QApplication exists unless a shared GL context was
    # requested first, and napari builds one as it starts. The Statmodels window's interactive
    # plots are hosted in a web view, so the flag has to be set here — before napari runs — or they
    # silently fall back to a "web engine unavailable" message.
    from nvitk.gui.panels.statmodels.plotly_view import prepare_webengine

    prepare_webengine()
    install_napari_display_warnings()
    apply_cli_backend(backend)
    run_app()


if __name__ == "__main__":
    main()
