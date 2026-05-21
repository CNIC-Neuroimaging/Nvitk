"""gPET pipeline stub (not implemented)."""

from __future__ import annotations

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger

log = Logger()


@click.command("nvitk-gpetpy")
@backend_click_option()
def main(backend: str) -> None:  # noqa: ARG001
    """Placeholder for future gPET pipeline."""
    log.error("gPET pipeline (gpetpy) is not implemented yet.")
    raise NotImplementedError("gPET pipeline (gpetpy) is not implemented yet.")


if __name__ == "__main__":
    main()
