"""gPET pipeline stub (not implemented)."""

from __future__ import annotations

import click

from nvitk.core.logger import Logger

log = Logger()


@click.command("nvitk-gpetpy")
def main() -> None:
    """Placeholder for future gPET pipeline."""
    log.error("gPET pipeline (gpetpy) is not implemented yet.")
    raise NotImplementedError("gPET pipeline (gpetpy) is not implemented yet.")


if __name__ == "__main__":
    main()
