"""g_pet pipeline stub (not implemented)."""

from __future__ import annotations

import click

from nvitk.core.logger import Logger

log = Logger()


@click.command("nvitk-pesa-brain-gpet")
def main() -> None:
    """Placeholder for future g_pet pipeline."""
    log.error("g_pet pipeline is not implemented yet.")
    raise NotImplementedError("g_pet pipeline is not implemented yet.")


if __name__ == "__main__":
    main()
