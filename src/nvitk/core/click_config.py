"""Shared ``--config-dir`` option for nvitk CLIs and pipeline entry points.

Mirrors :mod:`nvitk.core.click_backend`, but for the configuration search path rather than the
array backend.

The option is **eager**: Click resolves it before any other parameter, so
:func:`~nvitk.core.config_paths.set_config_dir` runs before the command body reads a setting.
That is enough for every lazily-read value. A module that reads configuration at *import* time
is resolved before Click runs at all and cannot be redirected by a flag — for those,
``NVITK_CONFIG_DIR`` in the environment is the mechanism, since it is set before the process
starts.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import click

from nvitk.core import config_paths

F = TypeVar("F", bound=Callable[..., Any])


def _apply(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Eager Click callback: point nvitk at *value* before the rest of the command runs."""
    if value:
        config_paths.set_config_dir(value)
    return value


def config_dir_click_option() -> Callable[[F], F]:
    """Decorator: add an eager ``--config-dir`` that redirects nvitk's configuration lookup."""

    def decorator(f: F) -> F:
        """Attach ``--config-dir`` to *f*."""
        return click.option(
            "--config-dir",
            "config_dir",
            type=click.Path(file_okay=False, dir_okay=True),
            default=None,
            expose_value=False,
            is_eager=True,
            callback=_apply,
            help=(
                "Directory holding nvitk's sge.json / settings.json / xnat.json "
                "(default: $NVITK_CONFIG_DIR, then ~/.config/nvitk). "
                "See `nvitk-config path`."
            ),
        )(f)

    return decorator


def preparse_config_dir(argv: list[str] | None = None) -> str | None:
    """Apply ``--config-dir`` straight from *argv*, before any nvitk import happens.

    For argparse-based entry points, and for the case where configuration has to be redirected
    ahead of module-level reads. Call it as the very first statement of ``main()``, before
    importing anything that touches configuration::

        def main() -> None:
            preparse_config_dir()
            from .window import StatmodelsWindow   # noqa: E402

    Accepts both ``--config-dir X`` and ``--config-dir=X``. The flag is left in ``argv`` for the
    real parser to consume and report normally.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    value: str | None = None
    for index, token in enumerate(args):
        if token == "--config-dir" and index + 1 < len(args):
            value = args[index + 1]
        elif token.startswith("--config-dir="):
            value = token.split("=", 1)[1]
    if value:
        config_paths.set_config_dir(value)
    return value


__all__ = ["config_dir_click_option", "preparse_config_dir"]
