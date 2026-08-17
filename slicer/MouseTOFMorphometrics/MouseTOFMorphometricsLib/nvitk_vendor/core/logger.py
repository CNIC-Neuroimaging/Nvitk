"""Stdlib-logging replacement for ``nvitk.core.logger`` (hand-written; not synced).

Upstream this is a Rich-backed singleton with progress bars and file handlers.
Inside Slicer, messages should land in Slicer's own Python log, so this forwards
to :mod:`logging` and drops the Rich dependency entirely.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER_NAME = "MouseTOFMorphometrics.nvitk"


class Logger:
    """Minimal drop-in for nvitk's singleton logger."""

    _instance: "Logger | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._log = logging.getLogger(_LOGGER_NAME)
        return cls._instance

    # -- levels -------------------------------------------------------------
    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log.debug(str(msg), *args)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log.info(str(msg), *args)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log.warning(str(msg), *args)

    warn = warning

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log.error(str(msg), *args)

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log.exception(str(msg), *args)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log.critical(str(msg), *args)

    # -- no-op knobs kept for signature compatibility -----------------------
    def set_level(self, *args: Any, **kwargs: Any) -> None:
        return None

    def enable_debug_locals(self, *args: Any, **kwargs: Any) -> None:
        return None

    def progress(self, *args: Any, **kwargs: Any) -> None:
        return None


__all__ = ["Logger"]
