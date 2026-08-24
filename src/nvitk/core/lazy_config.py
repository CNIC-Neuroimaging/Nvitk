"""Resolve module-level configuration constants on first use rather than at import.

The pipeline and tool ``config.py`` modules expose their settings as module constants
(``cfg.SGE_QUEUE``, ``cfg.CONTAINER_PATH``, …). Those used to be computed at *import* time by
reading ``sge.json`` and rebinding the names, which had three costs:

* a ``--config-dir`` flag could never affect them, because the import happens before the
  command line is parsed;
* importing a config module did disk I/O, so ``--help`` paid for reading a cluster config it
  was never going to use;
* the file was re-read and re-parsed several times per module.

:func:`module_getattr` builds a :pep:`562` ``__getattr__`` that keeps the same
``cfg.CONSTANT`` spelling but computes each value the first time it is read, caching it until
the configuration search path changes (tracked by
:func:`nvitk.core.config_paths.generation`).

Usage::

    _RESOLVERS = {
        "SGE_QUEUE":      lambda: _pipe().get("sge_queue"),
        "CONTAINER_PATH": lambda: sge_json.resolve_nvitk_container(pipe=_pipe()),
    }
    __getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)

``from cfg import CONSTANT`` still works — :pep:`562` fires for that too — but it snapshots the
value at the importing module's import time. Prefer ``import … as cfg`` plus ``cfg.CONSTANT``
anywhere a late ``--config-dir`` needs to be honoured.
"""

from __future__ import annotations

from typing import Any, Callable

from nvitk.core import config_paths

Resolver = Callable[[], Any]


def module_getattr(
    resolvers: dict[str, Resolver],
    *,
    module_name: str,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build ``(__getattr__, __dir__)`` for a module whose constants come from configuration.

    *resolvers* maps each constant name to a zero-argument callable producing its value.
    Results are cached per name and dropped whenever the configuration generation changes, so a
    mid-process :func:`~nvitk.core.config_paths.set_config_dir` is picked up rather than
    silently ignored.
    """
    cache: dict[str, Any] = {}
    cached_generation = -1

    def __getattr__(name: str) -> Any:
        """Resolve *name* from configuration, caching until the config generation changes."""
        nonlocal cached_generation
        current = config_paths.generation()
        if current != cached_generation:
            cache.clear()
            cached_generation = current
        if name in cache:
            return cache[name]
        try:
            resolver = resolvers[name]
        except KeyError:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from None
        value = resolver()
        cache[name] = value
        return value

    def __dir__() -> list[str]:
        """Include the lazily-resolved names so tab-completion and ``dir()`` still show them."""
        return sorted(resolvers)

    return __getattr__, __dir__


def first_set(*values: Any) -> Any:
    """The first argument that is neither ``None`` nor blank, else ``None``.

    For settings that accept more than one spelling in ``sge.json`` — several pipelines read
    both ``default_sge_model_root`` and ``models_dir``, for instance.
    """
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


__all__ = ["Resolver", "first_set", "module_getattr"]
