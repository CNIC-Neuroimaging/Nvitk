"""Locate nvitk's JSON configuration files.

nvitk keeps its site configuration in three JSON files — ``sge.json`` (cluster settings and
pipeline path roots), ``settings.json`` (dataset and atlas roots) and ``xnat.json`` (XNAT
connection profile). This module is the single place that decides *where those files live*.

It exists because the previous arrangement had three separate copies of a
``_find_repo_root()`` that ascended from ``__file__`` looking for ``pyproject.toml`` and
``src/nvitk/``. That directory only exists in a source checkout, so for anyone who installed
nvitk from conda or pip every lookup failed and every consumer silently fell back to a
hardcoded path. Configuration is now found in user-owned locations first, and a failed lookup
can say where it looked (:func:`describe_search`).

Precedence, highest first:

1. :func:`set_config_dir` — the ``--config-dir`` flag
2. a per-file environment variable (``NVITK_SGE_JSON``, ``NVITK_SETTINGS_JSON``,
   ``NVITK_XNAT_CONFIG``) — for that one file only
3. ``$NVITK_CONFIG_DIR``
4. ``$NVITK_HOME/.nvitk`` — retained; this already worked, undocumented
5. ``$XDG_CONFIG_HOME/nvitk`` or ``~/.config/nvitk`` — the standard location for an installed
   package, and where ``nvitk.db.xnat_config`` already looked for its profile
6. ``~/.nvitk``
7. ``./.nvitk``
8. the ``.nvitk`` of a source checkout, if this file is running from one

First match wins; the candidates are not layered. Merging several files would make "which
value am I actually using?" unanswerable, which is the failure mode this module was written to
end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator

#: Directory name used everywhere except the XDG location, which is unprefixed by convention.
CONFIG_DIR_NAME = ".nvitk"

ENV_CONFIG_DIR = "NVITK_CONFIG_DIR"
ENV_HOME = "NVITK_HOME"

#: Per-file overrides. ``NVITK_XNAT_CONFIG`` predates this module and is kept under its
#: original name so existing scripts and the ``--xnat-config`` CLI defaults keep working.
FILE_ENV_VARS: dict[str, str] = {
    "sge.json": "NVITK_SGE_JSON",
    "settings.json": "NVITK_SETTINGS_JSON",
    "xnat.json": "NVITK_XNAT_CONFIG",
}

# Set by set_config_dir(); wins over every environment variable.
_override_dir: Path | None = None

# Parsed documents, keyed by filename. Cleared whenever the search path changes.
_doc_cache: dict[str, dict[str, Any]] = {}

# Bumped by reload(); see generation().
_generation: int = 0

# Callables re-run on reload(); see register_reload_hook().
_reload_hooks: list[Callable[[], None]] = []


def _expand(raw: str | os.PathLike[str] | None) -> Path | None:
    """Expand ``~`` and strip whitespace; ``None`` for an empty or missing value."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return Path(os.path.expanduser(text))


def candidate_dirs() -> list[tuple[Path, str]]:
    """Every directory that may hold config, in precedence order.

    Each entry is ``(path, why)``, where *why* is a short human-readable description used by
    :func:`describe_search` to build error messages. Directories are returned whether or not
    they exist — callers filter.
    """
    out: list[tuple[Path, str]] = []

    if _override_dir is not None:
        out.append((_override_dir, "--config-dir"))

    env_dir = _expand(os.environ.get(ENV_CONFIG_DIR))
    if env_dir is not None:
        out.append((env_dir, f"${ENV_CONFIG_DIR}"))

    env_home = _expand(os.environ.get(ENV_HOME))
    if env_home is not None:
        out.append((env_home / CONFIG_DIR_NAME, f"${ENV_HOME}/{CONFIG_DIR_NAME}"))

    out.append((default_config_dir(), "~/.config/nvitk"))

    out.append((Path.home() / CONFIG_DIR_NAME, f"~/{CONFIG_DIR_NAME}"))
    out.append((Path.cwd() / CONFIG_DIR_NAME, f"./{CONFIG_DIR_NAME}"))

    repo = source_checkout_root()
    if repo is not None:
        out.append((repo / CONFIG_DIR_NAME, f"{repo}/{CONFIG_DIR_NAME} (source checkout)"))

    return _dedup(out)


def _dedup(items: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Drop repeated directories, keeping the highest-precedence occurrence of each."""
    seen: set[Path] = set()
    out: list[tuple[Path, str]] = []
    for path, why in items:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append((path, why))
    return out


def source_checkout_root() -> Path | None:
    """The repo root if this module is running from a source tree, else ``None``.

    Deliberately last in the config search order: it is a developer convenience, and relying on
    it is exactly what made configuration unreachable for installed users. It remains the right
    answer for locating *repo assets* that genuinely only exist in a checkout — the
    ``registry/`` submodules, for instance — which is why it is public.
    """
    here = Path(__file__).resolve()
    for anc in [here.parent, *here.parents]:
        if (anc / "pyproject.toml").is_file() and (anc / "src" / "nvitk").is_dir():
            return anc
    return None


def set_config_dir(path: str | os.PathLike[str] | None) -> None:
    """Point nvitk at *path* for the rest of this process, or clear the override with ``None``.

    Also exported into the environment as ``$NVITK_CONFIG_DIR`` so that subprocesses — SGE job
    scripts, Singularity containers, the GUI shelling out to a CLI — resolve the same
    configuration as their parent.
    """
    global _override_dir
    resolved = _expand(path)
    _override_dir = resolved
    if resolved is None:
        os.environ.pop(ENV_CONFIG_DIR, None)
    else:
        os.environ[ENV_CONFIG_DIR] = str(resolved)
    reload()


def register_reload_hook(hook: "Callable[[], None]") -> "Callable[[], None]":
    """Register *hook* to run whenever the configuration search path changes.

    For modules that compute their constants eagerly at import: they register the function that
    performs the merge, and it is re-run when :func:`set_config_dir` redirects the lookup, so a
    late ``--config-dir`` reaches them too. Modules built on
    :func:`nvitk.core.lazy_config.module_getattr` do not need this — they re-resolve on access.
    """
    _reload_hooks.append(hook)
    return hook


def reload() -> None:
    """Drop cached documents so the next read re-resolves and re-parses from disk."""
    global _generation
    _doc_cache.clear()
    _generation += 1
    for hook in list(_reload_hooks):
        try:
            hook()
        except Exception:  # a stale module must not break redirecting the whole process
            import warnings

            warnings.warn(
                f"Config reload hook {getattr(hook, '__qualname__', hook)!r} failed.",
                stacklevel=2,
            )


def generation() -> int:
    """A counter bumped every time the search path changes.

    Lets other modules cache values derived from configuration and notice when that
    configuration has been redirected underneath them — see :mod:`nvitk.core.lazy_config`.
    """
    return _generation


def default_config_dir() -> Path:
    """Where a new configuration should be created.

    The XDG location, derived the same way :func:`candidate_dirs` derives it — so a directory
    created here is one the search will actually find. Computing it independently is how
    ``nvitk-config init`` came to write to ``~/.config/nvitk`` on a machine whose
    ``$XDG_CONFIG_HOME`` pointed somewhere else.
    """
    xdg = _expand(os.environ.get("XDG_CONFIG_HOME")) or (Path.home() / ".config")
    return xdg / "nvitk"


def default_data_dir() -> Path:
    """Where nvitk keeps user data it manages itself (datasets, and similar).

    ``$XDG_DATA_HOME/nvitk`` or ``~/.local/share/nvitk`` — the standard location for
    application data on Linux, and the counterpart of :func:`default_config_dir`. Notably *not*
    a repository-shaped path: someone who installed from conda has no checkout, and a
    ``~/nvitk/dataset/...`` directory appearing in their home both looks like a clone and
    invites confusion with a real one.
    """
    xdg = _expand(os.environ.get("XDG_DATA_HOME")) or (Path.home() / ".local" / "share")
    return xdg / "nvitk"


def config_dir() -> Path | None:
    """The first existing configuration directory, or ``None`` if none exists."""
    for path, _why in candidate_dirs():
        if path.is_dir():
            return path
    return None


def config_file(name: str) -> Path | None:
    """Locate config file *name* (e.g. ``"sge.json"``), or ``None``.

    A per-file environment variable, if set, names the file directly and takes precedence over
    every directory candidate — including ``--config-dir`` — because it is the more specific
    instruction.
    """
    env_var = FILE_ENV_VARS.get(name)
    if env_var:
        direct = _expand(os.environ.get(env_var))
        if direct is not None and direct.is_file():
            return direct

    for directory, _why in candidate_dirs():
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def iter_candidate_files(name: str) -> Iterator[tuple[Path, str]]:
    """Yield every place *name* would be looked for, as ``(path, why)``, in precedence order."""
    env_var = FILE_ENV_VARS.get(name)
    if env_var:
        direct = _expand(os.environ.get(env_var))
        if direct is not None:
            yield direct, f"${env_var}"
    for directory, why in candidate_dirs():
        yield directory / name, why


def load_json(name: str) -> dict[str, Any]:
    """Parse config file *name*, or return ``{}`` if it is absent or unreadable.

    Returning ``{}`` for a *missing* file is intentional — most settings are optional and have
    defaults. Callers that need a value must ask for it explicitly via :func:`require`, which
    produces an error naming the key and the search path. A file that exists but cannot be
    parsed always warns, since that is a mistake rather than a choice.
    """
    if name in _doc_cache:
        return _doc_cache[name]

    path = config_file(name)
    doc: dict[str, Any] = {}
    if path is not None:
        try:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            doc = loaded if isinstance(loaded, dict) else {}
            if not isinstance(loaded, dict):
                import warnings

                warnings.warn(
                    f"{path} does not contain a JSON object; ignoring it.", stacklevel=2
                )
        except (OSError, json.JSONDecodeError) as exc:
            import warnings

            warnings.warn(f"Could not read {path}: {exc}", stacklevel=2)
            doc = {}

    _doc_cache[name] = doc
    return doc


def describe_search(name: str | None = None) -> str:
    """Every location *name* was looked for, as a single line, for an error message.

    A resolver that fails should say where it looked: "no configuration found" sends the reader
    to the documentation, this sends them to the path they need to create.
    """
    if name is None:
        parts = [why for _path, why in candidate_dirs()]
    else:
        parts = [f"{why} ({path})" for path, why in iter_candidate_files(name)]
    return "; ".join(parts)


class ConfigError(RuntimeError):
    """A required configuration value is missing or unusable."""


def require(
    value: Any,
    *,
    key: str,
    file: str = "sge.json",
    hint: str = "",
) -> Any:
    """Return *value*, or raise :class:`ConfigError` naming *key* and where *file* was sought.

    Used in place of the institution-specific path constants this codebase used to fall back
    on. Failing here — at the point the value is needed, with the key name in the message — is
    far more useful than defaulting to somebody else's filesystem layout and surfacing as a
    ``FileNotFoundError`` several stages into a pipeline run.
    """
    if value is not None and str(value).strip():
        return value
    message = (
        f'Required setting "{key}" is not configured in {file}.\n'
        f"Looked in: {describe_search(file)}\n"
        "Run `nvitk-config init` to create a starter configuration, then set the key."
    )
    if hint:
        message = f"{message}\n{hint}"
    raise ConfigError(message)


__all__ = [
    "CONFIG_DIR_NAME",
    "ENV_CONFIG_DIR",
    "ENV_HOME",
    "FILE_ENV_VARS",
    "ConfigError",
    "candidate_dirs",
    "config_dir",
    "config_file",
    "default_config_dir",
    "default_data_dir",
    "source_checkout_root",
    "describe_search",
    "generation",
    "iter_candidate_files",
    "load_json",
    "register_reload_hook",
    "reload",
    "require",
    "set_config_dir",
]
