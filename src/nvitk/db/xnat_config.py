"""Load XNAT connection settings from JSON/YAML and resolve credentials.

**Do not store passwords in dataset Parquet or commit them to git.** Use
environment variables (``XNAT_USER`` / ``XNAT_PASSWORD``), ``~/.netrc``, optional
``keyring``, or a **local** config file outside the repo.

Merge precedence for :func:`resolve_xnat_connection` for ``server`` / ``project`` / ``user``:

1. Explicit **overrides** (e.g. CLI flags that are set)
2. Values from the profile file (JSON/YAML)
3. Environment variables (``XNAT_SERVER``, ``XNAT_PROJECT``, ``XNAT_USER``)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


def _coerce_xnat_identifier(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is required")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError(f"{field} must be non-empty")
        return s
    if isinstance(value, (set, frozenset, list, tuple)):
        if len(value) != 1:
            raise TypeError(
                f"{field} must be a single string, not {type(value).__name__} with {len(value)} elements"
            )
        return _coerce_xnat_identifier(next(iter(value)), field)
    return _coerce_xnat_identifier(str(value), field)


@dataclass(frozen=True)
class XnatConnectionConfig:
    server: str
    project: str
    user: str | None = None
    password: str | None = None
    netrc_file: str | None = None
    verify: bool = False
    default_timeout: int = 600

    def __post_init__(self) -> None:
        object.__setattr__(self, "server", _coerce_xnat_identifier(self.server, "server"))
        object.__setattr__(self, "project", _coerce_xnat_identifier(self.project, "project"))

KEYRING_SERVICE = "nvitk"


def _default_config_paths() -> list[Path]:
    base = Path.home() / ".config" / "nvitk"
    return [base / "xnat.yaml", base / "xnat.yml", base / "xnat.json"]


def load_xnat_profile(path: Path | None = None) -> dict[str, Any]:
    """Load ``xnat`` profile dict from JSON or YAML.

    If ``path`` is ``None``, uses ``NVITK_XNAT_CONFIG`` if set, else the first
    existing file among ``~/.config/nvitk/xnat.{yaml,yml,json}``.

    YAML requires PyYAML (``pip install pyyaml`` or ``nvitk[xnat]``).
    """
    if path is not None:
        return _load_file(Path(path))

    env_path = os.getenv("NVITK_XNAT_CONFIG")
    if env_path:
        return _load_file(Path(env_path).expanduser())

    for candidate in _default_config_paths():
        if candidate.exists():
            return _load_file(candidate)

    return {}


def _load_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config suffix for {path}: use .json, .yaml, or .yml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"XNAT config root must be a mapping, got {type(data)}")
    return data


def _password_from_keyring(server: str) -> str | None:
    try:
        import keyring
    except ImportError:
        return None
    return keyring.get_password(KEYRING_SERVICE, f"xnat:{server}")


def resolve_xnat_connection(
    profile: Mapping[str, Any] | None = None,
    *,
    server: str | None = None,
    project: str | None = None,
    user: str | None = None,
    password: str | None = None,
    netrc_file: str | None = None,
    verify: bool | None = None,
    default_timeout: int | None = None,
) -> XnatConnectionConfig:
    """Build :class:`XnatConnectionConfig` from profile, environment, and overrides.

    Password resolution when not explicitly provided:

    1. ``profile['password']`` (discouraged for shared files)
    2. ``password_env`` env var name in profile → ``os.environ``
    3. ``password_keyring: true`` in profile → keyring for ``xnat:{server}``
    4. ``XNAT_PASSWORD`` environment variable
    """
    p = dict(profile or {})

    def pick_str(key: str, override: str | None, env_key: str) -> str | None:
        if override is not None and str(override).strip():
            return str(override).strip()
        v = p.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
        ev = os.getenv(env_key)
        if ev is not None and str(ev).strip():
            return str(ev).strip()
        return None

    srv = pick_str("server", server, "XNAT_SERVER")
    proj = pick_str("project", project, "XNAT_PROJECT")
    usr = pick_str("user", user, "XNAT_USER")

    if not srv or not proj:
        raise ValueError(f"XNAT server and project are required (server={srv!r}, project={proj!r}).")

    pwd: str | None = None
    if password is not None and str(password).strip():
        pwd = str(password).strip()
    elif p.get("password") is not None and str(p.get("password")).strip():
        pwd = str(p.get("password")).strip()
    else:
        env_name = p.get("password_env")
        if isinstance(env_name, str) and env_name.strip():
            pwd = os.getenv(env_name.strip())
        if p.get("password_keyring") is True and pwd is None:
            pwd = _password_from_keyring(srv)
        if pwd is None:
            pwd = os.getenv("XNAT_PASSWORD")
        if pwd is not None and not str(pwd).strip():
            pwd = None

    netrc = netrc_file
    if netrc is None and p.get("netrc_file"):
        netrc = str(p["netrc_file"])
    if netrc is None and os.getenv("XNAT_NETRC"):
        netrc = os.getenv("XNAT_NETRC")
    if netrc is None:
        default_netrc = Path.home() / ".netrc"
        if default_netrc.is_file():
            netrc = str(default_netrc)

    # CNIC XNAT uses an internal CA; default to verify=False unless profile/CLI enables it.
    ver = verify if verify is not None else bool(p.get("verify", False))
    if not isinstance(ver, bool):
        ver = bool(ver)

    timeout = default_timeout if default_timeout is not None else p.get("default_timeout", 300)
    if not isinstance(timeout, int):
        timeout = int(timeout)

    return XnatConnectionConfig(
        server=srv,
        project=proj,
        user=usr,
        password=pwd,
        netrc_file=netrc,
        verify=ver,
        default_timeout=timeout,
    )


def finalize_xnat_connection(
    config: XnatConnectionConfig,
    *,
    prompt_password: bool = True,
    force_prompt_password: bool = False,
) -> XnatConnectionConfig:
    """Optionally prompt for XNAT password on an interactive terminal.

    When *force_prompt_password* is True (e.g. ``--xnat-config`` without an explicit
    password), netrc/keyring passwords are ignored and the user is prompted.
    """
    if not prompt_password:
        return config

    from .xnat import _resolve_xnat_login

    resolved_user, resolved_password, _ = _resolve_xnat_login(config)
    user = resolved_user or config.user
    password = config.password or resolved_password

    if force_prompt_password:
        password = None
    elif config.password:
        password = config.password

    if password and not force_prompt_password:
        return replace(config, user=user, password=password)

    if not sys.stdin.isatty():
        return replace(config, user=user, password=password)

    import getpass

    if not user:
        user = input(f"XNAT username for {config.server}: ").strip()
    if not user:
        raise ValueError(f"XNAT username is required for {config.server!r}")
    pwd = getpass.getpass(f"XNAT password for {user}: ")
    if not pwd:
        raise ValueError("XNAT password is required")
    return replace(config, user=user, password=pwd)


def keyring_set_main() -> None:
    """CLI: store XNAT password in the system keyring (``keyring`` package required)."""
    import getpass

    try:
        import click
    except ImportError as exc:
        raise ImportError('Install click: pip install "click>=8"') from exc
    try:
        import keyring
    except ImportError as exc:
        raise ImportError("Install keyring: pip install keyring") from exc

    @click.command("nvitk-xnat-keyring-set")
    @click.option("--server", type=str, required=True, help="XNAT server URL (same as in config).")
    def cli(server: str) -> None:
        pw = getpass.getpass("Password: ")
        keyring.set_password(KEYRING_SERVICE, f"xnat:{server}", pw)
        click.echo(f"Stored password for {KEYRING_SERVICE} / xnat:{server}")

    cli()
