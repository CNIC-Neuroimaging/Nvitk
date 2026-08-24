"""``nvitk-config`` — inspect and scaffold nvitk's configuration.

nvitk reads its site settings from three JSON files (``sge.json``, ``settings.json``,
``xnat.json``) found in a configuration directory. This command answers the three questions a
new install raises: *where does that directory go*, *which file am I actually using*, and
*what do I have to fill in*.

.. code-block:: bash

    nvitk-config path              # where nvitk looks, and what it found
    nvitk-config init              # create a starter config from the bundled templates
    nvitk-config show              # resolved values, and which file each came from
    nvitk-config validate          # report settings that are missing or still placeholders
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

from nvitk.core import config_paths

#: Templates shipped inside the package, so ``init`` works from an installed distribution
#: rather than only from a source checkout.
TEMPLATE_DIR = Path(__file__).resolve().parent / "config_templates"

CONFIG_FILES = ("sge.json", "settings.json", "xnat.json")

#: A value that is still the shape the template shipped with, i.e. not yet filled in.
_PLACEHOLDER_PREFIX = "<"


def _iter_settings(doc: object, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a parsed config into ``(dotted.key, value)`` pairs, skipping ``_comment`` keys."""
    out: list[tuple[str, object]] = []
    if isinstance(doc, dict):
        for key, value in doc.items():
            if str(key).startswith("_comment"):
                continue
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.extend(_iter_settings(value, dotted))
            else:
                out.append((dotted, value))
    return out


def _placeholders(doc: object) -> list[tuple[str, object]]:
    """Settings whose value is still a ``<PLACEHOLDER>`` from the template."""
    return [
        (key, value)
        for key, value in _iter_settings(doc)
        if isinstance(value, str) and value.strip().startswith(_PLACEHOLDER_PREFIX)
    ]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Inspect and scaffold nvitk's configuration files."""


@main.command("path")
def path_cmd() -> None:
    """Print every directory nvitk searches, marking the one in use."""
    active = config_paths.config_dir()
    click.echo("Configuration directories, highest precedence first:\n")
    for directory, why in config_paths.candidate_dirs():
        if directory.is_dir():
            mark = "*" if directory == active else " "
            state = "exists"
        else:
            mark = " "
            state = "not present"
        click.echo(f" {mark} {str(directory):<55} {state:<12} [{why}]")

    click.echo("")
    if active is None:
        click.echo("No configuration directory found. Run `nvitk-config init` to create one.")
        return
    click.echo(f"Using: {active}")
    for name in CONFIG_FILES:
        found = config_paths.config_file(name)
        click.echo(f"  {name:<16} {found if found is not None else '-- not found --'}")


@main.command("init")
@click.option(
    "--dir", "target_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to create the configuration in (default: the standard user config dir).",
)
@click.option("--force", is_flag=True, help="Overwrite files that already exist.")
@click.option(
    "--only",
    type=click.Choice(CONFIG_FILES),
    multiple=True,
    help="Only create these files (repeatable). Default: all three.",
)
def init_cmd(target_dir: Path | None, force: bool, only: tuple[str, ...]) -> None:
    """Create a starter configuration from the bundled templates."""
    if target_dir is None:
        target_dir = config_paths.default_config_dir()
    target_dir = Path(target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    wanted = only or CONFIG_FILES
    written: list[Path] = []
    skipped: list[Path] = []

    for name in wanted:
        template = TEMPLATE_DIR / name.replace(".json", ".example.json")
        if not template.is_file():
            raise click.ClickException(f"Bundled template missing: {template}")
        destination = target_dir / name
        if destination.exists() and not force:
            skipped.append(destination)
            continue
        shutil.copyfile(template, destination)
        written.append(destination)

    for path in written:
        click.echo(f"created  {path}")
    for path in skipped:
        click.echo(f"exists   {path}  (use --force to overwrite)")

    if written:
        click.echo(
            "\nEdit the files above and replace every <PLACEHOLDER>, then run "
            "`nvitk-config validate`."
        )
        if target_dir != config_paths.default_config_dir():
            click.echo(
                f"\n{target_dir} is not a default search location — point nvitk at it with:\n"
                f"  export NVITK_CONFIG_DIR={target_dir}\n"
                f"or pass --config-dir {target_dir} to any nvitk command."
            )


@main.command("show")
@click.option(
    "--file", "which",
    type=click.Choice(CONFIG_FILES),
    default=None,
    help="Show only this file. Default: all three.",
)
def show_cmd(which: str | None) -> None:
    """Print resolved settings and the file each came from."""
    names = (which,) if which else CONFIG_FILES
    for name in names:
        source = config_paths.config_file(name)
        click.echo(click.style(f"── {name}", bold=True))
        if source is None:
            click.echo(f"   not found (looked in: {config_paths.describe_search(name)})\n")
            continue
        click.echo(f"   source: {source}")
        doc = config_paths.load_json(name) if name != "xnat.json" else _read_any(source)
        settings = _iter_settings(doc)
        if not settings:
            click.echo("   (empty)\n")
            continue
        for key, value in settings:
            rendered = "null" if value is None else str(value)
            if key.rsplit(".", 1)[-1] in {"password", "user"}:
                rendered = "<redacted>"
            click.echo(f"   {key:<50} {rendered}")
        click.echo("")


def _read_any(path: Path) -> dict:
    """Parse a JSON config for display; ``{}`` if it is not readable JSON."""
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


@main.command("validate")
def validate_cmd() -> None:
    """Report configuration that is missing or still contains template placeholders."""
    problems = 0
    for name in CONFIG_FILES:
        source = config_paths.config_file(name)
        if source is None:
            click.echo(f"{name}: not found")
            continue
        doc = config_paths.load_json(name) if name != "xnat.json" else _read_any(source)
        pending = _placeholders(doc)
        if pending:
            problems += len(pending)
            click.echo(click.style(f"{name}: {len(pending)} unfilled placeholder(s)", fg="yellow"))
            for key, value in pending:
                click.echo(f"   {key:<50} {value}")
        else:
            click.echo(click.style(f"{name}: ok", fg="green"))

    if problems:
        click.echo(
            f"\n{problems} setting(s) still hold template placeholders. "
            "Anything you do not need can be deleted rather than filled in — "
            "nvitk reports a missing key by name when a command actually requires it."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
