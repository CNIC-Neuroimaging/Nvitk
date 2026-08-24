"""``nvitk-dataset`` — fetch the nvitk dataset with DVC.

The dataset itself is not distributed with nvitk: it is research data, and it lives on a
storage location only its owners can reach. What *is* public is the set of DVC pointer files
committed in the repository — small text files naming a content hash. ``dvc get`` reads those
straight out of the repository over HTTPS and then pulls the matching content from the
configured remote, so a conda install with no checkout can still retrieve the data.

Access control sits at the remote, not at the pointers: a hash reveals nothing, and someone
without access to the storage location simply cannot download the content it names.

.. code-block:: bash

    nvitk-dataset pull            # tables + catalog  (~19 MB)
    nvitk-dataset pull --all      # + the prebuilt SQLite index (~1.3 GB)
    nvitk-dataset status          # what is configured, and what is present locally
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import click

from nvitk.core import config_paths
from nvitk.db.settings_paths import load_db_settings_block

#: Default source repository holding the DVC pointer files.
DEFAULT_DVC_REPO = "https://github.com/ignacio-ms/Nvitk.git"

#: Path of the dataset inside that repository.
REPO_DATASET_PATH = "dataset/nvitk-dataset"

#: Targets, in the order they are fetched. ``heavy`` ones need ``--all``.
TARGETS: tuple[tuple[str, bool, str], ...] = (
    ("catalog", False, "manifests and schemas (~150 KB)"),
    ("tables", False, "measurement tables (~19 MB)"),
    ("cache", True, "prebuilt SQLite index (~1.3 GB, rebuildable in seconds)"),
)


def _setting(key: str, default: str = "") -> str:
    """A ``db.<key>`` value from ``settings.json``, or *default*."""
    raw = load_db_settings_block().get(key)
    return str(raw).strip() if raw is not None and str(raw).strip() else default


def _require_dvc() -> str:
    """Path to the ``dvc`` executable, or a :class:`click.ClickException` explaining how to get it."""
    found = shutil.which("dvc")
    if found is None:
        raise click.ClickException(
            "dvc is not installed.\n"
            "  conda install -c conda-forge dvc     (or: pip install dvc)"
        )
    return found


def _dataset_root() -> Path:
    """Where the dataset should land, from ``db.root``."""
    root = _setting("root")
    if not root:
        raise click.ClickException(
            "No dataset location configured.\n"
            f'Set "db.root" in settings.json — looked in: '
            f"{config_paths.describe_search('settings.json')}\n"
            "Run `nvitk-config init` if you have no configuration yet."
        )
    return Path(root).expanduser()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Fetch the nvitk dataset from its DVC remote."""


@main.command("pull")
@click.option("--all", "want_all", is_flag=True, help="Also fetch the prebuilt SQLite index (~1.3 GB).")
@click.option("--rev", default=None, help="Git revision of the pointers (tag/branch/commit). Default: the repo's default branch.")
@click.option("--force", is_flag=True, help="Overwrite targets that already exist locally.")
@click.option("--dry-run", is_flag=True, help="Print the dvc commands without running them.")
def pull_cmd(want_all: bool, rev: str | None, force: bool, dry_run: bool) -> None:
    """Download the dataset into ``db.root``."""
    dvc = _require_dvc()
    root = _dataset_root()
    repo = _setting("dvc_repo", DEFAULT_DVC_REPO)
    remote_url = _setting("dvc_remote_url")

    if not dry_run:
        # After the dry-run check: --dry-run must not touch the filesystem. A missing parent
        # here almost always means a network share is not mounted, which deserves a sentence
        # rather than a MemoryError-deep traceback.
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise click.ClickException(
                f"Cannot create the dataset directory {root}\n"
                f"  {exc.strerror}\n"
                'If "db.root" is on a network share, mount it first; otherwise point '
                '"db.root" somewhere writable.'
            ) from exc

    selected = [t for t in TARGETS if want_all or not t[1]]
    skipped = [t for t in TARGETS if not (want_all or not t[1])]

    click.echo(f"repository : {repo}")
    click.echo(f"destination: {root}")
    if remote_url:
        click.echo(f"remote     : {remote_url}  (from db.dvc_remote_url)")
    else:
        click.echo("remote     : as configured in the repository's .dvc/config")
    click.echo("")

    failures = 0
    for name, _heavy, description in selected:
        destination = root / name
        if destination.exists() and not force:
            click.echo(f"  {name:9} already present — skipping (use --force to replace)")
            continue

        command = [dvc, "get", repo, f"{REPO_DATASET_PATH}/{name}", "-o", str(destination)]
        if rev:
            command += ["--rev", rev]
        # Lets a user whose mount differs from the committed one override it without editing
        # the repository's .dvc/config.
        if remote_url:
            command += ["--remote-config", f"url={remote_url}"]

        if dry_run:
            click.echo("  " + " ".join(command))
            continue

        click.echo(f"  {name:9} {description}")
        result = subprocess.run(command)
        if result.returncode != 0:
            failures += 1
            click.echo(
                click.style(f"  {name:9} FAILED", fg="red")
                + " — is the storage location mounted, and do you have access to it?"
            )

    for name, _heavy, description in skipped:
        click.echo(f"\n  {name} not fetched ({description}). Use --all to include it, or rebuild it with:")
        click.echo(f"    python -m nvitk.db.sqlite_index --dataset-root {root}")

    if failures:
        raise SystemExit(1)


@main.command("status")
def status_cmd() -> None:
    """Show the configured source and destination, and what is present locally."""
    click.echo(f"repository      : {_setting('dvc_repo', DEFAULT_DVC_REPO)}")
    click.echo(f"remote override : {_setting('dvc_remote_url') or '(none — uses the repo default)'}")
    try:
        root = _dataset_root()
    except click.ClickException as exc:
        click.echo(f"destination     : {click.style('not configured', fg='yellow')}")
        click.echo(f"\n{exc.message}")
        return
    click.echo(f"destination     : {root}")
    click.echo(f"dvc             : {shutil.which('dvc') or click.style('not installed', fg='yellow')}")
    click.echo("")
    for name, heavy, description in TARGETS:
        present = (root / name).exists()
        mark = click.style("present", fg="green") if present else "missing"
        click.echo(f"  {name:9} {mark:18} {description}{'  [--all]' if heavy else ''}")


if __name__ == "__main__":
    main()
