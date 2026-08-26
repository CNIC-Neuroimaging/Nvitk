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
    """Where the dataset should land.

    Uses the *same* resolution as :class:`~nvitk.db.repo.DataRepo` rather than reading
    ``db.root`` directly — ``local_fallback_root`` takes precedence over ``root`` there, so
    resolving it independently here would download the data to one directory while the rest of
    nvitk read from another.
    """
    from nvitk.db.repo import _local_dataset_root_from_settings

    db = load_db_settings_block()
    if not db:
        raise click.ClickException(
            "No dataset location configured.\n"
            f'Set "db.root" in settings.json — looked in: '
            f"{config_paths.describe_search('settings.json')}\n"
            "Run `nvitk-config init` if you have no configuration yet."
        )
    if not (str(db.get("local_fallback_root") or "").strip() or str(db.get("root") or "").strip()):
        raise click.ClickException(
            'Neither "db.root" nor "db.local_fallback_root" is set in settings.json.\n'
            f"Looked in: {config_paths.describe_search('settings.json')}"
        )
    return Path(_local_dataset_root_from_settings(db)).expanduser()


def _entries(path: Path) -> list[Path]:
    """Visible entries directly under *path*, or ``[]`` when it is not a readable directory."""
    if not path.is_dir():
        return []
    try:
        return [p for p in path.iterdir() if not p.name.startswith(".")]
    except OSError:
        return []


def _file_count(path: Path) -> int:
    """Number of files anywhere under *path* (0 when it is missing or holds only directories)."""
    if not path.is_dir():
        return 0
    try:
        return sum(1 for f in path.rglob("*") if f.is_file() and not f.name.startswith("."))
    except OSError:
        return 0


def _is_populated(path: Path) -> bool:
    """True when *path* contains at least one file, anywhere beneath it.

    Counting *files* rather than entries matters: ``DatasetCatalog.create_scaffold`` leaves a
    ``catalog/schema/`` subdirectory behind, so an "is it empty" test based on directory
    entries calls that tree populated and skips the download — the user then sees a confident
    "already present" over an empty dataset.
    """
    return _file_count(path) > 0


def _nested_inside(path: Path, name: str) -> Path | None:
    """The ``path/name`` directory when *path* holds only that — the shape a nested fetch leaves.

    ``dvc get -o DIR`` places the fetched directory *inside* ``DIR`` when ``DIR`` already
    exists, producing ``tables/tables/…``. Detecting it lets the fetch repair itself and lets
    :func:`status_cmd` report a tree left broken by an earlier run.
    """
    inner = path / name
    entries = _entries(path)
    if inner.is_dir() and len(entries) == 1 and entries[0].name == name:
        return inner
    return None


def _describe(path: Path, name: str) -> tuple[str, str]:
    """``(state, detail)`` for *path*: what is there, and how much of it."""
    if not path.exists():
        return "missing", ""
    if not path.is_dir():
        return "not a directory", ""
    if _nested_inside(path, name) is not None:
        return "NESTED", f"contents are under {name}/{name}/ — re-run with --force"
    files = _file_count(path)
    if files == 0:
        return "empty", "no files — not yet downloaded"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return "present", f"{files} files, {total / 1e6:.1f} MB"


def _root_origin() -> str:
    """Which setting produced the destination, for output that can be sanity-checked."""
    if _setting("local_fallback_root"):
        return "from db.local_fallback_root"
    if _setting("root"):
        return "from db.root"
    return "default location — set db.local_fallback_root to choose another"


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
    click.echo(f"destination: {root}   ({_root_origin()})")
    if remote_url:
        click.echo(f"remote     : {remote_url}  (from db.dvc_remote_url)")
    else:
        click.echo("remote     : as configured in the repository's .dvc/config")
    click.echo("")

    failures = 0
    for name, _heavy, description in selected:
        destination = root / name
        if _is_populated(destination) and _nested_inside(destination, name) is None and not force:
            state, detail = _describe(destination, name)
            click.echo(f"  {name:9} already present ({detail}) — skipping, use --force to replace")
            continue

        # Fetch into a staging path that cannot already exist. `dvc get -o DIR` puts the
        # fetched directory *inside* DIR when DIR exists, which yields tables/tables/… and
        # leaves the data where nothing can find it. Staging also means a failed or partial
        # download never touches whatever is already in place.
        staging = root / f".{name}.incoming"
        command = [dvc, "get", repo, f"{REPO_DATASET_PATH}/{name}", "-o", str(staging)]
        if rev:
            command += ["--rev", rev]
        # Lets a user whose mount differs from the committed one override it without editing
        # the repository's .dvc/config.
        if remote_url:
            command += ["--remote-config", f"url={remote_url}"]

        if dry_run:
            click.echo("  " + " ".join(command))
            click.echo(f"  {'':9} then move {staging.name} -> {name}")
            continue

        click.echo(f"  {name:9} {description}")
        shutil.rmtree(staging, ignore_errors=True)
        result = subprocess.run(command)

        if result.returncode != 0 or not _is_populated(staging):
            failures += 1
            shutil.rmtree(staging, ignore_errors=True)
            click.echo(
                click.style(f"  {name:9} FAILED", fg="red")
                + " — is the storage location mounted, and do you have access to it?"
            )
            continue

        # Defensive: if a future DVC still nests despite the fresh path, flatten it rather
        # than handing back an unusable tree.
        inner = _nested_inside(staging, name)
        if inner is not None:
            lifted = root / f".{name}.lifted"
            shutil.rmtree(lifted, ignore_errors=True)
            inner.rename(lifted)
            shutil.rmtree(staging, ignore_errors=True)
            lifted.rename(staging)

        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        staging.rename(destination)

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
    click.echo(f"destination     : {root}   ({_root_origin()})")
    click.echo(f"dvc             : {shutil.which('dvc') or click.style('not installed', fg='yellow')}")
    click.echo("")
    problems = 0
    for name, heavy, description in TARGETS:
        state, detail = _describe(root / name, name)
        colour = {"present": "green", "missing": None, "empty": "yellow"}.get(state, "red")
        if state not in {"present", "missing"}:
            problems += 1
        label = click.style(state, fg=colour) if colour else state
        click.echo(f"  {name:9} {label:24} {detail or description}{'  [--all]' if heavy else ''}")

    if problems:
        click.echo(
            "\nSomething is only partly there. `nvitk-dataset pull --force` replaces a target "
            "cleanly; an empty directory is fetched by a plain `pull`."
        )


if __name__ == "__main__":
    main()
