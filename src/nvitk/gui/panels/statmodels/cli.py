"""
``nvitk-statsmodels`` — launch the Statmodels explorer as a standalone window.

Description
-----------
The same window the Napari right-tab launcher opens, without starting Napari. Useful when the work
is modeling rather than image inspection: the viewer, its GPU context and its plugin scan are a
substantial cost to pay for a statistics panel.

Examples
--------
.. code-block:: bash

    nvitk-statsmodels                          # the dataset from .nvitk/settings.json
    nvitk-statsmodels -d /data/pesa-brain      # a specific dataset root
    nvitk-statsmodels --kind asl               # open on a pipeline kind
    nvitk-statsmodels --load my_model          # restore a saved model's configuration
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import json
import sys
from pathlib import Path

from nvitk.core.logger import Logger

from .constants import PIPELINE_KIND_ITEMS, PIPELINE_KIND_QVTPY

log = Logger()


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the launcher."""
    parser = argparse.ArgumentParser(
        prog="nvitk-statsmodels",
        description=(
            "Open the NVITK Statmodels explorer: mixed models, GLMs, non-linear fits and "
            "mediation over 4D-flow, ASL, T1, FLAIR and TOF measurements."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models are saved under db.statmodels_root (see `nvitk-config show`). "
            "Use --load <name> to reopen one."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        metavar="PATH",
        help="Dataset root. Defaults to the one configured in .nvitk/settings.json.",
    )
    parser.add_argument(
        "-k",
        "--kind",
        choices=[key for _label, key in PIPELINE_KIND_ITEMS],
        default=PIPELINE_KIND_QVTPY,
        help="Pipeline kind to open on (default: %(default)s).",
    )
    parser.add_argument(
        "--load",
        metavar="NAME_OR_PATH",
        help=(
            "Restore a saved model configuration: a name under the configured model directory, or a "
            "path to a config.json."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Build the analysis frame immediately instead of waiting for 'Reload data'.",
    )
    parser.add_argument(
        "--config-dir",
        metavar="PATH",
        help=(
            "Directory holding nvitk's sge.json / settings.json / xnat.json "
            "(default: $NVITK_CONFIG_DIR, then ~/.config/nvitk). See `nvitk-config path`."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["NONE", "DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: %(default)s).",
    )
    return parser


def _resolve_config(path_or_name: str, repo_root: Path) -> Path:
    """Locate a saved ``config.json`` from a bare model name or an explicit path.

    Bare names resolve against the configured model directory, so ``--load`` looks wherever
    ``_on_save`` actually wrote — not at a second, independently-built guess.
    """
    candidate = Path(path_or_name)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        return candidate / "config.json"
    from .helpers import statmodels_root

    saved = statmodels_root() / path_or_name
    if saved.is_dir():
        return saved / "config.json"
    raise FileNotFoundError(
        f"No saved model {path_or_name!r} — looked in {saved} and as a literal path."
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``nvitk-statsmodels`` command."""
    # Before argparse and before the Qt/window imports below, so that anything reading
    # configuration as it is imported already sees the redirected directory.
    from nvitk.core.click_config import preparse_config_dir

    preparse_config_dir(argv)

    args = build_parser().parse_args(argv)
    Logger(level=args.log_level)

    # Qt is imported here rather than at module import so ``--help`` works on a headless box with no
    # Qt platform plugin available.
    from qtpy.QtWidgets import QApplication

    from .window import StatmodelsWindow

    if args.dataset:
        root = Path(args.dataset).expanduser().resolve()
        if not root.is_dir():
            log.error("Dataset root does not exist: %s", root)
            return 2
        _use_dataset(root)

    # Must happen before the QApplication exists: Qt will not load the web engine afterwards
    # without a shared GL context, and the interactive plots are hosted in one.
    from .plotly_view import prepare_webengine

    if not prepare_webengine():
        log.warning(
            "Qt's web engine is unavailable — interactive plots will not render. "
            "Install it with: pip install PyQt6-WebEngine"
        )
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = StatmodelsWindow(initial_pipeline_kind=args.kind)
    except Exception as exc:
        log.exception("Could not open the Statmodels window.")
        log.error(
            "Is a dataset configured? Pass --dataset PATH, or set one in .nvitk/settings.json. (%s)",
            exc,
        )
        return 1

    if args.load:
        try:
            config_path = _resolve_config(args.load, Path(window._repo.root))
            window._apply_config(json.loads(config_path.read_text(encoding="utf-8")))
            log.info("Loaded configuration from %s", config_path)
        except Exception as exc:
            log.error("Could not load %r: %s", args.load, exc)

    window.show_maximized_floating()
    if args.reload:
        window._on_reload()

    return int(app.exec())


def _use_dataset(root: Path) -> None:
    """
    Point the repo accessor at *root* for this process.

    The window resolves its dataset through ``nvitk.db.repo.get_repo_from_settings``; overriding that
    lookup is less invasive than threading a root through every panel, and it keeps ``--dataset`` a
    property of this launcher rather than of the window.
    """
    from nvitk.db.repo import DataRepo

    from . import window as window_module

    window_module.open_repo = lambda _root=root: DataRepo(_root)
    log.info("Using dataset %s", root)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
