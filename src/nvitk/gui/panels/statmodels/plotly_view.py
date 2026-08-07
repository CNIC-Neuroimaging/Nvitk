"""
Interactive plot surface — a Plotly figure embedded in a Qt web view.

Description
-----------
Matplotlib renders a picture; reading a value off it means squinting at an axis. These plots are
about *which subject* and *what number*, so the figure has to answer questions on hover: this point
is subject ``sub-0142``'s left MCA at 146 mL/min, this line has slope −0.31 per year, this marginal
mean is 0.82 ± 0.04.

Embedding
---------
Plotly emits HTML plus a ~4.7 MB JavaScript bundle. Inlining that per figure would push megabytes
through the web view on every redraw, and Qt's ``setHtml`` caps at 2 MB regardless. Instead the
bundle is written **once** into a per-session directory and each figure is a small HTML file loaded
from a local URL beside it.

Static export still goes through Kaleido, so *Export PNG* produces the same figure as a file.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import tempfile
import uuid
from pathlib import Path
from typing import Any

from qtpy.QtCore import QUrl
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from nvitk.core.logger import Logger

from .theme import COLOR_ERROR, muted_label_style

log = Logger()

#: Where the shared plotly bundle and the per-figure HTML files live for this session.
_ASSET_DIR: Path | None = None


def prepare_webengine() -> bool:
    """
    Make Qt ready to host a web view. **Call before constructing the QApplication.**

    Qt refuses to import ``QtWebEngineWidgets`` once a ``QCoreApplication`` exists unless
    ``AA_ShareOpenGLContexts`` was set first — the web engine needs a shared GL context, and that
    can only be arranged before the application object is built. Since the Statmodels window is
    created long after the app starts, the flag has to be set in the entry point rather than here.

    Returns ``True`` when the web engine is usable. Never raises: a missing web engine degrades the
    plots, it does not stop the application from starting.
    """
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        if QApplication.instance() is None:
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        # Importing before the application exists is what actually registers the engine.
        import qtpy.QtWebEngineWidgets  # noqa: F401

        return True
    except Exception as exc:
        log.debug("QtWebEngine is not available: %s", exc)
        return False


WEBENGINE_HINT = (
    "Interactive plots need Qt's web engine. Install it with:\n"
    "    pip install PyQt6-WebEngine\n"
    "and relaunch — it has to be initialised before the application window is created."
)


def asset_dir() -> Path:
    """
    Session-scoped directory holding ``plotly.min.js`` and the rendered figures.

    One directory for the whole session so the bundle is written once; the OS reclaims it, and
    nothing here is worth persisting between runs.
    """
    global _ASSET_DIR
    if _ASSET_DIR is None or not _ASSET_DIR.exists():
        _ASSET_DIR = Path(tempfile.mkdtemp(prefix="nvitk-statmodels-plots-"))
        log.debug("Plotly asset directory: %s", _ASSET_DIR)
    return _ASSET_DIR


class PlotlyView(QWidget):
    """
    Host for one interactive figure.

    Mirrors the matplotlib canvas host it replaces — :meth:`show_figure`, :meth:`show_error`,
    :meth:`clear`, :meth:`save_figure` — so the plot pane does not need to know which backend drew
    what. Plotly owns zoom, pan and the mode bar, which is why this has no axis-limit sliders.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the web view and its placeholder."""
        super().__init__(parent)
        self._figure: Any = None
        self._view: Any = None
        self._html_path: Path | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder = QLabel("Interactive plots appear here after fitting.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(muted_label_style())
        self._layout.addWidget(self._placeholder)

    # ---- construction ---------------------------------------------------------
    def _ensure_view(self) -> Any:
        """Create the web view on first use — it is expensive, and many sessions never plot."""
        if self._view is not None:
            return self._view
        try:
            from qtpy.QtWebEngineWidgets import QWebEngineView
        except ImportError as exc:
            raise RuntimeError(
                f"{WEBENGINE_HINT}\n\n({exc})"
            ) from exc

        self._placeholder.setVisible(False)
        self._view = QWebEngineView(self)
        self._layout.addWidget(self._view, stretch=1)
        return self._view

    # ---- figure ---------------------------------------------------------------
    def show_figure(self, figure: Any) -> None:
        """Render *figure* (a ``plotly.graph_objects.Figure``) into the view."""
        import plotly.io as pio

        view = self._ensure_view()
        self._figure = figure
        directory = asset_dir()
        # A fresh filename per redraw: the web view caches aggressively, and reusing one path
        # shows the previous figure until something forces a reload.
        path = directory / f"figure-{uuid.uuid4().hex}.html"
        pio.write_html(
            figure,
            file=str(path),
            include_plotlyjs="directory",
            full_html=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        )
        self._discard_previous()
        self._html_path = path
        view.load(QUrl.fromLocalFile(str(path)))
        view.setVisible(True)

    def _discard_previous(self) -> None:
        """Delete the previously rendered HTML; the shared JS bundle stays."""
        if self._html_path is None:
            return
        try:
            self._html_path.unlink(missing_ok=True)
        except OSError as exc:
            log.debug("Could not remove the previous figure file: %s", exc)
        self._html_path = None

    def clear(self) -> None:
        """Drop the current figure and go back to the placeholder."""
        self._discard_previous()
        self._figure = None
        if self._view is not None:
            self._view.setVisible(False)
        self._placeholder.setText("Interactive plots appear here after fitting.")
        self._placeholder.setStyleSheet(muted_label_style())
        self._placeholder.setVisible(True)

    def show_error(self, message: str) -> None:
        """Replace the figure with an error message."""
        self.clear()
        self._placeholder.setText(message)
        self._placeholder.setStyleSheet(f"color: {COLOR_ERROR}; font-weight: normal;")

    def has_figure(self) -> bool:
        """Whether a figure is currently displayed."""
        return self._figure is not None

    def figure(self) -> Any:
        """The displayed figure, or ``None``."""
        return self._figure

    def save_figure(self, path: Any, *, scale: float = 2.0) -> Path:
        """
        Write the displayed figure to an image file, through Kaleido.

        The interactive state a user set in the browser — a zoom, a hidden trace — lives in the web
        view, not in the Python figure, so the export is the figure **as drawn**, not as currently
        panned. Use the mode bar's own camera button to capture the on-screen state instead.

        Raises
        ------
        RuntimeError
            When nothing is displayed, or Kaleido is not installed.
        """
        if self._figure is None:
            raise RuntimeError("There is no plot to export — fit a model first.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._figure.write_image(str(out), scale=scale)
        except Exception as exc:
            raise RuntimeError(
                f"Could not write the image ({exc}). Static export needs Kaleido: "
                f"pip install kaleido"
            ) from exc
        return out

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the legend, re-rendering in place."""
        if self._figure is None:
            return
        self._figure.update_layout(showlegend=bool(visible))
        self.show_figure(self._figure)


__all__ = ["PlotlyView", "asset_dir"]
