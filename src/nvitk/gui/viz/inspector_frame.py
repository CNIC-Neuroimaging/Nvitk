"""One frame for every QVTPy inspection panel: identity, figure and table export, help.

The inspection panels (cross-sections, hemodynamics, cohort violins, CPR, morphometrics,
lesions, QC measurements, brain-vascular map) grew one at a time, each with its own title, its
own figure buttons (or none) and its own margins. Here they share one frame:

* the dock title bar carries the panel's icon and accent colour, and the same icon actions
  everywhere: save the figures (PNG / SVG / PDF), copy a figure to the clipboard, save the
  table shown (CSV / Excel), and a help tip saying what the panel shows and how to drive it;
* the body sits in an :class:`InspectorFrame`: an accent stripe at its left edge, the design
  spacing around it, and an info line under it that a panel may write to
  (:func:`set_inspector_info`).

The actions find what to export by walking the panel's widgets (Matplotlib canvases, table
views), so the panels need no changes to gain them; a panel can still offer its own figures or
tables through ``inspector_figures()`` / ``inspector_tables()`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout, QWidget

from nvitk.gui.core import design


@dataclass(frozen=True)
class InspectorKind:
    """How one inspection panel presents itself."""

    icon: str
    hue: str
    help: str


#: Dock object name -> identity. Unknown docks get :data:`_DEFAULT`.
INSPECTORS: dict[str, InspectorKind] = {
    "nvitk_vessel_cross_section_dock": InspectorKind(
        "vessel", "blue",
        "Oblique cross-section of the vessel at the centreline point you click in 3D, with the "
        "lumen mask and the flow waveforms of the neighbouring stations."),
    "nvitk_hemodynamics_plot_dock": InspectorKind(
        "wave", "red",
        "Hemodynamic parameters along the picked centreline: flow, velocity, area, pulsatility."),
    "nvitk_cohort_violin_dock": InspectorKind(
        "plot_box", "purple",
        "Where this subject's values fall in the cohort's distribution, per vessel segment."),
    "nvitk_vessel_cpr_dock": InspectorKind(
        "vessel", "cyan",
        "Curved planar reformation along a centreline path, with its cross-sections and profile."),
    "nvitk_morphometrics_dock": InspectorKind(
        "ruler", "green",
        "Vessel morphometrics: radius, tortuosity and lesions along each centreline segment."),
    "nvitk_morpho_lesion_dock": InspectorKind(
        "target", "orange",
        "Candidate lesions (stenoses, aneurysms) with their profile and where they sit."),
    "nvitk_qc_measurements_dock": InspectorKind(
        "qc", "olive",
        "The subject's QC measurements against the expected ranges, for review and sign-off."),
    "nvitk_brain_vasc_dock": InspectorKind(
        "atlas", "pink",
        "Brain regions mapped to the arteries that feed them, with per-region summaries."),
}

_DEFAULT = InspectorKind("inspect", "grey", "")


def inspector_kind(object_name: str) -> InspectorKind:
    """Identity of the dock named *object_name* (a neutral default for unknown ones)."""
    if object_name in INSPECTORS:
        return INSPECTORS[object_name]
    for key, kind in INSPECTORS.items():  # tolerate suffixed names (one dock per layer...)
        if object_name.startswith(key):
            return kind
    return _DEFAULT


class InspectorFrame(QFrame):
    """The body of an inspection dock: *panel* inside an accent-striped frame with an info line."""

    def __init__(self, panel: QWidget, *, object_name: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.kind = inspector_kind(object_name)
        self.setObjectName("nvitkInspectorFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(design.SPACE_TIGHT + 2, design.SPACE_TIGHT, design.SPACE_TIGHT, design.SPACE_TIGHT)
        layout.setSpacing(design.SPACE_TIGHT)
        layout.addWidget(panel, 1)
        self._info = QLabel("", self)
        self._info.setWordWrap(True)
        self._info.setVisible(False)
        layout.addWidget(self._info)
        panel._nvitk_inspector_frame = self
        self._restyle()
        design.add_theme_listener(self._restyle)

    def set_info(self, text: str) -> None:
        """Show *text* under the panel (empty hides the line)."""
        self._info.setText(str(text or ""))
        self._info.setVisible(bool(text))

    def _restyle(self) -> None:
        colour = design.accent(self.kind.hue) if self.kind.hue in design.ACCENTS else design.COLOR_BORDER
        self.setStyleSheet(
            f"QFrame#nvitkInspectorFrame {{ border: none; border-left: 2px solid {colour}; }}"
        )
        self._info.setStyleSheet(f"color: {design.COLOR_MUTED}; font-size: {design.FONT_S}px; border: none;")


def frame_of(panel: Any) -> InspectorFrame | None:
    """The :class:`InspectorFrame` around *panel*, if it has one."""
    return getattr(panel, "_nvitk_inspector_frame", None)


def set_inspector_info(panel: Any, text: str) -> None:
    """Write *text* on *panel*'s info line (no-op for a panel without a frame)."""
    frame = frame_of(panel)
    if frame is not None:
        frame.set_info(text)


# What a panel shows
def _panel(dock_or_widget: Any) -> Any:
    widget = dock_or_widget.widget() if hasattr(dock_or_widget, "widget") and callable(dock_or_widget.widget) else dock_or_widget
    return getattr(widget, "panel", widget)


def panel_figures(panel: Any) -> list[Any]:
    """The Matplotlib figures *panel* shows (visible canvases first, in reading order)."""
    own = getattr(panel, "inspector_figures", None)
    if callable(own):
        return [f for f in own() if f is not None]
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except Exception:  # noqa: BLE001
        return []
    canvases = [c for c in panel.findChildren(FigureCanvasQTAgg)]
    shown = [c for c in canvases if c.isVisibleTo(panel)] or canvases
    shown.sort(key=lambda c: (c.mapTo(panel, c.rect().topLeft()).y(), c.mapTo(panel, c.rect().topLeft()).x()))
    return [c.figure for c in shown if getattr(c, "figure", None) is not None and c.figure.axes]


def panel_tables(panel: Any) -> list[Any]:
    """The table views *panel* shows, largest first."""
    own = getattr(panel, "inspector_tables", None)
    if callable(own):
        return [t for t in own() if t is not None]
    from qtpy.QtWidgets import QTableView

    tables = [t for t in panel.findChildren(QTableView) if t.isVisibleTo(panel) and t.model() is not None
              and t.model().rowCount() > 0]
    tables.sort(key=lambda t: -(t.model().rowCount() * max(t.model().columnCount(), 1)))
    return tables


def table_frame(view: Any) -> Any:
    """The rows and columns *view* shows (hidden ones left out, sorted order kept) as a DataFrame."""
    import pandas as pd

    model = view.model()
    cols = [c for c in range(model.columnCount()) if not view.isColumnHidden(c)]
    headers = [str(model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or f"col{c}") for c in cols]
    rows = []
    for r in range(model.rowCount()):
        if view.isRowHidden(r):
            continue
        rows.append([model.data(model.index(r, c), Qt.DisplayRole) for c in cols])
    return pd.DataFrame(rows, columns=headers)


# Actions
def _status(panel: Any, text: str) -> None:
    frame = frame_of(panel)
    if frame is not None:
        frame.set_info(text)


def save_figures(panel: Any, parent: QWidget | None = None, *, path: str | None = None) -> list[Path]:
    """Save every figure *panel* shows; one file, or ``<stem>_<n>`` files for several."""
    figures = panel_figures(panel)
    if not figures:
        _status(panel, "This panel shows no figure yet.")
        return []
    if path is None:
        path, _ = QFileDialog.getSaveFileName(parent, "Save figure", "figure.png", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if not path:
            return []
    out = Path(path)
    if out.suffix.lower() not in (".png", ".svg", ".pdf"):
        out = out.with_suffix(".png")
    written = []
    for i, figure in enumerate(figures):
        target = out if len(figures) == 1 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        figure.savefig(target, dpi=200, bbox_inches="tight")
        written.append(target)
    _status(panel, f"Saved {len(written)} figure(s) to {out.parent}")
    return written


def copy_figure(panel: Any) -> bool:
    """Copy the first figure *panel* shows to the clipboard, as an image."""
    from qtpy.QtWidgets import QApplication

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except Exception:  # noqa: BLE001
        return False
    figures = panel_figures(panel)
    if not figures:
        _status(panel, "This panel shows no figure yet.")
        return False
    canvas = figures[0].canvas
    if isinstance(canvas, FigureCanvasQTAgg):
        # An image, not a pixmap: on X11 Qt hands the clipboard to the clipboard manager at
        # exit, after the application is gone, and a pixmap there aborted the interpreter
        # ("Must construct a QGuiApplication before a QPixmap").
        QApplication.clipboard().setImage(canvas.grab().toImage())
        _status(panel, "Figure copied to the clipboard.")
        return True
    return False


def save_table(panel: Any, parent: QWidget | None = None, *, path: str | None = None) -> Path | None:
    """Save the largest table *panel* shows to CSV or Excel."""
    from nvitk.gui.viz.table_export import export_table

    tables = panel_tables(panel)
    if not tables:
        _status(panel, "This panel shows no table.")
        return None
    if path is None:
        path, _ = QFileDialog.getSaveFileName(parent, "Save table", "table.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return None
    out = export_table(table_frame(tables[0]), path, title="Table")
    _status(panel, f"Saved {out.name}")
    return out


def title_bar_actions(dock: Any, kind: InspectorKind) -> list[QWidget]:
    """Icon actions for *dock*'s title bar; each acts on whatever panel the dock holds now."""
    from nvitk.gui.core.widgets import IconButton

    buttons = []
    for name, tip, fn in (
        ("save", "Save the panel's figures (PNG, SVG, PDF)", lambda: save_figures(_panel(dock), dock)),
        ("copy", "Copy the figure to the clipboard", lambda: copy_figure(_panel(dock))),
        ("table", "Save the panel's table (Excel, CSV)", lambda: save_table(_panel(dock), dock)),
    ):
        button = IconButton(name, tip, icon_size=13)
        button.setFixedSize(22, 22)
        button.clicked.connect(lambda _=False, f=fn: f())
        buttons.append(button)
    if kind.help:
        tip = IconButton("help", kind.help, icon_size=13)
        tip.setFixedSize(22, 22)
        tip.clicked.connect(lambda: _status(_panel(dock), kind.help))
        buttons.append(tip)
    return buttons


def title_icon(kind: InspectorKind) -> QLabel:
    """The panel's icon, in its accent, for the title bar."""
    from nvitk.gui.core.icons import pixmap

    label = QLabel()
    label.setFixedSize(16, 16)

    def restyle() -> None:
        colour = design.accent(kind.hue) if kind.hue in design.ACCENTS else design.COLOR_MUTED
        label.setPixmap(pixmap(kind.icon, 14, colour))
        label.setStyleSheet("background: transparent;")

    restyle()
    design.add_theme_listener(restyle)
    return label


__all__ = [
    "INSPECTORS",
    "InspectorFrame",
    "InspectorKind",
    "copy_figure",
    "frame_of",
    "inspector_kind",
    "panel_figures",
    "panel_tables",
    "save_figures",
    "save_table",
    "set_inspector_info",
    "table_frame",
    "title_bar_actions",
    "title_icon",
]
