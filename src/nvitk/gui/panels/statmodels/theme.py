"""
Dark theme for the Statmodels explorer.

Description
-----------
The explorer runs as a floating window that must stay legible next to Napari's dark chrome, so it
carries its own palette and stylesheet rather than inheriting the host application's. Every widget
family the explorer uses needs an explicit rule: ``QMenu``, ``QDialog`` and ``QToolTip`` are
top-level windows that otherwise fall back to the OS palette and render light on a dark page.

Matplotlib figures go the other way — :func:`whiten_figure` forces a white canvas, so plots stay
readable and export cleanly regardless of the surrounding chrome.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QWidget

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
COLOR_WINDOW = "#2b2b2b"
COLOR_BASE = "#1e1e1e"
COLOR_TEXT = "#e0e0e0"
COLOR_MUTED = "#a0a0a0"
COLOR_BORDER = "#555"
COLOR_ACCENT = "#3d6ea5"
COLOR_WARN = "#e5a25b"
COLOR_ERROR = "#e06c6c"

# Backgrounds for the p-value column of the coefficient table, dark enough to keep white text
# readable while still ranking at a glance.
SIGNIFICANCE_COLORS: dict[str, str] = {
    "***": "#1e4620",
    "**": "#2a5a2c",
    "*": "#3a6b34",
    ".": "#4a4a2a",
}

DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #555;
    border-radius: 4px;
    margin-top: 7px;
    padding-top: 6px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QListWidget, QTableWidget, QTableView,
QSpinBox, QDoubleSpinBox, QAbstractItemView {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #555;
    selection-background-color: #3d6ea5;
    selection-color: #ffffff;
}
QComboBox QAbstractItemView {
    border: 1px solid #666;
}
QPushButton, QToolButton {
    background-color: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #666;
    padding: 4px 10px;
    border-radius: 3px;
}
QPushButton:hover, QToolButton:hover {
    background-color: #4a4a4a;
}
QPushButton:pressed, QToolButton:pressed, QPushButton:checked, QToolButton:checked {
    background-color: #3d6ea5;
}
QPushButton:disabled, QToolButton:disabled {
    color: #808080;
    border-color: #444;
}
QHeaderView::section {
    background-color: #333;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 4px;
}
QTableView {
    gridline-color: #444;
    alternate-background-color: #262626;
}
QTabWidget::pane {
    border: 1px solid #555;
    border-radius: 3px;
    top: -1px;
}
QTabBar::tab {
    background-color: #333;
    color: #c8c8c8;
    border: 1px solid #555;
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    padding: 5px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #2b2b2b;
    color: #ffffff;
    border-color: #3d6ea5;
}
QTabBar::tab:hover:!selected {
    background-color: #3d3d3d;
}
QMenu {
    background-color: #2f2f2f;
    color: #e0e0e0;
    border: 1px solid #666;
}
QMenu::item {
    padding: 5px 22px 5px 22px;
}
QMenu::item:selected {
    background-color: #3d6ea5;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #808080;
}
QMenu::separator {
    height: 1px;
    background-color: #555;
    margin: 4px 8px;
}
QDialog {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
QToolTip {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #666;
    padding: 3px;
}
QProgressBar {
    background-color: #1e1e1e;
    border: 1px solid #555;
    border-radius: 3px;
    text-align: center;
    color: #e0e0e0;
}
QProgressBar::chunk {
    background-color: #3d6ea5;
    border-radius: 2px;
}
QSplitter::handle {
    background-color: #444;
}
QScrollArea {
    border: none;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #262626;
    border: none;
}
QScrollBar:vertical { width: 12px; }
QScrollBar:horizontal { height: 12px; }
QScrollBar::handle {
    background-color: #4d4d4d;
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:hover {
    background-color: #5d5d5d;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px;
    width: 0px;
}
QCheckBox, QRadioButton, QLabel {
    background: transparent;
}
"""


def apply_dark_theme(widget: QWidget) -> None:
    """Apply the explorer's dark Qt palette and stylesheet to *widget*."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(38, 38, 38))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(58, 58, 58))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Highlight, QColor(61, 110, 165))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    widget.setPalette(palette)
    widget.setStyleSheet(DARK_STYLESHEET)


def muted_label_style() -> str:
    """Stylesheet for secondary/hint text (grey, non-bold even inside a group box)."""
    return f"color: {COLOR_MUTED}; font-weight: normal;"


def whiten_figure(fig: Any) -> None:
    """Force a white figure/axes background with dark text, so plots stay readable against the
    explorer's dark chrome and export cleanly."""
    fig.patch.set_facecolor("white")
    # Grouped displays carry a suptitle, which belongs to the figure rather than to any axes.
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_color("#111111")
    for ax in fig.axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.tick_params(colors="#222222", which="both")
        ax.xaxis.label.set_color("#111111")
        ax.yaxis.label.set_color("#111111")
        ax.title.set_color("#111111")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("#999999")
            for text in legend.get_texts():
                text.set_color("#111111")


__all__ = [
    "COLOR_ACCENT",
    "COLOR_BASE",
    "COLOR_BORDER",
    "COLOR_ERROR",
    "COLOR_MUTED",
    "COLOR_TEXT",
    "COLOR_WARN",
    "COLOR_WINDOW",
    "DARK_STYLESHEET",
    "SIGNIFICANCE_COLORS",
    "apply_dark_theme",
    "muted_label_style",
    "whiten_figure",
]
