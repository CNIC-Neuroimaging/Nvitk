"""Shared visual language for every nvitk Qt surface.

One palette, one spacing scale, one stylesheet, and the handful of primitives
(:class:`Card`, :func:`chip`, :func:`kv_row`, …) that the panels compose. Panels
should reach for a token or a primitive here rather than spelling out a hex
value, so a change to the look lands everywhere at once.

The palette is deliberately low-contrast — near-neutral greys carrying the
structure, with saturated colour reserved for the few things that genuinely
signal something (an accent for selection, amber for a translation column, the
anatomical axis colours). Napari's own chrome is dark, and the docks sit inside
it, so this is a dark theme; every value is a token, so a light variant is a
matter of swapping this block rather than editing the panels.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
#: Page background, behind cards and panes.
COLOR_BG = "#2b2b2b"
#: Card / raised surface, one step lighter than the page.
COLOR_SURFACE = "#323232"
#: Input wells and tables, one step darker than the page.
COLOR_WELL = "#242424"
#: Control chrome (buttons, tabs, headers).
COLOR_CONTROL = "#3a3a3a"
COLOR_CONTROL_HOVER = "#464646"

COLOR_TEXT = "#e8e8e8"
COLOR_MUTED = "#9a9a9a"
COLOR_FAINT = "#6f6f6f"
COLOR_DISABLED = "#767676"

COLOR_BORDER = "#454545"
COLOR_BORDER_STRONG = "#585858"

COLOR_ACCENT = "#6fa8dc"
COLOR_ACCENT_DEEP = "#3d6ea5"
COLOR_OK = "#7bb47b"
COLOR_WARN = "#e5a25b"
COLOR_ERROR = "#e06c6c"

#: Anatomical direction → axis chip colour, so an orientation reads at a glance.
AXIS_COLORS: dict[str, str] = {
    "R": COLOR_ERROR,
    "L": COLOR_ERROR,
    "A": COLOR_OK,
    "P": COLOR_OK,
    "S": COLOR_ACCENT,
    "I": COLOR_ACCENT,
}

#: Backgrounds for a pass / warn / fail status chip or table row: dark enough to
#: keep light text readable, saturated enough to rank at a glance.
STATUS_COLORS: dict[str, str] = {
    "ok": "#1e4620",
    "warn": "#5a4a1e",
    "bad": "#5a2424",
    "neutral": COLOR_BG,
    "unknown": "#333333",
}

#: Backgrounds for a coefficient table's significance column, dark enough to keep
#: light text readable while still ranking at a glance.
SIGNIFICANCE_COLORS: dict[str, str] = {
    "***": "#1e4620",
    "**": "#2a5a2c",
    "*": "#3a6b34",
    ".": "#4a4a2a",
}

# ──────────────────────────────────────────────────────────────────────────────
# Spacing
# ──────────────────────────────────────────────────────────────────────────────
#: Gap between tightly related controls on one row.
SPACE_TIGHT = 6
#: Standard gap between sibling widgets.
SPACE = 10
#: Gap between distinct groups / cards.
SPACE_LOOSE = 14
#: Inner padding of a card or pane.
PAD = 12
RADIUS = 5

# ──────────────────────────────────────────────────────────────────────────────
# Stylesheet
# ──────────────────────────────────────────────────────────────────────────────
STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
}}
QLabel, QCheckBox, QRadioButton {{
    background: transparent;
}}
QGroupBox {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS}px;
    margin-top: 14px;
    padding: {PAD}px;
    /* Normal weight: a group box's font is inherited by every child, so bolding
       the box bolds all its content. Only the title is bold. */
    font-weight: normal;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {COLOR_MUTED};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox,
QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView, QAbstractItemView {{
    background-color: {COLOR_WELL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    selection-background-color: {COLOR_ACCENT_DEEP};
    selection-color: #ffffff;
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    padding: 4px 6px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLOR_ACCENT};
}}
QComboBox {{
    background-color: {COLOR_CONTROL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 18px;
}}
QComboBox:hover {{
    background-color: {COLOR_CONTROL_HOVER};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_WELL};
    border: 1px solid {COLOR_BORDER_STRONG};
    selection-background-color: {COLOR_ACCENT_DEEP};
    outline: none;
}}
QPushButton, QToolButton {{
    background-color: {COLOR_CONTROL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 5px 12px;
}}
QPushButton:hover, QToolButton:hover {{
    background-color: {COLOR_CONTROL_HOVER};
    border-color: {COLOR_BORDER_STRONG};
}}
QPushButton:pressed, QToolButton:pressed,
QPushButton:checked, QToolButton:checked {{
    background-color: {COLOR_ACCENT_DEEP};
    border-color: {COLOR_ACCENT};
    color: #ffffff;
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {COLOR_DISABLED};
    border-color: #3a3a3a;
    background-color: #303030;
}}
QToolButton::menu-indicator {{
    subcontrol-position: right center;
    subcontrol-origin: padding;
    right: 4px;
}}
QHeaderView::section {{
    background-color: {COLOR_CONTROL};
    color: {COLOR_MUTED};
    border: none;
    border-right: 1px solid {COLOR_BORDER};
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 5px 6px;
    font-weight: bold;
}}
/* Deliberately no ``::item {{ background-color }}`` rule: a stylesheet rule on
   ::item overrides QTableWidgetItem.setBackground, which silently disables
   per-cell tinting (the QC review table paints its pass/fail rows that way). */
QTableView, QTableWidget {{
    gridline-color: {COLOR_BORDER};
    alternate-background-color: #2a2a2a;
}}
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS}px;
    top: -1px;
    background-color: {COLOR_BG};
}}
QTabBar::tab {{
    background: transparent;
    color: {COLOR_MUTED};
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    padding: 6px 14px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    color: {COLOR_TEXT};
    border-bottom: 2px solid {COLOR_ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {COLOR_TEXT};
    background-color: {COLOR_SURFACE};
}}
QTabBar::tab:disabled {{
    color: {COLOR_FAINT};
}}
QMenu {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 4px;
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 20px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {COLOR_ACCENT_DEEP};
    color: #ffffff;
}}
QMenu::item:disabled {{
    color: {COLOR_DISABLED};
}}
QMenu::separator {{
    height: 1px;
    background-color: {COLOR_BORDER};
    margin: 4px 8px;
}}
QDialog {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
}}
QMainWindow::separator {{
    background-color: {COLOR_BG};
    width: {SPACE}px;
    height: {SPACE}px;
}}
QMainWindow::separator:hover {{
    background-color: {COLOR_ACCENT_DEEP};
}}
QDockWidget {{
    color: {COLOR_MUTED};
    font-size: 10px;
    font-weight: bold;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background-color: {COLOR_CONTROL};
    border: 1px solid {COLOR_BORDER};
    border-top-left-radius: {RADIUS}px;
    border-top-right-radius: {RADIUS}px;
    padding: 6px 10px;
    text-align: left;
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 0px;
    icon-size: 12px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {COLOR_CONTROL_HOVER};
    border-radius: 3px;
}}
QMenuBar {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    border-bottom: 1px solid {COLOR_BORDER};
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background-color: {COLOR_CONTROL};
}}
QStatusBar {{
    background-color: {COLOR_BG};
    color: {COLOR_MUTED};
    border-top: 1px solid {COLOR_BORDER};
}}
QStatusBar::item {{
    border: none;
}}
QToolTip {{
    background-color: {COLOR_WELL};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER_STRONG};
    padding: 4px 6px;
}}
QProgressBar {{
    background-color: {COLOR_WELL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    text-align: center;
    color: {COLOR_TEXT};
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT_DEEP};
    border-radius: 3px;
}}
QSplitter::handle {{
    background-color: transparent;
}}
QSplitter::handle:horizontal {{
    width: {SPACE}px;
}}
QSplitter::handle:vertical {{
    height: {SPACE}px;
}}
QSplitter::handle:hover {{
    background-color: {COLOR_ACCENT_DEEP};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    border: none;
    margin: 0;
}}
QScrollBar:vertical {{ width: 11px; }}
QScrollBar:horizontal {{ height: 11px; }}
QScrollBar::handle {{
    background-color: #4d4d4d;
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:hover {{
    background-color: #5f5f5f;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0px;
    width: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
}}
QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {{
    background-color: {COLOR_WELL};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 3px;
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border: 1px solid {COLOR_ACCENT};
    border-radius: 3px;
}}
QRadioButton::indicator {{
    border-radius: 7px;
}}
"""


#: Id under which the nvitk palette is registered with Napari.
NAPARI_THEME_ID = "nvitk"


def napari_theme_colors() -> dict[str, str]:
    """The nvitk palette expressed in Napari's theme vocabulary.

    Napari paints its own chrome — the layer list, the dims sliders, the menus,
    the area around the canvas — from a registered theme rather than from a
    stylesheet, so matching it to the docks means restating the same tokens in
    its field names rather than styling those widgets ourselves.
    """
    return {
        "id": NAPARI_THEME_ID,
        "label": "nvitk",
        "syntax_style": "native",
        # The canvas stays true black: it is image data, not chrome, and a tinted
        # surround shifts how the intensities in it read.
        "canvas": "black",
        "console": COLOR_WELL,
        "background": COLOR_BG,
        "foreground": COLOR_CONTROL,
        "primary": COLOR_BORDER_STRONG,
        "secondary": COLOR_MUTED,
        "highlight": COLOR_CONTROL_HOVER,
        "text": COLOR_TEXT,
        "icon": "#cfcfcf",
        "warning": COLOR_WARN,
        "error": COLOR_ERROR,
        "current": COLOR_ACCENT_DEEP,
        "font_size": "9pt",
    }


def register_napari_theme() -> str:
    """Register the nvitk palette as a Napari theme and return its id.

    Returns Napari's own ``"dark"`` if registration fails, so a Napari whose theme
    API has moved falls back to a sane theme instead of breaking startup.
    """
    try:
        from napari.utils.theme import available_themes, register_theme

        if NAPARI_THEME_ID not in available_themes():
            register_theme(NAPARI_THEME_ID, napari_theme_colors(), "nvitk")
        return NAPARI_THEME_ID
    except Exception:
        return "dark"


def apply_theme(widget: QWidget) -> None:
    """Apply the nvitk palette and stylesheet to *widget* and its children.

    Top-level windows need the ``QPalette`` too: menus, dialogs and tooltips are
    separate native windows that otherwise fall back to the OS palette and render
    light-on-dark.
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLOR_BG))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base, QColor(COLOR_WELL))
    palette.setColor(QPalette.AlternateBase, QColor("#2a2a2a"))
    palette.setColor(QPalette.Text, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button, QColor(COLOR_CONTROL))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ToolTipBase, QColor(COLOR_WELL))
    palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight, QColor(COLOR_ACCENT_DEEP))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    widget.setPalette(palette)
    widget.setStyleSheet(STYLESHEET)


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────
def mono_font(size: int = 10) -> QFont:
    """Monospace font at *size* points, for numbers that should stay in columns."""
    font = QFont("Monospace")
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(size)
    return font


def muted_label_style() -> str:
    """Stylesheet for secondary / hint text (grey, non-bold even inside a group box)."""
    return f"color: {COLOR_MUTED}; font-weight: normal;"


def fmt_number(value: float | None, digits: int = 4) -> str:
    """Format a float compactly, dropping trailing zeros; ``—`` when unknown."""
    if value is None:
        return "—"
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def clear_layout(layout: Any) -> None:
    """Empty *layout*, unparenting each widget so it stops painting immediately.

    ``deleteLater`` alone defers destruction to the next event-loop pass, during
    which the old widgets still draw over the newly built ones.
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                clear_layout(child)


class Card(QFrame):
    """Titled container grouping one family of related information."""

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        """Build a bordered card, with *title* as a small caps heading when given."""
        super().__init__(parent)
        self.setObjectName("card")
        # Scoped to #card: a bare ``QFrame`` rule would cascade the border onto
        # every child container and box each row.
        self.setStyleSheet(
            f"QFrame#card {{ background-color: {COLOR_SURFACE};"
            f" border: 1px solid {COLOR_BORDER}; border-radius: {RADIUS}px; }}"
            " QWidget { background: transparent; border: none; }"
        )
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(PAD, PAD - 4, PAD, PAD - 2)
        self._root.setSpacing(SPACE_TIGHT)
        if title:
            self._root.addWidget(section_heading(title))

    def add(self, widget: QWidget) -> None:
        """Append *widget* to the card body."""
        self._root.addWidget(widget)

    def add_layout(self, layout: Any) -> None:
        """Append a nested *layout* to the card body."""
        self._root.addLayout(layout)

    def body(self) -> QVBoxLayout:
        """The card's body layout, for callers that need finer control."""
        return self._root


def section_heading(text: str) -> QLabel:
    """Small caps heading that titles a card or a group of controls."""
    label = QLabel(text.upper())
    label.setStyleSheet(
        f"color: {COLOR_MUTED}; font-size: 10px; font-weight: bold;"
        " letter-spacing: 1px; border: none;"
    )
    return label


def chip(text: str, color: str = COLOR_ACCENT) -> QLabel:
    """Small rounded badge label in *color*."""
    label = QLabel(text)
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet(
        f"color: {color}; border: 1px solid {color}; border-radius: 3px;"
        " padding: 1px 6px; font-weight: bold; font-size: 10px;"
    )
    label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
    return label


def cell(
    text: str,
    *,
    color: str = COLOR_TEXT,
    mono: bool = False,
    align: Any = Qt.AlignLeft,
) -> QLabel:
    """One grid cell as a styled, selectable label."""
    label = QLabel(text)
    label.setStyleSheet(f"color: {color}; border: none;")
    if mono:
        label.setFont(mono_font())
    label.setAlignment(align | Qt.AlignVCenter)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def measure(
    value: str,
    unit: str = "mm",
    *,
    color: str = COLOR_TEXT,
    align: Any = Qt.AlignRight,
) -> QLabel:
    """A number with its unit set in muted type, so the figures stay dominant."""
    label = QLabel(f"{value}<span style='color:{COLOR_MUTED};font-size:10px;'> {unit}</span>")
    label.setTextFormat(Qt.RichText)
    label.setStyleSheet(f"color: {color}; border: none;")
    label.setFont(mono_font())
    label.setAlignment(align | Qt.AlignVCenter)
    return label


def column_heading(text: str, align: Any = Qt.AlignLeft) -> QLabel:
    """Column heading for a property grid, underlined to separate it from the rows."""
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {COLOR_MUTED}; border: none; border-bottom: 1px solid {COLOR_BORDER};"
        " padding-bottom: 3px; font-size: 10px; font-weight: bold;"
    )
    label.setAlignment(align | Qt.AlignVCenter)
    return label


def kv_row(
    label: str,
    value: str,
    *,
    mono: bool = True,
    color: str = COLOR_TEXT,
    key_width: int = 96,
) -> QWidget:
    """A single ``label: value`` line with the keys aligned in a column."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(SPACE)
    key = cell(label, color=COLOR_MUTED)
    key.setMinimumWidth(key_width)
    row.addWidget(key)
    row.addWidget(cell(value, color=color, mono=mono), stretch=1)
    return holder


def matrix_grid(matrix: Any, *, digits: int = 4, mark_last_column: bool = False) -> QWidget:
    """Render *matrix* as an aligned numeric grid.

    With *mark_last_column*, the final column of a homogeneous transform is
    tinted so the translation reads apart from the rotation/scale block.
    """
    import numpy as np

    holder = QWidget()
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(3)
    arr = np.asarray(matrix, dtype=float)
    rows, cols = arr.shape[0], arr.shape[1]
    for r in range(rows):
        for c in range(cols):
            tinted = mark_last_column and c == cols - 1 and r < rows - 1
            grid.addWidget(
                cell(
                    fmt_number(arr[r, c], digits),
                    color=COLOR_WARN if tinted else COLOR_TEXT,
                    mono=True,
                    align=Qt.AlignRight,
                ),
                r,
                c,
            )
    for c in range(cols):
        grid.setColumnStretch(c, 1)
    return holder


def style_figure(fig: Any) -> None:
    """Force a white Matplotlib canvas with dark text.

    Plots keep a light background regardless of the surrounding chrome, so they
    stay readable and export cleanly into papers and reports.
    """
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
    "AXIS_COLORS",
    "COLOR_ACCENT",
    "COLOR_ACCENT_DEEP",
    "COLOR_BG",
    "COLOR_BORDER",
    "COLOR_BORDER_STRONG",
    "COLOR_CONTROL",
    "COLOR_DISABLED",
    "COLOR_ERROR",
    "COLOR_FAINT",
    "COLOR_MUTED",
    "COLOR_OK",
    "COLOR_SURFACE",
    "COLOR_TEXT",
    "COLOR_WARN",
    "COLOR_WELL",
    "PAD",
    "RADIUS",
    "SIGNIFICANCE_COLORS",
    "STATUS_COLORS",
    "SPACE",
    "SPACE_LOOSE",
    "SPACE_TIGHT",
    "NAPARI_THEME_ID",
    "STYLESHEET",
    "Card",
    "apply_theme",
    "cell",
    "chip",
    "clear_layout",
    "column_heading",
    "fmt_number",
    "kv_row",
    "matrix_grid",
    "measure",
    "mono_font",
    "muted_label_style",
    "napari_theme_colors",
    "register_napari_theme",
    "section_heading",
    "style_figure",
]
