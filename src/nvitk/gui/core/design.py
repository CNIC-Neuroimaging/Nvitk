"""Shared visual language for every nvitk Qt surface.

One palette, one spacing scale, one stylesheet, and the handful of primitives
(:class:`Card`, :func:`chip`, :func:`kv_row`, …) that the panels compose. Panels
should reach for a token or a primitive here rather than spelling out a hex
value, so a change to the look lands everywhere at once.

The palette is deliberately low-contrast — near-neutral greys carrying the
structure, with saturated colour reserved for the few things that genuinely
signal something (an accent for selection, amber for a translation column, the
anatomical axis colours).

Two themes share that structure. The constants below *are* the dark theme, the
one Napari's own chrome is built for; :data:`LIGHT_TOKENS` overrides each of them
for the light theme. :func:`toggle_theme` swaps between the two live — it rebinds
the tokens, rebuilds the stylesheet, rewrites the inline stylesheets already on
screen, and moves Napari's chrome with them. Reach for a token or a primitive
here rather than a hex value and a panel follows the theme for free.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
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

#: Every other row of a table, a shade off the well it sits in.
COLOR_ALT_ROW = "#2a2a2a"
#: Scrollbar handle, and its hover state. No track: the groove stays transparent.
COLOR_SCROLL = "#4d4d4d"
COLOR_SCROLL_HOVER = "#5f5f5f"
#: An unavailable button: filled a shade off the page so it still reads as a
#: control, but flatter than an enabled one.
COLOR_BUTTON_OFF = "#303030"
COLOR_BORDER_OFF = "#3a3a3a"

#: The one saturated colour in the interface: selection, focus, active state.
COLOR_ACCENT = "#ffa400"
#: Filled backgrounds behind light text (pressed buttons, selected rows), dark
#: enough that white stays legible on it. The same amber in both themes — it is
#: read against its own white text, not against the page.
COLOR_ACCENT_DEEP = "#b36f00"
#: Text and icons drawn on top of :data:`COLOR_ACCENT_DEEP`.
COLOR_ON_ACCENT = "#ffffff"
COLOR_OK = "#7bb47b"
COLOR_WARN = "#e5a25b"
COLOR_ERROR = "#e06c6c"

#: Superior/Inferior chip colour. Anatomical axis colours are data, not chrome:
#: they stay R/L-red, A/P-green, S/I-blue whatever the interface accent is.
COLOR_AXIS_SI = "#6fa8dc"

#: Anatomical direction → axis chip colour, so an orientation reads at a glance.
AXIS_COLORS: dict[str, str] = {
    "R": COLOR_ERROR,
    "L": COLOR_ERROR,
    "A": COLOR_OK,
    "P": COLOR_OK,
    "S": COLOR_AXIS_SI,
    "I": COLOR_AXIS_SI,
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
# Themes
# ──────────────────────────────────────────────────────────────────────────────
#: The light theme, token by token. Every name above appears here exactly once —
#: :func:`_check_palettes` fails the import if one is missing, so a token added to
#: the dark block cannot be forgotten here.
#:
#: Not the dark values inverted: light chrome needs its *text* darkened and its
#: signal colours deepened, or they wash out. The accent is the clearest case —
#: the dark theme's #ffa400 on a white page is barely a colour, so the light one
#: carries the same amber several steps down.
LIGHT_TOKENS: dict[str, Any] = {
    "COLOR_BG": "#eeeef1",
    "COLOR_SURFACE": "#fdfdfe",
    "COLOR_WELL": "#f7f7f9",
    "COLOR_CONTROL": "#e7e7ea",
    "COLOR_CONTROL_HOVER": "#dcdce0",
    "COLOR_TEXT": "#1b1b1f",
    "COLOR_MUTED": "#5c5c66",
    "COLOR_FAINT": "#8b8b96",
    "COLOR_DISABLED": "#a2a2ac",
    "COLOR_BORDER": "#d0d0d6",
    "COLOR_BORDER_STRONG": "#b4b4bc",
    "COLOR_ALT_ROW": "#f6f6f8",
    "COLOR_SCROLL": "#c2c2c9",
    "COLOR_SCROLL_HOVER": "#a9a9b2",
    "COLOR_BUTTON_OFF": "#ededf0",
    # One step off the fill, the way the dark block has it — which lands on the
    # same value as COLOR_CONTROL in both themes. They have to agree: a single hex
    # cannot have two answers when a live restyle maps it back.
    "COLOR_BORDER_OFF": "#e7e7ea",
    "COLOR_ACCENT": "#a66300",
    # Unchanged: both are read against the white text on top of them.
    "COLOR_ACCENT_DEEP": COLOR_ACCENT_DEEP,
    "COLOR_ON_ACCENT": COLOR_ON_ACCENT,
    "COLOR_OK": "#2f7d32",
    "COLOR_WARN": "#a96a00",
    "COLOR_ERROR": "#c02b2b",
    "COLOR_AXIS_SI": "#2f6fb0",
    # Anatomy keeps its hues — R/L red, A/P green, S/I blue — at the light
    # theme's darker values, so a chip still reads as its axis on a white page.
    "AXIS_COLORS": {
        "R": "#c02b2b",
        "L": "#c02b2b",
        "A": "#2f7d32",
        "P": "#2f7d32",
        "S": "#2f6fb0",
        "I": "#2f6fb0",
    },
    # Row tints, light enough to keep dark text on them readable.
    "STATUS_COLORS": {
        "ok": "#dcefdc",
        "warn": "#f8ecc9",
        "bad": "#f7d6d6",
        "neutral": "#eeeef1",
        "unknown": "#e4e4e8",
    },
    "SIGNIFICANCE_COLORS": {
        # Shares the 'ok' tint, as the dark theme shares its green: the hex is
        # what a live restyle maps, so one dark value cannot become two light ones.
        "***": "#dcefdc",
        "**": "#e2f2e2",
        "*": "#e9f6e6",
        ".": "#f1f1da",
    },
}

#: Snapshot of the dark theme, read back off this module so the documented block
#: above stays the single place a dark value is written.
DARK_TOKENS: dict[str, Any] = {name: globals()[name] for name in LIGHT_TOKENS}

#: Theme name → its token table.
THEMES: dict[str, dict[str, Any]] = {"dark": DARK_TOKENS, "light": LIGHT_TOKENS}

#: Preferences key holding the theme to open with.
THEME_PREF_KEY = "gui.theme"

_ACTIVE_THEME = "dark"


def _hex_values(tokens: dict[str, Any]) -> dict[str, str]:
    """Token → hex, flattened so the dict-valued tokens take part one entry each."""
    out: dict[str, str] = {}
    for name, value in tokens.items():
        if isinstance(value, dict):
            for key, hexval in value.items():
                out[f"{name}[{key}]"] = str(hexval)
        else:
            out[name] = str(value)
    return out


def _check_palettes() -> None:
    """Fail the import if the two token tables cannot map onto one another.

    A live restyle rewrites the hex values sitting in stylesheets that are already
    on screen, which only works if the mapping is a function: one hex in, one hex
    out, in both directions. Two tokens may share a hex (the green behind 'ok' and
    behind '***' is deliberately one colour), but then they have to share it in
    *both* themes or the swap back would have to pick a winner.
    """
    missing = sorted(set(DARK_TOKENS) - set(LIGHT_TOKENS))
    if missing:
        raise ValueError(f"LIGHT_TOKENS is missing: {', '.join(missing)}")
    dark, light = _hex_values(DARK_TOKENS), _hex_values(LIGHT_TOKENS)
    if set(dark) != set(light):
        raise ValueError(
            "The themes disagree on their colour keys: "
            f"{sorted(set(dark) ^ set(light))}"
        )
    for name, (source, target) in {
        "dark → light": (dark, light),
        "light → dark": (light, dark),
    }.items():
        seen: dict[str, str] = {}
        for key, value in source.items():
            mapped = target[key]
            if seen.setdefault(value.lower(), mapped).lower() != mapped.lower():
                raise ValueError(
                    f"{name} is ambiguous: {value} maps to both "
                    f"{seen[value.lower()]} and {mapped} (see {key})."
                )


_check_palettes()


def active_theme() -> str:
    """Name of the theme in force — ``"dark"`` or ``"light"``."""
    return _ACTIVE_THEME


def _remap(text: str, mapping: dict[str, str]) -> str:
    """Rewrite every hex colour in *text* that *mapping* has an entry for.

    One pass, so a value that is another entry's key cannot be substituted twice.
    """
    lookup = {k.lower(): v for k, v in mapping.items()}
    return re.sub(
        r"#[0-9a-fA-F]{3,8}",
        lambda m: lookup.get(m.group(0).lower(), m.group(0)),
        text,
    )


def _hex_mapping(outgoing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, str]:
    """``{old hex: new hex}`` for the tokens that differ between two themes."""
    old, new = _hex_values(outgoing), _hex_values(incoming)
    return {
        value: new[key] for key, value in old.items() if value.lower() != new[key].lower()
    }


def _rebind_tokens(outgoing: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Point every token at the incoming theme's value, here and wherever it was copied.

    ``from ...design import COLOR_TEXT`` binds a *copy* into the importing module,
    which updating this module's globals cannot reach — and there are a few dozen
    such copies. Each one is rebound by name, but only while it still holds the
    outgoing theme's value: a module that computed or overrode its own colour is
    left alone.
    """
    globals().update(incoming)
    this_module = sys.modules[__name__]
    for module in list(sys.modules.values()):
        if module is this_module or module is None:
            continue
        if not str(getattr(module, "__name__", "")).startswith("nvitk."):
            continue
        for token, value in incoming.items():
            try:
                if getattr(module, token, None) == outgoing[token]:
                    setattr(module, token, value)
            except Exception:  # noqa: BLE001 — a module that refuses is not fatal
                continue


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
def stylesheet() -> str:
    """The application stylesheet, built from the tokens as they stand now.

    A function rather than a constant because the tokens move: called again after
    :func:`set_theme`, it comes back in the new theme's colours.
    """
    return f"""
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
    selection-color: {COLOR_ON_ACCENT};
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
    color: {COLOR_ON_ACCENT};
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {COLOR_DISABLED};
    border-color: {COLOR_BORDER_OFF};
    background-color: {COLOR_BUTTON_OFF};
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
    alternate-background-color: {COLOR_ALT_ROW};
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
    color: {COLOR_ON_ACCENT};
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
    background-color: {COLOR_SCROLL};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:hover {{
    background-color: {COLOR_SCROLL_HOVER};
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


#: The stylesheet as of the theme in force. Rebuilt by :func:`set_theme`; callers
#: that apply it themselves should prefer :func:`stylesheet`, which cannot be stale.
STYLESHEET = stylesheet()


#: Id under which the nvitk palette is registered with Napari, per theme.
NAPARI_THEME_IDS: dict[str, str] = {"dark": "nvitk", "light": "nvitk-light"}
#: The dark theme's id, kept as a name because callers had it before there were two.
NAPARI_THEME_ID = NAPARI_THEME_IDS["dark"]

#: Napari icons are drawn in one flat colour, and it has to carry the chrome it
#: sits on rather than the text colour: light glyphs on dark, dark on light.
_NAPARI_ICON = {"dark": "#cfcfcf", "light": "#3c3c44"}


def napari_theme_colors(theme: str | None = None) -> dict[str, str]:
    """The nvitk palette expressed in Napari's theme vocabulary.

    Napari paints its own chrome — the layer list, the dims sliders, the menus,
    the area around the canvas — from a registered theme rather than from a
    stylesheet, so matching it to the docks means restating the same tokens in
    its field names rather than styling those widgets ourselves.

    *theme* names which of :data:`THEMES` to express; it defaults to the one in
    force. The tokens are read out of the table rather than off this module, so
    either theme can be described without switching to it.
    """
    name = str(theme or active_theme()).lower()
    tokens = THEMES.get(name, DARK_TOKENS)
    return {
        "id": NAPARI_THEME_IDS.get(name, NAPARI_THEME_ID),
        "label": f"nvitk {name}",
        # Pygments' dark palette is unreadable on a light console and vice versa.
        "syntax_style": "native" if name == "dark" else "default",
        # The canvas stays true black in both themes: it is image data, not
        # chrome, and a tinted surround shifts how the intensities in it read.
        "canvas": "black",
        "console": tokens["COLOR_WELL"],
        "background": tokens["COLOR_BG"],
        "foreground": tokens["COLOR_CONTROL"],
        "primary": tokens["COLOR_BORDER_STRONG"],
        "secondary": tokens["COLOR_MUTED"],
        "highlight": tokens["COLOR_CONTROL_HOVER"],
        "text": tokens["COLOR_TEXT"],
        "icon": _NAPARI_ICON.get(name, "#cfcfcf"),
        "warning": tokens["COLOR_WARN"],
        "error": tokens["COLOR_ERROR"],
        "current": tokens["COLOR_ACCENT_DEEP"],
        "font_size": "9pt",
    }


def register_napari_theme(theme: str | None = None) -> str:
    """Register an nvitk palette as a Napari theme and return its id.

    Returns Napari's own ``"dark"`` if registration fails, so a Napari whose theme
    API has moved falls back to a sane theme instead of breaking startup.
    """
    name = str(theme or active_theme()).lower()
    theme_id = NAPARI_THEME_IDS.get(name, NAPARI_THEME_ID)
    try:
        from napari.utils.theme import available_themes, register_theme

        if theme_id not in available_themes():
            register_theme(theme_id, napari_theme_colors(name), theme_id)
        return theme_id
    except Exception:
        return "dark"


#: Qt property marking a widget :func:`apply_theme` owns. A live theme switch
#: restyles these trees and leaves everything else — Napari's own widgets above
#: all — to whoever styled them.
THEMED_PROPERTY = "nvitkThemed"


def apply_theme(widget: QWidget) -> None:
    """Apply the nvitk palette and stylesheet to *widget* and its children.

    Top-level windows need the ``QPalette`` too: menus, dialogs and tooltips are
    separate native windows that otherwise fall back to the OS palette and render
    light-on-dark.

    The colours are whichever theme is in force when this runs, so a panel built
    after a switch opens in the new theme. ``widget`` is marked as ours, which is
    what lets :func:`switch_theme` find it again later.
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLOR_BG))
    palette.setColor(QPalette.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Base, QColor(COLOR_WELL))
    palette.setColor(QPalette.AlternateBase, QColor(COLOR_ALT_ROW))
    palette.setColor(QPalette.Text, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Button, QColor(COLOR_CONTROL))
    palette.setColor(QPalette.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ToolTipBase, QColor(COLOR_WELL))
    palette.setColor(QPalette.ToolTipText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.Highlight, QColor(COLOR_ACCENT_DEEP))
    palette.setColor(QPalette.HighlightedText, QColor(COLOR_ON_ACCENT))
    widget.setPalette(palette)
    # A Qt-side property rather than a Python attribute: the wrapper object a
    # later ``findChildren`` hands back is not guaranteed to be the same one, and
    # a Python attribute would not survive that.
    widget.setProperty(THEMED_PROPERTY, True)
    widget.setStyleSheet(stylesheet())


# ──────────────────────────────────────────────────────────────────────────────
# Switching
# ──────────────────────────────────────────────────────────────────────────────
def set_theme(name: str) -> str:
    """Make *name* the theme in force, without touching anything already drawn.

    Rebinds the tokens and rebuilds :data:`STYLESHEET`; widgets built from here on
    come out in the new colours. :func:`switch_theme` is the one to call for a
    window that is already open.
    """
    global _ACTIVE_THEME, STYLESHEET

    target = str(name or "").strip().lower()
    if target not in THEMES:
        raise ValueError(f"Unknown theme {name!r}. Available: {', '.join(sorted(THEMES))}")
    if target != _ACTIVE_THEME:
        _rebind_tokens(THEMES[_ACTIVE_THEME], THEMES[target])
        _ACTIVE_THEME = target
        STYLESHEET = stylesheet()
    return _ACTIVE_THEME


def _themed_roots() -> list[QWidget]:
    """Every live widget :func:`apply_theme` has been called on."""
    app = QApplication.instance()
    if app is None:
        return []
    roots: list[QWidget] = []
    for widget in app.allWidgets():
        try:
            if widget.property(THEMED_PROPERTY):
                roots.append(widget)
        except Exception:  # noqa: BLE001 — a widget mid-destruction is not ours
            continue
    return roots


def _remap_item_colours(widget: QWidget, mapping: dict[str, str]) -> int:
    """Rewrite the brushes painted *into* a list/table/tree's items.

    A stylesheet cannot reach these: ``item.setBackground(QColor(...))`` stores the
    colour in the item, which is why the QC review table's pass/fail tint and the
    DICOM tree's key colouring would otherwise keep the theme they were built in —
    dark text on a light page — until whatever refreshes them ran again.

    Views that paint through a model (the statmodels tables answer
    ``Qt.BackgroundRole`` from :data:`SIGNIFICANCE_COLORS`) need none of this: the
    role is read at paint time, so they come back in the new colours on their own.
    """
    from qtpy.QtGui import QBrush
    from qtpy.QtWidgets import QListWidget, QTableWidget, QTreeWidget, QTreeWidgetItemIterator

    def _swap(item: Any, column: int | None = None) -> int:
        """Remap one item's foreground and background, if it was given either.

        *column* is for tree items, whose accessors are per column where a table's
        and a list's take none.
        """
        if column is None:
            roles = (
                (item.foreground, item.setForeground),
                (item.background, item.setBackground),
            )
        else:
            roles = (
                (lambda: item.foreground(column), lambda b: item.setForeground(column, b)),
                (lambda: item.background(column), lambda b: item.setBackground(column, b)),
            )
        count = 0
        for get, set_ in roles:
            try:
                brush = get()
                # An item never given a colour answers with a default NoBrush one,
                # whose colour is black. Painting *that* would tint every plain cell.
                if brush is None or brush.style() == Qt.NoBrush:
                    continue
                new = mapping.get(brush.color().name().lower())
                if new:
                    set_(QBrush(QColor(new)))
                    count += 1
            except Exception:  # noqa: BLE001 — one odd item must not stop the rest
                continue
        return count

    changed = 0
    if isinstance(widget, QTableWidget):
        for row in range(widget.rowCount()):
            for column in range(widget.columnCount()):
                item = widget.item(row, column)
                if item is not None:
                    changed += _swap(item)
    elif isinstance(widget, QListWidget):
        for row in range(widget.count()):
            item = widget.item(row)
            if item is not None:
                changed += _swap(item)
    elif isinstance(widget, QTreeWidget):
        iterator = QTreeWidgetItemIterator(widget)
        while iterator.value():
            item = iterator.value()
            for column in range(max(item.columnCount(), 1)):
                changed += _swap(item, column)
            iterator += 1
    return changed


def _restyle_live_widgets(mapping: dict[str, str]) -> int:
    """Move the widgets already on screen to the new palette; returns how many changed.

    Three passes over the nvitk trees, because their colour arrives three ways. The
    panels' own inline stylesheets — ``setStyleSheet(f"color: {COLOR_MUTED}")`` and
    its several dozen siblings — have the old hex baked into them, so those are
    rewritten value by value through *mapping*; so are the brushes painted into
    list, table and tree items. The themed roots then have the fresh application
    stylesheet and palette applied over the top.

    Rewriting what is already there is what makes the switch immediate rather than
    something you see as each panel happens to be rebuilt. It also means a panel
    need do nothing to take part, which is why it is worth the regex.
    """
    changed = 0
    roots = _themed_roots()
    for root in roots:
        for widget in [root, *root.findChildren(QWidget)]:
            try:
                sheet = widget.styleSheet()
                if sheet:
                    swapped = _remap(sheet, mapping)
                    if swapped != sheet:
                        widget.setStyleSheet(swapped)
                        changed += 1
                changed += _remap_item_colours(widget, mapping)
            except Exception:  # noqa: BLE001 — skip anything being destroyed
                continue
    for root in roots:
        try:
            apply_theme(root)
        except Exception:  # noqa: BLE001
            continue
    return changed


def _current_napari_viewer() -> Any:
    """The live Napari viewer, when this process has one.

    Read out of ``sys.modules`` rather than imported: the statmodels explorer runs
    standalone too, and importing Napari to ask the question would cost seconds in
    a process that never wanted it. A window that does not hold a viewer can still
    switch the whole application this way.
    """
    napari = sys.modules.get("napari")
    if napari is None:
        return None
    try:
        return napari.current_viewer()
    except Exception:  # noqa: BLE001 — no viewer, or a Napari mid-teardown
        return None


def _set_napari_theme(viewer: Any, name: str) -> None:
    """Point *viewer* at the nvitk theme for *name*, registering it on first use."""
    viewer = viewer if viewer is not None else _current_napari_viewer()
    if viewer is None:
        return
    theme_id = register_napari_theme(name)
    try:
        viewer.theme = theme_id
    except Exception:  # noqa: BLE001 — a Napari whose theme API moved is not fatal
        pass


def switch_theme(name: str, *, viewer: Any = None, remember: bool = True) -> str:
    """Switch to *name* and repaint everything already on screen.

    Returns the theme now in force. *viewer* is the Napari viewer, whose own chrome
    is painted from a registered theme rather than from our stylesheet and so has
    to be moved separately. With *remember*, the choice is stored in the GUI
    preferences and is what the next launch opens with.
    """
    target = str(name or "").strip().lower()
    if target not in THEMES:
        raise ValueError(f"Unknown theme {name!r}. Available: {', '.join(sorted(THEMES))}")
    if target != _ACTIVE_THEME:
        # Before the tokens move: the mapping is built from the outgoing values.
        mapping = _hex_mapping(THEMES[_ACTIVE_THEME], THEMES[target])
        set_theme(target)
        _restyle_live_widgets(mapping)
    _set_napari_theme(viewer, target)
    if remember:
        try:
            from nvitk.gui.core.prefs import save_prefs

            save_prefs({THEME_PREF_KEY: target})
        except Exception:  # noqa: BLE001 — an unwritable config is not fatal
            pass
    return target


def toggle_theme(viewer: Any = None) -> str:
    """Flip between the dark and light themes; returns the one now in force."""
    return switch_theme("light" if active_theme() == "dark" else "dark", viewer=viewer)


def stored_theme() -> str:
    """The theme the preferences ask for, or ``"dark"`` when they say nothing."""
    try:
        from nvitk.gui.core.prefs import load_prefs

        name = str(load_prefs().get(THEME_PREF_KEY) or "").strip().lower()
    except Exception:  # noqa: BLE001
        return "dark"
    return name if name in THEMES else "dark"


def theme_toggle_button(viewer: Any = None, parent: QWidget | None = None) -> QToolButton:
    """A button that flips the interface between the dark and light themes.

    Labelled with the theme it would switch *to*, so it reads as an action rather
    than as a status. Its geometry is set explicitly: this control is meant to sit
    in someone else's bar — a tab corner, a dock header — and a host stylesheet
    that clamps button sizes otherwise leaves it laid out, clickable and blank.
    """
    button = QToolButton(parent)
    button.setObjectName("nvitkThemeToggle")
    button.setCursor(Qt.PointingHandCursor)
    button.setFocusPolicy(Qt.NoFocus)
    button.setMinimumSize(78, 22)
    button.setStyleSheet(
        f"QToolButton#nvitkThemeToggle {{ padding: 2px 8px; margin: 0px 2px;"
        f" min-width: 78px; min-height: 22px; max-height: 26px;"
        f" background-color: {COLOR_CONTROL}; color: {COLOR_TEXT};"
        f" border: 1px solid {COLOR_BORDER}; border-radius: 4px; font-size: 11px; }}"
        f"QToolButton#nvitkThemeToggle:hover {{ background-color: {COLOR_CONTROL_HOVER};"
        f" border-color: {COLOR_BORDER_STRONG}; }}"
    )

    def _sync() -> None:
        """Label and tooltip for the theme this would switch to."""
        going_light = active_theme() == "dark"
        button.setText("☀  Light" if going_light else "☾  Dark")
        button.setToolTip(
            "Switch the interface to the light theme."
            if going_light
            else "Switch the interface back to the dark theme."
        )

    def _clicked() -> None:
        toggle_theme(viewer)
        _sync()

    button.clicked.connect(_clicked)
    _sync()
    return button


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


def chip(text: str, color: str = "") -> QLabel:
    """Small rounded badge label in *color* (the accent when not given)."""
    color = color or COLOR_ACCENT
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
    color: str = "",
    mono: bool = False,
    align: Any = Qt.AlignLeft,
) -> QLabel:
    """One grid cell as a styled, selectable label."""
    color = color or COLOR_TEXT
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
    color: str = "",
    align: Any = Qt.AlignRight,
) -> QLabel:
    """A number with its unit set in muted type, so the figures stay dominant."""
    color = color or COLOR_TEXT
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
    color: str = "",
    key_width: int = 96,
) -> QWidget:
    """A single ``label: value`` line with the keys aligned in a column."""
    color = color or COLOR_TEXT
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
                    color=COLOR_ACCENT if tinted else COLOR_TEXT,
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


def style_image_figure(fig: Any) -> None:
    """Theme a Matplotlib figure that carries an image, for the dark chrome.

    Greyscale is read against dark and these canvases sit inside Napari's own dark
    viewer, so — unlike :func:`style_figure`, whose plots stay report-ready white —
    an image figure takes the surrounding background.
    """
    fig.patch.set_facecolor(COLOR_BG)
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_color(COLOR_TEXT)
    for ax in fig.axes:
        ax.set_facecolor(COLOR_WELL)
        for spine in ax.spines.values():
            spine.set_color(COLOR_BORDER)
        ax.tick_params(colors=COLOR_MUTED, which="both")
        ax.xaxis.label.set_color(COLOR_MUTED)
        ax.yaxis.label.set_color(COLOR_MUTED)
        ax.title.set_color(COLOR_TEXT)


__all__ = [
    "AXIS_COLORS",
    "COLOR_ACCENT",
    "COLOR_ACCENT_DEEP",
    "COLOR_ALT_ROW",
    "COLOR_AXIS_SI",
    "COLOR_BG",
    "COLOR_BORDER",
    "COLOR_BORDER_OFF",
    "COLOR_BORDER_STRONG",
    "COLOR_BUTTON_OFF",
    "COLOR_CONTROL",
    "COLOR_DISABLED",
    "COLOR_ERROR",
    "COLOR_FAINT",
    "COLOR_MUTED",
    "COLOR_OK",
    "COLOR_ON_ACCENT",
    "COLOR_SCROLL",
    "COLOR_SCROLL_HOVER",
    "COLOR_SURFACE",
    "COLOR_TEXT",
    "COLOR_WARN",
    "COLOR_WELL",
    "DARK_TOKENS",
    "LIGHT_TOKENS",
    "PAD",
    "RADIUS",
    "SIGNIFICANCE_COLORS",
    "STATUS_COLORS",
    "SPACE",
    "SPACE_LOOSE",
    "SPACE_TIGHT",
    "NAPARI_THEME_ID",
    "NAPARI_THEME_IDS",
    "STYLESHEET",
    "THEMED_PROPERTY",
    "THEMES",
    "THEME_PREF_KEY",
    "Card",
    "active_theme",
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
    "set_theme",
    "stored_theme",
    "style_figure",
    "style_image_figure",
    "stylesheet",
    "switch_theme",
    "theme_toggle_button",
    "toggle_theme",
]
