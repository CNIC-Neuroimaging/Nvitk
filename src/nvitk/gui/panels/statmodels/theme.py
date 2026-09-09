"""Statmodels-facing view of the shared nvitk design system.

The explorer runs as a floating window rather than inside Napari's chrome, so it
has to carry a palette explicitly — but that palette is now the same one every
nvitk dock uses. Everything here is re-exported from
:mod:`nvitk.gui.core.design`; add new tokens or primitives there, not here, so
the explorer and the docks cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import QWidget

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_ACCENT_DEEP,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_ERROR,
    COLOR_FAINT,
    COLOR_MUTED,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_WARN,
    COLOR_WELL,
    SIGNIFICANCE_COLORS,
    STYLESHEET,
    apply_theme,
    clear_layout,
    muted_label_style,
    style_figure,
)

# Names the explorer grew up with, kept so its modules read unchanged.
COLOR_WINDOW = COLOR_BG
COLOR_BASE = COLOR_WELL
DARK_STYLESHEET = STYLESHEET


def apply_dark_theme(widget: QWidget) -> None:
    """Apply the nvitk palette and stylesheet to *widget* (alias of :func:`apply_theme`)."""
    apply_theme(widget)


def whiten_figure(fig: Any) -> None:
    """Force a white Matplotlib canvas with dark text (alias of :func:`style_figure`)."""
    style_figure(fig)


__all__ = [
    "COLOR_ACCENT",
    "COLOR_ACCENT_DEEP",
    "COLOR_BASE",
    "COLOR_BORDER",
    "COLOR_ERROR",
    "COLOR_FAINT",
    "COLOR_MUTED",
    "COLOR_SURFACE",
    "COLOR_TEXT",
    "COLOR_WARN",
    "COLOR_WINDOW",
    "DARK_STYLESHEET",
    "SIGNIFICANCE_COLORS",
    "apply_dark_theme",
    "apply_theme",
    "clear_layout",
    "muted_label_style",
    "style_figure",
    "whiten_figure",
]
