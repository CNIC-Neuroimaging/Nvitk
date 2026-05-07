"""HTML embedding helpers for portability (single-file reports)."""

from __future__ import annotations

import html


def escape_srcdoc(text: str) -> str:
    """Escape HTML so it can be placed inside an iframe ``srcdoc=\"...\"`` attribute."""
    # Must escape &, <, >, " and also normalize newlines.
    # Keep it compact to reduce HTML size slightly.
    return html.escape(text, quote=True).replace("\n", "&#10;")


def iframe_srcdoc(html_text: str, *, height_px: int = 420, title: str = "embedded") -> str:
    esc = escape_srcdoc(html_text)
    return (
        f'<div class="iframe-wrap"><iframe title="{html.escape(title)}" '
        f'style="width:100%;height:{int(height_px)}px;border:0;background:#0b0f17;border-radius:10px" '
        f'srcdoc="{esc}"></iframe></div>'
    )


__all__ = ["escape_srcdoc", "iframe_srcdoc"]

