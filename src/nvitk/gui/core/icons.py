"""nvitk's icon set: small line icons, tinted from the theme at draw time.

Every icon is a 24x24 SVG body drawn with ``currentColor`` strokes, kept here as
text rather than as files so the set ships with the module and needs no package
data, and so a new icon is one dictionary entry. :func:`icon` renders one into a
:class:`QIcon` in the colours of the theme in force, with a dimmed disabled state
and a light "on" state for checked buttons sitting on the accent fill.

The geometry is nvitk's own, drawn in the common outline style (2px-ish stroke,
round joins) so the set sits comfortably beside Napari's chrome. A name not in
:data:`ICONS` falls back to Napari's bundled icons (``"paint"``, ``"erase"``,
``"visibility"``...), recoloured the same way.

Icons are baked at a colour, so a widget showing one has to ask again after a
theme switch: the primitives in :mod:`nvitk.gui.core.widgets` do this for you.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QByteArray, QRectF, QSize, Qt
from qtpy.QtGui import QIcon, QPainter, QPixmap

from nvitk.gui.core import design

#: Line icons, by name. Each value is the inside of a ``viewBox="0 0 24 24"`` SVG
#: whose strokes are ``currentColor``; a filled detail says ``fill="currentColor"``.
ICONS: dict[str, str] = {
    # Viewport layouts
    "layout_2x2": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18M3 12h18"/>',
    "layout_3plus1": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M3 9h6M3 15h6"/>',
    "layout_1plus3": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18M15 9h6M15 15h6"/>',
    "layout_3d": (
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="M12 7.5l4 2.2v4.6L12 16.5l-4-2.2V9.7z M8 9.7l4 2.2 4-2.2 M12 11.9v4.6"/>'
    ),
    "layout_axes_row": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18"/>',
    "layout_3d_over_axes": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 15h18M9 15v6M15 15v6"/>',
    "layout_axes_col": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18"/>',
    "layout_single": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 8v8M8 12h8"/>',
    "layout_3d_axis": (
        '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M10 3v18"/>'
        '<path d="M15.5 8.5l3 1.7v3.6l-3 1.7-3-1.7v-3.6z M12.5 10.2l3 1.7 3-1.7 M15.5 11.9v3.6"/>'
    ),
    # Viewing
    "crosshair": '<circle cx="12" cy="12" r="6.5"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/>',
    "crosshair_3d": (
        '<path d="M12 3v18M3 12h18"/><path d="M5.5 18.5l13-13"/>'
        '<path d="M8 4h8l4 4v8M4 8v8l4 4h8" stroke-dasharray="2 2.2"/>'
    ),
    "crosshair_click": (
        '<path d="M14 4.1 12 6M5.1 8l-2.9-.8M6 12l-1.9 2M7.2 2.2 8 5.1"/>'
        '<path d="M9.04 9.69a.5.5 0 0 1 .65-.65l11 4.5a.5.5 0 0 1-.07.95l-4.35 1.04a1 1 0 0 0-.74.74l-1.04 4.35a.5.5 0 0 1-.95.07z"/>'
    ),
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "eye_off": (
        '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'
        '<path d="M3 3l18 18"/>'
    ),
    "sliders": (
        '<path d="M4 6h3M11 6h9M4 12h9M17 12h3M4 18h1M9 18h11"/>'
        '<circle cx="9" cy="6" r="2"/><circle cx="15" cy="12" r="2"/><circle cx="7" cy="18" r="2"/>'
    ),
    "maximize": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    "restore": '<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "chevron_down": '<path d="M6 9l6 6 6-6"/>',
    "chevron_up": '<path d="M6 15l6-6 6 6"/>',
    "chevron_right": '<path d="M9 6l6 6-6 6"/>',
    "chevron_left": '<path d="M15 6l-6 6 6 6"/>',
    "search": '<circle cx="11" cy="11" r="6"/><path d="M15.5 15.5L20 20"/>',
    "filter": '<path d="M4 5h16l-6 7.5V19l-4-2v-4.5z"/>',
    "pin": '<path d="M12 16v5M8 4h8l-1 6 3 3H6l3-3z"/>',
    "refresh": '<path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5"/>',
    "replay": '<path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v5h5"/><path d="M10.5 9.5v5l4-2.5z"/>',
    "undo": '<path d="M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3"/>',
    "redo": '<path d="M15 14l5-5-5-5M20 9H10a6 6 0 0 0 0 12h3"/>',
    "more": (
        '<circle cx="6" cy="12" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/>'
        '<circle cx="18" cy="12" r="1.3" fill="currentColor"/>'
    ),
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><circle cx="12" cy="8" r=".6" fill="currentColor"/>',
    "help": (
        '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.6v.6"/>'
        '<circle cx="12" cy="17" r=".6" fill="currentColor"/>'
    ),
    "warning": '<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><circle cx="12" cy="17" r=".6" fill="currentColor"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "sun": (
        '<circle cx="12" cy="12" r="4"/>'
        '<path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'
    ),
    "moon": '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>',
    "contrast": '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor"/>',
    "droplet": '<path d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11z"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "film": (
        '<rect x="3" y="4" width="18" height="16" rx="1"/>'
        '<path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>'
    ),
    "play": '<path d="M7 4l12 8-12 8z"/>',
    "pause": '<path d="M9 5v14M15 5v14"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    "record": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5" fill="currentColor"/>',
    "clock": '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
    # Shell modules
    "database": (
        '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/>'
        '<path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>'
    ),
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "tools": '<path d="M15 4a5 5 0 0 0-4.6 6.9L4 17.3V20h2.7l6.4-6.4A5 5 0 0 0 20 9l-3 3-3-1-1-3z"/>',
    "brush": '<path d="M4 20l1-4L16 5l3 3L8 19z"/><path d="M14 7l3 3"/>',
    "mesh": '<path d="M3 18L8 6l5 12zM8 6l10 2-5 10M18 8l3 9-8 1"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    "qc": '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "chart": '<path d="M4 20h16M7 20v-6M12 20V8M17 20v-9"/>',
    "export": '<path d="M12 4v11M7 10l5 5 5-5M4 20h16"/>',
    "import": '<path d="M12 15V4M7 9l5-5 5 5M4 20h16"/>',
    "inspect": (
        '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "panel_left": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    "panel_right": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    "panel_bottom": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 15h18"/>',
    "popout": (
        '<path d="M14 4h6v6M20 4l-9 9"/>'
        '<path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'
    ),
    # Files and data
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="M21 16l-5-5-9 9"/>',
    "folder": '<path d="M3 6a1 1 0 0 1 1-1h5l2 2h8a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/>',
    "save": (
        '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
        '<path d="M7 3v5h8V3"/><rect x="7" y="13" width="10" height="8"/>'
    ),
    "copy": (
        '<rect x="8" y="8" width="12" height="12" rx="2"/>'
        '<path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>'
    ),
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "table": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 14h18M9 4v16"/>',
    "tag": '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.2"/>',
    "atlas": '<path d="M4 5l5-2 6 2 5-2v16l-5 2-6-2-5 2z"/><path d="M9 3v16M15 5v16"/>',
    "code": '<path d="M8 7l-5 5 5 5M16 7l5 5-5 5"/>',
    "server": (
        '<rect x="4" y="4" width="16" height="7" rx="1"/><rect x="4" y="13" width="16" height="7" rx="1"/>'
        '<circle cx="8" cy="7.5" r=".8" fill="currentColor"/><circle cx="8" cy="16.5" r=".8" fill="currentColor"/>'
    ),
    "gpu": (
        '<rect x="6" y="6" width="12" height="12" rx="1"/><rect x="9" y="9" width="6" height="6"/>'
        '<path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3"/>'
    ),
    "external": (
        '<path d="M14 4h6v6M20 4l-9 9"/>'
        '<path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'
    ),
    "star": '<path d="M12 3l2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3l-5.6 2.9 1.1-6.2L3 9.6l6.2-.9z"/>',
    # Plots
    "plot_scatter": (
        '<path d="M4 4v16h16"/><circle cx="8" cy="15" r="1.3" fill="currentColor"/>'
        '<circle cx="11" cy="10" r="1.3" fill="currentColor"/><circle cx="15" cy="13" r="1.3" fill="currentColor"/>'
        '<circle cx="18" cy="7" r="1.3" fill="currentColor"/>'
    ),
    "plot_line": '<path d="M4 4v16h16"/><path d="M7 15l4-5 3 3 5-7"/>',
    "plot_box": '<path d="M8 4v4M8 16v4M16 6v3M16 15v3"/><rect x="5" y="8" width="6" height="8"/><rect x="13" y="9" width="6" height="6"/>',
    "plot_hist": '<path d="M4 20h16"/><path d="M5 20v-5h3v5M8 20V9h3v11M11 20V5h3v15M14 20v-8h3v8M17 20v-3h3v3"/>',
    # Tool sections
    "wand": '<path d="M4 20L14 10"/><path d="M17 3v4M15 5h4M20 9v2M19 10h2M12 4v2M11 5h2"/>',
    "masks": '<circle cx="9" cy="12" r="6"/><circle cx="15" cy="12" r="6"/>',
    "blob": '<path d="M7 6c3-3 9-2 11 2s0 9-4 10-9 0-10-4 0-6 3-8z"/>',
    "sparkle": '<path d="M12 3l2 5 5 2-5 2-2 5-2-5-5-2 5-2z"/><path d="M19 16v4M17 18h4"/>',
    "restore_image": '<path d="M3 9l2-2 2 3 2-4 2 5 2-3"/><path d="M13 16c2.5-2 5.5-2 8 0M3 18c2.5-2 5.5-2 8 0"/>',
    "align": '<rect x="4" y="4" width="11" height="11" rx="1"/><rect x="9" y="9" width="11" height="11" rx="1" stroke-dasharray="2.5 2"/>',
    "move": '<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
    "ruler": '<path d="M3 17L17 3l4 4L7 21z"/><path d="M7 13l2 2M10 10l2 2M13 7l2 2"/>',
    "vessel": '<path d="M4 20c4 0 6-3 7-8s3-8 9-8M11 12c2 0 5 1 9 5"/>',
    "wave": '<path d="M3 12c2-5 4-5 6 0s4 5 6 0 4-5 6 0"/>',
    "surface": '<circle cx="12" cy="12" r="8"/><ellipse cx="12" cy="12" rx="8" ry="3"/><ellipse cx="12" cy="12" rx="3" ry="8"/>',
    "pipeline": '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 6h10M6 8l5 8M18 8l-5 8"/>',
    "convert": '<path d="M4 8h14l-3-3M20 16H6l3 3"/>',
    "roi_box": (
        '<rect x="5" y="5" width="14" height="14" stroke-dasharray="3 2"/>'
        '<rect x="3.5" y="3.5" width="3" height="3" fill="currentColor"/>'
        '<rect x="17.5" y="17.5" width="3" height="3" fill="currentColor"/>'
    ),
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14M2 6h14a2 2 0 0 1 2 2v14"/>',
    "grid": (
        '<rect x="4" y="4" width="4" height="4"/><rect x="10" y="4" width="4" height="4"/><rect x="16" y="4" width="4" height="4"/>'
        '<rect x="4" y="10" width="4" height="4"/><rect x="10" y="10" width="4" height="4"/><rect x="16" y="10" width="4" height="4"/>'
        '<rect x="4" y="16" width="4" height="4"/><rect x="10" y="16" width="4" height="4"/><rect x="16" y="16" width="4" height="4"/>'
    ),
}

#: Sizes each icon is rendered at, so a button of any common size gets a crisp
#: pixmap rather than a scaled one.
_SIZES = (12, 14, 16, 18, 20, 24, 32, 48)
_STROKE = 1.8
_CACHE: dict[tuple[Any, ...], QIcon] = {}


def has_icon(name: str) -> bool:
    """Whether *name* is in nvitk's set (Napari's fallback icons are not counted)."""
    return name in ICONS


def available_icons() -> list[str]:
    """Every name in nvitk's own set, sorted."""
    return sorted(ICONS)


def svg_markup(name: str, color: str, *, stroke: float = _STROKE) -> str:
    """The full SVG document for icon *name* drawn in *color*."""
    body = ICONS[name].replace("currentColor", color)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


def _render(name: str, color: str, size: int, ratio: float) -> QPixmap:
    """One pixmap of icon *name* at *size* logical pixels and device ratio *ratio*."""
    from qtpy.QtSvg import QSvgRenderer

    side = max(1, int(round(size * ratio)))
    pixmap = QPixmap(side, side)
    pixmap.fill(Qt.transparent)
    renderer = QSvgRenderer(QByteArray(svg_markup(name, color).encode("utf-8")))
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter, QRectF(0, 0, side, side))
    finally:
        painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def _device_ratio() -> float:
    """The primary screen's device pixel ratio, or 1 before there is a screen."""
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            return float(app.devicePixelRatio() or 1.0)
    except Exception:  # noqa: BLE001
        pass
    return 1.0


def _napari_icon(name: str, color: str, disabled: str, on: str) -> QIcon:
    """Napari's bundled icon *name* in the same three colours, or an empty icon."""
    try:
        from napari._qt.qt_resources import QColoredSVGIcon

        base = QColoredSVGIcon.from_resources(name)
        out = QIcon()
        for mode, state, tone in (
            (QIcon.Normal, QIcon.Off, color),
            (QIcon.Active, QIcon.Off, color),
            (QIcon.Disabled, QIcon.Off, disabled),
            (QIcon.Normal, QIcon.On, on),
            (QIcon.Active, QIcon.On, on),
            (QIcon.Disabled, QIcon.On, disabled),
        ):
            colored = base.colored(color=tone)
            for size in _SIZES:
                out.addPixmap(colored.pixmap(QSize(size, size)), mode, state)
        return out
    except Exception:  # noqa: BLE001 (an unknown name is an empty icon, not a crash)
        return QIcon()


def icon(name: str, color: str | None = None, *, on_color: str | None = None) -> QIcon:
    """Icon *name* in the theme's colours.

    *color* is the normal tint (the text colour by default); a disabled button gets
    :data:`~nvitk.gui.core.design.COLOR_DISABLED`; a checked one *on_color*, which
    defaults to the colour drawn over the accent fill. Names outside :data:`ICONS`
    come from Napari's own set. Cached per colour, so asking again is free.
    """
    normal = color or design.COLOR_TEXT
    on = on_color or design.COLOR_ON_ACCENT
    disabled = design.COLOR_DISABLED
    key = (name, normal.lower(), on.lower(), disabled.lower())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if name not in ICONS:
        result = _napari_icon(name, normal, disabled, on)
    else:
        result = QIcon()
        ratio = _device_ratio()
        for size in _SIZES:
            plain = _render(name, normal, size, ratio)
            result.addPixmap(plain, QIcon.Normal, QIcon.Off)
            result.addPixmap(plain, QIcon.Active, QIcon.Off)
            result.addPixmap(_render(name, disabled, size, ratio), QIcon.Disabled, QIcon.Off)
            lit = _render(name, on, size, ratio)
            result.addPixmap(lit, QIcon.Normal, QIcon.On)
            result.addPixmap(lit, QIcon.Active, QIcon.On)
            result.addPixmap(_render(name, disabled, size, ratio), QIcon.Disabled, QIcon.On)
    _CACHE[key] = result
    return result


def pixmap(name: str, size: int = 16, color: str | None = None) -> QPixmap:
    """A single pixmap of icon *name*, for a label rather than a button."""
    if name in ICONS:
        return _render(name, color or design.COLOR_TEXT, size, _device_ratio())
    return icon(name, color).pixmap(QSize(size, size))


__all__ = ["ICONS", "available_icons", "has_icon", "icon", "pixmap", "svg_markup"]
