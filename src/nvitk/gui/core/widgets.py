"""Shared widgets the redesigned panels are built from.

:mod:`nvitk.gui.core.design` holds the tokens (colours, type scale, spacing); this
module holds the pieces made from them, so every panel draws its buttons, section
headers, badges and pop-overs the same way:

* :class:`IconButton`: a flat icon-only button with a tooltip (the default way to
  offer an action now that text labels are kept for what needs words).
* :class:`Section`: a collapsible group with a coloured accent stripe, an optional
  icon, and room in its header for a count or a badge.
* :class:`Badge`: a small tinted label naming a kind of thing, usually an external
  suite ("FSL", "ANTsPyNet"...), in that suite's accent.
* :class:`Popover`: a frameless panel that opens next to a button, closes on an
  outside click, and can be pinned to stay open as a small floating window.
* :class:`ChipFilterBar`: wrapping toggle chips for filtering a list.
* :class:`SegmentedButtons`: one choice out of a few, as a joined button row.
* :class:`PaneHeader`: the slim title strip over a viewport pane or inspector.

Every widget here restyles itself after a theme switch (through
:func:`~nvitk.gui.core.design.add_theme_listener`): their fills use ``rgba()``
tints and baked icon pixmaps, neither of which the hex-rewriting live restyle can
reach.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from qtpy.QtCore import QEvent, QPoint, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.flow_layout import FlowRow
from nvitk.gui.core.icons import icon as make_icon
from nvitk.gui.core.icons import pixmap as make_pixmap


def _hue_color(hue: str | None, fallback: str | None = None) -> str:
    """Hex for *hue*: an accent name, a suite name, or a literal colour."""
    if not hue:
        return fallback or design.COLOR_MUTED
    if hue in design.ACCENTS:
        return design.accent(hue)
    if hue in design.SUITE_HUES:
        return design.suite_color(hue)
    return hue


# Buttons
class IconButton(QToolButton):
    """A flat, icon-only button: transparent until hovered, accent-filled when checked.

    Parameters
    ----------
    icon_name
        A name from :data:`nvitk.gui.core.icons.ICONS` (or one of Napari's icons).
    tooltip
        What the button does; icon-only buttons depend on it.
    checkable
        Toggle button, drawn on the accent fill while checked.
    icon_size
        Icon side in pixels; the button is that plus a small margin.
    text
        Optional label drawn beside the icon, for the few places a word is clearer.
    hue
        Tint the icon in an accent (name, suite or hex) rather than the text colour.
    """

    def __init__(
        self,
        icon_name: str,
        tooltip: str = "",
        *,
        checkable: bool = False,
        icon_size: int = 16,
        text: str = "",
        hue: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon_name = icon_name
        self._hue = hue
        self.setObjectName("nvitkIconButton")
        self.setFocusPolicy(Qt.NoFocus)
        self.setCheckable(checkable)
        self.setAutoRaise(True)
        if tooltip:
            self.setToolTip(tooltip)
        self.setIconSize(QSize(icon_size, icon_size))
        side = icon_size + 8
        if text:
            self.setText(text)
            self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.setMinimumHeight(side)
        else:
            # Explicit geometry: a host stylesheet that clamps button sizes otherwise
            # leaves an icon-only button laid out, clickable and blank.
            self.setMinimumSize(side, side)
        self._restyle()
        design.add_theme_listener(self._restyle)

    def set_icon_name(self, name: str) -> None:
        """Show icon *name* from now on (a play button becoming a pause button)."""
        self._icon_name = name
        self.setIcon(make_icon(name, _hue_color(self._hue, design.COLOR_TEXT)))

    def icon_name(self) -> str:
        """The name of the icon shown."""
        return self._icon_name

    def _restyle(self) -> None:
        """Icon and stylesheet in the theme in force."""
        self.setIcon(make_icon(self._icon_name, _hue_color(self._hue, design.COLOR_TEXT)))
        self.setStyleSheet(
            "QToolButton#nvitkIconButton { background: transparent;"
            " border: 1px solid transparent; border-radius: 4px; padding: 2px;"
            f" color: {design.COLOR_TEXT}; font-size: {design.FONT_S}px; }}"
            "QToolButton#nvitkIconButton:hover {"
            f" background-color: {design.COLOR_CONTROL_HOVER}; }}"
            "QToolButton#nvitkIconButton:checked, QToolButton#nvitkIconButton:pressed {"
            f" background-color: {design.COLOR_ACCENT_DEEP}; border-color: {design.COLOR_ACCENT};"
            f" color: {design.COLOR_ON_ACCENT}; }}"
            "QToolButton#nvitkIconButton:disabled { background: transparent; }"
            "QToolButton#nvitkIconButton::menu-indicator { image: none; width: 0px; }"
        )


def popover_button(
    icon_name: str,
    tooltip: str,
    content: QWidget | Callable[[], QWidget],
    *,
    title: str = "",
    parent: QWidget | None = None,
    icon_size: int = 16,
) -> IconButton:
    """An :class:`IconButton` that opens a :class:`Popover` holding *content*.

    *content* may be a factory, called on the first click, so a pop-over that is
    never opened costs nothing to build. The pop-over is kept and reused.
    """
    button = IconButton(icon_name, tooltip, icon_size=icon_size, parent=parent)
    holder: dict[str, Popover] = {}

    def _toggle() -> None:
        pop = holder.get("pop")
        if pop is None:
            widget = content() if callable(content) and not isinstance(content, QWidget) else content
            pop = Popover(widget, title=title, parent=button.window())
            holder["pop"] = pop
        pop.toggle_at(button)

    button.clicked.connect(_toggle)
    button.popover = lambda: holder.get("pop")  # type: ignore[attr-defined]
    return button


# Sections and labels
class _ClickableHeader(QWidget):
    """A header row that reports a left click (the whole row toggles its section)."""

    clicked = Signal()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class Section(QWidget):
    """A collapsible group: header (chevron, icon, title, extras) over a body.

    Parameters
    ----------
    title
        Header text.
    icon_name
        Optional icon before the title.
    hue
        Accent (name, suite or hex) for the stripe down the header's left edge and
        the icon; the section reads as belonging to that kind of thing.
    expanded
        Initial state, unless *pref_key* has one stored.
    pref_key
        When given, the expanded state is remembered between launches under it.
    """

    toggled = Signal(bool)

    #: Preferences key holding ``{pref_key: expanded}`` for every remembered section.
    PREFS_KEY = "gui.sections"

    def __init__(
        self,
        title: str,
        *,
        icon_name: str | None = None,
        hue: str | None = None,
        expanded: bool = True,
        pref_key: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._hue = hue
        self._icon_name = icon_name
        self._pref_key = pref_key
        if pref_key:
            expanded = bool(self._stored().get(pref_key, expanded))
        self._expanded = bool(expanded)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = _ClickableHeader(self)
        self._header.setObjectName("nvitkSectionHeader")
        # A plain QWidget paints no stylesheet background (nor the accent stripe) without this.
        self._header.setAttribute(Qt.WA_StyledBackground, True)
        row = QHBoxLayout(self._header)
        row.setContentsMargins(6, 3, 4, 3)
        row.setSpacing(design.SPACE_TIGHT)
        self._chevron = QLabel(self._header)
        self._chevron.setFixedSize(12, 12)
        row.addWidget(self._chevron)
        self._icon = QLabel(self._header)
        self._icon.setFixedSize(16, 16)
        self._icon.setVisible(bool(icon_name))
        row.addWidget(self._icon)
        self._title = QLabel(title, self._header)
        row.addWidget(self._title, 1)
        self._extras = QHBoxLayout()
        self._extras.setContentsMargins(0, 0, 0, 0)
        self._extras.setSpacing(design.SPACE_TIGHT)
        row.addLayout(self._extras)
        self._header.clicked.connect(self.toggle)
        root.addWidget(self._header)

        self._body = QWidget(self)
        self._body.setObjectName("nvitkSectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(design.SPACE, design.SPACE_TIGHT, 2, design.SPACE)
        self._body_layout.setSpacing(design.SPACE_TIGHT)
        root.addWidget(self._body)

        self._apply_state()
        self._restyle()
        design.add_theme_listener(self._restyle)

    # contents
    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        """Append *widget* to the body; returns it."""
        self._body_layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout: Any) -> None:
        """Append a nested *layout* to the body."""
        self._body_layout.addLayout(layout)

    def body(self) -> QVBoxLayout:
        """The body layout, for callers that need finer control."""
        return self._body_layout

    def add_header_widget(self, widget: QWidget) -> QWidget:
        """Put *widget* (a count, a badge, an action button) at the header's right end."""
        self._extras.addWidget(widget)
        return widget

    def set_title(self, text: str) -> None:
        """Change the header text."""
        self._title.setText(text)

    def title(self) -> str:
        """The header text."""
        return self._title.text()

    # state
    def is_expanded(self) -> bool:
        """Whether the body is showing."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the body (and remember it, with a *pref_key*)."""
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._apply_state()
        if self._pref_key:
            stored = self._stored()
            stored[self._pref_key] = expanded
            try:
                from nvitk.gui.core.prefs import save_prefs

                save_prefs({self.PREFS_KEY: stored})
            except Exception:  # noqa: BLE001
                pass
        self.toggled.emit(expanded)

    def toggle(self) -> None:
        """Flip between expanded and collapsed."""
        self.set_expanded(not self._expanded)

    def _stored(self) -> dict[str, Any]:
        try:
            from nvitk.gui.core.prefs import load_prefs

            value = load_prefs().get(self.PREFS_KEY)
        except Exception:  # noqa: BLE001
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _apply_state(self) -> None:
        self._body.setVisible(self._expanded)
        name = "chevron_down" if self._expanded else "chevron_right"
        self._chevron.setPixmap(make_pixmap(name, 12, design.COLOR_MUTED))

    def _restyle(self) -> None:
        color = _hue_color(self._hue, design.COLOR_BORDER_STRONG)
        if self._icon_name:
            self._icon.setPixmap(make_pixmap(self._icon_name, 16, _hue_color(self._hue, design.COLOR_TEXT)))
        self._apply_state()
        self._header.setStyleSheet(
            "QWidget#nvitkSectionHeader {"
            f" background-color: {design.COLOR_SURFACE};"
            f" border: none; border-left: 3px solid {color}; border-radius: 3px; }}"
            "QWidget#nvitkSectionHeader:hover {"
            f" background-color: {design.COLOR_CONTROL}; }}"
        )
        self._title.setStyleSheet(
            f"color: {design.COLOR_TEXT}; font-weight: bold; font-size: {design.FONT_M}px;"
            " background: transparent;"
        )


class Badge(QLabel):
    """A small tinted label naming a kind of thing, most often an external suite.

    *hue* is an accent name, a suite name (``"FSL"`` picks the FSL colour) or a hex;
    when it is omitted, *text* itself is tried as a suite name.
    """

    def __init__(
        self,
        text: str,
        hue: str | None = None,
        *,
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._hue = hue or (text if text in design.SUITE_HUES else "grey")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        if tooltip:
            self.setToolTip(tooltip)
        self._restyle()
        design.add_theme_listener(self._restyle)

    def set_hue(self, hue: str) -> None:
        """Recolour the badge."""
        self._hue = hue
        self._restyle()

    def _restyle(self) -> None:
        color = _hue_color(self._hue)
        self.setStyleSheet(
            f"color: {color}; background-color: {design.tint(color, 0.16)};"
            f" border: 1px solid {design.tint(color, 0.45)}; border-radius: 3px;"
            f" padding: 0px 4px; font-size: {design.FONT_XS}px; font-weight: bold;"
        )


# Pop-overs
class Popover(QFrame):
    """A frameless panel that opens beside a button and closes on an outside click.

    Pinning (the pin in its header) turns it into a small floating tool window that
    stays open and can be dragged by its header, for options someone is iterating
    on while watching the canvas.
    """

    closed = Signal()

    #: A click on the anchor closes an open pop-over (it is outside it) and then
    #: arrives at the anchor, which would reopen it at once. Clicks within this many
    #: seconds of a close are taken as that, not as a request to reopen.
    _REOPEN_GUARD = 0.25

    def __init__(
        self,
        content: QWidget | None = None,
        *,
        title: str = "",
        pinnable: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self._pinned = False
        self._closed_at = 0.0
        self._drag_from: QPoint | None = None
        design.apply_theme(self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._frame = QFrame(self)
        self._frame.setObjectName("nvitkPopoverBody")
        outer.addWidget(self._frame)
        body = QVBoxLayout(self._frame)
        body.setContentsMargins(design.PAD - 2, 6, design.PAD - 2, design.PAD - 2)
        body.setSpacing(design.SPACE_TIGHT)

        self._header = QWidget(self._frame)
        head = QHBoxLayout(self._header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(2)
        self._title = QLabel(title.upper(), self._header)
        head.addWidget(self._title, 1)
        self._pin = IconButton("pin", "Pin: keep this panel open as a floating window", checkable=True, icon_size=12)
        self._pin.setVisible(pinnable)
        self._pin.toggled.connect(self.set_pinned)
        head.addWidget(self._pin)
        close = IconButton("close", "Close", icon_size=12)
        close.clicked.connect(self.close)
        head.addWidget(close)
        self._header.installEventFilter(self)
        body.addWidget(self._header)

        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        body.addLayout(self._content_layout, 1)
        # A corner to resize it by: the content's own size is only where it starts.
        from qtpy.QtWidgets import QSizeGrip

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self._grip = QSizeGrip(self._frame)
        self._grip.setToolTip("Drag to resize")
        grip_row.addWidget(self._grip, 0, Qt.AlignRight | Qt.AlignBottom)
        body.addLayout(grip_row)
        self._content: QWidget | None = None
        if content is not None:
            self.set_content(content)
        self._restyle()
        design.add_theme_listener(self._restyle)

    def set_content(self, widget: QWidget) -> None:
        """Show *widget* as the pop-over's body, replacing any previous one."""
        if self._content is not None:
            self._content.hide()
            self._content.setParent(None)
        self._content = widget
        self._content_layout.addWidget(widget)
        widget.show()

    def content(self) -> QWidget | None:
        """The widget shown in the body."""
        return self._content

    def set_title(self, text: str) -> None:
        """Change the header text."""
        self._title.setText(str(text).upper())

    # placement
    def show_at(self, anchor: QWidget, *, side: str = "below") -> None:
        """Open beside *anchor* (``"below"``, ``"below_end"``, ``"above"``, ``"left"`` or
        ``"right"``), kept on the anchor's screen."""
        self.adjustSize()
        size = self.sizeHint().expandedTo(self.minimumSizeHint())
        top_left = anchor.mapToGlobal(QPoint(0, 0))
        w, h = anchor.width(), anchor.height()
        if side == "above":
            pos = QPoint(top_left.x(), top_left.y() - size.height() - 2)
        elif side == "left":
            pos = QPoint(top_left.x() - size.width() - 2, top_left.y())
        elif side == "right":
            pos = QPoint(top_left.x() + w + 2, top_left.y())
        elif side == "below_end":
            # Right edges aligned, opening leftwards: for a button at a pane's right end.
            pos = QPoint(top_left.x() + w - size.width(), top_left.y() + h + 2)
        else:
            pos = QPoint(top_left.x(), top_left.y() + h + 2)
        screen = anchor.screen() if hasattr(anchor, "screen") else None
        if screen is not None:
            area = screen.availableGeometry()
            x = min(max(pos.x(), area.left()), area.right() - size.width())
            y = min(max(pos.y(), area.top()), area.bottom() - size.height())
            pos = QPoint(x, y)
        self.move(pos)
        self.show()
        self.raise_()

    def show_at_point(self, global_pos: QPoint) -> None:
        """Open with the top-left corner at *global_pos* (a context-menu position), on screen."""
        from qtpy.QtGui import QGuiApplication

        self.adjustSize()
        size = self.sizeHint().expandedTo(self.minimumSizeHint())
        pos = QPoint(global_pos)
        screen = QGuiApplication.screenAt(global_pos)
        if screen is not None:
            area = screen.availableGeometry()
            pos = QPoint(
                min(max(pos.x(), area.left()), area.right() - size.width()),
                min(max(pos.y(), area.top()), area.bottom() - size.height()),
            )
        self.move(pos)
        self.show()
        self.raise_()

    def toggle_at(self, anchor: QWidget, *, side: str = "below") -> None:
        """Open beside *anchor*, or close if open (see :attr:`_REOPEN_GUARD`)."""
        if self.isVisible():
            self.close()
            return
        if not self._pinned and time.monotonic() - self._closed_at < self._REOPEN_GUARD:
            return
        self.show_at(anchor, side=side)

    # pinning
    def is_pinned(self) -> bool:
        """Whether the pop-over stays open as a floating window."""
        return self._pinned

    def set_pinned(self, pinned: bool) -> None:
        """Pin (floating tool window) or unpin (closes on the next outside click)."""
        pinned = bool(pinned)
        if pinned == self._pinned:
            return
        self._pinned = pinned
        visible = self.isVisible()
        pos = self.pos()
        flags = (Qt.Tool if pinned else Qt.Popup) | Qt.FramelessWindowHint
        self.setWindowFlags(flags)
        if self._pin.isChecked() != pinned:
            self._pin.setChecked(pinned)
        if visible:
            self.move(pos)
            self.show()

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt naming
        """Drag a pinned pop-over around by its header."""
        if obj is self._header and self._pinned:
            kind = event.type()
            if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_from = event.globalPosition().toPoint() - self.pos() if hasattr(
                    event, "globalPosition"
                ) else event.globalPos() - self.pos()
                return True
            if kind == QEvent.MouseMove and self._drag_from is not None:
                point = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
                self.move(point - self._drag_from)
                return True
            if kind == QEvent.MouseButtonRelease:
                self._drag_from = None
                return True
        return super().eventFilter(obj, event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self._closed_at = time.monotonic()
        super().hideEvent(event)
        self.closed.emit()

    def _restyle(self) -> None:
        self._frame.setStyleSheet(
            "QFrame#nvitkPopoverBody {"
            f" background-color: {design.COLOR_SURFACE};"
            f" border: 1px solid {design.COLOR_BORDER_STRONG}; border-radius: {design.RADIUS}px; }}"
        )
        self._title.setStyleSheet(
            f"color: {design.COLOR_MUTED}; font-size: {design.FONT_S}px; font-weight: bold;"
            " letter-spacing: 1px; background: transparent;"
        )


# Filters and choices
class ChipFilterBar(FlowRow):
    """Wrapping row of toggle chips, for narrowing a list by kind.

    Chips are ``(key, label, hue, tooltip)`` tuples (hue and tooltip optional). With
    *exclusive*, at most one is on. An "All" chip leads the row and is lit while
    nothing else is; :meth:`selected` is then empty, meaning "no filter".
    :attr:`changed` carries the selection as a ``frozenset`` of keys.
    """

    changed = Signal(object)

    def __init__(
        self,
        chips: Iterable[tuple[Any, ...]] = (),
        *,
        exclusive: bool = False,
        all_label: str | None = "All",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, h_spacing=design.SPACE_TIGHT, v_spacing=design.SPACE_TIGHT)
        self._exclusive = bool(exclusive)
        self._all_label = all_label
        self._buttons: dict[Any, QToolButton] = {}
        self._hues: dict[Any, str | None] = {}
        self._all: QToolButton | None = None
        self.set_chips(chips)
        design.add_theme_listener(self._restyle)

    def set_chips(self, chips: Iterable[tuple[Any, ...]]) -> None:
        """Replace the chips (the selection is cleared)."""
        flow = self.flow()
        while flow.count():
            item = flow.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self._buttons.clear()
        self._hues.clear()
        self._all = None
        if self._all_label:
            self._all = self._make_chip(self._all_label, None, "Show everything")
            self._all.setChecked(True)
            self._all.clicked.connect(lambda: self.set_selected(()))
            self.add(self._all)
        for spec in chips:
            key, label = spec[0], spec[1]
            hue = spec[2] if len(spec) > 2 else None
            tooltip = spec[3] if len(spec) > 3 else ""
            button = self._make_chip(str(label), hue, tooltip)
            button.toggled.connect(lambda on, k=key: self._on_toggled(k, on))
            self._buttons[key] = button
            self._hues[key] = hue
            self.add(button)
        self._restyle()
        self.refresh()

    def selected(self) -> frozenset:
        """Keys of the lit chips; empty means no filter."""
        return frozenset(k for k, b in self._buttons.items() if b.isChecked())

    def set_selected(self, keys: Iterable[Any], *, emit: bool = True) -> None:
        """Light exactly *keys*."""
        wanted = set(keys)
        for key, button in self._buttons.items():
            button.blockSignals(True)
            button.setChecked(key in wanted)
            button.blockSignals(False)
        self._sync_all()
        if emit:
            self.changed.emit(self.selected())

    def _on_toggled(self, key: Any, on: bool) -> None:
        if on and self._exclusive:
            for other, button in self._buttons.items():
                if other != key and button.isChecked():
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.blockSignals(False)
        self._sync_all()
        self.changed.emit(self.selected())

    def _sync_all(self) -> None:
        if self._all is not None:
            self._all.blockSignals(True)
            self._all.setChecked(not self.selected())
            self._all.blockSignals(False)

    def _make_chip(self, label: str, hue: str | None, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("nvitkChip")
        button.setText(label)
        button.setCheckable(True)
        button.setFocusPolicy(Qt.NoFocus)
        button.setProperty("nvitkHue", hue or "")
        if tooltip:
            button.setToolTip(tooltip)
        return button

    def _restyle(self) -> None:
        buttons = list(self._buttons.values()) + ([self._all] if self._all is not None else [])
        for button in buttons:
            hue = button.property("nvitkHue") or None
            color = _hue_color(hue, design.COLOR_ACCENT)
            button.setStyleSheet(
                "QToolButton#nvitkChip {"
                f" background: transparent; color: {design.COLOR_MUTED};"
                f" border: 1px solid {design.COLOR_BORDER}; border-radius: 9px;"
                f" padding: 1px 7px; font-size: {design.FONT_XS + 1}px; }}"
                "QToolButton#nvitkChip:hover {"
                f" color: {design.COLOR_TEXT}; border-color: {design.COLOR_BORDER_STRONG}; }}"
                "QToolButton#nvitkChip:checked {"
                f" color: {color}; background-color: {design.tint(color, 0.18)};"
                f" border-color: {design.tint(color, 0.6)}; font-weight: bold; }}"
            )


class SegmentedButtons(QWidget):
    """One choice out of a few, drawn as a joined row of buttons.

    *options* are ``(key, label, icon_name, tooltip)`` tuples (the last two
    optional). With *icon_only*, labels move into the tooltips.
    :attr:`changed` carries the key of the new choice.
    """

    changed = Signal(object)

    def __init__(
        self,
        options: Iterable[tuple[Any, ...]],
        *,
        current: Any = None,
        icon_only: bool = False,
        icon_size: int = 14,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Only as wide as its buttons: given more, a box layout spreads them apart.
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[Any, QToolButton] = {}
        self._icons: dict[Any, str | None] = {}
        specs = list(options)
        for index, spec in enumerate(specs):
            key, label = spec[0], str(spec[1])
            icon_name = spec[2] if len(spec) > 2 else None
            tooltip = spec[3] if len(spec) > 3 else ""
            button = QToolButton(self)
            button.setObjectName("nvitkSegment")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.setProperty("segment", "first" if index == 0 else "last" if index == len(specs) - 1 else "mid")
            if icon_name:
                button.setIconSize(QSize(icon_size, icon_size))
            if icon_only and icon_name:
                button.setToolTip(tooltip or label)
                button.setMinimumSize(icon_size + 10, icon_size + 8)
            else:
                button.setText(label)
                if tooltip:
                    button.setToolTip(tooltip)
                if icon_name:
                    button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self._group.addButton(button)
            self._buttons[key] = button
            self._icons[key] = icon_name
            row.addWidget(button)
        if specs:
            self.set_current(current if current in self._buttons else specs[0][0], emit=False)
        self._restyle()
        design.add_theme_listener(self._restyle)

    def current(self) -> Any:
        """Key of the chosen option."""
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return None

    def set_current(self, key: Any, *, emit: bool = False) -> None:
        """Choose *key* (signal only with *emit*)."""
        button = self._buttons.get(key)
        if button is None:
            return
        button.setChecked(True)
        if emit:
            self.changed.emit(key)

    def button(self, key: Any) -> QToolButton | None:
        """The button for option *key*."""
        return self._buttons.get(key)

    def _restyle(self) -> None:
        for key, button in self._buttons.items():
            name = self._icons.get(key)
            if name:
                button.setIcon(make_icon(name, design.COLOR_TEXT))
        radius = 4
        self.setStyleSheet(
            "QToolButton#nvitkSegment {"
            f" background-color: {design.COLOR_CONTROL}; color: {design.COLOR_TEXT};"
            f" border: 1px solid {design.COLOR_BORDER}; border-radius: 0px; margin: 0px;"
            f" padding: 2px 7px; font-size: {design.FONT_S}px; }}"
            'QToolButton#nvitkSegment[segment="first"] {'
            f" border-top-left-radius: {radius}px; border-bottom-left-radius: {radius}px; }}"
            'QToolButton#nvitkSegment[segment="last"] {'
            f" border-top-right-radius: {radius}px; border-bottom-right-radius: {radius}px; }}"
            'QToolButton#nvitkSegment[segment="mid"], QToolButton#nvitkSegment[segment="last"] {'
            " border-left: none; }"
            "QToolButton#nvitkSegment:hover {"
            f" background-color: {design.COLOR_CONTROL_HOVER}; }}"
            "QToolButton#nvitkSegment:checked {"
            f" background-color: {design.COLOR_ACCENT_DEEP}; color: {design.COLOR_ON_ACCENT};"
            f" border-color: {design.COLOR_ACCENT}; }}"
        )


# Pane chrome
class PaneHeader(QWidget):
    """Slim title strip over a viewport pane or an inspector.

    Small-caps *title* in *color* (an axis colour for a slice view, say), a muted
    info text on the right (``"51 / 101"``), and icon actions after it.
    """

    HEIGHT = 22

    def __init__(
        self,
        title: str = "",
        *,
        color: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._color = color
        self.setObjectName("nvitkPaneHeader")
        self.setFixedHeight(self.HEIGHT)
        self.setAttribute(Qt.WA_StyledBackground, True)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 2, 0)
        row.setSpacing(2)
        self._title = QLabel(title.upper(), self)
        row.addWidget(self._title)
        self._lead = QHBoxLayout()
        self._lead.setContentsMargins(4, 0, 0, 0)
        self._lead.setSpacing(2)
        row.addLayout(self._lead)
        row.addStretch(1)
        self._info = QLabel("", self)
        row.addWidget(self._info)
        self._actions = QHBoxLayout()
        self._actions.setContentsMargins(4, 0, 0, 0)
        self._actions.setSpacing(0)
        row.addLayout(self._actions)
        self._restyle()
        design.add_theme_listener(self._restyle)

    def set_title(self, text: str, color: str | None = None) -> None:
        """Change the title (and its colour, when given)."""
        self._title.setText(str(text).upper())
        if color is not None:
            self._color = color
            self._restyle()

    def set_info(self, text: str) -> None:
        """The muted text at the right (slice index, value readout...)."""
        self._info.setText(str(text))

    def add_lead_widget(self, widget: QWidget) -> QWidget:
        """Put *widget* just after the title (a layer chip, a mode selector)."""
        self._lead.addWidget(widget)
        return widget

    def add_widget(self, widget: QWidget) -> QWidget:
        """Put *widget* among the actions at the right end."""
        self._actions.addWidget(widget)
        return widget

    def add_action(
        self,
        icon_name: str,
        tooltip: str,
        callback: Callable[..., Any] | None = None,
        *,
        checkable: bool = False,
    ) -> IconButton:
        """Add an icon action at the right end; returns its button."""
        button = IconButton(icon_name, tooltip, checkable=checkable, icon_size=13, parent=self)
        button.setMinimumSize(20, 20)
        if callback is not None:
            if checkable:
                button.toggled.connect(callback)
            else:
                button.clicked.connect(lambda _=False: callback())
        self._actions.addWidget(button)
        return button

    def _restyle(self) -> None:
        self.setStyleSheet(
            "QWidget#nvitkPaneHeader {"
            f" background-color: {design.COLOR_SURFACE}; border: none;"
            f" border-bottom: 1px solid {design.COLOR_BORDER}; }}"
        )
        self._title.setStyleSheet(
            f"color: {self._color or design.COLOR_MUTED}; font-size: {design.FONT_S}px;"
            " font-weight: bold; letter-spacing: 1px; background: transparent;"
        )
        self._info.setStyleSheet(
            f"color: {design.COLOR_MUTED}; font-size: {design.FONT_S}px; background: transparent;"
        )


def hint_label(text: str = "", parent: QWidget | None = None) -> QLabel:
    """Muted small text for a hint or a status line under a control."""
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {design.COLOR_MUTED}; font-size: {design.FONT_S}px; font-weight: normal;")
    return label


def separator(parent: QWidget | None = None, *, vertical: bool = False) -> QFrame:
    """A one-pixel rule in the border colour."""
    line = QFrame(parent)
    line.setFrameShape(QFrame.VLine if vertical else QFrame.HLine)
    line.setStyleSheet(f"color: {design.COLOR_BORDER}; background-color: {design.COLOR_BORDER}; border: none;")
    if vertical:
        line.setFixedWidth(1)
    else:
        line.setFixedHeight(1)
    return line


# Resizing
class _HeightGrip(QWidget):
    """A thin bar under a widget; dragging it sets the widget's height."""

    def __init__(self, target: QWidget, owner: "ResizableBox") -> None:
        super().__init__(owner)
        self._target = target
        self._owner = owner
        self._start: tuple[int, int] | None = None
        self.setFixedHeight(7)
        self.setCursor(Qt.SizeVerCursor)  # a drag handle, not a button: the resize cursor stays
        self.setToolTip("Drag to resize")
        self.setMouseTracking(True)
        self._hover = False

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        from qtpy.QtGui import QColor, QPainter, QPen

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(design.COLOR_ACCENT if self._hover else design.COLOR_BORDER_STRONG))
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        y = self.height() // 2
        half = min(24, self.width() // 6)
        painter.drawLine(self.width() // 2 - half, y, self.width() // 2 + half, y)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            y = event.globalPosition().toPoint().y() if hasattr(event, "globalPosition") else event.globalPos().y()
            self._start = (y, self._target.height())

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._start is None:
            return
        y = event.globalPosition().toPoint().y() if hasattr(event, "globalPosition") else event.globalPos().y()
        self._owner.set_box_height(self._start[1] + y - self._start[0])

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if self._start is not None:
            self._start = None
            self._owner.remember()

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        self._owner.set_box_height(self._owner.default_height)
        self._owner.remember()


class ResizableBox(QWidget):
    """*widget* with a drag handle under it: its height is the user's, remembered per *pref_key*.

    For lists, tables and text boxes that used to be capped at a fixed height: the cap is now
    where they start, and dragging the handle (double-click: back to the default) changes it.
    """

    #: Preferences key holding ``{pref_key: height}``.
    PREFS_KEY = "gui.heights"

    def __init__(self, widget: QWidget, *, height: int = 160, minimum: int = 48,
                 pref_key: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.widget = widget
        self.default_height = int(height)
        self._minimum = int(minimum)
        self._pref_key = pref_key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        widget.setMinimumHeight(self._minimum)
        widget.setMaximumHeight(16777215)
        lay.addWidget(widget)
        lay.addWidget(_HeightGrip(widget, self))
        stored = self._stored()
        self.set_box_height(stored if stored else self.default_height)

    def _stored(self) -> int | None:
        if not self._pref_key:
            return None
        try:
            from nvitk.gui.core.prefs import load_prefs

            value = load_prefs().get(self.PREFS_KEY, {}).get(self._pref_key)
            return int(value) if value else None
        except Exception:  # noqa: BLE001
            return None

    def set_box_height(self, height: int) -> None:
        """Give the wrapped widget *height* pixels (never less than its minimum)."""
        self.widget.setFixedHeight(max(self._minimum, int(height)))

    def box_height(self) -> int:
        return self.widget.height()

    def remember(self) -> None:
        if not self._pref_key:
            return
        try:
            from nvitk.gui.core.prefs import load_prefs, save_prefs

            heights = dict(load_prefs().get(self.PREFS_KEY, {}) or {})
            heights[self._pref_key] = int(self.widget.height())
            save_prefs({self.PREFS_KEY: heights})
        except Exception:  # noqa: BLE001
            pass


def resizable(widget: QWidget, *, height: int = 160, minimum: int = 48, pref_key: str | None = None) -> ResizableBox:
    """*widget* in a :class:`ResizableBox` (a drag handle under it, height remembered)."""
    return ResizableBox(widget, height=height, minimum=minimum, pref_key=pref_key)


__all__ = [
    "Badge",
    "ResizableBox",
    "resizable",
    "ChipFilterBar",
    "IconButton",
    "PaneHeader",
    "Popover",
    "Section",
    "SegmentedButtons",
    "hint_label",
    "popover_button",
    "separator",
]
