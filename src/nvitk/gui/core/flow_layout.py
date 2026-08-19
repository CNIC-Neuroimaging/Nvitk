"""
A layout that wraps its items onto more rows instead of demanding more width.

Description
-----------
A toolbar of controls laid out with ``QHBoxLayout`` has a minimum width equal to the sum of every
control in it. That is invisible while the row is short and becomes a hard failure once it is not:
Qt propagates the layout's minimum up to the window, so the window can no longer be resized below
it — and if the sum exceeds the display, the window cannot be made to fit the screen at all. The
controls that fall past the edge are then unreachable, and no amount of dragging recovers them,
because the constraint is a *minimum*, not a preference.

This layout wraps instead. Its minimum width is the widest single item, not the sum, so a row of
twenty controls constrains a window exactly as much as a row of one.

Adapted from Qt's own flow-layout example. The arithmetic is theirs; the reason it is here is the
one above.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any

from qtpy.QtCore import QMargins, QPoint, QRect, QSize, Qt
from qtpy.QtWidgets import QLayout, QSizePolicy, QStyle, QWidget


class FlowLayout(QLayout):
    """
    Left-to-right layout that wraps to a new row when it runs out of width.

    Parameters
    ----------
    margin : int
        Outer margin, in pixels.
    h_spacing, v_spacing
        Horizontal / vertical gap between items. ``-1`` takes the style's default.
    """

    def __init__(
        self,
        parent: Any = None,
        *,
        margin: int = 0,
        h_spacing: int = 6,
        v_spacing: int = 4,
    ) -> None:
        """Create an empty flow layout."""
        super().__init__(parent)
        self._items: list[Any] = []
        self._h_spacing = int(h_spacing)
        self._v_spacing = int(v_spacing)
        self.setContentsMargins(QMargins(margin, margin, margin, margin))
        if parent is not None:
            # Without this the parent layout hands the widget ``sizeHint().height()`` — one row —
            # and every wrapped row below it is clipped out of view.
            policy = parent.sizePolicy()
            policy.setHeightForWidth(True)
            parent.setSizePolicy(policy)

    # ---- QLayout plumbing -----------------------------------------------------
    def addItem(self, item: Any) -> None:  # noqa: N802 - Qt naming
        """Append *item* (called by ``addWidget`` / ``addLayout``)."""
        self._items.append(item)

    def count(self) -> int:
        """Number of items."""
        return len(self._items)

    def itemAt(self, index: int) -> Any:  # noqa: N802 - Qt naming
        """Item at *index*, or ``None`` when out of range."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Any:  # noqa: N802 - Qt naming
        """Remove and return the item at *index*, or ``None`` when out of range."""
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Any:  # noqa: N802 - Qt naming
        """Nothing: the layout takes the width it is given and wraps within it.

        ``Qt.Orientation(0)`` rather than ``Qt.Orientations(...)`` — PyQt6 dropped the plural
        flag alias, and qtpy does not paper over it.
        """
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        """Height depends on width — that is the whole point of wrapping."""
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        """Height needed to lay the items out within *width*."""
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt naming
        """Place the items inside *rect*."""
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        """Preferred size: everything on one row."""
        margins = self.contentsMargins()
        width = height = 0
        for item in self._items:
            hint = item.sizeHint()
            width += hint.width() + self._h_spacing
            height = max(height, hint.height())
        return QSize(
            width + margins.left() + margins.right(),
            height + margins.top() + margins.bottom(),
        )

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt naming
        """
        The **widest single item**, not the sum.

        This is the line that fixes the locked window: with a horizontal layout the minimum is the
        sum of every control, and that sum is what Qt forces the window to be at least as wide as.
        """
        margins = self.contentsMargins()
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return QSize(
            size.width() + margins.left() + margins.right(),
            size.height() + margins.top() + margins.bottom(),
        )

    # ---- geometry -------------------------------------------------------------
    def _spacing(self, orientation: Any) -> int:
        """Configured gap, or the style's default for *orientation*."""
        configured = self._h_spacing if orientation == Qt.Horizontal else self._v_spacing
        if configured >= 0:
            return configured
        parent = self.parent()
        if parent is None or not hasattr(parent, "style"):
            return 6
        return parent.style().pixelMetric(
            QStyle.PM_LayoutHorizontalSpacing
            if orientation == Qt.Horizontal
            else QStyle.PM_LayoutVerticalSpacing,
            None,
            parent,
        )

    def _layout(self, rect: QRect, *, apply: bool) -> int:
        """Place (or merely measure) the items within *rect*; returns the total height."""
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y, row_height = effective.x(), effective.y(), 0

        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                # A hidden control must not reserve a slot — the brain-map options are hidden
                # unless that display is selected, and reserving their width would put the gap
                # back that this layout exists to remove.
                continue

            hint = item.sizeHint()
            width, height = hint.width(), hint.height()
            # An item wider than the whole row is clamped rather than allowed to run past the edge.
            # Nested wrapping rows are the case that matters: their sizeHint is "everything on one
            # line", so left unclamped they overflow instead of wrapping inside themselves, and the
            # controls past the edge become unreachable.
            if width > effective.width():
                width = effective.width()
                if item.hasHeightForWidth():
                    height = item.heightForWidth(width)
                elif widget is not None and widget.hasHeightForWidth():
                    height = widget.heightForWidth(width)

            next_x = x + width + self._spacing(Qt.Horizontal)
            if next_x - self._spacing(Qt.Horizontal) > effective.right() and row_height > 0:
                x = effective.x()
                y = y + row_height + self._spacing(Qt.Vertical)
                next_x = x + width + self._spacing(Qt.Horizontal)
                row_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), QSize(width, height)))
            x = next_x
            row_height = max(row_height, height)

        return y + row_height - rect.y() + margins.bottom()


class FlowRow(QWidget):
    """
    A widget holding a :class:`FlowLayout` that keeps its own height correct as it wraps.

    ``QBoxLayout`` is documented to honour ``heightForWidth``, and in practice does not do so
    reliably for a child whose height depends on its width: the row computes the right answer and
    the parent allocates one line anyway, clipping every wrapped line below it. Rather than fight
    that, the row pins its own minimum height on each resize — the height then always matches what
    the layout actually needs, whatever the parent decides to do.

    Use :meth:`add` to append controls.
    """

    def __init__(self, parent: Any = None, *, h_spacing: int = 6, v_spacing: int = 4) -> None:
        """Create an empty wrapping row."""
        super().__init__(parent)
        self._flow = FlowLayout(self, margin=0, h_spacing=h_spacing, v_spacing=v_spacing)

    def add(self, widget: QWidget) -> None:
        """Append *widget* to the row."""
        self._flow.addWidget(widget)

    def flow(self) -> FlowLayout:
        """The underlying layout, for callers that need to add a non-widget item."""
        return self._flow

    def _sync_height(self) -> None:
        """Pin the minimum height to what the wrapped content needs at the current width."""
        needed = self._flow.heightForWidth(max(self.width(), 1))
        if needed > 0 and needed != self.minimumHeight():
            self.setMinimumHeight(needed)
            self.updateGeometry()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        """Re-pin the height whenever the row is given a new width."""
        super().resizeEvent(event)
        self._sync_height()

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        """Controls are shown and hidden as the display changes, so re-measure on show."""
        super().showEvent(event)
        self._sync_height()

    def refresh(self) -> None:
        """Re-measure after controls have been shown or hidden."""
        self._flow.invalidate()
        self._sync_height()


__all__ = ["FlowLayout", "FlowRow"]
