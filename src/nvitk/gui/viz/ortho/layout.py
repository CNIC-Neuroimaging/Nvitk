"""Pane layout and persisted preferences of the orthogonal views."""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import (
    Qt,
    Signal,
)
from qtpy.QtWidgets import (
    QSplitter,
    QSplitterHandle,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import SPACE

#: Preference key holding the 3D card's toggles (``resample``, ``orientation_marker``)
#: and the panes' sizes (``panes``).
_ORTHO_PREFS_KEY = "ortho_view"


def _ortho_prefs() -> dict[str, Any]:
    """The 3D card's stored toggles (empty when nothing is stored)."""
    try:
        from nvitk.gui.core.prefs import load_prefs

        stored = load_prefs().get(_ORTHO_PREFS_KEY)
    except Exception:  # noqa: BLE001 (a preference must never break the panel)
        return {}
    return dict(stored) if isinstance(stored, dict) else {}


def _save_ortho_pref(key: str, value: Any) -> None:
    """Remember one 3D-card toggle for the next session."""
    try:
        from nvitk.gui.core.prefs import save_prefs

        save_prefs({_ORTHO_PREFS_KEY: {**_ortho_prefs(), key: value}})
    except Exception:  # noqa: BLE001
        pass


class _PaneHandle(QSplitterHandle):
    """A divider that evens the panes out on a double click."""

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        grid = getattr(self.splitter(), "grid", None)
        if grid is not None:
            grid.even()
        event.accept()


class _PaneSplitter(QSplitter):
    grid: "PaneGrid | None" = None

    def createHandle(self) -> QSplitterHandle:  # noqa: N802 - Qt naming
        return _PaneHandle(self.orientation(), self)


class PaneGrid(QWidget):
    """Two rows of two panes with draggable dividers.

    The row divider sizes the two rows; the column divider sizes the two
    columns, and dragging it in either row moves it in both, so the panes stay a
    grid. A pane dragged down to nothing collapses (drag the divider back to
    bring it out). A double click on a divider evens them all out again;
    :meth:`toggle_maximized` gives one pane the whole area and back.
    """

    #: The user moved a divider (or evened / maximised the panes).
    changed = Signal()

    def __init__(self, cells: list[list[QWidget]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows = _PaneSplitter(Qt.Vertical)
        self._cols = [_PaneSplitter(Qt.Horizontal), _PaneSplitter(Qt.Horizontal)]
        for splitter, row in zip(self._cols, cells):
            for widget in row:
                splitter.addWidget(widget)
            self._rows.addWidget(splitter)
        for splitter in (self._rows, *self._cols):
            splitter.grid = self
            splitter.setChildrenCollapsible(True)
            splitter.setHandleWidth(SPACE)
        self._cols[0].splitterMoved.connect(lambda *_: self._follow(0))
        self._cols[1].splitterMoved.connect(lambda *_: self._follow(1))
        self._rows.splitterMoved.connect(lambda *_: self._moved())
        #: ``(row, col)`` of the pane shown alone, and the sizes to go back to.
        self._maximized: tuple[int, int] | None = None
        self._before: dict[str, list[float]] | None = None
        self.set_fractions((1, 1), (1, 1))
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._rows)

    def _follow(self, moved: int) -> None:
        """The column divider was dragged in one row: the other row follows."""
        self._cols[1 - moved].setSizes(self._cols[moved].sizes())
        self._moved()

    def _moved(self) -> None:
        self._maximized = None
        self.changed.emit()

    @staticmethod
    def _shares(sizes: list[int]) -> list[float]:
        total = float(sum(sizes))
        return [s / total for s in sizes] if total > 0 else [0.5, 0.5]

    def fractions(self) -> dict[str, list[float]]:
        """The rows' and columns' shares of the area (each pair sums to 1)."""
        return {"rows": self._shares(self._rows.sizes()), "cols": self._shares(self._cols[0].sizes())}

    def set_fractions(self, rows: Any = None, cols: Any = None) -> bool:
        """Size the rows and columns by their shares; False when the values are unusable."""
        def _pair(values: Any) -> list[int] | None:
            try:
                pair = [max(float(v), 0.0) for v in values]
            except (TypeError, ValueError):
                return None
            if len(pair) != 2 or sum(pair) <= 0:
                return None
            # QSplitter spreads its size by these weights.
            return [int(round(10000 * v / sum(pair))) for v in pair]

        rows_px, cols_px = _pair(rows if rows is not None else (1, 1)), _pair(cols if cols is not None else (1, 1))
        if rows_px is None or cols_px is None:
            return False
        self._rows.setSizes(rows_px)
        for splitter in self._cols:
            splitter.setSizes(cols_px)
        return True

    def even(self) -> None:
        """All four panes the same size."""
        self.set_fractions((1, 1), (1, 1))
        self._moved()

    def maximized(self) -> tuple[int, int] | None:
        return self._maximized

    def toggle_maximized(self, row: int, col: int) -> None:
        """Show the pane at (*row*, *col*) alone, or put the grid back as it was."""
        if self._maximized == (row, col):
            before = self._before or {}
            self.set_fractions(before.get("rows"), before.get("cols"))
            self._maximized = None
        else:
            if self._maximized is None:
                self._before = self.fractions()
            self.set_fractions((1, 0) if row == 0 else (0, 1), (1, 0) if col == 0 else (0, 1))
            self._maximized = (row, col)
        self.changed.emit()


#: A layout: a pane id, or ``(orientation, children)`` / ``(orientation, children, weights)``
#: with orientation ``"h"`` (side by side) or ``"v"`` (stacked) and children layouts themselves.
LayoutSpec = Any


def spec_panes(spec: LayoutSpec) -> list[str]:
    """Pane ids a layout shows, in reading order."""
    if isinstance(spec, str):
        return [spec]
    out: list[str] = []
    for child in spec[1]:
        out.extend(spec_panes(child))
    return out


def _validate_spec(spec: LayoutSpec, known: set[str]) -> None:
    """Raise ``ValueError`` for a malformed layout or one naming an unknown pane."""
    if isinstance(spec, str):
        if spec not in known:
            raise ValueError(f"Unknown pane {spec!r} (known: {', '.join(sorted(known))})")
        return
    if not isinstance(spec, (tuple, list)) or len(spec) not in (2, 3) or spec[0] not in ("h", "v"):
        raise ValueError(f"A layout node is ('h' | 'v', [children], [weights]?), not {spec!r}")
    children = list(spec[1])
    if not children:
        raise ValueError("A layout node needs at least one child")
    if len(spec) == 3 and len(spec[2]) != len(children):
        raise ValueError("A layout node's weights must match its children")
    for child in children:
        _validate_spec(child, known)
    ids = spec_panes(spec)
    if len(ids) != len(set(ids)):
        raise ValueError("A pane can appear only once in a layout")


class LayoutGrid(QWidget):
    """Panes arranged by a nested splitter tree, rebuildable at runtime.

    *panes* maps an id (``"ax"``, ``"cor"``, ``"sag"``, ``"3d"``...) to its widget; a layout
    spec arranges any subset of them, e.g. today's 2x2::

        ("v", [("h", ["ax", "cor"]), ("h", ["sag", "3d"])])

    or three small axes beside a large 3D view::

        ("h", [("v", ["ax", "cor", "sag"]), "3d"], [1, 3])

    Panes a layout leaves out are parked in a hidden holder inside this widget: they keep
    their parent window, which a GL canvas needs (moved to another top-level window it loses
    its context). Every divider evens its panes on a double click. Sibling splitters of the
    same orientation and size move together (*link_siblings*), so a 2x2 stays a grid.
    :meth:`toggle_maximized` gives one pane the whole area and back; :meth:`set_pane_visible`
    hides one and lets its neighbours take the room.

    *floating* panes are never reparented after construction: a placeholder takes their slot in
    the splitters and the real widget, a direct child of the grid, is kept over it. That is for a
    GL canvas (Napari's 3D view): moved between splitters through a hidden parent, it comes back
    with invalid GL state; resized and moved in place, it does not.
    """

    #: A divider was moved, the panes evened or one maximised / restored.
    changed = Signal()

    def __init__(
        self,
        panes: dict[str, QWidget],
        spec: LayoutSpec,
        *,
        link_siblings: bool = True,
        floating: tuple[str, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        #: The real pane widgets, and what sits in the splitters for each (a placeholder for a
        #: floating pane, the widget itself otherwise).
        self._widgets = dict(panes)
        self._slots: dict[str, QWidget] = {}
        for pane_id in floating:
            real = self._widgets.get(pane_id)
            if real is None:
                continue
            slot = QWidget()
            slot.setObjectName(f"nvitkLayoutSlot_{pane_id}")
            slot.setMinimumSize(real.minimumSize())
            slot.installEventFilter(self)
            self._slots[pane_id] = slot
            real.setParent(self)
        self._panes = {pid: self._slots.get(pid, w) for pid, w in self._widgets.items()}
        self._link = bool(link_siblings)
        self._spec: LayoutSpec = None
        self._root_splitter: QSplitter | None = None
        self._splitters: dict[str, _PaneSplitter] = {}
        self._maximized: str | None = None
        self._before: dict[str, list[float]] | None = None
        self._hidden: set[str] = set()
        self._parking = QWidget(self)
        self._parking.setVisible(False)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.set_spec(spec)

    # structure
    def spec(self) -> LayoutSpec:
        """The layout in force."""
        return self._spec

    def pane_ids(self) -> list[str]:
        """Ids of the panes the layout shows."""
        return spec_panes(self._spec)

    def pane(self, pane_id: str) -> QWidget | None:
        """The widget registered as *pane_id*."""
        return self._widgets.get(pane_id)

    def set_spec(self, spec: LayoutSpec, fractions: dict[str, list[float]] | None = None) -> None:
        """Arrange the panes by *spec* (and size them by *fractions*, from :meth:`fractions`)."""
        _validate_spec(spec, set(self._panes))
        # Park everything first, so a pane moving between splitters never goes parentless.
        for widget in self._panes.values():
            widget.setParent(self._parking)
        old = self._root_splitter
        if old is not None:
            self._layout.removeWidget(old)
            old.hide()
            old.setParent(None)
            old.deleteLater()
        self._splitters = {}
        self._maximized = None
        self._spec = spec
        if isinstance(spec, str):
            # A single pane still sits in a splitter, so the API stays the same.
            spec = ("h", [spec])
        self._root_splitter = self._build(spec, "0")
        self._layout.addWidget(self._root_splitter)
        for pane_id in spec_panes(self._spec):
            self._panes[pane_id].setVisible(pane_id not in self._hidden)
        if fractions:
            self.set_fractions(fractions)
        else:
            self._apply_weights(spec, "0")
        self._sync_empty()
        self._sync_floating_later()
        self.changed.emit()

    def _build(self, spec: LayoutSpec, path: str) -> _PaneSplitter:
        splitter = _PaneSplitter(Qt.Horizontal if spec[0] == "h" else Qt.Vertical)
        splitter.grid = self  # type: ignore[assignment]
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(SPACE)
        splitter.setObjectName(f"nvitkLayoutSplitter_{path}")
        self._splitters[path] = splitter
        for index, child in enumerate(spec[1]):
            if isinstance(child, str):
                splitter.addWidget(self._panes[child])
            else:
                splitter.addWidget(self._build(child, f"{path}.{index}"))
        splitter.splitterMoved.connect(lambda *_a, p=path: self._on_moved(p))
        splitter.splitterMoved.connect(lambda *_a: self._sync_floating())
        return splitter

    def _apply_weights(self, spec: LayoutSpec, path: str) -> None:
        if isinstance(spec, str):
            return
        weights = list(spec[2]) if len(spec) == 3 else [1] * len(spec[1])
        self._splitters[path].setSizes(_weights_px(weights))
        for index, child in enumerate(spec[1]):
            self._apply_weights(child, f"{path}.{index}")

    # sizes
    def fractions(self) -> dict[str, list[float]]:
        """Each splitter's shares, keyed by its path in the tree (``"0"``, ``"0.1"``...)."""
        return {path: PaneGrid._shares(s.sizes()) for path, s in self._splitters.items()}

    def set_fractions(self, fractions: dict[str, list[float]] | None) -> bool:
        """Size the splitters by shares from :meth:`fractions`; False if they do not fit."""
        if not isinstance(fractions, dict):
            return False
        ok = True
        for path, shares in fractions.items():
            splitter = self._splitters.get(str(path))
            if splitter is None or not isinstance(shares, (list, tuple)) or len(shares) != splitter.count():
                ok = False
                continue
            try:
                splitter.setSizes(_weights_px([max(float(v), 0.0) for v in shares]))
            except (TypeError, ValueError, ZeroDivisionError):
                ok = False
        self._sync_floating_later()
        return ok

    def even(self) -> None:
        """Every splitter's panes the same size."""
        for splitter in self._splitters.values():
            splitter.setSizes(_weights_px([1] * splitter.count()))
        self._maximized = None
        self.changed.emit()

    def _on_moved(self, path: str) -> None:
        if self._link:
            moved = self._splitters[path]
            parent_path = path.rsplit(".", 1)[0] if "." in path else None
            for other_path, other in self._splitters.items():
                if (
                    other is not moved
                    and parent_path is not None
                    and other_path.rsplit(".", 1)[0] == parent_path
                    and other.orientation() == moved.orientation()
                    and other.count() == moved.count()
                ):
                    other.setSizes(moved.sizes())
        self._maximized = None
        self.changed.emit()

    # maximise / hide
    def maximized(self) -> str | None:
        """Id of the pane shown alone, if any."""
        return self._maximized

    def toggle_maximized(self, pane_id: str) -> None:
        """Show *pane_id* alone, or put the layout back as it was."""
        if pane_id not in self.pane_ids():
            return
        if self._maximized == pane_id:
            self.set_fractions(self._before or {})
            self._maximized = None
        else:
            if self._maximized is None:
                self._before = self.fractions()
            self._focus(pane_id)
            self._maximized = pane_id
        self.changed.emit()

    def _focus(self, pane_id: str) -> None:
        """Give every splitter on the way down to *pane_id* wholly to that branch."""
        target = self._panes[pane_id]
        for splitter in self._splitters.values():
            for index in range(splitter.count()):
                child = splitter.widget(index)
                if child is target or (isinstance(child, QSplitter) and child.isAncestorOf(target)):
                    sizes = [0] * splitter.count()
                    sizes[index] = 10000
                    splitter.setSizes(sizes)
                    break

    def is_pane_visible(self, pane_id: str) -> bool:
        """Whether *pane_id* is in the layout and not hidden."""
        return pane_id in self.pane_ids() and pane_id not in self._hidden

    def set_pane_visible(self, pane_id: str, visible: bool) -> None:
        """Hide *pane_id* (its neighbours take the room) or bring it back."""
        if pane_id not in self._panes:
            return
        if visible:
            self._hidden.discard(pane_id)
        else:
            self._hidden.add(pane_id)
        if pane_id in self.pane_ids():
            self._panes[pane_id].setVisible(bool(visible))
            self._sync_empty()
        self._sync_floating_later()
        self.changed.emit()

    # floating panes
    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt naming
        """A placeholder moved, resized, appeared or disappeared: its pane follows."""
        from qtpy.QtCore import QEvent

        if obj in self._slots.values() and event.type() in (
            QEvent.Resize, QEvent.Move, QEvent.Show, QEvent.Hide
        ):
            if event.type() == QEvent.Hide:
                self._sync_floating_later()
            else:
                self._sync_floating()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._sync_floating_later()

    def _sync_floating_later(self) -> None:
        """Sync now and once more after the layouts have settled (sizes land a turn later)."""
        from qtpy.QtCore import QTimer

        self._sync_floating()
        QTimer.singleShot(0, lambda: self._sync_floating(allow_hide=True))

    def _sync_floating(self, *, allow_hide: bool = False) -> None:
        """Lay each floating pane exactly over its placeholder, or hide it with it.

        Hiding waits for the settled pass (*allow_hide*): a layout switch parks every
        placeholder for a moment, and hiding a GL canvas only to show it again in the same turn
        is what leaves it drawing with stale GL state.
        """
        from qtpy.QtCore import QPoint, QRect

        shown = set(self.pane_ids())
        for pane_id, slot in self._slots.items():
            real = self._widgets[pane_id]
            try:
                visible = pane_id in shown and pane_id not in self._hidden and slot.isVisible()
            except RuntimeError:  # the slot is being torn down
                continue
            if visible and slot.width() > 0 and slot.height() > 0:
                real.setGeometry(QRect(slot.mapTo(self, QPoint(0, 0)), slot.size()))
                if not real.isVisible():
                    real.show()
                real.raise_()
            elif allow_hide and real.isVisible():
                real.hide()

    def _sync_empty(self) -> None:
        """Hide a splitter all of whose panes are hidden (and show it again)."""
        for path in sorted(self._splitters, key=len, reverse=True):
            splitter = self._splitters[path]
            any_shown = any(not splitter.widget(i).isHidden() for i in range(splitter.count()))
            if path != "0":
                splitter.setVisible(any_shown)


def _weights_px(weights: list[float]) -> list[int]:
    """Splitter sizes from relative weights (QSplitter spreads its size by these)."""
    total = float(sum(weights))
    if total <= 0:
        return [10000 // max(len(weights), 1)] * len(weights)
    return [int(round(10000 * float(w) / total)) for w in weights]
