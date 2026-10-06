"""Folders and subfolders in Napari's layer list.

Napari's layer list is flat. This adds folders to it without replacing it — so
the eye toggles, renaming, drag-reordering, thumbnails and nvitk's label ▾ chips
all keep working — by treating a folder as a *run of consecutive layers that
share a folder path*:

* a layer's folder is a ``/``-separated path stored in its metadata
  (``"CT/Segmentations"``); subfolders are simply longer paths;
* each folder's header is painted above the first row of its run, and its
  members are indented beneath it (the delegate in
  :mod:`nvitk.gui.labels.layer_list` draws both);
* collapsing a folder hides its member rows; the first one stays as a
  header-only row, so a collapsed folder still occupies one line.

Interaction
-----------
On a header: the arrow folds and unfolds, the box shows or hides every layer in
the folder, a click selects them all (so Napari's delete, duplicate… act on the
folder), a double click renames it and the right button opens the folder menu.
Dragging a layer *between two members* of a folder puts it in that folder, and
dragging it out takes it out; dragging a collapsed folder's row moves the whole
folder. The 📁 button beside Napari's layer buttons creates folders from the
selection and moves layers in and out.

Folders exist through their members: one whose last layer leaves disappears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from qtpy.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen

#: Metadata key holding a layer's folder path (``""`` / missing = top level).
FOLDER_KEY = "nvitk_folder"
#: Height of one folder header band, and the indent per nesting level (px).
HEADER_H = 24
INDENT = 14
_ARROW_W = 14
_BOX = 13


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────


def normalize_path(path: Any) -> str:
    """``" a //b/ "`` → ``"a/b"``: no empty parts, no surrounding blanks."""
    parts = [p.strip() for p in str(path or "").replace("\\", "/").split("/")]
    return "/".join(p for p in parts if p)


def folder_of(layer: Any) -> str:
    """*layer*'s folder path (``""`` at the top level)."""
    meta = getattr(layer, "metadata", None)
    if not isinstance(meta, dict):
        return ""
    return normalize_path(meta.get(FOLDER_KEY, ""))


def _set_folder(layer: Any, path: str) -> None:
    """Record *path* as *layer*'s folder (removing the key at the top level)."""
    meta = getattr(layer, "metadata", None)
    if not isinstance(meta, dict):
        return
    path = normalize_path(path)
    if path:
        meta[FOLDER_KEY] = path
    else:
        meta.pop(FOLDER_KEY, None)


def ancestors(path: str) -> list[str]:
    """``"a/b/c"`` → ``["a", "a/b", "a/b/c"]``."""
    parts = normalize_path(path).split("/") if path else []
    return ["/".join(parts[: i + 1]) for i in range(len(parts))]


def within(path: str, folder: str) -> bool:
    """True when *path* is *folder* or one of its subfolders."""
    if not folder:
        return True
    return path == folder or path.startswith(folder + "/")


def common_folder(a: str, b: str) -> str:
    """The deepest folder containing both *a* and *b* (``""`` when none)."""
    pa = a.split("/") if a else []
    pb = b.split("/") if b else []
    out = []
    for x, y in zip(pa, pb):
        if x != y:
            break
        out.append(x)
    return "/".join(out)


def folder_name(path: str) -> str:
    """The last part of a folder path."""
    return path.rsplit("/", 1)[-1] if path else ""


def parent_folder(path: str) -> str:
    """The folder containing *path* (``""`` for a top-level folder)."""
    return path.rsplit("/", 1)[0] if "/" in path else ""


@dataclass
class RowEntry:
    """How one layer's row is drawn.

    ``headers`` are the folders whose header band sits on top of this row (outer
    first); ``depth`` indents the layer item; ``item_visible`` is False on the
    header-only row of a collapsed folder; ``hidden`` rows are not shown at all.
    """

    headers: list[str] = field(default_factory=list)
    depth: int = 0
    item_visible: bool = True
    hidden: bool = False

    @property
    def band(self) -> int:
        """Height of the header band, in pixels."""
        return HEADER_H * len(self.headers)


def compute_layout(layers: Iterable[Any], collapsed: set[str], *, enabled: bool = True) -> dict[int, RowEntry]:
    """Row entries by ``id(layer)`` for *layers* in list order (bottom → top).

    Rows are laid out top-down as Napari shows them; a folder's header starts on
    the first row of each run of its members.
    """
    ordered = list(layers)
    entries: dict[int, RowEntry] = {}
    previous: list[str] = []
    for layer in reversed(ordered):
        path = folder_of(layer) if enabled else ""
        parts = path.split("/") if path else []
        shared = 0
        for x, y in zip(previous, parts):
            if x != y:
                break
            shared += 1
        chain = ["/".join(parts[: i + 1]) for i in range(len(parts))]
        folded = next((i for i, p in enumerate(chain) if p in collapsed), None)
        if folded is None:
            entry = RowEntry(chain[shared:], len(parts), True, False)
        elif folded >= shared:
            # A collapsed folder starts here: its header (and any enclosing new
            # ones) on a row of its own, the members below it hidden.
            entry = RowEntry(chain[shared: folded + 1], folded, False, False)
        else:
            entry = RowEntry([], len(parts), False, True)
        entries[id(layer)] = entry
        previous = parts
    return entries


def header_depth(path: str) -> int:
    """Nesting level of a folder header (0 for a top-level folder)."""
    return max(0, len(path.split("/")) - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Header geometry and painting (shared by the delegate and the mouse filter)
# ──────────────────────────────────────────────────────────────────────────────


def header_rects(row_rect: QRect, entry: RowEntry) -> list[tuple[str, QRect]]:
    """``(folder, rect)`` of each header band on a row."""
    out = []
    for j, path in enumerate(entry.headers):
        indent = header_depth(path) * INDENT
        out.append((
            path,
            QRect(row_rect.left() + indent, row_rect.top() + j * HEADER_H, row_rect.width() - indent, HEADER_H),
        ))
    return out


def header_parts(rect: QRect) -> dict[str, QRect]:
    """Hit areas inside one header: fold arrow, visibility box, the rest (name)."""
    arrow = QRect(rect.left() + 2, rect.top(), _ARROW_W, rect.height())
    box = QRect(rect.right() - _BOX - 6, rect.top() + (rect.height() - _BOX) // 2, _BOX, _BOX)
    name = QRect(arrow.right() + 2, rect.top(), max(box.left() - arrow.right() - 6, 0), rect.height())
    return {"arrow": arrow, "box": box, "name": name}


def item_rect(row_rect: QRect, entry: RowEntry) -> QRect:
    """Where the layer item itself is drawn on a row: below the headers, indented."""
    indent = entry.depth * INDENT
    return QRect(
        row_rect.left() + indent,
        row_rect.top() + entry.band,
        row_rect.width() - indent,
        row_rect.height() - entry.band,
    )


def _folder_icon(painter: QPainter, rect: QRect, colour: QColor, open_: bool) -> None:
    """A small painted folder (no font or icon-theme dependency)."""
    w, h = 14.0, 10.0
    x = rect.left() + (rect.width() - w) / 2.0
    y = rect.top() + (rect.height() - h) / 2.0
    path = QPainterPath()
    path.moveTo(x, y + 2)
    path.lineTo(x + 5, y + 2)
    path.lineTo(x + 6.5, y)
    path.lineTo(x + w * 0.55, y)
    path.lineTo(x + w * 0.62, y + 2)
    path.lineTo(x + w, y + 2)
    path.lineTo(x + w, y + h)
    path.lineTo(x, y + h)
    path.closeSubpath()
    fill = QColor(colour)
    fill.setAlphaF(0.35 if open_ else 0.6)
    painter.setPen(QPen(colour, 1.1))
    painter.setBrush(fill)
    painter.drawPath(path)


def paint_header(
    painter: QPainter,
    option: Any,
    rect: QRect,
    path: str,
    *,
    collapsed: bool,
    count: int,
    visibility: str,
    selected: bool,
) -> None:
    """Draw one folder header band."""
    pal = option.palette
    text = pal.color(pal.ColorRole.Text)
    muted = QColor(text)
    muted.setAlphaF(0.6)
    accent = pal.color(pal.ColorRole.Highlight)
    parts = header_parts(rect)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    bg = QColor(accent if selected else pal.color(pal.ColorRole.AlternateBase))
    bg.setAlphaF(0.35 if selected else 0.55)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(bg)
    painter.drawRoundedRect(QRectF(rect).adjusted(1, 1.5, -1, -1.5), 4, 4)

    # Fold arrow.
    painter.setPen(text)
    painter.drawText(parts["arrow"], int(Qt.AlignmentFlag.AlignCenter), "▸" if collapsed else "▾")
    # Folder glyph, name and count.
    name_rect = parts["name"]
    icon_rect = QRect(name_rect.left(), name_rect.top(), 18, name_rect.height())
    _folder_icon(painter, icon_rect, text, not collapsed)
    fm = painter.fontMetrics()
    label_rect = QRect(icon_rect.right() + 5, name_rect.top(), max(name_rect.width() - 23, 0), name_rect.height())
    suffix = f"  {count}"
    name = fm.elidedText(folder_name(path), Qt.TextElideMode.ElideRight, max(label_rect.width() - fm.horizontalAdvance(suffix), 10))
    font = painter.font()
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(text)
    painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), name)
    used = painter.fontMetrics().horizontalAdvance(name)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(muted)
    painter.drawText(
        label_rect.adjusted(used, 0, 0, 0),
        int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
        suffix,
    )
    # Visibility box: filled when every layer shows, half when some do.
    box = QRectF(parts["box"]).adjusted(0.5, 0.5, -0.5, -0.5)
    painter.setPen(QPen(accent if visibility != "none" else muted, 1.2))
    painter.setBrush(accent if visibility == "all" else Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(box, 2.5, 2.5)
    if visibility == "some":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(box.adjusted(3, 3, -3, -3), 1.5, 1.5)
    elif visibility == "all":
        tick = QPainterPath()
        tick.moveTo(box.left() + 2.5, box.center().y() + 0.5)
        tick.lineTo(box.left() + 5.0, box.bottom() - 2.5)
        tick.lineTo(box.right() - 2.0, box.top() + 3.0)
        painter.setPen(QPen(pal.color(pal.ColorRole.HighlightedText), 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(tick)
    painter.restore()


def paint_guides(painter: QPainter, option: Any, row_rect: QRect, entry: RowEntry) -> None:
    """Faint vertical lines marking the folders an indented row sits in."""
    if entry.depth <= 0:
        return
    pal = option.palette
    colour = QColor(pal.color(pal.ColorRole.Text))
    colour.setAlphaF(0.18)
    painter.save()
    painter.setPen(QPen(colour, 1))
    top = row_rect.top() + entry.band
    for level in range(entry.depth):
        x = row_rect.left() + level * INDENT + INDENT // 2
        painter.drawLine(x, top, x, row_rect.bottom())
    painter.restore()


# ──────────────────────────────────────────────────────────────────────────────
# The model
# ──────────────────────────────────────────────────────────────────────────────


class LayerFolders(QObject):
    """Folder state of one viewer's layer list, and the operations on it."""

    changed = Signal()

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        self._viewer = viewer
        self.enabled = True
        self.collapsed: set[str] = set()
        self._layout: dict[int, RowEntry] | None = None
        self._view: Any | None = None
        self._delegate: Any | None = None
        self._moving = False
        self._pending_moves: list[Any] = []
        self._sync_timer = QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.setInterval(0)
        self._sync_timer.timeout.connect(self.sync_view)
        self._moves_timer = QTimer(self)
        self._moves_timer.setSingleShot(True)
        self._moves_timer.setInterval(0)
        self._moves_timer.timeout.connect(self._resolve_moves)
        events = viewer.layers.events
        events.inserted.connect(self._on_inserted)
        events.removed.connect(self._on_changed)
        events.moved.connect(self._on_moved)
        events.reordered.connect(self._on_changed)
        for layer in viewer.layers:
            self._watch(layer)

    # -- queries ----------------------------------------------------------------

    def layers(self) -> list[Any]:
        """The viewer's layers, bottom to top."""
        return list(self._viewer.layers)

    def layout(self) -> dict[int, RowEntry]:
        """Row entries by layer id, recomputed after any change."""
        if self._layout is None:
            self._layout = compute_layout(self.layers(), self.collapsed, enabled=self.enabled)
        return self._layout

    def entry(self, layer: Any) -> RowEntry:
        """How *layer*'s row is drawn (a plain row when unknown)."""
        if layer is None:
            return RowEntry()
        return self.layout().get(id(layer), RowEntry())

    def members(self, folder: str) -> list[Any]:
        """Layers in *folder* or its subfolders, bottom to top."""
        return [layer for layer in self.layers() if folder and within(folder_of(layer), folder)]

    def folders(self) -> list[str]:
        """Every folder path in use (with its ancestors), top of the list first."""
        seen: dict[str, None] = {}
        for layer in reversed(self.layers()):
            for path in ancestors(folder_of(layer)):
                seen.setdefault(path, None)
        return list(seen)

    def visibility(self, folder: str) -> str:
        """``"all"``, ``"none"`` or ``"some"`` of *folder*'s layers visible."""
        states = [bool(getattr(layer, "visible", True)) for layer in self.members(folder)]
        if states and all(states):
            return "all"
        if not any(states):
            return "none"
        return "some"

    def is_selected(self, folder: str) -> bool:
        """True when every layer of *folder* is selected."""
        members = self.members(folder)
        selection = self._viewer.layers.selection
        return bool(members) and all(layer in selection for layer in members)

    # -- operations ---------------------------------------------------------------

    def unique_name(self, parent: str, name: str = "Folder") -> str:
        """*name* under *parent*, numbered if a folder there already has it."""
        name = normalize_path(name).replace("/", "-") or "Folder"
        existing = set(self.folders())
        candidate, n = name, 2
        while (f"{parent}/{candidate}" if parent else candidate) in existing:
            candidate = f"{name} {n}"
            n += 1
        return candidate

    def create_folder(self, layers: Iterable[Any], name: str = "Folder", parent: str = "") -> str:
        """Put *layers* in a new folder *name* (under *parent*); returns its path."""
        layers = [layer for layer in layers if layer is not None]
        if not layers:
            raise ValueError("Select the layers to put in the folder.")
        parent = normalize_path(parent)
        path = f"{parent}/{self.unique_name(parent, name)}" if parent else self.unique_name("", name)
        self.move_to_folder(layers, path)
        return path

    def move_to_folder(self, layers: Iterable[Any], folder: str) -> None:
        """Put *layers* in *folder* (``""`` = top level), keeping every folder in one run."""
        folder = normalize_path(folder)
        layers = [layer for layer in layers if layer is not None]
        if not layers:
            return
        moving = {id(layer) for layer in layers}
        old = {id(layer): folder_of(layer) for layer in layers}
        for layer in layers:
            _set_folder(layer, folder)
        # Where the run goes: on top of the deepest existing folder on the path,
        # or — leaving every folder — just above the top-level run it was in.
        anchor = None
        targets = list(reversed(ancestors(folder))) if folder else []
        for path in targets:
            others = [i for i, ly in enumerate(self.layers()) if id(ly) not in moving and within(folder_of(ly), path)]
            if others:
                anchor = max(others) + 1
                break
        if anchor is None and not folder:
            tops = {ancestors(p)[0] for p in old.values() if p}
            others = [
                i for i, ly in enumerate(self.layers())
                if id(ly) not in moving and any(within(folder_of(ly), t) for t in tops)
            ]
            if others:
                anchor = max(others) + 1
        if anchor is None:
            anchor = max(self._index(layer) for layer in layers) + 1
        self._move(layers, anchor)
        self.invalidate()

    def remove_from_folder(self, layers: Iterable[Any]) -> None:
        """Take *layers* out to the top level."""
        self.move_to_folder(layers, "")

    def rename_folder(self, folder: str, new_name: str) -> str:
        """Rename *folder* (its subfolders follow); returns the new path."""
        folder = normalize_path(folder)
        new_name = normalize_path(new_name).replace("/", "-")
        if not folder or not new_name:
            return folder
        parent = parent_folder(folder)
        if new_name == folder_name(folder):
            return folder
        target = f"{parent}/{self.unique_name(parent, new_name)}" if parent else self.unique_name("", new_name)
        for layer in self.members(folder):
            path = folder_of(layer)
            _set_folder(layer, target + path[len(folder):])
        self.collapsed = {target + p[len(folder):] if within(p, folder) else p for p in self.collapsed}
        self.invalidate()
        return target

    def ungroup(self, folder: str) -> None:
        """Dissolve *folder*: its layers and subfolders move up one level."""
        folder = normalize_path(folder)
        parent = parent_folder(folder)
        for layer in self.members(folder):
            rest = folder_of(layer)[len(folder):].lstrip("/")
            _set_folder(layer, "/".join(p for p in (parent, rest) if p))
        self.collapsed = {p for p in self.collapsed if not within(p, folder)}
        self.invalidate()

    def delete_folder(self, folder: str) -> int:
        """Remove *folder*'s layers from the viewer; returns how many."""
        members = self.members(folder)
        for layer in members:
            self._viewer.layers.remove(layer)
        self.collapsed = {p for p in self.collapsed if not within(p, folder)}
        self.invalidate()
        return len(members)

    def set_collapsed(self, folder: str, collapsed: bool) -> None:
        """Fold or unfold *folder*."""
        if collapsed:
            self.collapsed.add(folder)
        else:
            self.collapsed.discard(folder)
        self.invalidate()

    def toggle_collapsed(self, folder: str) -> bool:
        """Flip *folder*'s fold; True when it is now collapsed."""
        now = folder not in self.collapsed
        self.set_collapsed(folder, now)
        return now

    def set_all_collapsed(self, collapsed: bool) -> None:
        """Fold or unfold every folder."""
        self.collapsed = set(self.folders()) if collapsed else set()
        self.invalidate()

    def set_folder_visible(self, folder: str, visible: bool) -> None:
        """Show or hide every layer in *folder*."""
        for layer in self.members(folder):
            layer.visible = bool(visible)
        self.invalidate()

    def select_folder(self, folder: str, *, add: bool = False) -> None:
        """Select every layer in *folder* (added to the selection with *add*)."""
        members = self.members(folder)
        if not members:
            return
        selection = self._viewer.layers.selection
        if not add:
            selection.clear()
        selection.update(members)
        # Not ``active``: setting it in Napari selects that one layer only.
        try:
            selection.current = members[-1]
        except Exception:  # noqa: BLE001 — older Napari without ``current``
            pass

    def set_enabled(self, enabled: bool) -> None:
        """Show the folder structure, or the plain flat list (memberships are kept)."""
        self.enabled = bool(enabled)
        self.invalidate()

    # -- view ---------------------------------------------------------------------

    def attach_view(self, view: Any, delegate: Any) -> None:
        """Keep *view*'s hidden rows and row heights in step with the folders."""
        self._view = view
        self._delegate = delegate
        self.invalidate()

    def invalidate(self) -> None:
        """Forget the layout and re-sync the view on the next event-loop turn."""
        self._layout = None
        self._sync_timer.start()
        self.changed.emit()

    def sync_view(self) -> None:
        """Hide the rows inside collapsed folders and have the view re-measure rows."""
        view = self._view
        if view is None:
            return
        try:
            from napari._qt.containers._base_item_model import ItemRole

            model = view.model()
            layout = self.layout()
            for row in range(model.rowCount()):
                layer = model.index(row, 0).data(ItemRole)
                entry = layout.get(id(layer), RowEntry())
                if view.isRowHidden(row) != entry.hidden:
                    view.setRowHidden(row, entry.hidden)
            view.doItemsLayout()
            view.viewport().update()
        except RuntimeError:  # the view is being torn down
            pass

    # -- list events ----------------------------------------------------------------

    def _index(self, layer: Any) -> int:
        return self.layers().index(layer)

    def _move(self, layers: list[Any], anchor: int) -> None:
        """Move *layers* (kept in their order) to sit just below list index *anchor*."""
        indices = sorted(self._index(layer) for layer in layers)
        if indices == list(range(anchor - len(indices), anchor)):
            return
        self._moving = True
        try:
            self._viewer.layers.move_multiple(indices, anchor)
        finally:
            self._moving = False

    def _watch(self, layer: Any) -> None:
        """Repaint folder headers when a member's visibility changes."""
        emitter = getattr(getattr(layer, "events", None), "visible", None)
        if emitter is not None:
            emitter.connect(self._on_visibility)

    def _on_visibility(self, _event: Any = None) -> None:
        view = self._view
        if view is not None:
            try:
                view.viewport().update()
            except RuntimeError:
                pass

    def _on_changed(self, _event: Any = None) -> None:
        self.invalidate()

    def _on_inserted(self, event: Any) -> None:
        layer = getattr(event, "value", None)
        if layer is not None:
            self._watch(layer)
            if not self._moving:
                self._pending_moves.append(layer)
                self._moves_timer.start()
        self.invalidate()

    def _on_moved(self, event: Any) -> None:
        if self._moving:
            return
        layer = getattr(event, "value", None)
        if layer is not None:
            self._pending_moves.append(layer)
            self._moves_timer.start()
        self.invalidate()

    def _resolve_moves(self) -> None:
        """Give dropped or inserted layers the folder of where they landed.

        A layer between two members of a folder joins it; one that no longer
        touches its folder leaves it. A collapsed folder dragged by its row
        brings the rest of the folder along.
        """
        pending, self._pending_moves = self._pending_moves, []
        layers = self.layers()
        alive = [ly for ly in dict.fromkeys(pending) if ly in layers]
        for layer in alive:
            own = folder_of(layer)
            folded = next((p for p in ancestors(own) if p in self.collapsed), None)
            if folded is not None:
                others = [ly for ly in self.members(folded) if ly is not layer]
                if others:
                    self._move(others, self._index(layer))
                continue
            layers = self.layers()
            idx = layers.index(layer)
            above = folder_of(layers[idx + 1]) if idx + 1 < len(layers) else ""
            below = folder_of(layers[idx - 1]) if idx > 0 else ""
            if own and (within(above, own) or within(below, own)):
                continue
            _set_folder(layer, common_folder(above, below))
        self.invalidate()


# ──────────────────────────────────────────────────────────────────────────────
# Mouse on the headers
# ──────────────────────────────────────────────────────────────────────────────


class _HeaderMouse(QObject):
    """Handles clicks on folder headers before the list view sees them.

    Consumed here, a header click never changes the selection by accident, starts
    a drag, or opens Napari's layer context menu.
    """

    def __init__(self, view: Any, folders: LayerFolders) -> None:
        super().__init__(view)
        self._view = view
        self._folders = folders

    def _hit(self, pos: QPoint) -> tuple[str, str] | None:
        """``(folder, part)`` under *pos* — part is arrow / box / name — or ``None``."""
        from napari._qt.containers._base_item_model import ItemRole

        index = self._view.indexAt(pos)
        if not index.isValid():
            return None
        entry = self._folders.entry(index.data(ItemRole))
        if not entry.headers:
            return None
        for path, rect in header_rects(self._view.visualRect(index), entry):
            if rect.contains(pos):
                for part, area in header_parts(rect).items():
                    if area.contains(pos):
                        return path, part
                return path, "name"
            if rect.top() <= pos.y() <= rect.bottom():
                return path, "name"  # in the band, left of the indent
        return None

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt naming
        kind = event.type()
        if kind not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):
            return False
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        hit = self._hit(pos)
        if hit is None:
            return False
        path, part = hit
        button = event.button()
        if kind == QEvent.Type.MouseButtonPress:
            if button == Qt.MouseButton.LeftButton:
                if part == "arrow":
                    self._folders.toggle_collapsed(path)
                elif part == "box":
                    self._folders.set_folder_visible(path, self._folders.visibility(path) != "all")
                else:
                    add = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
                    self._folders.select_folder(path, add=add)
            elif button == Qt.MouseButton.RightButton:
                gpos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
                folder_menu(self._folders, path, self._view).exec_(gpos)
            return True
        if kind == QEvent.Type.MouseButtonDblClick and button == Qt.MouseButton.LeftButton:
            if part == "name":
                prompt_rename(self._folders, path, self._view)
            elif part == "arrow":
                self._folders.toggle_collapsed(path)
            return True
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Menus and the toolbar button
# ──────────────────────────────────────────────────────────────────────────────


def prompt_rename(folders: LayerFolders, folder: str, parent: Any = None) -> None:
    """Ask for a new name for *folder*."""
    from qtpy.QtWidgets import QInputDialog

    name, ok = QInputDialog.getText(parent, "Rename folder", "Folder name:", text=folder_name(folder))
    if ok and name.strip():
        folders.rename_folder(folder, name)


def prompt_new_folder(folders: LayerFolders, layers: list[Any], parent_path: str = "", parent: Any = None) -> str | None:
    """Ask for a name and put *layers* in a new folder; returns its path."""
    from qtpy.QtWidgets import QInputDialog

    from nvitk.gui.tools.runner import notify

    if not layers:
        notify("Select the layers to put in the new folder.", error=True)
        return None
    suggestion = folders.unique_name(parent_path, "Folder")
    name, ok = QInputDialog.getText(parent, "New folder", "Folder name:", text=suggestion)
    if not ok or not name.strip():
        return None
    return folders.create_folder(layers, name, parent=parent_path)


def _selected_layers(folders: LayerFolders) -> list[Any]:
    selection = folders._viewer.layers.selection
    return [layer for layer in folders.layers() if layer in selection]


def folder_menu(folders: LayerFolders, folder: str, parent: Any = None) -> Any:
    """Context menu of one folder header."""
    from qtpy.QtWidgets import QMenu, QMessageBox

    menu = QMenu(parent)
    collapsed = folder in folders.collapsed
    menu.addAction("Expand" if collapsed else "Collapse", lambda: folders.toggle_collapsed(folder))
    menu.addAction("Select its layers", lambda: folders.select_folder(folder))
    vis = folders.visibility(folder)
    menu.addAction("Hide its layers" if vis == "all" else "Show its layers",
                   lambda: folders.set_folder_visible(folder, vis != "all"))
    menu.addSeparator()
    menu.addAction("Rename…", lambda: prompt_rename(folders, folder, parent))
    menu.addAction(
        "New subfolder from selected layers…",
        lambda: prompt_new_folder(folders, _selected_layers(folders), folder, parent),
    )
    menu.addAction("Move selected layers here", lambda: folders.move_to_folder(_selected_layers(folders), folder))
    menu.addSeparator()
    menu.addAction("Ungroup (keep the layers)", lambda: folders.ungroup(folder))

    def _delete() -> None:
        n = len(folders.members(folder))
        answer = QMessageBox.question(
            parent, "Delete folder",
            f"Remove the folder “{folder_name(folder)}” and its {n} layer(s) from the viewer?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            folders.delete_folder(folder)

    menu.addAction("Delete folder and its layers…", _delete)
    return menu


def folders_menu(folders: LayerFolders, parent: Any = None) -> Any:
    """The toolbar button's menu: create, move in and out, fold, switch off."""
    from qtpy.QtWidgets import QMenu

    menu = QMenu(parent)
    selected = _selected_layers(folders)
    new = menu.addAction("New folder from selected layers…", lambda: prompt_new_folder(folders, selected, "", parent))
    new.setEnabled(bool(selected))
    move = menu.addMenu("Move selected layers to")
    move.setEnabled(bool(selected))
    for path in folders.folders():
        move.addAction("    " * header_depth(path) + folder_name(path), lambda p=path: folders.move_to_folder(selected, p))
    if folders.folders():
        move.addSeparator()
    move.addAction("New folder…", lambda: prompt_new_folder(folders, selected, "", parent))
    out = menu.addAction("Take selected layers out of their folder", lambda: folders.remove_from_folder(selected))
    out.setEnabled(any(folder_of(layer) for layer in selected))
    menu.addSeparator()
    has = bool(folders.folders())
    menu.addAction("Collapse all folders", lambda: folders.set_all_collapsed(True)).setEnabled(has)
    menu.addAction("Expand all folders", lambda: folders.set_all_collapsed(False)).setEnabled(has)
    menu.addSeparator()
    show = menu.addAction("Show folders in the layer list")
    show.setCheckable(True)
    show.setChecked(folders.enabled)
    show.toggled.connect(folders.set_enabled)
    return menu


def _toolbar_button(viewer: Any, folders: LayerFolders) -> Any | None:
    """A folder button beside Napari's new-layer buttons."""
    from qtpy.QtWidgets import QPushButton

    try:
        buttons = viewer.window._qt_viewer.layerButtons
    except Exception:  # noqa: BLE001
        return None
    button = QPushButton()
    button.setObjectName("nvitkFolderButton")
    button.setToolTip(
        "Layer folders: new folder from the selected layers, move layers in or out, "
        "collapse / expand all. Right-click a folder header for its own menu."
    )
    try:
        from napari._qt.qt_resources import QColoredSVGIcon

        bg = buttons.palette().color(buttons.palette().ColorRole.Window).red()
        button.setIcon(QColoredSVGIcon.from_resources("folder").colored(theme="dark" if bg < 128 else "light"))
    except Exception:  # noqa: BLE001 — a text button works too
        button.setText("📁")
    button.setFixedSize(28, 28)
    button.setStyleSheet("QPushButton { padding: 0px; margin: 0px; min-width: 28px; min-height: 28px; }")
    button.clicked.connect(lambda: folders_menu(folders, button).exec_(button.mapToGlobal(QPoint(0, button.height()))))
    layout = buttons.layout()
    # After the new-points / shapes / labels buttons, before the stretch.
    layout.insertWidget(3, button)
    return button


def install_layer_folders(viewer: Any, delegate: Any = None) -> LayerFolders | None:
    """Turn on folders in *viewer*'s layer list; ``None`` when the list is unreachable."""
    existing = getattr(viewer, "_nvitk_layer_folders", None)
    if existing is not None:
        return existing
    try:
        view = viewer.window._qt_viewer.layers
    except Exception:  # noqa: BLE001 — headless
        return None
    folders = LayerFolders(viewer)
    if delegate is None:
        delegate = view.itemDelegate()
    if hasattr(delegate, "set_folders"):
        delegate.set_folders(folders)
    folders.attach_view(view, delegate)
    mouse = _HeaderMouse(view, folders)
    view.viewport().installEventFilter(mouse)
    folders._mouse_filter = mouse
    folders._button = _toolbar_button(viewer, folders)
    viewer._nvitk_layer_folders = folders
    return folders


__all__ = [
    "FOLDER_KEY",
    "LayerFolders",
    "RowEntry",
    "compute_layout",
    "folder_of",
    "install_layer_folders",
]
