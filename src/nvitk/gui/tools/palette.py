"""Alt+Space command palette: type a few letters, run any nvitk tool.

The tool registry has grown past a hundred entries across a dozen categories, and
finding one means remembering which category it lives under. This indexes every
tool — plus the quick image operations and the dock actions that are not tools at
all — behind one fuzzy search, the way an editor's command palette works.

Matching is subsequence-based, so ``total`` finds *TotalSegmentator*, ``orthog``
finds *Orthogonal views*, and ``gauss`` finds *Gaussian blur*, without anyone
having to type the category first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_TEXT,
    SPACE_TIGHT,
    apply_theme,
)

#: How many results the list shows before it stops adding more.
MAX_RESULTS = 40


@dataclass
class Command:
    """One entry in the palette."""

    key: str
    title: str
    #: Where it comes from — the tool category, or a group like "Image".
    group: str = ""
    #: Extra words that should match it (``"threshold"`` finding *Binarize*).
    keywords: tuple[str, ...] = ()
    #: Called with no arguments when the entry is chosen.
    run: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def haystack(self) -> str:
        """Everything this command can be matched against, lowercased."""
        return " ".join([self.title, self.group, self.key, *self.keywords]).lower()


def subsequence_score(query: str, text: str) -> int | None:
    """How well *query* matches *text* as a subsequence; ``None`` if it does not.

    Lower is better. A run of consecutive characters scores better than the same
    letters scattered, and a match at the start of a word better than mid-word —
    so ``total`` puts *TotalSegmentator* above *Subtotal volume*.
    """
    q, t = query.strip().lower(), text.lower()
    if not q:
        return 0
    if q in t:
        # A literal substring always beats a scattered subsequence.
        return t.index(q)
    score = 0
    position = 0
    previous = -2
    for ch in q:
        found = t.find(ch, position)
        if found < 0:
            return None
        # Penalise gaps, and reward landing at the start of a word.
        gap = found - previous - 1
        if gap and not (found and t[found - 1] == " "):
            score += gap
        previous = found
        position = found + 1
    return 1000 + score


def rank_commands(commands: list[Command], query: str) -> list[Command]:
    """*commands* that match *query*, best first."""
    if not query.strip():
        return list(commands)[:MAX_RESULTS]
    scored: list[tuple[int, int, Command]] = []
    for index, command in enumerate(commands):
        score = subsequence_score(query, command.haystack)
        if score is not None:
            # Index breaks ties so equal matches keep the registry's order.
            scored.append((score, index, command))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [command for _s, _i, command in scored[:MAX_RESULTS]]


class CommandPalette(QDialog):
    """Frameless search-and-run overlay over the viewer."""

    def __init__(self, commands: list[Command], parent: QWidget | None = None) -> None:
        """Build the search field and result list for *commands*."""
        super().__init__(parent)
        self._commands = list(commands)
        self._chosen: Command | None = None

        self.setWindowTitle("nvitk commands")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Run a tool…  (try “total”, “orthog”, “gauss”)")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._refilter)
        self._search.returnPressed.connect(self._accept_current)

        self._list = QListWidget()
        self._list.itemActivated.connect(lambda _i: self._accept_current())
        self._list.itemClicked.connect(lambda _i: self._accept_current())

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT)
        root.setSpacing(SPACE_TIGHT)
        root.addWidget(self._search)
        root.addWidget(self._list, stretch=1)

        apply_theme(self)
        self.setStyleSheet(
            self.styleSheet()
            + f"QDialog {{ border: 1px solid {COLOR_ACCENT}; border-radius: 6px; }}"
        )
        self.resize(560, 420)
        self._refilter("")

    def _refilter(self, query: str) -> None:
        """Rebuild the result list for *query*."""
        self._list.clear()
        for command in rank_commands(self._commands, query):
            item = QListWidgetItem(command.title)
            item.setData(Qt.UserRole, command.key)
            if command.group:
                item.setToolTip(f"{command.group} · {command.title}")
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _accept_current(self) -> None:
        """Remember the highlighted command and close."""
        item = self._list.currentItem()
        if item is None:
            return
        key = str(item.data(Qt.UserRole))
        self._chosen = next((c for c in self._commands if c.key == key), None)
        self.accept()

    def keyPressEvent(self, event: Any) -> None:
        """Let Up/Down drive the list while the cursor stays in the search field."""
        if event.key() in (Qt.Key_Down, Qt.Key_Up):
            row = self._list.currentRow() + (1 if event.key() == Qt.Key_Down else -1)
            if 0 <= row < self._list.count():
                self._list.setCurrentRow(row)
            return
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def chosen(self) -> Command | None:
        """The command the user picked, or ``None`` if they dismissed the palette."""
        return self._chosen


#: Quick image operations, as ``(key, title, keywords, callable-name, kwargs)``.
#: Registry tools already carry their own metadata; these do not, so they are
#: described here rather than being faked into the tool registry.
_QUICK_OPS: tuple[tuple[str, str, tuple[str, ...], str, dict[str, Any]], ...] = (
    # An empty kwargs dict means "ask": the entry opens its options popup. Entries
    # that carry kwargs are one-click presets and run straight away.
    ("qk_threshold_display", "Threshold…  (live preview)",
     ("threshold", "binarize", "manual", "mask", "level"), "threshold_at_display", {}),
    ("qk_threshold_otsu", "Threshold (Otsu, automatic)",
     ("threshold", "binarize", "auto", "otsu"), "threshold_otsu", {}),
    ("qk_auto_contrast", "Brightness / contrast: auto window…",
     ("brightness", "contrast", "window", "levels"), "auto_contrast", {}),
    ("qk_reset_contrast", "Brightness / contrast: reset to full range",
     ("brightness", "contrast", "reset"), "reset_contrast", {}),
    ("qk_gaussian", "Gaussian blur…",
     ("filter", "smooth", "blur"), "gaussian", {}),
    ("qk_gaussian1", "Gaussian blur (sigma 1)",
     ("filter", "smooth", "blur"), "gaussian", {"sigma": 1.0}),
    ("qk_median", "Median filter…",
     ("filter", "denoise", "salt"), "median", {}),
    ("qk_invert", "Invert intensities", ("negative", "complement"), "invert", {}),
    ("qk_project", "Projection…",
     ("projection", "mip", "flatten", "maximum", "mean"), "project", {}),
    ("qk_proj_max", "Maximum intensity projection",
     ("projection", "mip", "flatten"), "project", {"how": "max", "axis": 0}),
    ("qk_rotate", "Rotate…",
     ("rotate", "transpose", "orientation", "90"), "rotate90", {}),
    ("qk_crop", "Crop to content…",
     ("crop", "bounding box", "trim"), "crop_to_content", {}),
    ("qk_convert", "Convert image type…",
     ("type", "dtype", "cast", "8-bit", "16-bit", "32-bit",
      "uint8", "uint16", "int16", "int32", "float32", "float64"),
     "convert_dtype", {}),
)


def build_commands(viewer: Any, run_tool: Callable[[str], None]) -> list[Command]:
    """Every tool, quick operation and panel action, as one searchable list.

    *run_tool* is called with a registry tool id; the palette does not know how to
    drive the tool form itself, so selecting a tool hands it back to the dock.
    """
    from nvitk.gui.tools import quick_ops
    from nvitk.gui.tools.registry import all_tools

    commands: list[Command] = []

    for spec in all_tools():
        commands.append(
            Command(
                key=spec.id,
                title=f"{spec.category}: {spec.label}",
                group=spec.category,
                keywords=tuple(spec.id.split("_")),
                run=(lambda tid=spec.id: run_tool(tid)),
            )
        )

    def _quick(name: str, kwargs: dict[str, Any], title: str) -> Callable[[], None]:
        """Bind one quick operation to the viewer, reporting whatever it returns."""

        def _run() -> None:
            from nvitk.gui.tools.quick_dialog import run_quick_op
            from nvitk.gui.tools.runner import log_tool_failure, notify

            try:
                parent = viewer.window._qt_window
            except Exception:
                parent = None
            try:
                message = run_quick_op(viewer, name, kwargs, title=title, parent=parent)
            except Exception as exc:  # noqa: BLE001
                log_tool_failure(exc)
                notify(str(exc), error=True)
                return
            # None means the options popup was cancelled; say nothing.
            if message:
                notify(str(message))

        return _run

    for key, title, keywords, fn_name, kwargs in _QUICK_OPS:
        commands.append(
            Command(
                key=key,
                title=f"Image: {title}",
                group="Image",
                keywords=keywords,
                run=_quick(fn_name, kwargs, title),
            )
        )

    return commands


#: Qt's name for the Windows / Super key is ``Meta``. "Win+Space" and
#: "Super+Space" both parse to an *empty* key sequence, which binds nothing and
#: fails silently — hence the spelling here and the check in the installer.
PALETTE_SHORTCUT = "Meta+Space"

#: What to call that key in the interface, where "Meta" means nothing to anyone.
PALETTE_SHORTCUT_LABEL = "Win+Space"


def install_command_palette(
    viewer: Any,
    run_tool: Callable[[str], None],
    *,
    shortcut: str = PALETTE_SHORTCUT,
) -> Callable[[], None]:
    """Bind *shortcut* on the Napari window to open the palette; returns the opener.

    The opener is returned whether or not the binding took, so a button can still
    reach the palette on a desktop whose window manager claims the key first.
    """
    from qtpy.QtGui import QKeySequence, QShortcut

    def _open() -> None:
        """Show the palette and run whatever the user picks."""
        try:
            parent = viewer.window._qt_window
        except Exception:
            parent = None
        dialog = CommandPalette(build_commands(viewer, run_tool), parent=parent)
        if parent is not None:
            # Centred on the window, near the top, the way a palette should sit.
            geo = parent.geometry()
            dialog.move(
                geo.x() + (geo.width() - dialog.width()) // 2,
                geo.y() + max(int(geo.height() * 0.12), 0),
            )
        if dialog.exec() != dialog.Accepted:
            return
        command = dialog.chosen()
        if command is not None and command.run is not None:
            command.run()

    sequence = QKeySequence(shortcut)
    if sequence.isEmpty():
        from nvitk.gui.core.log_panel import gui_log

        gui_log(
            f"Command palette: {shortcut!r} is not a key sequence Qt understands "
            "(the Windows key is spelled 'Meta'). Use the search button instead.",
            error=True,
        )
        return _open

    try:
        parent = viewer.window._qt_window
        hotkey = QShortcut(sequence, parent)
        hotkey.setContext(Qt.ApplicationShortcut)
        hotkey.activated.connect(_open)
        # Keep it alive with the window rather than letting it be collected.
        parent._nvitk_palette_shortcut = hotkey
    except Exception:
        pass
    return _open


__all__ = [
    "MAX_RESULTS",
    "PALETTE_SHORTCUT",
    "PALETTE_SHORTCUT_LABEL",
    "build_commands",
    "install_command_palette",
    "Command",
    "CommandPalette",
    "rank_commands",
    "subsequence_score",
]
