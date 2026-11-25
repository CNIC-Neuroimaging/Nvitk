"""Tool browser: every tool in task sections, filtered by modality / domain / species / suite.

The tree replaces picking a Category and then an Operation from two drop-downs: sections
(Segmentation, Masks & labels, Registration...) fold open into subsections and tools; each tool
shows the suite that does the work as a coloured tag (FSL, FastSurfer, ANTsPyNet...), and its
description on hover. Chips narrow the tree to an imaging modality (MRA/TOF, 4D flow, CT/CTA,
PET...), a domain, a species or a suite; the search box matches names, descriptions and suites.
Choosing a tool emits :attr:`ToolBrowser.tool_selected`; the tools dock then shows that tool's
form.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.icons import icon as make_icon
from nvitk.gui.core.widgets import ChipFilterBar, Section
from nvitk.gui.tools.taxonomy import DOMAINS, MODALITIES, SECTIONS, SPECIES, matches, meta_for

_TOOL_ROLE = Qt.ItemDataRole.UserRole + 1
_PREFS_KEY = "gui.tool_browser"


def _is_group(value: str) -> bool:
    """A section / subsection / panel row rather than a tool (tool ids may hold ``:``, ``ml:...``)."""
    return value.startswith(("section:", "sub:", "panel:"))


class ToolBrowser(QWidget):
    """Section tree of the registry's tools with chip filters and search."""

    tool_selected = Signal(str)
    #: A panel picked from the search (its dock object name).
    panel_selected = Signal(str)

    def __init__(
        self,
        tools: Iterable[Any] | None = None,
        *,
        sections: Iterable[Any] | None = None,
        meta_fn: Callable[[Any], Any] = meta_for,
        help_fn: Callable[[Any], str] | None = None,
        tag_filters: bool = True,
        prefs_key: str = _PREFS_KEY,
        panels: Callable[[], list[tuple[str, str]]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """*tools* defaults to the registry's; *sections* / *meta_fn* / *help_fn* let another
        catalogue (the mesh operations) use the same browser. *tag_filters* off hides the
        modality / domain / species chips (keeping the suite filter)."""
        super().__init__(parent)
        self._tools = list(tools) if tools is not None else None
        self._sections = list(sections) if sections is not None else list(SECTIONS)
        self._meta = meta_fn
        self._help_fn = help_fn
        self._prefs_key = prefs_key
        #: ``() -> [(dock object name, title)]``: panels the search also finds (opened on click).
        self._panels = panels
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        self._search = QLineEdit()
        self._search.setPlaceholderText(
            "Search tools and panels (name, description, suite)…" if panels is not None
            else "Search tools (name, description, suite)…")
        self._search.setClearButtonEnabled(True)
        self._search.addAction(make_icon("search", design.COLOR_MUTED), QLineEdit.LeadingPosition)
        self._search.textChanged.connect(lambda _t: self.refresh())
        root.addWidget(self._search)

        self._modalities = ChipFilterBar([(k, label, hue) for k, (label, hue) in MODALITIES.items()])
        self._modalities.changed.connect(lambda _s: self.refresh())
        root.addWidget(self._modalities)
        self._modalities.setVisible(tag_filters)

        more = Section("More filters", icon_name="filter", expanded=False, pref_key="tool_browser.filters")
        self._domains = ChipFilterBar([(k, label, hue) for k, (label, hue) in DOMAINS.items()])
        self._species = ChipFilterBar([(k, label, hue) for k, (label, hue) in SPECIES.items()])
        self._suites = ChipFilterBar([])
        for bar in (self._domains, self._species, self._suites):
            bar.changed.connect(lambda _s: self.refresh())
            more.add(bar)
        self._domains.setVisible(tag_filters)
        self._species.setVisible(tag_filters)
        root.addWidget(more)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(12)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tree.header().setStretchLastSection(False)
        self._tree.itemClicked.connect(self._on_item)
        self._tree.itemActivated.connect(self._on_item)
        self._tree.itemExpanded.connect(lambda _i: self._remember_expanded())
        self._tree.itemCollapsed.connect(lambda _i: self._remember_expanded())
        self._tree.setMinimumHeight(80)
        root.addWidget(self._tree, 1)

        self._expanded: set[str] = set(self._stored_expanded())
        self._fill_suites()
        self.refresh()
        design.add_theme_listener(self.refresh)

    # data
    def _all_tools(self) -> list[Any]:
        if self._tools is not None:
            return self._tools
        from nvitk.gui.tools.registry import _GATED_TOOL_IDS, all_tools, visible_tool_ids

        # Gated tools (FreeSurfer programs...) only where they can run; everything else always.
        shown = visible_tool_ids()
        return [t for t in all_tools() if t.id not in _GATED_TOOL_IDS or t.id in shown]

    def _fill_suites(self) -> None:
        suites = sorted({self._meta(t).suite for t in self._all_tools()}, key=lambda s: (s != "nvitk", s.lower()))
        self._suites.set_chips([(s, s, s if s in design.SUITE_HUES else "grey") for s in suites])

    def _help(self, tool: Any) -> str:
        if self._help_fn is not None:
            return str(self._help_fn(tool) or "")
        try:
            from nvitk.gui.tools.registry import operation_help_text

            return operation_help_text(tool.id) or str(getattr(tool, "description", "") or "")
        except Exception:  # noqa: BLE001
            return str(getattr(tool, "description", "") or "")

    # view
    def refresh(self) -> None:
        """Rebuild the tree for the current filters and search."""
        text = self._search.text().strip().lower()
        filters = dict(
            modalities=self._modalities.selected(),
            domains=self._domains.selected(),
            species=self._species.selected(),
            suites=self._suites.selected(),
        )
        filtering = bool(text or any(filters.values()))
        selected = self.current_tool()
        self._tree.setUpdatesEnabled(False)
        self._tree.clear()
        grouped: dict[str, dict[str, list[tuple[Any, Any]]]] = {}
        for tool in self._all_tools():
            meta = self._meta(tool)
            if not matches(meta, **filters):
                continue
            if text:
                haystack = " ".join(
                    [str(tool.label), str(tool.id), meta.suite, meta.subsection, self._help(tool)]
                ).lower()
                if not all(word in haystack for word in text.split()):
                    continue
            grouped.setdefault(meta.section, {}).setdefault(meta.subsection or "", []).append((tool, meta))
        bold = QFont(self.font())
        bold.setBold(True)
        if text and self._panels is not None:
            try:
                panels = [(n, t) for n, t in self._panels() if all(w in t.lower() or w in n.lower() for w in text.split())]
            except Exception:  # noqa: BLE001
                panels = []
            if panels:
                top = QTreeWidgetItem(["Panels", str(len(panels))])
                top.setIcon(0, make_icon("panel_left", design.accent("blue")))
                top.setFont(0, bold)
                top.setForeground(1, QBrush(QColor(design.COLOR_MUTED)))
                top.setToolTip(0, "nvitk panels (tabs): click one to open it")
                top.setData(0, _TOOL_ROLE, "section:__panels__")
                self._tree.addTopLevelItem(top)
                for name, title in panels:
                    leaf = QTreeWidgetItem([title, "panel"])
                    leaf.setData(0, _TOOL_ROLE, f"panel:{name}")
                    leaf.setToolTip(0, f"Open the {title} panel")
                    leaf.setForeground(1, QBrush(QColor(design.COLOR_MUTED)))
                    top.addChild(leaf)
                top.setExpanded(True)
        for section in self._sections:
            subs = grouped.get(section.key)
            if not subs:
                continue
            count = sum(len(v) for v in subs.values())
            top = QTreeWidgetItem([section.title, str(count)])
            top.setIcon(0, make_icon(section.icon, design.accent(section.hue)))
            top.setFont(0, bold)
            top.setForeground(1, QBrush(QColor(design.COLOR_MUTED)))
            top.setToolTip(0, section.description)
            top.setData(0, _TOOL_ROLE, f"section:{section.key}")
            self._tree.addTopLevelItem(top)
            for sub_name in sorted(subs, key=lambda s: (s == "", s.lower())):
                parent = top
                if sub_name and len(subs) > 1:
                    parent = QTreeWidgetItem([sub_name, ""])
                    parent.setForeground(0, QBrush(QColor(design.COLOR_MUTED)))
                    parent.setData(0, _TOOL_ROLE, f"sub:{section.key}/{sub_name}")
                    top.addChild(parent)
                    parent.setExpanded(True)
                for tool, meta in sorted(subs[sub_name], key=lambda tm: str(tm[0].label).lower()):
                    leaf = QTreeWidgetItem([str(tool.label), "" if meta.suite == "nvitk" else meta.suite])
                    leaf.setData(0, _TOOL_ROLE, tool.id)
                    tags = ", ".join(MODALITIES[m][0] for m in meta.modalities if m in MODALITIES)
                    tip = self._help(tool)
                    leaf.setToolTip(0, (f"[{tags}]  " if tags else "") + tip)
                    if meta.suite != "nvitk":
                        leaf.setForeground(1, QBrush(QColor(design.suite_color(meta.suite))))
                        leaf.setToolTip(1, f"Runs {meta.suite}")
                    parent.addChild(leaf)
                    if tool.id == selected:
                        self._tree.setCurrentItem(leaf)
            top.setExpanded(filtering or section.key in self._expanded)
        self._tree.setUpdatesEnabled(True)

    def current_tool(self) -> str | None:
        """Id of the highlighted tool, if any."""
        item = self._tree.currentItem()
        value = item.data(0, _TOOL_ROLE) if item is not None else None
        return value if isinstance(value, str) and not _is_group(value) else None

    def select_tool(self, tool_id: str, *, emit: bool = False) -> None:
        """Highlight *tool_id* (opening its section)."""
        iterator = [self._tree.topLevelItem(i) for i in range(self._tree.topLevelItemCount())]
        while iterator:
            item = iterator.pop()
            if item.data(0, _TOOL_ROLE) == tool_id:
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                if emit:
                    self.tool_selected.emit(tool_id)
                return
            iterator.extend(item.child(i) for i in range(item.childCount()))

    def _on_item(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        value = item.data(0, _TOOL_ROLE)
        if isinstance(value, str) and value.startswith("panel:"):
            self.panel_selected.emit(value.split(":", 1)[1])
        elif isinstance(value, str) and not _is_group(value):
            self.tool_selected.emit(value)
        elif item.childCount():
            item.setExpanded(not item.isExpanded())

    # remembered sections
    def _stored_expanded(self) -> list[str]:
        try:
            from nvitk.gui.core.prefs import load_prefs

            return list(load_prefs().get(self._prefs_key, {}).get("expanded", []))
        except Exception:  # noqa: BLE001
            return []

    def _remember_expanded(self) -> None:
        if self._search.text().strip() or self._modalities.selected():
            return  # filtering expands everything; that is not a choice to remember
        expanded = set()
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            key = str(item.data(0, _TOOL_ROLE) or "").split(":", 1)[-1]
            if item.isExpanded():
                expanded.add(key)
        if expanded != self._expanded:
            self._expanded = expanded
            try:
                from nvitk.gui.core.prefs import save_prefs

                save_prefs({self._prefs_key: {"expanded": sorted(expanded)}})
            except Exception:  # noqa: BLE001
                pass


__all__ = ["ToolBrowser"]
