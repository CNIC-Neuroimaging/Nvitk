"""File dialogs Qt does not offer ready-made."""

from __future__ import annotations

from typing import Any


def choose_directories(parent: Any = None, caption: str = "Choose folders", start: str = "") -> list[str]:
    """Several folders at once (Ctrl / Shift-click in the list), or ``[]`` when cancelled.

    Qt's own folder picker takes one; this is Qt's non-native file dialog in folder
    mode with multi-selection switched on in its views.
    """
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QFileDialog,
        QFileSystemModel,
        QListView,
        QTreeView,
    )

    dlg = QFileDialog(parent, caption, start or "")
    dlg.setFileMode(QFileDialog.FileMode.Directory)
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
    for view in dlg.findChildren(QListView) + dlg.findChildren(QTreeView):
        if isinstance(view.model(), (QFileSystemModel,)) or view.objectName() in ("listView", "treeView"):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    dlg.setLabelText(QFileDialog.DialogLabel.Accept, "Choose")
    dlg.setToolTip("Ctrl- or Shift-click to choose several folders.")
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return []
    chosen = [p for p in dlg.selectedFiles() if p]
    # Qt also lists the folder being browsed when nothing inside it was picked.
    return list(dict.fromkeys(chosen))


__all__ = ["choose_directories"]
