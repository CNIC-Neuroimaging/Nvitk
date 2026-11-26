"""The Export & record module's "Save layer to file" options, for one layer, in a pop-over.

Opened from the layer list's context menu (Export…): output path (with a Browse… dialog), the
format override and, for images, whether to write the file's original affine. Export writes
*that* layer, whichever is active, and journals the step like the Save layer panel does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.widgets import Popover, hint_label

#: Save-dialog filters and default suffix per layer kind.
_FILTERS = {
    "Surface": ("STL (*.stl);;PLY (*.ply);;OBJ (*.obj);;VTK PolyData (*.vtp);;OFF (*.off);;GIfTI (*.gii);;All (*)",
                ".stl"),
    "Points": ("XYZ (*.xyz);;CSV (*.csv);;PLY (*.ply);;VTK PolyData (*.vtp);;All (*)", ".xyz"),
    "image": ("NIfTI (*.nii *.nii.gz);;TIFF (*.tif *.tiff);;MetaImage (*.mha);;All (*)", ".nii.gz"),
}
#: Format override choices for images ("" = from the file extension).
_IMAGE_FORMATS = (("From the extension", ""), ("NIfTI", "nifti"), ("MetaImage", "mha"), ("TIFF", "tiff"))


def _kind(layer: Any) -> str:
    kind = type(layer).__name__
    return kind if kind in ("Surface", "Points") else "image"


def default_export_path(layer: Any) -> Path:
    """``<folder>/<layer name><suffix>``: beside the layer's source file when it has one."""
    from nvitk.gui.tools.external_io import layer_source_path

    suffix = _FILTERS[_kind(layer)][1]
    safe = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in str(layer.name)).strip() or "layer"
    src = layer_source_path(layer)
    folder = src.parent if src is not None and src.parent.is_dir() else Path.cwd()
    return folder / f"{safe}{suffix}"


class ExportOptions(QWidget):
    """Path, format and affine options for exporting *layer*, and the Export button."""

    def __init__(self, viewer: Any, layer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._layer = layer
        self._kind = _kind(layer)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(design.SPACE_TIGHT)
        path_row = QHBoxLayout()
        path_row.setSpacing(design.SPACE_TIGHT)
        self.path = QLineEdit(str(default_export_path(layer)))
        self.path.setMinimumWidth(320)
        self.path.returnPressed.connect(self.export)
        self.browse = QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        path_row.addWidget(self.path, 1)
        path_row.addWidget(self.browse)
        form.addRow("Output", path_row)
        self.format = QComboBox()
        self.use_file_affine = QCheckBox("Use the original file affine (from nvitk metadata)")
        self.use_file_affine.setChecked(True)
        if self._kind == "image":
            for label, value in _IMAGE_FORMATS:
                self.format.addItem(label, value)
            form.addRow("Format", self.format)
            form.addRow(self.use_file_affine)
        else:
            self.format.setVisible(False)
            self.use_file_affine.setVisible(False)
            form.addRow(hint_label("The format follows the extension; meshes are written in world mm."))
        root.addLayout(form)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.button = QPushButton("Export")
        self.button.setDefault(True)
        self.button.clicked.connect(self.export)
        buttons.addWidget(self.button)
        root.addLayout(buttons)
        self.status = hint_label("")
        root.addWidget(self.status)

    def _popover(self) -> Popover | None:
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, Popover):
            widget = widget.parentWidget()
        return widget

    def _browse(self) -> None:
        # A file dialog takes the focus, which closes an unpinned pop-over (and deletes it, with
        # this form, while the dialog is still open): pin it first.
        popover = self._popover()
        if popover is not None:
            popover.set_pinned(True)
        filters, suffix = _FILTERS[self._kind]
        out, _ = QFileDialog.getSaveFileName(self.window(), f"Export {self._layer.name}", self.path.text(), filters)
        if out:
            if not Path(out).suffix:
                out += suffix
            self.path.setText(out)

    def export(self) -> Path | None:
        """Write the layer to the path in the form; ``None`` when it failed (the reason is shown)."""
        from nvitk.gui.core.journal import record
        from nvitk.gui.io.export import export_layer
        from nvitk.gui.tools.runner import notify
        from nvitk.io._common import guess_write_type

        text = self.path.text().strip()
        if not text:
            self.status.setText("Set an output path.")
            return None
        out = Path(text).expanduser()
        force = (self.format.currentData() or None) if self._kind == "image" else None
        try:
            if self._kind == "image" and force is None:
                if out.suffix.lower() == ".stl":
                    force = "stl"
                else:
                    guess_write_type(out)  # an unknown extension fails here, with the reason
            out.parent.mkdir(parents=True, exist_ok=True)
            export_layer(self._layer, out, use_file_affine=self.use_file_affine.isChecked(), force_type=force)
        except Exception as exc:  # noqa: BLE001 (shown in the form, and as a notification)
            self.status.setText(f"Export failed: {exc}")
            notify(f"Export failed: {exc}", error=True)
            return None
        record(self._viewer, "export", str(out), params={"use_file_affine": self.use_file_affine.isChecked(),
                                                         "force_type": force},
               inputs=[str(self._layer.name)], outputs=[str(out)])
        self.status.setText(f"Saved → {out}")
        notify(f"Exported {self._layer.name} → {out}")
        return out


def show_export_popover(viewer: Any, layer: Any, global_pos: QPoint | None = None) -> Popover:
    """The export options for *layer* in a pop-over at *global_pos* (the cursor by default)."""
    from qtpy.QtGui import QCursor

    content = ExportOptions(viewer, layer)
    design.apply_theme(content)
    popover = Popover(content, title=f"Export · {layer.name}", parent=viewer.window._qt_window)
    popover.setAttribute(Qt.WA_DeleteOnClose, True)
    popover.show_at_point(global_pos if global_pos is not None else QCursor.pos())
    content.path.setFocus()
    return popover


__all__ = ["ExportOptions", "default_export_path", "show_export_popover"]
