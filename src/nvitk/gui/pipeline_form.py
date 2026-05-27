"""Dynamic Qt form for pipeline Click CLI options."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import click
from click.core import UNSET as CLICK_UNSET
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

PIPELINE_FORM_SCROLL_MAX = 300


def _cli_long_option(param: click.Parameter) -> str:
    """Primary ``--long-option`` for argv (uses Click opts, not Python ``param.name``)."""
    if isinstance(param, click.Option):
        long_opts = [
            o for o in param.opts if o.startswith("--") and not o.startswith("---")
        ]
        if long_opts:
            return max(long_opts, key=len)
    return f"--{param.name.replace('_', '-')}"


def _load_click_command(script_name: str) -> click.Command | None:
    for ep in entry_points(group="console_scripts"):
        if ep.name != script_name:
            continue
        mod_name, attr = ep.value.split(":", 1)
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        if isinstance(obj, click.Command):
            return obj
        if isinstance(obj, click.Group):
            return obj
    return None


class _Field:
    def __init__(self, param: click.Parameter, widget: QWidget, browse: QPushButton | None = None) -> None:
        self.param = param
        self.widget = widget
        self.browse = browse

    def value(self) -> Any:
        p = self.param
        w = self.widget
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        text = w.text().strip() if hasattr(w, "text") else ""
        if not text and p.required:
            raise ValueError(f"Required option --{p.name} is empty.")
        if not text:
            return None
        if isinstance(p.type, click.Path):
            return text
        py_type = getattr(p.type, "name", None) or str(p.type)
        if py_type in ("int", "integer"):
            return int(text)
        if py_type in ("float",):
            return float(text)
        if py_type in ("bool", "boolean"):
            return text.lower() in ("1", "true", "yes", "on")
        return text


class PipelineCliForm(QGroupBox):
    """Build inputs from a pipeline's Click command definition."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Pipeline CLI arguments", parent)
        self._script = ""
        self._fields: list[_Field] = []
        self._hint = QLabel("Select a pipeline operation to edit CLI options.")
        self._hint.setWordWrap(True)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(160)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._form_host = QWidget()
        self._form = QFormLayout()
        self._form_host.setLayout(self._form)
        self._scroll.setWidget(self._form_host)
        root = QVBoxLayout()
        root.addWidget(self._hint)
        root.addWidget(self._scroll, stretch=1)
        self.setLayout(root)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def set_expanded(self, expanded: bool) -> None:
        """Show pipeline options in a bounded scroll area."""
        cap = PIPELINE_FORM_SCROLL_MAX if expanded else 200
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setSizePolicy(policy)
        self._scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._scroll.setMinimumHeight(120 if expanded else 0)
        self._scroll.setMaximumHeight(cap)

    def set_script(self, script_name: str) -> None:
        if script_name == self._script:
            return
        self._script = script_name
        self._clear_form()
        if not script_name:
            self._hint.setText("Select a pipeline operation.")
            return
        cmd = _load_click_command(script_name)
        if cmd is None:
            self._hint.setText(f"Could not load Click command for {script_name!r}.")
            return
        self._hint.setText(f"Options for {script_name} (required *).")
        for param in cmd.params:
            if getattr(param, "hidden", False):
                continue
            flag = _cli_long_option(param)
            label = f"{flag.lstrip('-')}{' *' if param.required else ''}"
            if isinstance(param, click.Option) and param.is_flag:
                w = QCheckBox(param.help or name)
                default = param.default
                w.setChecked(bool(default) if default is not CLICK_UNSET else False)
                self._form.addRow(label, w)
                self._fields.append(_Field(param, w))
                continue
            edit = QLineEdit()
            if param.default is not None and param.default is not CLICK_UNSET:
                edit.setPlaceholderText(str(param.default))
            if param.help:
                edit.setToolTip(param.help)
            row = QWidget()
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(edit)
            browse = None
            if isinstance(param.type, click.Path):
                browse = QPushButton("…")
                browse.setFixedWidth(28)

                def _pick(le=edit, prm=param) -> None:
                    if getattr(prm.type, "dir_okay", True) and not getattr(
                        prm.type, "exists", False
                    ):
                        path = QFileDialog.getExistingDirectory(None, "Select directory")
                    else:
                        path, _ = QFileDialog.getOpenFileName(None, "Select path")
                    if path:
                        le.setText(path)

                browse.clicked.connect(_pick)
                h.addWidget(browse)
            row.setLayout(h)
            self._form.addRow(label, row)
            self._fields.append(_Field(param, edit, browse))

    def _clear_form(self) -> None:
        self._fields.clear()
        while self._form.rowCount():
            self._form.removeRow(0)

    def build_argv(self, exe: str) -> list[str]:
        argv = [exe]
        for field in self._fields:
            val = field.value()
            if val is None:
                continue
            flag = _cli_long_option(field.param)
            if isinstance(field.widget, QCheckBox):
                if val:
                    argv.append(flag)
                continue
            argv.extend([flag, str(val)])
        return argv
