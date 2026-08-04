"""Stage-based pipeline form for the Napari GUI."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import click
from click.core import UNSET as CLICK_UNSET
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from nvitk.gui.pipeline.stages import (
    PipelineStageDef,
    PipelineStageSpec,
    StageInputBinding,
    binding_requires_layer,
    input_label_for_key,
    pipeline_def_for_script,
    resolve_stage_input_bindings,
    visible_stage_inputs,
)

PIPELINE_FORM_SCROLL_MIN = 120


def _cli_long_option(param: click.Parameter) -> str:
    """The longest ``--flag`` name registered for a Click *param* (or a derived ``--param-name``)."""
    if isinstance(param, click.Option):
        long_opts = [
            o for o in param.opts if o.startswith("--") and not o.startswith("---")
        ]
        if long_opts:
            return max(long_opts, key=len)
    return f"--{param.name.replace('_', '-')}"


def _load_click_command(script_name: str) -> click.Command | None:
    """Resolve an installed console-script's underlying Click command/group by its entry-point name."""
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


class _ParamField:
    """Pairs a Click CLI parameter with the Qt widget collecting its value in the pipeline form."""

    def __init__(self, param: click.Parameter, widget: QWidget) -> None:
        """Bind *param* to its input *widget*."""
        self.param = param
        self.widget = widget

    def value(self) -> Any:
        """Read and type-coerce the widget's current value per the Click parameter's type; raises
        ``ValueError`` if a required text field is empty."""
        p = self.param
        w = self.widget
        if isinstance(w, QCheckBox):
            return bool(w.isChecked())
        text = w.text().strip() if hasattr(w, "text") else ""
        if not text and p.required:
            raise ValueError(f"Required option {_cli_long_option(p)} is empty.")
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


class _StageRow:
    """One pipeline stage's row in the form: its enable checkbox, input-widget container, and the
    per-slot input widgets themselves."""

    def __init__(
        self,
        stage: PipelineStageSpec,
        checkbox: QCheckBox,
        inputs_host: QWidget,
        inputs_form: QFormLayout,
        input_widgets: dict[str, QWidget],
    ) -> None:
        """Bind *stage* to its checkbox, inputs container/layout, and (initially empty) widget map."""
        self.stage = stage
        self.checkbox = checkbox
        self.inputs_host = inputs_host
        self.inputs_form = inputs_form
        self.input_widgets = input_widgets


class PipelineStageForm(QGroupBox):
    """Pipeline stages, layer inputs, and global CLI options."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the scrollable stage-checkbox and CLI-parameter form containers."""
        super().__init__("Pipeline", parent)
        self._viewer: Any | None = None
        self._script = ""
        self._pipeline_def: PipelineStageDef | None = None
        self._stage_rows: list[_StageRow] = []
        self._param_fields: list[_ParamField] = []
        self._hint = QLabel("Select a pipeline operation.")
        self._hint.setWordWrap(True)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(160)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._content = QWidget()
        self._content_layout = QVBoxLayout()
        self._content_layout.setAlignment(Qt.AlignTop)
        self._stages_host = QWidget()
        self._stages_layout = QVBoxLayout()
        self._stages_layout.setAlignment(Qt.AlignTop)
        self._stages_host.setLayout(self._stages_layout)
        self._params_host = QWidget()
        self._params_form = QFormLayout()
        self._params_host.setLayout(self._params_form)
        self._content_layout.addWidget(self._stages_host)
        self._content_layout.addWidget(self._params_host)
        self._content.setLayout(self._content_layout)
        self._scroll.setWidget(self._content)
        root = QVBoxLayout()
        root.addWidget(self._hint)
        root.addWidget(self._scroll, stretch=1)
        self.setLayout(root)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def set_viewer(self, viewer: Any | None) -> None:
        """Bind the Napari viewer used to populate layer-selection combo boxes."""
        self._viewer = viewer
        self.refresh_layer_combos()

    def set_expanded(self, expanded: bool) -> None:
        """Fill remaining dock height when visible; collapse when hidden."""
        if expanded:
            expanding = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.setSizePolicy(expanding)
            self._scroll.setSizePolicy(expanding)
            self._scroll.setMinimumHeight(PIPELINE_FORM_SCROLL_MIN)
            self._scroll.setMaximumHeight(16777215)
        else:
            compact = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.setSizePolicy(compact)
            self._scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self._scroll.setMinimumHeight(0)
            self._scroll.setMaximumHeight(200)

    def set_script(self, script_name: str) -> None:
        """Rebuild the form for *script_name*'s registered pipeline stages and CLI parameters."""
        if script_name == self._script:
            return
        self._script = script_name
        self._clear_form()
        if not script_name:
            self._hint.setText("Select a pipeline operation.")
            return

        self._pipeline_def = pipeline_def_for_script(script_name)
        if self._pipeline_def is None:
            self._hint.setText(f"No stage definition for {script_name!r}.")
            return

        self._hint.setText(f"{self._pipeline_def.title}: choose stages and options.")
        self._build_stage_rows(self._pipeline_def.stages)
        self._build_param_fields(script_name, self._pipeline_def)
        self._refresh_stage_inputs()

    def _clear_form(self) -> None:
        """Remove every stage-row and parameter widget from the form."""
        self._stage_rows.clear()
        self._param_fields.clear()
        self._pipeline_def = None
        while self._stages_layout.count():
            item = self._stages_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        while self._params_form.rowCount():
            self._params_form.removeRow(0)

    def _build_stage_rows(self, stages: tuple[PipelineStageSpec, ...]) -> None:
        """Add one enable-checkbox group box (with an empty inputs sub-form) per stage in *stages*."""
        for stage in stages:
            box = QGroupBox(stage.label)
            box_layout = QVBoxLayout()
            cb = QCheckBox(f"Run {stage.id}")
            cb.setChecked(stage.default_enabled)
            cb.toggled.connect(self._refresh_stage_inputs)
            desc = QLabel(stage.description)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: palette(mid);")
            inputs_host = QWidget()
            inputs_form = QFormLayout()
            inputs_host.setLayout(inputs_form)
            input_widgets = {}
            box_layout.addWidget(cb)
            if stage.description:
                box_layout.addWidget(desc)
            box_layout.addWidget(inputs_host)
            box.setLayout(box_layout)
            self._stages_layout.addWidget(box)
            self._stage_rows.append(
                _StageRow(stage, cb, inputs_host, inputs_form, input_widgets)
            )

    def _build_param_fields(self, script_name: str, pipeline_def: PipelineStageDef) -> None:
        """Build a form row (checkbox for flags, line edit with optional file/dir browse button
        otherwise) for every visible CLI parameter of *script_name*'s Click command."""
        cmd = _load_click_command(script_name)
        if cmd is None:
            return
        for param in cmd.params:
            if getattr(param, "hidden", False):
                continue
            if param.name in pipeline_def.hidden_params:
                continue
            flag = _cli_long_option(param)
            label = f"{flag.lstrip('-')}{' *' if param.required else ''}"
            if isinstance(param, click.Option) and param.is_flag:
                w = QCheckBox(param.help or param.name)
                default = param.default
                w.setChecked(bool(default) if default is not CLICK_UNSET else False)
                self._params_form.addRow(label, w)
                self._param_fields.append(_ParamField(param, w))
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
            if isinstance(param.type, click.Path):
                browse = QPushButton("…")
                browse.setFixedWidth(28)

                def _pick(le=edit, prm=param) -> None:
                    """Open a file/directory browser (per *prm*'s Path type) and fill *le* with the choice."""
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
            self._params_form.addRow(label, row)
            self._param_fields.append(_ParamField(param, edit))

    def _enabled_stages(self) -> dict[str, bool]:
        """``{stage_id: is_checked}`` for every stage row in the form."""
        return {row.stage.id: row.checkbox.isChecked() for row in self._stage_rows}

    def _layer_names(self) -> list[str]:
        """Names of every layer in the bound viewer, or ``[]`` if no viewer is bound."""
        viewer = self._viewer
        if viewer is None or not getattr(viewer, "layers", None):
            return []
        return [str(layer.name) for layer in viewer.layers]

    def refresh_layer_combos(self) -> None:
        """Repopulate every layer-selection combo box (and active-layer labels) with the current
        viewer layer names, preserving each combo's selection where still valid."""
        names = self._layer_names()
        active = self._active_layer_name()
        for row in self._stage_rows:
            for key, widget in row.input_widgets.items():
                if key.endswith(":active") and isinstance(widget, QLabel):
                    widget.setText(active or "(no active layer)")
                    continue
                if not isinstance(widget, QComboBox):
                    continue
                current = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItem("(none)", None)
                for name in names:
                    widget.addItem(name, name)
                idx = widget.findText(current)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                elif active and widget.count() > 1:
                    idx = widget.findText(active)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                widget.blockSignals(False)

    def _active_layer_name(self) -> str | None:
        """Name of the viewer's active (or last) layer, or ``None`` if there are no layers/viewer."""
        viewer = self._viewer
        if viewer is None or not viewer.layers:
            return None
        layer = viewer.layers.selection.active or viewer.layers[-1]
        return str(layer.name) if layer is not None else None

    def _refresh_stage_inputs(self) -> None:
        """Rebuild each stage row's input-slot widgets (layer combo boxes or an active-layer label)
        based on which stages are currently enabled."""
        if self._pipeline_def is None:
            return
        enabled = self._enabled_stages()
        stages = self._pipeline_def.stages
        active_name = self._active_layer_name()

        for idx, row in enumerate(self._stage_rows):
            while row.inputs_form.rowCount():
                row.inputs_form.removeRow(0)
            row.input_widgets.clear()
            row.inputs_host.setVisible(enabled.get(row.stage.id, False))

            visible_keys = visible_stage_inputs(stages, enabled, idx)
            if not visible_keys:
                continue

            for key in visible_keys:
                label = input_label_for_key(stages, key)
                if key.endswith(":active"):
                    value = active_name or "(no active layer)"
                    w = QLabel(value)
                    w.setWordWrap(True)
                    row.inputs_form.addRow(f"{label}:", w)
                    row.input_widgets[key] = w
                    continue

                combo = QComboBox()
                combo.addItem("(none)", None)
                for name in self._layer_names():
                    combo.addItem(name, name)
                if active_name:
                    combo_idx = combo.findText(active_name)
                    if combo_idx >= 0:
                        combo.setCurrentIndex(combo_idx)
                row.inputs_form.addRow(f"{label}:", combo)
                row.input_widgets[key] = combo

    def _layer_selections(self) -> dict[tuple[str, str], str | None]:
        """Current ``{(stage_id, slot): selected_layer_name}`` mapping from every layer combo box."""
        out = {}
        for row in self._stage_rows:
            for key, widget in row.input_widgets.items():
                if key.endswith(":active") or not isinstance(widget, QComboBox):
                    continue
                if ":" not in key:
                    continue
                stage_id, slot = key.split(":", 1)
                out[(stage_id, slot)] = widget.currentData()
        return out

    def build_layer_bindings(
        self,
        *,
        active_layer = None,
    ) -> list[StageInputBinding]:
        """Resolve every stage's input slot to a concrete layer-name binding, given the current form
        selections and the viewer's active layer."""
        if self._pipeline_def is None:
            return []
        active_name = str(active_layer.name) if active_layer is not None else self._active_layer_name()
        return resolve_stage_input_bindings(
            self._pipeline_def.stages,
            self._enabled_stages(),
            active_layer_name=active_name,
            layer_selections=self._layer_selections(),
        )

    def build_argv(
        self,
        exe: str,
        *,
        active_layer = None,
    ) -> list[str]:
        """Build the full CLI argv for *exe*: enabled ``--stages``, every filled CLI parameter, and
        resolved layer-input bindings."""
        if self._pipeline_def is None:
            raise ValueError("Select a pipeline operation.")

        enabled_ids = [
            row.stage.id for row in self._stage_rows if row.checkbox.isChecked()
        ]
        if not enabled_ids:
            raise ValueError("Select at least one pipeline stage.")

        argv = [exe, "--stages", ",".join(enabled_ids)]
        for field in self._param_fields:
            val = field.value()
            if val is None:
                continue
            flag = _cli_long_option(field.param)
            if isinstance(field.widget, QCheckBox):
                if val:
                    argv.append(flag)
                continue
            argv.extend([flag, str(val)])

        bindings = self.build_layer_bindings(active_layer=active_layer)
        stages = self._pipeline_def.stages
        for binding in bindings:
            if binding_requires_layer(stages, binding) and not binding.layer_name:
                label = input_label_for_key(
                    stages, f"{binding.stage_id}:{binding.input_name}"
                )
                raise ValueError(
                    f"Stage {binding.stage_id!r} requires {label!r}."
                )

        return argv


# Backward-compatible alias used by tools_dock.
PipelineCliForm = PipelineStageForm
