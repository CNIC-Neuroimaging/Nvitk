"""Tools dock: magicgui panel + label picker + pipeline CLI form + TotalSeg ROIs."""

from __future__ import annotations

from typing import Any, Callable

from qtpy.QtWidgets import QVBoxLayout, QWidget

from nvitk.gui.gpu_toggle import build_gpu_toggle_button
from nvitk.gui.label_catalog import guess_schema_from_layer, schema_for_totalsegmentator_task
from nvitk.gui.label_selector import LabelSelectorWidget
from nvitk.gui.pipeline_form import PipelineCliForm
from nvitk.gui.tool_panel import build_tool_panel
from nvitk.gui.tools_registry import (
    TOOL_IDS_USING_LABEL_PICKER,
    tool_by_id,
    tool_id_from_label,
)
from nvitk.gui.totalseg_selector import TotalSegRoiWidget


def _show_label_picker(category: str, tool_id: str, target_mode: str) -> bool:
    if target_mode in ("label", "all_labels") and category in (
        "Morphology",
        "Measure",
        "Filters",
        "Restoration",
        "Transform",
    ):
        return True
    return tool_id in TOOL_IDS_USING_LABEL_PICKER


def build_tools_dock(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    layer_display_kwargs: Callable[..., dict[str, Any]],
    on_layers_changed: Callable[[], None],
    record_step: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[QWidget, Any]:
    """Return (dock widget, magicgui tool_panel)."""
    container = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)

    label_selector = LabelSelectorWidget()
    pipeline_form = PipelineCliForm()
    totalseg_roi = TotalSegRoiWidget()
    pipeline_form.setVisible(False)
    totalseg_roi.setVisible(False)

    def _active_layer() -> Any | None:
        if not viewer.layers:
            return None
        return viewer.layers.selection.active or viewer.layers[-1]

    def _get_label_ids() -> list[int]:
        layer = _active_layer()
        if label_selector.isVisible():
            picked = label_selector.selected_ids()
            if picked:
                return picked
        if layer is not None:
            from nvitk.gui.label_selector import unique_layer_labels
            from nvitk.core.array import to_numpy

            return unique_layer_labels(to_numpy(layer.data))
        return []

    def _get_totalseg_roi() -> list[str] | None:
        if totalseg_roi.isVisible():
            return totalseg_roi.selected_roi_names()
        return None

    tool_panel = build_tool_panel(
        viewer,
        app_state,
        layer_display_kwargs=layer_display_kwargs,
        on_layers_changed=on_layers_changed,
        record_step=record_step,
        get_label_ids=_get_label_ids,
        get_pipeline_argv_builder=lambda: pipeline_form,
        get_totalseg_roi=_get_totalseg_roi,
        label_selector=label_selector,
    )

    def _sync_aux_panels() -> None:
        cat = tool_panel.category.value
        op = tool_panel.operation.value
        tid = tool_id_from_label(cat, op) or ""
        spec = tool_by_id(tid)
        tm = tool_panel.target_mode.value

        is_ts = tid == "seg_totalsegmentator"
        show_labels = _show_label_picker(cat, tid, tm)
        label_selector.setVisible(show_labels)
        if show_labels:
            layer = _active_layer()
            if is_ts:
                task = getattr(tool_panel, "task", None)
                task_val = str(task.value if task is not None else "total")
                label_selector.set_schema_key(schema_for_totalsegmentator_task(task_val))
            else:
                guessed = guess_schema_from_layer(layer)
                if guessed:
                    label_selector.set_schema_key(guessed)
            label_selector.refresh_from_layer(layer)

        is_pipeline = spec is not None and spec.run_mode == "pipeline"
        pipeline_form.setVisible(is_pipeline)
        if is_pipeline and spec:
            pipeline_form.set_script(spec.cli_command)

        totalseg_roi.setVisible(is_ts)
        if is_ts:
            task = getattr(tool_panel, "task", None)
            task_val = str(task.value if task is not None else "total")
            totalseg_roi.set_task(task_val)

        tool_panel.label_ids.visible = (not show_labels) and tm == "label"
        if hasattr(tool_panel, "correction_ids"):
            tool_panel.correction_ids.visible = (tid == "siphon_correct") and (not show_labels)

    layout.addWidget(build_gpu_toggle_button())
    layout.addWidget(tool_panel.native)
    layout.addWidget(label_selector)
    layout.addWidget(totalseg_roi)
    layout.addWidget(pipeline_form)
    container.setLayout(layout)

    def _signal_value(event: Any) -> Any:
        return event.value if hasattr(event, "value") else event

    tool_panel.category.changed.connect(lambda e: _sync_aux_panels())
    tool_panel.operation.changed.connect(lambda e: _sync_aux_panels())
    tool_panel.target_mode.changed.connect(lambda e: _sync_aux_panels())
    if hasattr(tool_panel, "task"):
        tool_panel.task.changed.connect(lambda e: _sync_aux_panels())

    @viewer.layers.events.changed.connect
    def _on_layers_event(_event) -> None:
        if label_selector.isVisible():
            label_selector.refresh_from_layer(_active_layer())

    def _refresh_label_selector() -> None:
        layer = _active_layer()
        if label_selector.schema_key() == "generic":
            guessed = guess_schema_from_layer(layer)
            if guessed:
                label_selector.set_schema_key(guessed)
        label_selector.refresh_from_layer(layer)

    label_selector._btn_refresh.clicked.connect(_refresh_label_selector)
    _sync_aux_panels()
    return container, tool_panel
