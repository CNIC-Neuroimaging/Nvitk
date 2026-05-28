"""Tools dock: magicgui panel + label picker + pipeline CLI form + TotalSeg ROIs."""

from __future__ import annotations

from typing import Any, Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from nvitk.gui.gpu_toggle import build_gpu_toggle_button
from nvitk.gui.label_catalog import guess_schema_from_layer, schema_for_totalsegmentator_task
from nvitk.gui.label_selector import LabelSelectorWidget
from nvitk.gui.pipeline_form import PipelineCliForm
from nvitk.gui.tool_presets import cursor_voxel_indices
from nvitk.gui.tool_panel import build_tool_panel
from nvitk.gui.tools_registry import (
    TOOL_IDS_USING_LABEL_PICKER,
    tool_by_id,
    tool_id_from_label,
)
from nvitk.gui.totalseg_selector import TotalSegRoiWidget


def _compact_magicgui_panel(native: QWidget) -> None:
    """Keep tool controls compact; the parent scroll area caps total height."""
    lay = native.layout()
    if lay is None:
        return
    lay.setAlignment(Qt.AlignTop)
    lay.setSpacing(6)
    native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)


def _show_label_picker(category: str, tool_id: str, target_mode: str) -> bool:
    if target_mode in ("label", "all_labels") and category in (
        "Morphology",
        "Segmentation",
        "Centerline",
        "Measure",
        "Filters",
        "Restoration",
        "Transform",
        "Visualization",
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
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

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

        _update_aux_panel_layout(show_labels)

        totalseg_roi.setVisible(is_ts)
        if is_ts:
            task = getattr(tool_panel, "task", None)
            task_val = str(task.value if task is not None else "total")
            totalseg_roi.set_task(task_val)

        _sync_sge_button()

        tool_panel.label_ids.visible = (not show_labels) and tm == "label"
        if hasattr(tool_panel, "correction_ids"):
            tool_panel.correction_ids.visible = (tid == "siphon_correct") and (not show_labels)
        if hasattr(tool_panel, "pipeline_preset"):
            tool_panel.pipeline_preset.visible = tid == "seg_region_grow"
        if hasattr(tool_panel, "seed_from_label"):
            tool_panel.seed_from_label.visible = tid == "seg_region_grow"
        cursor_row.setVisible(tid == "seg_region_grow")

    cursor_row = QWidget()
    cursor_layout = QHBoxLayout()
    cursor_layout.setContentsMargins(0, 0, 0, 0)
    btn_cursor_seed = QPushButton("Use cursor as seed")
    cursor_layout.addWidget(btn_cursor_seed)
    cursor_row.setLayout(cursor_layout)

    def _apply_cursor_seed() -> None:
        layer = _active_layer()
        if layer is None:
            return
        try:
            z, y, x = cursor_voxel_indices(viewer, layer)
        except Exception as exc:
            from nvitk.gui.tool_runner import notify

            notify(str(exc), error=True)
            return
        for name, val in (("seed_z", z), ("seed_y", y), ("seed_x", x)):
            w = getattr(tool_panel, name, None)
            if w is not None:
                w.value = val

    btn_cursor_seed.clicked.connect(_apply_cursor_seed)

    _compact_magicgui_panel(tool_panel.native)
    tool_scroll = QScrollArea()
    tool_scroll.setWidgetResizable(True)
    tool_scroll.setWidget(tool_panel.native)
    tool_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    tool_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    tool_scroll.setMaximumHeight(440)
    tool_scroll.setMinimumHeight(140)

    layout.addWidget(build_gpu_toggle_button(), 0)

    from nvitk.gui.tools_registry import is_sge_capable, sge_block_reason
    from nvitk.gui.sge_retrieve import import_sge_job
    from nvitk.gui.sge_submit import submit_gui_sge

    btn_run_sge = QPushButton("Run SGE")
    btn_import_sge = QPushButton("Import SGE results")
    btn_run_sge.setEnabled(False)
    btn_import_sge.setToolTip(
        "Re-import the latest finished job using session credentials "
        "(or retry if auto-import was skipped)."
    )

    def _import_sge(job_id: str | None = None) -> bool:
        return import_sge_job(
            viewer,
            app_state,
            job_id=job_id,
            parent=container,
            on_layers_changed=on_layers_changed,
        )

    app_state["import_sge_job"] = _import_sge

    def _sync_sge_button() -> None:
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value) or ""
        capable = is_sge_capable(tid)
        btn_run_sge.setEnabled(capable)
        reason = sge_block_reason(tid)
        btn_run_sge.setToolTip(
            reason
            or "Export layer, upload via SFTP, and submit Singularity job on the cluster."
        )

    def _on_run_sge() -> None:
        submit_gui_sge(
            viewer,
            tool_panel,
            app_state,
            get_label_ids=_get_label_ids,
            get_totalseg_roi=_get_totalseg_roi,
            parent=container,
        )

    def _on_import_sge() -> None:
        _import_sge()

    btn_run_sge.clicked.connect(_on_run_sge)
    btn_import_sge.clicked.connect(_on_import_sge)

    sge_row = QWidget()
    sge_layout = QHBoxLayout()
    sge_layout.setContentsMargins(0, 0, 0, 0)
    sge_layout.addWidget(btn_run_sge)
    sge_layout.addWidget(btn_import_sge)
    sge_row.setLayout(sge_layout)
    layout.addWidget(sge_row, 0)
    layout.addWidget(tool_scroll, 0)
    layout.addWidget(cursor_row, 0)
    layout.addWidget(label_selector, 0)
    layout.addWidget(totalseg_roi, 0)
    layout.addWidget(pipeline_form, 0)
    layout.addStretch(1)
    container.setLayout(layout)

    _row_label = layout.indexOf(label_selector)
    _row_pipeline = layout.indexOf(pipeline_form)
    _row_totalseg = layout.indexOf(totalseg_roi)
    _row_spacer = layout.count() - 1

    def _update_aux_panel_layout(show_labels: bool) -> None:
        is_pipeline = pipeline_form.isVisible()
        is_ts = totalseg_roi.isVisible()
        label_selector.set_expanded(show_labels)
        pipeline_form.set_expanded(is_pipeline)

        expand_row: int | None = None
        if show_labels:
            expand_row = _row_label
        elif is_pipeline:
            expand_row = _row_pipeline
        elif is_ts:
            expand_row = _row_totalseg

        for i in range(layout.count()):
            layout.setStretch(i, 1 if i == expand_row else 0)

        # If no aux panel is expanded/visible, anchor everything to the top by
        # putting the extra space into the final spacer stretch.
        if expand_row is None:
            layout.setStretch(_row_spacer, 1)
        else:
            layout.setStretch(_row_spacer, 0)

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
