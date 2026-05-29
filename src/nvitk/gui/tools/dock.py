"""Tools dock: magicgui panel + label picker + pipeline CLI form + TotalSeg ROIs."""

from __future__ import annotations

from typing import Any, Callable

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from nvitk.gui.tools.gpu_toggle import build_gpu_toggle_button
from nvitk.gui.labels.catalog import guess_schema_from_layer, schema_for_totalsegmentator_task
from nvitk.gui.labels.selector import LabelSelectorWidget
from nvitk.gui.labels.visibility import (
    apply_label_visibility,
    is_label_like_layer,
    layer_in_viewer,
    restore_label_visibility,
)
from nvitk.gui.pipeline.form import PipelineCliForm
from nvitk.gui.tools.presets import cursor_voxel_indices
from nvitk.gui.tools.panel import build_tool_panel
from nvitk.gui.tools.registry import (
    TOOL_IDS_USING_LABEL_PICKER,
    tool_by_id,
    tool_id_from_label,
)
from nvitk.gui.tools.totalseg_selector import TotalSegRoiWidget


def _compact_magicgui_panel(native: QWidget) -> None:
    """Keep tool controls compact; the parent scroll area caps total height."""
    lay = native.layout()
    if lay is None:
        return
    lay.setAlignment(Qt.AlignTop)
    lay.setSpacing(6)
    native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)


def _show_label_picker(category: str, tool_id: str, layer: Any | None) -> bool:
    if not is_label_like_layer(layer):
        return False
    if tool_id in TOOL_IDS_USING_LABEL_PICKER:
        return True
    return category in (
        "Morphology",
        "Segmentation",
        "Centerline",
        "Measure",
        "Filters",
        "Restoration",
        "Transform",
        "Visualization",
    )


def build_tools_dock(
    viewer: Any,
    app_state: dict[str, Any],
    *,
    layer_display_kwargs: Callable[..., dict[str, Any]],
    on_layers_changed: Callable[[], None],
    record_step = None,
) -> tuple[QWidget, Any]:
    """Return (dock widget, magicgui tool_panel)."""
    container = QWidget()
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    label_selector = LabelSelectorWidget()
    pipeline_form = PipelineCliForm()
    pipeline_form.set_viewer(viewer)
    totalseg_roi = TotalSegRoiWidget()
    pipeline_form.setVisible(False)
    totalseg_roi.setVisible(False)
    _filtered_layer = None
    _last_active_layer_id = None

    _visibility_timer = QTimer()
    _visibility_timer.setSingleShot(True)
    _visibility_timer.setInterval(120)

    _active_sync_timer = QTimer()
    _active_sync_timer.setSingleShot(True)
    _active_sync_timer.setInterval(0)

    def _active_layer() -> Any | None:
        if not viewer.layers:
            return None
        return viewer.layers.selection.active or viewer.layers[-1]

    def _restore_filtered_layer() -> None:
        nonlocal _filtered_layer
        if _filtered_layer is not None:
            restore_label_visibility(_filtered_layer, viewer=viewer)
            _filtered_layer = None

    def _drop_filtered_layer(layer: Any | None) -> None:
        """Drop live-filter state without touching a layer that is being removed."""
        nonlocal _filtered_layer
        if layer is not None and _filtered_layer is layer:
            _filtered_layer = None

    def _apply_label_visibility() -> None:
        nonlocal _filtered_layer
        if not label_selector.isVisible():
            _restore_filtered_layer()
            return
        layer = _active_layer()
        if _filtered_layer is not None and _filtered_layer is not layer:
            restore_label_visibility(_filtered_layer, viewer=viewer)
            _filtered_layer = None
        if layer is None or not is_label_like_layer(layer):
            return
        if not layer_in_viewer(layer, viewer):
            return
        apply_label_visibility(layer, label_selector.selected_ids())
        _filtered_layer = layer

    def _schedule_label_visibility() -> None:
        _visibility_timer.start()

    def _sync_label_picker_for_layer(layer: Any | None) -> None:
        """Lightweight update when the active layer changes (no full tool resync)."""
        nonlocal _last_active_layer_id, _filtered_layer
        layer_id = id(layer) if layer is not None else None
        if layer_id == _last_active_layer_id:
            return
        _last_active_layer_id = layer_id

        cat = tool_panel.category.value
        op = tool_panel.operation.value
        tid = tool_id_from_label(cat, op) or ""
        show_labels = _show_label_picker(cat, tid, layer)

        if _filtered_layer is not None and _filtered_layer is not layer:
            restore_label_visibility(_filtered_layer, viewer=viewer)
            _filtered_layer = None

        label_selector.setVisible(show_labels)
        if show_labels and layer is not None:
            if tid == "seg_totalsegmentator":
                task = getattr(tool_panel, "task", None)
                task_val = str(task.value if task is not None else "total")
                label_selector.set_schema_key(schema_for_totalsegmentator_task(task_val))
            else:
                guessed = guess_schema_from_layer(layer)
                if guessed:
                    label_selector.set_schema_key(guessed)
            label_selector.refresh_from_layer(layer)
            _apply_label_visibility()
            _filtered_layer = layer if is_label_like_layer(layer) else None
        else:
            _restore_filtered_layer()

        tool_panel.label_ids.visible = (not show_labels) and is_label_like_layer(layer)
        _update_aux_panel_layout(show_labels)

    def _get_label_ids() -> list[int]:
        layer = _active_layer()
        if label_selector.isVisible():
            picked = label_selector.selected_ids()
            if picked:
                return picked
        if layer is not None:
            from nvitk.gui.labels.visibility import label_source_data, unique_layer_labels

            return unique_layer_labels(label_source_data(layer))
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
        layer = _active_layer()

        is_ts = tid == "seg_totalsegmentator"
        show_labels = _show_label_picker(cat, tid, layer)
        label_selector.setVisible(show_labels)
        if show_labels and layer is not None:
            if is_ts:
                task = getattr(tool_panel, "task", None)
                task_val = str(task.value if task is not None else "total")
                label_selector.set_schema_key(schema_for_totalsegmentator_task(task_val))
            else:
                guessed = guess_schema_from_layer(layer)
                if guessed:
                    label_selector.set_schema_key(guessed)
            label_selector.refresh_from_layer(layer)
            _apply_label_visibility()
            nonlocal _filtered_layer, _last_active_layer_id
            _filtered_layer = layer if is_label_like_layer(layer) else None
            _last_active_layer_id = id(layer)
        else:
            _restore_filtered_layer()
            _last_active_layer_id = id(layer) if layer is not None else None

        is_pipeline = spec is not None and spec.run_mode == "pipeline"
        pipeline_form.setVisible(is_pipeline)
        if is_pipeline and spec:
            pipeline_form.set_script(spec.cli_command)
            pipeline_form.refresh_layer_combos()

        _update_aux_panel_layout(show_labels)

        totalseg_roi.setVisible(is_ts)
        if is_ts:
            task = getattr(tool_panel, "task", None)
            task_val = str(task.value if task is not None else "total")
            totalseg_roi.set_task(task_val)

        _sync_sge_button()

        tool_panel.label_ids.visible = (not show_labels) and is_label_like_layer(layer)
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
            from nvitk.gui.tools.runner import notify

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

    from nvitk.gui.tools.registry import is_sge_capable, sge_block_reason
    from nvitk.gui.sge.retrieve import import_sge_job
    from nvitk.gui.sge.submit import submit_gui_sge

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

        expand_row = None
        if show_labels:
            expand_row = _row_label
        elif is_pipeline:
            expand_row = _row_pipeline
        elif is_ts:
            expand_row = _row_totalseg

        for i in range(layout.count()):
            layout.setStretch(i, 1 if i == expand_row else 0)

        if expand_row is None:
            layout.setStretch(_row_spacer, 1)
        else:
            layout.setStretch(_row_spacer, 0)

    def _signal_value(event: Any) -> Any:
        return event.value if hasattr(event, "value") else event

    _visibility_timer.timeout.connect(_apply_label_visibility)
    _active_sync_timer.timeout.connect(
        lambda: (
            _sync_label_picker_for_layer(_active_layer()),
            pipeline_form.refresh_layer_combos(),
        )
    )

    def _schedule_active_layer_sync() -> None:
        _active_sync_timer.start()

    def _layer_from_removing_event(event: Any) -> Any | None:
        idx = getattr(event, "index", None)
        if idx is None:
            return None
        try:
            return viewer.layers[int(idx)]
        except (IndexError, TypeError, ValueError):
            return None

    @viewer.layers.events.removing.connect
    def _on_layer_removing(event: Any) -> None:
        """Avoid restoring/modifying a layer while Napari removes it from the list."""
        _visibility_timer.stop()
        _active_sync_timer.stop()
        layer = _layer_from_removing_event(event)
        _drop_filtered_layer(layer)
        if layer is not None and label_selector._layer_ref is layer:
            label_selector._layer_ref = None
            _last_active_layer_id = None

    @viewer.layers.events.removed.connect
    def _on_layer_removed_refresh_pipeline(_event: Any) -> None:
        pipeline_form.refresh_layer_combos()
        _schedule_active_layer_sync()

    @viewer.layers.events.inserted.connect
    def _on_layer_inserted_refresh_pipeline(_event: Any) -> None:
        pipeline_form.refresh_layer_combos()

    tool_panel.category.changed.connect(lambda e: _sync_aux_panels())
    tool_panel.operation.changed.connect(lambda e: _sync_aux_panels())
    if hasattr(tool_panel, "task"):
        tool_panel.task.changed.connect(lambda e: _sync_aux_panels())

    @viewer.layers.selection.events.active.connect
    def _on_active_layer_for_labels(_event) -> None:
        _schedule_active_layer_sync()

    label_selector.selection_changed.connect(_schedule_label_visibility)

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
