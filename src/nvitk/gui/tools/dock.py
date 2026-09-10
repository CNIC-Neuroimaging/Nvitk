"""Tools dock: magicgui panel + label picker + pipeline CLI form + TotalSeg ROIs."""

from __future__ import annotations

from typing import Any, Callable

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import COLOR_MUTED, SPACE, SPACE_TIGHT

from nvitk.gui.tools.gpu_toggle import build_gpu_toggle_button
from nvitk.gui.tools.orient_quick import build_orientation_quick_button
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


#: Widest a parameter name may get before it wraps. magicgui lays each parameter
#: out as ``[label | widget]``; without a cap, one long name (``"Reset affine to
#: target codes (ignore wrong header)"``) sets the width of the entire dock and
#: the value column falls off the right edge.
_TOOL_LABEL_MAX_WIDTH = 150

#: Height cap for the tool description caption.
_TOOL_HELP_MAX_HEIGHT = 90

#: Floor for the tool form's scroll area. Its ceiling is a share of the dock's
#: own height rather than a constant, so maximising the window actually gives the
#: form more room instead of leaving it pinned at a fixed size.
_TOOL_SCROLL_MIN_HEIGHT = 140
_TOOL_SCROLL_HEIGHT_SHARE = 0.55


def _compact_magicgui_panel(native: QWidget) -> None:
    """Keep tool controls compact and inside the dock's width.

    The parent scroll area caps total height; this caps the width, wrapping
    parameter names and letting the value widgets shrink so a narrow dock shows
    a whole form rather than a clipped one.
    """
    lay = native.layout()
    if lay is None:
        return
    lay.setAlignment(Qt.AlignTop)
    lay.setSpacing(SPACE_TIGHT)
    native.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    _cap_form_labels(native)
    for combo in native.findChildren(QComboBox):
        combo.setMinimumContentsLength(8)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    for edit in native.findChildren(QLineEdit):
        edit.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    for box in native.findChildren(QCheckBox):
        # A checkbox carries its own caption and cannot wrap it; elide instead of
        # letting it dictate the dock width.
        box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        if not box.toolTip():
            box.setToolTip(box.text())
    for text in native.findChildren(QTextEdit):
        text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)


def _cap_form_labels(native: QWidget) -> None:
    """Wrap magicgui's parameter names and stop them setting the panel's width.

    magicgui aligns the label column by giving every label the *widest* label's
    width as a hard ``minimumWidth``. That silently defeats a maximum width — Qt
    resolves max < min in favour of min — so the floor has to be cleared first.
    Re-applied whenever the form changes, because magicgui re-runs that alignment.
    """
    for label in native.findChildren(QLabel):
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setMaximumWidth(_TOOL_LABEL_MAX_WIDTH)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


def _style_operation_help(panel: Any) -> None:
    """Show the tool description as a caption instead of a form field.

    magicgui renders it as a labelled, bordered ``TextEdit`` that grows with the
    text — for a wordy tool that pushes the actual parameters off the panel. Drop
    the label and the well, cap the height, and let it read as help text.
    """
    widget = getattr(panel, "operation_help", None)
    native = getattr(widget, "native", None)
    if native is None:
        return
    native.setFrameShape(QFrame.NoFrame)
    native.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    native.setStyleSheet(
        f"QTextEdit {{ background: transparent; border: none; color: {COLOR_MUTED};"
        " padding: 0; }"
    )

    def _fit_to_text() -> None:
        """Size the caption to its wrapped text, up to the cap."""
        height = int(native.document().size().height()) + 4
        native.setFixedHeight(max(18, min(height, _TOOL_HELP_MAX_HEIGHT)))

    # The document relays out whenever the text or the available width changes,
    # so a short description never leaves a block of empty space behind it.
    native.document().documentLayout().documentSizeChanged.connect(
        lambda _size: _fit_to_text()
    )
    _fit_to_text()

    # The row is [label | text]; the caption speaks for itself.
    row = native.parentWidget()
    if row is not None:
        for label in row.findChildren(QLabel):
            label.hide()


def _show_label_picker(category: str, tool_id: str, layer: Any | None) -> bool:
    """True if the label-selector widget should be shown for *tool_id* given the active *layer* and
    its category (label-picker tools always; certain categories when the layer is label-like)."""
    if not is_label_like_layer(layer):
        return False
    if tool_id in TOOL_IDS_USING_LABEL_PICKER:
        return True
    if tool_id == "viz_vessel_cross_sections":
        return False
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
    layout.setSpacing(SPACE)

    label_selector = LabelSelectorWidget()
    label_selector.set_viewer(viewer)
    pipeline_form = PipelineCliForm()
    pipeline_form.set_viewer(viewer)
    totalseg_roi = TotalSegRoiWidget()
    pipeline_form.setVisible(False)
    totalseg_roi.setVisible(False)
    _last_active_layer_id = None

    _visibility_timer = QTimer()
    _visibility_timer.setSingleShot(True)
    _visibility_timer.setInterval(120)

    _active_sync_timer = QTimer()
    _active_sync_timer.setSingleShot(True)
    _active_sync_timer.setInterval(0)

    def _active_layer() -> Any | None:
        """The viewer's active (or last) layer, or ``None`` if there are no layers."""
        if not viewer.layers:
            return None
        return viewer.layers.selection.active or viewer.layers[-1]

    def _apply_label_visibility() -> None:
        """Filter the picker's bound layer to the checked label ids.

        Each filter is recorded on the layer itself, so a layer keeps the selection
        it was given while other layers are active; checking every id present is
        what clears it again.
        """
        if not label_selector.isVisible():
            return
        layer = label_selector.current_layer()
        if layer is None or not is_label_like_layer(layer):
            return
        if not layer_in_viewer(layer, viewer):
            return
        ids = label_selector.selected_ids()
        present = set(label_selector.available_ids())
        if ids and present and set(ids) >= present:
            restore_label_visibility(layer, viewer=viewer)
            return
        apply_label_visibility(layer, ids)

    def _schedule_label_visibility() -> None:
        """Debounce a call to :func:`_apply_label_visibility` via the visibility timer."""
        _visibility_timer.start()

    def _sync_label_picker_for_layer(layer: Any | None) -> None:
        """Lightweight update when the active layer changes (no full tool resync)."""
        nonlocal _last_active_layer_id
        layer_id = id(layer) if layer is not None else None
        if layer_id == _last_active_layer_id:
            return
        _last_active_layer_id = layer_id

        cat = tool_panel.category.value
        op = tool_panel.operation.value
        tid = tool_id_from_label(cat, op) or ""
        show_labels = _show_label_picker(cat, tid, layer)

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
            # refresh may promote Image → Labels; track the live layer.
            layer = label_selector.current_layer() or _active_layer()
            _last_active_layer_id = id(layer) if layer is not None else None
            _apply_label_visibility()

        tool_panel.label_ids.visible = (not show_labels) and is_label_like_layer(layer)
        _update_aux_panel_layout(show_labels)

    def _get_label_ids() -> list[int]:
        """Selected label ids from the picker if visible and non-empty, else every label present in
        the active layer."""
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
        """Selected TotalSegmentator ROI names if the ROI widget is visible, else ``None``."""
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
        """Full resync of every auxiliary panel (label picker, pipeline form, TotalSeg ROI widget,
        cursor/CoW rows, SGE button) for the currently selected category/operation."""
        nonlocal _last_active_layer_id
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
            layer = label_selector.current_layer() or _active_layer()
            _apply_label_visibility()
            _last_active_layer_id = id(layer) if layer is not None else None
        else:
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
        _fit_timer.start()

        tool_panel.label_ids.visible = (not show_labels) and is_label_like_layer(layer)
        if hasattr(tool_panel, "correction_ids"):
            tool_panel.correction_ids.visible = (tid == "siphon_correct") and (not show_labels)
        if hasattr(tool_panel, "pipeline_preset"):
            tool_panel.pipeline_preset.visible = tid == "seg_region_grow"
        if hasattr(tool_panel, "seed_from_label"):
            tool_panel.seed_from_label.visible = tid == "seg_region_grow"
        cursor_row.setVisible(tid == "seg_region_grow")
        _sync_cow_row()

    cursor_row = QWidget()
    cursor_layout = QHBoxLayout()
    cursor_layout.setContentsMargins(0, 0, 0, 0)
    btn_cursor_seed = QPushButton("Use cursor as seed")
    cursor_layout.addWidget(btn_cursor_seed)
    cursor_row.setLayout(cursor_layout)

    def _apply_cursor_seed() -> None:
        """Fill the region-grow seed coordinate widgets from the current cursor voxel position."""
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

    # Mouse TOF CoW Stage-2 controls (visible while tool selected or session active).
    cow_row = QWidget()
    cow_layout = QVBoxLayout()
    cow_layout.setContentsMargins(0, 0, 0, 0)
    cow_layout.setSpacing(4)
    cow_status = QLabel("Mouse TOF CoW Stage 2: idle")
    cow_status.setWordWrap(True)
    cow_btn_row = QWidget()
    cow_btn_layout = QHBoxLayout()
    cow_btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_cow_add = QPushButton("Add CC to tree")
    btn_cow_deselect = QPushButton("Deselect")
    btn_cow_done = QPushButton("Tree done")
    btn_cow_cancel = QPushButton("Cancel")
    cow_btn_layout.addWidget(btn_cow_add)
    cow_btn_layout.addWidget(btn_cow_deselect)
    cow_btn_layout.addWidget(btn_cow_done)
    cow_btn_layout.addWidget(btn_cow_cancel)
    cow_btn_row.setLayout(cow_btn_layout)
    cow_layout.addWidget(cow_status)
    cow_layout.addWidget(cow_btn_row)
    cow_row.setLayout(cow_layout)
    cow_row.setVisible(False)

    def _sync_cow_row() -> None:
        """Show/hide and update the Mouse TOF CoW Stage-2 status row and buttons based on session state."""
        from nvitk.gui.lab.mouse_tof_cow import get_session, session_active

        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value) or ""
        active = session_active(viewer)
        show = tid == "lab_mouse_tof_cow" or active
        cow_row.setVisible(show)
        sess = get_session(viewer)
        if sess is not None:
            cow_status.setText(sess.status_text())
            enabled = True
        else:
            cow_status.setText(
                "Mouse TOF CoW: Run the tool on a TOF Image to start Stage 1, "
                "then click CCs on the labeled layer."
            )
            enabled = False
        btn_cow_add.setEnabled(enabled)
        btn_cow_deselect.setEnabled(enabled)
        btn_cow_done.setEnabled(enabled)
        btn_cow_cancel.setEnabled(active)

    def _cow_add() -> None:
        """Assign the currently highlighted CC to the tree being built."""
        from nvitk.gui.lab.mouse_tof_cow import get_session
        from nvitk.gui.tools.runner import notify

        sess = get_session(viewer)
        if sess is None:
            notify("No active Mouse TOF CoW session. Run the tool first.", error=True)
            return
        sess.add_highlighted_cc()
        _sync_cow_row()

    def _cow_deselect() -> None:
        """Clear the currently highlighted CC without assigning it."""
        from nvitk.gui.lab.mouse_tof_cow import get_session
        from nvitk.gui.tools.runner import notify

        sess = get_session(viewer)
        if sess is None:
            notify("No active Mouse TOF CoW session. Run the tool first.", error=True)
            return
        sess.clear_highlight()
        _sync_cow_row()

    def _cow_done() -> None:
        """Finish the current vessel tree and advance to the next (or finalize if this was the last)."""
        from nvitk.gui.lab.mouse_tof_cow import get_session
        from nvitk.gui.tools.runner import notify

        sess = get_session(viewer)
        if sess is None:
            notify("No active Mouse TOF CoW session. Run the tool first.", error=True)
            return
        sess.finish_current_tree()
        _sync_cow_row()

    def _cow_cancel() -> None:
        """Cancel the active Mouse TOF CoW Stage-2 session without finalizing."""
        from nvitk.gui.lab.mouse_tof_cow import cancel_session

        cancel_session(viewer)
        _sync_cow_row()

    btn_cow_add.clicked.connect(_cow_add)
    btn_cow_deselect.clicked.connect(_cow_deselect)
    btn_cow_done.clicked.connect(_cow_done)
    btn_cow_cancel.clicked.connect(_cow_cancel)

    from nvitk.gui.lab.mouse_tof_cow import set_ui_hooks

    set_ui_hooks(status=lambda text: cow_status.setText(text), visibility=_sync_cow_row)

    _compact_magicgui_panel(tool_panel.native)
    _style_operation_help(tool_panel)
    tool_scroll = QScrollArea()
    tool_scroll.setWidgetResizable(True)
    tool_scroll.setWidget(tool_panel.native)
    tool_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    tool_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    tool_scroll.setMinimumHeight(_TOOL_SCROLL_MIN_HEIGHT)

    _fit_timer = QTimer()
    _fit_timer.setSingleShot(True)
    _fit_timer.setInterval(0)

    def _fit_tool_scroll() -> None:
        """Give the tool form exactly the height it needs, up to the cap.

        A ``QScrollArea`` reports a size hint of its own that can fall short of
        the form inside it, which left the Run button below the fold while the
        label picker underneath sat half empty. Each tool has a different set of
        parameters, so this is recomputed whenever the form changes.
        """
        _cap_form_labels(tool_panel.native)
        # A wrapped label's height depends on the width it ends up with, which is
        # not settled until the form has been laid out at the viewport width — so
        # ask for the height at that width when the layout can answer, and fall
        # back to the plain hint when it cannot.
        width = tool_scroll.viewport().width() or tool_scroll.width()
        needed = tool_panel.native.sizeHint().height()
        form_layout = tool_panel.native.layout()
        if width > 0 and form_layout is not None and form_layout.hasHeightForWidth():
            needed = max(needed, form_layout.heightForWidth(width))
        # The ceiling tracks the dock, so a taller window shows more of the form
        # rather than handing every new pixel to the panel underneath.
        available = container.height() or tool_scroll.height()
        ceiling = max(_TOOL_SCROLL_MIN_HEIGHT, int(available * _TOOL_SCROLL_HEIGHT_SHARE))
        tool_scroll.setMaximumHeight(ceiling)
        tool_scroll.setMinimumHeight(max(_TOOL_SCROLL_MIN_HEIGHT, min(needed, ceiling)))

    # Widget visibility settles during the current event-loop pass, so measure
    # the form on the next one.
    _fit_timer.timeout.connect(_fit_tool_scroll)

    top_row = QWidget()
    top_row_layout = QHBoxLayout()
    top_row_layout.setContentsMargins(0, 0, 0, 0)
    top_row_layout.setSpacing(6)
    top_row_layout.addWidget(build_gpu_toggle_button(), 1)
    top_row_layout.addWidget(build_orientation_quick_button(viewer), 1)
    top_row.setLayout(top_row_layout)
    layout.addWidget(top_row, 0)

    btn_ortho = QPushButton("Orthogonal views")
    btn_ortho.setToolTip(
        "Open the axial / coronal / sagittal views of the active layer in their own "
        "dock, with the 3D plane and see-inside controls."
    )

    def _open_ortho_views() -> None:
        """Open (or re-focus) the orthogonal-views dock on the active layer."""
        from nvitk.gui.tools.runner import log_tool_failure, notify

        layer = _active_layer()
        try:
            from nvitk.gui.viz.ortho_panel import open_ortho_views

            open_ortho_views(viewer, layer)
        except Exception as exc:  # noqa: BLE001
            log_tool_failure(exc)
            notify(f"Could not open the orthogonal views: {exc}", error=True)

    btn_ortho.clicked.connect(_open_ortho_views)
    layout.addWidget(btn_ortho, 0)

    from nvitk.gui.tools.registry import is_sge_capable, sge_block_reason
    from nvitk.gui.sge.submit import submit_gui_sge

    btn_run_sge = QPushButton("Run SGE")
    btn_run_sge.setEnabled(False)

    def _sync_sge_button() -> None:
        """Enable/disable the Run SGE button and set its tooltip for the currently selected tool."""
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value) or ""
        capable = is_sge_capable(tid)
        btn_run_sge.setEnabled(capable)
        reason = sge_block_reason(tid)
        btn_run_sge.setToolTip(
            reason
            or "Export layer, upload via SFTP, and submit Singularity job on the cluster."
        )

    def _on_run_sge() -> None:
        """Handle the Run SGE button: submit the current tool invocation as a remote SGE job."""
        submit_gui_sge(
            viewer,
            tool_panel,
            app_state,
            get_label_ids=_get_label_ids,
            get_totalseg_roi=_get_totalseg_roi,
            parent=container,
        )

    btn_run_sge.clicked.connect(_on_run_sge)
    layout.addWidget(btn_run_sge, 0)
    layout.addWidget(tool_scroll, 0)
    layout.addWidget(cursor_row, 0)
    layout.addWidget(cow_row, 0)
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
        """Give the currently relevant auxiliary panel (label picker, pipeline form, or TotalSeg ROI
        widget) the stretch factor in the dock layout, collapsing the others."""
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
        """Extract the new value from a magicgui change *event* (or pass through a raw value)."""
        return event.value if hasattr(event, "value") else event

    _visibility_timer.timeout.connect(_apply_label_visibility)
    _active_sync_timer.timeout.connect(
        lambda: (
            _sync_label_picker_for_layer(_active_layer()),
            pipeline_form.refresh_layer_combos(),
        )
    )

    def _schedule_active_layer_sync() -> None:
        """Debounce a call to resync the label picker and pipeline form for the active layer."""
        _active_sync_timer.start()

    def _layer_from_removing_event(event: Any) -> Any | None:
        """Layer instance about to be removed, from a Napari ``removing`` event's index."""
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
        nonlocal _last_active_layer_id
        _visibility_timer.stop()
        _active_sync_timer.stop()
        layer = _layer_from_removing_event(event)
        if layer is not None and label_selector._layer_ref is layer:
            label_selector._layer_ref = None
            _last_active_layer_id = None

    @viewer.layers.events.removed.connect
    def _on_layer_removed_refresh_pipeline(_event: Any) -> None:
        """Refresh layer-picker widgets across the dock after a layer is removed."""
        pipeline_form.refresh_layer_combos()
        from nvitk.gui.tools.panel import _update_reference_layers

        _update_reference_layers(tool_panel, viewer)
        _schedule_active_layer_sync()

    @viewer.layers.events.inserted.connect
    def _on_layer_inserted_refresh_pipeline(_event: Any) -> None:
        """Refresh layer-picker widgets across the dock after a layer is added."""
        pipeline_form.refresh_layer_combos()
        from nvitk.gui.tools.panel import _update_reference_layers

        _update_reference_layers(tool_panel, viewer)

    class _RefitOnResize(QObject):
        """Re-run the tool-form fit whenever the dock changes height."""

        def eventFilter(self, obj: Any, event: Any) -> bool:
            """Schedule a refit on resize, without consuming the event."""
            if event.type() == QEvent.Resize:
                _fit_timer.start()
            return False

    _refit_filter = _RefitOnResize(container)
    container.installEventFilter(_refit_filter)

    tool_panel.category.changed.connect(lambda e: _sync_aux_panels())
    tool_panel.operation.changed.connect(lambda e: _sync_aux_panels())
    if hasattr(tool_panel, "task"):
        tool_panel.task.changed.connect(lambda e: _sync_aux_panels())

    @viewer.layers.selection.events.active.connect
    def _on_active_layer_for_labels(_event) -> None:
        """Debounce a resync of the label picker/pipeline form when the active layer changes."""
        _schedule_active_layer_sync()

    label_selector.selection_changed.connect(_schedule_label_visibility)

    def _refresh_label_selector() -> None:
        """Manually re-guess the schema (if still generic) and refresh the label picker for the
        active layer."""
        layer = _active_layer()
        if label_selector.schema_key() == "generic":
            guessed = guess_schema_from_layer(layer)
            if guessed:
                label_selector.set_schema_key(guessed)
        label_selector.refresh_from_layer(layer)

    label_selector._btn_refresh.clicked.connect(_refresh_label_selector)
    _sync_aux_panels()
    return container, tool_panel
