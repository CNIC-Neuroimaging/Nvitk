"""Imaging (tools) dock: magicgui panel + pipeline CLI form + TotalSeg ROIs.

Label-layer tools run on the labels currently *shown* (the live filter set in
the Labels tab or from a label layer's ▾ in the layer list), so the dock no
longer carries a picker of its own.
"""

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
from nvitk.gui.labels.catalog import layer_schema_key
from nvitk.gui.labels.visibility import (
    is_label_like_layer,
    stored_visible_ids,
)
from nvitk.gui.pipeline.form import PipelineCliForm
from nvitk.gui.tools.presets import cursor_voxel_indices
from nvitk.gui.tools.panel import build_tool_panel
from nvitk.gui.tools.registry import (
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

#: Floor for the tool form's scroll area. While a pipeline form or the ROI list
#: shares the dock, the form's ceiling is a share of the dock's own height, so the
#: other panel keeps the rest; on its own the form takes the whole dock.
_TOOL_SCROLL_MIN_HEIGHT = 140
_TOOL_SCROLL_HEIGHT_SHARE = 0.4


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
    width as a hard ``minimumWidth``. That silently defeats a maximum width (Qt
    resolves max < min in favour of min), so the floor has to be cleared first.
    Re-applied whenever the form changes, because magicgui re-runs that alignment.
    """
    # A word-wrapped label only reports the height of the lines it needs
    # if its size policy says its height depends on its width. Without this the
    # label reports one line, the row is laid out one line tall, and the second
    # line is drawn outside it, which is also why the heightForWidth branch in
    # _fit_tool_scroll never fired.
    policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    policy.setHeightForWidth(True)
    labels = native.findChildren(QLabel)
    # The dock stylesheet's font reaches the labels that existed when the dock was
    # first polished; a parameter widget inserted (or re-inserted) later comes in
    # at the default weight and reads as a different kind of field. Match them.
    reference = next((lb.font() for lb in labels if lb.font().bold()), None)
    for label in labels:
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setMaximumWidth(_TOOL_LABEL_MAX_WIDTH)
        label.setSizePolicy(policy)
        if reference is not None and not label.font().bold() and not label.styleSheet():
            label.setFont(reference)


def _style_operation_help(panel: Any) -> None:
    """Show the tool description as a caption instead of a form field.

    magicgui renders it as a labelled, bordered ``TextEdit`` that grows with the
    text, for a wordy tool that pushes the actual parameters off the panel. Drop
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


#: Categories whose tools act on the selected labels of a label layer.
_LABEL_CATEGORIES = frozenset({
    "Morphology",
    "Segmentation",
    "Centerline",
    "Measure",
    "Filters",
    "Restoration",
    "Transform",
    "Interpolation",
    "Visualization",
})


def _tool_uses_labels(category: str, tool_id: str) -> bool:
    """True when *tool_id* runs on a label layer's selected labels."""
    from nvitk.gui.tools.registry import TOOL_IDS_USING_LABEL_PICKER

    if tool_id in TOOL_IDS_USING_LABEL_PICKER:
        return True
    if tool_id == "viz_vessel_cross_sections":
        return False
    return category in _LABEL_CATEGORIES


def _panel_entries(viewer: Any) -> list[tuple[str, str]]:
    """``(object name, title)`` of every nvitk panel, for the search."""
    try:
        from nvitk.gui.core.workspace import list_panels

        return list_panels(viewer)
    except Exception:  # noqa: BLE001
        return []


def _raise_panel(viewer: Any, object_name: str) -> None:
    from nvitk.gui.core.workspace import raise_panel

    raise_panel(viewer, object_name)


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

    pipeline_form = PipelineCliForm()
    pipeline_form.set_viewer(viewer)
    totalseg_roi = TotalSegRoiWidget()
    pipeline_form.setVisible(False)
    totalseg_roi.setVisible(False)

    # Parented to the dock: an ownerless QTimer outlives the widgets its
    # callback touches, and a single-shot still pending when the window closes
    # then fires against deleted C++ objects on the way out.
    _active_sync_timer = QTimer(container)
    _active_sync_timer.setSingleShot(True)
    _active_sync_timer.setInterval(0)

    def _active_layer() -> Any | None:
        """The viewer's active (or last) layer, or ``None`` if there are no layers."""
        if not viewer.layers:
            return None
        return viewer.layers.selection.active or viewer.layers[-1]

    def _sync_label_ids_field(layer: Any | None) -> None:
        """Offer the id field on label layers, for tools that act per label."""
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value) or ""
        tool_panel.label_ids.visible = _tool_uses_labels(
            tool_panel.category.value, tid
        ) and is_label_like_layer(layer)

    def _get_label_ids() -> list[int]:
        """Label ids a tool runs on: the ids typed in the field, else the labels shown.

        "Shown" is the layer's live filter (set in the Labels tab or from the
        layer's ▾ in the layer list), so what a tool touches is what is on screen;
        with no filter, every label present.
        """
        from nvitk.gui.tools.runner import parse_label_ids

        layer = _active_layer()
        typed = parse_label_ids(str(tool_panel.label_ids.value or ""))
        if typed:
            return typed
        if layer is None:
            return []
        shown = stored_visible_ids(layer)
        if shown is not None:
            return sorted(int(i) for i in shown)
        from nvitk.gui.labels.visibility import layer_label_ids

        return layer_label_ids(layer)

    def _get_label_schema() -> str:
        """The active layer's label vocabulary, for naming label ids in results."""
        return layer_schema_key(_active_layer()) or "generic"

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
        get_label_schema=_get_label_schema,
        get_pipeline_argv_builder=lambda: pipeline_form,
        get_totalseg_roi=_get_totalseg_roi,
    )

    def _sync_aux_panels() -> None:
        """Full resync of every auxiliary panel (pipeline form, TotalSeg ROI widget,
        cursor/CoW rows, SGE button) for the currently selected category/operation."""
        cat = tool_panel.category.value
        op = tool_panel.operation.value
        tid = tool_id_from_label(cat, op) or ""
        spec = tool_by_id(tid)
        layer = _active_layer()

        is_ts = tid == "seg_totalsegmentator"
        is_pipeline = spec is not None and spec.run_mode == "pipeline"
        pipeline_form.setVisible(is_pipeline)
        if is_pipeline and spec:
            pipeline_form.set_script(spec.cli_command)
            pipeline_form.refresh_layer_combos()

        # Visibility first: _update_aux_panel_layout decides who gets the dock's
        # spare height from isVisible().
        totalseg_roi.setVisible(is_ts)
        if is_ts:
            task = getattr(tool_panel, "task", None)
            task_val = str(task.value if task is not None else "total")
            totalseg_roi.set_task(task_val)

        _update_aux_panel_layout()

        _sync_sge_button()
        _fit_timer.start()

        _sync_label_ids_field(layer)
        if hasattr(tool_panel, "correction_ids"):
            tool_panel.correction_ids.visible = tid == "siphon_correct"
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

    _fit_timer = QTimer(tool_scroll)
    _fit_timer.setSingleShot(True)
    _fit_timer.setInterval(0)
    # "user": the splitter handle under the tool form was dragged (kept until another tool is
    # chosen); "applying": the fit itself is moving it.
    _fit_state = {"user": False, "applying": False}

    def _fit_tool_scroll() -> None:
        """Give the tool form the height it needs.

        On its own the form fills the dock (the scroll area takes the stretch and
        any spare height sits below the Run button, inside the form). Beside a
        pipeline form or the ROI list it is held to what it needs, up to a share
        of the dock, so that panel gets the rest instead of a strip at the bottom.
        A ``QScrollArea``'s own size hint can fall short of the form inside it,
        which left the Run button below the fold, so the height is measured here.
        """
        _cap_form_labels(tool_panel.native)
        # A wrapped label's height depends on the width it ends up with, which is
        # not settled until the form has been laid out at the viewport width, so
        # ask for the height at that width when the layout can answer, and fall
        # back to the plain hint when it cannot.
        width = tool_scroll.viewport().width() or tool_scroll.width()
        needed = tool_panel.native.sizeHint().height()
        form_layout = tool_panel.native.layout()
        if width > 0 and form_layout is not None and form_layout.hasHeightForWidth():
            needed = max(needed, form_layout.heightForWidth(width))
        # The form is top-aligned, so where the Run button ends is where its
        # content ends: the layout's own hint overshoots that by the spacing of
        # every hidden parameter row, which showed as a band of empty form.
        button = getattr(getattr(tool_panel, "_call_button", None), "native", None)
        if button is not None and button.isVisible() and button.height() > 0:
            from qtpy.QtCore import QPoint

            bottom = button.mapTo(tool_panel.native, QPoint(0, button.height())).y()
            margin = form_layout.contentsMargins().bottom() if form_layout is not None else 0
            if bottom > 0:
                needed = bottom + margin + SPACE_TIGHT
        needed += 2 * tool_scroll.frameWidth()
        if not (pipeline_form.isVisible() or totalseg_roi.isVisible()):
            tool_scroll.setMinimumHeight(min(needed, _TOOL_SCROLL_MIN_HEIGHT))
            tool_scroll.setMaximumHeight(16777215)
            return
        # Beside a pipeline form / ROI list the two share a splitter: the form starts at what it
        # needs (up to a share of the dock) and the handle between them is the user's to drag.
        tool_scroll.setMinimumHeight(min(needed, 80))
        tool_scroll.setMaximumHeight(16777215)
        if _fit_state["user"]:
            return
        total = sum(aux_split.sizes())
        available = total or container.height() or tool_scroll.height()
        ceiling = max(_TOOL_SCROLL_MIN_HEIGHT, int(available * _TOOL_SCROLL_HEIGHT_SHARE))
        height = max(_TOOL_SCROLL_MIN_HEIGHT, min(needed, ceiling))
        top = height + sum(w.sizeHint().height() for w in (cursor_row, cow_row) if w.isVisible())
        if total > top:
            sizes = [top] + [0] * (aux_split.count() - 1)
            rest = [i for i in range(1, aux_split.count()) if aux_split.widget(i).isVisible()]
            for i in rest:
                sizes[i] = (total - top) // len(rest)
            _fit_state["applying"] = True
            aux_split.setSizes(sizes)
            _fit_state["applying"] = False

    # Widget visibility settles during the current event-loop pass, so measure
    # the form on the next one.
    _fit_timer.timeout.connect(_fit_tool_scroll)

    # Wraps rather than widening the dock: three side-by-side buttons set a floor
    # under the whole dock's width, and a narrower dock then scrolled sideways.
    from nvitk.gui.core.flow_layout import FlowRow
    from nvitk.gui.core.performance import build_performance_button

    # The compute backend and CPU threads moved to the activity bar's foot (shell/compute.py);
    # reorienting is in the layer right-click menu (Reorient selected). Without the shell (a
    # bare viewer, a test) the old buttons are kept, without the orientation one.
    if getattr(viewer, "_nvitk_shell", None) is None and not getattr(viewer, "_nvitk_shell_pending", False):
        top_row = FlowRow()
        for button in (build_gpu_toggle_button(), build_performance_button()):
            top_row.add(button)
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
    # The views are in the central viewport now (Layout menu); the button is kept for the
    # tool-dock API but not shown.
    btn_ortho.setVisible(False)
    layout.addWidget(btn_ortho, 0)

    def _select_tool(tool_id: str) -> None:
        """Point the tool form at *tool_id* and run it, from the command palette."""
        from nvitk.gui.tools.registry import tool_by_id
        from nvitk.gui.tools.runner import notify

        spec = tool_by_id(tool_id)
        if spec is None:
            notify(f"Unknown tool {tool_id!r}.", error=True)
            return
        # Drive the existing form rather than bypassing it: the tool's parameters,
        # label picker and reference-layer wiring all hang off these two combos.
        tool_panel.category.value = spec.category
        tool_panel.operation.value = spec.label
        browser = getattr(container, "_nvitk_tool_browser", None)
        if browser is not None and browser.current_tool() != tool_id:
            browser.select_tool(tool_id)  # chosen from the search: show where it lives
        _sync_aux_panels()
        from nvitk.gui.core.workspace import raise_panel

        raise_panel(viewer, "nvitk:tools")  # the Imaging tab, wherever it is docked
        notify(f"{spec.category} → {spec.label} selected. Set any parameters, then Run tool.")

    from nvitk.gui.tools.palette import (
        CommandSearchBar,
        build_commands,
        install_command_palette,
    )

    open_palette = install_command_palette(viewer, _select_tool)

    # The tab gets a real search field; the shortcut still opens the popup, which
    # is what reaches the palette on a desktop whose window manager claims the key.
    search_tools = CommandSearchBar(lambda: build_commands(viewer, _select_tool))
    layout.addWidget(search_tools, 0)
    # The tool browser's own search covers this one; Alt+F still opens the palette everywhere.
    search_tools.setVisible(False)

    # Every tool by task section, filterable by modality / domain / species / suite; picking one
    # drives the form below exactly as the Category / Operation combos do.
    from nvitk.gui.tools.browser import ToolBrowser

    tool_browser = ToolBrowser(panels=lambda: _panel_entries(viewer))
    tool_browser.tool_selected.connect(_select_tool)
    tool_browser.panel_selected.connect(lambda name: _raise_panel(viewer, name))
    container._nvitk_tool_browser = tool_browser
    # The browser is the tool selector: the old Category / Operation dropdowns still exist (they
    # carry the dispatch) but are not shown. Its height is the user's (drag the handle under it).
    from nvitk.gui.core.widgets import resizable

    layout.addWidget(resizable(tool_browser, height=330, minimum=140, pref_key="tools.browser"), 0)
    for hidden in (tool_panel.category, tool_panel.operation):
        hidden.visible = False
    tool_title = QLabel("")
    tool_title.setWordWrap(True)
    tool_title.setTextFormat(Qt.RichText)

    def _sync_tool_title(*_args: Any) -> None:
        """Name the tool the form below belongs to (and where it sits in the browser)."""
        from nvitk.gui.core import design as _design
        from nvitk.gui.tools.registry import tool_by_id
        from nvitk.gui.tools.taxonomy import SECTIONS, meta_for

        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value)
        spec = tool_by_id(tid) if tid else None
        if spec is None:
            tool_title.setText("")
            return
        meta = meta_for(spec)
        section = next((sec.title for sec in SECTIONS if sec.key == meta.section), meta.section)
        where = " › ".join(p for p in (section, meta.subsection) if p)
        suite = "" if meta.suite == "nvitk" else (
            f" &nbsp;<span style='color:{_design.suite_color(meta.suite)}'>{meta.suite}</span>")
        tool_title.setText(
            f"<b style='font-size:{_design.FONT_L}px'>{spec.label}</b>{suite}"
            f"<br><span style='color:{_design.COLOR_MUTED}; font-size:{_design.FONT_S}px'>{where}</span>"
        )

    tool_panel.operation.changed.connect(_sync_tool_title)
    tool_panel.category.changed.connect(_sync_tool_title)
    _sync_tool_title()
    layout.addWidget(tool_title, 0)

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
            or "Export layer, upload over sshfs, and submit Singularity job on the cluster."
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
    # Heavy tools compute on a worker thread (tools/background.py): the switch, and whether the
    # selected tool is one of them.
    from nvitk.gui.tools.background import background_enabled, can_run_in_background, set_background_enabled

    background_box = QCheckBox("Run in the background")
    background_box.setChecked(background_enabled())
    background_box.toggled.connect(set_background_enabled)
    container._nvitk_background_box = background_box

    def _sync_background_box(*_args: Any) -> None:
        tid = tool_id_from_label(tool_panel.category.value, tool_panel.operation.value) or ""
        able = bool(tid) and can_run_in_background(tid)
        background_box.setEnabled(able)
        background_box.setToolTip(
            "Compute on a worker thread: the viewer stays usable (scroll, paint, run other tools) and "
            "the result is added when it is done. The status bar shows it running."
            if able else
            "This tool runs on the GUI thread: it manages its own background job (FSL, FastSurfer), "
            "opens windows or plots, or interacts with the viewer."
        )

    tool_panel.operation.changed.connect(_sync_background_box)
    tool_panel.category.changed.connect(_sync_background_box)
    _sync_background_box()
    run_row = QHBoxLayout()
    run_row.setContentsMargins(0, 0, 0, 0)
    run_row.addWidget(background_box, 1)
    run_row.addWidget(btn_run_sge, 0)
    layout.addLayout(run_row)
    # The tool form and the panel that may come with it (the pipeline form, the TotalSegmentator
    # ROI list) share a splitter, so neither is squeezed by the other: drag the handle between.
    from qtpy.QtWidgets import QSplitter

    top_pane = QWidget()
    top_layout = QVBoxLayout(top_pane)
    top_layout.setContentsMargins(0, 0, 0, 0)
    top_layout.setSpacing(layout.spacing())
    top_layout.addWidget(tool_scroll, 1)
    top_layout.addWidget(cursor_row, 0)
    top_layout.addWidget(cow_row, 0)
    aux_split = QSplitter(Qt.Vertical)
    aux_split.setObjectName("nvitk_tool_form_split")
    aux_split.setChildrenCollapsible(False)
    aux_split.setHandleWidth(6)
    aux_split.addWidget(top_pane)
    aux_split.addWidget(totalseg_roi)
    aux_split.addWidget(pipeline_form)
    aux_split.splitterMoved.connect(
        lambda *_a: _fit_state.__setitem__("user", True) if not _fit_state["applying"] else None)
    container._nvitk_tool_form_split = aux_split
    layout.addWidget(aux_split, 1)
    container.setLayout(layout)

    def _update_aux_panel_layout() -> None:
        """Give the dock's spare height to the panel that can use it: the pipeline
        form or the TotalSeg ROI list when one is showing, else the tool form."""
        is_pipeline = pipeline_form.isVisible()
        is_ts = totalseg_roi.isVisible()
        pipeline_form.set_expanded(is_pipeline)
        totalseg_roi.set_expanded(is_ts and not is_pipeline)
        tool_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        aux = is_pipeline or is_ts
        aux_split.setStretchFactor(0, 0 if aux else 1)
        aux_split.setStretchFactor(1, 1 if is_ts and not is_pipeline else 0)
        aux_split.setStretchFactor(2, 1 if is_pipeline else 0)
        _fit_state["user"] = False  # another tool: its own fit

    _active_sync_timer.timeout.connect(
        lambda: (
            _sync_label_ids_field(_active_layer()),
            pipeline_form.refresh_layer_combos(),
        )
    )

    def _schedule_active_layer_sync() -> None:
        """Debounce a call to resync the id field and pipeline form for the active layer."""
        _active_sync_timer.start()

    @viewer.layers.events.removing.connect
    def _on_layer_removing(_event: Any) -> None:
        """Avoid touching a layer while Napari removes it from the list."""
        _active_sync_timer.stop()

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
        """Debounce a resync of the id field / pipeline form when the active layer changes."""
        _schedule_active_layer_sync()

    _sync_aux_panels()
    return container, tool_panel
