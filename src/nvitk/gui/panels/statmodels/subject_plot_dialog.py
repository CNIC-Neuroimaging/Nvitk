"""
One subject's measurements, drawn on the anatomy they were measured in.

Description
-----------
The analysis dataframe is a cohort: one row per subject × region, and every plot in the main window
is a statement about the population. But the question that follows a suspicious cell is about *one
scan* — is this subject's left MCA really carrying twice its right, or is that one bad segmentation?
A number in a table cannot answer that; a picture of the whole circle of Willis can.

So this window takes a subject and draws their measurements the way the model plots draw a fit:

============================  ===========================================================
4D-flow scalar                the circle-of-Willis schematic (:mod:`nvitk.stats.vascular_map`)
4D-flow ``flow_tseries``      the same schematic, animated over the cardiac cycle
ASL / T1 parcel measurement   the cortical surface (:mod:`nvitk.stats.brain_map`)
============================  ===========================================================

Grouping levels are respected throughout. A frame grouped per hemisphere carries one ``ICA`` value
for both carotids and a lobe-grouped one carries a value for every parcel of that lobe; both are
resolved the same way the model maps resolve them, so the drawing shows what was actually measured
rather than the subset that happens to be spelled like a single structure.

Animation
---------
The cardiac frames share **one colour scale**, computed over the whole cycle. Rescaling per frame
would animate the scale rather than the flow — systole and diastole would look identical, each
saturating its own range, which is the opposite of what a waveform is worth looking at.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.core.geometry import fit_dialog
from nvitk.stats._hemodynamic_frames import frame_columns
from nvitk.stats._statmodels_frames import TIMESERIES_FEATURES

from .figure_host import FigureHostMixin
from .theme import muted_label_style

log = Logger()

#: Frames per second for the cardiac animation. Slow enough to read a single vessel's colour change,
#: fast enough that the cycle reads as one motion rather than a slideshow.
ANIMATION_FPS: float = 6.0

#: Columns that identify a row rather than measure something — never offered as a measurement.
_IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {"subject_uid", "territory", "group_key", "region_id", "visit_id", "session_id",
     "modality", "pipeline_id"}
)


class SubjectPlotDialog(FigureHostMixin, QDialog):
    """
    Anatomical view of one subject's measurements.

    Non-modal and self-owning, like :class:`~.column_plot_dialog.ColumnPlotDialog`: it is something
    you leave open beside the main window while working through a cohort, not a step in a workflow.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        frame: pd.DataFrame,
        subject: str,
        region_column: str = "territory",
        atlas: str = "",
        default_directory: Path | None = None,
    ) -> None:
        """Build the controls and draw the first figure."""
        super().__init__(parent)
        self.setWindowTitle(f"Subject — {subject}")
        fit_dialog(self, 940, 700)
        self.setWindowFlag(Qt.Window, True)
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self._frame = frame
        self._region_column = region_column if region_column in frame.columns else "territory"
        self._atlas = atlas
        self._directory = default_directory
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000.0 / ANIMATION_FPS))
        self._timer.timeout.connect(self._advance_frame)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QLabel("Subject"))
        self._subject = QComboBox()
        self._subject.setMinimumWidth(150)
        for uid in self._subjects():
            self._subject.addItem(uid, uid)
        index = self._subject.findData(subject)
        if index >= 0:
            self._subject.setCurrentIndex(index)
        self._subject.currentIndexChanged.connect(self._on_subject_changed)
        controls.addWidget(self._subject)

        controls.addWidget(QLabel("Measurement"))
        self._measurement = QComboBox()
        self._measurement.setMinimumWidth(190)
        for name in self._measurements():
            self._measurement.addItem(name, name)
        self._measurement.currentIndexChanged.connect(self._on_measurement_changed)
        controls.addWidget(self._measurement, stretch=1)

        self._cmap = QComboBox()
        for label, key in (
            ("auto", ""), ("viridis", "viridis"), ("magma", "magma"),
            ("RdBu", "RdBu_r"), ("coolwarm", "coolwarm"),
        ):
            self._cmap.addItem(label, key)
        self._cmap.currentIndexChanged.connect(lambda *_: self._redraw())
        controls.addWidget(QLabel("cmap"))
        controls.addWidget(self._cmap)

        self._btn_export = QPushButton("Export…")
        self._btn_export.setToolTip(
            "Save the current view — a PNG for a still, an animated GIF for a cardiac cycle."
        )
        self._btn_export.clicked.connect(self._on_export)
        controls.addWidget(self._btn_export)
        lay.addLayout(controls)

        # ---- Cardiac transport, shown only for a time-resolved measurement -------
        self._transport = QWidget()
        transport = QHBoxLayout(self._transport)
        transport.setContentsMargins(0, 0, 0, 0)
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.clicked.connect(self._toggle_play)
        transport.addWidget(self._btn_play)
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.valueChanged.connect(lambda *_: self._redraw())
        transport.addWidget(self._frame_slider, stretch=1)
        self._frame_label = QLabel("—")
        self._frame_label.setMinimumWidth(110)
        self._frame_label.setStyleSheet(muted_label_style())
        transport.addWidget(self._frame_label)
        lay.addWidget(self._transport)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(muted_label_style())
        lay.addWidget(self._status)

        self._build_figure_host(lay)
        self._on_measurement_changed()

    # ---- frame introspection --------------------------------------------------
    def _subjects(self) -> list[str]:
        """Subjects present in the frame, in a stable order."""
        if "subject_uid" not in self._frame.columns:
            return []
        return sorted({str(v) for v in self._frame["subject_uid"].dropna()})

    def _measurements(self) -> list[str]:
        """
        Numeric columns worth drawing, with each timeseries collapsed to a single entry.

        A 16-frame waveform is one measurement, not sixteen: listing ``flow_tseries_f1`` … as
        separate entries would bury every other column and offer the reader a choice between
        cardiac frames that the animation makes for them.
        """
        numeric = [
            str(c) for c in self._frame.columns
            if str(c) not in _IDENTIFIER_COLUMNS
            and pd.api.types.is_numeric_dtype(self._frame[c])
        ]
        hidden: set[str] = set()
        for stem in TIMESERIES_FEATURES:
            columns = frame_columns(self._frame, stem)
            if len(columns) > 1:
                hidden.update(columns[1:])
        return [c for c in numeric if c not in hidden]

    def _timeseries_columns(self, measurement: str) -> list[str]:
        """The cardiac frames of *measurement*, or ``[]`` when it is a plain scalar."""
        stem = str(measurement)
        if stem not in TIMESERIES_FEATURES and not any(
            stem.startswith(f"{t}") for t in TIMESERIES_FEATURES
        ):
            # An aliased timeseries keeps the ``_fN`` convention on whatever stem it was renamed to,
            # so the shape of the columns is what identifies it, not the name.
            columns = frame_columns(self._frame, stem)
            return columns if len(columns) > 1 else []
        columns = frame_columns(self._frame, stem)
        return columns if len(columns) > 1 else []

    def _subject_rows(self) -> pd.DataFrame:
        """The current subject's rows."""
        subject = str(self._subject.currentData() or "")
        if "subject_uid" not in self._frame.columns:
            return self._frame.iloc[0:0]
        return self._frame.loc[self._frame["subject_uid"].astype(str) == subject]

    def _is_vascular(self, rows: pd.DataFrame) -> bool:
        """
        Whether this subject's regions are vessels rather than cortical parcels.

        Decided from the data, not from the measurement's name: the same frame can carry a 4D-flow
        column and an ASL one, and which map to draw depends on what the *region* column holds.
        """
        from nvitk.stats.vascular_map import nodes_for_label

        levels = [str(v) for v in rows[self._region_column].dropna().unique()]
        if not levels:
            return False
        resolved = sum(1 for level in levels if nodes_for_label(level))
        return resolved * 2 >= len(levels)

    # ---- controls -------------------------------------------------------------
    def _on_subject_changed(self) -> None:
        """Retitle and redraw for the newly selected subject."""
        self.setWindowTitle(f"Subject — {self._subject.currentData()}")
        self._redraw()

    def _on_measurement_changed(self) -> None:
        """Show or hide the cardiac transport, size the slider, and redraw."""
        columns = self._timeseries_columns(str(self._measurement.currentData() or ""))
        self._transport.setVisible(bool(columns))
        if columns:
            self._frame_slider.blockSignals(True)
            self._frame_slider.setMaximum(len(columns) - 1)
            self._frame_slider.setValue(0)
            self._frame_slider.blockSignals(False)
        else:
            self._stop()
        self._redraw()

    def _toggle_play(self) -> None:
        """Start or stop the cardiac animation."""
        if self._timer.isActive():
            self._stop()
        else:
            self._timer.start()
            self._btn_play.setText("■ Stop")

    def _stop(self) -> None:
        """Stop the animation and reset the button."""
        self._timer.stop()
        self._btn_play.setText("▶ Play")

    def _advance_frame(self) -> None:
        """Step the slider one cardiac frame, wrapping — a cycle has no end."""
        maximum = self._frame_slider.maximum()
        if maximum <= 0:
            self._stop()
            return
        self._frame_slider.setValue((self._frame_slider.value() + 1) % (maximum + 1))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Stop the timer before the widgets it draws into are destroyed."""
        self._stop()
        super().closeEvent(event)

    # ---- drawing --------------------------------------------------------------
    def _values_for(self, rows: pd.DataFrame, column: str) -> dict[str, float]:
        """``{region label: value}`` for one measurement column of one subject."""
        out: dict[str, float] = {}
        for label, value in zip(rows[self._region_column], rows[column]):
            number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(number):
                out[str(label)] = float(number)
        return out

    def _scale_over_cycle(self, rows: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
        """
        ``(vmin, vmax)`` across the whole cardiac cycle.

        Shared by every frame so the animation shows flow changing, not the scale changing.
        """
        values = pd.to_numeric(rows[columns].to_numpy().ravel(), errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def _redraw(self) -> None:
        """Draw the current subject / measurement / cardiac frame."""
        try:
            self._draw()
        except Exception as exc:
            log.debug("Subject view failed: %s", exc, exc_info=True)
            self._clear_static()
            self._status.setText(f"Cannot draw this measurement: {exc}")

    def _draw(self) -> None:
        """Build and show the figure (see :meth:`_redraw` for the error path)."""
        import matplotlib.pyplot as plt

        measurement = str(self._measurement.currentData() or "")
        rows = self._subject_rows()
        if rows.empty or not measurement:
            self._status.setText("No rows for this subject.")
            return

        series = self._timeseries_columns(measurement)
        if series:
            index = min(self._frame_slider.value(), len(series) - 1)
            column = series[index]
            vmin, vmax = self._scale_over_cycle(rows, series)
            self._frame_label.setText(f"frame {index + 1} / {len(series)}")
        else:
            column, index, vmin, vmax = measurement, 0, None, None
            self._frame_label.setText("—")

        values = self._values_for(rows, column)
        if not values:
            self._status.setText(f"{column!r} has no value for this subject.")
            self._clear_static()
            return

        subject = str(self._subject.currentData() or "")
        cmap = str(self._cmap.currentData() or "") or None
        with plt.style.context("default"):
            if self._is_vascular(rows):
                figure, drawn, note = self._draw_vascular(
                    values, measurement=measurement, subject=subject, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    frame=(index, len(series)) if series else None,
                )
            else:
                figure, drawn, note = self._draw_brain(
                    values, measurement=measurement, subject=subject, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    frame=(index, len(series)) if series else None,
                )
        self._show_static(figure)
        self._status.setText(note + f"  ·  {drawn} region(s) drawn from {len(values)} measured.")

    def _draw_vascular(
        self, values: dict[str, float], *, measurement: str, subject: str,
        cmap: str | None, vmin: float | None, vmax: float | None,
        frame: tuple[int, int] | None,
    ) -> tuple[Any, int, str]:
        """One subject's vessel values on the circle-of-Willis schematic."""
        from nvitk.stats.vascular_map import nodes_for_label, plot_vascular_map

        # ``nodes_for_label`` rather than ``canonical_node``: a hemisphere-melted level such as
        # ``ICA`` is one measured number that belongs on *both* carotids, and a root-grouped
        # ``L_ICA`` on the whole left tree.
        painted: dict[str, float] = {}
        for label, value in values.items():
            for node in nodes_for_label(label):
                painted[node] = value
        if not painted:
            raise ValueError(
                f"None of this subject's {len(values)} region(s) are vessels — "
                f"e.g. {', '.join(list(values)[:4])}."
            )

        title = f"{subject} — {measurement}"
        if frame is not None:
            title += f"  (frame {frame[0] + 1}/{frame[1]})"
        figure = plot_vascular_map(
            painted,
            mask_nonsignificant=False,
            cmap=cmap,
            # A single subject's measurement is a magnitude, not an effect, so it gets a sequential
            # colormap and no zero midpoint unless the values genuinely straddle one.
            center=None,
            vmin=vmin,
            vmax=vmax,
            title=title,
            label=measurement,
        )
        return figure, len(painted), "Colour is this subject's measured value, not a model estimate."

    def _draw_brain(
        self, values: dict[str, float], *, measurement: str, subject: str,
        cmap: str | None, vmin: float | None, vmax: float | None,
        frame: tuple[int, int] | None,
    ) -> tuple[Any, int, str]:
        """One subject's parcel values on the cortical surface."""
        from nvitk.stats.brain_map import parcel_resolver, plot_brain_map

        atlas = self._atlas or "desikan"
        resolve, _ = parcel_resolver(atlas)
        painted: dict[int, float] = {}
        for label, value in values.items():
            for index in resolve(label):
                painted[int(index)] = value
        if not painted:
            raise ValueError(
                f"None of this subject's {len(values)} region(s) are parcels of the {atlas!r} "
                f"atlas — e.g. {', '.join(list(values)[:4])}."
            )

        title = f"{subject} — {measurement}"
        if frame is not None:
            title += f"  (frame {frame[0] + 1}/{frame[1]})"
        figure = plot_brain_map(
            painted,
            mode="estimate",
            mask_nonsignificant=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            atlas=atlas,
            hemisphere="both",
            views=("lateral", "medial"),
            title=title,
            label=measurement,
        )
        return figure, len(painted), f"This subject's measured values on the {atlas} atlas."

    # ---- export ---------------------------------------------------------------
    def _on_export(self) -> None:
        """Save a still as PNG, or a whole cardiac cycle as an animated GIF."""
        measurement = str(self._measurement.currentData() or "")
        subject = str(self._subject.currentData() or "")
        series = self._timeseries_columns(measurement)
        base = self._directory or Path.home()
        suffix = "gif" if series else "png"
        suggested = base / f"{subject}_{measurement}.{suffix}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", str(suggested),
            "Animated GIF (*.gif)" if series else "PNG image (*.png)",
        )
        if not path:
            return
        try:
            if series and str(path).lower().endswith(".gif"):
                self._export_gif(Path(path), series)
            elif self._static_figure is not None:
                self._static_figure.savefig(path, dpi=200, bbox_inches="tight")
            self._status.setText(f"Saved {path}")
        except Exception as exc:
            log.debug("Subject export failed: %s", exc, exc_info=True)
            self._status.setText(f"Could not save: {exc}")

    def _export_gif(self, path: Path, series: list[str]) -> None:
        """
        Write the cardiac cycle as an animated GIF.

        Each frame is rendered independently and appended with Pillow rather than driven through
        ``FuncAnimation``: the map plotters build a whole figure per call (they have no artists to
        update in place), so there is nothing for an animation to mutate.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        self._stop()
        rows = self._subject_rows()
        vmin, vmax = self._scale_over_cycle(rows, series)
        subject = str(self._subject.currentData() or "")
        measurement = str(self._measurement.currentData() or "")
        cmap = str(self._cmap.currentData() or "") or None
        vascular = self._is_vascular(rows)

        images: list[Image.Image] = []
        for index, column in enumerate(series):
            values = self._values_for(rows, column)
            if not values:
                continue
            with plt.style.context("default"):
                draw = self._draw_vascular if vascular else self._draw_brain
                figure, _n, _note = draw(
                    values, measurement=measurement, subject=subject, cmap=cmap,
                    vmin=vmin, vmax=vmax, frame=(index, len(series)),
                )
            canvas = figure.canvas
            canvas.draw()
            images.append(Image.frombytes(
                "RGBA", canvas.get_width_height(), bytes(canvas.buffer_rgba())
            ).convert("RGB"))
            plt.close(figure)

        if not images:
            raise ValueError("No cardiac frame has a value for this subject.")
        images[0].save(
            path, save_all=True, append_images=images[1:],
            duration=int(1000.0 / ANIMATION_FPS), loop=0,
        )
        log.info("Wrote %d-frame animation to %s.", len(images), path)


__all__ = ["ANIMATION_FPS", "SubjectPlotDialog"]
