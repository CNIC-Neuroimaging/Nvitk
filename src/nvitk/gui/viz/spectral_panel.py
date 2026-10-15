"""The Spectral panel: Philips spectral base images (SBI) turned into spectral results, live.

Load an SBI series (from the DICOM browser — SBI series are listed there as *spectral
base* — or with *Open SBI series…* here) and the panel keeps its two basis volumes, the
photoelectric and Compton-scatter parts of the attenuation
(:mod:`nvitk.io.conversors.philips_sbi`). From them it shows, in one live layer per
series:

* a **virtual monoenergetic** image at any energy, following the keV slider;
* **iodine density** (mg/mL), the **virtual non-contrast** image (HU), the **effective
  atomic number**, the **electron density** (% of water's), the **conventional** image
  and the two basis parts themselves.

*Add a copy* keeps the current result as a layer of its own (to compare 40 and 70 keV
side by side); *Energy series* stacks monoenergetic images into one 3D+E layer, which the
time tools scroll and plot. *Curve at the cursor* plots HU against keV for the voxel under
the mouse, *Curve over a label* the mean of a mask's label — the spectral attenuation
curve, computed from the basis pair rather than from stored images.

The results are nvitk's derivation (:mod:`nvitk.transform.spectral`), not Philips'
calibrated ones; the constants that tie the pair to physics are under *Calibration*.
Results are computed on a worker thread (on the GPU when GPU computing is on), so the
slider stays responsive; the layer takes each result as it comes, the last request winning.
"""

from __future__ import annotations

import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy
from nvitk.core.backend import get_global_backend, using
from nvitk.gui.core.design import COLOR_MUTED, SPACE, SPACE_TIGHT, Card

#: Results offered: key → (label, units, colormap, default display range or None for auto).
RESULTS: dict[str, tuple[str, str, str, tuple[float, float] | None]] = {
    "monoenergetic": ("Virtual monoenergetic", "HU", "gray", None),
    "conventional": ("Conventional", "HU", "gray", None),
    "iodine": ("Iodine density", "mg/mL", "magma", (0.0, 15.0)),
    "vnc": ("Virtual non-contrast", "HU", "gray", None),
    "zeff": ("Effective Z", "", "turbo", (5.0, 15.0)),
    "electron_density": ("Electron density", "% water", "gray", (80.0, 140.0)),
    "photo": ("Photoelectric part", "HU", "gray", None),
    "scatter": ("Compton-scatter part", "HU", "gray", None),
}
#: The CT window the HU results open with when the series carries none (level, width).
_DEFAULT_WINDOW = (90.0, 750.0)


def _notify(message: str, *, error: bool = False) -> None:
    try:
        from nvitk.gui.tools.runner import notify

        notify(message, error=error)
    except Exception:  # noqa: BLE001
        print(message, flush=True)


class _Relay(QObject):
    """Hands a worker thread's result to the GUI thread (signals queue across threads)."""

    finished = Signal(object, object, object)   # (dataset, request, result or exception)


class SpectralDataset:
    """One loaded SBI series and the layer that shows its current result."""

    def __init__(self, sbi: Any, source: str = "") -> None:
        self.sbi = sbi
        self.source = source
        self.layer_ref: Any = lambda: None
        #: The display range the user gave each result, kept across keV changes.
        self.ranges: dict[str, tuple[float, float]] = {}
        meta = sbi.metadata
        self.label = str(meta.get("sbi_label") or "SBI")
        try:
            level = float(meta.get("WindowCenter") or _DEFAULT_WINDOW[0])
            width = float(meta.get("WindowWidth") or _DEFAULT_WINDOW[1])
        except (TypeError, ValueError):
            level, width = _DEFAULT_WINDOW
        self.window = (level - width / 2.0, level + width / 2.0)

    @property
    def layer(self) -> Any:
        return self.layer_ref()

    def summary(self) -> str:
        m = self.sbi.metadata
        shape = " × ".join(str(n) for n in self.sbi.shape)
        bits = [f"Philips SBI V{m.get('sbi_version', '?')}", m.get("sbi_recon", ""),
                f"{m.get('KVP')} kV" if m.get("KVP") else "", f"kernel {m.get('ConvolutionKernel')}"
                if m.get("ConvolutionKernel") else "", shape]
        return " · ".join(b for b in bits if b)


class SpectralPanel(QWidget):
    """Spectral results of the loaded SBI series."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from nvitk.transform.spectral import PHILIPS_SBI_V51

        self._viewer = viewer
        self._datasets: list[SpectralDataset] = []
        self._calibration = PHILIPS_SBI_V51
        self._relay = _Relay(self)
        self._relay.finished.connect(self._on_finished)
        self._busy: SpectralDataset | None = None
        self._pending: dict[int, dict[str, Any]] = {}
        self._latest: dict[int, int] = {}
        self._serial = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._request_current)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT)
        root.setSpacing(SPACE)
        root.setAlignment(Qt.AlignTop)
        root.addWidget(self._build_data_card())
        root.addWidget(self._build_result_card())
        root.addWidget(self._build_curve_card())
        root.addWidget(self._build_calibration_card())
        root.addStretch(1)
        self._sync_enabled()

    # ── cards ────────────────────────────────────────────────────────────────

    def _build_data_card(self) -> Card:
        card = Card("Spectral data (Philips SBI)")
        self._dataset_box = QComboBox()
        self._dataset_box.currentIndexChanged.connect(lambda _i: self._on_dataset())
        self._open_btn = QPushButton("Open SBI series…")
        self._open_btn.setToolTip("Pick a DICOM folder: its Philips spectral base image (SBI) series are listed "
                                  "to choose from. The DICOM browser loads them too (they show as "
                                  "'spectral base').")
        self._open_btn.clicked.connect(self._open_dialog)
        row = QHBoxLayout()
        row.addWidget(self._dataset_box, 1)
        row.addWidget(self._open_btn)
        card.add_layout(row)
        self._info = QLabel("No SBI series loaded.")
        self._info.setWordWrap(True)
        self._info.setStyleSheet(f"color: {COLOR_MUTED};")
        card.add(self._info)
        return card

    def _build_result_card(self) -> Card:
        card = Card("Result")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self._result_box = QComboBox()
        for key, (label, units, _cmap, _rng) in RESULTS.items():
            self._result_box.addItem(f"{label} ({units})" if units else label, key)
        self._result_box.currentIndexChanged.connect(lambda _i: self._on_result_changed())
        form.addRow("Show", self._result_box)
        kev_row = QHBoxLayout()
        self._kev_slider = QSlider(Qt.Horizontal)
        self._kev_slider.setRange(40, 200)
        self._kev_slider.setValue(70)
        self._kev_spin = QSpinBox()
        self._kev_spin.setRange(40, 200)
        self._kev_spin.setValue(70)
        self._kev_spin.setSuffix(" keV")
        self._kev_slider.valueChanged.connect(self._on_kev_slider)
        self._kev_spin.valueChanged.connect(self._on_kev_spin)
        kev_row.addWidget(self._kev_slider, 1)
        kev_row.addWidget(self._kev_spin)
        self._kev_host = QWidget()
        self._kev_host.setLayout(kev_row)
        kev_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Energy", self._kev_host)
        card.add_layout(form)
        buttons = QHBoxLayout()
        self._copy_btn = QPushButton("Add a copy")
        self._copy_btn.setToolTip("Keep the current result as a layer of its own (the live layer moves on).")
        self._copy_btn.clicked.connect(self.add_copy)
        self._series_btn = QPushButton("Energy series…")
        self._series_btn.setToolTip("Monoenergetic images over a range of energies, stacked as one 3D+E layer "
                                    "(scroll the energy; the time tools plot HU against keV).")
        self._series_btn.clicked.connect(self._energy_series_dialog)
        buttons.addWidget(self._copy_btn)
        buttons.addWidget(self._series_btn)
        card.add_layout(buttons)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")
        card.add(self._status)
        note = QLabel("nvitk's derivation from the spectral base images (photoelectric / Compton model), "
                      "not Philips' calibrated results — expect the same picture, not the same numbers to the HU.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        card.add(note)
        return card

    def _build_curve_card(self) -> Card:
        card = Card("Spectral curve (HU vs keV)")
        row = QHBoxLayout()
        self._cursor_btn = QPushButton("Curve at the cursor")
        self._cursor_btn.setToolTip("The voxel under the mouse in the viewer (hover it, then click here); "
                                    "else the orthogonal views' crosshair when they show this series.")
        self._cursor_btn.clicked.connect(self.curve_at_cursor)
        row.addWidget(self._cursor_btn)
        card.add_layout(row)
        roi = QHBoxLayout()
        self._roi_layer = QComboBox()
        self._roi_layer.setToolTip("A label layer on this series' grid")
        self._roi_id = QSpinBox()
        self._roi_id.setRange(0, 1_000_000)
        self._roi_id.setValue(1)
        self._roi_id.setToolTip("Label id (0: every labelled voxel)")
        self._roi_btn = QPushButton("Curve over a label")
        self._roi_btn.clicked.connect(self.curve_over_label)
        roi.addWidget(self._roi_layer, 1)
        roi.addWidget(self._roi_id)
        roi.addWidget(self._roi_btn)
        card.add_layout(roi)
        self._viewer.layers.events.inserted.connect(lambda _e=None: self._refresh_roi_layers())
        self._viewer.layers.events.removed.connect(lambda _e=None: self._refresh_roi_layers())
        return card

    def _build_calibration_card(self) -> Card:
        from nvitk.transform.spectral import PHILIPS_SBI_V51

        card = Card("Calibration")
        card.setToolTip("The constants that tie the vendor's photoelectric / scatter pair to physics "
                        "(nvitk.transform.spectral). The defaults were measured for Philips SBI V5.1, 120 kVp.")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        def spin(lo: float, hi: float, value: float, step: float, decimals: int, tip: str) -> QDoubleSpinBox:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setToolTip(tip)
            box.editingFinished.connect(self._on_calibration)
            return box

        c = PHILIPS_SBI_V51
        self._cal_ref = spin(40.0, 120.0, c.reference_kev, 1.0, 1,
                             "The energy the basis pair refers to — the conventional image's equivalent.")
        self._cal_water = spin(0.01, 0.5, c.water_photo_share, 0.005, 3,
                               "Share of water's attenuation in the photoelectric image.")
        self._cal_iodine = spin(0.3, 1.2, c.iodine_photo_share, 0.005, 3,
                                "Share of iodine's attenuation in the photoelectric image.")
        self._cal_zexp = spin(2.0, 4.5, c.zeff_exponent, 0.01, 2, "Power law of the effective Z (Mayneord: 2.94).")
        form.addRow("Reference energy (keV)", self._cal_ref)
        form.addRow("Water photoelectric share", self._cal_water)
        form.addRow("Iodine photoelectric share", self._cal_iodine)
        form.addRow("Z_eff exponent", self._cal_zexp)
        card.add_layout(form)
        reset = QPushButton("Philips SBI V5.1 defaults")
        reset.clicked.connect(self._reset_calibration)
        card.add(reset)
        return card

    # ── state ────────────────────────────────────────────────────────────────

    def datasets(self) -> list[SpectralDataset]:
        return list(self._datasets)

    def current(self) -> SpectralDataset | None:
        k = self._dataset_box.currentIndex()
        return self._datasets[k] if 0 <= k < len(self._datasets) else None

    def result_key(self) -> str:
        return str(self._result_box.currentData() or "monoenergetic")

    def kev(self) -> float:
        return float(self._kev_spin.value())

    def calibration(self) -> Any:
        return self._calibration

    def _sync_enabled(self) -> None:
        ds = self.current()
        for w in (self._result_box, self._kev_host, self._copy_btn, self._series_btn, self._cursor_btn,
                  self._roi_btn, self._roi_layer, self._roi_id):
            w.setEnabled(ds is not None)
        self._kev_host.setEnabled(ds is not None and self.result_key() == "monoenergetic")
        self._info.setText(ds.summary() if ds is not None else "No SBI series loaded.")

    def _on_dataset(self) -> None:
        self._sync_enabled()
        self._refresh_roi_layers()

    def _on_result_changed(self) -> None:
        self._sync_enabled()
        self._schedule(now=True)

    def _on_kev_slider(self, value: int) -> None:
        if self._kev_spin.value() != value:
            self._kev_spin.blockSignals(True)
            self._kev_spin.setValue(value)
            self._kev_spin.blockSignals(False)
        self._schedule()

    def _on_kev_spin(self, value: int) -> None:
        if self._kev_slider.value() != value:
            self._kev_slider.blockSignals(True)
            self._kev_slider.setValue(value)
            self._kev_slider.blockSignals(False)
        self._schedule()

    def set_result(self, key: str, kev: float | None = None) -> None:
        """Show *key* (and *kev* for the monoenergetic image) — for scripts and tests."""
        if kev is not None:
            self._kev_spin.setValue(int(round(kev)))
        index = self._result_box.findData(key)
        if index >= 0:
            self._result_box.setCurrentIndex(index)
        self._schedule(now=True)

    def _on_calibration(self) -> None:
        from nvitk.transform.spectral import SpectralCalibration

        self._calibration = SpectralCalibration(
            reference_kev=float(self._cal_ref.value()),
            water_photo_share=float(self._cal_water.value()),
            iodine_photo_share=float(self._cal_iodine.value()),
            zeff_exponent=float(self._cal_zexp.value()),
        )
        self._schedule(now=True)

    def _reset_calibration(self) -> None:
        from nvitk.transform.spectral import PHILIPS_SBI_V51

        for box, value in ((self._cal_ref, PHILIPS_SBI_V51.reference_kev),
                           (self._cal_water, PHILIPS_SBI_V51.water_photo_share),
                           (self._cal_iodine, PHILIPS_SBI_V51.iodine_photo_share),
                           (self._cal_zexp, PHILIPS_SBI_V51.zeff_exponent)):
            box.setValue(value)
        self._on_calibration()

    # ── loading ──────────────────────────────────────────────────────────────

    def load_sbi(self, paths: list[str], *, source: str = "") -> SpectralDataset | None:
        """Read an SBI series (its files) and show it, at 70 keV to start with."""
        from nvitk.gui.core.activity import activity, run_in_background
        from nvitk.io.conversors.philips_sbi import read_sbi_series

        with activity("Reading spectral base images", total=len(paths)) as task:
            sbi = run_in_background(read_sbi_series, list(paths),
                                    progress=lambda done, total: task.update(done=done, total=total))
        ds = SpectralDataset(sbi, source=source)
        self._datasets.append(ds)
        self._dataset_box.blockSignals(True)
        self._dataset_box.addItem(f"{ds.label} ({sbi.shape[-1]} slices)")
        self._dataset_box.setCurrentIndex(len(self._datasets) - 1)
        self._dataset_box.blockSignals(False)
        self._sync_enabled()
        self._refresh_roi_layers()
        self._show_now(ds)
        _notify(f"{ds.label}: spectral base images read ({ds.summary()}).")
        return ds

    def _open_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder with the SBI series")
        if folder:
            self.open_folder(folder)

    def open_folder(self, folder: str, *, choose: Any = None) -> SpectralDataset | None:
        """Find the SBI series under *folder* and load one (asks which when there are several;
        *choose* picks one from their labels instead)."""
        from nvitk.gui.core.activity import activity, run_in_background
        from nvitk.io.dicom_index import scan_dicom

        with activity("Looking for spectral base images"):
            studies = run_in_background(scan_dicom, folder)
        series = [s for st in studies for s in st.series if s.kind == "spectral base"]
        if not series:
            self._status.setText(f"No Philips spectral base images (SBI) under {folder}.")
            return None
        labels = [f"{s.label} ({len(s.files)} files)" for s in series]
        if len(series) == 1:
            pick = 0
        elif choose is not None:
            pick = int(choose(labels))
        else:
            text, ok = QInputDialog.getItem(self, "SBI series", "Spectral base series to load:", labels, 0, False)
            if not ok:
                return None
            pick = labels.index(text)
        return self.load_sbi(list(series[pick].paths), source=folder)

    # ── computing ────────────────────────────────────────────────────────────

    def _request(self, ds: SpectralDataset) -> dict[str, Any]:
        self._serial += 1
        self._latest[id(ds)] = self._serial
        return {"serial": self._serial, "result": self.result_key(), "kev": self.kev(),
                "calibration": self._calibration}

    def _schedule(self, *, now: bool = False) -> None:
        if self.current() is None:
            return
        if now:
            self._timer.stop()
            self._request_current()
        else:
            self._timer.start()

    def _request_current(self) -> None:
        ds = self.current()
        if ds is None:
            return
        request = self._request(ds)
        if self._busy is not None:
            self._pending[id(ds)] = request      # the newest replaces any waiting one
            return
        self._start(ds, request)

    @staticmethod
    def _compute(ds: SpectralDataset, request: dict[str, Any]) -> np.ndarray:
        """The result volume (host, float32) — runs on a worker thread."""
        from nvitk.transform.spectral import spectral_result

        sbi = ds.sbi
        with using(get_global_backend()):
            out = spectral_result(request["result"], sbi.photo, sbi.scatter, kev=request["kev"],
                                  conventional=sbi.components.get("conventional"),
                                  calibration=request["calibration"])
            return to_numpy(out)

    def _start(self, ds: SpectralDataset, request: dict[str, Any]) -> None:
        self._busy = ds
        self._status.setText(f"Computing {self._describe(request)}…")

        def work() -> None:
            try:
                result: Any = self._compute(ds, request)
            except Exception as exc:  # noqa: BLE001 — handed to the GUI thread
                result = exc
            self._relay.finished.emit(ds, request, result)

        threading.Thread(target=work, name="nvitk-spectral", daemon=True).start()

    def _on_finished(self, ds: SpectralDataset, request: dict[str, Any], result: Any) -> None:
        self._busy = None
        pending = self._pending.pop(id(ds), None)
        if isinstance(result, Exception):
            self._status.setText(f"Could not compute {self._describe(request)}: {result}")
        elif self._latest.get(id(ds)) == request["serial"] or pending is None:
            self._apply(ds, request, result)
        if pending is not None:
            self._start(ds, pending)
        else:
            # A request for another series may be waiting.
            for other in self._datasets:
                waiting = self._pending.pop(id(other), None)
                if waiting is not None:
                    self._start(other, waiting)
                    break

    def _show_now(self, ds: SpectralDataset) -> None:
        """Compute and show the current result at once (on this thread)."""
        request = self._request(ds)
        from nvitk.gui.core.activity import activity, run_in_background

        with activity(f"Spectral: {self._describe(request)}"):
            result = run_in_background(self._compute, ds, request)
        self._apply(ds, request, result)

    def wait(self, timeout: float = 120.0) -> None:
        """Let the queued computations finish (for scripts and tests)."""
        import time

        from qtpy.QtWidgets import QApplication

        deadline = time.monotonic() + timeout
        while (self._busy is not None or self._pending or self._timer.isActive()) and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.01)
        QApplication.processEvents()

    @staticmethod
    def _describe(request: dict[str, Any]) -> str:
        label = RESULTS[request["result"]][0]
        return f"{label} at {request['kev']:g} keV" if request["result"] == "monoenergetic" else label

    # ── layers ───────────────────────────────────────────────────────────────

    def _layer_name(self, ds: SpectralDataset, request: dict[str, Any]) -> str:
        return f"{ds.label} · {self._describe(request)}"

    def _display_range(self, ds: SpectralDataset, key: str, data: np.ndarray) -> tuple[float, float]:
        if key in ds.ranges:
            return ds.ranges[key]
        preset = RESULTS[key][3]
        if preset is not None:
            return preset
        if RESULTS[key][1] == "HU" and key not in ("photo", "scatter"):
            return ds.window
        flat = data.reshape(-1)[:: max(1, data.size // 400_000)]
        lo, hi = (float(v) for v in np.percentile(flat, (1.0, 99.5)))
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def _apply(self, ds: SpectralDataset, request: dict[str, Any], result: np.ndarray) -> None:
        """Put *result* into the series' live layer (made on first use)."""
        key = request["result"]
        layer = ds.layer
        previous_key = getattr(layer, "_nvitk_spectral_key", None) if layer is not None else None
        if layer is not None and layer not in self._viewer.layers:
            layer = None
        if layer is not None and previous_key is not None:
            # Remember the range the user set for what was shown.
            try:
                ds.ranges[previous_key] = tuple(float(v) for v in layer.contrast_limits)  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass
        name = self._layer_name(ds, request)
        if layer is None:
            layer = self._add_layer(ds, result, name)
            ds.layer_ref = weakref.ref(layer)
        else:
            layer.data = self._display_data(ds, result)
            layer.name = name
        layer._nvitk_spectral_key = key
        cmap = RESULTS[key][2]
        try:
            layer.colormap = cmap
            lo, hi = self._display_range(ds, key, result)
            span = max(hi - lo, 1e-6)
            layer.contrast_limits_range = (min(lo, float(np.nanmin(result))) - span,
                                           max(hi, float(np.nanmax(result))) + span)
            layer.contrast_limits = (lo, hi)
        except Exception:  # noqa: BLE001 — display defaults are a convenience
            pass
        units = RESULTS[key][1]
        self._status.setText(f"{name}" + (f" ({units})" if units else "") + ".")

    def _image(self, ds: SpectralDataset, data: np.ndarray, name: str) -> Any:
        return ds.sbi.image("photo", data=data, image_name=name)

    def _display_data(self, ds: SpectralDataset, data: np.ndarray) -> Any:
        """*data* laid out as the series' layers are (the reader's ``X x Y x Z``, through the
        same preparation a loaded volume goes through)."""
        from nvitk.gui.io.napari_io import _prepare_layer_tuple

        prepared, _meta, _kind = _prepare_layer_tuple(self._image(ds, data, "spectral"), Path(ds.source or "sbi"))
        return prepared

    def _add_layer(self, ds: SpectralDataset, data: np.ndarray, name: str) -> Any:
        from nvitk.gui.viz.time_frequency import add_image_layer

        layer = add_image_layer(self._viewer, self._image(ds, data, name), name=name, source=ds.source)
        layer.metadata = {**dict(layer.metadata or {}), "spectral_result": True}
        return layer

    def add_copy(self) -> Any:
        """Keep the current result as a layer of its own."""
        ds = self.current()
        if ds is None:
            return None
        request = self._request(ds)
        from nvitk.gui.core.activity import activity, run_in_background

        with activity(f"Spectral: {self._describe(request)}"):
            result = run_in_background(self._compute, ds, request)
        key = request["result"]
        layer = self._add_layer(ds, result, self._layer_name(ds, request))
        try:
            layer.colormap = RESULTS[key][2]
            layer.contrast_limits = self._display_range(ds, key, result)
        except Exception:  # noqa: BLE001
            pass
        return layer

    def _energy_series_dialog(self) -> None:
        ds = self.current()
        if ds is None:
            return
        text, ok = QInputDialog.getText(
            self, "Energy series", "Energies (keV): first, last, step", text="40, 200, 20")
        if not ok:
            return
        try:
            first, last, step = (float(v) for v in text.replace(";", ",").split(","))
        except ValueError:
            self._status.setText("Give three numbers: first, last, step (keV).")
            return
        self.add_energy_series(first, last, step)

    def add_energy_series(self, first: float = 40.0, last: float = 200.0, step: float = 20.0) -> Any:
        """Monoenergetic images from *first* to *last* keV every *step*, as one 3D+E layer
        (HU, int16), ordered by energy."""
        from nvitk.gui.core.activity import activity, run_in_background
        from nvitk.gui.viz.time_frequency import add_image_layer
        from nvitk.transform.spectral import monoenergetic
        from nvitk.types import Image

        ds = self.current()
        if ds is None:
            return None
        kevs = [float(k) for k in np.arange(first, last + 1e-6, step) if 40.0 <= k <= 200.0]
        if not kevs:
            self._status.setText("No energy between 40 and 200 keV in that range.")
            return None
        sbi, calibration = ds.sbi, self._calibration

        def build(task: Any) -> np.ndarray:
            stack = np.empty((*sbi.shape, len(kevs)), dtype=np.int16)
            with using(get_global_backend()):
                for k, kev in enumerate(kevs):
                    vol = to_numpy(monoenergetic(sbi.photo, sbi.scatter, kev, calibration))
                    stack[..., k] = np.clip(np.rint(vol), -32768, 32767).astype(np.int16)
                    task.update(done=k + 1)
            return stack

        with activity(f"Spectral: energy series {kevs[0]:g}–{kevs[-1]:g} keV", total=len(kevs)) as task:
            stack = run_in_background(build, task)
        meta = dict(sbi.metadata)
        meta.update(affine=sbi.affine.copy(), axes="XYZT", spectral_energies_kev=kevs, spectral_stack=True,
                    spacing=sbi.spacing, shape=tuple(stack.shape))
        meta["x_res"], meta["y_res"], meta["z_res"] = sbi.spacing
        name = f"{ds.label} · monoenergetic {kevs[0]:g}–{kevs[-1]:g} keV"
        layer = add_image_layer(self._viewer, Image(data=stack, metadata=meta, axes="XYZT", name=name),
                                name=name, source=ds.source)
        try:
            layer.contrast_limits = ds.window
        except Exception:  # noqa: BLE001
            pass
        self._status.setText(f"{name}: {len(kevs)} energies, {stack.nbytes / 1e9:.1f} GB.")
        return layer

    # ── curves ───────────────────────────────────────────────────────────────

    def _voxel_at_cursor(self, ds: SpectralDataset) -> tuple[int, int, int] | None:
        """The series' voxel under the mouse; else the ortho views' crosshair when they
        show this series."""
        layer = ds.layer
        if layer is None or layer not in self._viewer.layers:
            return None
        shape = np.asarray(layer.data.shape[-3:])
        try:
            index = np.rint(np.asarray(layer.world_to_data(self._viewer.cursor.position), dtype=float)[-3:])
            if not (np.any(index < 0) or np.any(index >= shape)):
                return tuple(int(v) for v in index)  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            pass
        ortho = getattr(self._viewer, "_nvitk_ortho_panel", None)
        if ortho is not None and getattr(ortho, "_layer", None) is layer and hasattr(ortho, "position"):
            try:
                return tuple(int(v) for v in ortho.position())  # type: ignore[return-value]
            except Exception:  # noqa: BLE001
                pass
        return None

    def _plot(self, ds: SpectralDataset, photo: float, scatter: float, label: str) -> list[float]:
        from nvitk.gui.viz.time_frequency import show_time_curve_window
        from nvitk.transform.spectral import spectral_curve

        kevs = [float(k) for k in range(40, 201, 5)]
        values = spectral_curve(photo, scatter, kevs, self._calibration)
        show_time_curve_window(self._viewer, kevs, values, label=label, unit="keV", y_unit="HU")
        return values

    def curve_at_cursor(self) -> list[float] | None:
        """HU against keV at the voxel under the mouse."""
        ds = self.current()
        if ds is None:
            return None
        voxel = self._voxel_at_cursor(ds)
        if voxel is None:
            self._status.setText("Hover the series' layer in the viewer first (the curve is taken under the mouse).")
            return None
        layer_data = ds.layer.data if ds.layer is not None else None
        same_layout = layer_data is not None and tuple(layer_data.shape) == tuple(ds.sbi.shape)
        if not same_layout:
            self._status.setText("The live layer is laid out differently from the series; use a label instead.")
            return None
        p = float(ds.sbi.photo[voxel])
        s = float(ds.sbi.scatter[voxel])
        values = self._plot(ds, p, s, f"{ds.label} · voxel {voxel}")
        self._status.setText(f"Voxel {voxel}: {values[0]:.0f} HU at 40 keV, {values[6]:.0f} at 70, "
                             f"{values[12]:.0f} at 100.")
        return values

    def _refresh_roi_layers(self) -> None:
        ds = self.current()
        current = self._roi_layer.currentText()
        self._roi_layer.blockSignals(True)
        self._roi_layer.clear()
        if ds is not None:
            from nvitk.gui.labels.visibility import is_label_like_layer

            for layer in self._viewer.layers:
                if getattr(layer, "data", None) is None or tuple(np.shape(layer.data)) != tuple(ds.sbi.shape):
                    continue
                if type(layer).__name__ == "Labels" or is_label_like_layer(layer):
                    self._roi_layer.addItem(layer.name)
        if current:
            index = self._roi_layer.findText(current)
            if index >= 0:
                self._roi_layer.setCurrentIndex(index)
        self._roi_layer.blockSignals(False)

    def curve_over_label(self) -> list[float] | None:
        """HU against keV for the mean over a label of a mask on this series' grid."""
        ds = self.current()
        name = self._roi_layer.currentText()
        if ds is None or not name:
            self._status.setText("Pick a label layer on this series' grid.")
            return None
        layer = next((ly for ly in self._viewer.layers if ly.name == name), None)
        if layer is None:
            return None
        from nvitk.gui.labels.visibility import label_source_data

        labels = to_numpy(label_source_data(layer))
        lid = int(self._roi_id.value())
        mask = labels > 0 if lid == 0 else labels == lid
        n = int(mask.sum())
        if not n:
            self._status.setText(f"Label {lid} is empty on “{name}”.")
            return None
        p = float(ds.sbi.photo[mask].astype(np.float64).mean())
        s = float(ds.sbi.scatter[mask].astype(np.float64).mean())
        what = "all labels" if lid == 0 else f"label {lid}"
        values = self._plot(ds, p, s, f"{name} · {what}")
        self._status.setText(f"{name}, {what} ({n:,} voxels): {values[0]:.0f} HU at 40 keV, "
                             f"{values[6]:.0f} at 70, {values[12]:.0f} at 100.")
        return values


def build_spectral_panel(viewer: Any) -> SpectralPanel:
    panel = SpectralPanel(viewer)
    viewer._nvitk_spectral_panel = panel
    return panel


__all__ = ["RESULTS", "SpectralDataset", "SpectralPanel", "build_spectral_panel"]
