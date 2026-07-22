"""Statmodels explorer: interactive MixedLM formula builder over the NVITK DB."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.gui.tools.runner import notify
from nvitk.pipes.qvtpy.common.db_publish import QVTPY_PIPELINE_ID
from nvitk.stats import (
    aggregate_territory_measurements,
    build_analysis_df_from_repo_frames,
    fit_or_load_mixedlm,
    melt_imaging_territories,
    plot_mixedlm_params,
    print_mixedlm_info,
)

log = Logger()

PIPELINE_QVTPY = "qvtpy"

_DEFAULT_FORMULA = (
    "flow ~ C(tacsctot_group, Treatment('None')) "
    "* C(territory, Treatment('MCA')) + age_c + sex + Hematocrit"
)
_DEFAULT_GROUPS = "territory"
_DEFAULT_RE = "0"
_DEFAULT_VC = '{"patient": "0 + C(subject_uid)"}'


def _repo() -> DataRepo:
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def _statmodels_root(repo: DataRepo) -> Path:
    root = Path(repo.root) / "nvitk-statmodels"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_vc_formula(text: str) -> dict[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"vc_formula must be a Python dict literal: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("vc_formula must be a dict, e.g. {\"patient\": \"0 + C(subject_uid)\"}")
    return {str(k): str(v) for k, v in value.items()}


def _load_qvtpy_analysis_frame(
    repo: DataRepo,
    *,
    feature: str,
    atlas: str,
    clinical_vars: list[str],
) -> pd.DataFrame:
    """Build a long analysis frame for qvtpy image measurements + clinical covariates."""
    feature = str(feature).strip() or "flow_mean"
    # Map friendly names to DB variable ids.
    feature_aliases = {
        "flow": "flow_mean",
        "pwv": "pwv",
        "pi": "pi",
        "ri": "ri",
        "pitc": "pitc_slope",
    }
    variable_id = feature_aliases.get(feature, feature)

    image = repo.image(
        modality="4dflow",
        pipeline="latest",
        variables=[variable_id],
        wide=True,
    )
    if image.empty:
        raise ValueError(
            f"No 4dflow/qvtpy image measurements for variable={variable_id!r}."
        )

    melt_kwargs: dict[str, Any] = {"flow_vars": [variable_id], "asl_vars": []}
    atlas_key = str(atlas or "flow").strip().lower()
    if atlas_key in {"vascular-8", "desikan"}:
        # ASL atlases: still melt flow vars but keep default territory map.
        pass
    territory_df = melt_imaging_territories(image, **melt_kwargs)
    if territory_df.empty:
        raise ValueError("melt_imaging_territories returned no rows.")

    clinical = pd.DataFrame()
    if clinical_vars:
        try:
            clinical = repo.clinical(variables=clinical_vars, wide=True)
        except Exception as exc:
            log.warning("Could not load clinical covariates: %s", exc)
            # Try without variable filter.
            try:
                clinical = repo.clinical(wide=True)
            except Exception:
                clinical = pd.DataFrame()

    if clinical is None or clinical.empty:
        # Allow fitting without covariates if formula does not need them.
        analysis = aggregate_territory_measurements(
            territory_df, [variable_id], agg="mean"
        )
    else:
        # Only request covariates that exist.
        present = [c for c in clinical_vars if c in clinical.columns]
        analysis = build_analysis_df_from_repo_frames(
            territory_df,
            clinical,
            imaging_variable_ids=[variable_id],
            covariate_cols=present,
        )

    rename = {}
    if "territory_base" in analysis.columns and "territory" not in analysis.columns:
        rename["territory_base"] = "territory"
    if "Hematocrit" not in analysis.columns and "hematocrit" in analysis.columns:
        rename["hematocrit"] = "Hematocrit"
    if rename:
        analysis = analysis.rename(columns=rename)
    if "patient_id" not in analysis.columns and "subject_uid" in analysis.columns:
        analysis["patient_id"] = analysis["subject_uid"]
    if "age" in analysis.columns and "age_c" not in analysis.columns:
        age = pd.to_numeric(analysis["age"], errors="coerce")
        analysis["age_c"] = age - age.mean(skipna=True)
    elif "age_at_mri" in analysis.columns and "age_c" not in analysis.columns:
        age = pd.to_numeric(analysis["age_at_mri"], errors="coerce")
        analysis["age_c"] = age - age.mean(skipna=True)
    # Convenience alias so formulas can use `flow` as the LHS.
    if variable_id in analysis.columns and "flow" not in analysis.columns:
        if feature == "flow" or variable_id == "flow_mean":
            analysis["flow"] = analysis[variable_id]
    return analysis


class StatmodelsWindow(QMainWindow):
    """Floating / maximizable MixedLM explorer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("nvitk Statmodels")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1280, 800)
        self._repo = _repo()
        self._last_result = None
        self._last_df: pd.DataFrame | None = None
        self._last_meta: dict[str, Any] | None = None
        self._plot_canvas = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        form = QFormLayout()
        self._feature = QLineEdit("flow")
        self._atlas = QComboBox()
        self._atlas.addItem("flow vessels / territories", "flow")
        self._atlas.addItem("ASL vascular-8", "vascular-8")
        self._atlas.addItem("ASL Desikan", "desikan")
        self._formula = QPlainTextEdit(_DEFAULT_FORMULA)
        self._formula.setFixedHeight(80)
        self._groups = QLineEdit(_DEFAULT_GROUPS)
        self._re_formula = QLineEdit(_DEFAULT_RE)
        self._vc_formula = QLineEdit(_DEFAULT_VC)
        self._clinical = QLineEdit("age,sex,Hematocrit,tacsctot_group")
        self._model_name = QLineEdit("qvtpy_flow_cacs_territory")

        form.addRow("Feature (image variable)", self._feature)
        form.addRow("Territory / atlas", self._atlas)
        form.addRow("Clinical covariates", self._clinical)
        form.addRow("mm_formula", self._formula)
        form.addRow("groups", self._groups)
        form.addRow("re_formula", self._re_formula)
        form.addRow("vc_formula", self._vc_formula)
        form.addRow("Model name (save)", self._model_name)
        root.addLayout(form)

        btn_row = QHBoxLayout()
        self._btn_fit = QPushButton("Fit model")
        self._btn_save = QPushButton("Save model")
        self._btn_load = QPushButton("Load model…")
        self._btn_plot = QPushButton("Refresh plot")
        btn_row.addWidget(self._btn_fit)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_plot)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        splitter = QSplitter(Qt.Horizontal)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText("MixedLM summary will appear here.")
        splitter.addWidget(self._report)

        self._plot_host = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_hint = QLabel("Parameter / EMM plots appear here after fitting.")
        self._plot_hint.setWordWrap(True)
        self._plot_layout.addWidget(self._plot_hint)
        splitter.addWidget(self._plot_host)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        self._status = QLabel(
            f"Dataset: {self._repo.root}  |  pipeline default: {QVTPY_PIPELINE_ID}"
        )
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._btn_fit.clicked.connect(self._on_fit)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load.clicked.connect(self._on_load)
        self._btn_plot.clicked.connect(self._on_plot)

    def show_maximized_floating(self) -> None:
        self.show()
        self.showMaximized()
        self.raise_()
        self.activateWindow()

    def _clinical_list(self) -> list[str]:
        return [c.strip() for c in self._clinical.text().split(",") if c.strip()]

    def _on_fit(self) -> None:
        feature = self._feature.text().strip() or "flow"
        formula = self._formula.toPlainText().strip()
        groups = self._groups.text().strip() or "territory"
        re_formula = self._re_formula.text().strip() or "0"
        try:
            vc = _parse_vc_formula(self._vc_formula.text())
            df = _load_qvtpy_analysis_frame(
                self._repo,
                feature=feature,
                atlas=str(self._atlas.currentData() or "flow"),
                clinical_vars=self._clinical_list(),
            )
            # Ensure LHS column exists.
            lhs = formula.split("~", 1)[0].strip()
            if lhs and lhs not in df.columns and feature in df.columns:
                df = df.rename(columns={feature: lhs})
            result, model_df, meta = fit_or_load_mixedlm(
                data=df,
                formula=formula,
                groups=groups,
                re_formula=re_formula,
                vc_formula=vc,
                overwrite=True,
                dropna_columns=None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Fit failed", str(exc))
            notify(f"Statmodels fit failed: {exc}", error=True)
            return

        self._last_result = result
        self._last_df = model_df
        self._last_meta = meta
        report = print_mixedlm_info(result, outcome_name=lhs or feature, group_name=groups)
        self._report.setPlainText(report)
        self._status.setText(
            f"Fitted n={meta.get('n_rows')}  |  groups={groups}  |  "
            f"re={re_formula!r}  |  dataset={self._repo.root}"
        )
        self._on_plot()
        notify("MixedLM fit complete.")

    def _clear_plot(self) -> None:
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._plot_canvas = None

    def _on_plot(self) -> None:
        if self._last_result is None or self._last_df is None:
            return
        self._clear_plot()
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            import matplotlib.pyplot as plt

            feature = self._feature.text().strip() or "flow"
            y = "flow" if "flow" in self._last_df.columns else feature
            if y not in self._last_df.columns:
                # Prefer first numeric imaging-like column.
                for cand in ("flow_mean", "pi", "pwv", "pitc_slope"):
                    if cand in self._last_df.columns:
                        y = cand
                        break
            x = "tacsctot_group" if "tacsctot_group" in self._last_df.columns else (
                "age_c" if "age_c" in self._last_df.columns else None
            )
            group = self._groups.text().strip() or "territory"
            if x is None or group not in self._last_df.columns or y not in self._last_df.columns:
                raise ValueError(
                    f"Cannot plot: need y/group columns (have {list(self._last_df.columns)})"
                )

            plot_mixedlm_params(
                result=self._last_result,
                df_fit=self._last_df,
                x=x,
                y=y,
                group=group,
                mode="auto",
                title=f"MixedLM: {y} ~ {x} | {group}",
            )
            fig = plt.gcf()
            canvas = FigureCanvasQTAgg(fig)
            self._plot_layout.addWidget(canvas)
            self._plot_canvas = canvas
            canvas.draw_idle()
        except Exception as exc:
            err = QLabel(f"Plot unavailable: {exc}")
            err.setWordWrap(True)
            self._plot_layout.addWidget(err)

    def _config_dict(self) -> dict[str, Any]:
        return {
            "pipeline": PIPELINE_QVTPY,
            "feature": self._feature.text().strip(),
            "atlas": str(self._atlas.currentData() or "flow"),
            "clinical": self._clinical.text().strip(),
            "mm_formula": self._formula.toPlainText().strip(),
            "groups": self._groups.text().strip(),
            "re_formula": self._re_formula.text().strip(),
            "vc_formula": self._vc_formula.text().strip(),
            "model_name": self._model_name.text().strip(),
            "pipeline_id": QVTPY_PIPELINE_ID,
        }

    def _apply_config(self, cfg: dict[str, Any]) -> None:
        if "feature" in cfg:
            self._feature.setText(str(cfg["feature"]))
        if "clinical" in cfg:
            self._clinical.setText(str(cfg["clinical"]))
        if "mm_formula" in cfg:
            self._formula.setPlainText(str(cfg["mm_formula"]))
        if "groups" in cfg:
            self._groups.setText(str(cfg["groups"]))
        if "re_formula" in cfg:
            self._re_formula.setText(str(cfg["re_formula"]))
        if "vc_formula" in cfg:
            self._vc_formula.setText(str(cfg["vc_formula"]))
        if "model_name" in cfg:
            self._model_name.setText(str(cfg["model_name"]))
        atlas = str(cfg.get("atlas") or "")
        if atlas:
            idx = self._atlas.findData(atlas)
            if idx >= 0:
                self._atlas.setCurrentIndex(idx)

    def _on_save(self) -> None:
        if self._last_result is None:
            notify("Fit a model before saving.", error=True)
            return
        name = self._model_name.text().strip() or "model"
        out_dir = _statmodels_root(self._repo) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl = out_dir / "model.pkl"
        cfg_path = out_dir / "config.json"
        info_path = out_dir / "info.txt"
        try:
            self._last_result.save(str(pkl))
            cfg_path.write_text(json.dumps(self._config_dict(), indent=2), encoding="utf-8")
            info_path.write_text(self._report.toPlainText(), encoding="utf-8")
        except Exception as exc:
            notify(f"Save failed: {exc}", error=True)
            return
        notify(f"Saved model → {out_dir}")
        self._status.setText(f"Saved {out_dir}")

    def _on_load(self) -> None:
        start = str(_statmodels_root(self._repo))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Statmodels config or pickle",
            start,
            "Config/Model (config.json model.pkl);;All (*)",
        )
        if not path:
            return
        p = Path(path)
        model_dir = p.parent if p.name in {"config.json", "model.pkl", "info.txt"} else p
        cfg_path = model_dir / "config.json"
        pkl = model_dir / "model.pkl"
        try:
            if cfg_path.is_file():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self._apply_config(cfg)
            if pkl.is_file():
                from statsmodels.regression.mixed_linear_model import MixedLMResults

                self._last_result = MixedLMResults.load(str(pkl))
                report = print_mixedlm_info(self._last_result)
                self._report.setPlainText(report)
                # Refit frame for plotting covariates if possible.
                try:
                    self._last_df = _load_qvtpy_analysis_frame(
                        self._repo,
                        feature=self._feature.text().strip() or "flow",
                        atlas=str(self._atlas.currentData() or "flow"),
                        clinical_vars=self._clinical_list(),
                    )
                    self._on_plot()
                except Exception:
                    pass
            notify(f"Loaded model from {model_dir}")
        except Exception as exc:
            notify(f"Load failed: {exc}", error=True)


class StatmodelsPanel(QWidget):
    """Right-tab launcher for the floating Statmodels window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._window: StatmodelsWindow | None = None

        self._pipeline = QComboBox()
        self._pipeline.addItem("qvtpy (4D flow hemodynamics)", PIPELINE_QVTPY)

        self._btn = QPushButton("Open Statmodels window")
        self._btn.clicked.connect(self._open_window)

        hint = QLabel(
            "Explore MixedLM formulas over image / clinical / cognitive measurements "
            "from the dataset catalog. Models are saved under "
            "<dataset>/nvitk-statmodels/."
        )
        hint.setWordWrap(True)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("Pipeline"))
        lay.addWidget(self._pipeline)
        lay.addWidget(self._btn)
        lay.addWidget(hint)
        lay.addStretch(1)
        self.setLayout(lay)

    def _open_window(self) -> None:
        if self._window is None:
            self._window = StatmodelsWindow()
        self._window.show_maximized_floating()
