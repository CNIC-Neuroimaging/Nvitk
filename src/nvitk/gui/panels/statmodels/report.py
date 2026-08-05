"""
Model info panel — stat chips over tabbed tables.

Description
-----------
Replaces the monospace summary dump with something you can read at a glance and act on: a strip of
stat chips (n, groups, convergence, AIC/BIC/LLF) over tabs holding real tables. The coefficient table
sorts numerically, right-aligns its numbers, shades p-values by significance, and copies as TSV.

The classic fixed-width report is still there, verbatim, in the **Raw** tab — nothing that used to be
visible has been taken away.

Mediation results reuse the same shell: the tabs become Paths / By level / Raw, and a warning banner
carries the engine's caveat (pooled OLS, skipped levels, non-converged draws).
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtGui import QColor, QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from nvitk.stats.mixedlm import significance_stars

from .theme import COLOR_ACCENT, COLOR_MUTED, COLOR_WARN, SIGNIFICANCE_COLORS, muted_label_style

# Columns whose cells get the significance shading, keyed by the frame they appear in.
_PVALUE_COLUMNS = {"p_value", "pval"}


# ──────────────────────────────────────────────────────────────────────────────
# Table model
# ──────────────────────────────────────────────────────────────────────────────
class ResultTableModel(QAbstractTableModel):
    """
    Read-only model over a results frame, with numeric sorting and p-value shading.

    Numbers are shown with ``%.4g`` and right-aligned; the raw float is exposed under
    :data:`SORT_ROLE` so a proxy sorts ``1e-135`` below ``0.04`` instead of alphabetically.
    """

    SORT_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        """Start empty; call :meth:`set_frame`."""
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._headers: list[str] = []

    def set_frame(self, df: pd.DataFrame | None, *, headers: Mapping[str, str] | None = None) -> None:
        """Show *df*, optionally renaming columns for display via *headers*."""
        self.beginResetModel()
        self._df = pd.DataFrame() if df is None else df.reset_index(drop=True)
        mapping = dict(headers or {})
        self._headers = [mapping.get(str(c), str(c)) for c in self._df.columns]
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        """The frame currently displayed."""
        return self._df

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Number of rows."""
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Number of columns."""
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        """Formatted text, alignment, significance shading and the numeric sort key."""
        if not index.isValid():
            return None
        column = str(self._df.columns[index.column()])
        value = self._df.iloc[index.row(), index.column()]

        if role == self.SORT_ROLE:
            return float(value) if isinstance(value, (int, float, np.floating)) and not pd.isna(value) else str(value)
        if role == Qt.TextAlignmentRole:
            numeric = pd.api.types.is_numeric_dtype(self._df.dtypes.iloc[index.column()])
            return int(Qt.AlignRight | Qt.AlignVCenter) if numeric else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and column in _PVALUE_COLUMNS:
            colour = SIGNIFICANCE_COLORS.get(significance_stars(_as_float(value)))
            return QColor(colour) if colour else None
        if role == Qt.ToolTipRole:
            # Elided parameter names are unreadable without this.
            return None if pd.isna(value) else str(value)
        if role != Qt.DisplayRole:
            return None

        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4g}"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        return str(value)

    def headerData(self, section: int, orientation: int, role: int = Qt.DisplayRole) -> Any:
        """Column names (possibly renamed for display) and 1-based row numbers."""
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._headers[section] if section < len(self._headers) else None
        return str(section + 1)


def _as_float(value: Any) -> float:
    """Best-effort float conversion, ``nan`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class ResultTableView(QWidget):
    """A sortable results table with a Copy-as-TSV button and ``Ctrl+C`` on the selection."""

    def __init__(self, parent: QWidget | None = None, *, copy_button: bool = True) -> None:
        """Build the view over a numeric-sorting proxy."""
        super().__init__(parent)
        self._model = ResultTableModel(self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(ResultTableModel.SORT_ROLE)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        # Interaction terms produce names like
        # ``C(tacsctot_group, Treatment('g0'))[T.g3]:C(territory, ...)[T.Posterior Circulation]``.
        # Sized to content they consume the whole viewport and push every number out of sight, so
        # cap the width and elide — the full name stays available as a tooltip.
        header.setMaximumSectionSize(360)
        header.setStretchLastSection(True)
        self._table.setTextElideMode(Qt.ElideMiddle)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._table, stretch=1)

        if copy_button:
            row = QHBoxLayout()
            row.addStretch(1)
            button = QPushButton("Copy table")
            button.setToolTip("Copy the whole table as tab-separated text.")
            button.clicked.connect(self.copy_all)
            row.addWidget(button)
            lay.addLayout(row)

        shortcut = QShortcut(QKeySequence.Copy, self._table)
        shortcut.activated.connect(self.copy_selection)

    def set_frame(self, df: pd.DataFrame | None, *, headers: Mapping[str, str] | None = None) -> None:
        """Display *df*."""
        self._model.set_frame(df, headers=headers)

    def copy_all(self) -> None:
        """Put the whole table on the clipboard as TSV."""
        frame = self._model.frame()
        if not frame.empty:
            QApplication.clipboard().setText(frame.to_csv(sep="\t", index=False))

    def copy_selection(self) -> None:
        """Put the selected cells on the clipboard as TSV, falling back to the whole table."""
        indexes = self._table.selectionModel().selectedIndexes()
        if not indexes:
            self.copy_all()
            return
        rows = sorted({i.row() for i in indexes})
        cols = sorted({i.column() for i in indexes})
        lines = [
            "\t".join(
                str(self._proxy.index(r, c).data(Qt.DisplayRole) or "") for c in cols
            )
            for r in rows
        ]
        QApplication.clipboard().setText("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Stat chips
# ──────────────────────────────────────────────────────────────────────────────
class _StatChip(QFrame):
    """A single labeled statistic, rendered as a bordered chip."""

    def __init__(self, title: str, value: str, *, accent: str | None = None) -> None:
        """Build a chip showing *value* under *title*, optionally tinted with *accent*."""
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(0)

        caption = QLabel(title)
        caption.setStyleSheet(f"color: {COLOR_MUTED}; font-weight: normal; font-size: 10px;")
        body = QLabel(value)
        body.setStyleSheet(
            f"color: {accent or '#e0e0e0'}; font-weight: bold; font-size: 13px; border: none;"
        )
        lay.addWidget(caption)
        lay.addWidget(body)
        self.setStyleSheet(
            f"QFrame {{ border: 1px solid {COLOR_ACCENT}; border-radius: 4px; }} "
            "QLabel { border: none; }"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Report panel
# ──────────────────────────────────────────────────────────────────────────────
class ModelReportPanel(QGroupBox):
    """
    Model info: a chip strip over Summary / Coefficients / Random effects / Raw tabs.

    Both :meth:`set_mixedlm` and :meth:`set_mediation` rebuild the same shell, so switching between
    analysis types does not leave a stale tab behind.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the chip strip, the warning banner and the tab widget."""
        super().__init__("Model info", parent)
        lay = QVBoxLayout(self)

        self._chip_row = QHBoxLayout()
        self._chip_row.setSpacing(6)
        lay.addLayout(self._chip_row)

        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(f"color: {COLOR_WARN}; font-weight: normal;")
        self._banner.setVisible(False)
        lay.addWidget(self._banner)

        self._tabs = QTabWidget()
        lay.addWidget(self._tabs, stretch=1)

        self._placeholder = QLabel("Fit a model to see its summary here.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(muted_label_style())
        self._placeholder.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._placeholder)

        self.clear()

    # ---- shell ----------------------------------------------------------------
    def clear(self) -> None:
        """Drop every chip and tab, showing the placeholder."""
        self._set_chips([])
        self._tabs.clear()
        self._tabs.setVisible(False)
        self._banner.setVisible(False)
        self._placeholder.setVisible(True)

    def _set_chips(self, chips: Sequence[tuple[str, str, str | None]]) -> None:
        """Replace the chip strip with ``(title, value, accent)`` triples."""
        while self._chip_row.count():
            item = self._chip_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for title, value, accent in chips:
            self._chip_row.addWidget(_StatChip(title, value, accent=accent))
        self._chip_row.addStretch(1)

    def _set_banner(self, message: str) -> None:
        """Show *message* as a warning banner, or hide the banner when blank."""
        self._banner.setText(message)
        self._banner.setVisible(bool(message))

    def _add_text_tab(self, title: str, text: str) -> None:
        """Append a read-only monospace tab."""
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(text)
        view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._tabs.addTab(view, title)

    def _add_table_tab(self, title: str, frame: pd.DataFrame, *, headers: Mapping[str, str] | None = None) -> None:
        """Append a sortable table tab."""
        view = ResultTableView()
        view.set_frame(frame, headers=headers)
        self._tabs.addTab(view, title)

    # ---- MixedLM --------------------------------------------------------------
    def set_mixedlm(self, info: dict[str, Any], *, raw_text: str, note: str = "") -> None:
        """
        Render a :func:`~nvitk.stats.mixedlm.mixedlm_info_dict` bundle.

        Parameters
        ----------
        raw_text : str
            The classic report, shown verbatim in the Raw tab.
        note : str
            Extra context for the banner, e.g. the dropped-incomplete-rows note.
        """
        header = dict(info.get("header") or {})
        stats = dict(info.get("fit_statistics") or {})

        chips: list[tuple[str, str, str | None]] = [
            ("Observations", f"{int(header.get('n_obs') or 0):,}", None)
        ]
        if header.get("n_groups") is not None:
            chips.append((f"{header.get('group_name', 'Group')} groups", f"{int(header['n_groups']):,}", None))
        if "converged" in stats:
            converged = bool(stats["converged"])
            chips.append(("Converged", "yes" if converged else "no", None if converged else COLOR_WARN))
        for key, label in (("llf", "Log-likelihood"), ("aic", "AIC"), ("bic", "BIC")):
            if key in stats and np.isfinite(stats[key]):
                chips.append((label, f"{stats[key]:.4g}", None))
        if "resid_sd" in stats:
            chips.append(("Residual SD", f"{stats['resid_sd']:.4g}", None))
        self._set_chips(chips)

        messages = [note] if note else []
        if stats.get("converged") is False:
            messages.append(
                "The optimizer did not report convergence — treat the standard errors and p-values "
                "with caution, and check for near-collinear terms or an over-specified random "
                "structure."
            )
        self._set_banner("  ".join(messages))

        self._tabs.clear()
        self._add_text_tab("Summary", _mixedlm_summary_text(header, stats, note))
        self._add_table_tab(
            "Coefficients",
            info["fixed_effects"],
            headers={
                "parameter": "Parameter",
                "coef": "Coef",
                "std_err": "Std.Err",
                "z": "z",
                "p_value": "P-value",
                "ci_low": "CI low",
                "ci_high": "CI high",
                "sig": "Sig",
            },
        )
        self._add_table_tab(
            "Random effects",
            info["random_effects"],
            headers={"component": "Component", "kind": "Kind", "var": "Variance", "sd": "SD"},
        )
        self._add_text_tab("Raw", raw_text)
        self._tabs.setCurrentIndex(1)  # Coefficients is what you look at first
        self._tabs.setVisible(True)
        self._placeholder.setVisible(False)

    # ---- Mediation ------------------------------------------------------------
    def set_mediation(self, bundle: Mapping[str, Any], *, raw_text: str) -> None:
        """Render a :func:`~nvitk.stats.mediation.run_mediation` bundle."""
        from nvitk.stats.mediation import ENGINE_LABELS

        spec = bundle["spec"]
        paths: pd.DataFrame = bundle["paths"]
        indirect = paths.loc[paths["path"] == "Indirect"]

        chips: list[tuple[str, str, str | None]] = [
            ("Engine", ENGINE_LABELS.get(spec.engine, spec.engine).split("(")[0].strip(), None),
            ("Path", f"{spec.x} → {spec.m} → {spec.y}", None),
        ]
        if not indirect.empty:
            row = indirect.iloc[0]
            significant = bool(np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
                               and (row["ci_low"] > 0 or row["ci_high"] < 0))
            chips.append(
                ("Indirect", f"{row['coef']:.4g}", COLOR_ACCENT if significant else None)
            )
            chips.append(("Indirect CI", f"{row['ci_low']:.3g} … {row['ci_high']:.3g}", None))
            chips.append(("p", f"{row['pval']:.3g}", None))
            chips.append(("n", f"{int(row['n']):,}", None))
        self._set_chips(chips)
        self._set_banner(str(bundle.get("note") or ""))

        self._tabs.clear()
        self._add_table_tab(
            "Paths",
            paths[["path", "coef", "ci_low", "ci_high", "pval", "n"]],
            headers={
                "path": "Path",
                "coef": "Coef",
                "ci_low": "CI low",
                "ci_high": "CI high",
                "pval": "p",
                "n": "n",
            },
        )
        summary = bundle.get("summary")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            self._add_table_tab(f"By {spec.group_col}", summary)
        self._add_text_tab("Raw", raw_text)
        self._tabs.setCurrentIndex(0)
        self._tabs.setVisible(True)
        self._placeholder.setVisible(False)

    def set_message(self, message: str) -> None:
        """Show a single informational message in place of a result."""
        self.clear()
        self._placeholder.setText(message)


def _mixedlm_summary_text(header: Mapping[str, Any], stats: Mapping[str, Any], note: str) -> str:
    """Compact header/fit-statistics block for the Summary tab."""
    lines: list[str] = []
    if note:
        lines += [note, ""]
    if header.get("formula"):
        lines.append(f"Formula        : {header['formula']}")
    lines.append(f"Outcome        : {header.get('outcome_name', '—')}")
    lines.append(f"Observations   : {int(header.get('n_obs') or 0):,}")
    if header.get("n_groups") is not None:
        lines.append(f"{header.get('group_name', 'Group'):<15}: {int(header['n_groups']):,} groups")
    lines.append("")
    lines.append("Fit statistics")
    lines.append("-" * 40)
    for key, label in (
        ("llf", "Log-likelihood"),
        ("aic", "AIC"),
        ("bic", "BIC"),
        ("scale", "Residual variance"),
        ("resid_sd", "Residual SD"),
    ):
        if key in stats:
            lines.append(f"{label:<20}: {stats[key]:.6g}")
    if "converged" in stats:
        lines.append(f"{'Converged':<20}: {bool(stats['converged'])}")
    return "\n".join(lines)


__all__ = ["ModelReportPanel", "ResultTableModel", "ResultTableView"]
