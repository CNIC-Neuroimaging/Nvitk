"""Repo access, formula parsing and small widget helpers shared across the Statmodels panels."""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import ast
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.stats import formula_columns

# Bare Python identifiers — the only formula left-hand sides that can be resolved to a real column.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

log = Logger()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset repo
# ──────────────────────────────────────────────────────────────────────────────
def open_repo() -> DataRepo:
    """Open the configured dataset repo, unwrapping the ``(repo, ...)`` tuple form if returned."""
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def statmodels_root(repo: DataRepo | None = None) -> Path:
    """Ensure and return the directory holding saved model configurations and exports.

    Read from ``settings.json`` ``db.statmodels_root``. Saved models are a researcher's own
    working output rather than dataset content — they are written far more often than the
    dataset changes, and are usefully kept on a backed-up share — so they get their own
    location instead of living inside the dataset tree.

    Falls back to ``<dataset>/nvitk-statmodels`` when the key is unset, which is where models
    were kept before this was configurable, so an existing setup keeps working untouched.
    """
    from nvitk.db.settings_paths import load_db_settings_block

    configured = load_db_settings_block().get("statmodels_root")
    if configured is not None and str(configured).strip():
        root = Path(os.path.expanduser(str(configured).strip()))
    else:
        if repo is None:
            repo = open_repo()
        root = Path(repo.root) / "nvitk-statmodels"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ──────────────────────────────────────────────────────────────────────────────
# Grouping columns
# ──────────────────────────────────────────────────────────────────────────────
#: How few distinct values a *float* column needs before it counts as a factor. Dtype alone
#: cannot tell a numeric-coded factor from a measurement: one missing value upcasts an integer
#: ``sex`` to float64, and excluding floats outright is what kept ``sex`` out of every hue and
#: split picker. A factor coded numerically — sex, a binarized flag, a visit number, a bin index —
#: never has more levels than this; a continuous measurement essentially always does, even in a
#: small frame, where a bare level cap would let it through.
MAX_FLOAT_GROUP_LEVELS: int = 12


def groupable_levels(series: pd.Series, *, cap: int) -> list[str]:
    """
    The levels of *series* if it can be grouped by, in natural order; ``[]`` if it cannot.

    "Can be grouped by" is about how many distinct values there are, not about dtype — at least
    two to compare, and few enough to draw or lay out.
    """
    from nvitk.stats.region_groups import natural_level_key

    values = series.dropna()
    if values.empty:
        return []
    levels = [str(v) for v in pd.unique(values)]
    limit = int(cap)
    if pd.api.types.is_float_dtype(series) and not isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        # Two ways a float says it is a measurement rather than a factor: too many distinct
        # values, or values that never repeat. The second matters on a frame filtered down to a
        # handful of rows, where a measurement has few enough levels to clear the cap on count
        # alone — and a factor's levels repeat at any size.
        limit = min(limit, MAX_FLOAT_GROUP_LEVELS)
        if len(levels) >= len(values):
            return []
    if not 2 <= len(levels) <= limit:
        return []
    return sorted(levels, key=natural_level_key)


def is_groupable(series: pd.Series, *, cap: int) -> bool:
    """Whether *series* can be used as a grouping — see :func:`groupable_levels`."""
    return bool(groupable_levels(series, cap=cap))


def grouping_columns(
    frame: pd.DataFrame | None,
    *,
    cap: int,
    exclude: Sequence[str] = (),
    extra: Sequence[str] = (),
) -> list[str]:
    """
    Every column of *frame* worth offering as a grouping, in the frame's own order.

    Parameters
    ----------
    cap : int
        Most levels a column may have and still be offered. Readability differs by use — overlaid
        violins turn to slivers well before panels do — so the caller sets it.
    exclude : sequence of str
        Columns to leave out regardless, e.g. the one being summarized.
    extra : sequence of str
        Columns to offer even when they blow the cap — a model's own grouping factor belongs in
        the picker whether or not it has 510 levels.
    """
    if frame is None or frame.empty:
        return [name for name in extra if name not in set(exclude)]

    skip = {str(c) for c in exclude}
    forced = {str(c) for c in extra}
    out: list[str] = []
    for column in frame.columns:
        name = str(column)
        if name in skip or name in out:
            continue
        if name in forced or is_groupable(frame[column], cap=cap):
            out.append(name)
    # A forced column the frame does not have is still worth listing: the picker is what tells the
    # user the fit grouped on something the plotting frame no longer carries.
    out += [name for name in forced if name not in set(out) and name not in skip]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Formula helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_vc_formula(text: str) -> dict[str, str] | None:
    """Parse the variance-components formula field (a Python dict literal, e.g.
    ``{"patient": "0 + C(subject_uid)"}``) into a ``{group: formula}`` dict, or ``None`` if blank;
    raises ``ValueError`` if it isn't a dict literal."""
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"vc_formula must be a Python dict literal: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError('vc_formula must be a dict, e.g. {"patient": "0 + C(subject_uid)"}')
    return {str(k): str(v) for k, v in value.items()}


def formula_lhs(formula: str) -> str:
    """Left-hand side of a patsy formula, or ``""`` when there is no ``~``."""
    text = str(formula or "")
    return text.split("~", 1)[0].strip() if "~" in text else ""


def resolve_outcome_column(
    df: pd.DataFrame,
    formula: str,
    measurement_columns: Sequence[str],
) -> tuple[pd.DataFrame, str | None]:
    """
    Resolve the formula's left-hand side to a real frame column, renaming only when unambiguous.

    Three cases, in order:

    1. The LHS is a bare identifier that already exists — use it as-is.
    2. The LHS is *not* a bare identifier (``log(pi)``, ``np.log(pi)``) — leave it to patsy and take
       the plotting outcome from the first formula token that is a column. Prefer a derived column
       for this: a real ``log_pi`` works as a plot axis and a filter target, a formula-level
       transform does not.
    3. The LHS is a bare identifier that is missing — rename the measurement column onto it, but
       only when exactly one measurement is loaded. With several, "the measurement" is ambiguous and
       a guess would silently model the wrong variable.

    Returns
    -------
    (df, outcome_column)
        The frame (renamed only in case 3) and the column to plot on the y axis, or ``None`` when
        nothing suitable was found.

    Raises
    ------
    ValueError
        In case 3 with more than one measurement loaded; the message lists the candidates.
    """
    lhs = formula_lhs(formula)

    # ---- 1. Bare identifier already present ------------------------------------
    if lhs and IDENTIFIER_RE.match(lhs) and lhs in df.columns:
        return df, lhs

    # ---- 2. Transformed LHS: patsy evaluates it, we only need a y for the plot --
    if lhs and not IDENTIFIER_RE.match(lhs):
        for token in formula_columns(df.columns, lhs):
            return df, token
        return df, None

    # ---- 3. Bare identifier missing from the frame -----------------------------
    if lhs:
        present = [c for c in measurement_columns if c in df.columns]
        if len(present) == 1:
            return df.rename(columns={present[0]: lhs}), lhs
        if len(present) > 1:
            raise ValueError(
                f"The formula outcome {lhs!r} is not a column, and {len(present)} measurements are "
                f"loaded ({', '.join(present)}) so it is ambiguous which one it means. Use one of "
                "them as the outcome, or add a derived column named "
                f"{lhs!r}."
            )
        raise ValueError(f"The formula outcome {lhs!r} is not a column of the analysis frame.")

    return df, None


# ──────────────────────────────────────────────────────────────────────────────
# Fit reporting
# ──────────────────────────────────────────────────────────────────────────────
def dropped_rows_note(meta: dict[str, Any]) -> str:
    """Human-readable note about rows dropped for missing values during a fit, or ``""`` if none were."""
    dropped = int(meta.get("n_rows_dropped") or 0)
    if dropped <= 0:
        return ""
    by_col = dict(meta.get("dropped_by_column") or {})
    detail = ", ".join(f"{col} ({n})" for col, n in sorted(by_col.items(), key=lambda kv: -kv[1]))
    return (
        f"NOTE: dropped {dropped} of {meta.get('n_rows_input')} rows with missing values "
        f"before fitting (n={meta.get('n_rows')})."
        + (f" Missing per column: {detail}." if detail else "")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checkable list widgets
# ──────────────────────────────────────────────────────────────────────────────
#: Item role holding a row's searchable text, since the row is a widget and carries no item text.
SEARCH_ROLE = Qt.UserRole + 1


class CovariateRow(QWidget):
    """
    One covariate: a checkbox, and which visit its values should come from.

    A cohort does not measure everything at every visit — carotid plaque exists at visits 3 and 4
    while the rest of the clinical panel exists only at 4 — and collapsing to one row per subject
    has to pick one. Leaving that to a "latest wins" policy makes the answer depend on which visits
    happen to be loaded, so a variable with a choice to make offers it here, right beside the
    checkbox that includes it. A variable recorded at a single visit shows that visit as plain text:
    there is nothing to choose, but the row still says where the number came from.
    """

    def __init__(
        self,
        variable_id: str,
        label: str,
        visits: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        """Build the row for *variable_id*, offering a picker only when *visits* has a choice."""
        super().__init__(parent)
        self.variable_id = str(variable_id)
        self._visits = [str(v) for v in visits if str(v).strip()]

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(4)

        self.check = QCheckBox(f"{label} ({variable_id})")
        self.check.setToolTip(f"{label}\nvariable_id: {variable_id}")
        lay.addWidget(self.check, stretch=1)

        self.combo: QComboBox | None = None
        if len(self._visits) > 1:
            self.combo = QComboBox()
            self.combo.setToolTip(
                f"{variable_id} has values at visits {', '.join(self._visits)}. Pick which one "
                "feeds the model; 'latest' follows the collapse policy, as before."
            )
            # "latest" first so the stored default keeps the behaviour a config saved before this
            # picker existed described — an explicit visit is opt-in.
            self.combo.addItem("latest", "")
            for visit in self._visits:
                self.combo.addItem(f"v{visit}", visit)
            self.combo.setMaximumWidth(84)
            lay.addWidget(self.combo)
        elif len(self._visits) == 1:
            badge = QLabel(f"v{self._visits[0]}")
            badge.setToolTip(f"{variable_id} is only recorded at visit {self._visits[0]}.")
            badge.setEnabled(False)
            lay.addWidget(badge)

    def search_text(self) -> str:
        """What the search box matches against."""
        return self.check.text()

    def visit(self) -> str:
        """The chosen visit id, or ``""`` when there is nothing to choose or 'latest' is selected."""
        return str(self.combo.currentData() or "") if self.combo is not None else ""

    def set_visit(self, visit: str) -> None:
        """
        Select *visit*, falling back to "latest" when this row does not offer it.

        A restore has to be deterministic: leaving a stale pick in place because a saved config
        names a visit the data no longer has would silently fit a different model from the one the
        config describes.
        """
        if self.combo is None:
            return
        wanted = str(visit or "")
        idx = self.combo.findData(wanted)
        if idx < 0:
            log.warning(
                "%s has no visit %r (available: %s) — falling back to the collapse policy.",
                self.variable_id, wanted, ", ".join(self._visits) or "none",
            )
            idx = max(self.combo.findData(""), 0)
        self.combo.setCurrentIndex(idx)


def _rows(widget: QListWidget) -> list[tuple[QListWidgetItem, CovariateRow]]:
    """Every (item, row widget) pair of *widget*, skipping any item without one."""
    out: list[tuple[QListWidgetItem, CovariateRow]] = []
    for i in range(widget.count()):
        item = widget.item(i)
        row = widget.itemWidget(item)
        if isinstance(row, CovariateRow):
            out.append((item, row))
    return out


def populate_checklist(
    widget: QListWidget,
    entries: list[dict[str, Any]],
    *,
    visits: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """
    Fill *widget* with one unchecked :class:`CovariateRow` per variable in *entries*.

    Parameters
    ----------
    visits : mapping, optional
        ``{variable_id: [visit_id, …]}`` from
        :meth:`~nvitk.db.repo.DataRepo.variable_visits`. A variable listed with more than one visit
        gets a picker; one with a single visit gets a label; one that is absent gets neither.
    """
    widget.clear()
    by_id = dict(visits or {})
    for entry in entries:
        vid = str(entry.get("variable_id", "")).strip()
        if not vid:
            continue
        label = str(entry.get("label") or vid)
        row = CovariateRow(vid, label, by_id.get(vid, ()))
        item = QListWidgetItem()
        item.setData(Qt.UserRole, vid)
        item.setData(SEARCH_ROLE, row.search_text())
        item.setSizeHint(row.sizeHint())
        widget.addItem(item)
        widget.setItemWidget(item, row)


def checked_variable_ids(widget: QListWidget) -> list[str]:
    """Variable ids of every checked row in *widget*."""
    return [row.variable_id for _item, row in _rows(widget) if row.check.isChecked()]


def set_checked_variable_ids(widget: QListWidget, ids: list[str]) -> None:
    """Check the rows in *widget* whose variable id is in *ids*, uncheck the rest."""
    want = {str(v).strip() for v in ids if str(v).strip()}
    for _item, row in _rows(widget):
        row.check.setChecked(row.variable_id in want)


def checked_variable_visits(widget: QListWidget) -> dict[str, str]:
    """
    ``{variable_id: visit_id}`` for every checked row that pinned a visit.

    Rows left on "latest", rows with a single visit and unchecked rows are all absent, so the result
    is exactly the set of deliberate choices — which is what
    :func:`~nvitk.stats._statmodels_frames.collapse_visits_to_subject` expects as its overrides.
    """
    out: dict[str, str] = {}
    for _item, row in _rows(widget):
        visit = row.visit()
        if row.check.isChecked() and visit:
            out[row.variable_id] = visit
    return out


def set_variable_visits(widget: QListWidget, visits: Mapping[str, str]) -> None:
    """Restore saved visit picks; a variable that no longer offers its saved visit is left alone."""
    wanted = {str(k): str(v) for k, v in (visits or {}).items()}
    for _item, row in _rows(widget):
        row.set_visit(wanted.get(row.variable_id, ""))


def variable_visit_labels(widget: QListWidget) -> dict[str, list[str]]:
    """``{variable_id: [visit, …]}`` as currently offered — the inverse of *populate_checklist*."""
    return {row.variable_id: list(row._visits) for _item, row in _rows(widget)}


def filter_list_widget(widget: QListWidget, needle: str) -> None:
    """Hide rows of *widget* whose label does not contain *needle* (case-insensitive)."""
    text = str(needle or "").strip().lower()
    for item, _row in _rows(widget):
        haystack = str(item.data(SEARCH_ROLE) or item.text()).lower()
        item.setHidden(bool(text) and text not in haystack)


__all__ = [
    "IDENTIFIER_RE",
    "SEARCH_ROLE",
    "CovariateRow",
    "checked_variable_ids",
    "checked_variable_visits",
    "dropped_rows_note",
    "filter_list_widget",
    "formula_lhs",
    "open_repo",
    "parse_vc_formula",
    "populate_checklist",
    "resolve_outcome_column",
    "set_checked_variable_ids",
    "set_variable_visits",
    "statmodels_root",
    "variable_visit_labels",
]
