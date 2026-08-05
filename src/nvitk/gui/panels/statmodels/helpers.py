"""Repo access, formula parsing and small widget helpers shared across the Statmodels panels."""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import ast
import re
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QListWidget, QListWidgetItem

from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.stats import formula_columns

# Bare Python identifiers — the only formula left-hand sides that can be resolved to a real column.
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset repo
# ──────────────────────────────────────────────────────────────────────────────
def open_repo() -> DataRepo:
    """Open the configured dataset repo, unwrapping the ``(repo, ...)`` tuple form if returned."""
    got = get_repo_from_settings()
    if isinstance(got, tuple):
        return got[0]
    return got


def statmodels_root(repo: DataRepo) -> Path:
    """Ensure and return the ``nvitk-statmodels`` scratch directory under the dataset root, for cached
    fits and saved configs."""
    root = Path(repo.root) / "nvitk-statmodels"
    root.mkdir(parents=True, exist_ok=True)
    return root


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
def populate_checklist(widget: QListWidget, entries: list[dict[str, Any]]) -> None:
    """Fill *widget* with one unchecked, checkable item per variable *entries*, labeled
    ``"<label> (<variable_id>)"``."""
    widget.clear()
    for entry in entries:
        vid = str(entry.get("variable_id", "")).strip()
        if not vid:
            continue
        label = str(entry.get("label") or vid)
        item = QListWidgetItem(f"{label} ({vid})")
        item.setData(Qt.UserRole, vid)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        widget.addItem(item)


def checked_variable_ids(widget: QListWidget) -> list[str]:
    """Variable ids of every checked item in *widget*."""
    out: list[str] = []
    for i in range(widget.count()):
        item = widget.item(i)
        if item.checkState() == Qt.Checked:
            vid = item.data(Qt.UserRole)
            if vid:
                out.append(str(vid))
    return out


def set_checked_variable_ids(widget: QListWidget, ids: list[str]) -> None:
    """Check the items in *widget* whose variable id is in *ids*, uncheck the rest."""
    want = {str(v).strip() for v in ids if str(v).strip()}
    for i in range(widget.count()):
        item = widget.item(i)
        vid = str(item.data(Qt.UserRole) or "")
        item.setCheckState(Qt.Checked if vid in want else Qt.Unchecked)


def filter_list_widget(widget: QListWidget, needle: str) -> None:
    """Hide items of *widget* whose text does not contain *needle* (case-insensitive)."""
    text = str(needle or "").strip().lower()
    for i in range(widget.count()):
        item = widget.item(i)
        item.setHidden(bool(text) and text not in item.text().lower())


__all__ = [
    "IDENTIFIER_RE",
    "checked_variable_ids",
    "dropped_rows_note",
    "filter_list_widget",
    "formula_lhs",
    "open_repo",
    "parse_vc_formula",
    "populate_checklist",
    "resolve_outcome_column",
    "set_checked_variable_ids",
    "statmodels_root",
]
