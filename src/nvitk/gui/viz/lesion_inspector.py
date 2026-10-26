"""Inspect the lesions of a morphometrics run: hover card, calibre profile, extents.

The lesion points (:func:`nvitk.gui.viz.morpho_viz.add_morpho_lesion_points`) mark each
stenosis / enlargement at its midpoint. This module makes them explorable:

- **hover** a lesion point: a card with the vessel, segment, severity (diameter / area % against
  the healthy reference, taper %), length, position and the confidence with each penalty
  spelled out;
- **click** it (or its row in the results window's lesion table): the *Lesion inspector* dock
  plots the calibre along the measured path; inscribed (detection) and cross-section radius, the
  healthy taper reference and the threshold the radius had to cross, the lesion's run, the
  reference windows (and the interpolated reference between them), the siphon mask and the path's
  other lesions, i.e. why the lesion was called and how much to trust it. ◀ / ▶ step through the
  lesions;
- **extents**: each lesion's run is drawn along its centerline (``Morpho lesion extents``), not
  just its midpoint;
- **filter** by confidence (:func:`confidence_filter_controls`, above the results window's lesion
  table): lesions below the chosen confidence are hidden from the viewer (points and extents)
  and skipped by hover, click and ◀ / ▶. The table rows are tinted by confidence class
  (:data:`LESION_CLASS_COLOURS`: green high, yellow moderate, red low).

Everything is read back from the run's folder: ``lesions.csv`` and the per-path point tables
(``points/<path_id>.csv``, through ``Sheet_Index``); the thresholds come from
``morphometrics_provenance.json``.
"""

from __future__ import annotations

import json
import re
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LESION_EXTENTS_LAYER = "Morpho lesion extents"
#: Confidence class -> colour (table rows, filter labels).
LESION_CLASS_COLOURS: dict[str, str] = {"high": "#5fb35f", "moderate": "#d4b83a", "low": "#d9534f"}
#: Filter presets: label -> minimum confidence (the class boundaries of the lesion table).
CONFIDENCE_PRESETS: dict[str, float] = {
    "all lesions": 0.0,
    "moderate and high (≥ 0.4)": 0.4,
    "high only (≥ 0.7)": 0.7,
}
_DOCK_NAME = "nvitk_morpho_lesion_dock"
_STATE_ATTR = "_nvitk_lesion_state"
#: Extents-layer metadata: the lesion each path belongs to, and its full colour.
_EXTENT_OWNERS = "nvitk_lesion_index"
_EXTENT_COLOURS = "nvitk_lesion_colour"

#: Confidence flags -> (what it means, factor). ``{near_end}`` / ``{near_bif}`` are filled in mm.
FLAG_TEXT: dict[str, tuple[str, float]] = {
    "reference_distal": ("no proximal reference window: distal one used", 0.8),
    "reference_none": ("no reference window", 0.8),
    "reference_inconsistent": ("proximal and distal references disagree beyond taper", 0.8),
    "reference_fallback": ("no healthy window in its stretch: plain windows used", 0.8),
    "in_siphon": ("overlaps the siphon (tight-bend) mask", 0.7),
    "near_vessel_end": ("within {near_end:g} mm of a path end", 0.6),
    "short": ("shorter than 1.5 × the minimum length", 0.8),
    "near_bifurcation": ("within {near_bif:g} mm of a junction", 0.5),
    "radius_disagreement": ("cross-section and inscribed radii disagree", 0.7),
    "cross_section_gaps": ("> 20 % of it has no cross-section", 0.8),
    # Hypoplasia (absent main vessels): what was missing, and what the topology says.
    "absent": ("absent from the mask (no flow signal: aplasia, severe hypoplasia or occlusion)", None),
    "too_small": ("in the mask but too small for a centerline", None),
    "neighbours_absent": ("its neighbouring vessels are absent too (field of view / segmentation?)", 0.3),
    "fetal_pca": ("fetal-type PCA: PComm and P2 present", None),
    "a1_aplasia": ("A1 aplasia: AComm and the other ACA present", None),
}


@dataclass
class LesionRun:
    """The lesions of one morphometrics output folder and what is needed to explain them."""

    stage7_dir: Path
    lesions: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)
    _points_csv: dict[str, Path] = field(default_factory=dict)
    _paths: dict[str, pd.DataFrame | None] = field(default_factory=dict)

    def param(self, name: str, default: Any) -> Any:
        return self.params.get(name, default)

    def path_points(self, path_id: str) -> pd.DataFrame | None:
        """The point table of a measured path (cached)."""
        if path_id not in self._paths:
            csv = self._points_csv.get(path_id)
            if csv is None:
                safe = re.sub(r"[^\w.-]+", "_", path_id)
                for cand in (self.stage7_dir / "points" / f"{path_id}.csv", self.stage7_dir / "points" / f"{safe}.csv"):
                    if cand.is_file():
                        csv = cand
                        break
            try:
                self._paths[path_id] = pd.read_csv(csv) if csv is not None else None
            except Exception:  # noqa: BLE001
                self._paths[path_id] = None
        return self._paths[path_id]

    def span(self, i: int) -> tuple[pd.DataFrame, int, int] | None:
        """``(path points, first index, last index)`` of lesion *i* on its path."""
        row = self.lesions.iloc[int(i)]
        pts = self.path_points(str(row.get("path_id", "")))
        if pts is None or pts.empty or "s_mm" not in pts.columns:
            return None
        n = len(pts)
        if pd.notna(row.get("point_start")) and pd.notna(row.get("point_end")):
            a, b = int(row["point_start"]), int(row["point_end"])
        else:  # outputs written before the lesion table kept the indices
            s = pts["s_mm"].to_numpy(dtype=float)
            s0 = float(row.get("start_s_from_vessel_origin_mm", 0.0))
            a = int(np.argmin(np.abs(s - s0)))
            b = int(np.argmin(np.abs(s - (s0 + float(row.get("length_mm", 0.0))))))
        a, b = sorted((max(0, min(a, n - 1)), max(0, min(b, n - 1))))
        return pts, a, b


def load_lesion_run(stage7_dir: Path | str) -> LesionRun | None:
    """The lesions of a run folder, ``None`` when it has none."""
    d = Path(stage7_dir)
    try:
        lesions = pd.read_csv(d / "lesions.csv")
    except Exception:  # noqa: BLE001
        return None
    if lesions.empty:
        return None
    params: dict[str, Any] = {}
    try:
        params = dict(json.loads((d / "morphometrics_provenance.json").read_text()).get("parameters", {}))
    except Exception:  # noqa: BLE001
        params = {}
    csvs: dict[str, Path] = {}
    try:
        index = pd.read_excel(d / "case_metrics_donut_tree.xlsx", sheet_name="Sheet_Index")
        csvs = {str(k): d / str(v) for k, v in zip(index["path_id"], index["points_csv"])}
    except Exception:  # noqa: BLE001
        csvs = {}
    return LesionRun(stage7_dir=d, lesions=lesions.reset_index(drop=True), params=params, _points_csv=csvs)


def _num(row: Any, key: str) -> float:
    try:
        v = float(row.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return v


def flag_lines(row: Any, run: LesionRun | None = None) -> list[str]:
    """Each confidence penalty of a lesion, in words, with its factor."""
    near_end = float(run.param("LESION_NEAR_END_MM", 5.0)) if run else 5.0
    near_bif = float(run.param("LESION_NEAR_BIFURCATION_MM", 3.0)) if run else 3.0
    out = []
    for flag in [f for f in str(row.get("confidence_flags", "") or "").split(";") if f and f != "nan"]:
        text, factor = FLAG_TEXT.get(flag, (flag, float("nan")))
        text = text.format(near_end=near_end, near_bif=near_bif)
        out.append(text if factor is None else f"{text} ×{factor:g}")
    return out


def lesion_card(row: Any, run: LesionRun | None = None, *, html: bool = True) -> str:
    """A short description of one lesion (hover card / inspector header)."""
    kind = str(row.get("kind", "lesion"))
    wider = kind == "enlargement"
    d_pct, a_pct, taper = _num(row, "diameter_percent"), _num(row, "area_percent"), _num(row, "taper_percent")
    ref, ext = _num(row, "reference_radius_mm"), _num(row, "extreme_radius_mm")
    conf = _num(row, "confidence")
    sev = "wider" if wider else "narrower"
    if kind == "hypoplasia":
        lines = [f"{row.get('vessel_name', '')} — hypoplasia",
                 f"confidence {conf:.2f} ({row.get('confidence_class', '')})"]
        flags = flag_lines(row, run)
        if not html:
            return "\n".join(lines + [f"  – {f}" for f in flags])
        return f"<b>{lines[0]}</b><br>{lines[1]}" + "".join(f"<br>&nbsp;&nbsp;– {f}" for f in flags)
    lines = [
        f"{row.get('vessel_name', '')} — {kind}" + (f" · segment {row.get('segment')}" if row.get("segment") else ""),
        (f"{d_pct:.0f} % {sev} in diameter, {a_pct:.0f} % in area vs the {row.get('reference', '')} reference "
         f"(radius {ref:.2f} → {ext:.2f} mm)") if np.isfinite(d_pct) else "no reference window for a diameter %",
        f"taper {taper:.0f} % · length {_num(row, 'length_mm'):.1f} mm · "
        f"from the vessel origin {_num(row, 'start_s_from_vessel_origin_mm'):.1f} mm",
        f"confidence {conf:.2f} ({row.get('confidence_class', '')})",
    ]
    flags = flag_lines(row, run)
    if not html:
        return "\n".join(lines + [f"  – {f}" for f in flags])
    head = f"<b>{lines[0]}</b>"
    body = "<br>".join(lines[1:])
    tail = "".join(f"<br>&nbsp;&nbsp;– {f}" for f in flags) if flags else "<br>&nbsp;&nbsp;no penalty"
    return f"{head}<br>{body}{tail}"


# The calibre profile


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(np.append(np.asarray(flag, dtype=bool), False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def _reference(pts: pd.DataFrame, kind: str) -> np.ndarray | None:
    """The taper reference along the path (older outputs: recovered from the raw stenosis %)."""
    col = f"{kind}_reference_radius_mm"
    if col in pts.columns and np.isfinite(pd.to_numeric(pts[col], errors="coerce")).any():
        return pd.to_numeric(pts[col], errors="coerce").to_numpy(dtype=float)
    if kind == "stenosis" and {"stenosis_raw_percent_point", "stenosis_detection_radius_mm"} <= set(pts.columns):
        pct = pd.to_numeric(pts["stenosis_raw_percent_point"], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(pts["stenosis_detection_radius_mm"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = r / (1.0 - pct / 100.0)
        return np.where(np.isfinite(ref) & (pct < 100.0), ref, np.nan)
    return None


def draw_profile(ax: Any, run: LesionRun, i: int) -> str:
    """Plot lesion *i*'s calibre profile on *ax*; returns a note (empty when all went well)."""
    from nvitk.gui.core.design import COLOR_MUTED, COLOR_OK, COLOR_SURFACE, COLOR_TEXT
    from nvitk.gui.viz.morpho_viz import LESION_COLOURS

    ax.clear()
    row = run.lesions.iloc[int(i)]
    if str(row.get("kind", "")) == "hypoplasia":
        ax.text(0.5, 0.5, "No calibre profile: the vessel has no measured centerline.", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=COLOR_MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        return "A hypoplasia is an absent main vessel; see the flags above."
    found = run.span(i)
    if found is None:
        ax.text(0.5, 0.5, "Point table of this path not found", ha="center", va="center", color=COLOR_MUTED,
                transform=ax.transAxes)
        return "The run folder has no point table for this path."
    pts, a, b = found
    kind = str(row.get("kind", "stenosis"))
    colour = LESION_COLOURS.get(kind, (0.7, 0.7, 0.7))
    s = pts["s_mm"].to_numpy(dtype=float)
    r_det = pd.to_numeric(pts.get("stenosis_detection_radius_mm", pts.get("radius_mm")), errors="coerce").to_numpy(float)
    r_cs = pd.to_numeric(pts.get("radius_mm"), errors="coerce").to_numpy(float)
    # The radius this kind of lesion was tested on: stenoses on the detection (inscribed) radius;
    # enlargements on their own (ENLARGEMENT_RADIUS_SOURCE; cow: cross-section on a path's first
    # measurement, the detection radius on re-measured paths; outputs without the column used the
    # cross-section). Anything else than the detection radius is the cross-section one, possibly
    # its running median (ENLARGEMENT_SMOOTH_CROSS_SECTION).
    if kind == "enlargement":
        tested = (pd.to_numeric(pts["enlargement_radius_mm"], errors="coerce").to_numpy(float)
                  if "enlargement_radius_mm" in pts.columns else r_cs)
    else:
        tested = r_det
    on_cs = (not np.allclose(tested, r_det, equal_nan=True)) or bool(np.allclose(r_cs, r_det, equal_nan=True))
    smoothed = on_cs and not np.allclose(tested, r_cs, equal_nan=True)
    alt_r, alt_name = (r_det, "inscribed radius") if on_cs else (r_cs, "cross-section radius")
    tested_name = ("cross-section radius, median" if smoothed else "cross-section radius" if on_cs
                   else "inscribed radius") + " (tested)"

    # Context first (underneath): siphon mask, the path's other lesions, the reference window.
    if "siphon_mask_point" in pts.columns:
        for k, (p, q) in enumerate(_runs(pts["siphon_mask_point"].to_numpy(dtype=float) > 0.5)):
            ax.axvspan(s[p], s[q], color=COLOR_MUTED, alpha=0.15, lw=0, label="siphon mask" if k == 0 else None)
    same_path = run.lesions.index[run.lesions["path_id"].astype(str) == str(row.get("path_id"))]
    first_other = True
    for j in same_path:
        if j == i:
            continue
        other = run.span(j)
        if other is None:
            continue
        _, p, q = other
        ax.axvspan(s[p], s[q], color=LESION_COLOURS.get(str(run.lesions.iloc[j].get("kind")), colour), alpha=0.10,
                   lw=0, label="other lesions" if first_other else None)
        first_other = False
    lo, hi = (float(v) for v in run.param("LESION_PROXIMAL_REFERENCE_WINDOW_MM", (2.0, 10.0)))
    end_ex = float(run.param("LESION_REFERENCE_EXCLUDE_END_MM", 0.0) or 0.0)  # older runs: none
    ref_where = str(row.get("reference", ""))
    ref_r = _num(row, "reference_radius_mm")
    spans = {side: (_num(row, f"reference_{side}_start_point"), _num(row, f"reference_{side}_end_point"))
             for side in ("proximal", "distal")}
    if any(np.isfinite(p0) for p0, _ in spans.values()):
        # The windows the run used (LESION_REFERENCE_METHOD): both for an interpolated reference.
        anchors: dict[str, tuple[float, float]] = {}
        for side in (("proximal", "distal") if ref_where == "interpolated" else (ref_where,)):
            p0, p1 = spans.get(side, (np.nan, np.nan))
            if not (np.isfinite(p0) and np.isfinite(p1)):
                continue
            w0, w1 = s[int(p0)], s[int(p1)]
            ax.axvspan(w0, w1, color=COLOR_OK, alpha=0.16, lw=0, label=f"{side} reference window")
            median = _num(row, f"reference_{side}_radius_mm")
            anchors[side] = (0.5 * (w0 + w1), median)
            if np.isfinite(median):  # each window's own median, over the window
                ax.hlines(median, w0, w1, colors=COLOR_OK, lw=2.2,
                          label="window median" if len(anchors) == 1 else None)
        if len(anchors) == 2:
            (sp, rp), (sd, rd) = anchors["proximal"], anchors["distal"]
            ax.plot([sp, sd], [rp, min(rd, rp)], color=COLOR_OK, lw=1.2, ls="--", label="interpolated reference")
            if np.isfinite(rd) and np.isfinite(rp) and rd > rp:
                # A healthy vessel does not widen downstream: the distal median is capped at the proximal.
                ax.annotate("distal median capped\nat the proximal one", (sd, rd), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=6.5, color=COLOR_OK)
    elif ref_where in ("proximal", "distal"):  # outputs written before the windows were recorded
        w0, w1 = ((s[a] - hi, s[a] - lo) if ref_where == "proximal" else (s[b] + lo, s[b] + hi))
        w0, w1 = max(w0, s[0] + end_ex), min(w1, s[-1] - end_ex)
        ax.axvspan(w0, w1, color=COLOR_OK, alpha=0.16, lw=0, label=f"{ref_where} reference window")
        if np.isfinite(ref_r):
            ax.hlines(ref_r, w0, w1, colors=COLOR_OK, lw=1.4)
    ax.axvspan(s[a], s[b], color=colour, alpha=0.30, lw=0, label="this lesion")
    if "taper_stretch" in pts.columns:  # TAPER_REFERENCE_METHOD = "stretch": where it may step
        cuts = np.flatnonzero(np.diff(pts["taper_stretch"].to_numpy(dtype=float)) != 0)
        for k, c in enumerate(cuts):
            ax.axvline(s[c + 1], color=COLOR_MUTED, lw=1.0, ls=(0, (4, 3)),
                       label="junction (taper reference may step)" if k == 0 else None)

    ax.plot(s, alt_r, color=COLOR_MUTED, lw=0.8, label=alt_name)
    ax.plot(s, tested, color=COLOR_TEXT, lw=1.3, label=tested_name)
    ref = _reference(pts, kind)
    pct = float(run.param("STENOSIS_THRESHOLD_PCT" if kind == "stenosis" else "ENLARGEMENT_THRESHOLD_PCT", 10.0))
    if ref is not None:
        thr = ref * (1.0 - pct / 100.0) if kind == "stenosis" else ref * (1.0 + pct / 100.0)
        ax.plot(s, ref, "--", color=colour, lw=1.1, label="taper reference")
        ax.plot(s, thr, ":", color=colour, lw=1.1, label=f"{pct:g} % threshold")
    seg = tested[a:b + 1]
    if np.isfinite(seg).any():
        k = a + int(np.nanargmin(seg) if kind == "stenosis" else np.nanargmax(seg))
        ax.plot([s[k]], [tested[k]], "o", color=colour, ms=6, mec=COLOR_TEXT, mew=0.8)
        if np.isfinite(ref_r):  # where the diameter / area % read the reference
            ax.plot([s[k]], [ref_r], "D", color=COLOR_OK, ms=4.5, mec=COLOR_TEXT, mew=0.6,
                    label="reference at the lesion")
    pad = max(10.0, 2.0 * (s[b] - s[a]))
    ax.set_xlim(max(s[0], s[a] - pad - hi), min(s[-1], s[b] + pad))
    ax.set_xlabel("distance along the path (mm)", fontsize=8)
    ax.set_ylabel("radius (mm)", fontsize=8)
    ax.set_title(f"{row.get('vessel_name', '')} {kind} · path {row.get('path_id', '')}", fontsize=9)
    ax.tick_params(labelsize=7)
    # Below the axes: inside, it hides the start of the curve on a narrow dock.
    leg = ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, framealpha=0.6)
    for text in leg.get_texts():
        text.set_color(COLOR_TEXT)
    leg.get_frame().set_facecolor(COLOR_SURFACE)
    return "" if ref is not None else "This output has no taper reference for enlargements (re-run to plot it)."


# The inspector dock


def _make_inspector(viewer: Any) -> Any:
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    from nvitk.gui.core.design import COLOR_MUTED, SPACE_TIGHT

    class LesionInspector(QWidget):
        """Calibre profile and description of one lesion, with ◀ / ▶ through the run's lesions."""

        def __init__(self) -> None:
            super().__init__()
            self._viewer_ref = weakref.ref(viewer)
            self.index = 0
            root = QVBoxLayout(self)
            root.setSpacing(SPACE_TIGHT)
            nav = QHBoxLayout()
            self._prev, self._next = QPushButton("◀"), QPushButton("▶")
            self._prev.setToolTip("Previous lesion")
            self._next.setToolTip("Next lesion")
            self._where = QLabel("")
            self._centre = QPushButton("Show in viewer")
            self._centre.setToolTip("Select the lesion point and bring it into view")
            nav.addWidget(self._prev)
            nav.addWidget(self._where, stretch=1)
            nav.addWidget(self._next)
            nav.addWidget(self._centre)
            root.addLayout(nav)
            self._card = QLabel("")
            self._card.setWordWrap(True)
            root.addWidget(self._card)
            self._fig = self._ax = self._canvas = None
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                from matplotlib.figure import Figure

                self._fig = Figure(figsize=(5.0, 3.2), dpi=96, layout="constrained")
                self._ax = self._fig.add_subplot(111)
                self._canvas = FigureCanvasQTAgg(self._fig)
                root.addWidget(self._canvas, stretch=1)
            except Exception as exc:  # noqa: BLE001
                root.addWidget(QLabel(f"Matplotlib unavailable: {exc}"))
            self._note = QLabel("")
            self._note.setWordWrap(True)
            self._note.setStyleSheet(f"color: {COLOR_MUTED};")
            root.addWidget(self._note)
            self._prev.clicked.connect(lambda: self.step(-1))
            self._next.clicked.connect(lambda: self.step(+1))
            # The plot is drawn with the colour tokens, which a theme switch rebinds: redraw then.
            from nvitk.gui.core.design import add_theme_listener

            add_theme_listener(self._on_theme)
            self._centre.clicked.connect(lambda: _centre_on(viewer, self.index))

        def step(self, direction: int) -> None:
            """The previous / next lesion the confidence filter shows."""
            state = getattr(self._viewer_ref(), _STATE_ATTR, None)
            run = state.get("run") if state else None
            if run is None:
                return
            visible = visible_lesions(state)
            if not len(visible):
                return
            n = len(run.lesions)
            for k in range(1, n + 1):
                j = (self.index + direction * k) % n
                if j in visible:
                    self.show_lesion(j)
                    return

        def _on_theme(self) -> None:
            """Redraw the current lesion in the new theme's colours."""
            state = getattr(self._viewer_ref(), _STATE_ATTR, None)
            if state and state.get("run") is not None:
                self.show_lesion(self.index)

        def show_lesion(self, i: int) -> None:
            state = getattr(self._viewer_ref(), _STATE_ATTR, None)
            run = state.get("run") if state else None
            if run is None:
                return
            n = len(run.lesions)
            self.index = int(i) % n
            shown = len(visible_lesions(state))
            self._where.setText(f"lesion {self.index + 1} / {n}" + (f" ({shown} shown)" if shown < n else ""))
            self._card.setText(lesion_card(run.lesions.iloc[self.index], run))
            note = ""
            if self._ax is not None and self._canvas is not None:
                from nvitk.gui.core.design import style_image_figure

                note = draw_profile(self._ax, run, self.index)
                style_image_figure(self._fig)
                self._canvas.draw_idle()
            kind = str(run.lesions.iloc[self.index].get("kind", "stenosis"))
            if kind == "hypoplasia":
                self._note.setText(note)
                return
            pct = float(run.param("STENOSIS_THRESHOLD_PCT" if kind == "stenosis" else "ENLARGEMENT_THRESHOLD_PCT",
                                  10.0))
            self._note.setText(
                (note + " " if note else "")
                + f"The lesion is the run where the tested radius crosses the dotted threshold ({pct:g} % off "
                  "the dashed taper reference); its diameter / area % compare its extreme radius (dot) with "
                  "the green reference (diamond): the reference windows' median, interpolated between the "
                  "proximal and distal one when both exist."
            )

    return LesionInspector()


def inspect_lesion(viewer: Any, i: int) -> Any:
    """Show lesion *i* of the current run in the *Lesion inspector* dock (created on first use)."""
    from nvitk.gui.viz.left_dock import attach_left_inspection_dock

    state = getattr(viewer, _STATE_ATTR, None)
    if not state or state.get("run") is None:
        return None
    widget = state.get("inspector")
    alive = False
    if widget is not None:
        try:
            widget.isVisible()
            alive = True
        except RuntimeError:  # deleted by Qt with its dock
            alive = False
    if not alive:
        widget = _make_inspector(viewer)
        state["inspector"] = widget
        dock = attach_left_inspection_dock(
            viewer, widget, object_name=_DOCK_NAME, title="Lesion inspector",
            tabify_with=["nvitk_morphometrics_dock"], minimum_width=340,
        )
        state["dock"] = dock
    widget.show_lesion(i)
    dock = state.get("dock")
    if dock is not None:
        try:
            dock.show()
            dock.raise_()
        except Exception:  # noqa: BLE001
            pass
    return widget


def _centre_on(viewer: Any, i: int) -> None:
    state = getattr(viewer, _STATE_ATTR, None)
    layer = state["layer"]() if state and state.get("layer") is not None else None
    if layer is None or layer not in viewer.layers or not (0 <= int(i) < len(layer.data)):
        return
    run = state.get("run")
    if run is not None and int(i) < len(run.lesions) and not located(run.lesions)[int(i)]:
        return  # a hypoplasia has nowhere to go
    layer.selected_data = {int(i)}
    world = np.asarray(layer.data_to_world(np.asarray(layer.data[int(i)], dtype=float)), dtype=float)
    nd = len(viewer.dims.point)
    for axis, value in zip(range(nd - len(world), nd), world):
        viewer.dims.set_point(axis, float(value))
    try:
        viewer.camera.center = tuple(float(v) for v in world[-3:])
    except Exception:  # noqa: BLE001
        pass


# Hover, click and extents


def _lesion_under_cursor(layer: Any, event: Any) -> int | None:
    try:
        value = layer.get_value(
            event.position, view_direction=event.view_direction, dims_displayed=list(event.dims_displayed),
            world=True,
        )
    except Exception:  # noqa: BLE001
        return None
    if value is None:
        return None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < len(layer.data) else None


def _on_mouse_move(viewer: Any, event: Any) -> None:
    from qtpy.QtGui import QCursor
    from qtpy.QtWidgets import QToolTip

    state = getattr(viewer, _STATE_ATTR, None)
    layer = state["layer"]() if state and state.get("layer") is not None else None
    if layer is None or layer not in viewer.layers or not layer.visible:
        if state and state.get("hover") is not None:
            QToolTip.hideText()
            state["hover"] = None
        return
    idx = _lesion_under_cursor(layer, event)
    if idx == state.get("hover"):
        return
    state["hover"] = idx
    if idx is None:
        QToolTip.hideText()
        return
    run = state["run"]
    QToolTip.showText(QCursor.pos(), lesion_card(run.lesions.iloc[idx], run) + "<br><i>click to inspect</i>")


def _on_mouse_press(viewer: Any, event: Any) -> None:
    if getattr(event, "type", "") != "mouse_press" or getattr(event, "button", 1) != 1:
        return
    state = getattr(viewer, _STATE_ATTR, None)
    layer = state["layer"]() if state and state.get("layer") is not None else None
    if layer is None or layer not in viewer.layers or not layer.visible:
        return
    idx = _lesion_under_cursor(layer, event)
    if idx is not None:
        inspect_lesion(viewer, idx)


def add_lesion_extents(
    viewer: Any, run: LesionRun, *, reference_layer: Any | None, metadata: dict[str, Any] | None = None,
    edge_width: float = 1.2,
) -> Any | None:
    """Each lesion's run drawn along its centerline, in its kind's colour."""
    from nvitk.gui.viz.morpho_viz import LESION_COLOURS, _overlay_kwargs, _vtp_mm_to_data

    for lyr in list(viewer.layers):
        if lyr.name == LESION_EXTENTS_LAYER:
            viewer.layers.remove(lyr)
    paths, colours, owners = [], [], []
    for i in range(len(run.lesions)):
        found = run.span(i)
        if found is None:
            continue
        pts, a, b = found
        if not {"x_mm", "y_mm", "z_mm"} <= set(pts.columns):
            continue
        xyz = pts[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)[a:b + 1]
        if len(xyz) < 2:
            xyz = pts[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)[max(a - 1, 0):b + 2]
        if len(xyz) < 2:
            continue
        paths.append(_vtp_mm_to_data(xyz, reference_layer).astype(np.float32))
        colours.append((*LESION_COLOURS.get(str(run.lesions.iloc[i].get("kind")), (0.7, 0.7, 0.7)), 1.0))
        owners.append(i)
    if not paths:
        return None
    meta = {**dict(metadata or {}), _EXTENT_OWNERS: owners, _EXTENT_COLOURS: colours}
    shapes = viewer.add_shapes(
        paths, shape_type="path", name=LESION_EXTENTS_LAYER, edge_color=colours, edge_width=float(edge_width),
        opacity=0.9, metadata=meta, **_overlay_kwargs(reference_layer),
    )
    try:
        shapes.editable = False
    except Exception:  # noqa: BLE001
        pass
    return shapes


def install_lesion_tools(
    viewer: Any,
    stage7_dir: Path | str,
    *,
    lesion_layer: Any,
    reference_layer: Any | None,
    metadata: dict[str, Any] | None = None,
    extents: bool = True,
    extents_layer: Any | None = None,
) -> Any | None:
    """Hover cards, click-to-inspect and (with *extents*) the extents overlay for a run's lesion
    layer. Returns the extents layer (``None`` when not drawn or the point tables are missing).
    """
    run = load_lesion_run(stage7_dir)
    if run is None or lesion_layer is None:
        return None
    state = getattr(viewer, _STATE_ATTR, None) or {}
    state.update(run=run, layer=weakref.ref(lesion_layer), hover=None)
    state.setdefault("min_confidence", 0.0)
    setattr(viewer, _STATE_ATTR, state)
    if not state.get("callbacks"):
        viewer.mouse_move_callbacks.append(_on_mouse_move)
        viewer.mouse_drag_callbacks.append(_on_mouse_press)
        state["callbacks"] = True
    widget = state.get("inspector")
    if widget is not None:
        try:
            if widget.isVisible():
                widget.show_lesion(0)
        except RuntimeError:
            state["inspector"] = None
    if extents:
        extents_layer = add_lesion_extents(viewer, run, reference_layer=reference_layer, metadata=metadata)
    state["extents"] = weakref.ref(extents_layer) if extents_layer is not None else None
    # A new run keeps the filter the user had chosen.
    apply_confidence_filter(viewer, float(state.get("min_confidence", 0.0)))
    return extents_layer


# Filtering by confidence


def _confidences(run: LesionRun) -> np.ndarray:
    conf = pd.to_numeric(run.lesions.get("confidence"), errors="coerce")
    return conf.fillna(0.0).to_numpy(dtype=float) if conf is not None else np.zeros(len(run.lesions))


def located(lesions: pd.DataFrame) -> np.ndarray:
    """The lesions with a position (``x/y/z_mm``); a hypoplasia (an absent vessel) has none."""
    if lesions.empty or not {"x_mm", "y_mm", "z_mm"} <= set(lesions.columns):
        return np.zeros(len(lesions), dtype=bool)
    xyz = lesions[["x_mm", "y_mm", "z_mm"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.isfinite(xyz).all(axis=1)


def visible_lesions(state: dict[str, Any] | None) -> set[int]:
    """Indices of the lesions the confidence filter shows (and that have a place to be shown at)."""
    run = state.get("run") if state else None
    if run is None:
        return set()
    keep = (_confidences(run) >= float(state.get("min_confidence", 0.0)) - 1e-9) & located(run.lesions)
    return set(np.flatnonzero(keep).tolist())


def apply_confidence_filter(viewer: Any, min_confidence: float) -> tuple[int, int]:
    """Show only the lesions with confidence >= *min_confidence* (points and extents);
    returns ``(shown, total)``."""
    state = getattr(viewer, _STATE_ATTR, None)
    run = state.get("run") if state else None
    if run is None:
        return 0, 0
    state["min_confidence"] = float(min_confidence)
    keep = (_confidences(run) >= float(min_confidence) - 1e-9) & located(run.lesions)
    layer = state["layer"]() if state.get("layer") is not None else None
    if layer is not None and layer in viewer.layers and len(layer.data) == len(keep):
        layer.shown = keep
        layer.selected_data = {i for i in layer.selected_data if keep[i]}
    extents = state["extents"]() if state.get("extents") is not None else None
    if extents is not None and extents in viewer.layers:
        meta = dict(extents.metadata or {})
        owners = meta.get(_EXTENT_OWNERS)
        if owners is not None and len(owners) == len(extents.data):
            colours = np.asarray(meta[_EXTENT_COLOURS], dtype=float).copy()
            colours[:, 3] = np.where(keep[np.asarray(owners, dtype=int)], 1.0, 0.0)
            extents.edge_color = colours
    state["hover"] = None
    return int(keep.sum()), int(len(keep))


def confidence_filter_controls(viewer: Any) -> Any:
    """The results window's lesion filter: a preset and a minimum confidence, applied live."""
    from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

    from nvitk.gui.core.design import COLOR_MUTED, SPACE_TIGHT

    state = getattr(viewer, _STATE_ATTR, None) or {}
    current = float(state.get("min_confidence", 0.0))
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(SPACE_TIGHT)
    preset = QComboBox()
    preset.addItems([*CONFIDENCE_PRESETS, "custom"])
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.05)
    spin.setDecimals(2)
    spin.setValue(current)
    spin.setToolTip("Lesions with a lower confidence are hidden in the viewer (points and extents)")
    count = QLabel("")
    count.setStyleSheet(f"color: {COLOR_MUTED};")
    legend = QLabel(" ".join(f'<span style="color:{c}">■</span> {k}' for k, c in LESION_CLASS_COLOURS.items()))
    legend.setToolTip("Row colour: confidence class (high ≥ 0.7, moderate ≥ 0.4, low below)")

    def _sync_preset(value: float) -> None:
        name = next((k for k, v in CONFIDENCE_PRESETS.items() if abs(v - value) < 1e-9), "custom")
        preset.blockSignals(True)
        preset.setCurrentText(name)
        preset.blockSignals(False)

    def _apply(value: float) -> None:
        shown, total = apply_confidence_filter(viewer, float(value))
        count.setText(f"{shown} of {total} shown")
        _sync_preset(float(value))

    def _on_preset(name: str) -> None:
        if name in CONFIDENCE_PRESETS:
            spin.setValue(CONFIDENCE_PRESETS[name])  # -> valueChanged -> _apply

    preset.currentTextChanged.connect(_on_preset)
    spin.valueChanged.connect(_apply)
    lay.addWidget(QLabel("Show in the viewer"))
    lay.addWidget(preset)
    lay.addWidget(QLabel("confidence ≥"))
    lay.addWidget(spin)
    lay.addWidget(count)
    lay.addStretch(1)
    lay.addWidget(legend)
    _apply(current)
    return box


__all__ = [
    "CONFIDENCE_PRESETS",
    "FLAG_TEXT",
    "LESION_CLASS_COLOURS",
    "LESION_EXTENTS_LAYER",
    "LesionRun",
    "add_lesion_extents",
    "apply_confidence_filter",
    "confidence_filter_controls",
    "draw_profile",
    "flag_lines",
    "inspect_lesion",
    "install_lesion_tools",
    "lesion_card",
    "load_lesion_run",
    "visible_lesions",
]
