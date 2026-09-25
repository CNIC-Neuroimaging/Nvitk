# Stats GUI

`nvitk-statsmodels` is a standalone statistical-modeling workbench — mixed models, MMRM,
robust regression, SEM/network models, and mediation analysis over cohort measurement
tables, with cortical and vascular cohort-level plotting. It opens its own `QApplication`
**without** starting napari, deliberately, to avoid paying for the viewer/GPU-context/plugin
scan cost when the work is pure modeling. The same window is also reachable from inside
{doc}`the main GUI <../gui/index>`, via its "Statmodels" dock tab.

The window is a tab bar over **sessions** — independent workbenches, each with its own
dataframe and model — so a second hypothesis is a new tab rather than a reload that costs
the first one. See [Sessions](#sessions).

```{code-block} bash
nvitk-statsmodels --dataset /path/to/dataset --kind qvtpy
```

## Command reference

`nvitk-statsmodels` is an **argparse** CLI (unlike the rest of the toolkit, which is
click-based), so its options are listed here directly rather than auto-rendered:

| Option | Purpose |
|---|---|
| `-d`, `--dataset` | Path to the dataset (`DataRepo` root) to load measurements from. |
| `-k`, `--kind` | Pipeline kind to analyze — one of `qvtpy`, `asl`, `t1`, `flair`, `tof` (below). |
| `--load NAME_OR_PATH` | Restore a previously saved model configuration. |
| `--reload` | Force-reload data even if a cached frame exists. |
| `--log-level` | Logging verbosity. |

Model configurations are saved to and loaded from
`<dataset>/nvitk-statmodels/<name>/config.json`.

## Pipeline kinds

| Kind | Cohort data |
|---|---|
| `qvtpy` | 4D-flow hemodynamics (see {doc}`../pipelines/qvtpy`) |
| `asl` | ASL perfusion (CBF/ATT) |
| `t1` | T1 volumetry |
| `flair` | FLAIR white-matter hyperintensities |
| `tof` | eICAB TOF morphometrics |

## Sessions

`StatmodelsShell` (`nvitk.gui.panels.statmodels.sessions`) holds N `StatmodelsWindow` pages in
one tab bar. Each is a complete, independent explorer: its own dataset query, frame recipe,
fitted model, plot and panel layout. Nothing is shared, and nothing is rebuilt on a tab change —
every session is a live widget parked in the stack, so switching is a raise rather than a reload.
That is also why a page is built eagerly: a tab that constructs itself on first click is not one
you can flick between while comparing two fits.

| Action | How |
|---|---|
| New session | the **+** button on the tab bar, or `Ctrl+T` |
| Switch | click the tab, `Ctrl+Tab` / `Ctrl+Shift+Tab`, or `Ctrl+1`…`Ctrl+9` |
| Rename | double-click the tab |
| Close | the tab's ✕ or `Ctrl+W` (it asks first if there is a frame or a fit to lose) |
| Everything else | right-click any tab |

### Floating panels

Popping a panel out — the ⧉ button on the *Analysis dataframe* or *Results* title bar, a
double-click on it, or dragging the dock free — makes it a top-level window rather than a widget
inside the page. Qt does not take those down when the shell switches tabs, so they would otherwise
sit over the new session looking like its own panels.

The shell hides them with their session and brings them back, at the same size and position, when
that session is current again. The same applies to the modeless column- and subject-plot dialogs,
and to the shell itself: hiding or closing the window takes every session's detached panels off
screen with it rather than stranding them on the desktop. A panel you closed yourself stays
closed — a round trip through another tab does not bring it back.

### Ordering the levels

The axis order defaults to a natural sort — `g2` before `g10` — which is right for bin groups and
wrong for anything whose meaning is not alphabetical: a severity scale, a vessel sequence
following the circulation, a control group that belongs first. **Order…** on the distribution
window opens a reorder editor for whichever column the *by* picker is on: the x-axis groups when
split, the panels when panelled.

Drag or use the arrows, or take one of the computed orders — *Natural*, *A–Z*, *By median*
(ascending median of the plotted column), *By count* (most observations first), *Reverse*. The
button reads **Order ✓** while an order is in force, and each column keeps its own, so switching
the split and coming back does not lose it.

This applies to the counts view too. A categorical column is not drawn as a distribution but as
bar counts of its own levels, and those bars, their legend and the status line all follow the
chosen order. Without one they follow the order the column was **cut** in — a binned column
carries its own categories — rather than the order its rows happen to arrive in.

An order outlives the frame it was chosen on, so it selects and ranks rather than dictating: a
level a filter has since removed is skipped, and one that has since appeared is drawn after the
ordered set rather than silently dropped.

### Legend

Both the distribution window and the model plots carry a **Legend** toggle. It acts on the figure
in place — no refit, no redraw of the data — and survives the redraws that every other control
triggers. The violin, box and strip views name their levels on the axis instead, so there is no
legend to hide there.

### What can be a hue or a split

Every picker that asks "group by which column" — the distribution window's *by* and sub-split, the
model plot's *colour by*, the summary grouping, the reference-level menu, the SEM grouping — offers
a column on how many **distinct values** it has, not on its dtype. A factor is a factor whether it
is spelled `"M"`/`"F"` or `0`/`1`, and deciding by dtype is what kept every numerically coded one
out: a single missing value upcasts an integer `sex` to float64.

A continuous measurement is excluded by the level cap rather than by being numeric. Floats get a
tighter test, since dtype cannot separate a coded factor from a measurement: at most
`MAX_FLOAT_GROUP_LEVELS` (12) distinct values, **and** values that repeat — a measurement's do not,
which is what keeps it out of a frame filtered down to a handful of rows.

Levels are labelled through `nvitk.stats.group_counts.level_strings`, so a whole float reads as
`0` / `1` rather than `0.0` / `1.0`, and rows with no value stay missing instead of drawing as a
`nan` level of their own.

### Taking another session's dataframe

Right-clicking a session *other* than the current one offers to pull its dataframe across. The
submenu is named after the **destination** (`Import into "Session 2"`), so which way the frame
travels is not something to work out from which tab happened to be right-clicked.

| Menu entry | What arrives | When you want it |
|---|---|---|
| **Dataframe + transformations** | The frame *and* the recipe that built it — measurements, covariate picks, region combinations, derived columns, casts and reference levels, filters, melt/wide — all still editable | Fitting a second model to one frame. Pressing *Reload data* here re-runs the same query. |
| **Dataframe as new raw data** | The finished rows only, adopted as this session's raw input, with the recipe spent rather than carried | Building a *new* set of transformations on top of another session's output |
| **Duplicate into a new session** | The whole session — frame recipe plus formula, engine, plot options and panel layout — in a fresh tab | A copy to diverge from |

The second mode exists because a recipe replayed over its own output is not idempotent: a
row-producing region combination would append a second copy of every synthetic row, and a melt
would find no `flow_mean__LICA` columns left to melt. Carrying the recipe and spending it are
therefore two separate offers rather than one guess.

The **fit never crosses over**, in any mode. A result belongs to the frame it was fitted to, and
one left sitting under a frame it did not come from reads as live when it is not — press *Fit* to
reproduce it. The destination's model settings and plot *are* kept by the two import modes, which
is the point: same data, different model.

## Group sizes

A distribution says nothing without the N behind it: a territory drawn from eleven observations
looks exactly as solid as one drawn from ninety, and a level that quietly lost half its rows to a
QC filter has no way to say so. So every level of a column distribution carries its own count,
wherever that level appears:

| Where | Looks like |
|---|---|
| Categorical kinds (violin, box, strip) | under each x tick — `LICA` / `(n=21 +2 excl)` |
| Overlaid kinds (histogram, density, ECDF) | in the legend — `LICA (n=21 +2 excl)` |
| Panels | in each panel's heading — `LICA  (n=21 +2 excl)`, with the within-panel split labelled too |
| The whole display | the dialog's status line, e.g. `n = 52 +5 excl \| territory: LICA n=21 +2 excl · RICA n=19 · BASI n=12 +3 excl` |

`n` is the number of observations **kept** — rows that have a value for the column and that the
active filters did not remove. `+k excl` is rows drawn greyed out by *Grey excluded*: present on
the figure, absent from the analysis. Untick *Grey excluded* and they stop being drawn and stop
being counted, so the label always describes what is actually on screen. Rows with no value for
the column are never counted, since nothing is drawn for them.

The pooled `n` above the axes is the sum of the per-level counts, and the descriptive statistics
beside it (mean, SD, median, IQR) are computed over those same kept rows.

Both backends count through `nvitk.stats.group_counts`, which is what stops the Matplotlib figure
and the Plotly one from disagreeing about what they are showing.

## Significance brackets

The **Signif.** toggle, beside *Points* and *95% CI*, draws pairwise significance over the levels
on a model plot's categorical axis. Two modes:

| Mode | Draws |
|---|---|
| `all` | every comparison, `***` `**` `*` `.` `NS`, and `NA` for one with no p-value |
| `significant only` | just those with an adjusted p below 0.05 |

These are **level-versus-level** comparisons, not the coefficient table's contrasts against the
reference level: "is LICA different from RICA" is what a grouped plot is read for, and no
coefficient answers it. Each is a linear combination of the fitted coefficients,

```{code-block} text
estimate = c'β        se = √(c' V c)        t = estimate / se
```

evaluated with the other covariates held at their reference values — so the comparison is
adjusted, not a difference of raw means. That needs nothing from the model but its coefficients
and their covariance, which is why one implementation serves MixedLM, OLS and GLM; the R engines
have no patsy design matrix, so they go through `emmeans::pairs` instead.

### Multiplicity

Thirteen vessels are seventy-eight comparisons, and at α = 0.05 four of those are expected to
look significant with no effect present at all. p-values are therefore **Holm-adjusted**, in
Python for both backends rather than by each engine's own default, so a bracket means the same
thing whichever engine drew it. The correction is over the comparisons actually computed — untick
half the levels in the *Groups* list and the rest stop carrying their penalty.

### Correcting, or not

A regression table corrects nothing — it prints seventy raw p-values — so a corrected figure and
a coefficient table can never agree. The **adjust** selector settles which you are looking at:

| Setting | Shows |
|---|---|
| `Holm` (default) | corrected within each series — the responsible default |
| `raw p` | the uncorrected p of each comparison, exactly what the table prints |

The gap is not small. Six comparisons per territory turn a table's `p = 0.0145 *` into
`p = 0.087` — no longer significant, and invisible under *significant only*. Pick `raw p` to read
a model's interactions off the figure as the table reports them; the status line then says
**uncorrected (raw p, as the coefficient table)** rather than *Holm-adjusted*, because the two
figures look identical otherwise.

An uncorrected bracket is a real result about one comparison. It misleads only if read as though
every comparison on the figure had been tested at that level — which is what the correction
exists to prevent, and why it stays the default.

**What counts as one family** (when correcting) is the series, not the figure. When the plot splits by a second
factor, each of its levels is corrected on its own — `emmeans`' convention for a `by`
specification, and how a per-curve bracket is read. Pooling instead makes seventeen territories a
single family of 102 tests, where a real effect at raw p = 0.003 adjusts to 0.30: every bracket
reads `NS` and *significant only* draws nothing, which says more about the denominator than about
the data.

### What a bracket is testing

A bracket and an interaction coefficient answer different questions, so the **basis** selector
next to the mode says which one is drawn:

| Basis | A bracket between two plaque levels on RICA asks | Drawn |
|---|---|---|
| `within series` | does g1 differ from g0 **in RICA** — the simple effect | every pair |
| `interaction (table rows)` | does the g0→g1 change in RICA differ **from the reference territory's** | pairs against the factor's reference — one per table row |
| `interaction (all pairs)` | the same, for any two levels | every pair; the extras are differences of two interaction coefficients |

The second is exactly what the coefficient table's `plaque[g1]:territory[RICA]` row reports, and
it is checked against it: estimate, standard error and p agree to 1e-9 on the statsmodels engines
and to the table's own precision on lmrob and MMRM. Pick it to read a model's interactions
straight off the figure.

`interaction (table rows)` gives **exactly one bracket per interaction row** and no more. An
interaction *term* contrasts a level against the factor's own reference, so only pairs touching
that reference are drawn: with four plaque groups and seventeen territories that is 3 × 16 = 48
brackets against the table's 48 interaction rows.

`interaction (all pairs)` lifts that restriction. A g1-versus-g2 bracket is then the *difference
of two* interaction coefficients — a real contrast, but one the table never prints, so there is
nothing to check it against. The two bases carry identical numbers for the pairs they share; the
restriction removes comparisons rather than changing them, and shrinks the correction family with
them (three per series rather than six).

The reference series carries no brackets in that basis — it is the baseline the others are
measured from, so its own contrasts are zero by construction. Which level that is comes from the
fit rather than from a sort order: it is the one the coefficient table is silent about, so the
figure and the table always name the same baseline. The status line says which.

The `within series` basis cannot be read off the table at all: a simple effect's standard error
needs the covariance between the main and interaction terms, which the table does not carry.
Either basis can be significant while the other is not.

### One set of brackets per curve

When the plot colours by a second factor, the comparisons are made **within** each of its levels,
not averaged over them, and each series' brackets are drawn in that series' own colour. On a fit
like `log1p_pi ~ plaque * territory` that is the only defensible reading: the plot draws one
territory per line, and what a bracket over a line can claim is the plaque effect *in that
territory*. The averaged contrast would put one set of brackets over three curves that disagree —
and on a real interaction it does: a plaque effect present only in RICA shows as `***` on RICA's
band and `NS` on the others, where the averaged version dilutes it to nothing.

Each bracket carries the stars and the adjusted p beneath them — the marker to read at a glance,
the number to cite.

The **grouped** display needs no special handling: a panel holding three of thirteen territories
has three legend entries, so only those three series' brackets land on it, in that panel's own
colours.

### Factors the model has no fixed effect for

A contrast is a linear combination of *fixed-effect* coefficients, so a factor the model has none
for cannot be compared. The common case is a random-effects grouping:

```{code-block} text
log1p_pi ~ plaque + age_c + (1 + plaque | territory) + (1 | subject_uid)
```

Nothing about `territory` is in the fixed effects here. What the model has instead is one shrunk
prediction per territory, and a BLUP carries no null hypothesis — there is no p-value to put on a
bracket. Asking `emmeans` anyway answers *No variable named territory in the reference grid*, from
three frames below where the mistake was made, so the check happens up front instead.

* Asked to **compare** such a factor's levels, the plot says so and names the fix: add it as a
  fixed effect.
* Asked to **split by** one — the usual case, since it is the colouring column — the split is
  dropped and the comparison is pooled instead. The brackets that get drawn are correct; the
  status line says they are pooled and why. Refusing the whole annotation because the colouring
  column is not in the formula would take away brackets that were fine.

### What gets drawn

At most a dozen brackets per axes, most significant first; the status line reports how many were
left off. Thirteen levels of brackets would be taller than the figure and would hide the data they
are about. On a **continuous** plot there are no ticks to span — a bracket across a numeric axis
would claim a range of x it does not mean — so the comparisons between the colouring factor's
levels are listed in a corner box instead.

Available on the MixedLM/OLS/GLM, lme4, lmrob and MMRM plots, on both the Matplotlib and the
Plotly backend. Not on the non-linear fit (one curve over all rows, so there are no levels), the
anatomical maps (which already colour by significance) or SEM.

### Where the confidence band sits

The band has to describe the curve it is drawn around, and two different conventions made it
describe something else. The plotted curve holds each **categorical** covariate at its modal
level; `emmeans` averages over every factor outside its specification with equal weights. On a
model with a categorical covariate those are different numbers, so the band came out displaced
from the line by a constant — the modal level's effect minus the mean of them.

The held factors are now named in the specification and the held combination selected, so the
band's centre *is* the predicted curve, exactly. Continuous covariates need no pin: `emmeans`
already holds those at their mean, which is what the reference row does. Per-level bands are
unaffected — each still surrounds its own curve.

### Engines `emmeans` will not take

The R engines get their contrasts *and* their confidence bands from `emmeans`, which declines
some of the models this toolkit fits:

```{code-block} text
Can't handle an object of class "lmrob"
```

`robustbase` registers no `emm_basis` method, so a robust regression would get no marginal means,
no CI band and no brackets. (Two other `emmeans` footguns are handled in passing: `pairs` is an S3
method on `emmGrid` rather than an exported object — `emmeans::pairs` raises — so the exported
`contrast(em, "pairwise")` is used instead; and a pymer4 `lmer` has to be unwrapped to the R model
it holds before rpy2 will take it.) `nvitk.stats.r_basis` rebuilds just the part needed — the reference
grid, `X β` and `√(diag(X V X'))`, including `emmeans`' equal-weight averaging over factors
outside the specification — from the fit's own coefficients and robust covariance. It is used
only when `emmeans` refuses, and it is checked against `emmeans` on models `emmeans` *does*
accept: means, standard errors, intervals and pairwise contrasts all agree to 1e-9.

## Window layout and data flow

`StatmodelsWindow` (`nvitk.gui.panels.statmodels.window`) — one session — is laid out in three
draggable rows, per its own module docstring:

1. **Top** — what data to load (measurement pickers) and what to do with it (a MixedLM
   formula box, or the mediation form).
2. **Middle** — the plot pane and a model-info report, given most of the window's height.
3. **Bottom** — clinical/cognitive covariate pickers and the analysis dataframe table.

Data flows one way and is recomputed from scratch on every reload, so toggling one stage
never compounds on another's output:

```{code-block} text
measurements → analysis_df (raw, never mutated)
             → derived columns
             → filter rules
             → working_df (what actually gets fitted)
```

## Statistical capabilities

| Capability | Notes |
|---|---|
| **Mixed-effects models** | Patsy-style formula box, GLM family selection, and an R/`lme4` backend path. |
| **MMRM** | Mixed-model repeated measures. |
| **Robust regression** | `lmrob`-backed. |
| **Non-linear fits** | Dedicated non-linear model box. |
| **SEM / network models** | Backed by `nvitk.stats.interactive`'s forest/matrix/network plots. |
| **Mediation analysis** | X→M→Y with covariates; the slow path (an `n_boot`-draw cluster bootstrap respecting subject × territory nesting) runs on a cancellable background worker with progress/ETA. |
| **Domain plotting** | Brain-surface / cortical-parcel plots and circle-of-Willis vascular schematic plots. |
| **Pairwise significance** | A *Signif.* toggle on the model plots draws Holm-adjusted level-versus-level brackets \u2014 see [Significance brackets](#significance-brackets). |
| **Column distributions** | Right-click a column → *Plot* for a violin / box / strip / histogram / density / ECDF of it, split or panelled by another column. Every level is labelled with its own N — see [Group sizes](#group-sizes). |
| **Derived columns** | `transform` (canned function), `expression` (free-form over columns), or `bins` (continuous → labeled groups). |
| **Region combinations** | Row-wise arithmetic across a subject's regions (e.g. `TCBF = RICA + LICA + BASI`), with prefills for standard composites and vessel-network conservation-balance residuals. |
| **Report / export** | A stat-chip strip (n, groups, convergence, AIC/BIC/LLF) over sortable, significance-shaded coefficient tables; `.xlsx` export includes a second **provenance** sheet documenting how the frame was built. |
| **DB publish** | Upserts a derived column back into the dataset as a first-class variable, with a preview-before-write dialog since it's the one action that writes to shared state. |

```{seealso}
Full generated reference:
[`nvitk.gui.panels.statmodels`](../autoapi/nvitk/gui/panels/statmodels/index), and the
underlying modeling library at {doc}`../api/stats`.
```
