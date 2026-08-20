# Stats GUI

`nvitk-statsmodels` is a standalone statistical-modeling workbench — mixed models, MMRM,
robust regression, SEM/network models, and mediation analysis over cohort measurement
tables, with cortical and vascular cohort-level plotting. It opens its own `QApplication`
**without** starting napari, deliberately, to avoid paying for the viewer/GPU-context/plugin
scan cost when the work is pure modeling. The same window is also reachable from inside
{doc}`the main GUI <../gui/index>`, via its "Statmodels" dock tab.

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

## Window layout and data flow

`StatmodelsWindow` (`nvitk.gui.panels.statmodels.window`) is laid out in three draggable
rows, per its own module docstring:

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
| **Derived columns** | `transform` (canned function), `expression` (free-form over columns), or `bins` (continuous → labeled groups). |
| **Region combinations** | Row-wise arithmetic across a subject's regions (e.g. `TCBF = RICA + LICA + BASI`), with prefills for standard composites and vessel-network conservation-balance residuals. |
| **Report / export** | A stat-chip strip (n, groups, convergence, AIC/BIC/LLF) over sortable, significance-shaded coefficient tables; `.xlsx` export includes a second **provenance** sheet documenting how the frame was built. |
| **DB publish** | Upserts a derived column back into the dataset as a first-class variable, with a preview-before-write dialog since it's the one action that writes to shared state. |

```{seealso}
Full generated reference:
[`nvitk.gui.panels.statmodels`](../autoapi/nvitk/gui/panels/statmodels/index), and the
underlying modeling library at {doc}`../api/stats`.
```
