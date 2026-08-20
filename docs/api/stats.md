# Statistics

`nvitk.stats` is the statistical modeling engine behind {doc}`the Stats GUI
<../stats-gui/index>` — every model type reachable from `nvitk-statsmodels` is a plain
Python function here first, importable and scriptable independently of the GUI.

| Module | Purpose |
|---|---|
| `mixedlm` | statsmodels-based linear mixed-effects models. |
| `r_mixedlm`, `r_gam`, `r_mmrm`, `r_robust` | R/rpy2-backed alternates — `lme4` mixed models, GAMs, MMRM, and robust (`lmrob`) regression. Import-safe without R installed; availability is probed lazily at call time via each module's `r_backend_status()`. |
| `sem` | Structural equation / path modeling (also R/`semopy`-backed). |
| `mediation` | X→M→Y mediation analysis with a cluster-aware bootstrap. |
| `regression` | General regression utilities. |
| `frame_ops`, `_statmodels_frames`, `_hemodynamic_frames`, `_model_values` | Dataframe-shaping utilities that turn raw measurement tables into model-ready frames — includes the `TRANSFORMS` registry used by the Stats GUI's derived-column editor. |
| `region_algebra`, `region_groups` | Row-wise arithmetic across anatomical regions (e.g. composite flow variables) and named region groupings. |
| `qc_filters` | Filtering rules applied before fitting. |
| `brain_map`, `vascular_map`, `vessel_network`, `_vessel_territory_map` | Cohort-level spatial plotting — cortical-parcel and circle-of-Willis schematic maps. |
| `distribution_plots`, `violin_hemodynamics` | Distribution and violin-plot helpers. |
| `interactive`, `interactive_adapters` | Plotly-based interactive figures (forest/matrix/network plots) shared with the Stats GUI. |
| `summaries` | Model-summary formatting. |

```{note}
The R-backed modules (`r_mixedlm`, `r_gam`, `r_mmrm`, `r_robust`, `sem`) never import R at
module load time — only when a function that actually needs it is called. Their
*documented* functionality still requires `rpy2` and the corresponding R packages / `pymer4`
/ `semopy` at runtime (all included in the conda `nvitk[all]` install and the pixi `stats`
feature).
```

```{seealso}
Full generated reference: [`nvitk.stats`](../autoapi/nvitk/stats/index). For the interactive
workflow built on top of this module, see {doc}`../stats-gui/index`.
```
