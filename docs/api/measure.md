# Measure

`nvitk.measure` exposes every quantitative metric in the toolkit through two complementary
APIs: standalone functional primitives, and a `Measurer` orchestrator for chaining several
measurements over the same image/mask pair.

```{code-block} python
from nvitk.measure import volume_cc, masked_stats, Measurer

vol = volume_cc(mask)
stats = masked_stats(pet, mask, stats=("mean", "max"))
summary = Measurer(pet, mask).volume() | Measurer(pet, mask).suv(kinds=("bw",))
```

| Module | Purpose |
|---|---|
| `volume` | Voxel/physical volume. |
| `intensity` | Masked intensity statistics. |
| `suv` | Standardized Uptake Value (PET) computation. |
| `voxel` | Voxel-overlap metrics — Dice, Jaccard. |
| `surface` | Surface-based metrics (Hausdorff distance, surface area). |
| `radiomics` | Radiomics feature extraction (PyRadiomics-backed). |
| `compare` | Paired-image comparison utilities. |
| `hemodynamics`, `mask_hemodynamics` | Flow/pressure-derived hemodynamic indices used by the {doc}`QVTPy pipeline <../pipelines/qvtpy>`. |
| `cross_section` | Oblique-plane cross-section extraction — the shared primitive behind QVTPy's per-station flow measurements. |
| `measurer` | The `Measurer` orchestrator class itself. |

## Vascular morphometrics (`nvitk.measure.morpho`)

A large, dedicated subsystem for Circle-of-Willis-style vascular morphometrics (caliber,
tortuosity, stenosis, tree topology) — the engine behind {doc}`the QVTPy pipeline's stage 7
<../pipelines/qvtpy>` and the standalone TOF morphometrics tools. Key pieces:

| Module | Purpose |
|---|---|
| `anatomy`, `anatomy_axes`, `labels_util` | Anatomical region/label definitions. |
| `centerlines`, `skeleton` | Centerline/skeleton extraction feeding the morphometrics. |
| `caliber`, `metrics` | Per-segment caliber and derived morphometric measures. |
| `tree_regions`, `tree_segments`, `topology_io` | Vessel-tree topology construction and I/O. |
| `donut_loops`, `geometry`, `surface` | Cross-section geometry helpers. |
| `orchestration`, `run_case` | Full-case orchestration entry points. |
| `export_utils/` | Tortuosity metric export, radius histograms, PVSM scene generation, summary tables. |
| `topology/*.json` | Per-cohort topology definitions (eICAB, mouse-root, QVTPy). |

```{seealso}
Full generated reference: [`nvitk.measure`](../autoapi/nvitk/measure/index).
```
