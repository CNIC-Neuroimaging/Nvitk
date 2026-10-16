# Visualization

`nvitk.viz` holds standalone visualization helpers used by both the pipelines and the GUI's
Viz dock (`nvitk.gui.viz`, documented in {doc}`../gui/index`).

| Module | Purpose |
|---|---|
| `atlas_sources` | Brain/vascular atlas loading for overlay rendering. |
| `brainshow` | Brain-surface/parcellation display. |
| `flowshow` | 4D-flow interactive viewer — the engine behind `nvitk-qvtpy-flowshow`. Streamlines and pathlines are traced by VMTK ({doc}`vmtk`); PyVista only renders them. |
| `centerline_pick` | Interactive centerline point picking. |
| `pet_hotspots` | PET/Dixon hotspot visualization — the engine behind PESA-Fat's `nvitk-pesa-fat-hotspot`. |

```{seealso}
Full generated reference: [`nvitk.viz`](../autoapi/nvitk/viz/index).
```
