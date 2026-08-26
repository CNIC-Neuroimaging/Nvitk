# Pipelines

End-to-end, cohort-scale batch pipelines built on top of the {doc}`Main API Reference
<../api/index>`. Each pipeline runs identically whether dispatched locally or across an SGE
cluster (`--submit local|sge`), with per-subject array-job dispatch and dependency chaining
when running on a cluster.

```{toctree}
:maxdepth: 2
:hidden:

pesa-fat
qvtpy
qvtpy-hemodynamics
qvtpy-morphometrics
qvtpy-autoqc
topbrain
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`flame` PESA-Fat
:link: pesa-fat
:link-type: doc
Whole-body CT/PET and Dixon fat quantification — TotalSegmentator-based segmentation, SUV
and volume measurement, batch QC.
:::

:::{grid-item-card} {octicon}`pulse` QVTPy
:link: qvtpy
:link-type: doc
4D-flow MRI hemodynamics — eICAB segmentation, FSL registration, centerline-based flow
measurement, and TOF morphometrics, in 12 chainable stages.
:::

:::{grid-item-card} {octicon}`beaker` TopBrain
:link: topbrain
:link-type: doc
36-class whole-brain vessel segmentation for the ToPBrain / ToPAneu challenges —
self-supervised pre-training with nnssl, nnU-Net fine-tuning with selectable losses, and a
Grand Challenge submission container.
:::
::::

```{note}
This section currently covers PESA-Fat, QVTPy and TopBrain. Sibling cohort pipelines already exist in
the codebase (`nvitk-bbtpy`, and a GPETPy tool referenced from the {doc}`Main GUI
<../gui/index>`'s Pipelines category) and are planned for future documentation passes.
```
