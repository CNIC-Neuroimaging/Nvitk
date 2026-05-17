# PESA-Brain

Independent pipelines under `nvitk.pipes.pesa_brain` (no qvtpy / pesa_fat imports).

## Config (`black_blood/config.py`)

| Key | Purpose |
|-----|---------|
| `DEFAULT_DICOM_ROOT` | XNAT download target |
| `DEFAULT_NIFTI_ROOT` | `BlackBlood/vwi_bb.nii.gz` per subject |
| `DEFAULT_QVTPY_RESULTS_ROOT` | eICAB outputs (`<subject>/eicab/`) |
| `DEFAULT_RESULTS_ROOT` | `pesa_brain` pipeline outputs |
| `VWI_BB_REL_PATH` | `BlackBlood/vwi_bb.nii.gz` |

TOF and eICAB use the **qvtpy** layout under the same PESA-Brain data tree (see `notebooks/reports/qvtpy/methods.md`).

## BrainVIEW download (stage0_d)

XNAT series `csAI_3D_BrainVIEW_T1W` with variant **strong** > **default** > **weak** (one scan per subject). If `strong` is missing, a warning is logged and the next available variant is used.

## Example commands

```bash
# Full chain with XNAT download
nvitk-pesa-brain --pipeline black_blood --subjects PESA001 \
  --with-download --stages stage0_d,stage0_c,stage1,stage2 \
  --seg-strategy crop-resegment

# Convert + register + segment (DICOM already local)
nvitk-pesa-brain-bb-convert --subject PESA001
nvitk-pesa-brain-bb-reg --subject PESA001
nvitk-pesa-brain-bb-seg --subject PESA001 --seg-strategy centerline-growth
```

`g_pet` is a stub (`NotImplementedError`).
