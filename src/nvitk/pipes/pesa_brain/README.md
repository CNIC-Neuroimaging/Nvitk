# PESA-Brain

Independent pipelines under `nvitk.pipes.pesa_brain` (no qvtpy / pesa_fat imports).

## CLI entry points

| Command | Purpose |
|---------|---------|
| `nvitk-pesa-brain` | Master dispatcher (`--pipeline black_blood` or `g_pet`) |
| `nvitk-pesa-brain-bb` | Black-blood: all stages via `--stages` |
| `nvitk-pesa-brain-gpet` | g_pet stub (not implemented) |

Per-stage commands (`nvitk-pesa-brain-bb-reg`, etc.) are **not** exposed; use `--stages` on `nvitk-pesa-brain-bb`.

### Black-blood stages (`--stages`)

| Stage | Aliases | Description |
|-------|---------|-------------|
| `stage0_d` | `download` | XNAT → DICOM (`vwi_bb/`) |
| `stage0_c` | `convert`, `stage0` | DICOM → `BlackBlood/vwi_bb.nii.gz` |
| `stage1` | `reg`, `registration` | eICAB TOF_resampled → **vwi_bb** (FLIRT); segmentation in BB space |
| `stage2` | `seg`, `segmentation` | eICAB centerlines warped to vwi_bb + segmentation on native vwi_bb |

Default: `stage0_c,stage1,stage2`. Add `stage0_d` or use `--with-download`.

### eICAB mask (`--eicab-mask {cw,wb}`)

Stage 2 uses the requested eICAB Circle-of-Willis (`cw`) or whole-brain (`wb`) multilabel under `{qvtpy_results}/{subject}/eicab/`. If the requested mask is missing, the other is used with a warning (same policy as qvtpy).

## Config (`black_blood/config.py`)

- `DEFAULT_DICOM_ROOT`, `DEFAULT_NIFTI_ROOT`, `DEFAULT_RESULTS_ROOT`
- `DEFAULT_QVTPY_RESULTS_ROOT` — eICAB / TOF_resampled from qvtpy layout
- `VWI_BB_REL_PATH` — `BlackBlood/vwi_bb.nii.gz`

## Example

```bash
nvitk-pesa-brain-bb --subjects PESA001 \
  --with-download --stages stage0_d,stage0_c,stage1,stage2 \
  --eicab-mask cw --seg-strategy crop-resegment

nvitk-pesa-brain --pipeline black_blood --subjects PESA001 \
  --stages stage1,stage2 --eicab-mask wb
```
