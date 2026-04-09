# Study Data Guide

This repository now includes a local hybrid study database under `dataset/` and a Python API under `src/nvitk/db`.

For the full architecture and methodology reference, see `DATABASE_METHODOLOGY.md`.
For the stepwise Excel import workflow, see `TABLE_BY_TABLE_IMPORT_GUIDE.md`.

## Layout
- `dataset/catalog/repository.json`: dataset root manifest
- `dataset/catalog/tables.json`: table registry and schema hints
- `dataset/catalog/variables.json`: variable dictionary, aliases, and imported codebook metadata
- `dataset/tables/*.parquet`: canonical local tables
- `dataset/cache/index.sqlite`: optional generated query/index cache

## Main Python API
```python
from nvitk.db import DataRepo

repo = DataRepo("dataset")

clinical = repo.clinical(
    variables=["BPXSYM", "BMI"],
    filters={"subject_uid": ["PESA001", "PESA002"]},
    wide=True,
)

hemo = repo.image(
    modality="4dflow",
    variables=["flow_mean"],
    regions=["ICA_L", "ICA_R"],
    wide=True,
)

report_df = repo.join([clinical, hemo], on="subject_uid")
```

## Filters
Simple filters are supported for both Parquet-backed and SQLite-backed queries:

```python
repo.get(
    "clinical_measurements",
    filters={
        "subject_uid": "PESA001",
        "value_num": {"$ge": 120, "$lt": 140},
    },
)
```

Supported operators:
- scalar value: equality
- list/tuple/set: `IN`
- `{"$ge": ...}`, `{"$gt": ...}`, `{"$le": ...}`, `{"$lt": ...}`
- `{"$ne": ...}`, `{"$contains": ...}`, `{"$not_null": True}`

## Import Current Curated Excel Tables
If the current PESA-Brain Excel files live in an external directory:

```bash
PYTHONPATH=src nvitk-import-study-data --dataset-root dataset --db-base-path /path/to/PESA-Brain/DB --build-sqlite-index
```

Or from Python:

```python
from nvitk.db import DataRepo
from nvitk.db.importers import import_pesabrain_curated_tables

repo = DataRepo("dataset", auto_scaffold=True)
import_pesabrain_curated_tables(repo, "/path/to/PESA-Brain/DB", build_sqlite_index=True)
```

The importer also refreshes:
- `dataset/catalog/variables.json` from variable dictionaries and codebooks
- `source_tables` with one row per imported workbook/sheet/import-role

## Build Or Refresh The SQLite Cache
```bash
PYTHONPATH=src nvitk-build-sqlite-index --dataset-root dataset
```

Use it from Python:

```python
repo = DataRepo("dataset", use_sqlite=True)
```

## Sync XNAT Metadata And Optional DICOM Cache
Install the XNAT extra if needed:

```bash
pip install -e ".[xnat]"
```

Then sync:

```bash
PYTHONPATH=src nvitk-sync-xnat \
  --dataset-root dataset \
  --server https://xnat.example.org \
  --project PESA_Brain \
  --catalog-path /path/to/subject_catalog.csv \
  --download-root /path/to/local/DICOM \
  --download-dicoms \
  --build-sqlite-index
```

The sync updates:
- `subjects`
- `subject_ids`
- `sessions`
- `scans`
- `assets`

## 4DFlow Phase To Volume Conversion
Run on one patient:

```bash
PYTHONPATH=src phase2volume --input /path/to/PESA001 --dataset-root dataset
```

Or on a folder of patient directories:

```bash
PYTHONPATH=src phase2volume --input /path/to/NIFTI --multifile --dataset-root dataset
```

This generates:
- `Angiography_4D.nii` and `Angiography_3D.nii`
- `ComplexDifference_4D.nii` and `ComplexDifference_3D.nii`
- `VelocityMagnitude_4D.nii` and `VelocityMagnitude_3D.nii`

When `--dataset-root` is provided, the outputs are also registered into `dataset/tables/assets.parquet`.

## Notebook Migration Pattern
Replace direct Excel loads like:

```python
db_clinical = pd.read_excel(db_base_path / "PESABrain_Clinical_20260216.xlsx")
```

with:

```python
from nvitk.db import DataRepo

repo = DataRepo("dataset")
db_clinical = repo.clinical(wide=True)
```

and replace notebook-local merges with reusable `repo.clinical(...)`, `repo.image(...)`, and `repo.join(...)`.
