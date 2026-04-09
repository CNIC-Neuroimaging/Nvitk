# Table-By-Table Import Guide

This guide explains how to import the current PESA-Brain Excel files into the local `nvitk` study database step by step using the Python API.

It assumes:

- the canonical dataset root is `dataset/`
- the Excel directory is `/data_local/LabVF/PESA-Brain/DB/DB`
- you want to import source by source, not all at once

## 1. Current reset state

The dataset has been reset to an empty scaffold:

- `dataset/tables/*.parquet` has been removed
- `dataset/cache/index.sqlite` has been removed
- `dataset/catalog/variables.json` is empty again
- `dataset/catalog/tables.json` contains only schema definitions, not imported row counts or provenance

That means the next imports will repopulate the dataset from scratch.

## 2. The Python API to use

The clean table-by-table API is:

```python
from pathlib import Path

import pandas as pd

from nvitk.db import DataRepo, import_pesabrain_source, list_pesabrain_sources
```

Initialize the repo:

```python
db_base = Path("/data_local/LabVF/PESA-Brain/DB/DB")
repo = DataRepo("dataset", auto_scaffold=True)
```

See all configured importable workbook roles:

```python
pd.DataFrame(list_pesabrain_sources())
```

Each importable unit is defined by:

- `filename`
- `sheet`
- `source_kind`

The import call is:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_Clinical_20260216.xlsx",
    sheet="Sheet1",
    source_kind="clinical_wide",
)
```

By default this does all of the following:

1. imports that one configured source
2. updates the canonical table or tables
3. updates `source_tables`
4. updates `variables.json` if the source contains measurements or dictionaries
5. rebuilds `subjects`

## 3. Files that change during imports

For any stepwise import, the possible updated files are:

- `dataset/tables/subject_ids.parquet`
- `dataset/tables/sessions.parquet`
- `dataset/tables/scans.parquet`
- `dataset/tables/clinical_measurements.parquet`
- `dataset/tables/image_measurements.parquet`
- `dataset/tables/assets.parquet`
- `dataset/tables/cohort_membership.parquet`
- `dataset/tables/source_tables.parquet`
- `dataset/tables/subjects.parquet`
- `dataset/catalog/tables.json`
- `dataset/catalog/variables.json`

`tables.json` changes when a table is written because `DataRepo.write_table()` refreshes:

- `columns`
- `row_count`
- `last_updated`
- `provenance`

`variables.json` changes when a source introduces:

- observed variables from measurements
- metadata from variable dictionaries
- categorical choices from dropdown/codebook sheets

## 4. Recommended import order

The safest order is:

1. import identity and cohort files
2. import subject/session inventory
3. import clinical measurement files
4. import image measurement files
5. import dictionary/codebook files
6. optionally build SQLite at the end

That order keeps `subject_ids`, `sessions`, and `subjects` informative early.

## 5. Step-by-step imports

## 5.1 Subject ID mapping

Workbook:

- `PESABrain_All_IDs.xlsx`

Call:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_All_IDs.xlsx",
    sheet="Sheet1",
    source_kind="subject_ids",
)
```

Writes:

- `subject_ids`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates `subject_ids`, `source_tables`, and `subjects`
- `variables.json` does not change

What it means:

- this establishes the core mapping between `patient_id`, `mri_id`, `seqn`, and the internal `subject_uid`

## 5.2 4DFlow availability and extra IDs

Workbook:

- `PESABrain_All_4DFlow_IDs.xlsx`

This workbook has two roles on the same sheet.

First import it as extra IDs:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_All_4DFlow_IDs.xlsx",
    sheet="Sheet1",
    source_kind="subject_ids",
)
```

Then import it as cohort membership:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_All_4DFlow_IDs.xlsx",
    sheet="Sheet1",
    source_kind="cohort",
)
```

Writes:

- `subject_ids`
- `cohort_membership`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates the touched tables
- `variables.json` still does not change

What it means:

- `subject_ids` gains more identifier rows
- `cohort_membership` gains the `4dflow_available` cohort

## 5.3 Subject catalog / local session inventory

Workbook:

- `PESABrain_SubjectCatalog_AllXNAT_20260216.xlsx`

Call:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_SubjectCatalog_AllXNAT_20260216.xlsx",
    sheet="Datos",
    source_kind="subject_catalog",
)
```

Writes:

- `subject_ids`
- `sessions`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates `subject_ids`, `sessions`, `source_tables`, and `subjects`
- `variables.json` does not change

What it means:

- the database now knows about MR sessions even before XNAT sync

## 5.4 Main clinical table

Workbook:

- `PESABrain_Clinical_20260216.xlsx`

Call:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_Clinical_20260216.xlsx",
    sheet="Sheet1",
    source_kind="clinical_wide",
)
```

Writes:

- `clinical_measurements`
- `subject_ids`
- maybe `sessions` if a session-like column exists
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates every touched table
- `variables.json` gets one registry entry per imported clinical variable

What it means:

- each non-ID column becomes long-form rows in `clinical_measurements`
- each imported variable becomes discoverable by alias through `DataRepo.clinical()`

## 5.5 Clinical all-XNAT table

Workbook:

- `PESABrain_Clinical_AllXNAT_20260216.xlsx`

Call:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_Clinical_AllXNAT_20260216.xlsx",
    sheet="Datos",
    source_kind="clinical_wide",
)
```

Writes:

- `clinical_measurements`
- `subject_ids`
- `sessions`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates for all touched tables
- `variables.json` gains or merges variable entries

What it means:

- this source may add visit/session/date-linked clinical information and enrich the subject/session mapping

## 5.6 APOE, TAC, and plaque workbooks

Workbooks:

- `PESABrain_APOE_20260318.xlsx`
- `PESABrain_TAC_20260318.xlsx`
- `PESABrain_Echography_CarotidePlaque_20260216.xlsx`

Calls:

```python
import_pesabrain_source(repo, db_base, "PESABrain_APOE_20260318.xlsx", sheet="Sheet1", source_kind="clinical_wide")
import_pesabrain_source(repo, db_base, "PESABrain_TAC_20260318.xlsx", sheet="Sheet1", source_kind="clinical_wide")
import_pesabrain_source(repo, db_base, "PESABrain_Echography_CarotidePlaque_20260216.xlsx", sheet="Sheet1", source_kind="clinical_wide")
```

Writes:

- `clinical_measurements`
- `subject_ids`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates for those tables
- `variables.json` gains or merges `apoe`, calcium score, plaque-volume, and related variable definitions

## 5.7 4DFlow summary image tables

Workbooks:

- `PESABrain_4DFlow_LocalizedPI_20260216.xlsx`
- `PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx`

Calls:

```python
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedPI_20260216.xlsx", sheet="PESABrain_AnalysisDB_Batch1", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx", sheet="PESABrain_AnalysisDB_Batch1", source_kind="image_wide")
```

Writes:

- `image_measurements`
- `subject_ids`
- `sessions`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates the touched tables
- `variables.json` gains or merges image variable definitions such as `pi`, `flow_mean`, and `tcbf`

What it means:

- these are regional summary image measurements, not time series

## 5.8 4DFlow time-series tables

Workbooks:

- `PESABrain_4DFlow_LocalizedTimeseriesFlow_20260216.xlsx`
- `PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx`

Calls:

```python
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedTimeseriesFlow_20260216.xlsx", sheet="Datos", source_kind="image_timeseries_long")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx", sheet="Datos", source_kind="image_timeseries_wide")
```

Writes:

- `image_measurements`
- `subject_ids`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates `image_measurements`, `source_tables`, and `subjects`
- `variables.json` gains or merges entries such as `flow_tseries` and `phase_fraction`

What it means:

- these rows use `frame_index` to store time-resolved vessel data

## 5.9 ASL perfusion tables

Workbooks:

- `PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx`
- `PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx`

Calls:

```python
import_pesabrain_source(repo, db_base, "PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx", sheet="Sheet1", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx", sheet="Sheet1", source_kind="image_wide")
```

Writes:

- `image_measurements`
- `subject_ids`
- `sessions`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates the touched tables
- `variables.json` gains or merges ASL variable entries, mainly `mean_cbf`

## 5.10 Hybrid hemodynamic workbook

Workbook:

- `PESABrain_LocHemodynamic_20260406.xlsx`

Call:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_LocHemodynamic_20260406.xlsx",
    sheet="Sheet1",
    source_kind="hybrid_hemodynamic",
)
```

Writes:

- `clinical_measurements`
- `image_measurements`
- `subject_ids`
- `sessions`
- `source_tables`
- `subjects`

Catalog updates:

- `tables.json` updates all touched tables
- `variables.json` gains or merges both clinical and image variable entries

What it means:

- this is the most mixed workbook
- the parser splits vessel metrics into image measurements, ASL-like regional values into image measurements, and clinical covariates into clinical measurements

## 5.11 Variable dictionary workbook

Workbook:

- `PESABrain_Variables_20250312.xlsx`

Call:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_Variables_20250312.xlsx",
    sheet="Variables",
    source_kind="variable_dictionary",
)
```

Writes:

- `source_tables`
- `subjects` only because the helper rebuilds it

Catalog updates:

- `variables.json` is the important change here
- `tables.json` updates `source_tables` and `subjects`

What it means:

- this does not create measurement rows
- it enriches `variables.json` with descriptions, units, codebook names, missingness flags, and related metadata

## 5.12 Anatomical codebook workbook

Workbook:

- `PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx`

This workbook has four distinct roles.

Clinical-style variable dictionary:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx",
    sheet="Tests Cognitivos",
    source_kind="variable_dictionary",
)
```

Image-style variable dictionary:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx",
    sheet="Neuroimagen",
    source_kind="variable_dictionary",
)
```

Observed case-level neuroimage report table:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx",
    sheet="Casos",
    source_kind="image_wide",
)
```

Dropdown / categorical value dictionary:

```python
import_pesabrain_source(
    repo,
    db_base,
    "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx",
    sheet="DESPLEGABLES",
    source_kind="dropdown_dictionary",
)
```

Writes:

- `image_measurements` for `Casos`
- `source_tables`
- `subjects`

Catalog updates:

- `variables.json` is heavily enriched by the dictionary and dropdown sheets
- `tables.json` updates touched tables and the source inventory

What it means:

- the dictionaries define semantics
- the `Casos` sheet provides actual observed neuroimaging report values
- the dropdown sheet fills categorical allowed values like `0: No`, `1: ICAr`, etc.

## 6. How to inspect the result after each step

Useful checks:

```python
repo.get("subject_ids").head()
repo.get("clinical_measurements").head()
repo.get("image_measurements").head()
repo.get("source_tables").sort_values(["source_file", "source_sheet", "source_kind"])
```

Inspect variables:

```python
clinical_vars = repo.catalog.variable_entries(domain="clinical")
image_vars = repo.catalog.variable_entries(domain="image")
len(clinical_vars), len(image_vars)
```

Inspect a specific variable:

```python
variables = {entry["variable_id"]: entry for entry in repo.catalog.variable_entries()}
variables["age_at_mri"]
variables["aneurysm"]
```

## 7. Optional SQLite build

Do this at the end, not after every workbook, unless you really need it.

```python
repo.build_sqlite_index()
```

This creates:

- `dataset/cache/index.sqlite`

It does not change canonical data. It is only a query cache.

## 8. Full practical script

```python
from pathlib import Path

from nvitk.db import DataRepo, import_pesabrain_source

db_base = Path("/data_local/LabVF/PESA-Brain/DB/DB")
repo = DataRepo("dataset", auto_scaffold=True)

import_pesabrain_source(repo, db_base, "PESABrain_All_IDs.xlsx", sheet="Sheet1", source_kind="subject_ids")
import_pesabrain_source(repo, db_base, "PESABrain_All_4DFlow_IDs.xlsx", sheet="Sheet1", source_kind="subject_ids")
import_pesabrain_source(repo, db_base, "PESABrain_All_4DFlow_IDs.xlsx", sheet="Sheet1", source_kind="cohort")
import_pesabrain_source(repo, db_base, "PESABrain_SubjectCatalog_AllXNAT_20260216.xlsx", sheet="Datos", source_kind="subject_catalog")

import_pesabrain_source(repo, db_base, "PESABrain_Clinical_20260216.xlsx", sheet="Sheet1", source_kind="clinical_wide")
import_pesabrain_source(repo, db_base, "PESABrain_Clinical_AllXNAT_20260216.xlsx", sheet="Datos", source_kind="clinical_wide")
import_pesabrain_source(repo, db_base, "PESABrain_APOE_20260318.xlsx", sheet="Sheet1", source_kind="clinical_wide")
import_pesabrain_source(repo, db_base, "PESABrain_TAC_20260318.xlsx", sheet="Sheet1", source_kind="clinical_wide")
import_pesabrain_source(repo, db_base, "PESABrain_Echography_CarotidePlaque_20260216.xlsx", sheet="Sheet1", source_kind="clinical_wide")

import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedPI_20260216.xlsx", sheet="PESABrain_AnalysisDB_Batch1", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx", sheet="PESABrain_AnalysisDB_Batch1", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedTimeseriesFlow_20260216.xlsx", sheet="Datos", source_kind="image_timeseries_long")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx", sheet="Datos", source_kind="image_timeseries_wide")
import_pesabrain_source(repo, db_base, "PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx", sheet="Sheet1", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx", sheet="Sheet1", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_LocHemodynamic_20260406.xlsx", sheet="Sheet1", source_kind="hybrid_hemodynamic")

import_pesabrain_source(repo, db_base, "PESABrain_Variables_20250312.xlsx", sheet="Variables", source_kind="variable_dictionary")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx", sheet="Tests Cognitivos", source_kind="variable_dictionary")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx", sheet="Neuroimagen", source_kind="variable_dictionary")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx", sheet="Casos", source_kind="image_wide")
import_pesabrain_source(repo, db_base, "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx", sheet="DESPLEGABLES", source_kind="dropdown_dictionary")

repo.build_sqlite_index()
```

## 9. When to use the whole-directory importer instead

If you do not need stepwise control, use:

```python
from nvitk.db import DataRepo, import_pesabrain_db_directory

repo = DataRepo("dataset", auto_scaffold=True)
import_pesabrain_db_directory(repo, db_base, build_sqlite_index=True)
```

Use the whole-directory import when:

- you want the entire DB imported in one call
- you do not need to inspect intermediate states
- you do not need to control the order manually

Use table-by-table import when:

- you want to inspect each stage
- you want to debug one workbook
- you want to verify catalog changes incrementally
