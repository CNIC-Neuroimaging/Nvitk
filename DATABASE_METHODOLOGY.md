# Database Methodology

This document explains the methodology behind the local study database implemented in `nvitk`, with emphasis on the code under `src/nvitk/db` and the dataset manifests under `dataset/catalog`.

The goal is to make the system understandable as both:

- a data methodology: how the study data is modeled, versioned, and queried
- a code methodology: what each file, class, and important function is responsible for

## 1. Why this database exists

The database layer exists to replace ad hoc use of many versioned Excel files with a repository-managed, code-first data system that is:

- local and simple to inspect
- Git-friendly for manifests and canonical dataset structure
- practical for analytics in Python
- explicit about provenance
- extensible to both local curated tables and XNAT metadata

The central design choice is a hybrid model:

- canonical data lives in local `Parquet` tables under `dataset/tables/`
- canonical metadata lives in JSON manifests under `dataset/catalog/`
- an optional SQLite cache under `dataset/cache/index.sqlite` is generated for faster queries, but it is not the source of truth

This means the authoritative layer is file-based and transparent, while SQL is used only as a disposable acceleration layer.

## 2. Design principles

The implementation follows a small set of rules.

### 2.1 Canonical storage is file-based

The system treats `Parquet` plus JSON manifests as the real dataset. SQLite can be rebuilt at any time from those files.

### 2.2 Measurements are stored in long form

Clinical and image variables are stored in long-form tables rather than wide Excel-like tables. This makes it easier to:

- add new variables without schema churn
- combine sources from different files
- attach provenance per value
- pivot into wide form only when needed for analysis

### 2.3 Internal IDs are stable

The system uses `subject_uid` as the internal subject key, then maps external IDs like `patient_id`, `seqn`, `mri_id`, and XNAT labels through `subject_ids`.

### 2.4 Provenance is first-class

The code keeps track of where rows came from by storing fields such as:

- `source_batch_id`
- `source_file`
- `source_sheet`
- `source_column`
- `source_table`
- `pipeline_name`
- `pipeline_version`

### 2.5 The API is pandas-first

`DataRepo` returns `pandas.DataFrame` objects. The system is optimized for notebooks and analysis scripts, not for building a full database server.

### 2.6 Metadata drives behavior

The manifests in `dataset/catalog` are not passive documentation. They influence:

- table discovery
- wide pivot rules
- dtype coercion
- variable alias resolution
- SQLite index generation

## 3. Repository layout

The relevant pieces are:

```text
dataset/
  README.md
  catalog/
    repository.json
    tables.json
    variables.json
    schema/
      repository.schema.json
      tables.schema.json
      variables.schema.json
  tables/
    *.parquet
  cache/
    index.sqlite

src/nvitk/db/
  __init__.py
  exceptions.py
  storage.py
  filters.py
  catalog.py
  repo.py
  sqlite_index.py
  importers.py
  xnat.py

tests/db/
  test_catalog_and_repo.py
  test_sqlite_index.py
  test_xnat.py
  test_importers.py
```

Conceptually:

- `dataset/catalog/*` describes the dataset
- `dataset/tables/*` stores the dataset
- `src/nvitk/db/*` implements the dataset logic
- `tests/db/*` defines behavior that should keep working

## 4. The hybrid architecture

The full stack has four layers.

### 4.1 Manifest layer

This is the contract layer.

- `dataset/catalog/repository.json` says where the table root, cache root, and manifests live
- `dataset/catalog/tables.json` describes every table, its schema, keys, and pivot behavior
- `dataset/catalog/variables.json` stores the variable dictionary, aliases, and imported metadata

### 4.2 Canonical table layer

This is the authoritative data layer.

- Each table is a `Parquet` file.
- Reads and writes go through `DataRepo` and helpers in `storage.py`.
- New data updates the table manifests so row counts and inferred dtypes stay in sync.

### 4.3 Optional SQLite cache layer

This is the acceleration layer.

- `sqlite_index.py` copies `Parquet` tables into SQLite.
- Indexes are created for key and lookup columns.
- Reads can use SQLite when available.
- If SQLite fails, `DataRepo.get()` falls back to `Parquet`.

### 4.4 Import and sync layer

This is the ingestion layer.

- `importers.py` converts local Excel workbooks into canonical tables
- `xnat.py` synchronizes XNAT subjects, sessions, scans, and optionally DICOM assets

## 5. Core data model

The dataset uses a small number of canonical tables.

### 5.1 `subjects`

Purpose:

- one row per subject known to the repository

Role:

- lightweight entity table derived from the other tables

Typical fields:

- `subject_uid`
- `primary_patient_id`
- `primary_seqn`
- `sex`
- `birth_date`

Methodology:

- `subjects` is not the raw import source for IDs
- it is rebuilt from `subject_ids`, `clinical_measurements`, `image_measurements`, and `sessions`
- this makes it a derived summary table rather than the raw identity table

### 5.2 `subject_ids`

Purpose:

- map one internal subject to many external identifiers

Role:

- identity normalization layer

Typical fields:

- `subject_uid`
- `id_namespace`
- `id_value`
- `id_source`
- `is_primary`

Methodology:

- if a workbook contains `patient_id`, `seqn`, `mri_id`, or related identifier-like fields, they are harvested into this table
- this decouples identity management from the measurement tables

### 5.3 `sessions`

Purpose:

- represent imaging sessions or experiment-level rows

Role:

- bridge between subject-level identity and image-derived rows

Typical fields:

- `session_uid`
- `subject_uid`
- `project_id`
- `experiment_label`
- `modality`
- `acquired_at`

Methodology:

- local workbook imports can produce session rows when an `mri_id` or equivalent exists
- XNAT sync also writes to `sessions`

### 5.4 `scans`

Purpose:

- store scan-level XNAT inventory

Role:

- detailed imaging inventory table

Typical fields:

- `scan_uid`
- `session_uid`
- `subject_uid`
- `scan_id`
- `series_description`
- `modality`
- `orientation`
- `local_cache_path`

Methodology:

- only populated by the XNAT sync layer
- complements `sessions`; it does not replace `image_measurements`

### 5.5 `clinical_measurements`

Purpose:

- store long-form clinical variables

Role:

- main canonical table for clinical values

Typical fields:

- `subject_uid`
- `visit_id`
- `variable_id`
- `value_num`
- `value_text`
- `value_kind`
- `source_file`
- `source_sheet`
- `source_column`
- `measured_at`

Methodology:

- values remain in long form
- a variable can be numeric or text
- provenance fields ensure that the same logical variable imported from different sources can still be audited separately

### 5.6 `image_measurements`

Purpose:

- store long-form image-derived variables

Role:

- main canonical table for image measurements

Typical fields:

- `subject_uid`
- `session_id`
- `modality`
- `region_id`
- `region_label`
- `frame_index`
- `variable_id`
- `value_num`
- `value_text`
- `pipeline_name`
- `source_file`
- `source_sheet`
- `source_column`

Methodology:

- supports both scalar regional measurements and time-series rows
- `frame_index` is used for time-resolved signals such as 4DFlow time series
- `modality` and `region_id` are part of the canonical identity of image-derived values

### 5.7 `assets`

Purpose:

- catalog files associated with the study

Role:

- local file registry

Typical fields:

- `asset_uid`
- `subject_uid`
- `session_uid`
- `modality`
- `asset_type`
- `asset_path`
- `resource_label`
- `source`

Methodology:

- currently used mainly by XNAT DICOM caching
- intended as the natural place to register future derived files from imaging pipelines

### 5.8 `cohort_membership`

Purpose:

- define named subject subsets

Role:

- cohort availability and grouping layer

Typical fields:

- `cohort_id`
- `subject_uid`
- `membership_source`

Methodology:

- used for sets such as "subjects with 4DFlow available"

### 5.9 `source_tables`

Purpose:

- inventory imported workbook sheets and their original column layouts

Role:

- audit and coverage table

Typical fields:

- `source_uid`
- `source_file`
- `source_sheet`
- `source_kind`
- `domain`
- `modality`
- `layout`
- `n_rows`
- `n_columns`
- `columns_json`

Methodology:

- every recognized sheet is registered here
- unrecognized sheets can still be inventoried as `unmapped`
- this table is critical for understanding what was imported and what remains unmapped

## 6. Identity and provenance methodology

These rules are central to the whole implementation.

### 6.1 `subject_uid` is the internal anchor

The importer tries to infer `subject_uid` from likely columns such as:

- `patient_id`
- `seqn`
- `codigoimagen`
- `subject`
- `codi sub.`

This logic lives in `ensure_subject_uid()` in `src/nvitk/db/importers.py`.

If no candidate column exists, the importer can generate fallback row IDs like `row_000001`. That fallback is safe for technical ingestion, but it is not the preferred identity strategy for real study data.

### 6.2 Provenance is stored per row, not just per file

The code adds row-level provenance fields because the same canonical table can receive data from many workbooks and pipelines.

This is why the measurement tables include fields such as:

- `source_table`
- `source_file`
- `source_sheet`
- `source_column`
- `source_batch_id`

### 6.3 Key columns are methodological, not just technical

`tables.json` defines `key_columns` and `index_columns`.

- `key_columns` are used by `DataRepo.upsert_table()` to decide how duplicates are resolved
- `index_columns` are used by the SQLite builder to create SQL indexes

This separation is important:

- key logic expresses canonical identity
- index logic expresses query convenience

## 7. Manifest methodology

The manifest system is implemented mainly in `src/nvitk/db/catalog.py`.

### 7.1 `repository.json`

This is the root manifest.

It tells the system:

- the dataset name
- where tables live
- where the cache lives
- where to find the table manifest
- where to find the variable manifest

`DatasetCatalog.__init__()` loads this first.

### 7.2 `tables.json`

This is the table registry.

Each table entry contains:

- `path`
- `kind`
- `description`
- `key_columns`
- `index_columns`
- optionally `wide_index_columns`
- optionally `wide_key_columns`
- `columns`
- optionally dynamic fields like `row_count`, `last_updated`, and `provenance`

Methodology:

- static structure is seeded from the scaffold
- runtime metadata such as row counts and inferred dtypes are refreshed by `DatasetCatalog.update_table_schema()`

### 7.3 `variables.json`

This is the variable registry.

It stores:

- canonical `variable_id`
- domain
- target table
- aliases
- imported codebook metadata
- units
- descriptions
- allowed values
- source workbook/sheet

Methodology:

- observed variables and dictionary variables are merged into one registry
- alias resolution is used by `DataRepo.clinical()` and `DataRepo.image()`
- richer metadata should not be overwritten by poorer metadata

That is why `DatasetCatalog._merge_variable_entry()` only overwrites fields when the new data is meaningful, and it merges aliases instead of replacing them.

### 7.4 `TableDefinition`

`TableDefinition` is the in-memory representation of each table.

It is a frozen dataclass used by `DataRepo` and `SQLiteIndex` so the manifest is translated into a stable Python object.

### 7.5 Validation

`DatasetCatalog` validates:

- repository manifest required keys
- table manifest structure
- variable manifest structure

This is intentionally lightweight validation. The goal is to catch broken manifests early without turning the system into a full schema framework.

## 8. Storage methodology

The low-level storage helpers live in `src/nvitk/db/storage.py`.

### 8.1 JSON helpers

- `read_json()`
- `write_json()`
- `json_dumps()`

These standardize manifest I/O and keep JSON ASCII-safe and deterministic enough for Git diffs.

### 8.2 Parquet helpers

- `read_parquet_table()`
- `write_parquet_table()`

These isolate the physical storage mechanism from the rest of the code.

### 8.3 Dtype inference and coercion

This is one of the most important implementation details.

- `infer_manifest_dtypes()` records the table dtypes into `tables.json`
- `manifest_dtype_to_pandas()` maps manifest dtype names to pandas dtypes
- `empty_dataframe()` creates correctly typed empty frames when a table does not exist
- `coerce_dataframe_to_manifest()` forces loaded frames to conform to the manifest

Why this matters:

- `Parquet` and SQLite do not always give back the same dtypes
- after a SQLite read, datetime or nullable integer columns can come back as generic objects
- `DataRepo.get()` therefore coerces the result back to the manifest schema

Without that coercion, query behavior would differ depending on whether the SQLite cache was used.

### 8.4 String and bool normalization

- `normalize_string()` converts empty-like values to `None`
- `coerce_bool()` normalizes common truthy inputs
- `ensure_string_columns()` is a convenience helper when schemas need explicit string columns

## 9. Filter methodology

Filtering is implemented in `src/nvitk/db/filters.py`.

The system uses a simple filter DSL so the same filter spec can work for:

- pandas filtering on `Parquet`
- SQL filtering on SQLite

### 9.1 `FilterCondition`

`FilterCondition` is a normalized representation of a single condition:

- column
- operator
- value

### 9.2 Supported operators

The normalized operators are:

- `eq`
- `ne`
- `in`
- `not_in`
- `gt`
- `ge`
- `lt`
- `le`
- `contains`
- `is_null`
- `not_null`

Aliases like `$gte`, `$lte`, `$nin`, and `$notnull` are normalized by `_normalize_op()`.

### 9.3 Normalization flow

`normalize_filters()` converts user input into a canonical internal structure.

Examples:

```python
{"subject_uid": "PESA001"}
{"variable_id": ["bmi", "bpxsym"]}
{"value_num": {"$ge": 120, "$lt": 140}}
{"source_file": {"$contains": "Clinical"}}
```

Methodology:

- scalar values become equality
- sequences become `IN`
- `None` becomes `IS NULL`
- mapping specs become explicit operator conditions

### 9.4 Dual execution path

- `apply_filters()` applies normalized filters to a pandas DataFrame
- `build_sql_where()` compiles the same logic to SQL plus parameters

This duality is what keeps `Parquet` queries and SQLite queries behaviorally aligned.

### 9.5 Utility helpers

- `merge_filters()` combines filter dictionaries
- `ensure_list()` normalizes string-or-list inputs for domain helpers

## 10. Query methodology in `DataRepo`

`src/nvitk/db/repo.py` is the main user-facing API.

`DataRepo` is intentionally small. It is not an ORM. It is a convenience layer over manifests, `Parquet`, optional SQLite, and measurement pivots.

### 10.1 Dataset root resolution

`_default_dataset_root()` resolves the dataset location in this order:

1. `NVITK_DATASET_ROOT` if set
2. otherwise the repository `dataset/` directory

This lets notebooks and scripts share the same API while still allowing overrides.

### 10.2 Constructor

`DataRepo.__init__()` wires together:

- `DatasetCatalog`
- `SQLiteIndex`
- user preference for SQLite

If `auto_scaffold=True` and the dataset does not exist yet, it creates the scaffold before loading it.

### 10.3 `get()`

`get()` is the generic table reader.

It does the following:

1. resolve the table definition from the catalog
2. decide whether to use SQLite or `Parquet`
3. query the selected backend
4. coerce the result to manifest dtypes
5. optionally produce wide form

Methodology:

- SQLite is opportunistic, not mandatory
- failures in SQLite do not break reads because the code falls back to `Parquet`

### 10.4 `clinical()` and `image()`

These are domain-specific helpers.

`clinical()`:

- resolves variable aliases in the clinical domain
- merges those with any user filters
- reads from `clinical_measurements`

`image()`:

- resolves image variable aliases
- optionally filters by `modality`
- optionally filters by `regions`
- reads from `image_measurements`

These methods exist so notebooks can ask for data semantically rather than by raw table mechanics.

### 10.5 `assets()`

Convenience wrapper around `get("assets")`.

### 10.6 `join()`

`join()` performs sequential pandas merges over a list of frames.

Methodology:

- it is intentionally minimal
- the database layer does not try to create a query planner
- complex analysis joins are expected to happen in pandas

### 10.7 `write_table()` and `upsert_table()`

`write_table()`:

- writes a full table to `Parquet`
- updates the table manifest schema
- optionally refreshes SQLite

`upsert_table()`:

- reads the existing table
- concatenates old and new rows
- drops duplicates by `key_columns`
- keeps the last row for each key

Methodology:

- this is a simple last-write-wins merge strategy
- it is good enough for repository-managed tables
- it is not meant to provide transactional database semantics

### 10.8 Wide access

The database stores measurements in long form, but `DataRepo` can return wide form on demand.

The flow is:

1. `_resolve_measurement_values()` creates a combined `value` column from `value_num` and `value_text`
2. `_to_wide()` uses `tables.json` rules to pivot
3. `_compose_wide_keys()` builds column names from the configured key fields

For `clinical_measurements`, wide columns are based on:

- `variable_id`

For `image_measurements`, wide columns are based on:

- `modality`
- `region_id`
- `frame_index`
- `variable_id`

This keeps the canonical storage tidy while still supporting notebook-friendly wide data.

## 11. SQLite cache methodology

The SQL cache is implemented in `src/nvitk/db/sqlite_index.py`.

### 11.1 Why it exists

`Parquet` plus pandas is enough for many workloads, but SQLite helps for:

- repeated filtering
- lightweight SQL-style slicing
- avoiding full-table reads for some use cases

### 11.2 How it is built

`SQLiteIndex.build()`:

1. iterates through catalog tables
2. reads each `Parquet` table
3. writes it into SQLite with `to_sql(..., if_exists="replace")`
4. creates indexes for `key_columns` and `index_columns`
5. writes a small `_dataset_meta` table

### 11.3 How it is queried

`SQLiteIndex.query_table()` builds:

- a `SELECT` clause from requested columns
- a `WHERE` clause from `build_sql_where()`

Then it returns the result through `pandas.read_sql_query()`.

### 11.4 Important methodological rule

SQLite is not authoritative.

That is why:

- the file is rebuildable
- `.gitignore` excludes `dataset/cache/index.sqlite`
- `DataRepo.get()` always falls back to `Parquet` if the SQLite path fails

## 12. Import methodology for local Excel data

This is implemented in `src/nvitk/db/importers.py`.

This file is the bridge between messy real-world workbook structures and the canonical dataset model.

## 12.1 `SourceSpec`

`SourceSpec` describes one importable unit:

- workbook filename
- sheet name
- source kind
- domain
- layout
- optional modality
- optional cohort id

This allows one workbook to play multiple roles. For example, one sheet can populate both identity mappings and cohort membership.

### 12.2 `PESABRAIN_DB_SPECS`

This is the source registry for the current PESA-Brain workbook directory.

It says which workbook/sheet is interpreted as:

- `subject_ids`
- `cohort`
- `subject_catalog`
- `clinical_wide`
- `image_wide`
- `image_timeseries_long`
- `image_timeseries_wide`
- `hybrid_hemodynamic`
- `variable_dictionary`
- `dropdown_dictionary`

Methodology:

- this is the explicit import contract for the real Excel bundle
- adding a new workbook means adding or extending `SourceSpec` entries

### 12.3 Column normalization helpers

Key helpers:

- `normalize_variable_id()`
- `_region_id()`
- `_normalized_column_map()`
- `_first_matching_column()`
- `ensure_subject_uid()`

These functions let the importer tolerate inconsistent human naming while still producing stable canonical IDs.

### 12.4 Type inference helpers

Key helpers:

- `_parse_datetime_series()`
- `_coerce_numeric()`
- `_series_value_payload()`

Methodology:

- dates can be Excel serial dates or textual dates
- numeric columns can use decimal commas
- a column is classified as numeric when most observed values can be parsed as numbers
- otherwise it becomes text

### 12.5 Inventory registration

`_inventory_row()` and `_register_inventory_rows()` write sheet-level metadata into `source_tables`.

Methodology:

- every recognized source sheet is inventoried
- unmatched sheets are still recorded as `unmapped`
- this is what makes importer coverage auditable

### 12.6 Identity harvesting

`harvest_subject_ids_from_frame()` scans each row and extracts identifier-like columns into `subject_ids`.

Methodology:

- identifiers are recognized by namespace hints such as `patient`, `subject`, `seqn`, `mr`, `codigo`, `id`
- the measurement tables do not need to keep every external ID once the identity mapping exists

### 12.7 Session harvesting

`harvest_sessions_from_frame()` creates session rows when a workbook has a session-like column such as `mri_id` or `MR ID`.

Methodology:

- local workbook imports can create usable session entities even without XNAT
- this supports image measurements derived from files that only know about MRI IDs

### 12.8 Frame builders

The two primitive builders are:

- `_clinical_frame()`
- `_image_frame()`

They take a raw source column and turn it into canonical long-form rows plus a variable registry entry.

This is the heart of the importer because everything else reduces to repeated use of these two builders.

### 12.9 Generic parsers

`_parse_generic_clinical_wide()`:

- imports wide clinical sheets where each non-ID column is a variable

`_parse_generic_image_wide()`:

- imports wide image sheets where each non-ID column is an image variable or region-specific value

These are the default strategies for workbook-like tables.

### 12.10 Specialized parsers

`_parse_image_timeseries_long()`:

- imports long-form 4DFlow time series with explicit `frame`, `flow`, and `phase`

`_parse_image_timeseries_wide()`:

- imports wide time series where columns `0`, `1`, `2`, ... represent frames

`_parse_hybrid_hemodynamic()`:

- handles the merged hemodynamic workbook that mixes:
  - 4DFlow regional metrics
  - ASL regional metrics
  - clinical covariates

This parser is specialized because that workbook is semantically mixed and cannot be treated as purely image or purely clinical.

### 12.11 Variable dictionaries

`_parse_variable_dictionary()` imports rich metadata from codebooks and dictionaries into `variables.json`.

It extracts fields such as:

- original name
- export name
- descriptions
- units
- allowed ranges
- missingness rules
- allowed categorical values
- comments
- codebook source

`_parse_dropdown_dictionary()` complements this by reading dropdown-style allowed-value sheets and attaching those categorical choices to variables.

Methodology:

- the variable registry is not just a list of observed columns
- it is also a semantic dictionary

### 12.12 `rebuild_subjects_table()`

This function derives `subjects` by looking across:

- `subject_ids`
- `clinical_measurements`
- `image_measurements`
- `sessions`

It then chooses summary values such as:

- primary patient ID
- primary SEQN
- sex
- birth date when inferable

Methodology:

- the subject entity is synthesized from the richer canonical tables
- it is not trusted as an independent raw source

### 12.13 Orchestration entry points

`import_source_spec()`:

- dispatches one `SourceSpec` to the correct parser

`import_pesabrain_db_directory()`:

- scans a directory of Excel files
- applies all configured `SourceSpec` mappings
- inventories unmapped sheets
- rebuilds `subjects`
- optionally rebuilds SQLite

`import_pesabrain_curated_tables()`:

- compatibility wrapper around the directory-level importer

### 12.14 Import methodology summary

The importer follows this flow:

1. discover workbook files
2. classify workbook/sheet pairs with `PESABRAIN_DB_SPECS`
3. inventory every imported source
4. harvest IDs and sessions opportunistically
5. convert source columns to long-form measurements
6. update the variable registry from both observations and dictionaries
7. rebuild the derived `subjects` table
8. optionally rebuild SQLite

## 13. XNAT methodology

XNAT integration lives in `src/nvitk/db/xnat.py`.

This file replaces shell-based workflows with a Python API approach.

### 13.1 Connection model

`XnatConnectionConfig` captures:

- server
- project
- user/password
- optional `netrc` path
- TLS verification
- timeout

`connect_xnat()` turns that config into an `xnat.connect(...)` session.

### 13.2 Subject resolution

The sync can target subjects from:

- direct subject tokens
- a text file
- a catalog CSV

This logic is implemented in:

- `parse_subject_tokens()`
- `load_subject_catalog_rows()`
- `resolve_subject_labels()`

Methodology:

- the sync can work with either subject labels or MR IDs
- when MR IDs are provided and a catalog is available, they are mapped back to subject labels

### 13.3 Sequence classification

`classify_scan()` and `infer_flow_orientation()` encode the legacy sequence rules.

Current classification recognizes:

- TOF-like sequences
- 4DFlow sequences, with orientation inference such as AP, RL, FH

This is intentionally explicit and conservative. Unrecognized sequences are skipped rather than guessed.

### 13.4 DICOM download

`download_scan_dicoms()`:

- downloads a scan bundle
- extracts files to a target directory
- avoids file collisions
- can optionally keep the downloaded ZIP

### 13.5 Project sync

`sync_xnat_project()`:

1. resolves requested subjects
2. opens an XNAT session
3. iterates subjects, experiments, and scans
4. classifies scans
5. optionally downloads DICOMs
6. writes canonical rows into:
   - `subjects`
   - `subject_ids`
   - `sessions`
   - `scans`
   - `assets`

Methodology:

- XNAT sync is additive to the local dataset model
- it uses the same canonical tables, not a separate subsystem

## 14. Exceptions and failure model

The dataset-specific exceptions live in `src/nvitk/db/exceptions.py`.

- `DatasetError`: base dataset exception
- `ManifestError`: manifest is invalid or inconsistent
- `TableNotFoundError`: requested table is not defined
- `FilterError`: filter specification is invalid
- `XnatSyncError`: reserved for XNAT sync failures

Methodology:

- domain-specific errors make failures easier to reason about than generic `KeyError` or `ValueError`

## 15. Public API surface

`src/nvitk/db/__init__.py` defines what is considered public:

- `DataRepo`
- `DatasetCatalog`
- `TableDefinition`
- `SQLiteIndex`
- `XnatConnectionConfig`
- `classify_scan()`
- `connect_xnat()`
- `sync_xnat_project()`
- import helpers such as `import_pesabrain_db_directory()`

This file is important because it marks the intended entry points for notebooks and scripts.

## 16. Tests as behavioral contracts

The tests in `tests/db/` show what the code treats as important behavior.

### 16.1 `test_catalog_and_repo.py`

Verifies:

- dataset scaffold creation
- table listing
- clinical alias resolution
- wide access
- filter range handling
- combined `value` behavior for text measurements

### 16.2 `test_sqlite_index.py`

Verifies:

- SQLite-backed queries match `Parquet`-backed queries
- wide image queries still behave correctly through SQLite

### 16.3 `test_xnat.py`

Verifies:

- scan classification logic
- flow orientation inference
- MR ID to subject resolution via catalogs

### 16.4 `test_importers.py`

Verifies:

- directory-level workbook import
- `source_tables` inventory creation
- variable metadata registration from codebooks and dropdown dictionaries

Methodology:

- the tests document intended behavior just as much as they check correctness

## 17. Practical workflows

### 17.1 Create or load a dataset

```python
from nvitk.db import DataRepo

repo = DataRepo("dataset", auto_scaffold=True)
```

### 17.2 Import the local Excel directory

```python
from nvitk.db import DataRepo, import_pesabrain_db_directory

repo = DataRepo("dataset", auto_scaffold=True)
import_pesabrain_db_directory(repo, "/path/to/PESA-Brain/DB", build_sqlite_index=True)
```

### 17.3 Query clinical variables by alias

```python
clinical = repo.clinical(
    variables=["BPXSYM", "BMI"],
    filters={"subject_uid": ["PESA001", "PESA002"]},
    wide=True,
)
```

### 17.4 Query image variables by modality and region

```python
flow = repo.image(
    modality="4dflow",
    variables=["flow_mean"],
    regions=["lica", "rica"],
    wide=True,
)
```

### 17.5 Join analysis-ready frames

```python
report_df = repo.join([clinical, flow], on="subject_uid")
```

### 17.6 Rebuild the SQL cache

```python
repo.build_sqlite_index()
```

## 18. How to extend the system safely

### 18.1 Add a new source workbook

1. Add one or more `SourceSpec` entries in `PESABRAIN_DB_SPECS`
2. Reuse an existing parser if the layout matches
3. Add a specialized parser only if the workbook is semantically mixed or structurally unusual
4. Confirm the imported sheet appears in `source_tables`

### 18.2 Add a new canonical table

1. Add the table definition to `dataset/catalog/tables.json`
2. Ensure the table has meaningful `key_columns`
3. If wide queries should be supported, define `wide_index_columns` and `wide_key_columns`
4. Use `DataRepo.write_table()` or `upsert_table()` so the manifest stays synchronized

### 18.3 Add a new query helper

If a recurring domain pattern appears, add a thin helper in `DataRepo`, similar to `clinical()` or `image()`, rather than pushing users to rewrite the same filter logic in notebooks.

### 18.4 Add a new derived image pipeline

The intended pattern is:

1. write derived files
2. register the files in `assets`
3. register scalar or time-series outputs in `image_measurements`
4. register variable metadata in `variables.json`

That is the methodology the DB layer is designed for, even if some individual converters are still focused mainly on file generation.

## 19. Current limitations and trade-offs

The implementation is intentionally pragmatic. Important limitations are:

- `upsert_table()` is last-write-wins, not transactional
- the SQLite cache can become stale until rebuilt
- manifest dtypes are inferred from current data, so schema evolution should be reviewed when sources change
- wide pivots use `aggfunc="first"`, which assumes the long-form keys are already sufficiently unique
- workbook coverage depends on `PESABRAIN_DB_SPECS`
- XNAT scan classification is currently focused on TOF and 4DFlow rules, not every possible MR sequence
- `subjects` is derived from other tables, so the identity truth really lives in `subject_ids`

These are acceptable trade-offs for a repository-managed analytical data layer, but they should be kept in mind when extending the system.

## 20. Mental model to keep in mind

The easiest way to understand the codebase is to think of it as:

- `catalog.py`: what the dataset is
- `storage.py`: how bytes are stored and typed
- `filters.py`: how rows are selected
- `repo.py`: how users query and write data
- `sqlite_index.py`: how query acceleration is generated
- `importers.py`: how messy Excel inputs become canonical tables
- `xnat.py`: how remote imaging metadata becomes canonical tables
- `exceptions.py`: how domain failures are expressed

And the methodology can be summarized in one sentence:

Store study data canonically as typed `Parquet` tables plus JSON manifests, keep variable semantics and provenance explicit, and expose everything through a small pandas-first API with optional SQLite acceleration.
