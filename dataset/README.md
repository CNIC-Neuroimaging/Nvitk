# Study Dataset

This directory stores the repository-managed local study database for `nvitk`.

## Design
- Canonical tables live as local `Parquet` files under `dataset/tables/`.
- Small manifests and schemas live under `dataset/catalog/`.
- `dataset/cache/index.sqlite` is a generated query/index cache and is not the canonical source of truth.

## Main Tables
- `subjects`: one row per subject known to the repository.
- `subject_ids`: mappings between the internal `subject_uid` and external identifiers such as `patient_id`, `seqn`, or XNAT labels.
- `sessions` and `scans`: XNAT/session inventory and local cache references.
- `clinical_measurements`: long-form clinical variables.
- `image_measurements`: long-form image-derived variables.
- `assets`: local files such as cached DICOMs or derived NIfTI outputs.
- `cohort_membership`: named cohorts and subsets.
- `source_tables`: imported workbook/sheet inventory with original column layouts.

## How It Is Used
- `src/nvitk/db` provides the Python API used by notebooks and scripts.
- Importers convert the current curated Excel extracts into these canonical tables.
- XNAT sync updates the inventory and local asset cache.
- Image derivation tools such as `phase2volume` can register their generated assets here.
