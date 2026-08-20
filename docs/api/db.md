# Database & XNAT

`nvitk.db` is the dataset-persistence layer every pipeline and GUI panel reads from and
writes to — a local, file-backed dataset catalog with an optional XNAT sync/upload layer.

```{code-block} python
from nvitk.db import DataRepo

repo = DataRepo("/path/to/dataset")
```

| Class / module | Purpose |
|---|---|
| `DataRepo` (`repo`) | The central entry point — on-disk dataset access, wraps the catalog and storage layers. |
| `DatasetCatalog`, `SQLiteIndex` | The queryable index over subjects/sessions/measurements (`catalog`, `sqlite_index`). |
| `storage`, `filters`, `exceptions` | Storage-path resolution, query filters, and the DB-layer exception hierarchy. |
| `importers`, `derived_measurements`, `pipeline_assets` | Bulk import and pipeline-output ingestion helpers. |
| `local_dicom_assets`, `local_nifti_assets` | Local-filesystem asset discovery. |
| `xnat`, `xnat_config`, `xnat_upload`, `xnat_projects`, `xnat_pipeline_resources`, `xnat_scan_nifti_assets`, `xnat_4dflows_assets` | XNAT connectivity — sync, upload, and project/resource-specific helpers, backing `nvitk-xnat-sync`, `nvitk-xnat-pipeline-sync`, and each pipeline's own `*-xnat-upload` command. |
| `asl_atlases`, `t1_atlases`, `variable_units` | Atlas lookups and measurement-unit metadata used by {doc}`the Stats GUI <../stats-gui/index>`. |
| `qvtpy_anatomy`, `qvtpy_qc` | QVTPy-specific anatomy/QC lookups — see {doc}`../pipelines/qvtpy`. |

```{seealso}
Full generated reference: [`nvitk.db`](../autoapi/nvitk/db/index).
```
