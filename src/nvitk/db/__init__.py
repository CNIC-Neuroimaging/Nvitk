"""
Dataset catalog, SQLite index, XNAT helpers, imports, and :class:`~nvitk.db.repo.DataRepo`.

Configure the tree with ``NVITK_DATASET_ROOT`` or pass ``root=`` to ``DataRepo``.
"""

from __future__ import annotations

from .catalog import DatasetCatalog, TableDefinition
from .derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedImageMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    build_image_measurement_rows,
    publish_derived_measurements,
    register_derived_variable,
)
from .importers import (
    import_measurements_from_source,
    import_pesabrain_curated_tables,
    import_pesabrain_db_directory,
    import_pesabrain_source,
    import_subject_ids_from_source,
    list_pesabrain_sources,
    upsert_cohort_membership_for_subjects,
)
from .asl_atlases import ASL_ATLAS_REGIONS, regions_for_atlas
from .t1_atlases import register_t1_atlas_regions, regions_for_t1_atlas
from .repo import DEFAULT_COHORT_ID, DataRepo, get_repo_from_settings
from .sqlite_index import SQLiteIndex
from .local_dicom_assets import register_dicom_tree, upsert_dicom_assets
from .local_nifti_assets import register_nifti_tree, upsert_nifti_assets
from .xnat import (
    classify_scan,
    connect_xnat,
    list_scans_for_subject,
    list_subjects_for_project,
    resolve_xnat_scan_from_scan_row,
    sync_xnat_project,
    xnat_sequence_to_asset_slot,
)
from .xnat_projects import (
    classify_experiment_ia_pet_v5,
    classify_scan_for_project,
    classify_scan_ia_pet_v5,
    get_xnat_project,
    list_xnat_project_ids,
)
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection

__all__ = [
    "ASL_ATLAS_REGIONS",
    "DEFAULT_COHORT_ID",
    "DerivedClinicalMeasurementSpec",
    "DerivedImageMeasurementSpec",
    "DerivedVariableRegistration",
    "DataRepo",
    "get_repo_from_settings",
    "regions_for_atlas",
    "regions_for_t1_atlas",
    "register_t1_atlas_regions",
    "DatasetCatalog",
    "TableDefinition",
    "build_clinical_measurement_rows",
    "build_image_measurement_rows",
    "publish_derived_measurements",
    "register_derived_variable",
    "SQLiteIndex",
    "XnatConnectionConfig",
    "classify_scan",
    "classify_experiment_ia_pet_v5",
    "classify_scan_for_project",
    "classify_scan_ia_pet_v5",
    "connect_xnat",
    "get_xnat_project",
    "list_xnat_project_ids",
    "list_scans_for_subject",
    "list_subjects_for_project",
    "import_measurements_from_source",
    "import_pesabrain_curated_tables",
    "import_pesabrain_source",
    "import_subject_ids_from_source",
    "list_pesabrain_sources",
    "upsert_cohort_membership_for_subjects",
    "sync_xnat_project",
    "resolve_xnat_scan_from_scan_row",
    "xnat_sequence_to_asset_slot",
    "load_xnat_profile",
    "resolve_xnat_connection",
    "register_dicom_tree",
    "register_nifti_tree",
    "upsert_dicom_assets",
    "upsert_nifti_assets",
]
