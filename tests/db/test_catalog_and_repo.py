from __future__ import annotations

import pandas as pd

from nvitk.db import DataRepo, DatasetCatalog


def test_catalog_scaffold_and_table_listing(tmp_path):
    root = tmp_path / "dataset"
    DatasetCatalog.create_scaffold(root)

    catalog = DatasetCatalog(root)
    assert "clinical_measurements" in catalog.list_tables()
    assert catalog.sqlite_index_path == root / "cache" / "index.sqlite"


def test_repo_clinical_alias_resolution_and_wide_access(sample_repo: DataRepo):
    wide = sample_repo.clinical(variables=["BPXSYM", "BMI"], filters={"subject_uid": "PESA001"}, wide=True)

    assert list(wide["subject_uid"]) == ["PESA001"]
    assert list(wide["visit_id"]) == ["V1"]
    assert wide.loc[0, "bpxsym"] == 128.0
    assert wide.loc[0, "bmi"] == 26.5


def test_catalog_resolve_variable_ids_normalized_and_labels(tmp_path):
    root = tmp_path / "dataset"
    DatasetCatalog.create_scaffold(root)
    catalog = DatasetCatalog(root)
    catalog.register_variables(
        [
            {
                "variable_id": "flow_tseries",
                "source_column": "wide_frame",
                "aliases": ["wide_frame", "flow", "flow_tseries"],
                "domain": "image",
                "table": "image_measurements",
                "label": "Flow time series",
            }
        ],
        merge=False,
    )
    catalog.refresh()
    out = catalog.resolve_variable_ids(
        ["flow", "FLOW_TSERIES", "Flow time series", "wide_frame", "unknown_x"],
        domain="image",
    )
    assert out == ["flow_tseries", "flow_tseries", "flow_tseries", "flow_tseries", "unknown_x"]


def test_repo_generic_filters_support_range_conditions(sample_repo: DataRepo):
    filtered = sample_repo.get(
        "clinical_measurements",
        filters={"variable_id": "bpxsym", "value_num": {"$ge": 120, "$lt": 130}},
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["subject_uid"] == "PESA001"


def test_repo_returns_text_measurements_with_combined_value(sample_repo: DataRepo):
    clinical = sample_repo.clinical(variables=["APOE"], wide=False)

    assert len(clinical) == 1
    assert clinical.iloc[0]["value"] == "e3/e4"
    assert pd.isna(clinical.iloc[0]["value_num"])


def test_repo_image_wide_preserves_rows_when_session_id_all_na(tmp_path):
    """Optional session_id is often unset for 4DFlow timeseries sheets; pivot must not drop every row."""
    root = tmp_path / "dataset"
    repo = DataRepo(root, auto_scaffold=True)
    df = pd.DataFrame(
        [
            {
                "subject_uid": "PESA001",
                "session_id": pd.NA,
                "modality": "4dflow",
                "region_id": "lica",
                "frame_index": 0,
                "variable_id": "flow_tseries",
                "value_num": 10.0,
                "value_text": pd.NA,
                "unit": pd.NA,
                "value_kind": "numeric",
                "pipeline_name": "ts",
                "pipeline_version": pd.NA,
                "qc_status": pd.NA,
                "source_asset": pd.NA,
                "source_table": "t",
                "source_file": "long.xlsx",
                "source_sheet": "Datos",
                "source_column": "flow",
                "source_batch_id": "b",
                "measured_at": pd.NaT,
            },
            {
                "subject_uid": "PESA001",
                "session_id": pd.NA,
                "modality": "4dflow",
                "region_id": "lica",
                "frame_index": 1,
                "variable_id": "flow_tseries",
                "value_num": 11.0,
                "value_text": pd.NA,
                "unit": pd.NA,
                "value_kind": "numeric",
                "pipeline_name": "ts",
                "pipeline_version": pd.NA,
                "qc_status": pd.NA,
                "source_asset": pd.NA,
                "source_table": "t",
                "source_file": "long.xlsx",
                "source_sheet": "Datos",
                "source_column": "flow",
                "source_batch_id": "b",
                "measured_at": pd.NaT,
            },
        ]
    )
    df["session_id"] = pd.array([pd.NA, pd.NA], dtype="string")
    repo.write_table("image_measurements", df)
    wide = repo.image(wide=True)
    assert len(wide) == 1
    assert wide.iloc[0]["subject_uid"] == "PESA001"
    assert wide.iloc[0]["session_id"] == ""
    col_0 = "4dflow__lica__0__flow_tseries"
    col_1 = "4dflow__lica__1__flow_tseries"
    assert col_0 in wide.columns and col_1 in wide.columns
    assert wide.iloc[0][col_0] == 10.0
    assert wide.iloc[0][col_1] == 11.0
