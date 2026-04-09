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


def test_repo_generic_filters_support_range_conditions(sample_repo: DataRepo):
    filtered = sample_repo.get(
        "clinical_measurements",
        filters={"variable_id": "bpxsym", "value_num": {"$ge": 120, "$lt": 130}},
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["subject_uid"] == "PESA001"


def test_repo_returns_text_measurements_with_combined_value(sample_repo: DataRepo):
    clinical = sample_repo.clinical(variables=["APOE"])

    assert len(clinical) == 1
    assert clinical.iloc[0]["value"] == "e3/e4"
    assert pd.isna(clinical.iloc[0]["value_num"])
