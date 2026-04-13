"""Regression tests for measurement table filtering (variable specs vs structural columns, entity keys)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nvitk.db.repo import (
    DataRepo,
    _apply_variable_value_filters,
    _measurement_entity_tuple_series,
)


def test_measurement_entity_keys_include_session_and_region() -> None:
    df = pd.DataFrame(
        {
            "subject_uid": ["A", "A", "B"],
            "session_id": ["S1", "S1", "S2"],
            "modality": ["asl", "asl", "asl"],
            "pipeline_id": ["p1", "p1", "p1"],
            "region_id": ["r1", "r2", "r1"],
            "variable_id": ["flow_mean", "flow_mean", "flow_mean"],
            "value_num": [150.0, 50.0, 120.0],
            "value_text": pd.array([pd.NA, pd.NA, pd.NA], dtype="string"),
        }
    )
    keys = _measurement_entity_tuple_series(df)
    assert keys.iloc[0] != keys.iloc[1]
    filtered = _apply_variable_value_filters(
        df,
        [("flow_mean", {">=": 100, "<=": 200})],
    )
    assert len(filtered) == 2
    assert set(filtered["region_id"].astype(str)) == {"r1"}


@pytest.mark.parametrize(
    ("table", "key", "expect_variable"),
    [
        ("clinical_measurements", "bmi", True),
        ("clinical_measurements", "subject_uid", False),
        ("image_measurements", "flow_mean", True),
    ],
)
def test_split_measurement_filters_variable_vs_structural(
    table: str,
    key: str,
    expect_variable: bool,
) -> None:
    root = Path(__file__).resolve().parents[1] / "dataset"
    if not (root / "catalog" / "repository.json").exists():
        pytest.skip("dataset catalog not present")
    repo = DataRepo(root)
    struct, var_specs = repo._split_measurement_filters(
        {key: 1},
        domain="clinical" if table == "clinical_measurements" else "image",
        table_name=table,
    )
    if expect_variable:
        assert not struct
        assert var_specs
    else:
        assert struct.get(key) == 1
        assert not var_specs
