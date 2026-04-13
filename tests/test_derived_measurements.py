"""Tests for nvitk.db.derived_measurements."""

from __future__ import annotations

import pandas as pd

from nvitk.db.derived_measurements import (
    DerivedImageMeasurementSpec,
    DerivedVariableRegistration,
    build_image_measurement_rows,
)


def test_build_image_measurement_rows_minimal() -> None:
    agg = pd.DataFrame(
        {
            "subject_uid": ["S1", "S2"],
            "value_num": [10.0, 20.0],
        }
    )
    spec = DerivedImageMeasurementSpec(
        variable_id="v",
        modality="m",
        pipeline_id="p1",
        source_file="derived",
        source_sheet="v",
        source_column="v",
    )
    out = build_image_measurement_rows(agg, spec)
    assert len(out) == 2
    assert list(out["variable_id"]) == ["v", "v"]
    assert list(out["modality"]) == ["m", "m"]
    assert out["session_id"].isna().all()
    assert out["region_id"].isna().all()


def test_registration_to_catalog_entry() -> None:
    spec = DerivedImageMeasurementSpec(
        variable_id="psf",
        modality="4dflow",
        pipeline_id="p",
        source_file="derived",
        source_sheet="psf",
        source_column="psf",
        unit="mL/min",
    )
    reg = DerivedVariableRegistration.from_image_spec(
        spec,
        label="Peak systolic flow",
        aliases=["psf", "PSF"],
        parent_variable="flow_tseries",
    )
    d = reg.to_catalog_entry()
    assert d["variable_id"] == "psf"
    assert d["domain"] == "image"
    assert d["table"] == "image_measurements"
    assert d["parent_variable"] == "flow_tseries"
    assert d["unit"] == "mL/min"
