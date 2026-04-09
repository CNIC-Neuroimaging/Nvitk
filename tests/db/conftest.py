from __future__ import annotations

import pandas as pd
import pytest

from nvitk.db import DataRepo


@pytest.fixture()
def sample_repo(tmp_path):
    root = tmp_path / "dataset"
    repo = DataRepo(root, auto_scaffold=True)

    repo.write_table(
        "clinical_measurements",
        pd.DataFrame(
            [
                {
                    "subject_uid": "PESA001",
                    "visit_id": "V1",
                    "variable_id": "bpxsym",
                    "value_num": 128.0,
                    "value_text": pd.NA,
                    "unit": "mmHg",
                    "value_kind": "numeric",
                    "source_table": "clinical",
                    "source_batch_id": "seed",
                    "measured_at": pd.Timestamp("2026-01-01"),
                },
                {
                    "subject_uid": "PESA001",
                    "visit_id": "V1",
                    "variable_id": "bmi",
                    "value_num": 26.5,
                    "value_text": pd.NA,
                    "unit": "kg/m2",
                    "value_kind": "numeric",
                    "source_table": "clinical",
                    "source_batch_id": "seed",
                    "measured_at": pd.Timestamp("2026-01-01"),
                },
                {
                    "subject_uid": "PESA002",
                    "visit_id": "V1",
                    "variable_id": "apoe",
                    "value_num": float("nan"),
                    "value_text": "e3/e4",
                    "unit": pd.NA,
                    "value_kind": "text",
                    "source_table": "apoe",
                    "source_batch_id": "seed",
                    "measured_at": pd.Timestamp("2026-01-02"),
                },
            ]
        ),
    )
    repo.write_table(
        "image_measurements",
        pd.DataFrame(
            [
                {
                    "subject_uid": "PESA001",
                    "session_id": "MRI1",
                    "modality": "4dflow",
                    "region_id": "ICA_L",
                    "variable_id": "flow_mean",
                    "value_num": 240.0,
                    "value_text": pd.NA,
                    "unit": "mL/min",
                    "value_kind": "numeric",
                    "pipeline_name": "hemo",
                    "pipeline_version": "1.0.0",
                    "qc_status": "ok",
                    "source_asset": pd.NA,
                    "source_batch_id": "seed",
                    "measured_at": pd.Timestamp("2026-01-01"),
                },
                {
                    "subject_uid": "PESA001",
                    "session_id": "MRI1",
                    "modality": "4dflow",
                    "region_id": "ICA_R",
                    "variable_id": "flow_mean",
                    "value_num": 230.0,
                    "value_text": pd.NA,
                    "unit": "mL/min",
                    "value_kind": "numeric",
                    "pipeline_name": "hemo",
                    "pipeline_version": "1.0.0",
                    "qc_status": "ok",
                    "source_asset": pd.NA,
                    "source_batch_id": "seed",
                    "measured_at": pd.Timestamp("2026-01-01"),
                },
            ]
        ),
    )
    repo.register_variables(
        [
            {
                "variable_id": "bpxsym",
                "source_column": "BPXSYM",
                "aliases": ["BPXSYM"],
                "domain": "clinical",
                "table": "clinical_measurements",
                "label": "Systolic blood pressure",
                "value_kind": "numeric",
            },
            {
                "variable_id": "bmi",
                "source_column": "BMI",
                "aliases": ["BMI"],
                "domain": "clinical",
                "table": "clinical_measurements",
                "label": "Body mass index",
                "value_kind": "numeric",
            },
            {
                "variable_id": "apoe",
                "source_column": "APOE",
                "aliases": ["APOE"],
                "domain": "clinical",
                "table": "clinical_measurements",
                "label": "APOE genotype",
                "value_kind": "text",
            },
            {
                "variable_id": "flow_mean",
                "source_column": "Flow Mean",
                "aliases": ["flow_mean"],
                "domain": "image",
                "table": "image_measurements",
                "modality": "4dflow",
                "label": "Mean flow",
                "value_kind": "numeric",
            },
        ]
    )
    return repo
