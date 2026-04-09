from __future__ import annotations

import pandas as pd

from nvitk.db import DataRepo


def test_sqlite_index_matches_parquet_queries(sample_repo: DataRepo):
    sample_repo.build_sqlite_index()

    parquet_df = sample_repo.image(
        modality="4dflow",
        variables=["flow_mean"],
        regions=["ICA_L", "ICA_R"],
        use_sqlite=False,
    ).sort_values(["subject_uid", "region_id"]).reset_index(drop=True)

    sqlite_df = sample_repo.image(
        modality="4dflow",
        variables=["flow_mean"],
        regions=["ICA_L", "ICA_R"],
        use_sqlite=True,
    ).sort_values(["subject_uid", "region_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(parquet_df, sqlite_df, check_dtype=False)


def test_sqlite_index_supports_wide_image_queries(sample_repo: DataRepo):
    sample_repo.build_sqlite_index()

    wide = sample_repo.image(modality="4dflow", variables=["flow_mean"], wide=True, use_sqlite=True)

    assert "4dflow__ICA_L__flow_mean" in wide.columns
    assert "4dflow__ICA_R__flow_mean" in wide.columns
