"""Catalog-driven measurement pipelines: defaults and ``resolve_pipeline_selector``."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvitk.db.catalog import DatasetCatalog

_DATASET = Path(__file__).resolve().parents[2] / "dataset"


@pytest.fixture(scope="module")
def catalog() -> DatasetCatalog:
    if not (_DATASET / "catalog" / "repository.json").exists():
        pytest.skip("dataset/catalog not present")
    return DatasetCatalog(_DATASET)


def test_default_pipeline_ids(catalog: DatasetCatalog) -> None:
    assert catalog.default_pipeline_id("4dflow") == "pesabrain_4dflow_current"
    assert catalog.default_pipeline_id("asl") == "pesabrain_asl_current"


def test_resolve_legacy_alias(catalog: DatasetCatalog) -> None:
    assert catalog.resolve_pipeline_selector("legacy") == catalog.pipeline_ids_for_role("legacy")


def test_resolve_explicit_pipeline_id(catalog: DatasetCatalog) -> None:
    assert catalog.resolve_pipeline_selector("pesabrain_4dflow_current") == ["pesabrain_4dflow_current"]
