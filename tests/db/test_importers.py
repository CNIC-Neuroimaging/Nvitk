from __future__ import annotations

from pathlib import Path

import pandas as pd

from nvitk.db import DataRepo
from nvitk.db.importers import import_pesabrain_db_directory, import_pesabrain_source, list_pesabrain_sources


def _write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def test_import_directory_registers_inventory_and_variable_metadata(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    source_root = tmp_path / "db"
    source_root.mkdir()

    _write_workbook(
        source_root / "PESABrain_All_IDs.xlsx",
        {
            "Sheet1": pd.DataFrame(
                [
                    {
                        "patient_id": "PESA001",
                        "mri_id": "BMRI001",
                        "seqn": "1",
                    }
                ]
            )
        },
    )
    _write_workbook(
        source_root / "PESABrain_Clinical_20260216.xlsx",
        {
            "Sheet1": pd.DataFrame(
                [
                    {
                        "patient_id": "PESA001",
                        "seqn": "1",
                        "age_at_mri": 62.5,
                        "sex": "Male",
                    }
                ]
            )
        },
    )
    _write_workbook(
        source_root / "PESABrain_Variables_20250312.xlsx",
        {
            "Variables": pd.DataFrame(
                [
                    {
                        "Nombre": "AGE_AT_MRI",
                        "Nombre exportación": "age_at_mri",
                        "Descripción": "Age at MRI",
                        "Tipo": "Num",
                        "Unidades": "years",
                        "Rango posibilidad (missing)": "0",
                        "Codebook": "CLINICAL.csv",
                    }
                ]
            )
        },
    )
    _write_workbook(
        source_root / "PESABrain_4DFlow_AnatomicalCodebook_20260204.xlsx",
        {
            "Tests Cognitivos": pd.DataFrame(columns=["Nombre", "Nombre exportación"]),
            "Neuroimagen": pd.DataFrame(
                [
                    {
                        "Nombre": "Aneurysm",
                        "Nombre exportación": "Aneurysm",
                        "Descripción (Significado de la variable)": "Presencia de aneurismas cerebrales",
                        "Tipo Texto/Cat/Num/Fecha": "Cat",
                        "Unidades (Unidades en las que esta medida, Ej. cm, mg,)": "0: No; 1: ICAr",
                    }
                ]
            ),
            "Casos": pd.DataFrame(
                [
                    {
                        "Codi Sub.": "PESA001",
                        "Aneurysm": "0: No",
                    }
                ]
            ),
            "DESPLEGABLES": pd.DataFrame(
                [
                    {"Unnamed: 0": None, "Unnamed: 1": "Aneurysm"},
                    {"Unnamed: 0": None, "Unnamed: 1": "0: No"},
                    {"Unnamed: 0": None, "Unnamed: 1": "1: ICAr"},
                ]
            ),
        },
    )

    repo = DataRepo(dataset_root, auto_scaffold=True)
    import_pesabrain_db_directory(repo, source_root)

    source_tables = repo.get("source_tables")
    assert set(source_tables["source_kind"]) == {
        "subject_ids",
        "clinical_wide",
        "variable_dictionary",
        "image_wide",
        "dropdown_dictionary",
    }

    clinical = repo.get("clinical_measurements")
    assert "age_at_mri" in set(clinical["variable_id"])

    image = repo.get("image_measurements")
    aneurysm_rows = image[image["variable_id"] == "aneurysm"]
    assert not aneurysm_rows.empty
    assert aneurysm_rows.iloc[0]["modality"] == "neuroimage_report"

    variables = {entry["variable_id"]: entry for entry in repo.catalog.variable_entries()}
    aneurysm = variables["aneurysm"]
    assert aneurysm["description"] == "Presencia de aneurismas cerebrales"
    assert "0: No" in aneurysm["allowed_values"]
    assert "1: ICAr" in aneurysm["allowed_values"]
    assert aneurysm["modality"] == "neuroimage_report"


def test_import_single_pesabrain_source_supports_stepwise_loading(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    source_root = tmp_path / "db"
    source_root.mkdir()

    _write_workbook(
        source_root / "PESABrain_All_IDs.xlsx",
        {
            "Sheet1": pd.DataFrame(
                [
                    {
                        "patient_id": "PESA001",
                        "mri_id": "BMRI001",
                        "seqn": "1",
                    }
                ]
            )
        },
    )
    _write_workbook(
        source_root / "PESABrain_Clinical_20260216.xlsx",
        {
            "Sheet1": pd.DataFrame(
                [
                    {
                        "patient_id": "PESA001",
                        "seqn": "1",
                        "age_at_mri": 62.5,
                    }
                ]
            )
        },
    )

    repo = DataRepo(dataset_root, auto_scaffold=True)
    source_specs = list_pesabrain_sources()
    assert any(item["filename"] == "PESABrain_Clinical_20260216.xlsx" for item in source_specs)

    import_pesabrain_source(
        repo,
        source_root,
        "PESABrain_All_IDs.xlsx",
        sheet="Sheet1",
        source_kind="subject_ids",
    )
    import_pesabrain_source(
        repo,
        source_root,
        "PESABrain_Clinical_20260216.xlsx",
        sheet="Sheet1",
        source_kind="clinical_wide",
    )

    assert len(repo.get("subject_ids")) == 3
    assert "age_at_mri" in set(repo.get("clinical_measurements")["variable_id"])
    assert "PESA001" in set(repo.get("subjects")["subject_uid"])


def test_image_timeseries_wide_empty_session_stable_keys_no_extra_grain(tmp_path) -> None:
    """Wide timeseries sheets are subject-only; session_id must be '' so upserts do not duplicate rows."""
    dataset_root = tmp_path / "dataset"
    source_root = tmp_path / "db"
    source_root.mkdir()

    wide = pd.DataFrame(
        {
            "patient_id": ["PESA001"],
            "subject_uid": ["PESA001"],
            "vessel": ["Left ICA"],
            "vessel_code": ["LICA"],
            0: [10.0],
            1: [20.0],
        }
    )
    _write_workbook(
        source_root / "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx",
        {"Datos": wide},
    )
    repo = DataRepo(dataset_root, auto_scaffold=True)
    import_pesabrain_source(
        repo,
        source_root,
        "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx",
        sheet="Datos",
        source_kind="image_timeseries_wide",
        source_batch_id="wide_ts",
        rebuild_subjects=False,
    )
    img = repo.get("image_measurements")
    assert len(img) == 2
    assert img["session_id"].astype("string").tolist() == ["", ""]
    assert set(img["frame_index"].tolist()) == {0, 1}

    import_pesabrain_source(
        repo,
        source_root,
        "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx",
        sheet="Datos",
        source_kind="image_timeseries_wide",
        source_batch_id="wide_ts_repeat",
        rebuild_subjects=False,
    )
    img2 = repo.get("image_measurements")
    assert len(img2) == 2
