from __future__ import annotations

from pathlib import Path

from nvitk.db.xnat import classify_scan, infer_flow_orientation, resolve_subject_labels


def test_classify_scan_matches_legacy_sequence_rules():
    tof = classify_scan("cs3DI_MC_TOF", "usable")
    ap = classify_scan("4DQflowNeuro_AP", "usable")
    generic = classify_scan("4DQflowNeuro", "usable")
    rejected = classify_scan("4DQflowNeuro_AP", "questionable")

    assert tof == {"modality": "tof", "orientation": None, "sequence": "TOF"}
    assert ap == {"modality": "4dflow", "orientation": "AP", "sequence": "4DFLOW_AP"}
    assert generic == {"modality": "4dflow", "orientation": "GENERIC", "sequence": "4DFLOW_GENERIC"}
    assert rejected is None


def test_infer_flow_orientation_handles_multiple_legacy_aliases():
    assert infer_flow_orientation("4DQflowNeuro_PA") == "AP"
    assert infer_flow_orientation("4DQflowNeuro_LR") == "RL"
    assert infer_flow_orientation("4DQflowNeuro_HF") == "FH"


def test_resolve_subject_labels_can_map_mrid_from_catalog(tmp_path):
    catalog = tmp_path / "subject_catalog.csv"
    catalog.write_text(
        "Subject,MR ID,Scans\n"
        "PESA001,BMRI001,TOF;4DFLOW_AP\n"
        "PESA002,BMRI002,TOF;4DFLOW_RL\n",
        encoding="utf-8",
    )

    subjects = resolve_subject_labels(catalog_path=catalog, subjects=["BMRI001"], id_type="mrid")

    assert subjects == ["PESA001"]
