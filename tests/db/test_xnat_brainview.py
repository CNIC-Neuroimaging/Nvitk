"""Tests for BrainVIEW / VWI_BB XNAT classification."""

from __future__ import annotations

from nvitk.db.xnat import (
    classify_scan,
    parse_brainview_variant,
    select_preferred_vwi_bb_scan,
    xnat_sequence_to_asset_slot,
)


def test_parse_brainview_variant_from_description_prefix() -> None:
    assert parse_brainview_variant("strong csAI_3D_BrainVIEW_T1W", None) == "strong"
    assert parse_brainview_variant("default csAI_3D_BrainVIEW_T1W", None) == "default"


def test_classify_scan_vwi_bb() -> None:
    out = classify_scan("strong csAI_3D_BrainVIEW_T1W", "strong")
    assert out is not None
    assert out["sequence"] == "VWI_BB"
    assert out["variant"] == "strong"


def test_select_preferred_vwi_bb_scan_priority() -> None:
    chosen = select_preferred_vwi_bb_scan(
        [
            {"variant": "weak", "scan_id": "3"},
            {"variant": "strong", "scan_id": "1"},
        ],
        subject_label="PESA001",
    )
    assert chosen is not None
    assert chosen["scan_id"] == "1"


def test_xnat_sequence_to_asset_slot_vwi_bb() -> None:
    assert xnat_sequence_to_asset_slot("VWI_BB") == "vwi_bb"
