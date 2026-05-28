"""Tests for QC review status aggregation."""

from __future__ import annotations

from nvitk.pipes.pesa_fat.qc.review_policy import overall_status


def test_overall_ok_requires_all_expected() -> None:
    expected = ["A", "B", "C"]
    assert (
        overall_status({"A": "OK", "B": "OK", "C": "OK"}, expected_structures=expected)
        == "OK"
    )
    assert (
        overall_status({"A": "OK", "B": "OK"}, expected_structures=expected)
        == "PENDING"
    )
    assert (
        overall_status({"A": "OK", "B": "FAIL", "C": "OK"}, expected_structures=expected)
        == "FAIL"
    )


def test_overall_fail_on_any_fail() -> None:
    assert overall_status({"A": "OK", "B": "FAIL"}, expected_structures=["A", "B"]) == "FAIL"
