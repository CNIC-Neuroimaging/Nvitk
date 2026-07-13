"""QVTpy-specific SGE chunk helpers (stage counts for per-user job limits)."""

from __future__ import annotations


def count_sge_stages_per_subject(
    *,
    run_conv: bool,
    run_eicab: bool,
    run_s2: bool,
    run_s3: bool,
    run_s4: bool,
    run_s4t: bool,
    run_s5: bool,
    run_s6: bool,
    run_s7: bool,
) -> int:
    """Number of ``qsub`` jobs emitted per subject in a QVTpy SGE wave."""
    return sum(
        (
            run_conv,
            run_eicab,
            run_s2,
            run_s3,
            run_s4,
            run_s4t,
            run_s5,
            run_s6,
            run_s7,
        )
    )


__all__ = ["count_sge_stages_per_subject"]
