"""Shared dataclasses for GUI SGE remote jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SgeConnection:
    host: str
    user: str
    password: str


@dataclass
class SgePendingJob:
    job_id: str
    tool_id: str
    remote_job_root: str
    connection: SgeConnection
    output_name: str = "output.nii.gz"
    status: str = "submitted"
    done_payload: dict[str, Any] | None = None
    local_download_dir: str | None = None


@dataclass
class SgeDoneMarker:
    job_id: str
    exit_code: int
    output_files: list[str] = field(default_factory=list)
    finished_at: str = ""
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SgeDoneMarker:
        files = data.get("output_files") or []
        if isinstance(files, str):
            files = [files]
        return cls(
            job_id=str(data.get("job_id") or ""),
            exit_code=int(data.get("exit_code", 1)),
            output_files=[str(f) for f in files],
            finished_at=str(data.get("finished_at") or ""),
            error=(str(data["error"]) if data.get("error") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "exit_code": self.exit_code,
            "output_files": list(self.output_files),
            "finished_at": self.finished_at,
            "error": self.error,
        }


def remote_done_path(remote_job_root: str) -> str:
    return f"{remote_job_root.rstrip('/')}/output/.done"


def remote_output_dir(remote_job_root: str) -> str:
    return f"{remote_job_root.rstrip('/')}/output"


def verify_local_downloads(local_dir: Path, done: SgeDoneMarker) -> list[Path]:
    """Ensure every file listed in *done* was downloaded with non-zero size."""
    expected = done.output_files or []
    if not expected:
        raise ValueError("Done marker lists no output files.")
    paths = []
    for name in expected:
        lp = local_dir / name
        if not lp.is_file() or lp.stat().st_size <= 0:
            raise ValueError(f"Downloaded output missing or empty: {name}")
        paths.append(lp)
    return paths


__all__ = [
    "SgeConnection",
    "SgeDoneMarker",
    "SgePendingJob",
    "remote_done_path",
    "remote_output_dir",
    "verify_local_downloads",
]
