"""Shared dataclasses for GUI SGE remote jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SgeConnection:
    """SSH host/username/password for one SGE cluster connection."""

    host: str
    user: str
    password: str


@dataclass
class SgePendingJob:
    """A submitted GUI SGE job being tracked for completion (id, connection, status, results)."""

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
    """Parsed contents of a worker's ``.done`` marker file (exit code, output files, error)."""

    job_id: str
    exit_code: int
    output_files: list[str] = field(default_factory=list)
    finished_at: str = ""
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SgeDoneMarker:
        """Parse a JSON-decoded ``.done`` payload into an :class:`SgeDoneMarker`."""
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
        """Serialize this marker back to a plain dict (inverse of :meth:`from_dict`)."""
        return {
            "job_id": self.job_id,
            "exit_code": self.exit_code,
            "output_files": list(self.output_files),
            "finished_at": self.finished_at,
            "error": self.error,
        }


def remote_done_path(remote_job_root: str) -> str:
    """Remote path of the ``.done`` marker file for a job rooted at *remote_job_root*."""
    return f"{remote_job_root.rstrip('/')}/output/.done"


def remote_output_dir(remote_job_root: str) -> str:
    """Remote output directory for a job rooted at *remote_job_root*."""
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
