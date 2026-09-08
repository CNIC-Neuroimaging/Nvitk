"""Round-trip local data through the cluster for a single inference job.

``--submit sge`` normally assumes the images already live where the job can see them. When they
do not -- they are on the workstation and the cluster storage is not shared -- this module
uploads them to a scratch directory, lets the job run there, brings the results back and
removes what it created.

Why it blocks
-------------
Submitting and returning is right for training, which runs for days. A staged inference has to
wait: nothing can be downloaded until the job has produced it. So this polls ``qstat`` and only
returns once the job is gone from the queue.

Deleting on the far side
------------------------
Every removal is confined to a directory this module created, named with its own prefix and a
random suffix, under a staging root the pipeline owns. :func:`assert_removable` enforces that,
and it is checked immediately before the deletion rather than at construction -- the point is to
make "delete a path built from a user-supplied string" impossible to reach, not merely unlikely.
On failure nothing is deleted: the inputs and the job's output are what you need to diagnose it.
"""

from __future__ import annotations

import posixpath
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.logger import Logger

log = Logger()

#: Directory the pipeline owns for scratch transfers, beside the submission scripts.
STAGING_DIR_NAME: str = "staging"

#: Prefix every staged directory carries. Part of the deletion guard.
STAGING_PREFIX: str = "topbrain_infer_"

#: Seconds between ``qstat`` polls. Inference is minutes-to-hours; a tighter loop would only
#: add SSH round trips.
DEFAULT_POLL_SECONDS: float = 30.0

#: Give up waiting after this long, leaving the remote tree in place.
DEFAULT_TIMEOUT_SECONDS: float = 24 * 3600.0


def staging_root() -> str:
    """The remote directory staged transfers live under."""
    from nvitk.pipes.topbrain import config as cfg

    return posixpath.join(posixpath.dirname(str(cfg.SGE_SCRIPTS_DIR)), STAGING_DIR_NAME)


def assert_removable(remote_path: str) -> None:
    """Raise unless *remote_path* is a staging directory this module created.

    Raises
    ------
    ValueError
        For anything outside :func:`staging_root`, or whose basename lacks
        :data:`STAGING_PREFIX`. A recursive delete must never be reachable from a path the
        caller supplied.
    """
    normalised = posixpath.normpath(str(remote_path))
    root = posixpath.normpath(staging_root())
    if not normalised.startswith(root + "/"):
        raise ValueError(f"Refusing to delete {normalised!r}: it is not under {root!r}.")
    if not posixpath.basename(normalised).startswith(STAGING_PREFIX):
        raise ValueError(
            f"Refusing to delete {normalised!r}: its name does not start with "
            f"{STAGING_PREFIX!r}, so this module did not create it."
        )


def wait_for_jobs(
    job_ids: Sequence[str],
    *,
    host: str,
    user: str,
    password: str,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Block until none of *job_ids* is in the queue; returns whether they all finished.

    ``qstat`` lists queued and running jobs and says nothing about how a finished one ended, so
    this reports "no longer queued", not "succeeded". The caller decides success from what the
    job actually produced -- which for inference is the only evidence that matters.
    """
    from nvitk.cluster.remote_transfer import ssh_exec

    wanted = {str(j).strip() for j in job_ids if str(j).strip()}
    if not wanted:
        return True
    deadline = time.monotonic() + float(timeout_seconds)
    log.info("Waiting for job(s) %s (polling every %.0fs)...", ", ".join(sorted(wanted)),
             poll_seconds)
    while True:
        code, stdout, _stderr = ssh_exec(
            host=host, user=user, password=password, command=f"qstat -u {user}"
        )
        if code != 0:
            log.warning("qstat exited %d; assuming the job is still queued.", code)
        else:
            running = {
                line.split()[0] for line in stdout.splitlines()
                if line.strip() and line.split()[0].isdigit()
            }
            if not (wanted & running):
                log.ok("Job(s) no longer in the queue.")
                return True
        if time.monotonic() > deadline:
            log.warning("Timed out after %.0fs; the remote data is left in place.",
                        timeout_seconds)
            return False
        time.sleep(poll_seconds)


@dataclass
class StagedTransfer:
    """One staged round trip: what was uploaded, where, and what came back."""

    remote_root: str
    remote_input: str
    remote_output: str
    uploaded: int = 0
    retrieved: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "remote_root": self.remote_root, "uploaded": self.uploaded,
            "retrieved": self.retrieved, "notes": list(self.notes),
        }


def new_remote_root() -> str:
    """A fresh staging directory name: timestamped for humans, random for uniqueness."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return posixpath.join(staging_root(), f"{STAGING_PREFIX}{stamp}_{uuid.uuid4().hex[:8]}")


def upload_inputs(
    volumes: Sequence[Path], remote_input: str, *, host: str, user: str, password: str
) -> int:
    """Upload *volumes* into *remote_input*; returns the count.

    Names are preserved: stage 4 derives case ids from them, so renaming here would rename the
    outputs too.
    """
    from nvitk.cluster.remote_transfer import ensure_remote_dir, sftp_session, upload_file

    with sftp_session(host=host, user=user, password=password) as (_ssh, sftp):
        ensure_remote_dir(sftp, remote_input)
        for volume in volumes:
            upload_file(sftp, Path(volume), posixpath.join(remote_input, Path(volume).name))
    log.ok(f"uploaded {len(volumes)} volume(s) -> {remote_input}")
    return len(volumes)


def retrieve_outputs(
    remote_output: str, local_output: Path, *, host: str, user: str, password: str
) -> int:
    """Download everything under *remote_output* into *local_output*; returns the file count."""
    from nvitk.cluster.remote_transfer import download_directory_sftp, sftp_session

    Path(local_output).mkdir(parents=True, exist_ok=True)
    with sftp_session(host=host, user=user, password=password) as (_ssh, sftp):
        count = download_directory_sftp(sftp, remote_output, Path(local_output))
    log.ok(f"retrieved {count} file(s) -> {local_output}")
    return int(count)


def remove_remote_root(remote_root: str, *, host: str, user: str, password: str) -> bool:
    """Delete a staging directory, after :func:`assert_removable` allows it."""
    from nvitk.cluster.remote_transfer import ssh_exec

    assert_removable(remote_root)
    code, _out, err = ssh_exec(
        host=host, user=user, password=password,
        command=f"rm -rf -- {remote_root!r}",
    )
    if code != 0:
        log.warning("Could not remove %s (exit %d): %s", remote_root, code, err.strip()[:200])
        return False
    log.info("Removed staged data at %s", remote_root)
    return True


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "STAGING_DIR_NAME",
    "STAGING_PREFIX",
    "StagedTransfer",
    "assert_removable",
    "new_remote_root",
    "remove_remote_root",
    "retrieve_outputs",
    "staging_root",
    "upload_inputs",
    "wait_for_jobs",
]
