"""Generic SGE subject-chunk helpers and job-completion polling.

Pipeline-specific logic (e.g. how many jobs one subject produces) belongs in each
pipeline package, not here.
"""

from __future__ import annotations

import re
import shlex
import time
from typing import Iterable, Sequence

from nvitk.core.logger import Logger

log = Logger()

_JOB_ID_FROM_ECHO = re.compile(r"->\s*(\d+)\s*$", re.MULTILINE)
_BARE_JOB_ID = re.compile(r"^\s*(\d+)\s*$", re.MULTILINE)


def chunk_sequence(items: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Split *items* into chunks of at most *chunk_size* (``<=0`` means one chunk)."""
    seq = list(items)
    if chunk_size <= 0 or len(seq) <= chunk_size:
        return [seq] if seq else []
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def warn_if_chunk_exceeds_sge_limit(
    chunk_size: int,
    stages_per_subject: int,
    *,
    max_jobs: int = 1000,
    margin: int = 50,
) -> None:
    need = int(chunk_size) * int(stages_per_subject)
    cap = int(max_jobs) - int(margin)
    if stages_per_subject > 0 and need > cap:
        log.warning(
            f"SGE chunk may exceed per-user job limit: {chunk_size} subjects × "
            f"{stages_per_subject} stages = {need} jobs (cluster cap ~{max_jobs}). "
            f"Reduce --sge-subject-chunk-size to {max(1, cap // stages_per_subject)} or fewer."
        )


def parse_sge_submission_job_ids(stdout: str, stderr: str = "") -> list[str]:
    """Extract SGE job ids from driver-script stdout/stderr (``-> 12345`` lines)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for text in (stdout, stderr):
        for pattern in (_JOB_ID_FROM_ECHO, _BARE_JOB_ID):
            for match in pattern.finditer(text):
                jid = match.group(1).strip()
                if jid.isdigit() and jid not in seen:
                    seen.add(jid)
                    ordered.append(jid)
    return ordered


def query_active_sge_job_ids(
    host: str,
    user: str,
    password: str,
    *,
    port: int = 22,
    connect_timeout: float | None = 30.0,
) -> set[str]:
    """Return numeric job ids currently listed for *user* in ``qstat``."""
    from nvitk.cluster.remote_transfer import ssh_exec

    cmd = f"qstat -u {shlex.quote(user)} 2>/dev/null | awk 'NR>2 {{gsub(/@.*/, \"\", $1); if ($1 ~ /^[0-9]+$/) print $1}}'"
    code, out, err = ssh_exec(
        host=host,
        user=user,
        password=password,
        command=cmd,
        port=port,
        timeout=connect_timeout,
    )
    if code != 0 and not out.strip():
        log.warning(f"qstat query failed (exit {code}): {err.strip() or out.strip()}")
    active: set[str] = set()
    for line in out.splitlines():
        tok = line.strip().split()[0] if line.strip() else ""
        if tok.isdigit():
            active.add(tok)
    return active


def wait_for_sge_job_ids(
    host: str,
    user: str,
    password: str,
    job_ids: Iterable[str],
    *,
    poll_interval: float = 120.0,
    wait_timeout: float | None = None,
    port: int = 22,
    connect_timeout: float | None = 30.0,
) -> bool:
    """Poll ``qstat`` via SSH until none of *job_ids* remain active.

    *wait_timeout* is wall-clock seconds for the whole wait (``None`` = no limit).
    Each poll opens a short SSH session; there is no long-lived connection timeout
    while waiting for jobs to finish.
    """
    ids = [jid for jid in dict.fromkeys(str(j).strip() for j in job_ids) if jid.isdigit()]
    if not ids:
        log.info("SGE chunk: no job ids to wait on.")
        return True

    id_set = set(ids)
    log.info(f"SGE chunk: waiting for {len(id_set)} job(s) to leave the queue …")
    start = time.monotonic()
    poll_n = 0
    while True:
        poll_n += 1
        active = query_active_sge_job_ids(
            host, user, password, port=port, connect_timeout=connect_timeout
        )
        pending = id_set & active
        if not pending:
            log.info(f"SGE chunk: all {len(id_set)} job(s) finished (after {poll_n} poll(s)).")
            return True

        elapsed = time.monotonic() - start
        if wait_timeout is not None and elapsed >= float(wait_timeout):
            log.warning(
                f"SGE chunk wait timeout ({wait_timeout}s): "
                f"{len(pending)}/{len(id_set)} job(s) still active."
            )
            return False

        log.info(
            f"SGE chunk: {len(pending)}/{len(id_set)} job(s) still active "
            f"(poll {poll_n}, elapsed {int(elapsed)}s); sleeping {int(poll_interval)}s …"
        )
        time.sleep(float(poll_interval))


__all__ = [
    "chunk_sequence",
    "parse_sge_submission_job_ids",
    "query_active_sge_job_ids",
    "wait_for_sge_job_ids",
    "warn_if_chunk_exceeds_sge_limit",
]
