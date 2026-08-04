"""Generic SGE subject-chunk helpers and job-completion polling.

Pipeline-specific logic (e.g. how many jobs one subject produces) belongs in each
pipeline package, not here.
"""

from __future__ import annotations

import re
import shlex
import time
from collections.abc import Callable
from typing import Iterable, Sequence

from nvitk.core.logger import Logger

log = Logger()

# Terse qsub may print ``12345`` or array forms ``12345.1-7:1``.
_JOB_ID_FROM_ECHO = re.compile(
    r"->\s*(\d+)(?:\.\d+-\d+(?::\d+)?)?\s*$", re.MULTILINE
)
_BARE_JOB_ID = re.compile(
    r"^\s*(\d+)(?:\.\d+-\d+(?::\d+)?)?\s*$", re.MULTILINE
)


def chunk_sequence(items: Sequence[str], chunk_size: int) -> list[list[str]]:
    """Split *items* into chunks of at most *chunk_size* (``<=0`` means one chunk)."""
    seq = list(items)
    if chunk_size <= 0 or len(seq) <= chunk_size:
        return [seq] if seq else []
    return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]


def warn_if_chunk_exceeds_sge_limit(
    chunk_size: int,
    jobs_per_subject: int,
    *,
    max_jobs: int = 1000,
    margin: int = 10,
) -> None:
    """Log a warning if a chunk's total job count (chunk_size × jobs_per_subject) would approach the SGE array-job cap."""
    need = int(chunk_size) * int(jobs_per_subject)
    cap = int(max_jobs) - int(margin)
    if jobs_per_subject > 0 and need > cap:
        log.warning(
            f"SGE chunk may exceed per-user job limit: {chunk_size} subjects × "
            f"{jobs_per_subject} job(s)/subject = {need} jobs (cluster cap ~{max_jobs}). "
            f"Reduce --sge-subject-chunk-size to {max(1, cap // jobs_per_subject)} or fewer."
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


def count_active_sge_jobs(
    host: str,
    user: str,
    password: str,
    *,
    port: int = 22,
    connect_timeout: float | None = 30.0,
) -> int:
    """Return how many jobs *user* currently has in ``qstat``."""
    return len(
        query_active_sge_job_ids(
            host, user, password, port=port, connect_timeout=connect_timeout
        )
    )


def subjects_fit_in_sge_limit(
    active_job_count: int,
    jobs_per_subject: int,
    *,
    max_jobs: int = 1000,
    margin: int = 10,
) -> int:
    """How many whole subjects can be queued without exceeding *max_jobs*.

  *margin* reserves headroom below the cluster cap so a multi-stage driver
  script does not fail on its last ``qsub`` when the queue is near full.
    """
    if jobs_per_subject <= 0:
        return 0
    cap = int(max_jobs) - int(margin)
    free = cap - int(active_job_count)
    if free < jobs_per_subject:
        return 0
    return free // jobs_per_subject


def subject_fits_in_sge_limit(
    active_job_count: int,
    jobs_for_subject: int,
    *,
    max_jobs: int = 1000,
    margin: int = 10,
) -> bool:
    """Return whether *jobs_for_subject* can be queued given *active_job_count*."""
    if jobs_for_subject <= 0:
        return True
    cap = int(max_jobs) - int(margin)
    return int(active_job_count) + int(jobs_for_subject) <= cap


def drip_submit_subjects(
    pending_subjects: Sequence[str],
    submit_subject: Callable[[str], tuple[bool, list[str]]],
    *,
    host: str,
    user: str,
    password: str,
    jobs_per_subject: int | Callable[[str], int],
    poll_interval: float = 120.0,
    max_jobs: int = 1000,
    margin: int = 10,
    loop_timeout: float | None = None,
    serial: bool = False,
    wait_timeout: float | None = None,
    port: int = 22,
    connect_timeout: float | None = 30.0,
) -> bool:
    """Submit subjects when the user's queue has enough free slots.

    *submit_subject(subject) -> (ok, job_ids)* runs the remote driver for one
    subject and must return ``(False, [])`` on a fatal submission error. Each
    call queues all jobs for that subject before the next capacity check.

    When *serial* is true, each subject is submitted only after the previous
    subject's *job_ids* have left ``qstat`` (one subject in flight at a time).
    """
    queue = list(pending_subjects)
    if not queue:
        return True

    jobs_resolver: Callable[[str], int]
    if callable(jobs_per_subject):
        jobs_resolver = jobs_per_subject
        jobs_desc = "variable job(s)/subject"
    else:
        fixed = int(jobs_per_subject)
        jobs_resolver = lambda _subj: fixed  # noqa: E731
        jobs_desc = f"{fixed} job(s)/subject"

    mode = "serial" if serial else f"cap {max_jobs}, margin {margin}"
    log.info(
        f"SGE drip ({mode}): {len(queue)} subject(s) queued after initial batch "
        f"({jobs_desc})"
    )

    if serial:
        start = time.monotonic()
        while queue:
            subj = queue[0]
            next_jobs = jobs_resolver(subj)
            if next_jobs <= 0:
                queue.pop(0)
                log.info(
                    f"SGE drip: {subj!r} has no pending stages (--skip-processed); skipping."
                )
                continue

            elapsed = time.monotonic() - start
            if loop_timeout is not None and elapsed >= float(loop_timeout):
                log.warning(
                    f"SGE drip timeout ({loop_timeout}s): "
                    f"{len(queue)} subject(s) still pending."
                )
                return False

            log.info(
                f"SGE drip (serial): submitting {subj!r} "
                f"({next_jobs} job(s), {len(queue) - 1} subject(s) left after this)"
            )
            queue.pop(0)
            ok, job_ids = submit_subject(subj)
            if not ok:
                log.error(f"SGE drip: submission failed for {subj!r}; stopping.")
                return False
            if job_ids and not wait_for_sge_job_ids(
                host,
                user,
                password,
                job_ids,
                poll_interval=poll_interval,
                wait_timeout=wait_timeout,
                port=port,
                connect_timeout=connect_timeout,
            ):
                log.error(
                    f"SGE drip: timed out waiting for {subj!r} job(s); stopping."
                )
                return False

        log.info("SGE drip: all subjects submitted.")
        return True

    start = time.monotonic()
    poll_n = 0
    while queue:
        poll_n += 1
        active = count_active_sge_jobs(
            host, user, password, port=port, connect_timeout=connect_timeout
        )
        next_jobs = jobs_resolver(queue[0])
        if next_jobs <= 0:
            subj = queue.pop(0)
            log.info(
                f"SGE drip: {subj!r} has no pending stages (--skip-processed); skipping."
            )
            continue

        if subject_fits_in_sge_limit(
            active, next_jobs, max_jobs=max_jobs, margin=margin
        ):
            subj = queue.pop(0)
            log.info(
                f"SGE drip: submitting {subj!r} ({next_jobs} job(s), "
                f"{active}/{max_jobs} active, {len(queue)} subject(s) left)"
            )
            ok, _job_ids = submit_subject(subj)
            if not ok:
                log.error(f"SGE drip: submission failed for {subj!r}; stopping.")
                return False
            continue

        elapsed = time.monotonic() - start
        if loop_timeout is not None and elapsed >= float(loop_timeout):
            log.warning(
                f"SGE drip timeout ({loop_timeout}s): "
                f"{len(queue)} subject(s) still pending ({active}/{max_jobs} jobs active)."
            )
            return False

        log.info(
            f"SGE drip: {len(queue)} subject(s) pending, {active}/{max_jobs} jobs active "
            f"(poll {poll_n}, elapsed {int(elapsed)}s); sleeping {int(poll_interval)}s …"
        )
        time.sleep(float(poll_interval))

    log.info("SGE drip: all subjects submitted.")
    return True


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
    "count_active_sge_jobs",
    "drip_submit_subjects",
    "parse_sge_submission_job_ids",
    "query_active_sge_job_ids",
    "subject_fits_in_sge_limit",
    "subjects_fit_in_sge_limit",
    "wait_for_sge_job_ids",
    "warn_if_chunk_exceeds_sge_limit",
]
