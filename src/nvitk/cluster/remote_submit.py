"""Optional SSH helper to run an emitted SGE bash script on a login node."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


def run_sge_script_ssh(
    host: str,
    user: str,
    password: str,
    script_path: Path | str,
    *,
    local_script_path: Path | None = None,
    port: int = 22,
    timeout: float | None = None,
) -> bool:
    """Execute ``bash -lc <script>`` on *host* via Paramiko.

    *script_path* is the path on the **cluster** login node. When the script was
    staged locally and uploaded via SFTP, pass *local_script_path* for pre-flight
    summaries (optional).

    Returns ``True`` if the remote session reports exit status 0.
    Does not log passwords. On import/paramiko errors, returns ``False``.
    """
    from nvitk.core.logger import Logger

    log = Logger()
    try:
        import paramiko
    except ImportError:
        log.warning(
            "paramiko is not installed (pip install 'nvitk[cluster]' or paramiko); "
            "skipping SSH. Run the script on the cluster login node manually."
        )
        return False

    remote_script = str(script_path)
    remote_cmd = f"bash {shlex.quote(remote_script)}"

    quiet = os.environ.get("NVITK_QUIET_SGE_SUMMARY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not quiet:
        try:
            from nvitk.cluster.sge import format_sge_driver_script_variables

            summary_src = local_script_path if local_script_path is not None else Path(script_path)
            txt = summary_src.read_text(encoding="utf-8", errors="replace")
            print(
                format_sge_driver_script_variables(txt, Path(remote_script)),
                file=sys.stderr,
                flush=True,
            )
        except OSError as exc:
            log.warning("Could not read SGE driver script for summary: %s", exc)
        print(
            f"[nvitk|SGE] Remote exec via SSH: {user}@{host}:{port} "
            f"bash {shlex.quote(remote_script)}",
            file=sys.stderr,
            flush=True,
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            timeout=timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        _stdin, stdout, stderr = client.exec_command(remote_cmd)
        out_b = stdout.read()
        err_b = stderr.read()
        exit_status = stdout.channel.recv_exit_status()
        if out_b:
            log.info("SSH connection established | status: %s", exit_status)
            log.info("SSH stdout (tail): \n %s", out_b.decode(errors="replace")[-2000:])
        if err_b:
            log.error("SSH connection established | status: %s", exit_status)
            log.error("SSH stderr (tail): \n %s", err_b.decode(errors="replace")[-2000:])
        return exit_status == 0
    except Exception as exc:
        log.warning("SSH exec failed: %s", exc)
        return False
    finally:
        client.close()


__all__ = ["run_sge_script_ssh"]
