"""Optional SSH helper to run an emitted SGE bash script on a login node."""

from __future__ import annotations

import shlex
from pathlib import Path


def try_run_script_via_ssh(
    host: str,
    user: str,
    password: str,
    script_path: Path,
    *,
    port: int = 22,
    timeout: float | None = None,
) -> bool:
    """Execute ``bash -lc <script>`` on *host* via Paramiko.

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

    script_s = str(script_path)
    remote_cmd = f"bash {shlex.quote(script_s)}"

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
            log.info("SSH stdout (tail): %s", out_b.decode(errors="replace")[-2000:])
        if err_b:
            log.info("SSH stderr (tail): %s", err_b.decode(errors="replace")[-2000:])
        return exit_status == 0
    except Exception as exc:
        log.warning("SSH exec failed: %s", exc)
        return False
    finally:
        client.close()


__all__ = ["try_run_script_via_ssh"]
