"""Tests for interactive pyhelp helpers."""

from __future__ import annotations

import subprocess

import pytest

from nvitk.util.pyhelp_tree import run_command_help


def test_run_command_help_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nvitk.util.pyhelp_tree.shutil.which", lambda _c: "/usr/bin/nvitk-gui")

    code = run_command_help("nvitk-gui")
    assert code == 0
    assert calls == [["/usr/bin/nvitk-gui", "--help"]]


def test_run_command_help_missing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nvitk.util.pyhelp_tree.shutil.which", lambda _c: None)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_command_help("no-such-cmd-xyz") == 127
