"""Regression tests for how long an sshfs mount is kept alive.

A GUI cluster launch polls ``output/.done`` every five seconds, and each poll opens its own
short ``cluster_mount`` block. The first implementation unmounted as soon as the refcount hit
zero, so every tick paid a full SSH handshake and FUSE setup -- one to three seconds of work
and two log lines, forever, while a job ran::

    18:50:40 | Mounting cluster storage over sshfs: ... -> ...
    18:50:41 | sshfs mounted: ... -> ...
    18:50:45 | Mounting cluster storage over sshfs: ... -> ...

Refcounting alone cannot fix that: the blocks are sequential, never concurrent. These tests
pin the idle grace period that does, and the refcounting it sits on top of.

No cluster is involved -- ``mount``/``unmount``/``is_live`` are substituted so the tests
measure the lifecycle policy rather than sshfs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nvitk.cluster import sge_json, sshfs

ROOT = "/BIOIT_IMAGE/nvitk-sge/gui"
OTHER_ROOT = "/BIOIT_IMAGE/nvitk-sge/SGE_SCRIPTS"


class Recorder:
    """Counts mount/unmount calls in place of the real ones."""

    def __init__(self) -> None:
        self.mounted = 0
        self.unmounted = 0

    def mount(self, *, host, user, password, remote_root, options=None, timeout=None):
        self.mounted += 1
        return sshfs.SshfsMount(
            host=host, user=user, remote_root=remote_root,
            mountpoint=Path(f"/tmp/fake-sshfs/{self.mounted}"), owned=True,
        )

    def unmount(self, mountpoint, **_kwargs) -> bool:
        self.unmounted += 1
        return True


@pytest.fixture
def recorder(monkeypatch):
    """Substitute the mount primitives and isolate the module's registry per test."""
    rec = Recorder()
    monkeypatch.setattr(sshfs, "mount", rec.mount)
    monkeypatch.setattr(sshfs, "unmount", rec.unmount)
    monkeypatch.setattr(sshfs, "is_live", lambda *_a, **_k: True)
    monkeypatch.setattr(sshfs, "_ACTIVE", {})
    monkeypatch.setattr(sshfs, "_REFCOUNTS", {})
    monkeypatch.setattr(sshfs, "_IDLE_SINCE", {})
    return rec


def hold(root: str = ROOT):
    """One ``cluster_mount`` block, the shape a poll or a transfer uses."""
    return sshfs.cluster_mount(host="h", user="u", password="p", remote_root=root)


def poll(root: str = ROOT) -> None:
    """Acquire and release, as the GUI job monitor does every five seconds."""
    with hold(root):
        pass


class TestIdleReuse:
    """Sequential blocks inside the TTL must reuse the mount, not rebuild it."""

    def test_repeated_polls_mount_once(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        for _ in range(12):
            poll()
        assert (recorder.mounted, recorder.unmounted) == (1, 0)

    def test_the_same_mountpoint_is_handed_back(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        with hold() as first:
            first_point = first.mountpoint
        with hold() as second:
            assert second.mountpoint == first_point

    def test_distinct_roots_get_distinct_mounts(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        poll(ROOT)
        poll(OTHER_ROOT)
        poll(ROOT)
        assert recorder.mounted == 2

    def test_an_idle_mount_still_counts_as_active_for_the_leak_guard(
        self, recorder, monkeypatch
    ):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        poll()
        assert len(sshfs.active_mounts()) == 1


class TestRefcounting:
    """Concurrent holders keep a single mount; the last one out starts the idle clock."""

    def test_nesting_mounts_once(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        with hold():
            with hold():
                with hold():
                    pass
        assert (recorder.mounted, recorder.unmounted) == (1, 0)

    def test_a_held_mount_is_never_swept(self, recorder, monkeypatch):
        """TTL 0 must not pull the filesystem out from under an open block."""
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 0.0)
        with hold():
            poll()  # an inner acquire/release runs a sweep
            assert recorder.unmounted == 0, "swept a mount that was still held"
            assert recorder.mounted == 1, "remounted a mount that was still held"
        # Releasing only starts the idle clock; the sweep runs on the next acquisition, so
        # an operation that finishes and never comes back leaves cleanup to atexit.
        assert recorder.unmounted == 0
        poll()
        assert recorder.unmounted == 1


class TestIdleExpiry:
    """The grace period has to end, or a long-lived process hoards mounts."""

    def test_ttl_zero_unmounts_on_the_next_acquisition(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 0.0)
        poll()
        assert recorder.unmounted == 0, "release must not unmount inline"
        poll()
        assert recorder.unmounted == 1
        assert recorder.mounted == 2

    def test_an_expired_idle_mount_is_swept(self, recorder, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 60.0)
        monkeypatch.setattr(sshfs.time, "monotonic", lambda: clock["now"])
        poll()
        clock["now"] += 61.0
        poll()
        assert (recorder.mounted, recorder.unmounted) == (2, 1)

    def test_a_mount_used_within_the_ttl_is_kept(self, recorder, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 60.0)
        monkeypatch.setattr(sshfs.time, "monotonic", lambda: clock["now"])
        for _ in range(10):
            clock["now"] += 5.0  # the GUI's poll interval
            poll()
        assert (recorder.mounted, recorder.unmounted) == (1, 0)


class TestOwnership:
    """A mount this process did not create is reused but never torn down."""

    def test_an_unowned_mount_is_not_unmounted(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 0.0)
        monkeypatch.setattr(
            sshfs, "mount",
            lambda **kw: sshfs.SshfsMount(
                host=kw["host"], user=kw["user"], remote_root=kw["remote_root"],
                mountpoint=Path("/tmp/pre-existing"), owned=False,
            ),
        )
        poll()
        poll()
        assert recorder.unmounted == 0

    def test_unmount_all_releases_held_and_idle_mounts(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        poll(ROOT)
        poll(OTHER_ROOT)
        sshfs.unmount_all()
        assert recorder.unmounted == 2
        assert sshfs.active_mounts() == []


class TestStaleMount:
    """A mount whose connection died must be replaced, not handed back."""

    def test_a_dead_cached_mount_is_remounted(self, recorder, monkeypatch):
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 300.0)
        poll()
        monkeypatch.setattr(sshfs, "is_live", lambda *_a, **_k: False)
        poll()
        assert recorder.mounted == 2
        assert recorder.unmounted == 1
