"""Tests for reusing an sshfs mount the workstation already has.

A workstation that keeps cluster storage permanently mounted -- ``samwise:/BIOIT_IMAGE`` under
``~/NetVolumes`` -- should not gain a second connection to the same tree just because nvitk
wants a subdirectory of it. These tests pin the matching rules, which have to normalise two
things that differ in practice: the host (an existing mount usually names the cluster by its
ssh alias, while nvitk resolves it to an address) and the username (sshfs omits it from the
device string when it equals the local user).

``/proc/mounts`` is substituted throughout, so nothing here depends on what this machine
happens to have mounted.
"""

from __future__ import annotations

import getpass
from pathlib import Path

import pytest

from nvitk.cluster import sge_json, sshfs

LOCAL_USER = getpass.getuser()

#: The real shape of the entry this feature exists for: ssh alias, no username, trailing slash.
NETVOLUMES = (
    "samwise:/BIOIT_IMAGE/",
    "/home/imarcoss/NetVolumes/BIOIT_IMAGE",
    "fuse.sshfs",
    "rw,nosuid,nodev,relatime,user_id=11503",
)


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    """Fixed alias table, reuse enabled, and every mount considered live."""
    monkeypatch.setattr(
        sge_json, "resolve_host_alias",
        lambda host: {"samwise": "10.149.80.48"}.get(str(host).strip(), str(host).strip()),
    )
    monkeypatch.setattr(sge_json, "sshfs_reuse_existing_mounts", lambda: True)
    monkeypatch.setattr(sshfs, "is_live", lambda *_a, **_k: True)


def set_mounts(monkeypatch, *entries):
    """Substitute ``/proc/mounts`` with *entries* of ``(device, mountpoint, fstype, opts)``."""
    monkeypatch.setattr(sshfs, "_iter_proc_mounts", lambda: iter(entries))


def find(path, host="samwise", user="imarcoss"):
    """Look for an existing mount serving *path*."""
    return sshfs.find_external_mount(host=host, user=user, remote_path=path)


class TestDeviceParsing:
    """``[user@]host:[dir]`` -- the colon splits before the ``@`` does."""

    @pytest.mark.parametrize("device,expected", [
        ("samwise:/BIOIT_IMAGE/", ("", "samwise", "/BIOIT_IMAGE")),
        ("u@10.0.0.1:/a/b", ("u", "10.0.0.1", "/a/b")),
        ("u@host:/path/with@at", ("u", "host", "/path/with@at")),
        ("u@[fe80::1]:/data", ("u", "[fe80::1]", "/data")),
        ("[fe80::1]:/data", ("", "[fe80::1]", "/data")),
        ("host://double//slash/", ("", "host", "/double/slash")),
    ])
    def test_parses(self, device, expected):
        assert sshfs._parse_sshfs_device(device) == expected

    @pytest.mark.parametrize("device", [
        "host:relative/path",   # relative to the remote home; cannot map absolute paths
        "host:",                # the remote home itself
        "/dev/sda1",            # not an sshfs device at all
        "",
    ])
    def test_rejects_unusable_devices(self, device):
        assert sshfs._parse_sshfs_device(device) is None


class TestEnumeration:
    """Reading the sshfs entries out of ``/proc/mounts``."""

    def test_absent_username_defaults_to_the_local_user(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        found = list(sshfs.iter_external_sshfs_mounts())
        assert len(found) == 1
        assert found[0].user == LOCAL_USER
        assert found[0].remote_root == "/BIOIT_IMAGE"
        assert found[0].mountpoint == Path("/home/imarcoss/NetVolumes/BIOIT_IMAGE")

    def test_non_sshfs_filesystems_are_ignored(self, monkeypatch):
        set_mounts(
            monkeypatch,
            ("/dev/sda1", "/", "ext4", "rw"),
            ("tmpfs", "/run", "tmpfs", "rw"),
            NETVOLUMES,
        )
        assert [m.remote_root for m in sshfs.iter_external_sshfs_mounts()] == ["/BIOIT_IMAGE"]

    def test_read_only_is_recorded(self, monkeypatch):
        set_mounts(monkeypatch, ("h:/data", "/mnt/ro", "fuse.sshfs", "ro,nosuid"))
        assert list(sshfs.iter_external_sshfs_mounts())[0].read_only is True


class TestMatching:
    """Which existing mount, if any, serves a given cluster path."""

    def test_the_netvolumes_case(self, monkeypatch):
        """The whole point: alias host, no username, subdirectory of the mounted root."""
        set_mounts(monkeypatch, NETVOLUMES)
        found = find("/BIOIT_IMAGE/nvitk-sge/gui/job-1/output/seg.nii.gz", user=LOCAL_USER)
        assert found is not None
        assert found.owned is False, "a mount nvitk did not make must never be owned"
        assert found.local("/BIOIT_IMAGE/nvitk-sge/gui/job-1/output/seg.nii.gz") == Path(
            "/home/imarcoss/NetVolumes/BIOIT_IMAGE/nvitk-sge/gui/job-1/output/seg.nii.gz"
        )

    def test_host_matches_through_the_alias_table_in_both_directions(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        assert find("/BIOIT_IMAGE/x", host="10.149.80.48", user=LOCAL_USER) is not None
        assert find("/BIOIT_IMAGE/x", host="samwise", user=LOCAL_USER) is not None

    def test_a_different_host_does_not_match(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        assert find("/BIOIT_IMAGE/x", host="other-cluster", user=LOCAL_USER) is None

    def test_a_different_user_does_not_match(self, monkeypatch):
        set_mounts(monkeypatch, ("alice@samwise:/BIOIT_IMAGE", "/mnt/a", "fuse.sshfs", "rw"))
        assert find("/BIOIT_IMAGE/x", user="bob") is None

    def test_a_path_outside_the_mounted_root_does_not_match(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        assert find("/data_lab_MCC/imarcoss/RESULTS", user=LOCAL_USER) is None

    def test_a_sibling_sharing_the_root_prefix_does_not_match(self, monkeypatch):
        """``/BIOIT_IMAGE`` must not be treated as containing ``/BIOIT_IMAGE_OLD``."""
        set_mounts(monkeypatch, NETVOLUMES)
        assert find("/BIOIT_IMAGE_OLD/x", user=LOCAL_USER) is None

    def test_the_narrowest_mount_wins(self, monkeypatch):
        set_mounts(
            monkeypatch,
            NETVOLUMES,
            ("samwise:/BIOIT_IMAGE/nvitk-sge", "/mnt/narrow", "fuse.sshfs", "rw"),
        )
        found = find("/BIOIT_IMAGE/nvitk-sge/gui/job-1", user=LOCAL_USER)
        assert found.mountpoint == Path("/mnt/narrow")

    def test_read_only_mounts_are_skipped(self, monkeypatch):
        """nvitk uploads as well as reads; a read-only reuse would fail later and confusingly."""
        set_mounts(monkeypatch, ("samwise:/BIOIT_IMAGE", "/mnt/ro", "fuse.sshfs", "ro"))
        assert find("/BIOIT_IMAGE/x", user=LOCAL_USER) is None

    def test_a_dead_mount_is_skipped(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        monkeypatch.setattr(sshfs, "is_live", lambda *_a, **_k: False)
        assert find("/BIOIT_IMAGE/x", user=LOCAL_USER) is None

    def test_the_toggle_disables_reuse(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        monkeypatch.setattr(sge_json, "sshfs_reuse_existing_mounts", lambda: False)
        assert find("/BIOIT_IMAGE/x", user=LOCAL_USER) is None


class TestMountIntegration:
    """``mount`` must prefer an existing mount over making its own."""

    def test_mount_returns_the_existing_one_without_running_sshfs(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)

        def fail(*_a, **_k):
            raise AssertionError("ran sshfs despite an existing mount")

        monkeypatch.setattr(sshfs, "require_sshfs", fail)
        monkeypatch.setattr(sshfs, "mountpoint_for", fail)
        handle = sshfs.mount(
            host="samwise", user=LOCAL_USER, password="unused",
            remote_root="/BIOIT_IMAGE/nvitk-sge/gui",
        )
        assert handle.owned is False
        assert handle.mountpoint == Path("/home/imarcoss/NetVolumes/BIOIT_IMAGE")

    def test_a_reused_mount_is_never_unmounted(self, monkeypatch):
        set_mounts(monkeypatch, NETVOLUMES)
        monkeypatch.setattr(sshfs, "_ACTIVE", {})
        monkeypatch.setattr(sshfs, "_REFCOUNTS", {})
        monkeypatch.setattr(sshfs, "_IDLE_SINCE", {})
        monkeypatch.setattr(sge_json, "sshfs_idle_ttl_seconds", lambda: 0.0)
        unmounts: list[Path] = []
        monkeypatch.setattr(sshfs, "unmount", lambda mp, **_k: unmounts.append(mp) or True)

        for _ in range(3):
            with sshfs.cluster_mount(
                host="samwise", user=LOCAL_USER, password="unused",
                remote_root="/BIOIT_IMAGE/nvitk-sge/gui",
            ):
                pass
        sshfs.unmount_all()
        assert unmounts == []
