"""
TensorBoard mirroring and serving for nnU-Net / nnssl style training runs.

Description
-----------
Neither nnU-Net nor nnssl writes TensorBoard events. Both do write, once per epoch, a plain
text ``training_log_<timestamp>.txt`` inside every ``fold_*`` directory, in an identical
format (both derive from the same ``print_to_log_file`` helper)::

    2026-08-26 17:03:11.123456: Epoch 12
    2026-08-26 17:03:11.223456: Current learning rate: 0.00931
    2026-08-26 17:08:11.323456: train_loss -0.6412
    2026-08-26 17:08:11.423456: val_loss -0.5883
    2026-08-26 17:08:11.523456: Pseudo dice [np.float64(0.8123), np.float64(0.0)]
    2026-08-26 17:08:11.623456: Epoch time: 301.4 s

This module *mirrors* those logs into TensorBoard event files. Mirroring rather than
instrumenting the trainers is deliberate:

- the frameworks stay unpatched — nnssl is a vendored upstream clone and the in-tree nnU-Net
  build is already carrying enough local changes;
- it works whether the trainer runs in-process (stage 1) or as a subprocess (stage 2), and
  whether that process is on this host or on a cluster node;
- it works **after the fact** and **across a shared filesystem**: a run on the cluster can be
  watched live from the workstation by pointing the mirror at the mounted results root;
- nothing is lost — the per-epoch text log carries every value the frameworks' own loggers
  hold (``train_losses``, ``val_losses``, ``lrs``, per-class pseudo dice, epoch duration).

Concurrency
-----------
Two mirrors writing the same event directory would double every curve. A heartbeat lock file
in the event root makes the second one stand down (:class:`MirrorLock`); it is advisory and
NFS-safe enough for its purpose, since the loser degrades to "serve only".

Emitted scalars
---------------
============================  ==========================================================
``loss/train``                per-epoch training loss
``loss/val``                  per-epoch validation loss
``lr``                        learning rate at the start of the epoch
``dice/pseudo_mean``          mean of the per-class pseudo dice (nnU-Net only)
``dice/pseudo_ema``           nnU-Net's EMA of the above (``0.9 * prev + 0.1 * new``)
``dice_per_class/<name>``     one series per foreground class, named if names are supplied
``time/epoch_seconds``        wall-clock epoch duration
============================  ==========================================================

Steps are epoch indices and the event ``walltime`` is the real timestamp of the epoch's last
log line, so TensorBoard's *Relative* / *Wall* axes stay meaningful.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import click

from nvitk.core.logger import Logger

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Filesystem conventions
# ──────────────────────────────────────────────────────────────────────────────

#: Per-epoch text log both frameworks write inside every ``fold_*`` directory.
TRAINING_LOG_GLOB: str = "training_log_*.txt"

#: Event files written by ``SummaryWriter`` — removed on a full rebuild.
EVENT_GLOB: str = "events.out.tfevents.*"

#: Mirror bookkeeping, one per event directory (byte offsets + last epoch written).
STATE_FILE: str = "_nvitk_tb_state.json"

#: Heartbeat lock, one per event *root* — see :class:`MirrorLock`.
LOCK_FILE: str = "_nvitk_tb_mirror.lock.json"

#: Written by :class:`TensorBoardServer` so a client on another host can find the server.
SERVER_SIDECAR: str = "tensorboard_server.json"

#: Run directories, relative to a framework results root: ``<dataset>/<run>/fold_<n>``.
DEFAULT_RUN_PATTERN: str = "*/*/fold_*"

#: Default seconds between mirror passes. Epochs take minutes, so polling faster only costs
#: filesystem round-trips on a shared mount.
DEFAULT_INTERVAL: float = 20.0

#: Default TensorBoard port; :func:`resolve_port` walks upwards if it is taken.
DEFAULT_PORT: int = 6006


# ──────────────────────────────────────────────────────────────────────────────
# Log parsing
# ──────────────────────────────────────────────────────────────────────────────

#: ``2026-08-26 17:03:11.123456: <body>`` — the timestamp both frameworks prefix.
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?):\s?(?P<body>.*)$"
)

_EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s*$")
_LR_RE = re.compile(r"^Current learning rate:\s*(\S+)")
_TRAIN_LOSS_RE = re.compile(r"^train_loss\s+(\S+)")
_VAL_LOSS_RE = re.compile(r"^val_loss\s+(\S+)")
_PSEUDO_DICE_RE = re.compile(r"^Pseudo dice\s+\[(?P<items>.*)\]")
_EPOCH_TIME_RE = re.compile(r"^Epoch time:\s*(\S+)\s*s")

#: ``np.float64(0.1234)`` — NumPy >= 2 repr of the elements of the ``Pseudo dice`` list.
_NP_SCALAR_RE = re.compile(r"^np\.\w+\((?P<value>[^)]*)\)$")


def _as_float(token: str) -> float:
    """Parse one logged number, unwrapping a NumPy scalar repr; ``nan`` when unparseable.

    ``str(np.float64(0.1))`` is ``'0.1'`` on its own but ``'np.float64(0.1)'`` inside a list,
    and both spellings appear in the same log file (scalars vs the ``Pseudo dice`` list).
    """
    text = token.strip().rstrip(",")
    match = _NP_SCALAR_RE.match(text)
    if match is not None:
        text = match["value"].strip()
    try:
        return float(text)
    except ValueError:
        return math.nan


def _as_timestamp(text: str) -> float | None:
    """Log timestamp → POSIX seconds, or ``None`` if it does not parse."""
    try:
        return datetime.fromisoformat(text.replace("T", " ")).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class EpochRecord:
    """One completed epoch, as recovered from a training log."""

    epoch: int
    """Epoch index as the trainer counts it — becomes the TensorBoard step."""

    walltime: float | None
    """POSIX timestamp of the epoch's last log line, or ``None`` if unparsed."""

    scalars: dict[str, float] = field(default_factory=dict)
    """Tag → value for the single-valued series (``loss/train``, ``lr``, …)."""

    per_class_dice: tuple[float, ...] = ()
    """Pseudo dice per foreground class, in the trainer's label order (nnU-Net only)."""


def parse_training_log(chunk: bytes) -> tuple[list[EpochRecord], int]:
    """Parse *chunk* into completed epochs and report how many bytes were consumed.

    An epoch is only emitted once its ``Epoch time:`` line has been seen, which both trainers
    write last in ``on_epoch_end`` — so a record is never published half-filled.

    Returns
    -------
    records
        Completed epochs, in file order.
    consumed
        Byte offset just past the last completed record. A partially written epoch at the tail
        is deliberately *not* consumed, so the next pass re-reads and completes it. Callers
        must store this offset rather than the file size.
    """
    records: list[EpochRecord] = []
    consumed = 0
    offset = 0

    epoch: int | None = None
    scalars: dict[str, float] = {}
    per_class: tuple[float, ...] = ()
    walltime: float | None = None

    # Split on bytes so the offset arithmetic stays exact regardless of decoding.
    for raw in chunk.split(b"\n")[:-1]:  # the tail after the last \n is an incomplete line
        offset += len(raw) + 1
        line = raw.decode("utf-8", errors="replace").rstrip()
        match = _LINE_RE.match(line)
        if match is None:
            continue
        body, stamp = match["body"].strip(), _as_timestamp(match["ts"])

        start = _EPOCH_RE.match(body)
        if start is not None:
            # A new header abandons whatever was pending: the previous epoch never finished.
            epoch, scalars, per_class, walltime = int(start[1]), {}, (), stamp
            continue
        if epoch is None:
            continue  # preamble (plans dump, split info, …) before the first epoch

        walltime = stamp or walltime
        if (hit := _LR_RE.match(body)) is not None:
            scalars["lr"] = _as_float(hit[1])
        elif (hit := _TRAIN_LOSS_RE.match(body)) is not None:
            scalars["loss/train"] = _as_float(hit[1])
        elif (hit := _VAL_LOSS_RE.match(body)) is not None:
            scalars["loss/val"] = _as_float(hit[1])
        elif (hit := _PSEUDO_DICE_RE.match(body)) is not None:
            items = [t for t in hit["items"].split(",") if t.strip()]
            per_class = tuple(_as_float(t) for t in items)
        elif (hit := _EPOCH_TIME_RE.match(body)) is not None:
            scalars["time/epoch_seconds"] = _as_float(hit[1])
            records.append(
                EpochRecord(
                    epoch=epoch, walltime=walltime, scalars=dict(scalars),
                    per_class_dice=per_class,
                )
            )
            consumed = offset  # everything up to here is safely replayed
            epoch, scalars, per_class = None, {}, ()

    return records, consumed


def iter_training_logs(run_dir: Path) -> list[Path]:
    """Training logs in *run_dir*, oldest first.

    The frameworks name them ``training_log_<Y>_<m>_<d>_<H>_<M>_<S>.txt`` with **unpadded**
    month and day, so a lexicographic sort puts October before September. Sorted on the parsed
    numbers instead, falling back to mtime for anything that does not match.
    """
    def key(path: Path) -> tuple[int, ...]:
        """Sort key: the six timestamp fields, or the mtime when they are absent."""
        parts = path.stem.split("_")[2:]
        try:
            return (0, *(int(p) for p in parts))
        except ValueError:
            return (1, int(path.stat().st_mtime))

    return sorted(Path(run_dir).glob(TRAINING_LOG_GLOB), key=key)


def discover_run_dirs(root: Path, *, pattern: str = DEFAULT_RUN_PATTERN) -> list[Path]:
    """Fold directories under *root* that already hold at least one training log.

    Both frameworks lay out ``<results_root>/<dataset>/<trainer>__<plans>__<config>/fold_<n>``,
    which is what :data:`DEFAULT_RUN_PATTERN` matches. Directories without a log are skipped so
    a run that has only just been created does not appear as an empty TensorBoard run.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.glob(pattern)
        if d.is_dir() and any(d.glob(TRAINING_LOG_GLOB))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Event writing
# ──────────────────────────────────────────────────────────────────────────────


def summary_writer_class() -> Callable[..., Any]:
    """The available ``SummaryWriter`` implementation.

    Raises
    ------
    ImportError
        Naming the package to install. ``torch.utils.tensorboard`` is only a thin shim over
        the ``tensorboard`` package and raises a far less actionable error on its own.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except ImportError:
        pass
    try:
        from tensorboardX import SummaryWriter  # type: ignore[no-redef]

        return SummaryWriter
    except ImportError:
        raise ImportError(
            "TensorBoard support needs the 'tensorboard' package (or 'tensorboardX'). "
            "Install it with: pip install tensorboard"
        ) from None


def tensorboard_available() -> bool:
    """Whether events can be written and a server started in this environment."""
    try:
        summary_writer_class()
    except ImportError:
        return False
    return True


class TensorBoardMirror:
    """Mirror one training run's text log into one TensorBoard run directory.

    Incremental: each pass reads only the bytes appended since the last one. Resuming a run
    (``--c``) opens a *new* log file that carries on from the last epoch, and is mirrored as
    one continuous curve. Two situations instead force a **rebuild**, in which the event files
    are deleted and written again:

    - a source log shrank (truncated or replaced) — everything is replayed;
    - an epoch index went backwards, which means the training was restarted from scratch. Only
      the logs from that restart onwards are replayed: the earlier ones describe a run whose
      checkpoints the framework has already overwritten, and splicing the two together would
      produce a curve that never existed. The original text logs are untouched on disk.

    Parameters
    ----------
    source_dir
        The framework's ``fold_*`` directory, holding ``training_log_*.txt``.
    event_dir
        Where to write event files. Its path relative to the TensorBoard ``--logdir`` is what
        the UI shows as the run name.
    class_names
        Names for the per-class pseudo dice series, in the trainer's label order. Without
        them the series are numbered ``class_00``, ``class_01``, … — usable, but on a 36-class
        vessel model the names are the whole point.
    """

    def __init__(
        self,
        source_dir: Path,
        event_dir: Path,
        *,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.event_dir = Path(event_dir)
        self.class_names = tuple(class_names) if class_names else ()
        self._writer: Any | None = None
        self._state: dict[str, Any] = {"consumed": {}, "last_epoch": -1}
        self._loaded = False

    # ---- state ---------------------------------------------------------------

    @property
    def _state_path(self) -> Path:
        """Bookkeeping sidecar inside the event directory."""
        return self.event_dir / STATE_FILE

    def _load_state(self) -> None:
        """Read the sidecar once, tolerating a corrupt one by starting over."""
        if self._loaded:
            return
        self._loaded = True
        if self._state_path.is_file():
            try:
                self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("Unreadable mirror state at %s; rebuilding.", self._state_path)
                self._state = {"consumed": {}, "last_epoch": -1}

    def _save_state(self) -> None:
        """Persist the sidecar."""
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state, indent=2) + "\n", encoding="utf-8")

    def _reset(self) -> None:
        """Drop every event file and all bookkeeping for this run."""
        self.close()
        for event in self.event_dir.glob(EVENT_GLOB):
            event.unlink(missing_ok=True)
        self._state = {"consumed": {}, "last_epoch": -1}

    # ---- writing -------------------------------------------------------------

    @property
    def writer(self) -> Any:
        """Lazily opened ``SummaryWriter`` for this run."""
        if self._writer is None:
            self.event_dir.mkdir(parents=True, exist_ok=True)
            self._writer = summary_writer_class()(log_dir=str(self.event_dir))
        return self._writer

    def _class_tag(self, index: int) -> str:
        """Series name for the *index*-th foreground class."""
        if index < len(self.class_names):
            return f"dice_per_class/{index + 1:02d}_{self.class_names[index]}"
        return f"dice_per_class/class_{index:02d}"

    def _write(self, record: EpochRecord, ema: float | None) -> float | None:
        """Write one epoch; returns the updated pseudo-dice EMA."""
        writer = self.writer
        for tag, value in record.scalars.items():
            writer.add_scalar(tag, value, record.epoch, walltime=record.walltime)

        if record.per_class_dice:
            finite = [v for v in record.per_class_dice if not math.isnan(v)]
            for index, value in enumerate(record.per_class_dice):
                writer.add_scalar(
                    self._class_tag(index), value, record.epoch, walltime=record.walltime
                )
            if finite:
                mean = sum(finite) / len(finite)
                # nnU-Net's own definition (nnUNetLogger.log, 'ema_fg_dice'): the value its
                # "best" checkpoint is selected on, so it belongs on the same chart.
                ema = mean if ema is None else 0.9 * ema + 0.1 * mean
                writer.add_scalar(
                    "dice/pseudo_mean", mean, record.epoch, walltime=record.walltime
                )
                writer.add_scalar(
                    "dice/pseudo_ema", ema, record.epoch, walltime=record.walltime
                )
        return ema

    # ---- driving -------------------------------------------------------------

    def sync(self) -> int:
        """Mirror everything appended since the last call; returns the epochs written."""
        self._load_state()
        sources = iter_training_logs(self.source_dir)
        if not sources:
            return 0

        # ---- 1. Truncation check -> rebuild --------------------------------
        consumed: dict[str, int] = dict(self._state.get("consumed", {}))
        for path in sources:
            offset = int(consumed.get(path.name, 0))
            if offset and path.stat().st_size < offset:
                log.info("Mirror: %s shrank; rebuilding %s.", path.name, self.event_dir.name)
                self._reset()
                consumed = {}
                break

        # ---- 2. Replay the appended bytes of each log ----------------------
        for attempt in (1, 2):
            written, restart_at = self._sync_once(sources, consumed)
            if restart_at is None or attempt == 2:
                break
            # ---- 3. Epoch went backwards -> training restarted from scratch --
            log.info(
                "Mirror: %s restarts the epoch count; rebuilding %s from it and dropping the "
                "%d earlier log(s), which describe a superseded run.",
                sources[restart_at].name, self.event_dir.name, restart_at,
            )
            self._reset()
            consumed = {p.name: p.stat().st_size for p in sources[:restart_at]}
        self._state["consumed"] = consumed
        if written:
            self.writer.flush()
            self._save_state()
        return written

    def _sync_once(
        self, sources: Sequence[Path], consumed: dict[str, int]
    ) -> tuple[int, int | None]:
        """One replay pass.

        Returns
        -------
        written
            Epochs written to the event file.
        restart_at
            Index into *sources* of the log whose epoch numbering went backwards, or ``None``
            when the pass was clean. The caller rebuilds from that log — see :meth:`sync`.
        """
        written = 0
        ema = self._state.get("ema")
        last_epoch = int(self._state.get("last_epoch", -1))

        for index, path in enumerate(sources):
            offset = int(consumed.get(path.name, 0))
            with open(path, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
            if not chunk:
                continue
            records, used = parse_training_log(chunk)
            for record in records:
                if record.epoch <= last_epoch:
                    return written, index
                ema = self._write(record, ema)
                last_epoch = record.epoch
                written += 1
            consumed[path.name] = offset + used

        self._state["last_epoch"] = last_epoch
        self._state["ema"] = ema
        return written, None

    def close(self) -> None:
        """Close the writer, if one was opened."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ──────────────────────────────────────────────────────────────────────────────
# Advisory lock
# ──────────────────────────────────────────────────────────────────────────────


class MirrorLock:
    """Heartbeat lock making a second mirror of the same event root stand down.

    The stage job on a cluster node and an interactive session on the workstation can easily
    end up watching the same shared directory. Both writing would double every curve, so the
    later one degrades to read-only. Advisory by design: a stale heartbeat (owner killed) is
    simply taken over after *ttl* seconds.
    """

    def __init__(self, path: Path, *, ttl: float = 120.0) -> None:
        self.path = Path(path)
        self.ttl = float(ttl)
        self.owned = False
        #: Identifies this holder specifically. A pid is not enough: two watchers can live in
        #: one process (a pipeline run that serves as well as mirrors), and they must not each
        #: believe they hold the lock.
        self.owner = uuid.uuid4().hex
        self._warned = False

    @property
    def _identity(self) -> dict[str, Any]:
        """This holder's ownership record."""
        return {
            "owner": self.owner, "host": socket.gethostname(), "pid": os.getpid(),
            "heartbeat": time.time(),
        }

    def _holder(self) -> dict[str, Any] | None:
        """The current holder, or ``None`` when free or stale."""
        try:
            held = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(held.get("heartbeat", 0)) > self.ttl:
            return None
        return held

    def acquire(self) -> bool:
        """Claim the lock; ``False`` if a live mirror already holds it.

        Safe to retry: a holder whose heartbeat has gone stale (job killed, node lost) is
        taken over, so a standby mirror eventually picks the work up.
        """
        held = self._holder()
        if held is not None and held.get("owner") != self.owner:
            if not self._warned:
                log.info(
                    "TensorBoard mirror already running on %s (pid %s); this process stays on "
                    "standby and will take over if it stops.", held.get("host"), held.get("pid"),
                )
                self._warned = True
            return False
        self.refresh()
        self.owned = True
        self._warned = False
        return True

    def refresh(self) -> None:
        """Renew the heartbeat."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.write_text(json.dumps(self._identity) + "\n", encoding="utf-8")
        except OSError as exc:  # a read-only or full mount must not kill a training run
            log.debug("Could not refresh the TensorBoard mirror lock (%s).", exc)

    def release(self) -> None:
        """Drop the lock if this process holds it."""
        if self.owned:
            self.path.unlink(missing_ok=True)
            self.owned = False


# ──────────────────────────────────────────────────────────────────────────────
# Background watcher
# ──────────────────────────────────────────────────────────────────────────────


class TensorBoardWatcher:
    """Background thread mirroring every run under one or more framework results roots.

    Runs discovered on each pass, so a fold directory created after the watcher started is
    picked up without a restart — which is exactly what happens when stage 2 begins fold 1.

    Parameters
    ----------
    sources
        Label → framework results root. The label becomes the first path component of the
        TensorBoard run name, e.g. ``stage2/Dataset501_…/PretrainedTrainer__…/fold_0``.
    event_root
        The directory a TensorBoard server is pointed at.
    class_names
        Forwarded to every :class:`TensorBoardMirror` — see there.
    interval
        Seconds between passes.

    Examples
    --------
    >>> with TensorBoardWatcher({"stage2": results}, event_root=tb_dir):  # doctest: +SKIP
    ...     train()
    """

    def __init__(
        self,
        sources: Mapping[str, Path],
        event_root: Path,
        *,
        class_names: Sequence[str] | None = None,
        interval: float = DEFAULT_INTERVAL,
        run_pattern: str = DEFAULT_RUN_PATTERN,
    ) -> None:
        self.sources = {str(k): Path(v) for k, v in sources.items()}
        self.event_root = Path(event_root)
        self.class_names = tuple(class_names) if class_names else ()
        self.interval = float(interval)
        self.run_pattern = run_pattern
        self.lock = MirrorLock(self.event_root / LOCK_FILE, ttl=max(4 * self.interval, 120.0))
        self._mirrors: dict[Path, TensorBoardMirror] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- one pass ------------------------------------------------------------

    def _mirror_for(self, label: str, root: Path, run_dir: Path) -> TensorBoardMirror:
        """The mirror for *run_dir*, created on first sight."""
        if run_dir not in self._mirrors:
            relative = run_dir.relative_to(root)
            self._mirrors[run_dir] = TensorBoardMirror(
                run_dir, self.event_root / label / relative, class_names=self.class_names
            )
            log.info("TensorBoard run: %s/%s", label, relative)
        return self._mirrors[run_dir]

    def sync_once(self) -> int:
        """Mirror every discovered run once; returns the total epochs written."""
        written = 0
        for label, root in self.sources.items():
            for run_dir in discover_run_dirs(root, pattern=self.run_pattern):
                try:
                    written += self._mirror_for(label, root, run_dir).sync()
                except Exception as exc:  # never let monitoring abort a training run
                    log.warning("TensorBoard mirror failed for %s (%s).", run_dir, exc)
        return written

    # ---- thread lifecycle ----------------------------------------------------

    def _loop(self) -> None:
        """Poll until stopped, mirroring only while this process holds the lock.

        A pass that cannot take the lock does nothing but retry it, so a watcher started
        alongside a live mirror sits on standby and takes over if that mirror disappears.
        """
        while not self._stop.is_set():
            if self.lock.owned or self.lock.acquire():
                self.lock.refresh()
                self.sync_once()
            self._stop.wait(self.interval)
        if self.lock.owned:
            self.sync_once()  # final pass so the last epochs are not lost

    def start(self) -> bool:
        """Start the thread; returns whether this process is the one writing events.

        A ``False`` return is not a failure — the thread runs either way, on standby.
        """
        self.event_root.mkdir(parents=True, exist_ok=True)
        owned = self.lock.acquire()
        self._thread = threading.Thread(
            target=self._loop, name="nvitk-tensorboard-mirror", daemon=True
        )
        self._thread.start()
        if owned:
            log.info("TensorBoard mirror watching %s -> %s",
                     ", ".join(str(p) for p in self.sources.values()), self.event_root)
        return owned

    def stop(self) -> None:
        """Stop the thread, flush and close every writer, release the lock."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2 * self.interval, 30.0))
            self._thread = None
        for mirror in self._mirrors.values():
            mirror.close()
        self.lock.release()

    def __enter__(self) -> TensorBoardWatcher:
        """Start mirroring."""
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop mirroring."""
        self.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────────────────────


def resolve_port(preferred: int = DEFAULT_PORT, *, host: str = "", tries: int = 32) -> int:
    """First free TCP port at or above *preferred*.

    Raises
    ------
    RuntimeError
        If *tries* consecutive ports are all taken.
    """
    for port in range(int(preferred), int(preferred) + int(tries)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host or "", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port in [{preferred}, {preferred + tries}).")


class TensorBoardServer:
    """A ``tensorboard`` process serving *logdir*, as a context manager.

    Started as ``python -m tensorboard.main`` rather than through the ``tensorboard`` console
    script, which is not on ``PATH`` inside the Singularity container.

    Parameters
    ----------
    bind_all
        Listen on every interface. Required when the server runs on a cluster node and the
        browser is elsewhere; never use it to expose a workstation.
    sidecar
        Where to record the resolved host, port, URL and pid. On a cluster the compute node is
        only known once the job starts, so this file — written to the shared filesystem — is
        how the workstation learns where to tunnel.
    """

    def __init__(
        self,
        logdir: Path,
        *,
        port: int = DEFAULT_PORT,
        host: str = "127.0.0.1",
        bind_all: bool = False,
        reload_interval: int = 30,
        sidecar: Path | None = None,
    ) -> None:
        self.logdir = Path(logdir)
        self.port = int(port)
        self.host = "0.0.0.0" if bind_all else host
        self.bind_all = bool(bind_all)
        self.reload_interval = int(reload_interval)
        self.sidecar = Path(sidecar) if sidecar is not None else None
        self.process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        """Where to point a browser (the node's own hostname when bound to all interfaces)."""
        host = socket.gethostname() if self.bind_all else self.host
        return f"http://{host}:{self.port}"

    def _write_sidecar(self) -> None:
        """Record how to reach this server, for a client on another host."""
        if self.sidecar is None:
            return
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar.write_text(
            json.dumps(
                {
                    "url": self.url, "hostname": socket.gethostname(), "port": self.port,
                    "bind_all": self.bind_all, "logdir": str(self.logdir),
                    "pid": self.process.pid if self.process else None,
                    "started": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    def start(self) -> str:
        """Launch the server; returns its URL."""
        summary_writer_class()  # fail here, with an installable package name, not in Popen
        self.logdir.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(self.logdir),
            "--port", str(self.port),
            "--reload_interval", str(self.reload_interval),
        ]
        argv.extend(["--bind_all"] if self.bind_all else ["--host", self.host])
        log.info("$ %s", " ".join(argv))
        self.process = subprocess.Popen(argv)
        self._write_sidecar()
        log.ok(f"TensorBoard serving {self.logdir} at {self.url}")
        return self.url

    def stop(self, *, timeout: float = 10.0) -> None:
        """Terminate the server and remove its sidecar."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self.sidecar is not None:
            self.sidecar.unlink(missing_ok=True)

    def wait(self) -> int:
        """Block until the server exits; returns its exit code."""
        if self.process is None:
            raise RuntimeError("TensorBoardServer.wait() called before start().")
        return self.process.wait()

    def __enter__(self) -> TensorBoardServer:
        """Start the server."""
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the server."""
        self.stop()


def read_server_sidecar(path: Path) -> dict[str, Any] | None:
    """A server sidecar's contents, or ``None`` when absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ssh_tunnel_command(sidecar: Mapping[str, Any], *, login_host: str, local_port: int) -> str:
    """The ``ssh -L`` line that forwards a cluster-node TensorBoard to this workstation."""
    return (
        f"ssh -N -L {int(local_port)}:{sidecar.get('hostname')}:{sidecar.get('port')} "
        f"{login_host}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _parse_mirror_specs(specs: Iterable[str]) -> dict[str, Path]:
    """Parse ``--mirror`` values: ``label=/path`` or a bare ``/path`` (label from the name)."""
    sources: dict[str, Path] = {}
    for spec in specs:
        label, _, raw = str(spec).partition("=")
        if not raw:
            label, raw = Path(label).name, label
        sources[label] = Path(os.path.expanduser(raw)).resolve()
    return sources


@click.command("nvitk-tensorboard")
@click.option("--logdir", type=click.Path(path_type=Path), required=True,
              help="Event directory to serve (and to mirror into).")
@click.option("--mirror", "mirrors", multiple=True,
              help="Framework results root to mirror: 'label=/path' or '/path'. Repeatable.")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True)
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--bind-all", is_flag=True, default=False,
              help="Listen on all interfaces (for a server on a cluster node).")
@click.option("--reload-interval", type=int, default=30, show_default=True,
              help="How often TensorBoard rescans --logdir, in seconds.")
@click.option("--interval", type=float, default=DEFAULT_INTERVAL, show_default=True,
              help="How often the mirror rescans the training logs, in seconds.")
@click.option("--sidecar", type=click.Path(path_type=Path), default=None,
              help="Write host/port/URL here so another host can find this server.")
@click.option("--class-name", "class_names", multiple=True,
              help="Name for the per-class dice series, in label order. Repeatable.")
@click.option("--no-serve", is_flag=True, default=False,
              help="Mirror only — do not start a server.")
@click.option("--once", is_flag=True, default=False,
              help="Mirror a single pass and exit (implies --no-serve).")
@click.option("--log-level", type=str, default="INFO", show_default=True)
def main(
    logdir: Path, mirrors: tuple[str, ...], port: int, host: str, bind_all: bool,
    reload_interval: int, interval: float, sidecar: Path | None,
    class_names: tuple[str, ...], no_serve: bool, once: bool, log_level: str,
) -> None:
    """Mirror nnU-Net / nnssl training logs into TensorBoard, and serve them.

    Works against a live run or a finished one, on this host or on a mounted cluster
    filesystem — nothing needs to be running for the mirror to catch up.
    """
    Logger(level=log_level.upper())
    sources = _parse_mirror_specs(mirrors)

    if once:
        if not sources:
            raise click.UsageError("--once needs at least one --mirror root.")
        watcher = TensorBoardWatcher(
            sources, logdir, class_names=class_names or None, interval=interval
        )
        log.ok(f"mirrored {watcher.sync_once()} epoch(s) -> {logdir}")
        return

    watcher = (
        TensorBoardWatcher(sources, logdir, class_names=class_names or None, interval=interval)
        if sources else None
    )
    if watcher is not None:
        watcher.start()
    try:
        if no_serve:
            if watcher is None:
                raise click.UsageError("--no-serve needs at least one --mirror root.")
            log.info("Mirroring only; Ctrl-C to stop.")
            while True:
                time.sleep(interval)
        server = TensorBoardServer(
            logdir, port=resolve_port(port, host="" if bind_all else host), host=host,
            bind_all=bind_all, reload_interval=reload_interval, sidecar=sidecar,
        )
        with server:
            server.wait()
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        if watcher is not None:
            watcher.stop()


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_PORT",
    "DEFAULT_RUN_PATTERN",
    "EVENT_GLOB",
    "EpochRecord",
    "LOCK_FILE",
    "MirrorLock",
    "SERVER_SIDECAR",
    "STATE_FILE",
    "TRAINING_LOG_GLOB",
    "TensorBoardMirror",
    "TensorBoardServer",
    "TensorBoardWatcher",
    "discover_run_dirs",
    "iter_training_logs",
    "main",
    "parse_training_log",
    "read_server_sidecar",
    "resolve_port",
    "ssh_tunnel_command",
    "summary_writer_class",
    "tensorboard_available",
]


if __name__ == "__main__":
    main()
