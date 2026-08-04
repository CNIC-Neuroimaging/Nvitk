"""Console and file logging for NVITK.

Uses ANSI escape codes (see :mod:`nvitk.util.colors`) for level colors on ``stderr``,
plain formatting on file handlers, and Rich :class:`~rich.progress.Progress` in terminals
only (disabled in Jupyter). The :class:`Logger` is a process-wide singleton.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────

import logging
import inspect
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypeVar, Union

T = TypeVar("T")

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TaskID

from nvitk.core.patterns import Singleton
from nvitk.util.colors import bcolors

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None


# ──────────────────────────────────────────────────────────────────────────────
# Environment & helpers
# ──────────────────────────────────────────────────────────────────────────────


def in_notebook() -> bool:
    """Return True when running inside a Jupyter / IPython ZMQ kernel (no Rich progress)."""
    if get_ipython is None:
        return False
    try:
        ip = get_ipython()
        if ip is None:
            return False
        shell = ip.__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI CSI color sequences from *text* (for the in-memory log buffer)."""
    return _ANSI_ESCAPE_RE.sub("", text)


# ──────────────────────────────────────────────────────────────────────────────
# Console formatting
# ──────────────────────────────────────────────────────────────────────────────


class _AnsiLevelFormatter(logging.Formatter):
    """Format log lines as ``time | LEVEL | message`` with ANSI color on the level column only."""

    _LEVEL_PREFIX = {
        logging.DEBUG: bcolors.GRAY,
        logging.INFO: bcolors.OKBLUE,
        logging.WARNING: bcolors.WARNING,
        logging.ERROR: bcolors.FAIL,
        logging.CRITICAL: bcolors.BOLD + bcolors.FAIL,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as ``HH:MM:SS | LEVEL | message`` with ANSI on the level column."""
        asctime = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        msg = record.getMessage()
        lc = self._LEVEL_PREFIX.get(record.levelno, "")
        if lc:
            colored_level = f"{lc}{levelname:<8}{bcolors.ENDC}"
        else:
            colored_level = f"{levelname:<8}"
        return f"{asctime} | {colored_level} | {msg}"


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────


class Logger(metaclass=Singleton):
    """
    Singleton logger: stderr with ANSI level colors, optional file handlers, Rich progress in terminals.

    **Levels:** ``NONE`` (disable console), ``INFO``, ``DEBUG``. Debug can optionally append
    caller locals via :meth:`enable_debug_locals`.

    **Console:** :class:`logging.StreamHandler` on ``stderr`` with :class:`_AnsiLevelFormatter`
    (not Rich markup), so output is reliable in Jupyter and on HPC.

    **Progress:** :class:`~rich.progress.Progress` on a shared :class:`~rich.console.Console`
    when not in a notebook; otherwise disabled.

    **Typical import**

    .. code-block:: python

        from nvitk.core.logger import Logger
        logger = Logger(level="INFO")
        logger.info("message")

    Examples
    --------
    File logging and script setup use :meth:`add_file_handler` and :meth:`setup_script_logging`.
    """

    _levels = {
        "NONE": logging.CRITICAL + 10,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }

    def __init__(self, level: str = 'INFO', name: str = 'nvitk', start_progress: bool = False):
        """
        Configure the singleton: console handler, optional Rich progress (terminal only).

        Parameters
        ----------
        level
            ``'NONE'``, ``'INFO'``, or ``'DEBUG'``.
        name
            Root :class:`logging.Logger` name (default ``'nvitk'``).
        """
        self._name = name
        self._logger = logging.getLogger(name)
        self._logger.propagate = False
        self._file_handlers = []  # Track file handlers for cleanup

        if in_notebook():
            # Notebook mode: disable forced terminal behavior, do not record output, and disable progress.
            self._base_console = Console(
                soft_wrap=True,
                force_terminal=False,
                record=False,
                markup=False,
                theme=None,
            )
            self._progress = None  # Disable progress in notebooks.
        else:
            # Terminal mode: use live features (no Rich theme; log colors use ANSI via StreamHandler).
            self._base_console = Console(
                soft_wrap=True,
                force_terminal=True,
                record=True,
                markup=False,
                theme=None,
            )
            self._progress = None
            if start_progress: 
                self._progress = Progress(console=self._base_console, transient=False)
                self._progress.start()

        # Internal log buffer.
        self._log_buffer: list[str] = []
        self._replaced_stream_handler: logging.StreamHandler | None = None
        self._rich_handler: RichHandler | None = None

        self._show_debug_locals = False
        self.set_level(level)

    def set_level(self, level: str = 'INFO') -> None:
        """
        Replace console handlers with Null (``NONE``) or ANSI :class:`logging.StreamHandler`.

        File handlers registered via :meth:`add_file_handler` are left unchanged.

        Parameters
        ----------
        level
            ``'NONE'``, ``'INFO'``, or ``'DEBUG'``.
        """
        # Remove only console handlers, preserve file handlers
        handlers_to_remove = []
        for handler in self._logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                handlers_to_remove.append(handler)

        for handler in handlers_to_remove:
            self._logger.removeHandler(handler)

        level = level.upper().strip()
        level_value = self._levels.get(level, logging.INFO)
        self._logger.setLevel(level_value)

        if level == 'NONE':
            self._logger.addHandler(logging.NullHandler())
            return

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            _AnsiLevelFormatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        self._logger.addHandler(console_handler)

    def get_level(self) -> str:
        """
        Map the underlying logger level to ``'NONE'``, ``'INFO'``, or ``'DEBUG'``.

        Returns
        -------
        str
            Current level label; unknown numeric levels fall back to ``'INFO'``.
        """
        level_value = self._logger.level
        for name, value in self._levels.items():
            if value == level_value:
                return name
        return 'INFO'

    def add_file_handler(
        self,
        file_path: Union[str, Path],
        level: str = "INFO",
        format_string: Optional[str] = None,
        mode: str = "a"
    ) -> logging.FileHandler:
        """
        Append a :class:`logging.FileHandler` with plain (non-ANSI) formatting.

        Parent directories are created automatically. The handler is tracked for
        :meth:`remove_file_handlers`. Emits one :meth:`info` line to the console when added.

        Parameters
        ----------
        file_path
            Destination ``.log`` or ``.err`` path.
        level
            Minimum record level for this handler (standard logging names).
        format_string
            If omitted, ``"%(asctime)s - %(name)s - %(levelname)s - %(message)s"``.
        mode
            File open mode, default append ``'a'``.

        Returns
        -------
        logging.FileHandler
            The attached handler instance.

        Examples
        --------
        >>> logger = Logger()
        >>> logger.add_file_handler("logs/app.log")
        >>> logger.add_file_handler("logs/errors.log", level="ERROR")
        """
        file_path = Path(file_path)

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(file_path, mode=mode)

        # Set level
        level_value = getattr(logging, level.upper(), logging.INFO)
        file_handler.setLevel(level_value)

        # Set format
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)
        file_handler.setFormatter(formatter)

        # Add to logger
        self._logger.addHandler(file_handler)
        self._file_handlers.append(file_handler)

        self.info(f"Added file handler: {file_path} (level: {level})")

        return file_handler

    def setup_script_logging(
        self,
        script_name: str,
        log_dir: Union[str, Path] = "logs",
        log_level: str = "INFO",
        err_level: str = "ERROR"
    ) -> tuple[logging.FileHandler, logging.FileHandler]:
        """
        Create ``{script_name}.log`` (all messages ≥ *log_level*) and ``{script_name}.err`` (≥ *err_level*).

        Parameters
        ----------
        script_name
            Base filename without extension.
        log_dir
            Directory for both files.
        log_level
            Threshold for the general log file.
        err_level
            Threshold for the error-only file.

        Returns
        -------
        tuple[logging.FileHandler, logging.FileHandler]
            ``(log_handler, err_handler)``.

        Examples
        --------
        >>> logger = Logger()
        >>> logger.setup_script_logging("pipeline")
        """
        log_dir = Path(log_dir)

        # Create log and error file handlers
        log_file = log_dir / f"{script_name}.log"
        err_file = log_dir / f"{script_name}.err"

        log_handler = self.add_file_handler(log_file, level=log_level)
        err_handler = self.add_file_handler(err_file, level=err_level)

        self.info(f"Script logging configured for '{script_name}'")
        self.info(f"  Log file: {log_file}")
        self.info(f"  Error file: {err_file}")

        return log_handler, err_handler

    def remove_file_handlers(self) -> None:
        """
        Close and remove every handler added via :meth:`add_file_handler` (not console handlers).
        """
        for handler in self._file_handlers:
            handler.close()
            self._logger.removeHandler(handler)

        self._file_handlers.clear()
        self.info("All file handlers removed")

    def enable_debug_locals(self, enable: bool = True) -> None:
        """
        When *enable* is True, attach a filter so DEBUG records append ``| LOCALS=...`` from the caller frame.

        Parameters
        ----------
        enable
            Toggle the :class:`_DebugLocalsFilter` on the root logger.
        """
        self._show_debug_locals = enable
        if enable:
            self._logger.addFilter(self._DebugLocalsFilter())
        else:
            for f in list(self._logger.filters):
                if isinstance(f, self._DebugLocalsFilter):
                    self._logger.removeFilter(f)

    class _DebugLocalsFilter(logging.Filter):
        """Internal: extend DEBUG ``record.msg`` with ``f_locals`` of the direct caller."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Attach caller local variables to DEBUG records (opt-in verbose tracing)."""
            if record.levelno == logging.DEBUG:
                stack = inspect.stack()
                if len(stack) > 2:
                    caller_frame = stack[2].frame
                    caller_locals = caller_frame.f_locals
                    record.msg = f"{record.msg} | LOCALS={caller_locals}"
            return True

    def _swap_to_rich_handler(self) -> None:
        """Replace the stderr StreamHandler with a RichHandler sharing the progress console."""
        for h in list(self._logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, (logging.FileHandler, RichHandler)):
                self._logger.removeHandler(h)
                self._replaced_stream_handler = h
                break

        rh = RichHandler(
            console=self._base_console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=False,
        )
        rh.setLevel(self._logger.level)
        self._rich_handler = rh
        self._logger.addHandler(rh)

    def _swap_to_stream_handler(self) -> None:
        """Restore the original stderr StreamHandler after progress ends."""
        if self._rich_handler:
            self._logger.removeHandler(self._rich_handler)
            self._rich_handler = None
        if self._replaced_stream_handler:
            self._logger.addHandler(self._replaced_stream_handler)
            self._replaced_stream_handler = None

    def ensure_progress(self) -> None:
        """Start Rich :class:`~rich.progress.Progress` if not already running (terminal only)."""
        if self._progress is not None or in_notebook():
            return
        self._progress = Progress(console=self._base_console, transient=False)
        self._progress.start()
        self._swap_to_rich_handler()

    def stop_progress(self) -> None:
        """Stop and clear the Rich progress display (no-op in notebooks)."""
        if self._progress is None:
            return
        self._progress.stop()
        self._progress = None
        self._swap_to_stream_handler()

    def step(self, msg: str, *args, **kwargs) -> None:
        """Emit an indented INFO line for a sub-step within a pipeline stage."""
        self.info(f"  ▸ {msg}", *args, **kwargs)

    def progress(self, description: str, total: float) -> Optional[TaskID]:
        """
        Add a Rich progress task (no-op in notebooks: logs a single INFO and returns None).

        Parameters
        ----------
        description
            Task label shown in the progress bar.
        total
            Total units for completion (float for partial steps).

        Returns
        -------
        TaskID or None
            Task id for :meth:`update_progress`, or None if progress is disabled.
        """
        if self._progress is None:
            if in_notebook():
                self._logger.info("Progress tasks are disabled in notebook mode.")
                return None
            else:
                self._progress = Progress(console=self._base_console, transient=False)
                self._progress.start()
                self._swap_to_rich_handler()
        task_id = self._progress.add_task(
            description,
            total=total,
            remove_when_done=False
        )
        self._progress.refresh()
        return task_id

    def update_progress(self, task_id: TaskID, advance: float = 1.0) -> None:
        """
        Advance a Rich task created by :meth:`progress` (ignored if progress is disabled).
        """
        if self._progress is None:
            return
        self._progress.update(task_id, advance=advance)
        self._progress.refresh()

    def debug(self, msg, *args, **kwargs):
        """Emit a DEBUG record (``%``-style formatting supported like :meth:`logging.Logger.debug`)."""
        self._append_log_line("DEBUG", msg)
        self._logger.debug(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def info(self, msg, *args, **kwargs):
        """Emit an INFO record."""
        self._append_log_line("INFO", msg)
        self._logger.info(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def warning(self, msg, *args, **kwargs):
        """Emit a WARNING record."""
        self._append_log_line("WARNING", msg)
        self._logger.warning(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def error(self, msg, *args, **kwargs):
        """Emit an ERROR record."""
        self._append_log_line("ERROR", msg)
        self._logger.error(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def critical(self, msg, *args, **kwargs):
        """Emit a CRITICAL record."""
        self._append_log_line("CRITICAL", msg)
        self._logger.critical(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def exception(self, msg, *args, **kwargs):
        """Emit an ERROR record with exception info (call inside ``except``)."""
        self._append_log_line("ERROR", msg)
        self._logger.exception(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def ok(self, msg, *args, **kwargs):
        """Emit at INFO level with green ANSI on the message body (level column stays INFO color)."""
        styled_msg = f"{bcolors.OKGREEN}{msg}{bcolors.ENDC}"
        self._append_log_line("INFO", styled_msg)
        self._logger.info(styled_msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def _append_log_line(self, level_name: str, msg: str) -> None:
        """Strip ANSI and append one line to :attr:`_log_buffer` for testing or inspection."""
        now_str = datetime.now().strftime("%H:%M:%S")
        plain = _strip_ansi(str(msg))
        line = f"{now_str} - {level_name} - {plain}"
        self._log_buffer.append(line)

    def get_logger(self) -> logging.Logger:
        """Return the underlying :class:`logging.Logger` (same *name* as passed to ``__init__``)."""
        return self._logger

    def reset(self, restart_progress: bool = True) -> None:
        """
        Stop progress, clear buffers, remove file handlers, and restart Rich progress in terminals.

        Intended for tests or subprocess isolation.
        """
        if self._progress:
            self._progress.stop()
            self._swap_to_stream_handler()
        self._log_buffer.clear()
        self.remove_file_handlers()
        if not in_notebook() and restart_progress:
            self._progress = Progress(console=self._base_console, transient=False)
            self._progress.start()
            self._swap_to_rich_handler()

    def __repr__(self) -> str:
        """``Logger(name=..., level=..., file_handlers=...)``."""
        return f"Logger(name='{self._name}', level='{self._logger.level}', file_handlers={len(self._file_handlers)})"

    def __str__(self) -> str:
        """Same as :meth:`__repr__`."""
        return self.__repr__()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline run tracking (subjects × stages, log-only)
# ──────────────────────────────────────────────────────────────────────────────

_COHORT_STAGES = frozenset({"stage0_d"})


@dataclass
class PipelineRunTracker:
    """
    Step logging for multi-subject pipeline runs (no Rich progress bar).

    Counts one step per cohort stage (e.g. XNAT download) plus one per
    (subject, per-subject stage). Use :meth:`run_stage` to log begin/complete/fail.

    Examples
    --------
    >>> log = Logger()
    >>> with PipelineRunTracker(log, "black_blood", subjects, stages) as run:
    ...     run.run_stage("(cohort)", "stage0_d", lambda: download_all())
    ...     for subj in subjects:
    ...         run.run_stage(subj, "stage1", lambda: reg(subj))
    """

    logger: Logger
    pipeline: str
    subjects: list[str]
    stages: list[str]
    stage_labels: dict[str, str] = field(default_factory=dict)
    _per_subject_stages: list[str] = field(init=False)
    _cohort_stages: list[str] = field(init=False)
    _total: int = field(init=False)
    _done: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Split stages into cohort-wide vs per-subject and precompute the total step count."""
        self._cohort_stages = [s for s in self.stages if s in _COHORT_STAGES]
        self._per_subject_stages = [s for s in self.stages if s not in _COHORT_STAGES]
        self._total = len(self._cohort_stages) + len(self.subjects) * len(
            self._per_subject_stages
        )

    def _label(self, stage: str) -> str:
        """Human-friendly label for *stage* (falls back to the raw stage id)."""
        return self.stage_labels.get(stage, stage)

    def __enter__(self) -> PipelineRunTracker:
        """Log the run header (subjects, stage list) and start tracking."""
        self.logger.info(
            f"▶ {self.pipeline}: {len(self.subjects)} subject(s), "
            f"stages=[{', '.join(self._label(s) for s in self.stages)}], "
            f"{self._total} step(s)"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Log a success summary, or an abort line if the block raised."""
        if exc_type is None:
            self.logger.ok(
                f"✓ {self.pipeline} finished ({self._done}/{self._total} steps)"
            )
        else:
            self.logger.error(f"✗ {self.pipeline} aborted: {exc_val}")
        return False

    def _tick(self, caption: str, *, ok: bool = True) -> None:
        """Advance the completed-step counter and emit a ``[done/total]`` progress line."""
        self._done += 1
        prefix = f"[{self._done}/{self._total}]"
        if ok:
            self.logger.ok(f"{prefix} {caption}")
        else:
            self.logger.warning(f"{prefix} {caption}")

    def begin(self, subject: str, stage: str) -> None:
        """Log start of a pipeline step."""
        self.logger.info(f"▶ [{subject}] {self._label(stage)}")

    def complete(self, subject: str, stage: str, detail: str = "") -> None:
        """Mark a step complete and log ``[n/total]``."""
        tail = f" — {detail}" if detail else ""
        self._tick(f"[{subject}] {self._label(stage)}{tail}", ok=True)

    def fail(self, subject: str, stage: str, exc: BaseException | str) -> None:
        """Mark a step failed, log ``[n/total]``, and emit ERROR."""
        msg = str(exc)
        self.logger.error(f"[{subject}] {self._label(stage)} failed: {msg}")
        self._tick(f"[{subject}] {self._label(stage)} FAILED", ok=False)

    def run_stage(
        self,
        subject: str,
        stage: str,
        fn: Callable[[], T],
        *,
        detail: str | Callable[[T], str] | None = None,
        reraise: bool = False,
    ) -> T | None:
        """
        Run *fn* with begin/complete logging; on exception call :meth:`fail`.

        Returns the result of *fn*, or None if an exception was caught and
        *reraise* is False.
        """
        self.begin(subject, stage)
        try:
            result = fn()
        except Exception as exc:
            import traceback

            self.logger.exception(traceback.format_exc())
            self.fail(subject, stage, exc)
            if reraise:
                raise
            return None
        if callable(detail):
            detail_str = detail(result)
        elif detail:
            detail_str = str(detail)
        else:
            detail_str = ""
        self.complete(subject, stage, detail_str)
        return result
