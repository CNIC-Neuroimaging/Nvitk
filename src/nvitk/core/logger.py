import logging
import inspect
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from rich.console import Console, Theme
from rich.progress import Progress, TaskID
from rich.logging import RichHandler

from nvitk.core.patterns import Singleton

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None


def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except NameError:
        return False


class Logger(metaclass=Singleton):
    """
    Logger class for the PyMicra package.

    This class provides robust logging capabilities by integrating Python's built-in
    logging module with the Rich library. The Logger offers the following features:

      - **Unified Rich Console**: Uses a shared Rich Console instance for logging.
      - **Auto-Scrolling Logs**: Log messages are output via a RichHandler so they auto-scroll
        in the terminal.
      - **Flexible Logging Levels**: Supports 'NONE', 'INFO', and 'DEBUG' levels. In DEBUG mode,
        caller local variables can optionally be appended to log messages.
      - **Singleton Pattern**: Ensures a single Logger instance per process.
      - **Testing Friendly**: Maintains an internal log buffer that stores formatted log lines.
      - **Jupyter Compatibility**: In notebook environments, live updates and progress tracking
        are disabled to avoid notebook output issues.
      - **File Logging**: Supports saving logs to .log and .err files with automatic directory creation.
      - **Usage in Other Modules**: Since the Logger is a singleton, you can import it directly:
            >>> from nvitk.core.logger import Logger
            >>> logger = Logger(level='INFO')
            >>> logger.info("Hello, world!")

    Examples
    --------
    Basic usage:

        >>> logger = Logger(level='INFO')
        >>> logger.info("Starting process...")
        >>> task_id = logger.progress("Task A", total=10)  # In terminal mode only.
        >>> for i in range(10):
        ...     logger.info(f"Iteration {i}")
        ...     logger.update_progress(task_id, 1)
        >>> logger.info("Process completed!")

    File logging:

        >>> logger = Logger(level='INFO')
        >>> logger.add_file_handler("my_script.log")  # Saves all logs to file
        >>> logger.add_file_handler("my_script.err", level="ERROR")  # Saves only errors
        >>> logger.info("This goes to console and .log file")
        >>> logger.error("This goes to console, .log file, and .err file")

    Convenient script logging:

        >>> logger = Logger(level='INFO')
        >>> logger.setup_script_logging("my_script")  # Creates my_script.log and my_script.err
        >>> logger.info("Processing data...")
        >>> logger.error("Something went wrong!")

    Enabling debug locals:

        >>> logger.set_level("DEBUG")
        >>> logger.enable_debug_locals(True)
        >>> logger.debug("Debug message with local variables.")
    """

    _levels = {
        "NONE": logging.CRITICAL + 10,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }

    _theme = Theme({
        "info": "blue",
        "warning": "yellow",
        "error": "red",
        "critical": "bold red",
        "debug": "magenta",
        "ok": "green",
        "bold": "bold",
        "italic": "italic",
        "underline": "underline",
    })

    def __init__(self, level: str = 'INFO', name: str = 'pymicra'):
        """
        Initialize the Logger singleton.

        This constructor sets up the logging environment for PyMicra. In a terminal environment,
        a persistent Rich Progress object is created for progress tracking; in a Jupyter Notebook,
        live progress is disabled and the Console is configured for standard notebook output.

        Parameters
        ----------
        level : {'NONE', 'INFO', 'DEBUG'}, optional
            The desired logging level, default is 'INFO'.
        name : str, optional
            The name of the logger, by default 'pymicra'.
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
            # Terminal mode: use live features.
            self._base_console = Console(
                theme=self._theme,
                soft_wrap=True,
                force_terminal=True,
                record=True,
                markup=True
            )
            self._progress = Progress(console=self._base_console, transient=False)
            self._progress.start()

        # Internal log buffer.
        self._log_buffer: list[str] = []

        self._show_debug_locals = False
        self.set_level(level)

    def set_level(self, level: str = 'INFO') -> None:
        """
        Set the logging level for the base logger.

        Removes existing handlers and configures the logger with either a NullHandler (if level is 'NONE')
        or a RichHandler to render logs on the shared console. File handlers are preserved.

        Parameters
        ----------
        level : {'NONE', 'INFO', 'DEBUG'}
            The desired logging level.
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

        rich_handler = RichHandler(
            console=self._base_console,
            show_time=False,
            show_level=True,
            show_path=False,
            markup=True,
            log_time_format="%H:%M:%S"
        )
        self._logger.addHandler(rich_handler)

    def get_level(self) -> str:
        """
        Get the current logging level of the logger.

        Returns
        -------
        str
            The current logging level as a string ('NONE', 'INFO', 'DEBUG').
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
        Add a file handler to save logs to a file.

        Parameters
        ----------
        file_path : str or Path
            Path to the log file. Parent directories will be created if they don't exist.
        level : {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}, optional
            Minimum logging level for this file handler, default is 'INFO'.
        format_string : str, optional
            Custom format string for log messages. If None, uses a default format.
        mode : str, optional
            File mode for opening the log file, default is 'a' (append).

        Returns
        -------
        logging.FileHandler
            The created file handler.

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
        Set up convenient logging for a script with both .log and .err files.

        Creates two file handlers:
        - {script_name}.log: Contains all log messages at or above log_level
        - {script_name}.err: Contains only error messages at or above err_level

        Parameters
        ----------
        script_name : str
            Name of the script (without extension). Used as the base name for log files.
        log_dir : str or Path, optional
            Directory to save log files, default is "logs".
        log_level : str, optional
            Minimum level for the .log file, default is "INFO".
        err_level : str, optional
            Minimum level for the .err file, default is "ERROR".

        Returns
        -------
        tuple[logging.FileHandler, logging.FileHandler]
            A tuple containing (log_handler, err_handler).

        Examples
        --------
        >>> logger = Logger()
        >>> logger.setup_script_logging("data_processor")
        >>> logger.info("Processing started")  # Goes to console and data_processor.log
        >>> logger.error("Failed to process")  # Goes to console, data_processor.log, and data_processor.err
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

    def remove_file_handlers(self):
        """
        Remove all file handlers from the logger.

        This is useful for cleanup or when switching between different logging configurations.
        """
        for handler in self._file_handlers:
            handler.close()
            self._logger.removeHandler(handler)

        self._file_handlers.clear()
        self.info("All file handlers removed")

    def enable_debug_locals(self, enable: bool = True):
        """
        Enable or disable a filter that appends caller local variables to DEBUG log messages.

        Parameters
        ----------
        enable : bool
            True to enable, False to disable.
        """
        self._show_debug_locals = enable
        if enable:
            self._logger.addFilter(self._DebugLocalsFilter())
        else:
            for f in list(self._logger.filters):
                if isinstance(f, self._DebugLocalsFilter):
                    self._logger.removeFilter(f)

    class _DebugLocalsFilter(logging.Filter):
        """
        Appends local variables to DEBUG log messages.

        Example:
        >>> Logger().enable_debug_locals(True)
        >>> Logger().debug("Check variables")
        """
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno == logging.DEBUG:
                stack = inspect.stack()
                if len(stack) > 2:
                    caller_frame = stack[2].frame
                    caller_locals = caller_frame.f_locals
                    record.msg = f"{record.msg} | LOCALS={caller_locals}"
            return True

    def progress(self, description: str, total: float) -> Optional[TaskID]:
        """
        Create a new progress task (only in terminal mode).

        Parameters
        ----------
        description : str
            Description for the task.
        total : float
            Total steps for the task.

        Returns
        -------
        TaskID or None
            The created task's ID, or None if progress is disabled.
        """
        if self._progress is None:
            self._logger.info("Progress tasks are disabled in notebook mode.")
            return None
        task_id = self._progress.add_task(
            description,
            total=total,
            remove_when_done=False
        )
        self._progress.refresh()
        return task_id

    def update_progress(self, task_id: TaskID, advance: float = 1.0):
        """
        Advance a progress task (only in terminal mode).

        Parameters
        ----------
        task_id : TaskID
            The ID of the task.
        advance : float, optional
            Amount to advance (default is 1.0).
        """
        if self._progress is None:
            return
        self._progress.update(task_id, advance=advance)
        self._progress.refresh()

    def debug(self, msg, *args, **kwargs):
        """Log a DEBUG message."""
        self._append_log_line("DEBUG", msg)
        self._logger.debug(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def info(self, msg, *args, **kwargs):
        """Log an INFO message."""
        self._append_log_line("INFO", msg)
        self._logger.info(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def warning(self, msg, *args, **kwargs):
        """Log a WARNING message."""
        self._append_log_line("WARNING", msg)
        self._logger.warning(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def error(self, msg, *args, **kwargs):
        """Log an ERROR message."""
        self._append_log_line("ERROR", msg)
        self._logger.error(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def critical(self, msg, *args, **kwargs):
        """Log a CRITICAL message."""
        self._append_log_line("CRITICAL", msg)
        self._logger.critical(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def exception(self, msg, *args, **kwargs):
        """Log an exception."""
        self._append_log_line("ERROR", msg)
        self._logger.exception(msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def ok(self, msg, *args, **kwargs):
        """Log a message in green (OK message)."""
        styled_msg = f"[green]{msg}[/green]"
        self._append_log_line("INFO", styled_msg)
        self._logger.info(styled_msg, *args, **kwargs)
        if self._progress:
            self._progress.refresh()

    def _append_log_line(self, level_name: str, msg: str):
        """Append a formatted log line to the internal buffer."""
        now_str = datetime.now().strftime("%H:%M:%S")
        line = f"{now_str} - {level_name} - {msg}"
        self._log_buffer.append(line)

    def get_logger(self) -> logging.Logger:
        """Retrieve the underlying Python logger."""
        return self._logger

    def reset(self):
        """Reset the logger (useful for testing)."""
        if self._progress:
            self._progress.stop()
        self._log_buffer.clear()
        self.remove_file_handlers()
        if not in_notebook():
            self._progress = Progress(console=self._base_console, transient=False)
            self._progress.start()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Logger(name='{self._name}', level='{self._logger.level}', file_handlers={len(self._file_handlers)})"

    def __str__(self) -> str:
        return self.__repr__()
