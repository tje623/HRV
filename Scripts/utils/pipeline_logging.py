#!/usr/bin/env python3
"""
Shared logging utility for the ECG artifact-detection pipeline.

Usage in each pipeline script
------------------------------
    from utils.pipeline_logging import setup_logger

    # After argparse.parse_args():
    logger = setup_logger("data_pipeline", args=args, disable_log=args.no_log)

Each call produces:
  • A StreamHandler  → console output (INFO+)
  • A FileHandler    → /Volumes/xHRV/Outputs/Logs/<script>_<timestamp>.log  (DEBUG+)
    (suppressed when logging is disabled)

The log file starts with a human-readable header that records:
  - Script name, start date/time, full CLI argument list

Disabling logging
-----------------
Two mechanisms — either is sufficient:
  1. Pass ``disable_log=True`` to setup_logger()  (wire it to an argparse
     ``--no-log`` flag so the user can suppress file logs at call time).
  2. Set the environment variable  DISABLE_LOGGING=1  before running any script.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("/Volumes/xHRV/Outputs/Logs")

# Track which loggers we have already configured so re-imports are idempotent.
_configured: set[str] = set()


def setup_logger(
    script_name: str,
    args=None,
    log_dir: Path = _LOG_DIR,
    disable_log: bool = False,
    level_console: int = logging.INFO,
    level_file: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure and return a named logger that writes to console + log file.

    Parameters
    ----------
    script_name:
        Short identifier used in the logger name and log filename
        (e.g. "data_pipeline", "beat_features").
    args:
        ``argparse.Namespace`` returned by ``parse_args()``.  All key=value
        pairs are written into the log header.  Pass ``None`` to fall back
        to raw ``sys.argv``.
    log_dir:
        Directory for log files.  Created automatically if absent.
    disable_log:
        When True, skip file logging entirely (console still active).
        Also triggered by the environment variable ``DISABLE_LOGGING=1``.
    level_console:
        Minimum log level for the console handler (default INFO).
    level_file:
        Minimum log level for the file handler (default DEBUG so that
        verbose decision-trace messages are captured).

    Returns
    -------
    logging.Logger
        The configured logger.  Call ``logger.debug/info/warning/error``
        throughout each script.
    """
    logger_name = f"ecgclean.{script_name}"

    # Root logger: make sure basicConfig hasn't swallowed our records.
    # We propagate=True so the root level doesn't silently filter us out.
    root = logging.getLogger()
    if root.level == logging.WARNING:
        root.setLevel(logging.DEBUG)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Idempotency guard — calling setup_logger() twice in the same process
    # (e.g. in tests) must not create duplicate handlers.
    if logger_name in _configured:
        return logger
    _configured.add(logger_name)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(level_console)
    console_h.setFormatter(fmt)
    logger.addHandler(console_h)

    # ------------------------------------------------------------------
    # Decide whether to write a log file
    # ------------------------------------------------------------------
    env_disabled = os.environ.get("DISABLE_LOGGING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    write_file = not disable_log and not env_disabled

    if not write_file:
        logger.info(
            "File logging DISABLED (DISABLE_LOGGING env var or --no-log flag)."
        )
        return logger

    # ------------------------------------------------------------------
    # File handler
    # ------------------------------------------------------------------
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp_file = now.strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_human = now.strftime("%A, %B %d, %Y at %I:%M:%S %p")

    log_path = log_dir / f"{script_name}_{timestamp_file}.log"

    file_h = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setLevel(level_file)
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    # ------------------------------------------------------------------
    # Header block (written directly so it appears before any log records)
    # ------------------------------------------------------------------
    sep = "=" * 72
    if args is not None:
        cli_repr = "  ".join(f"{k}={v!r}" for k, v in vars(args).items())
    else:
        cli_repr = " ".join(sys.argv)

    header = "\n".join([
        "",
        sep,
        f"  Script  : {script_name}",
        f"  Started : {timestamp_human}",
        f"  Log file: {log_path}",
        f"  CLI args: {cli_repr}",
        sep,
        "",
    ])
    file_h.stream.write(header)
    file_h.stream.flush()

    logger.info("Logging to: %s", log_path)
    return logger


def add_logging_args(parser: "argparse.ArgumentParser") -> None:
    """
    Add ``--no-log`` flag to an existing ArgumentParser.

    Call this before ``parser.parse_args()`` so that every script gets a
    consistent disable-logging option.  Wire the result into ``setup_logger``::

        add_logging_args(parser)
        args = parser.parse_args()
        logger = setup_logger("my_script", args=args, disable_log=args.no_log)
    """
    parser.add_argument(
        "--no-log",
        action="store_true",
        default=False,
        help=(
            "Disable file logging.  Console output is unaffected.  "
            "Equivalent to setting DISABLE_LOGGING=1 in the environment."
        ),
    )
