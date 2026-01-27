# app_logging.py
from __future__ import annotations

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _log_dir(app_name_no_spaces: str) -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / app_name_no_spaces / "logs"
    # fallback for non-windows / dev
    return Path.home() / f".{app_name_no_spaces.lower()}" / "logs"
    
    
def init_logging(app_name: str, app_version: str) -> tuple[logging.Logger, Path]:
    """
    Initializes a rotating file logger + console logger (console is useful in dev).
    Safe to call once at startup.
    Returns (logger, log_path).
    """
    app_key = app_name.replace(" ", "")
    log_dir = _log_dir(app_key)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "PhotoPicker.log"

    logger = logging.getLogger("photopicker")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent double-handlers if init_logging is called twice accidentally
    if logger.handlers:
        return logger, log_path

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=2_000_000,   # 2MB per file
        backupCount=5,        # keep ~10MB total history
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info("==== %s starting (v%s) ====", app_name, app_version)
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Executable: %s", sys.executable)
    logger.info("Frozen: %s", bool(getattr(sys, "frozen", False)))
    logger.info("Log file: %s", str(log_path))

    _install_global_exception_hooks(logger)

    return logger, log_path


def _install_global_exception_hooks(logger: logging.Logger) -> None:
    """
    Capture uncaught exceptions in main thread + Python 3.8+ thread exceptions.
    Tkinter callback exceptions are typically printed to stderr; the sys.excepthook
    helps for many cases.
    """
    def excepthook(exc_type, exc, tb):
        logger.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))

    sys.excepthook = excepthook

    # Python 3.8+: threading.excepthook exists
    try:
        import threading  # local import
        if hasattr(threading, "excepthook"):
            old = threading.excepthook

            def th_hook(args):
                logger.critical(
                    "UNCAUGHT THREAD EXCEPTION in %s",
                    getattr(args, "thread", None).name if getattr(args, "thread", None) else "(unknown)",
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
                try:
                    old(args)
                except Exception:
                    pass

            threading.excepthook = th_hook
    except Exception:
        # If anything fails, we still have file logging.
        logger.exception("Failed to install threading exception hook")


def safe_log_exception(logger: logging.Logger, context: str) -> None:
    """Convenience wrapper for logging the current exception with traceback."""
    logger.error("%s", context, exc_info=True)