"""
Logging setup for pipeline runs, adapted from the KPF pipeline

Modules log via ``logger = logging.getLogger(__name__)`` and never configure
handlers. The entry point calls ``setup_logging`` once per run, which attaches a
file handler and a console handler to the root logger, so module loggers,
captured ``warnings``, and third-party libraries all reach the same log. The
file always records DEBUG; the console shows the requested level. With no
handlers installed (e.g. in tests), records below WARNING are dropped.
"""

__all__ = ["setup_logging", "teardown_logging"]

import logging
import os
import time
import warnings

from astropy.units import UnitsWarning

LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)-8s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# chatty third-party loggers held at WARNING
_THIRD_PARTY_PINS = [
    "matplotlib",
    "PIL",
    "filelock",
    "numexpr",
    "numba",
    "urllib3",
    "lightkurve",
    "asyncio",
    "h5py",
    "fsspec",
]

_installed_handlers = []
_warning_filters = []  # catch_warnings context restoring filters at teardown


def setup_logging(log_dir, target, run_id, level="INFO", append=False):
    """
    Log to the console (stderr) and to a new file for this run (or append to it)

    Parameters
    ----------
    log_dir : str
        the file is {log_dir}/alderaan_{target}_{run_id}.log
    target : str
    run_id : str
    level : str
        console level: DEBUG, INFO, WARNING, or ERROR; the file records DEBUG
    append : bool
        append to an existing log file; default False fails if the file exists

    Returns
    -------
    log_path : str
        absolute path of the log file
    """
    teardown_logging()

    os.makedirs(log_dir, exist_ok=True)
    file_handler = _open_file_handler(
        os.path.join(log_dir, f"alderaan_{target}_{run_id}.log"), append
    )

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime

    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.upper())

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _installed_handlers.append(handler)

    for name in _THIRD_PARTY_PINS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Suppress astropy UnitsWarning: astropy cannot parse Kepler FITS unit
    # strings ('e-/s', 'BJD - 2454833'), and alderaan does not use astropy
    # units. Teardown restores the caller's warning filters.
    context = warnings.catch_warnings()
    context.__enter__()
    _warning_filters.append(context)
    warnings.filterwarnings("ignore", category=UnitsWarning, module="astropy")
    logging.captureWarnings(True)
    return file_handler.baseFilename


def teardown_logging():
    """Remove and close handlers installed by setup_logging; restore warning filters"""
    root = logging.getLogger()
    while _installed_handlers:
        handler = _installed_handlers.pop()
        root.removeHandler(handler)
        handler.close()
    while _warning_filters:
        _warning_filters.pop().__exit__(None, None, None)
    logging.captureWarnings(False)


def _open_file_handler(path, append=False):
    """Open a UTF-8 log file; unless append, FileExistsError if path exists"""
    # exclusive create: fail loudly rather than share or overwrite a log file
    return logging.FileHandler(path, mode="a" if append else "x", encoding="utf-8")
