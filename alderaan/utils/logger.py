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

LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)-8s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# chatty third-party loggers held at WARNING
_THIRD_PARTY_PINS = [
    "matplotlib",
    "PIL",
    "theano",
    "aesara",
    "pymc3",
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


def setup_logging(log_dir, target, run_id, level="INFO"):
    """
    Log to the console (stderr) and to a new file for this run

    Parameters
    ----------
    log_dir : str
        logs root; the file is {log_dir}/{run_id}/alderaan_{target}_{UTstamp}.log
    target : str
    run_id : str
    level : str
        console level: DEBUG, INFO, WARNING, or ERROR; the file records DEBUG

    Returns
    -------
    log_path : str
    """
    teardown_logging()

    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run_dir = os.path.join(log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    file_handler = _open_file_handler(
        os.path.join(run_dir, f"alderaan_{target}_{stamp}.log")
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

    logging.captureWarnings(True)
    return file_handler.baseFilename


def teardown_logging():
    """Remove and close handlers installed by setup_logging"""
    root = logging.getLogger()
    while _installed_handlers:
        handler = _installed_handlers.pop()
        root.removeHandler(handler)
        handler.close()
    logging.captureWarnings(False)


def _open_file_handler(path):
    # exclusive create so concurrent runs never share a file
    base, ext = os.path.splitext(path)
    for i in range(1000):
        candidate = path if i == 0 else f"{base}.{i}{ext}"
        try:
            return logging.FileHandler(candidate, mode="x", encoding="utf-8")
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique log file near {path}")
