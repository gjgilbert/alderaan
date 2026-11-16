"""Tests for alderaan/utils/logger.py: per-run log file setup and teardown"""

import logging
import warnings

import pytest

from alderaan.utils.logger import setup_logging, teardown_logging


@pytest.fixture
def root_logger():
    """Root logger, restored to its prior level and handlers afterward"""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield root
    teardown_logging()
    root.setLevel(level)
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)


class TestLogging:
    def test_file_records_debug_from_module_loggers(self, tmp_path, root_logger):
        log_path = setup_logging(str(tmp_path / "logs"), "K00137", level="WARNING")
        assert log_path.startswith(str(tmp_path / "logs" / "alderaan_K00137_"))
        assert log_path.endswith(".log")

        logging.getLogger("alderaan.some_module").debug("debug message")
        teardown_logging()

        text = open(log_path).read()
        assert "DEBUG" in text
        assert "alderaan.some_module: debug message" in text

    def test_captures_warnings(self, tmp_path, root_logger):
        log_path = setup_logging(str(tmp_path), "K00137")
        warnings.warn("careful", UserWarning, stacklevel=1)
        teardown_logging()
        assert "careful" in open(log_path).read()

    def test_teardown_removes_handlers(self, tmp_path, root_logger):
        n_before = len(root_logger.handlers)
        log_path = setup_logging(str(tmp_path), "K00137")
        assert len(root_logger.handlers) == n_before + 2

        teardown_logging()
        assert len(root_logger.handlers) == n_before

        logging.getLogger("alderaan").warning("after teardown")
        assert "after teardown" not in open(log_path).read()

    def test_teardown_restores_warning_filters(self, tmp_path, root_logger):
        filters_before = list(warnings.filters)
        setup_logging(str(tmp_path), "K00137")
        assert warnings.filters != filters_before
        teardown_logging()
        assert warnings.filters == filters_before

    def test_setup_replaces_previous_handlers(self, tmp_path, root_logger):
        n_before = len(root_logger.handlers)
        setup_logging(str(tmp_path / "a"), "K00137")
        setup_logging(str(tmp_path / "b"), "K00137")
        assert len(root_logger.handlers) == n_before + 2
