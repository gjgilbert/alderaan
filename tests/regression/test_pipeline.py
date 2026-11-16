"""
Tests for alderaan/pipeline.py: main()'s mission check, logging, error
handling, and the guard against mixing two runs' outputs. The pipeline stages
(_run) are replaced by stubs; nothing runs end to end.
"""

import glob
import os

import pytest

from alderaan import pipeline
from alderaan.utils import logger as alderaan_logger

ARGS = dict(target="K00137", run_id="test", input_dir="in", catalog_dir="cat")


def _log_text(output_dir):
    (path,) = glob.glob(os.path.join(output_dir, "test", "logs", "*.log"))
    with open(path) as f:
        return f.read()


class TestMain:
    def test_rejects_non_kepler_before_creating_outputs(self, tmp_path):
        with pytest.raises(NotImplementedError, match="TESS"):
            pipeline.main(mission="TESS", output_dir=str(tmp_path), **ARGS)
        assert list(tmp_path.iterdir()) == []

    def test_success_is_logged_and_handlers_removed(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(*args):
            calls.append(args)
            return "results.fits"

        monkeypatch.setattr(pipeline, "_run", fake_run)
        pipeline.main(mission="Kepler", output_dir=str(tmp_path), visits=[4], **ARGS)

        assert calls == [("K00137", "Kepler", "test", "in", str(tmp_path), "cat", [4])]
        assert "pipeline completed" in _log_text(tmp_path)
        assert alderaan_logger._installed_handlers == []

    def test_failure_is_logged_with_traceback_and_reraised(self, tmp_path, monkeypatch):
        def fake_run(*args):
            raise RuntimeError("stage exploded")

        monkeypatch.setattr(pipeline, "_run", fake_run)
        with pytest.raises(RuntimeError, match="stage exploded"):
            pipeline.main(mission="Kepler", output_dir=str(tmp_path), **ARGS)

        text = _log_text(tmp_path)
        assert "pipeline aborted" in text
        assert "Traceback" in text and "stage exploded" in text
        assert alderaan_logger._installed_handlers == []

    def test_refuses_to_mix_outputs_of_two_runs(self, tmp_path):
        results_dir = tmp_path / "test" / "results" / "K00137"
        results_dir.mkdir(parents=True)
        (results_dir / "K00137_results.fits").touch()

        with pytest.raises(FileExistsError, match="not empty"):
            pipeline.main(mission="Kepler", output_dir=str(tmp_path), **ARGS)
        assert "pipeline aborted" in _log_text(tmp_path)
