"""
Tests for alderaan/pipeline.py: main()'s mission check, logging, error
handling, and the guard against mixing two runs' outputs. The pipeline stages
(_run) are replaced by stubs; nothing runs end to end.
"""

import glob
import os

import numpy as np
import pytest

from alderaan import pipeline
from alderaan.data_models import LiteCurve, Results, TransitTimes
from alderaan.errors import UnsupportedTargetError
from alderaan.utils import logger as alderaan_logger

ARGS = dict(target="K00137", run_id="test", input_dir="in", catalog="cat")


def _log_text(output_dir):
    (path,) = glob.glob(os.path.join(output_dir, "test", "logs", "*.log"))
    with open(path) as f:
        return f.read()


class TestMain:
    def test_rejects_non_kepler_before_creating_outputs(self, tmp_path):
        with pytest.raises(NotImplementedError, match="TESS"):
            pipeline.main(mission="TESS", output_dir=str(tmp_path), **ARGS)
        assert list(tmp_path.iterdir()) == []

    def test_success_logs_run_summary_and_removes_handlers(
        self, tmp_path, monkeypatch, make_posteriors
    ):
        time = np.arange(130.0, 200.0, 0.02)
        lc = LiteCurve(
            "Kepler",
            "K00137",
            time,
            np.ones(time.size),
            np.full(time.size, 1e-4),
            np.arange(time.size),
            np.full(time.size, 4),
            np.full(time.size, "long cadence"),
        )
        tt = TransitTimes.from_linear_ephemeris(
            "Kepler", "K00137", 7.6, 131.0, 130.0, 200.0
        )
        results = Results(make_posteriors(npl=1), [tt], lc)
        results.meta["PLNAME0"] = "K00137.01"
        calls = []

        def fake_run(*args):
            calls.append(args)
            return results, "results.fits"

        monkeypatch.setattr(pipeline, "_run", fake_run)
        pipeline.main(mission="Kepler", output_dir=str(tmp_path), visits=[4], **ARGS)

        assert calls == [("K00137", "test", "in", str(tmp_path), "cat", [4])]
        text = _log_text(tmp_path)
        assert "run summary: K00137 (run_id test)" in text
        assert "status:   ok" in text and "visits 4" in text
        assert "K00137.01" in text and "10/10" in text
        assert "results:  results.fits" in text
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
        assert "status:   FAILED" in text
        assert alderaan_logger._installed_handlers == []

    def test_unsupported_target_is_one_error_line_and_reraised(
        self, tmp_path, monkeypatch
    ):
        def fake_run(*args):
            raise UnsupportedTargetError("planet 0: 2 good transits (< 3)")

        monkeypatch.setattr(pipeline, "_run", fake_run)
        with pytest.raises(UnsupportedTargetError):
            pipeline.main(mission="Kepler", output_dir=str(tmp_path), **ARGS)

        text = _log_text(tmp_path)
        assert "ERROR" in text and "unsupported): planet 0: 2 good transits" in text
        assert "Traceback" not in text and "pipeline aborted" not in text
        assert "status:   unsupported (known limit): planet 0" in text
        assert alderaan_logger._installed_handlers == []

    def test_refuses_to_mix_outputs_of_two_runs(self, tmp_path):
        results_dir = tmp_path / "test" / "results" / "K00137"
        results_dir.mkdir(parents=True)
        (results_dir / "K00137_results.fits").touch()

        with pytest.raises(FileExistsError, match="not empty"):
            pipeline.main(mission="Kepler", output_dir=str(tmp_path), **ARGS)
        assert "pipeline aborted" in _log_text(tmp_path)
