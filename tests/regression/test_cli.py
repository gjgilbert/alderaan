"""
Tests for alderaan/cli: the `alderaan` subcommand dispatcher and `alderaan run`
argument parsing. alderaan.pipeline is replaced by a stub; nothing is run.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from alderaan import cli
from alderaan.cli import run


@pytest.fixture
def pipeline_calls(monkeypatch):
    """Stub alderaan.pipeline.main; returns the list of kwargs it was called with"""
    calls = []
    stub = types.ModuleType("alderaan.pipeline")
    stub.main = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "alderaan.pipeline", stub)
    return calls


class TestDispatch:
    def test_routes_run_with_forwarded_argv(self, monkeypatch):
        seen = []
        monkeypatch.setitem(cli._COMMANDS, "run", lambda rest: seen.append(rest) or 7)
        assert cli.main(["run", "-t", "K00137"]) == 7
        assert seen == [["-t", "K00137"]]

    def test_usage(self, capsys):
        assert cli.main(["--help"]) == 0
        assert "usage: alderaan <command>" in capsys.readouterr().out
        assert cli.main([]) == 2

    def test_unknown_command(self, capsys):
        assert cli.main(["frobnicate"]) == 2
        assert "unknown command 'frobnicate'" in capsys.readouterr().err


class TestRun:
    def test_maps_arguments_to_pipeline_main(self, pipeline_calls, tmp_path):
        argv = [
            "-t",
            "K00137",
            "-m",
            "Kepler",
            "--run_id",
            "r1",
            "--input_dir",
            str(tmp_path / "in"),
            "--output_dir",
            str(tmp_path / "out"),
            "--catalog_dir",
            str(tmp_path / "cat"),
            "--visits",
            "4",
            "5",
            "--log_level",
            "DEBUG",
        ]
        assert cli.main(["run", *argv]) == 0
        assert pipeline_calls == [
            dict(
                target="K00137",
                mission="Kepler",
                run_id="r1",
                input_dir=str((tmp_path / "in").resolve()),
                output_dir=str((tmp_path / "out").resolve()),
                catalog_dir=str((tmp_path / "cat").resolve()),
                visits=[4, 5],
                log_level="DEBUG",
            )
        ]

    def test_defaults(self, pipeline_calls):
        run.main(["-t", "K00137", "-m", "Kepler", "--run_id", "r1"])
        (kwargs,) = pipeline_calls
        base = Path(run.__file__).resolve().parents[2]
        assert kwargs["input_dir"] == str(base / "tests" / "data" / "MAST_downloads")
        assert kwargs["output_dir"] == str(base / "outputs")
        assert kwargs["catalog_dir"] == str(base / "tests" / "catalogs")
        assert kwargs["visits"] is None
        assert kwargs["log_level"] == "INFO"

    def test_unknown_mission_exits_before_pipeline(self, pipeline_calls):
        with pytest.raises(SystemExit) as exc:
            run.main(["-t", "K00137", "-m", "Hubble", "--run_id", "r1"])
        assert exc.value.code == 2
        assert pipeline_calls == []

    def test_help_does_not_import_pipeline(self):
        code = (
            "import sys\n"
            "from alderaan.cli import main\n"
            "try:\n"
            "    main(['run', '--help'])\n"
            "except SystemExit as e:\n"
            "    assert e.code == 0\n"
            "print('alderaan.pipeline' in sys.modules, 'pymc' in sys.modules)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        assert "--target" in out
        assert out.strip().endswith("False False")
