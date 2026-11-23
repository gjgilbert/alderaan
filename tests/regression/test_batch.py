import sys
import time

from alderaan.cli.batch import batch_summary, read_targets, run_targets


def _exit_task(tag, code, sleep=0.0):
    code = f"import sys, time; time.sleep({sleep}); sys.exit({code})"
    return tag, [sys.executable, "-c", code]


class TestReadTargets:
    def test_unique_in_catalog_order(self, tmp_path):
        catalog = tmp_path / "catalog.csv"
        catalog.write_text(
            "planet_name,koi_id\nK00005.01,K00005\nK00002.01,K00002\nK00005.02,K00005\n"
        )
        assert read_targets(catalog) == ["K00005", "K00002"]


class TestRunTargets:
    def test_returns_failed_targets(self):
        tasks = [_exit_task("a", 0), _exit_task("b", 1), _exit_task("c", 0)]
        assert run_targets(tasks, jobs=2) == ({"b": "exit 1"}, {})

    def test_exit_3_is_unsupported_not_failed(self):
        # the reason is the text after "(unsupported): " on the child's stderr
        msg = "ERROR alderaan.pipeline: target not processed (unsupported): 2 transits"
        unsupported = (
            "b",
            [
                sys.executable,
                "-c",
                f"import sys; sys.stderr.write({msg!r}); sys.exit(3)",
            ],
        )
        tasks = [_exit_task("a", 0), unsupported, _exit_task("c", 1)]
        assert run_targets(tasks, jobs=3) == ({"c": "exit 1"}, {"b": "2 transits"})

    def test_timeout_is_failure(self):
        tasks = [_exit_task("slow", 0, sleep=30), _exit_task("fast", 0)]
        failed, _ = run_targets(tasks, jobs=2, job_timeout=1)
        assert list(failed) == ["slow"] and "timed out" in failed["slow"]

    def test_launch_interval_spaces_launches(self):
        tasks = [_exit_task(t, 0) for t in "abc"]
        start = time.monotonic()
        run_targets(tasks, jobs=3, launch_interval=0.5)
        assert time.monotonic() - start >= 1.0


class TestBatchSummary:
    def test_counts_and_reasons(self):
        text = batch_summary(
            "r1", 5, {"K00003": "exit 1"}, {"K00002": "2 transits"}, "/logs", 7200.0
        )
        assert "batch summary: run_id r1" in text
        assert "5 (3 ok, 1 unsupported, 1 failed)" in text
        assert "K00002    2 transits" in text and "K00003    exit 1" in text
        assert "FAILED" in text and "2.00 hr" in text

    def test_all_ok_lists_no_targets(self):
        text = batch_summary("r1", 2, {}, {}, "/logs", 60.0)
        assert "2 (2 ok, 0 unsupported, 0 failed)" in text
        assert "FAILED" not in text and "unsupported (" not in text
