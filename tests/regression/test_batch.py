import sys
import time

from alderaan.cli.batch import read_targets, run_targets


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
        assert run_targets(tasks, jobs=2) == ({"b"}, set())

    def test_exit_3_is_unsupported_not_failed(self):
        tasks = [_exit_task("a", 0), _exit_task("b", 3), _exit_task("c", 1)]
        assert run_targets(tasks, jobs=3) == ({"c"}, {"b"})

    def test_timeout_is_failure(self):
        tasks = [_exit_task("slow", 0, sleep=30), _exit_task("fast", 0)]
        failed, _ = run_targets(tasks, jobs=2, job_timeout=1)
        assert failed == {"slow"}

    def test_launch_interval_spaces_launches(self):
        tasks = [_exit_task(t, 0) for t in "abc"]
        start = time.monotonic()
        run_targets(tasks, jobs=3, launch_interval=0.5)
        assert time.monotonic() - start >= 1.0
