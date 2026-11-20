"""End-to-end profile of ``alderaan.pipeline._run`` on K00137.

Ranks wall-clock by pipeline stage (every invocation combined), plus data I/O and
plots, so the report shows which algorithm dominates a run — the first thing to
know before optimizing anything. Function-level detail lives in the individual
``profile_<algorithm>.py`` harnesses, so there is no line-profiler pass here.

Run with ``make profile-pipeline`` or
``python tests/profiling/profile_pipeline.py``.
"""

import tempfile

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan import pipeline


def run():
    def setup():
        output_dir = tempfile.mkdtemp(prefix="alderaan_profile_")
        return (
            P.TARGET,
            P.MISSION,
            "profile",
            str(P.DATA_DIR),
            output_dir,
            str(P.CATALOG_DIR),
            P.VISITS,
        )

    P.run_profile(
        title="Pipeline (K00137)",
        report_name="pipeline",
        setup=setup,
        call=pipeline._run,
        line_pass=False,
        recipe=True,
    )


if __name__ == "__main__":
    run()
