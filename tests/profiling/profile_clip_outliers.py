"""Profile ``ClipOutliers`` (local_trend + transit_model) on K00137.

Run with ``make profile-clip_outliers`` or
``python tests/profiling/profile_clip_outliers.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import ClipOutliers


def run():
    return P.profile_algorithm(
        ClipOutliers, "ClipOutliers (local_trend + transit_model)", "clip_outliers"
    )


if __name__ == "__main__":
    run()
