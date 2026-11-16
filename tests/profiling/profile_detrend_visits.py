"""Profile ``DetrendVisits`` (wide_mask + tight_mask) on K00137.

Run with ``make profile-detrend_visits`` or
``python tests/profiling/profile_detrend_visits.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import DetrendVisits


def run():
    return P.profile_algorithm(
        DetrendVisits, "DetrendVisits (wide_mask + tight_mask)", "detrend_visits"
    )


if __name__ == "__main__":
    run()
