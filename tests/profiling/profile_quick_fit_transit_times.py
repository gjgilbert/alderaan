"""Profile ``QuickFitTransitTimes`` (pass 1) on K00137.

Run with ``make profile-quick_fit_transit_times`` or
``python tests/profiling/profile_quick_fit_transit_times.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import QuickFitTransitTimes


def run():
    return P.profile_algorithm(
        QuickFitTransitTimes, "QuickFitTransitTimes (pass 1)", "quick_fit_transit_times"
    )


if __name__ == "__main__":
    run()
