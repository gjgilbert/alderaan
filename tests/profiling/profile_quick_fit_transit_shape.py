"""Profile ``QuickFitTransitShape`` (both iterations) on K00137.

Run with ``make profile-quick_fit_transit_shape`` or
``python tests/profiling/profile_quick_fit_transit_shape.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import QuickFitTransitShape


def run():
    return P.profile_algorithm(
        QuickFitTransitShape,
        "QuickFitTransitShape (both iterations)",
        "quick_fit_transit_shape",
    )


if __name__ == "__main__":
    run()
