"""Profile ``FlagBadTransits`` (both passes) on K00137.

Run with ``make profile-flag_bad_transits`` or
``python tests/profiling/profile_flag_bad_transits.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import FlagBadTransits


def run():
    return P.profile_algorithm(
        FlagBadTransits, "FlagBadTransits (both passes)", "flag_bad_transits"
    )


if __name__ == "__main__":
    run()
