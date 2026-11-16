"""Profile ``RegularizeOMC`` (smooth + select) on K00137.

Run with ``make profile-regularize_omc`` or
``python tests/profiling/profile_regularize_omc.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import RegularizeOMC


def run():
    return P.profile_algorithm(
        RegularizeOMC, "RegularizeOMC (smooth + select)", "regularize_omc"
    )


if __name__ == "__main__":
    run()
