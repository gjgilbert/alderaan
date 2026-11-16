"""Profile ``SampleTransitModel`` (capped dynesty) on K00137.

Run with ``make profile-sample_transit_model`` or
``python tests/profiling/profile_sample_transit_model.py``.
"""

try:  # importable via `-m tests.profiling.profile_*` or runnable as a script
    from tests.profiling import _profiling as P
except ModuleNotFoundError:
    import _profiling as P

from alderaan.algorithms import SampleTransitModel


def run():
    return P.profile_algorithm(
        SampleTransitModel,
        "SampleTransitModel (capped dynesty)",
        "sample_transit_model",
    )


if __name__ == "__main__":
    run()
