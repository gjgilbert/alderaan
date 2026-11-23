"""Shared pytest configuration and fixtures.

Real-data fixtures load K00137 (Kepler-18, 3 planets) from tests/data and
reference/; tests that use them, or that run samplers, are @pytest.mark.slow.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data" / "MAST_downloads"
CATALOG = TESTS_DIR.parent / "reference" / "alderaan" / "kepler_merged_alderaan.csv"

TARGET = "K00137"
KIC_ID = 8644288
VISITS = [4, 5, 6]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: real photometry or samplers; skipped by test-fast",
    )


@pytest.fixture(scope="session")
def _k00137_session():
    from alderaan.utils.kepler import load_kepler_data

    return load_kepler_data(TARGET, str(DATA_DIR), str(CATALOG), visits=VISITS)


@pytest.fixture
def k00137(_k00137_session):
    """(star, planets, litecurve) for K00137 quarters 4-6; deep-copied per test."""
    return copy.deepcopy(_k00137_session)


@pytest.fixture
def make_posteriors():
    """Factory for synthetic Posteriors.

    make_posteriors(npl, nsamp, target, seed, **kwargs) builds uniform samples and
    dynesty-like sampler_info for npl <= 2 planets on a 130-200 d baseline;
    kwargs override Posteriors arguments.
    """
    from alderaan.data_models import Posteriors

    def _make(npl=2, nsamp=200, target=TARGET, seed=0, **kwargs):
        rng = np.random.default_rng(seed)
        labels = []
        for n in range(npl):
            labels += f"C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n}".split()
        labels += ["LD_Q1", "LD_Q2"]
        logl = np.sort(rng.normal(0, 1, nsamp))
        logvol = -np.arange(1, nsamp + 1) / 100
        args = dict(
            samples=pd.DataFrame(
                rng.uniform(0.01, 0.99, (nsamp, len(labels))), columns=labels
            ),
            sampler_info={
                "logl": logl,
                "logvol": logvol,
                **Posteriors._integrals(logl, logvol),
                "samples_n": np.full(nsamp, 100),
                "ncall": np.full(nsamp, 3),
                "samples_id": rng.integers(0, 150, nsamp),
                "samples_batch": rng.integers(0, 2, nsamp),
                "niter": nsamp,
                "eff": 12.5,
                "batch_nlive": [100, 50],
                "batch_logl_bounds": np.array([[-np.inf, np.inf], [-1.0, 1.0]]),
            },
            ref_period=np.array([3.5, 7.6])[:npl],
            ref_epoch=np.array([131.1, 133.4])[:npl],
            t_start=130.0,
            t_end=200.0,
            supersample={"long cadence": 15, "short cadence": 1},
        )
        args.update(kwargs)
        return Posteriors("Kepler", target, **args)

    return _make
