"""
Shared pytest configuration and fixtures

Real-data fixtures load K00137 (Kepler-18, 3 planets) from tests/data and
tests/catalogs; tests that use them, or that compile theano models or run
samplers, are marked @pytest.mark.slow.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data" / "MAST_downloads"
CATALOG_DIR = TESTS_DIR / "catalogs"

TARGET = "K00137"
KIC_ID = 8644288
VISITS = [4, 5, 6]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: real photometry, theano compiles, or samplers; skipped by test-fast",
    )


@pytest.fixture(scope="session")
def _k00137_session():
    from alderaan.utils.kepler import load_kepler_data

    return load_kepler_data(TARGET, str(DATA_DIR), str(CATALOG_DIR), visits=VISITS)


@pytest.fixture
def k00137(_k00137_session):
    """(star, planets, litecurve) for K00137 quarters 4-6; a fresh deep copy per test"""
    return copy.deepcopy(_k00137_session)


@pytest.fixture
def make_posteriors():
    """
    Factory for synthetic Posteriors: make_posteriors(npl, nsamp, target, **kwargs)
    builds uniform samples and dynesty-like sampler_info for npl planets with
    reference ephemerides spanning 130-200 d; kwargs override Posteriors arguments
    """
    from alderaan.data_models.posteriors import Posteriors

    def _make(npl=2, nsamp=200, target=TARGET, seed=0, **kwargs):
        rng = np.random.default_rng(seed)
        labels = []
        for n in range(npl):
            labels += f"C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n}".split()
        labels += ["LD_Q1", "LD_Q2"]
        logwt = rng.normal(-10, 2, nsamp)
        args = dict(
            samples=pd.DataFrame(
                rng.uniform(0.01, 0.99, (nsamp, len(labels))), columns=labels
            ),
            sampler_info={
                "logwt": logwt,
                "logl": np.sort(rng.normal(0, 1, nsamp)),
                "logz": np.logaddexp.accumulate(logwt),
                "logzerr": np.full(nsamp, 0.1),
                "logvol": -np.arange(nsamp) / 100,
                "samples_n": np.full(nsamp, 100),
                "niter": nsamp,
                "eff": 12.5,
                "batch_nlive": [100, 50],
            },
            ref_period=np.array([3.5, 7.6])[:npl],
            ref_epoch=np.array([131.1, 133.4])[:npl],
            t_start=130.0,
            t_end=200.0,
        )
        args.update(kwargs)
        return Posteriors("Kepler", target, **args)

    return _make
