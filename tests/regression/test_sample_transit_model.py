"""
Tests for alderaan/algorithms/sample_transit_model.py

Fast tier: prior transform bounds. Slow tier: a full dynesty run on a short
synthetic single-planet litecurve (~1 min), checking that the injected truth
lies within the posterior.
"""

import numpy as np
import pytest

from alderaan.algorithms import SampleTransitModel
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Posteriors, Star, TransitTimes
from alderaan.transit_model import TransitModel, exposure_offsets, transit_flux
from alderaan.utils.astro import ld_u_to_q

LC = "long cadence"
U1, U2 = 0.4, 0.25
PERIOD, EPOCH, ROR, IMPACT, T14 = 5.3, 102.1, 0.08, 0.3, 0.15
NOISE = 1e-4


def _synthetic_system(seed=0):
    """Star, one planet, and a 30-day detrended litecurve with injected transits"""
    rng = np.random.default_rng(seed)
    cadno = np.arange(int(30 / EXPTIME[LC]))
    time = 100.0 + cadno * EXPTIME[LC]
    index = np.arange(int((time.max() - EPOCH) / PERIOD) + 1)

    offsets = exposure_offsets(EXPTIME[LC], 15)
    flux = transit_flux(time, EPOCH, PERIOD, ROR, IMPACT, T14, U1, U2, offsets)
    flux += NOISE * rng.standard_normal(len(time))
    litecurve = LiteCurve(
        "Kepler",
        "SYN",
        time,
        flux,
        np.full_like(time, NOISE),
        cadno,
        np.ones_like(cadno),
        np.full(len(time), LC),
    )

    star = Star("Kepler", "SYN", 1.01, 1, U1, U2)
    planet = Planet("Kepler", "SYN", PERIOD, EPOCH, ROR, IMPACT, T14)
    planet.set_transit_times(
        TransitTimes("Kepler", "SYN", index, EPOCH + PERIOD * index)
    )
    planet.set_static_period_and_epoch()

    return star, [planet], litecurve


class TestPriorTransform:
    def test_maps_unit_cube_edges_to_prior_bounds(self):
        star, planets, litecurve = _synthetic_system()
        tm = TransitModel(star, planets, litecurve)

        lo = SampleTransitModel._prior_transform(np.zeros(7), tm)
        hi = SampleTransitModel._prior_transform(np.ones(7), tm)
        mid = SampleTransitModel._prior_transform(np.full(7, 0.5), tm)

        # [C0, C1, ror, b, T14, q1, q2]
        np.testing.assert_allclose(lo[2:], [1e-5, 0.0, 0.01 * T14, 0.0, 0.0])
        np.testing.assert_allclose(hi[2:], [0.99, 1.99, 3 * T14, 1.0, 1.0])
        np.testing.assert_allclose(mid[:2], [0.0, 0.0], atol=1e-12)
        # log-uniform ror and T14: median at the geometric mean of the bounds
        assert mid[2] == pytest.approx(np.sqrt(1e-5 * 0.99))
        assert mid[4] == pytest.approx(np.sqrt(0.01 * 3) * T14)
        # b is uniform on [0, 1 + ror]
        assert mid[3] == pytest.approx(0.5 * (1 + mid[2]))


@pytest.mark.slow
class TestSample:
    def test_posterior_contains_truth(self):
        star, planets, litecurve = _synthetic_system()
        posteriors = SampleTransitModel(star, planets, litecurve).perform()

        assert isinstance(posteriors, Posteriors)
        assert list(posteriors.samples.columns) == [
            "C0_0",
            "C1_0",
            "ROR_0",
            "IMPACT_0",
            "DUR14_0",
            "LD_Q1",
            "LD_Q2",
        ]
        assert posteriors.neff() > 100

        q1, q2 = ld_u_to_q(U1, U2)
        truth = {
            "C0_0": 0.0,
            "C1_0": 0.0,
            "ROR_0": ROR,
            "IMPACT_0": IMPACT,
            "DUR14_0": T14,
            "LD_Q1": q1,
            "LD_Q2": q2,
        }
        summary = posteriors.summary()
        for label, value in truth.items():
            assert summary.loc[label, "CI_2.5"] < value < summary.loc[label, "CI_97.5"]

        # the data, not the prior, set the posterior: 95% CI widths far narrower
        # than the prior ranges (timing in days)
        width = summary["CI_97.5"] - summary["CI_2.5"]
        assert width["ROR_0"] < 0.1 * ROR
        assert width["DUR14_0"] < 0.05 * T14
        assert width["C0_0"] < 2 / 1440
        assert width["C1_0"] < 2 / 1440
