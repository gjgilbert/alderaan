"""
Tests for alderaan/algorithms/quick_fit_transit_shape.py

Least-squares recovery of an injected transit shape (ror, b, T14) from a
synthetic detrended long cadence litecurve; fast tier only.
"""

import numpy as np
import pytest

from alderaan.algorithms import QuickFitTransitShape
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.transit_model import exposure_offsets, transit_flux
from alderaan.utils.astro import approx_ror, approx_transit_depth

LC = "long cadence"
U1, U2 = 0.4, 0.25
NOISE = 1e-4

# (period, epoch, ror, impact, T14) per planet; impacts are high enough for
# ingress/egress shape to constrain b
TRUTH = [(5.3, 102.1, 0.08, 0.7, 0.15), (11.7, 104.6, 0.05, 0.8, 0.2)]


def _synthetic_system(truth, depth_scale=1.3, duration_scale=1.2, seed=0):
    """
    Star, planets, and a 60-day detrended litecurve with injected transits;
    planets start from depths and durations offset from the truth
    """
    rng = np.random.default_rng(seed)
    cadno = np.arange(int(60 / EXPTIME[LC]))
    time = 100.0 + cadno * EXPTIME[LC]
    offsets = exposure_offsets(EXPTIME[LC], 15)
    star = Star("Kepler", "SYN", 1.01, 1, U1, U2)

    flux = np.ones_like(time)
    planets = []
    for period, epoch, ror, impact, T14 in truth:
        index = np.arange(int((time.max() - epoch) / period) + 1)
        flux += transit_flux(time, epoch, period, ror, impact, T14, U1, U2, offsets)
        flux -= 1.0

        p = Planet(
            "Kepler",
            "SYN",
            period,
            epoch,
            approx_ror(depth_scale * approx_transit_depth(ror, impact), 0.0),
            0.0,
            duration_scale * T14,
        )
        p.set_transit_times(
            TransitTimes("Kepler", "SYN", index, epoch + period * index)
        )
        p.set_static_period_and_epoch()
        planets.append(p)

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
    return star, planets, litecurve


class TestQuickFitTransitShape:
    @pytest.mark.parametrize("truth", [TRUTH[:1], TRUTH], ids=["1pl", "2pl"])
    def test_recovers_injected_shape(self, truth):
        star, planets, litecurve = _synthetic_system(truth)
        new_planets, new_litecurve = QuickFitTransitShape(
            star, planets, litecurve
        ).perform()

        for p, (_, _, ror, impact, T14) in zip(new_planets, truth):
            assert p.ror == pytest.approx(ror, rel=0.03)
            assert p.impact == pytest.approx(impact, abs=0.05)
            assert p.duration == pytest.approx(T14, rel=0.03)

        # shape only: ephemeris and photometry unchanged
        for old, new in zip(planets, new_planets):
            np.testing.assert_array_equal(
                new.transit_times.ttime_mod, old.transit_times.ttime_mod
            )
        np.testing.assert_array_equal(new_litecurve.flux, litecurve.flux)
