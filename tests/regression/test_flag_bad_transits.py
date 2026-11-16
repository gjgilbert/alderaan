"""Tests for alderaan.algorithms.flag_bad_transits.FlagBadTransits"""

import numpy as np
import pytest

from alderaan.algorithms import FlagBadTransits
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes

NOISE = 1e-4
PERIOD, EPOCH, DURATION = 3.1, 1.3, 0.2


def _system(t_max=80.0, seed=7):
    """Star, one planet, and a flat long cadence litecurve with white noise"""
    star = Star("Kepler", "SYN", koi_id=1, kic_id=1, u1=0.4, u2=0.25)
    p = Planet("Kepler", "SYN", PERIOD, EPOCH, 1e-3, DURATION, 0.3)
    p.set_transit_times(
        TransitTimes.from_linear_ephemeris("Kepler", "SYN", PERIOD, EPOCH, 0, t_max)
    )

    time = np.arange(0.0, t_max, EXPTIME["long cadence"])
    npts = len(time)
    rng = np.random.default_rng(seed)
    lc = LiteCurve(
        "Kepler",
        "SYN",
        time,
        1 + rng.normal(0, NOISE, npts),
        np.full(npts, NOISE),
        np.arange(npts),
        np.ones(npts, dtype=int),
        np.full(npts, "long cadence"),
    )
    return star, [p], lc


def _tc(planets, i):
    return planets[0].transit_times.ttime_mod[i]


def _run(star, planets, lc):
    planets_out, _ = FlagBadTransits(star, planets, lc).perform()
    return planets_out[0].transit_times.index


class TestCoverage:
    def test_keeps_all_well_covered_transits(self):
        star, planets, lc = _system()
        np.testing.assert_array_equal(
            _run(star, planets, lc), planets[0].transit_times.index
        )

    def test_drops_transit_in_a_data_gap(self):
        star, planets, lc = _system()
        tc = _tc(planets, 5)
        lc.select(np.abs(lc.time - tc) > 2 * DURATION)

        index = _run(star, planets, lc)
        assert 5 not in index
        assert len(index) == len(planets[0].transit_times.index) - 1

    def test_drops_transit_with_sparse_in_transit_coverage(self):
        # in-transit cadences thinned to 1 in 4 (< 50% of expected)
        star, planets, lc = _system()
        tc = _tc(planets, 8)
        in_transit = np.abs(lc.time - tc) < 0.5 * DURATION
        lc.select(~in_transit | (np.cumsum(in_transit) % 4 == 0))

        assert 8 not in _run(star, planets, lc)

    def test_keeps_transit_with_partial_baseline_coverage(self):
        # transit fully sampled; one side of the out-of-transit baseline missing
        star, planets, lc = _system()
        tc = _tc(planets, 8)
        lc.select(~((lc.time > tc + 0.5 * DURATION) & (lc.time < tc + 1.5 * DURATION)))

        assert 8 in _run(star, planets, lc)


class TestNoise:
    def test_drops_transit_with_high_noise(self):
        star, planets, lc = _system()
        tc = _tc(planets, 12)
        window = np.abs(lc.time - tc) < 1.5 * DURATION
        rng = np.random.default_rng(3)
        lc.flux[window] = 1 + rng.normal(0, 10 * NOISE, window.sum())

        index = _run(star, planets, lc)
        assert 12 not in index
        assert len(index) == len(planets[0].transit_times.index) - 1

    def test_overlapping_transit_exempt_from_noise_check(self):
        # a second planet transits at the same time as planet 0's transit 12
        star, planets, lc = _system()
        tc = _tc(planets, 12)
        p2 = Planet("Kepler", "SYN", 37.0, tc, 1e-3, DURATION, 0.3)
        p2.set_transit_times(
            TransitTimes.from_linear_ephemeris("Kepler", "SYN", 37.0, tc, 0, 80.0)
        )
        window = np.abs(lc.time - tc) < 1.5 * DURATION
        rng = np.random.default_rng(3)
        lc.flux[window] = 1 + rng.normal(0, 10 * NOISE, window.sum())

        planets_out, _ = FlagBadTransits(star, planets + [p2], lc).perform()
        assert 12 in planets_out[0].transit_times.index


@pytest.mark.slow
class TestFlagBadTransitsK00137:
    def test_keeps_covered_transits(self, k00137):
        star, planets, lc = k00137
        planets_out, lc_out = FlagBadTransits(star, planets, lc).perform()

        np.testing.assert_array_equal(lc_out.flux, lc.flux)
        for p, p_out in zip(planets, planets_out):
            tt = p_out.transit_times
            assert len(tt.index) >= 0.8 * len(p.transit_times.index)
            assert np.all(np.isin(tt.index, p.transit_times.index))
            # every kept transit has photometry at its midpoint
            dt = np.min(np.abs(lc.time[:, None] - tt.ttime_mod[None, :]), axis=0)
            assert np.all(dt < 0.5 * p.duration)
