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
    p = Planet("Kepler", "SYN", PERIOD, EPOCH, 1e-3, DURATION)
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


def _run(star, planets, lc, flag):
    """Index of transits passing one photometry flag; every transit is kept"""
    planets_out, _ = FlagBadTransits(star, planets, lc).perform()
    tt = planets_out[0].transit_times
    np.testing.assert_array_equal(tt.index, planets[0].transit_times.index)
    return tt.index[getattr(tt, flag)]


class TestCoverage:
    def test_keeps_omc_model(self):
        star, planets, lc = _system()
        tt = planets[0].transit_times
        planets[0].set_transit_times(
            TransitTimes(
                tt.mission,
                tt.target,
                tt.index,
                tt.ttime_obs,
                model_name="poly_1",
                model_dof=2.0,
                ttv_scatter=0.003,
            )
        )
        planets_out, _ = FlagBadTransits(star, planets, lc).perform()
        tt = planets_out[0].transit_times
        assert (tt.model_name, tt.model_dof, tt.ttv_scatter) == ("poly_1", 2.0, 0.003)

    def test_keeps_all_well_covered_transits(self):
        star, planets, lc = _system()
        for flag in ["phot_coverage", "phot_noise"]:
            np.testing.assert_array_equal(
                _run(star, planets, lc, flag), planets[0].transit_times.index
            )

    def test_flags_transit_in_a_data_gap(self):
        star, planets, lc = _system()
        tc = _tc(planets, 5)
        lc.select(np.abs(lc.time - tc) > 2 * DURATION)

        index = _run(star, planets, lc, "phot_coverage")
        assert 5 not in index
        assert len(index) == len(planets[0].transit_times.index) - 1

    def test_flags_transit_with_sparse_in_transit_coverage(self):
        # in-transit cadences thinned to 1 in 4 (< 50% of expected)
        star, planets, lc = _system()
        tc = _tc(planets, 8)
        in_transit = np.abs(lc.time - tc) < 0.5 * DURATION
        lc.select(~in_transit | (np.cumsum(in_transit) % 4 == 0))

        assert 8 not in _run(star, planets, lc, "phot_coverage")

    def test_keeps_transit_with_partial_baseline_coverage(self):
        # transit fully sampled; one side of the out-of-transit baseline missing
        star, planets, lc = _system()
        tc = _tc(planets, 8)
        lc.select(~((lc.time > tc + 0.5 * DURATION) & (lc.time < tc + 1.5 * DURATION)))

        assert 8 in _run(star, planets, lc, "phot_coverage")


class TestNoise:
    def test_flags_transit_with_high_noise(self):
        star, planets, lc = _system()
        tc = _tc(planets, 12)
        window = np.abs(lc.time - tc) < 1.5 * DURATION
        rng = np.random.default_rng(3)
        lc.flux[window] = 1 + rng.normal(0, 10 * NOISE, window.sum())

        index = _run(star, planets, lc, "phot_noise")
        assert 12 not in index
        assert len(index) == len(planets[0].transit_times.index) - 1

    def test_overlapping_transit_exempt_from_noise_check(self):
        # a second planet transits at the same time as planet 0's transit 12
        star, planets, lc = _system()
        tc = _tc(planets, 12)
        p2 = Planet("Kepler", "SYN", 37.0, tc, 1e-3, DURATION)
        p2.set_transit_times(
            TransitTimes.from_linear_ephemeris("Kepler", "SYN", 37.0, tc, 0, 80.0)
        )
        window = np.abs(lc.time - tc) < 1.5 * DURATION
        rng = np.random.default_rng(3)
        lc.flux[window] = 1 + rng.normal(0, 10 * NOISE, window.sum())

        planets_out, _ = FlagBadTransits(star, planets + [p2], lc).perform()
        assert planets_out[0].transit_times.phot_noise[12]

    def test_few_transits_skip_noise_check(self):
        # 7 transits: too few for the noise spread, but coverage is still checked
        star, planets, lc = _system(t_max=20.0)
        assert len(planets[0].transit_times.index) == 7
        window = np.abs(lc.time - _tc(planets, 3)) < 1.5 * DURATION
        rng = np.random.default_rng(3)
        lc.flux[window] = 1 + rng.normal(0, 10 * NOISE, window.sum())
        lc.select(np.abs(lc.time - _tc(planets, 5)) > 2 * DURATION)

        planets_out, _ = FlagBadTransits(star, planets, lc).perform()
        tt = planets_out[0].transit_times
        assert tt.phot_noise[3]
        assert not tt.phot_coverage[5]


class TestTooFewTransits:
    def test_keeps_planet_with_three_good_transits(self):
        star, planets, lc = _system(t_max=8.0)
        assert len(planets[0].transit_times.index) == 3
        planets_out, lc_out = FlagBadTransits(star, planets, lc).perform()
        assert len(planets_out) == 1
        np.testing.assert_array_equal(lc_out.time, lc.time)

    def test_single_planet_with_two_good_transits_raises(self):
        star, planets, lc = _system(t_max=8.0)
        lc.select(np.abs(lc.time - _tc(planets, 2)) > 2 * DURATION)
        with pytest.raises(ValueError, match="2 good transits"):
            FlagBadTransits(star, planets, lc).perform()

    def test_drops_planet_and_its_transits_in_multiplanet_system(self, caplog):
        star, planets, lc = _system()
        p2 = Planet("Kepler", "SYN", 37.0, 21.5, 1e-3, DURATION)
        p2.set_transit_times(
            TransitTimes.from_linear_ephemeris("Kepler", "SYN", 37.0, 21.5, 0, 80.0)
        )
        tc2 = p2.transit_times.ttime_mod
        assert len(tc2) == 2

        planets_out, lc_out = FlagBadTransits(star, planets + [p2], lc).perform()

        assert len(planets_out) == 1
        assert planets_out[0].period == pytest.approx(PERIOD)
        assert "planet 1: 2 good transits" in caplog.text
        # 3 T14 window
        near = np.any(np.abs(lc.time[:, None] - tc2) < 1.5 * DURATION, axis=1)
        np.testing.assert_array_equal(lc_out.time, lc.time[~near])
        np.testing.assert_array_equal(lc_out.flux, lc.flux[~near])


@pytest.mark.slow
class TestFlagBadTransitsK00137:
    def test_keeps_covered_transits(self, k00137):
        star, planets, lc = k00137
        planets_out, lc_out = FlagBadTransits(star, planets, lc).perform()

        np.testing.assert_array_equal(lc_out.flux, lc.flux)
        for p, p_out in zip(planets, planets_out):
            tt = p_out.transit_times
            np.testing.assert_array_equal(tt.index, p.transit_times.index)
            phot = tt.phot_coverage & tt.phot_noise
            assert np.mean(phot) >= 0.8
            # every good transit has photometry at its midpoint
            tc = tt.ttime_mod[phot]
            dt = np.min(np.abs(lc.time[:, None] - tc[None, :]), axis=0)
            assert np.all(dt < 0.5 * p.duration)
