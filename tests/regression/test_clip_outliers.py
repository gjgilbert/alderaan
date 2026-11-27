"""Tests for alderaan.algorithms.clip_outliers.ClipOutliers"""

import copy

import numpy as np
import pytest

from alderaan.algorithms import ClipOutliers, RegularizeOMC
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.transit_model import TransitModel

EXP_LC = EXPTIME["long cadence"]
NOISE = 1e-4


def _system(duration=0.125, depth=4e-3, t_max=60.0, seed=42):
    """
    Star, one planet (P = 4.3 d), and two-visit long cadence litecurve whose
    flux is the transit model plus white noise; cadence index is the cadno
    """
    star = Star("Kepler", "SYN", koi_id=1, kic_id=1, u1=0.4, u2=0.25)

    period, epoch = 4.3, 1.7
    p = Planet("Kepler", "SYN", period, epoch, np.sqrt(depth), 0.3, duration)
    p.set_transit_times(
        TransitTimes.from_linear_ephemeris("Kepler", "SYN", period, epoch, 0, t_max)
    )
    p.set_static_period_and_epoch()

    time = np.arange(0.0, t_max, EXP_LC)
    npts = len(time)
    lc = LiteCurve(
        "Kepler",
        "SYN",
        time,
        np.ones(npts),
        np.full(npts, NOISE),
        np.arange(npts),
        np.where(time < t_max / 2, 1, 2),
        np.full(npts, "long cadence"),
    )

    tm = TransitModel(star, [p], lc)
    theta = tm.theta_initial()
    theta[2], theta[3] = p.ror, p.impact
    rng = np.random.default_rng(seed)
    lc.flux = tm.model_flux(theta) + rng.normal(0, NOISE, npts)

    return star, [p], lc


def _in_transit(planets, lc):
    p = planets[0]
    return np.any(
        [np.abs(lc.time - tc) < 0.5 * p.duration for tc in p.transit_times.ttime_mod],
        axis=0,
    )


def _inject_outliers(lc, in_transit, n=20, amp=30 * NOISE, seed=1):
    """Alternating +/- spikes at random out-of-transit cadences; returns cadnos"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.flatnonzero(~in_transit), n, replace=False)
    lc.flux[idx] += amp * np.where(np.arange(n) % 2 == 0, 1, -1)
    return lc.cadno[idx]


@pytest.mark.parametrize("method", ["local_trend", "transit_model"])
def test_removes_outliers_and_keeps_transits(method):
    star, planets, lc = _system()
    in_transit = _in_transit(planets, lc)
    # few enough (< 0.3% of out-of-mask cadences per visit) not to skip rejection
    bad = _inject_outliers(lc, in_transit, n=4)

    # quiet star: raw and detrended photometry are the same
    _, lc_out = ClipOutliers(star, planets, lc).perform(method=method, raw_litecurve=lc)

    assert not np.any(np.isin(bad, lc_out.cadno))
    # few clean cadences are lost, and none in transit
    assert len(lc_out.time) >= len(lc.time) - len(bad) - 5
    assert np.all(np.isin(lc.cadno[in_transit], lc_out.cadno))
    # kept cadences are unchanged
    np.testing.assert_array_equal(lc_out.flux, lc.flux[lc_out.cadno])
    np.testing.assert_array_equal(lc_out.visit, lc.visit[lc_out.cadno])


class TestTransitModel:
    def test_removes_downward_outlier_inside_transit(self):
        # clipping about the transit model needs no transit mask
        star, planets, lc = _system()
        in_transit = _in_transit(planets, lc)
        dip = np.flatnonzero(in_transit)[len(np.flatnonzero(in_transit)) // 2]
        lc.flux[dip] -= 30 * NOISE

        _, lc_out = ClipOutliers(star, planets, lc).perform(
            method="transit_model", raw_litecurve=lc
        )

        assert lc.cadno[dip] not in lc_out.cadno
        kept = np.isin(lc.cadno[in_transit], lc_out.cadno)
        assert kept.sum() == in_transit.sum() - 1

    @pytest.mark.parametrize("variable", [False, True])
    def test_sigma_grows_with_stellar_variability(self, variable):
        # 8 sigma in transit: an outlier on a quiet star (sigma x 1.08), kept on
        # a variable star with rms ratio 7 (sigma x 3.0)
        star, planets, lc = _system()
        in_transit = _in_transit(planets, lc)
        dip = np.flatnonzero(in_transit)[len(np.flatnonzero(in_transit)) // 2]
        lc.flux[dip] -= 8 * NOISE
        raw = copy.deepcopy(lc)
        if variable:
            # raw rms 7 x NOISE: sinusoid std sqrt(48) x NOISE added to the noise
            raw.flux += np.sqrt(96) * NOISE * np.sin(2 * np.pi * raw.time)

        _, lc_out = ClipOutliers(star, planets, lc).perform(
            method="transit_model", raw_litecurve=raw
        )

        assert (lc.cadno[dip] in lc_out.cadno) == variable


class TestLocalTrend:
    @pytest.mark.parametrize("nspike", [1, 2])
    def test_in_transit_rate_limit(self, nspike):
        # < 100 in-transit cadences in the visit: one cosmic ray in transit is
        # rejected; two are > 1% and > 1 cadence, so that visit rejects nothing
        star, planets, lc = _system()
        in_transit = _in_transit(planets, lc)
        in_visit = np.flatnonzero(in_transit & (lc.visit == 1))
        assert len(in_visit) < 100
        spikes = in_visit[[len(in_visit) // 4, 3 * len(in_visit) // 4]][:nspike]
        lc.flux[spikes] += 100 * NOISE

        _, lc_out = ClipOutliers(star, planets, lc).perform(method="local_trend")

        kept = np.isin(lc.cadno[spikes], lc_out.cadno)
        assert np.all(kept) if nspike == 2 else not np.any(kept)

    def test_keeps_cadences_bordering_a_long_transit(self):
        # T14 = 8 hr is longer than the 13-cadence (6.4 hr) smoothing kernel
        star, planets, lc = _system(duration=8 / 24)
        near = np.any(
            [
                np.abs(lc.time - tc) < 1.5 * planets[0].duration
                for tc in planets[0].transit_times.ttime_mod
            ],
            axis=0,
        )

        _, lc_out = ClipOutliers(star, planets, lc).perform(method="local_trend")

        assert np.all(np.isin(lc.cadno[near], lc_out.cadno))


@pytest.mark.slow
class TestClipOutliersK00137:
    def test_smooth_then_local_trend(self, k00137):
        star, planets, lc = k00137
        planets, lc = RegularizeOMC(star, planets, lc).perform(method="smooth")
        planets_out, lc_out = ClipOutliers(star, planets, lc).perform(
            method="local_trend"
        )

        assert len(lc_out.time) > 0.98 * len(lc.time)
        assert np.all(np.isfinite(lc_out.flux))
        assert set(np.unique(lc_out.visit)) == set(np.unique(lc.visit))

        # every cadence within any transit window survives
        alg = ClipOutliers(star, planets, lc)
        in_transit = alg.make_transit_mask(rel_size=1.0)
        assert np.isin(lc.cadno[in_transit], lc_out.cadno).mean() > 0.99
        for p, p_out in zip(planets, planets_out):
            np.testing.assert_array_equal(
                p_out.transit_times.ttime_mod, p.transit_times.ttime_mod
            )
