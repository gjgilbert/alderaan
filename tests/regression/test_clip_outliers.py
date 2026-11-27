"""Tests for alderaan.algorithms.clip_outliers.ClipOutliers"""

import numpy as np
import pytest

from alderaan.algorithms import ClipOutliers, RegularizeOMC
from alderaan.algorithms.clip_outliers import FAR, IN, NEAR
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


def _dist(planets, lc):
    """Time from the nearest transit, in units of T14"""
    p = planets[0]
    d = np.min(np.abs(lc.time[:, None] - p.transit_times.ttime_mod[None, :]), axis=1)
    return d / p.duration


def _regions(planets, lc):
    """Boolean (in, near, far) masks: +/- 0.5 T14; wide mask (3 T14) out of it"""
    d = _dist(planets, lc)
    return d < 0.5, (d >= 0.5) & (d < 1.5), d >= 1.5


def _pick(mask, n, seed=1):
    return np.random.default_rng(seed).choice(np.flatnonzero(mask), n, replace=False)


def _clip(star, planets, lc, method):
    return ClipOutliers(star, planets, lc).perform(method=method, raw_litecurve=lc)[1]


@pytest.mark.parametrize("method", ["local_trend", "transit_model"])
def test_removes_far_outliers_and_keeps_transits(method):
    star, planets, lc = _system()
    in_t, near, far = _regions(planets, lc)
    bad = _pick(far, 20)
    lc.flux[bad] += 30 * NOISE * np.where(np.arange(20) % 2 == 0, 1, -1)

    lc_out = _clip(star, planets, lc, method)

    assert not np.any(np.isin(lc.cadno[bad], lc_out.cadno))
    assert np.all(np.isin(lc.cadno[in_t | near], lc_out.cadno))
    assert len(lc_out.time) >= len(lc.time) - len(bad) - 3
    # kept cadences are unchanged
    np.testing.assert_array_equal(lc_out.flux, lc.flux[lc_out.cadno])
    np.testing.assert_array_equal(lc_out.visit, lc.visit[lc_out.cadno])


def test_rejects_unknown_method():
    star, planets, lc = _system()
    with pytest.raises(ValueError, match="method"):
        ClipOutliers(star, planets, lc).perform(method="median")


def test_transit_model_requires_raw_litecurve():
    star, planets, lc = _system()
    with pytest.raises(ValueError, match="raw_litecurve"):
        ClipOutliers(star, planets, lc).perform(method="transit_model")


class TestSigmaThreshold:
    def _sigma(self, flux, lc):
        u = lc.visit == 1
        mask = np.zeros(np.sum(u), dtype=bool)
        return ClipOutliers._sigma_threshold(lc.time[u], flux[u], mask, EXP_LC)

    def test_white_noise_is_near_5(self):
        rng = np.random.default_rng(3)
        _, _, lc = _system()
        sigma, beta = self._sigma(1 + rng.normal(0, NOISE, len(lc.time)), lc)
        assert 0.8 < beta < 1.3
        assert 5.0 <= sigma < 6.5

    def test_coherent_variability_raises_sigma(self):
        # variability on the kernel timescale: beta >> 1
        rng = np.random.default_rng(3)
        _, _, lc = _system()
        flux = 1 + 20 * NOISE * np.sin(2 * np.pi * lc.time / 0.5)
        sigma, beta = self._sigma(flux + rng.normal(0, NOISE, len(flux)), lc)
        assert beta > 2
        assert sigma == pytest.approx(5 * beta)


class TestLocalTrend:
    def test_upper_outlier_in_transit_removed_lower_kept(self):
        # a transit cannot raise flux; a dip in transit is not an outlier
        star, planets, lc = _system()
        in_t, _, _ = _regions(planets, lc)
        up, down = _pick(in_t, 2)
        lc.flux[up] += 100 * NOISE
        lc.flux[down] -= 100 * NOISE

        lc_out = _clip(star, planets, lc, "local_trend")

        assert lc.cadno[up] not in lc_out.cadno
        assert lc.cadno[down] in lc_out.cadno

    def test_thresholds_by_region(self):
        # on the far scale: far both tails at sigma; near upper tail at
        # 2 x sigma; in transit upper tail at sigma
        rng = np.random.default_rng(2)
        region = np.repeat([FAR, NEAR, IN], [4000, 400, 400])
        resid = rng.normal(0, 1e-3, len(region))
        resid = np.clip(resid, -3e-3, 3e-3)  # no chance 5-sigma tails
        for i in (1, 4001, 4401):  # far, near, in transit
            resid[i : i + 3] = [8e-3, -8e-3, 12e-3]

        keep = ClipOutliers._clip_outliers(resid, 5.0, region, "local_trend")

        np.testing.assert_array_equal(keep[1:4], [False, False, False])
        np.testing.assert_array_equal(keep[4001:4004], [True, True, False])
        np.testing.assert_array_equal(keep[4401:4404], [False, True, False])
        assert keep.sum() == len(keep) - 3 - 1 - 2

    @pytest.mark.parametrize("depth, duration", [(0.02, 0.125), (4e-3, 8 / 24)])
    def test_keeps_deep_and_long_transits(self, depth, duration):
        # 2% depth (200 sigma) or T14 = 8 hr, longer than the 13-cadence kernel
        star, planets, lc = _system(depth=depth, duration=duration)
        in_t, near, _ = _regions(planets, lc)

        lc_out = _clip(star, planets, lc, "local_trend")

        assert np.all(np.isin(lc.cadno[in_t | near], lc_out.cadno))


class TestTransitModel:
    def test_removes_outliers_inside_transit(self):
        star, planets, lc = _system()
        in_t, _, _ = _regions(planets, lc)
        up, down = _pick(in_t, 2)
        lc.flux[up] += 30 * NOISE
        lc.flux[down] -= 30 * NOISE

        lc_out = _clip(star, planets, lc, "transit_model")

        assert lc.cadno[up] not in lc_out.cadno
        assert lc.cadno[down] not in lc_out.cadno
        assert np.isin(lc.cadno[in_t], lc_out.cadno).sum() == in_t.sum() - 2

    def test_each_region_uses_its_own_noise(self):
        # detrending noise 4 x higher near transit: near is not over-clipped,
        # and a far 8-sigma spike is still rejected
        star, planets, lc = _system()
        _, near, far = _regions(planets, lc)
        rng = np.random.default_rng(5)
        lc.flux[near] += rng.normal(0, np.sqrt(15) * NOISE, near.sum())
        f = _pick(far, 5)
        lc.flux[f] += 8 * NOISE

        lc_out = _clip(star, planets, lc, "transit_model")

        assert np.isin(lc.cadno[near], lc_out.cadno).sum() >= near.sum() - 1
        assert not np.any(np.isin(lc.cadno[f], lc_out.cadno))

    def test_model_error_in_deep_transit_is_not_clipped(self):
        # model 10% too large in ror: a smooth in-transit error, not outliers
        star, planets, lc = _system(depth=0.02)
        in_t, _, _ = _regions(planets, lc)
        planets[0].ror *= 1.1

        lc_out = _clip(star, planets, lc, "transit_model")

        assert np.isin(lc.cadno[in_t], lc_out.cadno).mean() > 0.95


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
