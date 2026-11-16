"""Tests for alderaan.algorithms.regularize_omc.RegularizeOMC"""

import numpy as np
import pytest

from alderaan.algorithms import RegularizeOMC
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes

PERIOD, EPOCH = 10.0, 5.0
TTV_AMP, TTV_PERIOD = 15 / 1440, 400.0  # 15 min sinusoid over a 400 d period
TIME_ERR = 1 / 1440  # 1 min


def _system(ntransit=140, outliers=(), missing=(), seed=11):
    """
    Star, one planet with sinusoidal TTVs plus white timing noise, and a
    placeholder litecurve; returns (star, planets, litecurve, true transit times)
    """
    star = Star("Kepler", "SYN", koi_id=1, kic_id=1, u1=0.4, u2=0.25)

    index = np.arange(ntransit)
    linear = EPOCH + PERIOD * index
    truth = linear + TTV_AMP * np.sin(2 * np.pi * linear / TTV_PERIOD)

    rng = np.random.default_rng(seed)
    ttime_obs = truth + rng.normal(0, TIME_ERR, ntransit)
    error_obs = np.full(ntransit, TIME_ERR)
    for i in outliers:
        ttime_obs[i] += 60 / 1440
    for i in missing:
        ttime_obs[i] = np.nan
        error_obs[i] = np.nan

    p = Planet("Kepler", "SYN", PERIOD, EPOCH, 1e-3, 0.15, 0.3)
    # unmeasured transits start on the linear ephemeris
    ttime_mod = np.where(np.isfinite(ttime_obs), ttime_obs, linear)
    p.set_transit_times(
        TransitTimes(
            "Kepler", "SYN", index, ttime_obs, error_obs=error_obs, ttime_mod=ttime_mod
        )
    )

    time = np.linspace(0.0, EPOCH + PERIOD * ntransit, 100)
    lc = LiteCurve(
        "Kepler",
        "SYN",
        time,
        np.ones(100),
        np.full(100, 1e-4),
        np.arange(100),
        np.ones(100, dtype=int),
        np.full(100, "long cadence"),
    )
    return star, [p], lc, truth


def _rms(x):
    return np.sqrt(np.mean(x**2))


class TestSmooth:
    def test_recovers_injected_ttv(self):
        star, planets, lc, truth = _system()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        tt = planets_out[0].transit_times
        residual = tt.ttime_mod - truth
        assert _rms(residual) < 0.75 * TIME_ERR
        # signal is kept, not smoothed away
        assert np.ptp(tt.ttime_mod - tt.linear_ephemeris()) > 1.5 * TTV_AMP
        np.testing.assert_allclose(planets_out[0].period, PERIOD, atol=1e-4)

    def test_flags_outlier_and_replaces_it(self):
        star, planets, lc, truth = _system(outliers=[40, 90])
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        tt = planets_out[0].transit_times
        np.testing.assert_array_equal(np.flatnonzero(tt.out_flag), [40, 90])
        assert np.all(np.abs(tt.ttime_mod[[40, 90]] - truth[[40, 90]]) < 3 * TIME_ERR)
        assert np.all(np.isnan(tt.out_prob))

    def test_fills_missing_epochs(self):
        star, planets, lc, truth = _system(missing=[0, 60, 61])
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        tt = planets_out[0].transit_times
        np.testing.assert_array_equal(tt.index, np.arange(140))
        assert np.all(np.isnan(tt.ttime_obs[[0, 60, 61]]))
        assert not np.any(tt.out_flag)
        assert np.all(
            np.abs(tt.ttime_mod[[0, 60, 61]] - truth[[0, 60, 61]]) < 3 * TIME_ERR
        )


class TestSelectBestModel:
    """AIC/BIC model selection on fixed model predictions"""

    @staticmethod
    def _fits(**trends):
        return {k: {"trend": v, "pred": v} for k, v in trends.items()}

    def test_prefers_model_matching_the_data(self):
        x = np.linspace(-1, 1, 50)
        yobs = 2 * x**2
        yerr = np.full(50, 0.01)
        q = np.ones(50, dtype=bool)
        fits = self._fits(poly_1=x, poly_2=yobs)

        _, name = RegularizeOMC._select_best_model(
            yobs, yerr, q, fits, {"poly_1": 2, "poly_2": 3}
        )
        assert name == "poly_2"

    def test_parametric_model_preferred_when_criteria_disagree_on_gp(self):
        # matern32 fits slightly better, at a cost AIC accepts but BIC rejects
        npts = 100
        yobs = np.zeros(npts)
        yerr = np.ones(npts)
        q = np.ones(npts, dtype=bool)
        fits = self._fits(matern32=np.zeros(npts), poly_1=np.full(npts, np.sqrt(0.06)))

        _, name = RegularizeOMC._select_best_model(
            yobs, yerr, q, fits, {"matern32": 4.0, "poly_1": 2}
        )
        assert name == "poly_1"


class TestPolyModel:
    def test_matches_weighted_least_squares_with_prior(self):
        # Normal(0, 10) prior on each coefficient = one extra data row per coefficient
        rng = np.random.default_rng(3)
        xtime = np.sort(rng.uniform(100, 1500, 60))
        yerr = rng.uniform(0.5, 2.0, 60) * 1e-3
        yobs = 1e-3 * np.cos(xtime / 300) + yerr * rng.normal(size=60)
        q = np.ones(60, dtype=bool)
        q[::9] = False

        fit = RegularizeOMC._poly_model(xtime, yobs, yerr, q, 2)

        x = 2 * (xtime - xtime.min()) / (xtime.max() - xtime.min()) - 1
        X = np.vander(x, N=3, increasing=True)
        A = np.vstack([X[q] / yerr[q, None], np.eye(3) / 10])
        b = np.concatenate([yobs[q] / yerr[q], np.zeros(3)])
        C = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(fit["pred"], X @ C, rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(fit["trend"], (X @ C)[q], rtol=1e-9, atol=1e-15)


class TestGibbsMixture:
    def test_recovers_injected_mixture(self):
        rng = np.random.default_rng(5)
        n = 500
        broad = rng.random(n) > 0.9
        res = np.where(broad, rng.normal(0, 3.0, n), rng.normal(0, 0.3, n))

        mu, tau = RegularizeOMC._gibbs_mixture(res)
        scales = np.median(1 / np.sqrt(tau), axis=0)

        # compare with each component's realized scatter
        realized = [np.std(res[broad]), np.std(res[~broad])]
        np.testing.assert_allclose(scales, realized, rtol=0.15)
        assert abs(np.median(mu)) < 0.1
        assert np.all(tau[:, 0] <= tau[:, 1])
        np.testing.assert_array_equal(RegularizeOMC._gibbs_mixture(res)[0], mu)


@pytest.mark.slow
class TestSample:
    def test_deterministic(self):
        def loglike(theta):
            return -0.5 * np.sum((theta - 0.3) ** 2 / 0.01)

        def prior_transform(u):
            return 2 * u - 1

        a = RegularizeOMC._sample(loglike, prior_transform, 2, "toy")
        b = RegularizeOMC._sample(loglike, prior_transform, 2, "toy")
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.median(a, 0), 0.3, atol=0.03)


@pytest.mark.slow
class TestSelect:
    def test_recovers_ttv_and_flags_outliers(self):
        outliers = [12, 30]
        star, planets, lc, truth = _system(ntransit=48, outliers=outliers, missing=[20])
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        tt = planets_out[0].transit_times
        good = np.setdiff1d(np.arange(48), outliers)
        assert _rms(tt.ttime_mod[good] - truth[good]) < TIME_ERR
        assert np.all(tt.out_flag[outliers])
        assert np.mean(tt.out_flag) < 0.3
        assert np.isfinite(tt.ttime_mod[20]) and not tt.out_flag[20]
        measured = np.isfinite(tt.ttime_obs)
        assert np.all((tt.out_prob[measured] >= 0) & (tt.out_prob[measured] <= 1))
        assert np.isnan(tt.out_prob[20])


@pytest.mark.slow
class TestRegularizeOMCK00137:
    def test_smooth(self, k00137):
        star, planets, lc = k00137
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        for p, p_out in zip(planets, planets_out):
            tt = p_out.transit_times
            np.testing.assert_array_equal(tt.index, p.transit_times.index)
            assert np.all(np.isfinite(tt.ttime_mod))
            np.testing.assert_allclose(p_out.period, p.period, rtol=1e-4)
            # regularized times stay within an hour of the measured ones
            measured = np.isfinite(tt.ttime_obs) & ~tt.out_flag
            assert np.all(np.abs(tt.ttime_mod - tt.ttime_obs)[measured] < 1 / 24)
