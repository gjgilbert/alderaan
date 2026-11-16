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
    def _traces(**trends):
        return {k: {"trend": v[None, :], "pred": v[None, :]} for k, v in trends.items()}

    def test_prefers_model_matching_the_data(self):
        x = np.linspace(-1, 1, 50)
        yobs = 2 * x**2
        yerr = np.full(50, 0.01)
        q = np.ones(50, dtype=bool)
        traces = self._traces(poly_1=x, poly_2=yobs)

        _, name = RegularizeOMC._select_best_model(
            yobs, yerr, q, traces, {"poly_1": 2, "poly_2": 3}
        )
        assert name == "poly_2"

    def test_parametric_model_preferred_when_criteria_disagree_on_gp(self):
        # matern32 fits slightly better, at a cost AIC accepts but BIC rejects
        npts = 100
        yobs = np.zeros(npts)
        yerr = np.ones(npts)
        q = np.ones(npts, dtype=bool)
        traces = self._traces(
            matern32=np.zeros(npts), poly_1=np.full(npts, np.sqrt(0.06))
        )

        _, name = RegularizeOMC._select_best_model(
            yobs, yerr, q, traces, {"matern32": 4.0, "poly_1": 2}
        )
        assert name == "poly_1"


@pytest.fixture
def fast_sampling(monkeypatch):
    """Shorten NUTS runs; selection and outlier logic are unchanged"""
    from alderaan.algorithms import regularize_omc

    sample = regularize_omc.pm.sample

    def _sample(*args, **kwargs):
        kwargs.update(tune=500, draws=500)
        return sample(*args, **kwargs)

    monkeypatch.setattr(regularize_omc.pm, "sample", _sample)


@pytest.mark.slow
class TestSelect:
    def test_recovers_ttv_and_flags_outliers(self, fast_sampling):
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
