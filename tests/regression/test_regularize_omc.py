"""Tests for alderaan.algorithms.regularize_omc.RegularizeOMC"""

import logging

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

    def test_low_quality_times_are_unmeasured(self):
        # low quality (FlagBadTransits) measured times are not fit: they come
        # back unmeasured, filled by the model
        star, planets, lc, truth = _system(outliers=[40, 90])
        tt = planets[0].transit_times
        quality = np.ones(len(tt.index), dtype=bool)
        quality[[40, 90]] = False
        planets[0].set_transit_times(
            TransitTimes(
                "Kepler",
                "SYN",
                tt.index,
                tt.ttime_obs,
                error_obs=tt.error_obs,
                ttime_mod=tt.ttime_mod,
                quality=quality,
            )
        )
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        tt = planets_out[0].transit_times
        assert np.all(np.isnan(tt.ttime_obs[[40, 90]]))
        assert np.all(np.isnan(tt.error_obs[[40, 90]]))
        assert not np.any(tt.out_flag)
        assert np.all(np.abs(tt.ttime_mod[[40, 90]] - truth[[40, 90]]) < 3 * TIME_ERR)


class TestSelectBestModel:
    """AIC/BIC model selection on fixed model predictions"""

    @staticmethod
    def _results(yobs, yerr, dofs, **preds):
        results = {}
        for k, pred in preds.items():
            _, _, lnlike = RegularizeOMC._calculate_outlier_probability(
                yobs, yerr, pred
            )
            good_mix = np.ones(len(yobs), dtype=bool)
            results[k] = {
                "pred": pred,
                "good_mix": good_mix,
                "lnlike": lnlike,
                "dof": dofs[k],
            }
        return results

    def test_prefers_model_matching_the_data(self):
        x = np.linspace(-1, 1, 50)
        yobs = 2 * x**2
        yerr = np.full(50, 0.01)
        results = self._results(
            yobs, yerr, {"poly_1": 2, "poly_2": 3}, poly_1=x, poly_2=yobs
        )
        assert RegularizeOMC._select_best_model(results, 50) == "poly_2"

    def test_parametric_model_preferred_when_criteria_disagree_on_gp(self):
        # matern32 fits slightly better, at a cost AIC accepts but BIC rejects
        npts = 100
        yobs = np.zeros(npts)
        yerr = np.ones(npts)
        results = self._results(
            yobs,
            yerr,
            {"matern32": 4.0, "poly_1": 2},
            matern32=np.zeros(npts),
            poly_1=np.full(npts, np.sqrt(0.06)),
        )
        assert RegularizeOMC._select_best_model(results, npts) == "poly_1"

    def test_rejecting_misfit_transits_is_not_rewarded(self):
        # 'wrong' misses 20 transits by 6 sigma and would reject them; scored on
        # its own kept transits it would beat 'right', which fits all of them
        rng = np.random.default_rng(8)
        npts = 200
        yerr = np.ones(npts)
        yobs = rng.normal(0, 1, npts)
        wrong = np.zeros(npts)
        wrong[:20] = 6.0
        results = self._results(
            yobs, yerr, {"right": 2, "wrong": 2}, right=np.zeros(npts), wrong=wrong
        )
        assert RegularizeOMC._select_best_model(results, npts) == "right"


def _contaminated_line(seed=1):
    """
    Linear OMC with 1 min white noise; 15% of transits late by 4-8 min, mostly
    inside the 5 sigma clip; returns (xtime, yobs, yerr, truth, bad)
    """
    rng = np.random.default_rng(seed)
    xtime = np.arange(300) * PERIOD
    truth = 3 / 1440 * (xtime / xtime.max() - 0.5)
    yerr = np.full(300, TIME_ERR)
    yobs = truth + rng.normal(0, TIME_ERR, 300)
    bad = rng.random(300) < 0.15
    yobs[bad] += rng.uniform(4, 8, bad.sum()) * TIME_ERR
    return xtime, yobs, yerr, truth, bad


def _poly_fits(xtime, yobs, yerr, good_clip, orders=(1, 2, 3)):
    results = {}
    for k in orders:
        results[f"poly_{k}"] = RegularizeOMC._fit_with_rejection(
            yobs,
            yerr,
            good_clip,
            lambda mask, k=k: RegularizeOMC._poly_model(xtime, yobs, yerr, mask, k),
        )
    return results


class TestFitWithRejection:
    def test_rejection_improves_fit(self):
        xtime, yobs, yerr, truth, bad = _contaminated_line()
        good_clip = RegularizeOMC._sigma_clip(yobs, yerr)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]
        plain = RegularizeOMC._poly_model(xtime, yobs, yerr, good_clip, 1)["pred"]

        assert np.sum(~r["good_mix"] & bad) > 0.9 * np.sum(bad)
        assert np.sum(~r["good_mix"] & ~bad) < 0.05 * np.sum(~bad)
        assert _rms(r["pred"] - truth) < 0.5 * _rms(plain - truth)

    def test_clean_data_unchanged(self):
        rng = np.random.default_rng(7)
        xtime = np.arange(300) * PERIOD
        yerr = np.full(300, TIME_ERR)
        yobs = rng.normal(0, TIME_ERR, 300)
        good_clip = RegularizeOMC._sigma_clip(yobs, yerr)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]
        plain = RegularizeOMC._poly_model(xtime, yobs, yerr, good_clip, 1)["pred"]

        np.testing.assert_array_equal(r["good_mix"], good_clip)
        np.testing.assert_array_equal(r["pred"], plain)

    def test_outliers_stay_a_minority(self):
        # 16 transits of a TTV a line can't fit (K01582.01's 8, twice): the
        # mixture would call 12 of them outliers
        xtime = np.arange(16) * 186.4
        yobs = np.tile([85.2, -5.3, -64.4, -65.9, -32.8, 31.5, 56.4, -4.7], 2) / 1440
        yerr = np.tile([4.2, 4.0, 4.0, 3.8, 3.7, 3.9, 4.1, 3.8], 2) / 1440
        good_clip = np.ones(16, dtype=bool)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]

        np.testing.assert_array_equal(r["good_mix"], good_clip)

    def test_leave_one_out_flags_single_outlier(self):
        # 12 transits on a line with 2 min noise; transit 5 is off by 20 sigma
        rng = np.random.default_rng(5)
        xtime = np.arange(12) * PERIOD
        yerr = np.full(12, 2 / 1440)
        yobs = rng.normal(0, 1, 12) * yerr
        yobs[5] += 20 * yerr[5]
        good_clip = np.ones(12, dtype=bool)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]

        np.testing.assert_array_equal(np.flatnonzero(~r["good_mix"]), [5])
        assert np.all(np.isnan(r["out_prob"]))

    def test_leave_one_out_false_flag_rate(self):
        # clean 12-transit planets are flagged at about alpha = 0.01
        rng = np.random.default_rng(6)
        xtime = np.arange(12) * PERIOD
        yerr = np.full(12, 2 / 1440)
        good_clip = np.ones(12, dtype=bool)
        nflag = 0
        for _ in range(500):
            yobs = rng.normal(0, 1, 12) * yerr
            r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]
            nflag += np.any(~r["good_mix"])
        assert nflag / 500 < 0.03


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
        assert fit["dof"] == 3

    def test_leverage_gives_leave_one_out_residuals(self):
        # r_i / (1 - S_ii) equals the residual of a refit without transit i
        rng = np.random.default_rng(8)
        xtime = np.sort(rng.uniform(100, 1500, 10))
        yerr = rng.uniform(0.5, 2.0, 10) * 1e-3
        yobs = 1e-3 * np.cos(xtime / 300) + yerr * rng.normal(size=10)
        q = np.ones(10, dtype=bool)

        fit = RegularizeOMC._poly_model(xtime, yobs, yerr, q, 2)
        loo = (yobs - fit["pred"]) / (1 - fit["leverage"])
        for i in range(10):
            # same [-1, 1] mapping: refit with transit i down-weighted to nothing
            err_i = yerr.copy()
            err_i[i] = 1e6
            refit = RegularizeOMC._poly_model(xtime, yobs, err_i, q, 2)
            np.testing.assert_allclose(loo[i], yobs[i] - refit["pred"][i], rtol=1e-6)
        np.testing.assert_allclose(np.sum(fit["leverage"]), 3, rtol=1e-3)


class TestSinModel:
    def test_recovers_line_plus_sinusoid(self):
        # a residual offset and slope (e.g. from a biased linear ephemeris) plus
        # a 400 d sinusoid; the LS period estimate is off by 0.5 / baseline
        rng = np.random.default_rng(9)
        xtime = np.arange(150) * PERIOD
        yerr = np.full(150, TIME_ERR)
        truth = (
            2 * TIME_ERR
            + 3 * TIME_ERR * xtime / xtime.max()
            + 4 * TIME_ERR * np.sin(2 * np.pi * xtime / 400 + 1.0)
        )
        yobs = truth + rng.normal(0, TIME_ERR, 150)
        q = np.ones(150, dtype=bool)
        ttv_period = 1 / (1 / 400 + 0.5 / np.ptp(xtime))

        fit = RegularizeOMC._sin_model(xtime, yobs, yerr, q, ttv_period)
        assert _rms(fit["pred"] - truth) < 0.25 * TIME_ERR


def _mixed_error_omc():
    """
    White-noise OMC alternating precise (1 min) and noisy (4 min) errors; points
    50 (precise) and 51 (noisy) are both offset by +8 min
    """
    rng = np.random.default_rng(3)
    yerr = np.where(np.arange(200) % 2 == 0, 1.0, 4.0) / 1440
    yobs = rng.normal(0, yerr)
    yobs[[50, 51]] = 8 / 1440
    return yobs, yerr


class TestOutlierScreen:
    def test_judged_against_own_errors(self):
        # 8 min is 8 sigma for the precise point, 2 sigma for the noisy one
        yobs, yerr = _mixed_error_omc()
        good_clip = RegularizeOMC._sigma_clip(yobs, yerr)
        np.testing.assert_array_equal(np.flatnonzero(~good_clip), [50])

    def test_few_transits_not_clipped(self):
        # 9 transits ending in a 140 min ramp (K00564.02): every point is TTV
        yobs = np.array([-1.6, -6.0, -29.3, -6.5, -3.1, 5.1, 34.3, 79.3, 139.1]) / 1440
        yerr = np.array([4.2, 3.6, 6.7, 4.1, 3.9, 4.1, 4.1, 4.0, 4.0]) / 1440
        assert np.all(RegularizeOMC._sigma_clip(yobs, yerr))

    def test_equal_errors_depend_only_on_scatter(self):
        yobs, _ = _mixed_error_omc()
        np.testing.assert_array_equal(
            RegularizeOMC._sigma_clip(yobs, np.full(200, 1 / 1440)),
            RegularizeOMC._sigma_clip(yobs, np.full(200, 7 / 1440)),
        )


class TestOutlierProbability:
    def test_judged_against_own_errors(self):
        yobs, yerr = _mixed_error_omc()
        out_prob, out_flag, _ = RegularizeOMC._calculate_outlier_probability(
            yobs, yerr, np.zeros_like(yobs)
        )
        assert out_flag[50]
        assert out_prob[50] > out_prob[51]

    def test_clean_residuals_with_excess_scatter_not_flagged(self):
        # Gaussian residuals 1.5x wider than the errors: jitter, not outliers
        rng = np.random.default_rng(4)
        yerr = np.full(400, 1 / 1440)
        yobs = rng.normal(0, 1.5 * yerr)
        _, out_flag, _ = RegularizeOMC._calculate_outlier_probability(
            yobs, yerr, np.zeros_like(yobs)
        )
        assert not np.any(out_flag)

    def test_recovers_injected_outliers(self):
        rng = np.random.default_rng(5)
        yerr = np.full(400, 1 / 1440)
        yobs = rng.normal(0, yerr)
        bad = rng.choice(400, 20, replace=False)
        yobs[bad] += rng.choice([-1, 1], 20) * rng.uniform(8, 20, 20) / 1440
        _, out_flag, _ = RegularizeOMC._calculate_outlier_probability(
            yobs, yerr, np.zeros_like(yobs)
        )
        np.testing.assert_array_equal(np.flatnonzero(out_flag), np.sort(bad))


@pytest.mark.slow
class TestSample:
    def test_deterministic(self):
        def loglike(theta):
            return -0.5 * np.sum((theta - 0.3) ** 2 / 0.01)

        def prior_transform(u):
            return 2 * u - 1

        a = RegularizeOMC._sample(loglike, prior_transform, 2)
        b = RegularizeOMC._sample(loglike, prior_transform, 2)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(np.median(a, 0), 0.3, atol=0.03)


def _system_with_omc(omc, err):
    """_system with the given OMC values and errors (days) as the measured times"""
    star, planets, lc, _ = _system(ntransit=len(omc))
    index = np.arange(len(omc))
    planets[0].set_transit_times(
        TransitTimes(
            "Kepler", "SYN", index, EPOCH + PERIOD * index + omc, error_obs=err
        )
    )
    return star, planets, lc


class TestIndependentTimesPreferred:
    def test_times_inconsistent_with_model(self):
        # 6 transits, 30 min TTV, 3 min errors, against a flat line
        yobs = 30 * np.sin(np.arange(6)) / 1440
        yerr = np.full(6, 3 / 1440)
        switch, _ = RegularizeOMC._independent_times_preferred(
            yobs, yerr, np.zeros(6), 2
        )
        assert switch

    def test_times_consistent_with_model(self):
        # residuals of exactly 1 sigma: chi2 = 6 on 4 dof, p = 0.2
        yerr = np.full(6, 3 / 1440)
        yobs = yerr * np.array([1, -1, 1, -1, 1, -1])
        switch, _ = RegularizeOMC._independent_times_preferred(
            yobs, yerr, np.zeros(6), 2
        )
        assert not switch


class TestSelectFewTransits:
    def test_ttv_adopts_measured_times(self):
        # 5 transits of a TTV a line can't fit (K01783.02)
        omc = np.array([44.2, -54.8, -23.2, 40.0, 8.2]) / 1440
        err = np.array([3.8, 3.9, 3.9, 4.4, 7.1]) / 1440
        star, planets, lc = _system_with_omc(omc, err)
        ttime_obs = planets[0].transit_times.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        tt = planets_out[0].transit_times
        np.testing.assert_allclose(tt.ttime_mod, ttime_obs, rtol=0, atol=1e-9)
        assert not np.any(tt.out_flag)
        assert np.all(np.isnan(tt.out_prob))
        np.testing.assert_allclose(planets_out[0].period, PERIOD, rtol=1e-3)

    @pytest.mark.parametrize("ntransit", [15, 16])
    def test_outlier_gated_below_16_transits(self, ntransit):
        # one 20 sigma outlier on a line: below 16 transits it fails the chi2
        # test (measured times adopted, flag cleared); at 16 the select model
        # is kept and the outlier flagged
        rng = np.random.default_rng(5)
        err = np.full(ntransit, 3 / 1440)
        omc = rng.normal(0, err)
        omc[7] += 20 * err[7]
        star, planets, lc = _system_with_omc(omc, err)
        ttime_obs = planets[0].transit_times.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        tt = planets_out[0].transit_times
        if ntransit < 16:
            np.testing.assert_allclose(tt.ttime_mod, ttime_obs, rtol=0, atol=1e-9)
            assert not np.any(tt.out_flag)
        else:
            assert tt.out_flag[7]
            assert np.sum(tt.out_flag) == 1

    def test_fewer_than_3_measured_times_raise(self):
        star, planets, lc, _ = _system(ntransit=5, missing=[0, 1, 2])
        with pytest.raises(ValueError, match="needs at least 3"):
            RegularizeOMC(star, planets, lc).perform(method="select")


@pytest.mark.slow
class TestSelect:
    def test_ttv_ramp_fit_not_flagged(self, caplog):
        # 9 transits ending in a 140 min ramp (K00564.02): a curved model fits
        # every point; a line would need to flag the ramp's end
        omc = np.array([-1.6, -6.0, -29.3, -6.5, -3.1, 5.1, 34.3, 79.3, 139.1]) / 1440
        err = np.array([4.2, 3.6, 6.7, 4.1, 3.9, 4.1, 4.1, 4.0, 4.0]) / 1440
        star, planets, lc = _system_with_omc(omc, err)
        with caplog.at_level(logging.INFO, logger="alderaan.algorithms.regularize_omc"):
            planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        assert not np.any(planets_out[0].transit_times.out_flag)
        assert "poly_1 OMC model;" not in caplog.text

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

    @staticmethod
    def _contaminated_sinusoid(seed, amp):
        """
        150 transits, 2 min errors, an amp-sigma 400 d TTV, and 30 transits offset
        by 4-8 sigma; returns (star, planets, lc, linear, ttv, err, xtime, yobs)
        """
        rng = np.random.default_rng(seed)
        n = 150
        index = np.arange(n)
        linear = EPOCH + PERIOD * index
        err = np.full(n, 2 / 1440)
        ttv = amp * err * np.sin(2 * np.pi * linear / 400 + rng.uniform(0, 6.28))
        omc = ttv + rng.normal(0, 1, n) * err
        bad = rng.choice(n, 30, replace=False)
        omc[bad] += rng.choice([-1, 1], 30) * rng.uniform(4, 8, 30) * err[bad]

        star, planets, lc, _ = _system(ntransit=n)
        planets[0].set_transit_times(
            TransitTimes("Kepler", "SYN", index, linear + omc, error_obs=err)
        )
        tt = planets[0].transit_times
        xtime = tt.linear_ephemeris()
        return star, planets, lc, linear, ttv, err, xtime, tt.ttime_obs - xtime

    @staticmethod
    def _first_search_significant(xtime, yobs, err):
        good_clip = RegularizeOMC._sigma_clip(yobs, err)
        _, fap = RegularizeOMC._lomb_scargle_peak(xtime, yobs, good_clip)
        return fap < 0.03

    def test_periodic_signal_from_outliers_is_rejected(self, caplog):
        # white noise: the outliers alone give a significant first-search peak,
        # and the sinusoid is selected unless the second search rejects it
        star, planets, lc, _, _, err, xtime, yobs = self._contaminated_sinusoid(15, 0)
        assert self._first_search_significant(xtime, yobs, err)

        with caplog.at_level(logging.INFO, logger="alderaan.algorithms.regularize_omc"):
            RegularizeOMC(star, planets, lc).perform(method="select")
        assert "sinusoid rejected" in caplog.text
        assert "sinusoid OMC model;" not in caplog.text

    def test_periodic_signal_hidden_by_outliers_is_recovered(self, caplog):
        # 0.8 sigma TTV: found only once the outliers are rejected
        star, planets, lc, linear, ttv, err, xtime, yobs = self._contaminated_sinusoid(
            2, 0.8
        )
        assert not self._first_search_significant(xtime, yobs, err)

        with caplog.at_level(logging.INFO, logger="alderaan.algorithms.regularize_omc"):
            planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        assert "sinusoid added" in caplog.text
        assert "sinusoid OMC model;" in caplog.text
        tt = planets_out[0].transit_times
        assert _rms((tt.ttime_mod - linear - ttv) / err) < 0.25


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
