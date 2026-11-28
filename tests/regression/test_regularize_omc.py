"""Tests for alderaan.algorithms.regularize_omc.RegularizeOMC"""

import numpy as np
import pytest
from astropy.stats import mad_std

from alderaan.algorithms import RegularizeOMC
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.errors import UnsupportedTargetError
from alderaan.transit_model import exposure_offsets, transit_flux

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
    ttime_err = np.full(ntransit, TIME_ERR)
    for i in outliers:
        ttime_obs[i] += 60 / 1440
    for i in missing:
        ttime_obs[i] = np.nan
        ttime_err[i] = np.nan

    p = Planet("Kepler", "SYN", PERIOD, EPOCH, 0.03, 0.0, 0.15)
    # unmeasured transits start on the linear ephemeris
    ttime_mod = np.where(np.isfinite(ttime_obs), ttime_obs, linear)
    p.set_transit_times(
        TransitTimes(
            "Kepler", "SYN", index, ttime_obs, ttime_err=ttime_err, ttime_mod=ttime_mod
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
        np.testing.assert_array_equal(np.flatnonzero(~tt.kept_ttv), [40, 90])
        assert np.all(np.abs(tt.ttime_mod[[40, 90]] - truth[[40, 90]]) < 3 * TIME_ERR)

    def test_fills_missing_epochs(self):
        star, planets, lc, truth = _system(missing=[0, 60, 61])
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        tt = planets_out[0].transit_times
        np.testing.assert_array_equal(tt.index, np.arange(140))
        assert np.all(np.isnan(tt.ttime_obs[[0, 60, 61]]))
        assert np.all(tt.kept_ttv)
        assert np.all(
            np.abs(tt.ttime_mod[[0, 60, 61]] - truth[[0, 60, 61]]) < 3 * TIME_ERR
        )

    def test_poor_photometry_times_are_unmeasured(self):
        # measured times with poor photometry (FlagBadTransits) are not fit: they
        # come back unmeasured, filled by the model
        star, planets, lc, truth = _system(outliers=[40, 90])
        tt = planets[0].transit_times
        noise = np.ones(len(tt.index), dtype=bool)
        noise[[40, 90]] = False
        planets[0].set_transit_times(
            TransitTimes(
                "Kepler",
                "SYN",
                tt.index,
                tt.ttime_obs,
                ttime_err=tt.ttime_err,
                ttime_mod=tt.ttime_mod,
                phot_noise=noise,
            )
        )
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="smooth")

        tt = planets_out[0].transit_times
        assert np.all(np.isnan(tt.ttime_obs[[40, 90]]))
        assert np.all(np.isnan(tt.ttime_err[[40, 90]]))
        assert np.all(tt.kept_ttv)
        assert np.all(np.abs(tt.ttime_mod[[40, 90]] - truth[[40, 90]]) < 3 * TIME_ERR)


class TestCompare:
    """AIC/BIC model selection on fixed model predictions"""

    @staticmethod
    def _results(yobs, yerr, dofs, **preds):
        results = {}
        for k, pred in preds.items():
            _, lnlike = RegularizeOMC._calculate_outlier_probability(yobs, yerr, pred)
            inlier = np.ones(len(yobs), dtype=bool)
            results[k] = {
                "pred": pred,
                "inlier": inlier,
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
        assert RegularizeOMC._compare(results, 50) == "poly_2"

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
        assert RegularizeOMC._compare(results, npts) == "poly_1"

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
        assert RegularizeOMC._compare(results, npts) == "right"


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
    return {
        f"poly_{k}": RegularizeOMC._robust_fit(
            f"poly_{k}", xtime, yobs, yerr, good_clip
        )
        for k in orders
    }


class TestRobustFit:
    def test_rejection_improves_fit(self):
        xtime, yobs, yerr, truth, bad = _contaminated_line()
        good_clip = RegularizeOMC._sigma_clip(yobs, yerr)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]
        plain = RegularizeOMC._poly_model(xtime, yobs, yerr, good_clip, 1)["pred"]

        assert np.sum(~r["inlier"] & bad) > 0.9 * np.sum(bad)
        assert np.sum(~r["inlier"] & ~bad) < 0.05 * np.sum(~bad)
        assert _rms(r["pred"] - truth) < 0.5 * _rms(plain - truth)

    def test_clean_data_unchanged(self):
        rng = np.random.default_rng(7)
        xtime = np.arange(300) * PERIOD
        yerr = np.full(300, TIME_ERR)
        yobs = rng.normal(0, TIME_ERR, 300)
        good_clip = RegularizeOMC._sigma_clip(yobs, yerr)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]
        plain = RegularizeOMC._poly_model(xtime, yobs, yerr, good_clip, 1)["pred"]

        np.testing.assert_array_equal(r["inlier"], good_clip)
        np.testing.assert_array_equal(r["pred"], plain)

    def test_outliers_stay_a_minority(self):
        # 16 transits of a TTV a line can't fit (K01582.01's 8, twice): the
        # mixture would call 12 of them outliers
        xtime = np.arange(16) * 186.4
        yobs = np.tile([85.2, -5.3, -64.4, -65.9, -32.8, 31.5, 56.4, -4.7], 2) / 1440
        yerr = np.tile([4.2, 4.0, 4.0, 3.8, 3.7, 3.9, 4.1, 3.8], 2) / 1440
        good_clip = np.ones(16, dtype=bool)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]

        np.testing.assert_array_equal(r["inlier"], good_clip)

    def test_leave_one_out_flags_single_outlier(self):
        # 12 transits on a line with 2 min noise; transit 5 is off by 20 sigma
        rng = np.random.default_rng(5)
        xtime = np.arange(12) * PERIOD
        yerr = np.full(12, 2 / 1440)
        yobs = rng.normal(0, 1, 12) * yerr
        yobs[5] += 20 * yerr[5]
        good_clip = np.ones(12, dtype=bool)
        r = _poly_fits(xtime, yobs, yerr, good_clip, orders=(1,))["poly_1"]

        np.testing.assert_array_equal(np.flatnonzero(~r["inlier"]), [5])

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
            nflag += np.any(~r["inlier"])
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


class TestLombScarglePeak:
    def test_short_baseline_after_outlier_cut_is_no_signal(self):
        # 8 transits; outliers at both ends leave a 5-period baseline, too short
        # for any searched period
        rng = np.random.default_rng(12)
        xtime = np.arange(8) * PERIOD
        yobs = rng.normal(0, TIME_ERR, 8)
        yobs[[0, 7]] += 100 * TIME_ERR
        mask = np.ones(8, dtype=bool)
        assert RegularizeOMC._lomb_scargle_peak(xtime, yobs, mask) == (None, 1.0)


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
        # 8 min is 8 sigma for the precise point, 2 sigma for the noisy one
        out_flag, _ = RegularizeOMC._calculate_outlier_probability(
            yobs, yerr, np.zeros_like(yobs)
        )
        assert out_flag[50]
        assert not out_flag[51]

    def test_clean_residuals_with_excess_scatter_not_flagged(self):
        # Gaussian residuals 1.5x wider than the errors: jitter, not outliers
        rng = np.random.default_rng(4)
        yerr = np.full(400, 1 / 1440)
        yobs = rng.normal(0, 1.5 * yerr)
        out_flag, _ = RegularizeOMC._calculate_outlier_probability(
            yobs, yerr, np.zeros_like(yobs)
        )
        assert not np.any(out_flag)

    def test_recovers_injected_outliers(self):
        rng = np.random.default_rng(5)
        yerr = np.full(400, 1 / 1440)
        yobs = rng.normal(0, yerr)
        bad = rng.choice(400, 20, replace=False)
        yobs[bad] += rng.choice([-1, 1], 20) * rng.uniform(8, 20, 20) / 1440
        out_flag, _ = RegularizeOMC._calculate_outlier_probability(
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
    """
    _system with the given OMC values and errors (days) as the measured times;
    transits with NaN omc are unmeasured
    """
    star, planets, lc, _ = _system(ntransit=len(omc))
    index = np.arange(len(omc))
    linear = EPOCH + PERIOD * index
    planets[0].set_transit_times(
        TransitTimes(
            "Kepler",
            "SYN",
            index,
            linear + omc,
            ttime_err=err,
            ttime_mod=np.where(np.isfinite(omc), linear + omc, linear),
        )
    )
    return star, planets, lc


def _transit_system(omc, err, dip_omc=None, seed=2):
    """
    _system_with_omc with a long cadence litecurve of 1000 ppm, 3.6 h transits
    at EPOCH + PERIOD * index + dip_omc (default omc) in 100 ppm white noise;
    transits with NaN omc are unmeasured and have no photometry
    """
    star, planets, _ = _system_with_omc(omc, err)
    p = planets[0]
    p.set_transit_shape(0.03, 0.3, 0.15)
    p.set_static_period_and_epoch()

    dip_omc = omc if dip_omc is None else dip_omc
    exptime = EXPTIME["long cadence"]
    time, flux = [], []
    for tlin, dt in zip(EPOCH + PERIOD * np.arange(len(omc)), dip_omc):
        if np.isnan(dt):
            continue
        t = np.arange(tlin - 1.0, tlin + 1.0, exptime)
        time.append(t)
        flux.append(
            transit_flux(
                t,
                tlin + dt,
                PERIOD,
                0.03,
                0.3,
                0.15,
                star.u1,
                star.u2,
                exposure_offsets(exptime, 7),
            )
        )
    time, flux = np.concatenate(time), np.concatenate(flux)
    flux += np.random.default_rng(seed).normal(0, 1e-4, len(time))
    ncad = len(time)
    lc = LiteCurve(
        "Kepler",
        "SYN",
        time,
        flux,
        np.full(ncad, 1e-4),
        np.arange(ncad),
        np.ones(ncad, dtype=int),
        np.full(ncad, "long cadence"),
    )
    return star, planets, lc


class TestChi2Pvalue:
    def test_times_inconsistent_with_model(self):
        # 6 transits, 30 min TTV, 3 min errors, against a flat line
        yobs = 30 * np.sin(np.arange(6)) / 1440
        yerr = np.full(6, 3 / 1440)
        p = RegularizeOMC._chi2_pvalue(yobs, yerr, np.zeros(6), np.ones(6, bool), 2)
        assert p < 0.01

    def test_times_consistent_with_model(self):
        # residuals of exactly 1 sigma: chi2 = 6 on 4 dof, p = 0.2
        yerr = np.full(6, 3 / 1440)
        yobs = yerr * np.array([1, -1, 1, -1, 1, -1])
        p = RegularizeOMC._chi2_pvalue(yobs, yerr, np.zeros(6), np.ones(6, bool), 2)
        assert p >= 0.01


class TestLassoPolyModel:
    def test_recovers_quartic_ttv(self):
        # 60 transits, 2 min errors, a quartic OMC poly_3 cannot follow
        x = np.linspace(-1, 1, 60)
        truth = 30 * (x**4 - x**2) / 1440
        yerr = np.full(60, 2 / 1440)
        yobs = truth + np.random.default_rng(4).normal(0, 2 / 1440, 60)
        fit = RegularizeOMC._lasso_poly_model(x, yobs, yerr, np.ones(60, bool), 5)
        assert _rms(fit["pred"] - truth) < 1 / 1440
        assert fit["dof"] <= 6

    def test_white_noise_shrinks_to_zero(self):
        yerr = np.full(60, 2 / 1440)
        yobs = np.random.default_rng(4).normal(0, 2 / 1440, 60)
        x = np.arange(60.0)
        fit = RegularizeOMC._lasso_poly_model(x, yobs, yerr, np.ones(60, bool), 5)
        assert fit["dof"] <= 1


class TestLassoUpgrade:
    @staticmethod
    def _upgrade(omc, err, order):
        """_lasso_upgrade of a poly_<order> fit to every transit"""
        star, planets, lc = _system_with_omc(omc, err)
        tlin = planets[0].transit_times.linear_ephemeris()
        mask = np.ones(len(omc), bool)
        fit = {**RegularizeOMC._poly_model(tlin, omc, err, mask, order), "inlier": mask}
        alg = RegularizeOMC(star, planets, lc)
        return alg._lasso_upgrade(tlin, omc, err, f"poly_{order}", fit)

    def test_misfit_smooth_ttv_upgraded(self):
        # 60 transits, 2 min errors, a 15 min deep quartic OMC: poly_2 fails chi2
        x = np.linspace(-1, 1, 60)
        omc = (60 * (x**4 - x**2) + np.random.default_rng(5).normal(0, 2, 60)) / 1440
        err = np.full(60, 2 / 1440)
        label, fit = self._upgrade(omc, err, 2)
        assert label in ("lasso_4", "lasso_5")
        assert (
            RegularizeOMC._chi2_pvalue(omc, err, fit["pred"], fit["inlier"], 6) > 0.001
        )

    def test_model_passing_chi2_kept(self):
        omc = np.random.default_rng(5).normal(0, 2, 60) / 1440
        label, _ = self._upgrade(omc, np.full(60, 2 / 1440), 1)
        assert label == "poly_1"

    def test_excess_white_scatter_kept(self):
        # scatter 3x the errors fails chi2, but no shape explains it
        omc = np.random.default_rng(5).normal(0, 6, 60) / 1440
        label, _ = self._upgrade(omc, np.full(60, 2 / 1440), 1)
        assert label == "poly_1"


class TestSelectFewTransits:
    @pytest.mark.slow
    def test_ttv_adopts_measured_times(self):
        # 5 transits of a TTV a line can't fit (K01783.02)
        omc = np.array([44.2, -54.8, -23.2, 40.0, 8.2]) / 1440
        err = np.array([3.8, 3.9, 3.9, 4.4, 7.1]) / 1440
        star, planets, lc = _system_with_omc(omc, err)
        ttime_obs = planets[0].transit_times.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        tt = planets_out[0].transit_times
        np.testing.assert_allclose(tt.ttime_mod, ttime_obs, rtol=0, atol=1e-9)
        assert np.all(tt.kept_ttv)
        np.testing.assert_allclose(planets_out[0].period, PERIOD, rtol=1e-3)

    @pytest.mark.parametrize("ntransit", [pytest.param(15, marks=pytest.mark.slow), 16])
    def test_outlier_gated_below_16_transits(self, ntransit):
        # one 20 sigma outlier on a line, its transit really shifted: below 16
        # transits it fails the chi2 test and the light curve supports it
        # (measured times adopted, flag cleared); at 16 the select model is
        # kept and the outlier flagged
        rng = np.random.default_rng(5)
        err = np.full(ntransit, 3 / 1440)
        omc = rng.normal(0, err)
        omc[7] += 20 * err[7]
        star, planets, lc = _transit_system(omc, err)
        ttime_obs = planets[0].transit_times.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        tt = planets_out[0].transit_times
        if ntransit < 16:
            np.testing.assert_allclose(tt.ttime_mod, ttime_obs, rtol=0, atol=1e-9)
            assert np.all(tt.kept_ttv)
        else:
            assert not tt.kept_ttv[7]
            assert np.sum(~tt.kept_ttv) == 1

    @pytest.mark.parametrize("ntransit", [15, 16])
    def test_matern32_needs_16_transits(self, ntransit):
        # 30 min TTV, 1 min errors: the GP is offered only from 16 transits
        err = np.full(ntransit, 1 / 1440)
        omc = 30 * np.sin(np.arange(ntransit)) / 1440
        star, planets, lc = _system_with_omc(omc, err)
        tt = planets[0].transit_times
        y = tt.ttime_obs - tt.linear_ephemeris()
        m = np.ones(ntransit, dtype=bool)
        models = RegularizeOMC._candidates(y, err, m, m, None)
        assert ("matern32" in models) == (ntransit >= 16)

    def test_max_polyorder(self):
        # 10 transits: poly_3 needs 16 unless the order is set explicitly
        y = np.zeros(10)
        err = np.ones(10)
        m = np.ones(10, dtype=bool)
        assert "poly_3" not in RegularizeOMC._candidates(y, err, m, m, None)
        assert "poly_3" in RegularizeOMC._candidates(y, err, m, m, None, 3)

    @pytest.mark.parametrize("scatter, wild", [(10, True), (50, False)])
    def test_wild_score(self, scatter, wild):
        # 12 times alternating +-scatter min about zero, 10 min errors, one at
        # 150 min: wild among times that scatter like their errors, not among
        # times that all scatter 5x their errors (shared excess scatter)
        from scipy import stats

        x = np.arange(12.0)
        err = np.full(12, 10 / 1440)
        y = scatter * (-1.0) ** np.arange(12) / 1440
        y[5] = 150 / 1440
        inlier = np.arange(12) != 5
        score = RegularizeOMC._wild_score(x, y, err, inlier, "poly_1", 5)
        assert (score >= stats.norm.isf(0.0027 / 24)) == wild

    @pytest.mark.slow
    def test_ttv_ramp(self):
        # 9 transits ending in a 140 min ramp (K00564.02): select prefers the
        # parabola over a line that flags the ramp's end; the parabola still
        # fails the chi2 test and the light curve supports the ramp, so the
        # measured times are adopted
        omc = np.array([-1.6, -6.0, -29.3, -6.5, -3.1, 5.1, 34.3, 79.3, 139.1]) / 1440
        err = np.array([4.2, 3.6, 6.7, 4.1, 3.9, 4.1, 4.1, 4.0, 4.0]) / 1440
        star, planets, lc = _transit_system(omc, err)
        tt = planets[0].transit_times
        x = tt.linear_ephemeris()
        y = tt.ttime_obs - x
        m = np.ones(9, dtype=bool)
        fits = {
            k: RegularizeOMC._robust_fit(k, x, y, err, m)
            for k in RegularizeOMC._candidates(y, err, m, m, None)
        }
        assert RegularizeOMC._compare(fits, 9) == "poly_2"

        ttime_obs = tt.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        tt_out = planets_out[0].transit_times
        np.testing.assert_allclose(tt_out.ttime_mod, ttime_obs, rtol=0, atol=1e-9)
        assert np.all(tt_out.kept_ttv)

    def test_fewer_than_3_measured_times_get_linear_ephemeris(self):
        star, planets, lc, _ = _system(ntransit=5, missing=[0, 1, 2])
        tt = planets[0].transit_times
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        tt_out = planets_out[0].transit_times
        assert tt_out.model_name == "linear"
        assert np.all(np.isnan(tt_out.ttime_obs))
        np.testing.assert_allclose(tt_out.ttime_mod, tt.linear_ephemeris())
        assert tt_out.ttv_scatter == np.median(tt.ttime_err[3:])

    def test_no_measured_times_raise(self):
        star, planets, lc, _ = _system(ntransit=5, missing=[0, 1, 2, 3, 4])
        with pytest.raises(UnsupportedTargetError, match="0 measured"):
            RegularizeOMC(star, planets, lc).perform(method="select")

    def test_linear_planet_passes_through(self):
        # low per-transit SNR (QuickFitTransitTimes): no measured times, kept as given
        star, planets, lc = _system_with_omc(np.full(5, np.nan), np.full(5, np.nan))
        tt = planets[0].transit_times
        linear = TransitTimes(
            "Kepler",
            "SYN",
            tt.index,
            tt.ttime_obs,
            ttime_mod=tt.ttime_mod,
            model_name="linear",
            model_dof=2.0,
            ttv_scatter=0.002,
        )
        planets[0].set_transit_times(linear)
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        out = planets_out[0].transit_times
        assert (out.model_name, out.model_dof, out.ttv_scatter) == (
            "linear",
            2.0,
            0.002,
        )
        np.testing.assert_array_equal(out.ttime_mod, linear.ttime_mod)
        assert np.all(np.isnan(out.ttime_obs))


@pytest.mark.slow
class TestSelectMatern32Gate:
    @staticmethod
    def _perform(omc, err):
        star, planets, lc = _system_with_omc(omc, err)
        ttime_obs = planets[0].transit_times.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        return planets_out[0].transit_times, ttime_obs

    def test_interpolating_gp_adopts_measured_times(self):
        # 20 transits of a random-walk TTV (10 min steps, 1 min errors) and one
        # 60 min bad time: matern32 is selected with dof/N > 0.5, so the
        # measured times are adopted; the bad time stays flagged, filled by a
        # GP on the kept times
        x = np.arange(20)
        omc = np.cumsum(np.random.default_rng(3).normal(0, 10, 20))
        omc -= np.polyval(np.polyfit(x, omc, 1), x)
        omc[9] += 60
        err = np.full(20, 1 / 1440)
        star, planets, lc = _system_with_omc(omc / 1440, err)
        tt_in = planets[0].transit_times
        tlin = tt_in.linear_ephemeris()
        ttime_obs = tt_in.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        tt = planets_out[0].transit_times

        assert tt.model_name == "independent"
        assert np.flatnonzero(~tt.kept_ttv).tolist() == [9]
        good = tt.kept_ttv
        np.testing.assert_allclose(
            tt.ttime_mod[good], ttime_obs[good], rtol=0, atol=1e-9
        )
        gp = RegularizeOMC._matern32_model(tlin, ttime_obs - tlin, err, good)
        np.testing.assert_allclose(
            tt.ttime_mod[9], tlin[9] + gp["pred"][9], rtol=0, atol=1e-9
        )

    def test_gp_dropped_when_errors_exceed_kept_scatter(self):
        # K04548.01: small-error outliers inflate mad_std enough to offer
        # matern32, which is selected; the kept times scatter less than their
        # errors, so a parametric model replaces it
        # fmt: off
        measured = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23,
        ]
        t_meas = [
            132.488562, 193.448646, 254.694434, 315.928451, 376.916916, 438.061402,
            499.063886, 560.073927, 621.103357, 743.383847, 865.495896, 926.555084,
            987.729482, 1048.717790, 1109.828307, 1170.799680, 1232.722790,
            1354.044219, 1476.312132, 1537.402198,
        ]
        e_meas = [
            0.032643, 0.036229, 0.036538, 0.056392, 0.049341, 0.025271, 0.032927,
            0.034442, 0.107103, 0.024333, 0.024221, 0.054674, 0.029041, 0.216283,
            0.050571, 0.024207, 0.024459, 0.283817, 0.094009, 0.122873,
        ]
        # fmt: on
        star, planets, lc, _ = _system(ntransit=24)
        index = np.arange(24)
        ttime_obs = np.full(24, np.nan)
        ttime_err = np.full(24, np.nan)
        ttime_obs[measured] = t_meas
        ttime_err[measured] = e_meas
        # modeled times of the unmeasured transits set the linear ephemeris
        ttime_mod = ttime_obs.copy()
        ttime_mod[[9, 11, 19, 21]] = [682.185284, 804.452014, 1292.91796, 1415.18586]
        planets[0] = Planet("Kepler", "SYN", 61.08, 132.49, 0.03, 0.0, 0.15)
        planets[0].set_transit_times(
            TransitTimes(
                "Kepler",
                "SYN",
                index,
                ttime_obs,
                ttime_err=ttime_err,
                ttime_mod=ttime_mod,
            )
        )
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        assert planets_out[0].transit_times.model_name not in (
            "matern32",
            "independent",
        )

    def test_regularizing_gp_kept(self):
        # 40 transits of a smooth 20 min bump, 2 min errors: matern32 is
        # selected with dof/N <= 0.5 and kept
        x = np.arange(40)
        omc = 20 * np.exp(-0.5 * ((x - 25) / 5) ** 2)
        omc += np.random.default_rng(1).normal(0, 2, 40)
        tt, ttime_obs = self._perform(omc / 1440, np.full(40, 2 / 1440))

        assert tt.model_name == "matern32"
        assert 0 < tt.model_dof <= 0.5 * len(tt.index)
        assert np.max(np.abs(tt.ttime_mod - ttime_obs)) > 1 / 1440


class TestSelectPhotometricOutliers:
    # 12 transits on a line, 3 min errors, transit 5 measured 120 min late
    ERR = np.full(12, 3 / 1440)
    OMC = np.random.default_rng(4).normal(0, 3, 12) / 1440

    def _perform(self, dip_shift):
        omc = self.OMC.copy()
        omc[5] += 120 / 1440
        dip_omc = self.OMC.copy()
        dip_omc[5] += dip_shift / 1440
        star, planets, lc = _transit_system(omc, self.ERR, dip_omc=dip_omc)
        ttime_obs = planets[0].transit_times.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        return planets_out[0].transit_times, ttime_obs

    def test_time_without_dip_is_flagged(self):
        # the light curve has the transit on the line: the time is flagged and
        # the line restored
        tt, _ = self._perform(dip_shift=0)
        assert np.flatnonzero(~tt.kept_ttv).tolist() == [5]
        truth = EPOCH + PERIOD * 5 + self.OMC[5]
        assert np.abs(tt.ttime_mod[5] - truth) < 3 * self.ERR[5]

    @pytest.mark.slow
    def test_time_with_dip_is_kept(self):
        # the transit really is 120 min late: the measured times are adopted
        tt, ttime_obs = self._perform(dip_shift=120)
        # each adopted time is a free parameter
        assert (tt.model_name, tt.model_dof) == ("independent", 12)
        assert np.all(tt.kept_ttv)
        np.testing.assert_allclose(tt.ttime_mod, ttime_obs, rtol=0, atol=1e-9)


class TestSelectUnmeasuredTransits:
    def test_parametric_model_predicts(self):
        # 13 transits of a 360 min parabola, 1 min errors, transit 11
        # unmeasured: it gets the parabola, 10 min off the interpolation
        # between its neighbors
        x = np.arange(13)
        truth = 360 * ((x - 6) / 6) ** 2 / 1440
        omc = truth + np.random.default_rng(6).normal(0, 1, 13) / 1440
        omc[11] = np.nan
        star, planets, lc = _system_with_omc(omc, np.full(13, 1 / 1440))
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        tt = planets_out[0].transit_times

        omc_mod = tt.ttime_mod - (EPOCH + PERIOD * x)
        omc_mod -= np.polyval(np.polyfit(x, omc_mod, 1), x)
        parabola = truth - np.polyval(np.polyfit(x, truth, 1), x)
        assert np.abs(omc_mod[11] - parabola[11]) < 1 / 1440
        assert np.abs(omc_mod[11] - 0.5 * (omc_mod[10] + omc_mod[12])) > 5 / 1440
        kept = tt.kept_ttv & np.isfinite(tt.ttime_obs)
        assert np.isclose(tt.ttv_scatter, mad_std((tt.ttime_obs - tt.ttime_mod)[kept]))

    @pytest.mark.slow
    def test_independent_times_filled_by_gp(self):
        # 5 transits of a TTV a line can't fit (K01783.02), transit 3
        # unmeasured: measured times adopted, transit 3 from a GP on them
        omc = np.array([44.2, -54.8, -23.2, np.nan, 40.0, 8.2]) / 1440
        err = np.array([3.8, 3.9, 3.9, np.nan, 4.4, 7.1]) / 1440
        star, planets, lc = _system_with_omc(omc, err)
        tt_in = planets[0].transit_times
        tlin = tt_in.linear_ephemeris()
        ttime_obs = tt_in.ttime_obs.copy()
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        tt = planets_out[0].transit_times

        measured = np.isfinite(ttime_obs)
        np.testing.assert_allclose(
            tt.ttime_mod[measured], ttime_obs[measured], rtol=0, atol=1e-9
        )
        gp = RegularizeOMC._matern32_model(tlin, ttime_obs - tlin, err, measured)
        np.testing.assert_allclose(
            tt.ttime_mod[3], tlin[3] + gp["pred"][3], rtol=0, atol=1e-9
        )
        # scatter about the GP, not about ttime_mod (= the measured times)
        resid = (ttime_obs - tlin - gp["pred"])[measured]
        assert tt.ttv_scatter > 0
        assert np.isclose(tt.ttv_scatter, mad_std(resid), rtol=0, atol=1e-9)


@pytest.mark.slow
class TestSelect:
    def test_recovers_ttv_and_flags_outliers(self):
        outliers = [12, 30]
        star, planets, lc, truth = _system(ntransit=48, outliers=outliers, missing=[20])
        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")

        tt = planets_out[0].transit_times
        good = np.setdiff1d(np.arange(48), outliers)
        assert _rms(tt.ttime_mod[good] - truth[good]) < TIME_ERR
        assert not np.any(tt.kept_ttv[outliers])
        assert np.mean(~tt.kept_ttv) < 0.3
        assert np.isfinite(tt.ttime_mod[20]) and tt.kept_ttv[20]

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
            TransitTimes("Kepler", "SYN", index, linear + omc, ttime_err=err)
        )
        tt = planets[0].transit_times
        xtime = tt.linear_ephemeris()
        return star, planets, lc, linear, ttv, err, xtime, tt.ttime_obs - xtime

    @staticmethod
    def _first_search_significant(xtime, yobs, err):
        good_clip = RegularizeOMC._sigma_clip(yobs, err)
        _, fap = RegularizeOMC._lomb_scargle_peak(xtime, yobs, good_clip)
        return fap < 0.03

    def test_periodic_signal_from_outliers_is_rejected(self):
        # white noise: the outliers alone give a significant first-search peak,
        # and the sinusoid is selected unless the second search rejects it
        star, planets, lc, _, _, err, xtime, yobs = self._contaminated_sinusoid(15, 0)
        assert self._first_search_significant(xtime, yobs, err)

        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        assert planets_out[0].transit_times.model_name != "sinusoid"

    def test_periodic_signal_hidden_by_outliers_is_recovered(self):
        # 0.8 sigma TTV: found only once the outliers are rejected
        star, planets, lc, linear, ttv, err, xtime, yobs = self._contaminated_sinusoid(
            2, 0.8
        )
        assert not self._first_search_significant(xtime, yobs, err)

        planets_out, _ = RegularizeOMC(star, planets, lc).perform(method="select")
        tt = planets_out[0].transit_times
        assert tt.model_name == "sinusoid"
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
            measured = np.isfinite(tt.ttime_obs) & tt.kept_ttv
            assert np.all(np.abs(tt.ttime_mod - tt.ttime_obs)[measured] < 1 / 24)
