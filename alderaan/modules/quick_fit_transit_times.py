__all__ = ["QuickFitTransitTimes"]

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from batman import _quadratic_ld, _rsky
from scipy.optimize import least_squares

from alderaan.modules.transit import TransitModel
from alderaan.utils.astro import bin_data, predict_tc_error
from alderaan.utils.plotting import plot_quick_fit_ttvs
from alderaan.utils.stats import robust_polyfit


class OptimizationTTVFitter(TransitModel):
    def __init__(self, litecurve, planets, star):
        super().__init__(litecurve, planets, star)

    def optimize(self, planet_no, relative_window_size=3.0):
        lc = self.litecurve
        overlap = self.identify_overlapping_transits(rtol=1.0, atol=1 / 24)[planet_no]
        obsmode = self.transit_obsmode[planet_no]

        ttime_old = self.planets[planet_no].transit_times.copy()
        ttime_new = np.zeros_like(ttime_old)

        for i, tc in enumerate(ttime_old):
            print(f"  {i}, tc = {np.round(tc, 1)}")

            if not overlap[i]:
                transit_window_size = (
                    relative_window_size * self.planets[planet_no].duration
                )
            else:
                transit_window_size = relative_window_size * (
                    self.planets[planet_no].duration
                    + np.max([p.duration for p in self.planets])
                )
                transit_window_size = np.min(
                    [transit_window_size, 0.5 * self.planets[planet_no].period]
                )

            theta = np.zeros(self.npl, dtype=float)
            fix = np.zeros(self.npl, dtype=bool)

            for n, p in enumerate(self.planets):
                theta[n] = p.transit_times[np.argmin(np.abs(p.transit_times - tc))]

                if np.abs(theta[n] - tc) > transit_window_size / 2:
                    fix[n] = True

            use = np.abs(lc.time - tc) < transit_window_size / 2
            t_obs = lc.time[use]
            f_obs = lc.flux[use]
            f_err = lc.error[use]

            print(f"optimizing logp for transit {i} (tc = {np.round(tc, 1)})")

            logp_initial = self.ln_likelihood(
                theta, self, obsmode[i], t_obs, f_obs, f_err
            ).copy()

            def _fxn(x, x0, fix, self, obsmode, t_obs, f_obs, f_err):
                _x = x0.copy()
                _x[~fix] = x
                return self.model_residuals(_x, self, obsmode, t_obs, f_obs, f_err)

            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
                args=(
                    theta,
                    fix,
                    self,
                    obsmode[i],
                    t_obs,
                    f_obs,
                    f_err,
                ),
            )

            theta[~fix] = result.x.copy()
            ttime_new[i] = theta[planet_no].copy()
            logp_final = self.ln_likelihood(
                theta, self, obsmode[i], t_obs, f_obs, f_err
            ).copy()

            print(f"logp: {logp_initial} -> {logp_final}")
            print(f"message: {result['message']}")

        return ttime_new

    @staticmethod
    def ln_likelihood(theta, transitmodel, obsmode, t_obs, f_obs, f_err):
        res = transitmodel.model_residuals(
            theta, transitmodel, obsmode, t_obs, f_obs, f_err
        )
        lnlike = -0.5 * np.sum(res**2)

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike

    @staticmethod
    def model_residuals(theta, transitmodel, obsmode, t_obs, f_obs, f_err):
        f_mod = transitmodel.model_flux(theta, transitmodel, obsmode, t_obs)
        return (f_obs - f_mod) / f_err

    @staticmethod
    def model_flux(theta, transitmodel, obsmode, t_obs):
        tm = transitmodel

        exptime_ioff = tm._exptime_offsets[obsmode]
        supersample = tm.supersample[obsmode]

        _t_supersample = (exptime_ioff + t_obs.reshape(t_obs.size, 1)).flatten()

        f_mod = np.ones_like(t_obs)

        for n, p in enumerate(tm.planets):
            nthreads = 1
            ds = _rsky._rsky(
                _t_supersample,
                theta[n],
                p.period,
                p.ror,
                p.impact,
                p.duration,
                1,
                nthreads,
            )

            qld_flux = _quadratic_ld._quadratic_ld(
                ds, np.abs(p.ror), tm.star.u1, tm.star.u2, nthreads
            )

            qld_flux = np.mean(qld_flux.reshape(-1, supersample), axis=1)

            f_mod += qld_flux - 1.0

        return f_mod


class CrossCorrelationTTVFitter(TransitModel):
    def __init__(self, litecurve, planets, star):
        super().__init__(litecurve, planets, star)

    def _construct_template(self, planet_no, obsmode, transit_window_size):
        """
        planet_no (int) : planet number index

        IMPORTANT!!! Planet extensions must have desired [P, Rp/Rs, b, T14]
        """
        # shorthand
        p = self.planets[planet_no]

        exptime = self.exptime[obsmode]
        supersample = self.supersample[obsmode]

        time_template = np.arange(
            0, transit_window_size / 2, exptime / supersample / 2, dtype=float
        )
        time_template = np.hstack([-time_template[:-1][::-1], time_template])
        flux_template = np.zeros_like(time_template)

        nthreads = 1
        ds = _rsky._rsky(
            time_template,
            0.0,
            p.period,
            p.ror,
            p.impact,
            p.duration,
            1,
            nthreads,
        )

        qld_flux = _quadratic_ld._quadratic_ld(
            ds, np.abs(p.ror), self.star.u1, self.star.u2, nthreads
        )

        flux_template += qld_flux

        return time_template, flux_template

    def mazeh13_holczer16_method(
        self,
        planet_no,
        rel_window_size=5.0,
        abs_window_size_buffer=1 / 24,
        quicklook_dir=None,
        target=None,
    ):
        """
        Docstring
        """
        # shorthand
        lc = self.litecurve
        p = self.planets[planet_no]

        ttime = np.nan * np.ones(len(p.transit_times))
        ttime_err = np.nan * np.ones(len(p.transit_times))

        overlap = self.identify_overlapping_transits(rtol=1.0, atol=1 / 24)

        assert len(p.transit_times) == len(overlap[planet_no]), (
            f"Mismatched sizes for ttime ({len(p.transit_times)}) "
            f"and overlap ({len(overlap[planet_no])})"
        )

        transit_obsmode = self.transit_obsmode[planet_no]

        for obsmode in self.unique_obsmodes:
            print(f"  Determining transit times using {obsmode} data")

            exptime = self.exptime[obsmode]
            supersample = self.supersample[obsmode]
            exptime_ioff = self._exptime_offsets[obsmode]

            transit_window_size = rel_window_size * p.duration + abs_window_size_buffer
            time_template, flux_template = self._construct_template(
                planet_no, obsmode, transit_window_size
            )

            # primary loop for fitting transit times
            for i, tc in enumerate(p.transit_times):
                if (
                    (not overlap[planet_no][i])
                    and (p.transit_quality[i])
                    and (transit_obsmode[i] == obsmode)
                ):
                    # STEP 0: pull data near a single transit
                    in_transit = np.abs(self.litecurve.time - tc) < p.duration / 2
                    in_window = (
                        np.abs(self.litecurve.time - tc) < transit_window_size / 2
                    )

                    if np.sum(in_transit) > 0:
                        _t = lc.time[in_window]
                        _f_obs = lc.flux[in_window]
                        _f_err = lc.error[in_window]

                        _t_supersample = (
                            exptime_ioff + _t.reshape(_t.size, 1)
                        ).flatten()

                        # remove any residual out-of-transit trend
                        use = ~in_transit[in_window]
                        if np.any(use):
                            _slope, _intercept = robust_polyfit(_t[use], _f_obs[use], 1)
                            _f_obs /= _intercept + _slope * _t

                        # STEP 1: generate tc_offset vs chisq vectors
                        gridstep = exptime / supersample / 1.618 / 3
                        tc_offset = np.arange(0, transit_window_size / 2, gridstep)
                        tc_offset = tc + np.hstack([-tc_offset[:-1][::-1], tc_offset])
                        chisq = np.zeros_like(tc_offset)

                        for j, tc_o in enumerate(tc_offset):
                            _f_mod = np.interp(
                                _t_supersample - tc_o, time_template, flux_template
                            )
                            _f_mod = bin_data(
                                _t_supersample, _f_mod, exptime, bin_centers=_t
                            )[1]

                            chisq[j] = np.sum(((_f_obs - _f_mod) / _f_err) ** 2)

                        # STEP 2: isolate relevant portions of
                        # {tc_offset, chisq} vectors
                        delta_chisq = 2.0

                        loop = True
                        while loop:
                            # grab data near chisq minimum
                            tc_fit = tc_offset[chisq < chisq.min() + delta_chisq]
                            x2_fit = chisq[chisq < chisq.min() + delta_chisq]

                            # eliminate points far from the local minimum
                            faraway = np.abs(tc_fit - np.median(tc_fit)) / np.median(
                                np.diff(tc_fit)
                            ) > 1 + 0.5 * len(tc_fit)

                            tc_fit = tc_fit[~faraway]
                            x2_fit = x2_fit[~faraway]

                            # check stopping conditions
                            if len(x2_fit) > 7:
                                loop = False
                            if delta_chisq >= 16:
                                loop = False

                            # increment chisq
                            delta_chisq *= np.sqrt(2)

                        # STEP 3: fit a parabola to local chisq minimum
                        if len(tc_fit) > 3:
                            # polynomial fitting
                            quad_coeffs = robust_polyfit(tc_fit, x2_fit, 2)
                            quad_model = np.polyval(quad_coeffs, tc_fit)
                            qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])
                            qtc_err = np.sqrt(1 / quad_coeffs[0])

                            # transit time and scaled error
                            _tti = np.nanmean([qtc_min, np.mean(tc_fit)])
                            _erri = qtc_err * (1 + np.std(x2_fit - quad_model))

                            # check that the fit is well-conditioned
                            convex_local_min = quad_coeffs[0] > 0
                            within_bounds = (_tti > tc_fit.min()) and (
                                _tti < tc_fit.max()
                            )

                            if convex_local_min and within_bounds:
                                ttime[i] = _tti.copy()
                                ttime_err[i] = _erri.copy()

                        # STEP 4: make quicklook plot
                        if (quicklook_dir is not None) and (len(_f_obs) > 0):
                            ttv_dir = os.path.join(quicklook_dir, "ttvs")
                            os.makedirs(ttv_dir, exist_ok=True)
                            _filepath = os.path.join(
                                ttv_dir, f"{target}_{planet_no}_ttv_{i}.png"
                            )

                            _t_obs = _t
                            _t_mod = _t_supersample
                            _f_mod = np.interp(
                                _t_supersample - _tti, time_template, flux_template
                            )

                            _ = plot_quick_fit_ttvs(
                                target,
                                planet_no,
                                _t_obs,
                                _t_mod,
                                _f_obs,
                                _f_mod,
                                tc,
                                tc_offset,
                                tc_fit,
                                x2_fit,
                                chisq,
                                transit_window_size,
                                filepath=_filepath,
                            )

        return ttime, ttime_err


class QuickFitTransitTimes:
    def __init__(self, shire, target, quicklook_dir):
        self.shire = deepcopy(shire)
        self.target = target
        self.quicklook_dir = quicklook_dir

    def perform(self):
        print("\n\nMODULE: QUICK FIT TRANSIT TIMES\n")

        star = self.shire.star
        planets = self.shire.planets
        litecurve = self.shire.litecurve
        npl = len(planets)

        # fit transit times using Mazeh+13 / Holczer+16 cross-correlation method
        print("\nApplying cross-correlation method")
        ttvmodel = CrossCorrelationTTVFitter(litecurve, planets, star)
        ttime_cc = [None] * npl
        error_cc = [None] * npl

        for n, p in enumerate(planets):
            print(f"\nPlanet {n} : fitting {np.sum(p.transit_quality)} transit times")

            ttime_cc[n], error_cc[n] = ttvmodel.mazeh13_holczer16_method(
                n, target=self.target, quicklook_dir=self.quicklook_dir
            )

            assert len(ttime_cc[n]) == len(p.transit_times)
            assert len(error_cc[n]) == len(p.transit_errors)

            nfit = np.sum(~np.isnan(ttime_cc[n]))
            ntot = np.sum(p.transit_quality)

            print(
                f"Planet {n} : {nfit} of {ntot} transit times "
                f"({nfit / ntot * 100:.1f}%) fit successfully"
            )

        # update ephemerides
        for n, p in enumerate(planets):
            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            p.transit_times[q] = ttime_cc[n][q]
            p.transit_errors[q] = error_cc[n][q]
            planets[n] = p.refit_period_and_epoch()

        # fit transit times using least squares optimization
        print("\nOptimizing independent transit times")
        ttvmodel = OptimizationTTVFitter(litecurve, planets, star)
        ttime_opt = [None] * npl

        for n in range(npl):
            print(f"\nPlanet {n}")
            ttime_opt[n] = ttvmodel.optimize(n)

        # update ephemerides, prioritizing cross-correlation transit times
        for n, p in enumerate(planets):
            error_obs = np.copy(error_cc[n])
            error_exp = np.zeros_like(error_obs)

            for obsmode in ttvmodel.unique_obsmodes:
                exptime = ttvmodel.exptime[obsmode]
                sigma_f = np.nanmedian(litecurve.error[litecurve.obsmode == obsmode])
                error_exp[ttvmodel.transit_obsmode[n] == obsmode] = predict_tc_error(
                    p.ror, p.impact, p.duration, exptime, sigma_f
                )

            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            ttime_new = np.copy(ttime_cc[n])
            ttime_new[~q] = ttime_opt[n][~q]
            error_new = np.sqrt(error_obs**2 + error_exp**2)
            error_new[~q] = np.nanpercentile(error_new, 90)

            p.transit_times = ttime_new
            p.transit_errors = error_new
            planets[n] = p.refit_period_and_epoch()

        plt.close("all")

        return self.shire
