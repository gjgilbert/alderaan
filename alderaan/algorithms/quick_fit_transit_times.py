"""Fit individual transit times by cross-correlation and optimization"""

import logging

import numpy as np
from astropy.stats import mad_std
from scipy.optimize import least_squares

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.data_models import TransitTimes
from alderaan.transit_model import TransitModel, transit_flux
from alderaan.utils.astro import approx_tc_error
from alderaan.utils.stats import robust_polyfit

logger = logging.getLogger(__name__)


class QuickFitTransitTimes(BaseAlg):
    """Measure individual transit times for each planet"""

    @staticmethod
    def _per_transit_snr(p, lc):
        """
        Single-transit SNR, ror^2 sqrt(T14 / exptime) / mad_std(flux), maximum
        over obsmodes
        """
        return max(
            p.ror**2
            * np.sqrt(p.duration / EXPTIME[om])
            / mad_std(lc.flux[lc.obsmode == om])
            for om in np.unique(lc.obsmode)
        )

    @staticmethod
    def _construct_template(tm, planet_no, obsmode, transit_window_size):
        """
        Model transit template for one planet, centered on t = 0

        From the planet's transit shape and the star's limb darkening, sampled at
        half the supersampled cadence and not exposure-integrated.

        Parameters
        ----------
        tm : TransitModel
        planet_no : int
            planet index
        obsmode : str
        transit_window_size : float
            full width of the template (days)

        Returns
        -------
        time_template, flux_template : ndarray
        """
        p = tm.planets[planet_no]

        exptime = EXPTIME[obsmode]
        supersample = tm.supersample[obsmode]

        time_template = np.arange(
            0, transit_window_size / 2, exptime / supersample / 2, dtype=float
        )
        time_template = np.hstack([-time_template[:-1][::-1], time_template])

        flux_template = transit_flux(
            time_template,
            0.0,
            p.period,
            p.ror,
            p.impact,
            p.duration,
            tm.star.u1,
            tm.star.u2,
        )

        return time_template, flux_template

    @staticmethod
    def _predicted_tc_error(tm, planet_no):
        """
        Predicted transit time uncertainty (days) at each transit of one planet

        From the transit shape and the median photometric error of the transit's
        visit and obsmode (Price & Rogers 2014).
        """
        p = tm.planets[planet_no]
        lc = tm.litecurve
        transit_visit = lc.visit_at(p.transit_times.ttime_mod)
        transit_obsmode = tm.transit_obsmode[planet_no]

        tc_error_pred = np.zeros(len(transit_visit))
        for visit, obsmode in set(zip(transit_visit, transit_obsmode)):
            use = (lc.visit == visit) & (lc.obsmode == obsmode)
            sigma_f = np.median(lc.error[use])
            tc_error_pred[(transit_visit == visit) & (transit_obsmode == obsmode)] = (
                approx_tc_error(p.ror, p.impact, p.duration, EXPTIME[obsmode], sigma_f)
            )

        return tc_error_pred

    @staticmethod
    def _fit_one_transit(
        tm,
        obsmode,
        tc,
        duration,
        tc_error_pred,
        transit_window_size,
        time_template,
        flux_template,
    ):
        """
        Cross-correlation fit of a single transit time

        Divides out a linear out-of-transit trend, slides the template across a
        grid of trial transit times, then fits a parabola to chi-square near its
        minimum.

        Parameters
        ----------
        tm : TransitModel
        obsmode : str
            observing mode of this transit
        tc : float
            current transit time estimate (days)
        duration : float
            planet transit duration (days)
        tc_error_pred : float
            predicted transit time uncertainty (days)
        transit_window_size : float
            full width of the data window around tc (days)
        time_template, flux_template : ndarray
            model transit template for this planet and obsmode

        Returns
        -------
        ttime, ttime_err : float
            fitted transit time and its uncertainty (days); NaN if not fit
        """
        lc = tm.litecurve
        exptime = EXPTIME[obsmode]
        supersample = tm.supersample[obsmode]
        exptime_ioff = tm.exptime_offsets[obsmode]

        # STEP 0: data near this transit
        in_window = np.abs(lc.time - tc) < transit_window_size / 2

        _t = lc.time[in_window]
        _f_obs = lc.flux[in_window]
        _f_err = lc.error[in_window]

        _t_supersample = (exptime_ioff + _t.reshape(_t.size, 1)).ravel()

        # remove residual out-of-transit trend, excluding +/- T14
        near_transit = np.abs(lc.time - tc) < duration
        use = ~near_transit[in_window]
        if np.any(use):
            _slope, _intercept = robust_polyfit(_t[use], _f_obs[use], 1)
            _f_obs /= _intercept + _slope * _t

        # STEP 1: chisq vs trial tc; gridstep resolves the tc error, and trial
        # transits stay fully inside the data window
        gridstep = min(exptime / supersample / 5, tc_error_pred)
        tc_offset = np.arange(0, (transit_window_size - duration) / 2, gridstep)
        tc_offset = tc + np.hstack([-tc_offset[:-1][::-1], tc_offset])
        chisq = np.zeros_like(tc_offset)

        for j, tc_o in enumerate(tc_offset):
            _f_mod = np.interp(_t_supersample - tc_o, time_template, flux_template)
            _f_mod = _f_mod.reshape(-1, len(exptime_ioff)).mean(axis=1)

            chisq[j] = np.sum(((_f_obs - _f_mod) / _f_err) ** 2)

        # no trial transit fits better than no transit at all
        if chisq.min() >= np.sum(((_f_obs - 1) / _f_err) ** 2):
            logger.debug("transit at %.3f not fit: no detection", tc)
            return np.nan, np.nan

        # STEP 2: widen delta_chisq until enough points near the minimum;
        # schedule (2 to 16, x sqrt(2)) tuned by trial and error
        delta_chisq = 2.0

        loop = True
        while loop:
            tc_fit = tc_offset[chisq < chisq.min() + delta_chisq]
            x2_fit = chisq[chisq < chisq.min() + delta_chisq]

            # eliminate points far from the local minimum
            faraway = np.abs(tc_fit - np.median(tc_fit)) / gridstep > 1 + 0.5 * len(
                tc_fit
            )

            tc_fit = tc_fit[~faraway]
            x2_fit = x2_fit[~faraway]

            if len(x2_fit) > 7:
                loop = False
            if delta_chisq >= 16:
                loop = False

            delta_chisq *= np.sqrt(2)

        # STEP 3: fit a parabola to local chisq minimum
        if len(tc_fit) <= 3:
            logger.debug("transit at %.3f not fit: too few chisq points", tc)
            return np.nan, np.nan

        quad_coeffs = robust_polyfit(tc_fit, x2_fit, 2)
        quad_model = np.polyval(quad_coeffs, tc_fit)
        qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])

        _tti = np.mean([qtc_min, np.mean(tc_fit)])

        convex_local_min = quad_coeffs[0] > 0
        within_bounds = (_tti > tc_fit.min()) and (_tti < tc_fit.max())

        if not (convex_local_min and within_bounds):
            logger.debug("transit at %.3f not fit: ill-conditioned", tc)
            return np.nan, np.nan

        # 1-sigma from the parabola curvature (delta chi2 = 1), inflated by the chi2
        # scatter about the parabola: a heuristic, tuned by trial and error
        qtc_err = np.sqrt(1 / quad_coeffs[0])
        return _tti, qtc_err * (1 + np.std(x2_fit - quad_model))

    def _cross_correlation_times(
        self,
        tm,
        planet_no,
        rel_window_size=5.0,
        abs_window_size_buffer=1 / 24,
    ):
        """
        Fit transit times using the Mazeh+13 / Holczer+16 cross-correlation method

        The data window around each transit is rel_window_size * T14 +
        abs_window_size_buffer (days) wide; transits with a sibling transit
        inside the window or no in-transit data (data gap) are skipped.

        Returns
        -------
        ttime, ttime_err : ndarray
            fitted transit times and uncertainties; NaN where not fit
        sibling_in_window : ndarray of bool
            a sibling transit is inside the data window (not fit)
        """
        p = tm.planets[planet_no]

        transit_obsmode = tm.transit_obsmode[planet_no]
        ntransit = len(p.transit_times.ttime_mod)
        transit_window_size = rel_window_size * p.duration + abs_window_size_buffer

        # a sibling transit inside the data window could capture the template
        sibling_in_window = np.zeros(ntransit, dtype=bool)
        for j, pj in enumerate(tm.planets):
            if j != planet_no:
                dt = np.abs(
                    p.transit_times.ttime_mod[:, None] - pj.transit_times.ttime_mod
                )
                sibling_in_window |= np.any(
                    dt < transit_window_size / 2 + pj.duration / 2, axis=1
                )

        templates = {
            obsmode: self._construct_template(
                tm, planet_no, obsmode, transit_window_size
            )
            for obsmode in tm.unique_obsmodes
        }
        tc_error_pred = self._predicted_tc_error(tm, planet_no)

        ttime = np.full(ntransit, np.nan)
        ttime_err = np.full(ntransit, np.nan)
        logger.debug("planet %d: cross-correlation fit", planet_no)
        # no in-transit data: the transit falls in a data gap
        in_gap = np.array(
            [
                not np.any(np.abs(tm.litecurve.time - tc) < p.duration / 2)
                for tc in p.transit_times.ttime_mod
            ],
            dtype=bool,
        )

        for i, tc in enumerate(p.transit_times.ttime_mod):
            if sibling_in_window[i] or in_gap[i]:
                continue

            obsmode = transit_obsmode[i]
            ttime[i], ttime_err[i] = self._fit_one_transit(
                tm,
                obsmode,
                tc,
                p.duration,
                tc_error_pred[i],
                transit_window_size,
                *templates[obsmode],
            )

        nfit = int(np.sum(~np.isnan(ttime)))
        nattempt = int(np.sum(~sibling_in_window & ~in_gap))
        logger.info(
            "planet %d: cross-correlation fit %d of %d transits "
            "(%d skipped: sibling in window; %d skipped: in data gap)",
            planet_no,
            nfit,
            ntransit,
            np.sum(sibling_in_window),
            np.sum(in_gap & ~sibling_in_window),
        )
        if nattempt - nfit > 0.5 * nattempt:
            logger.warning(
                "planet %d: cross-correlation fit failed for %d of %d attempted "
                "transits",
                planet_no,
                nattempt - nfit,
                nattempt,
            )

        return ttime, ttime_err, sibling_in_window

    @staticmethod
    def _window_flux(theta, tm, obsmode, t_obs):
        """Model flux at t_obs with planet n transiting at theta[n]; shapes fixed"""
        f_mod = np.ones_like(t_obs)

        for n, p in enumerate(tm.planets):
            qld_flux = transit_flux(
                t_obs,
                theta[n],
                p.period,
                p.ror,
                p.impact,
                p.duration,
                tm.star.u1,
                tm.star.u2,
                exptime_offsets=tm.exptime_offsets[obsmode],
            )

            f_mod += qld_flux - 1.0

        return f_mod

    def _window_residuals(self, theta, tm, obsmode, t_obs, f_obs, f_err):
        """Normalized residuals (f_obs - model) / f_err"""
        f_mod = self._window_flux(theta, tm, obsmode, t_obs)
        return (f_obs - f_mod) / f_err

    def _window_ln_likelihood(self, theta, tm, obsmode, t_obs, f_obs, f_err):
        """Gaussian log-likelihood in the window; -1e300 if not finite"""
        res = self._window_residuals(theta, tm, obsmode, t_obs, f_obs, f_err)
        lnlike = -0.5 * np.sum(res**2)

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike

    def _optimize_times(self, tm, planet_no, overlap, fit, relative_window_size=3.0):
        """
        Fit the transit times of one planet by least squares, one window at a time

        For each transit where fit is True, the window is relative_window_size x
        T14 wide (if the transit overlaps another planet, per overlap:
        relative_window_size x (T14 + the longest T14), at most P/2). The nearest
        transit of every planet inside the window varies, with a Normal(start,
        T14/3) prior bounded to the window. Returns times and errors (days, from
        the Jacobian); NaN where not fit, failed, or likelihood did not improve.
        """
        lc = tm.litecurve
        overlap = overlap[planet_no]
        obsmode = tm.transit_obsmode[planet_no]

        ttime_old = tm.planets[planet_no].transit_times.ttime_mod
        ttime_new = np.full_like(ttime_old, np.nan)
        error_new = np.full_like(ttime_old, np.nan)
        prior_sd = np.array([p.duration / 3 for p in tm.planets])
        nfail = 0

        for i, tc in enumerate(ttime_old):
            if not fit[i]:
                continue

            if not overlap[i]:
                transit_window_size = (
                    relative_window_size * tm.planets[planet_no].duration
                )
            else:
                transit_window_size = relative_window_size * (
                    tm.planets[planet_no].duration
                    + np.max([p.duration for p in tm.planets])
                )
                transit_window_size = np.min(
                    [transit_window_size, 0.5 * tm.planets[planet_no].period]
                )

            theta = np.zeros(tm.npl, dtype=float)
            fix = np.zeros(tm.npl, dtype=bool)

            for n, p in enumerate(tm.planets):
                theta[n] = p.transit_times.ttime_mod[
                    np.argmin(np.abs(p.transit_times.ttime_mod - tc))
                ]

                if np.abs(theta[n] - tc) > transit_window_size / 2:
                    fix[n] = True

            use = np.abs(lc.time - tc) < transit_window_size / 2
            t_obs = lc.time[use]
            f_obs = lc.flux[use]
            f_err = lc.error[use]

            logp_initial = self._window_ln_likelihood(
                theta, tm, obsmode[i], t_obs, f_obs, f_err
            )

            def _fxn(x, x0, fix, tm, obsmode, t_obs, f_obs, f_err):
                """Window residuals and prior terms, free entries (~fix) set to x"""
                _x = x0.copy()
                _x[~fix] = x
                resid = self._window_residuals(_x, tm, obsmode, t_obs, f_obs, f_err)
                prior = (x - x0[~fix]) / prior_sd[~fix]
                return np.concatenate([resid, prior])

            half_window = transit_window_size / 2
            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
                bounds=(theta[~fix] - half_window, theta[~fix] + half_window),
                args=(
                    theta,
                    fix,
                    tm,
                    obsmode[i],
                    t_obs,
                    f_obs,
                    f_err,
                ),
            )

            theta[~fix] = result.x
            logp_final = self._window_ln_likelihood(
                theta, tm, obsmode[i], t_obs, f_obs, f_err
            )

            logger.debug(
                "planet %d transit %d: tc %.5f -> %.5f d (%+.2f min), "
                "logp %.6g -> %.6g, nfev = %d",
                planet_no,
                i,
                tc,
                theta[planet_no],
                (theta[planet_no] - tc) * 1440,
                logp_initial,
                logp_final,
                result.nfev,
            )
            if result.success and logp_final >= logp_initial:
                k = np.sum(~fix[:planet_no])
                cov = np.linalg.inv(result.jac.T @ result.jac)
                ttime_new[i] = theta[planet_no]
                error_new[i] = np.sqrt(cov[k, k])
            else:
                nfail += 1

        if nfail > 0:
            logger.warning(
                "planet %d: optimizer failed for %d of %d transits (unmeasured)",
                planet_no,
                nfail,
                np.sum(fit),
            )

        return ttime_new, error_new

    def perform(self):
        """
        Measure transit times by cross-correlation, or by least squares if blocked

        Least squares fits the transits where a sibling transit blocked
        cross-correlation. Only transits with phot_coverage and phot_noise are
        measured. Each planet's TransitTimes is replaced: ttime_obs and ttime_err
        (floored at the predicted error) are NaN where unmeasured, and ttime_mod
        is ttime_obs where measured, else unchanged.

        A planet with per-transit SNR < 2, or T14 shorter than its shortest
        cadence, is not measured: it gets its linear ephemeris, model_name
        'linear', and no observed times.
        """
        star = self.star
        planets = self.planets
        litecurve = self.litecurve
        npl = len(planets)

        # below SNR ~ 2, or with T14 shorter than one cadence, a single-transit
        # time locks onto noise: linear ephemeris, with the predicted timing error
        # as its TTV scatter
        tm = TransitModel(star, planets, litecurve)
        exptime = min(EXPTIME[om] for om in np.unique(litecurve.obsmode))
        linear = []
        for n, p in enumerate(planets):
            snr = self._per_transit_snr(p, litecurve)
            p.meta["snr_per_transit"] = snr
            linear.append(snr < 2 or p.duration < exptime)
            if linear[n]:
                logger.info(
                    "planet %d: per-transit SNR = %.2f, T14 = %.1f min; linear",
                    n,
                    snr,
                    p.duration * 24 * 60,
                )
                tt = p.transit_times
                p.set_transit_times(
                    TransitTimes(
                        tt.mission,
                        tt.target,
                        tt.index,
                        ttime_obs=np.full(len(tt.index), np.nan),
                        ttime_mod=tt.linear_ephemeris(),
                        phot_coverage=tt.phot_coverage,
                        phot_noise=tt.phot_noise,
                        model_name="linear",
                        model_dof=2.0,
                        ttv_scatter=np.median(self._predicted_tc_error(tm, n)),
                    )
                )

        # Mazeh+13 / Holczer+16 cross-correlation; sibling models at linear times
        if any(linear):
            tm = TransitModel(star, planets, litecurve)
        ttime_cc = [None] * npl
        error_cc = [None] * npl
        sibling_in_window = [None] * npl

        for n, p in enumerate(planets):
            if linear[n]:
                continue
            ttime_cc[n], error_cc[n], sibling_in_window[n] = (
                self._cross_correlation_times(tm, n)
            )

            assert len(ttime_cc[n]) == len(p.transit_times.ttime_mod)

            # transits with poor photometry (FlagBadTransits) are not measured
            tt = p.transit_times
            phot = tt.phot_coverage & tt.phot_noise
            ttime_cc[n][~phot] = np.nan
            error_cc[n][~phot] = np.nan

            if np.all(np.isnan(error_cc[n])):
                continue
            # 8 x median tuned by trial and error
            large = error_cc[n] > 8 * np.nanmedian(error_cc[n])
            if np.any(large):
                logger.info(
                    "planet %d: %d cross-correlation fits discarded "
                    "(error > 8x median)",
                    n,
                    np.sum(large),
                )
            ttime_cc[n][large] = np.nan
            error_cc[n][large] = np.nan

        # optimizer starting times: cross-correlation, else the previous model
        for n, p in enumerate(planets):
            if linear[n]:
                continue
            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            tt = p.transit_times
            p.set_transit_times(
                TransitTimes(
                    tt.mission,
                    tt.target,
                    tt.index,
                    ttime_obs=np.where(q, ttime_cc[n], np.nan),
                    ttime_err=np.where(q, error_cc[n], np.nan),
                    ttime_mod=np.where(q, ttime_cc[n], tt.ttime_mod),
                    phot_coverage=tt.phot_coverage,
                    phot_noise=tt.phot_noise,
                )
            )

        # least squares optimization where a sibling transit blocked
        # cross-correlation
        tm = TransitModel(star, planets, litecurve)
        overlap = self.identify_overlapping_transits(rel_size=1.0, abs_size=1 / 24)
        ttime_opt = [None] * npl
        error_opt = [None] * npl

        for n, p in enumerate(planets):
            if linear[n]:
                continue
            tt = p.transit_times
            fit = tt.phot_coverage & tt.phot_noise & sibling_in_window[n]
            ttime_opt[n], error_opt[n] = self._optimize_times(tm, n, overlap, fit)

        # measured times: cross-correlation, else optimizer; ttime_mod = ttime_obs
        # where measured, else the previous model
        for n, p in enumerate(planets):
            if linear[n]:
                continue
            tt = p.transit_times
            good = tt.phot_coverage & tt.phot_noise
            tc_error_pred = self._predicted_tc_error(tm, n)

            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            measured = good & (q | np.isfinite(ttime_opt[n]))
            ttime_new = np.where(q, ttime_cc[n], ttime_opt[n])
            # predicted error ~ Cramer-Rao bound: floor on the measured error
            error_new = np.maximum(
                np.where(q, error_cc[n], error_opt[n]), tc_error_pred
            )
            ttime_new[~measured] = np.nan
            error_new[~measured] = np.nan

            p.set_transit_times(
                TransitTimes(
                    tt.mission,
                    tt.target,
                    tt.index,
                    ttime_obs=ttime_new,
                    ttime_err=error_new,
                    ttime_mod=np.where(measured, ttime_new, tt.ttime_mod),
                    phot_coverage=tt.phot_coverage,
                    phot_noise=tt.phot_noise,
                )
            )

            omc = p.transit_times.ttime_obs - p.transit_times.linear_ephemeris()
            logger.info(
                "planet %d: %d cross-correlation + %d optimizer times, "
                "%d unmeasured, %d low quality; "
                "P = %.6f d, epoch = %.5f BKJD, OMC rms = %.2f min, "
                "median error = %.2f min",
                n,
                np.sum(q),
                np.sum(~q & measured),
                np.sum(good & ~measured),
                np.sum(~good),
                planets[n].period,
                planets[n].epoch,
                np.nanstd(omc) * 1440,
                np.nanmedian(error_new) * 1440,
            )

        return self.planets, self.litecurve
