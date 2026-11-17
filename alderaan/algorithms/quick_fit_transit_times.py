"""Fit individual transit times by cross-correlation and optimization"""

import logging

import numpy as np
from scipy.optimize import least_squares

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.data_models import TransitTimes
from alderaan.transit_model import TransitModel, transit_flux
from alderaan.utils.astro import approx_tc_error
from alderaan.utils.stats import robust_polyfit

logger = logging.getLogger(__name__)


class QuickFitTransitTimes(BaseAlg):
    """
    Measure individual transit times for each planet

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    @staticmethod
    def _construct_template(tm, planet_no, obsmode, transit_window_size):
        """
        Model transit template for one planet, from its orbital period, ror,
        impact, and duration

        Parameters
        ----------
        tm : TransitModel
        planet_no : int
            planet index
        obsmode : str
        transit_window_size : float
            full width of the template (days)
        """
        # shorthand
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
        Predicted transit time uncertainty at each transit of one planet, from
        its transit shape and the median photometric error of the transit's
        visit and obsmode (Price & Rogers 2014)
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

        Slides the template across a grid of trial transit times, then fits a
        parabola to chi-square near its minimum

        Parameters
        ----------
        tm : TransitModel
        obsmode : str
            observing mode of this transit
        tc : float
            current transit time estimate
        duration : float
            planet transit duration (days)
        tc_error_pred : float
            predicted transit time uncertainty (days); caps the tc grid step
        transit_window_size : float
            full width of the data window around tc (days)
        time_template, flux_template : ndarray
            model transit template for this planet and obsmode

        Returns
        -------
        ttime, ttime_err : float
            fitted transit time and its uncertainty; NaN if the fit was skipped
        skip_reason : str or None
            why the fit was skipped, or None on success
        """
        lc = tm.litecurve
        exptime = EXPTIME[obsmode]
        supersample = tm.supersample[obsmode]
        exptime_ioff = tm.exptime_offsets[obsmode]

        # STEP 0: pull data near a single transit
        in_transit = np.abs(lc.time - tc) < duration / 2
        in_window = np.abs(lc.time - tc) < transit_window_size / 2

        if np.sum(in_transit) == 0:
            return np.nan, np.nan, "no in-transit data"

        _t = lc.time[in_window]
        _f_obs = lc.flux[in_window]
        _f_err = lc.error[in_window]

        _t_supersample = (exptime_ioff + _t.reshape(_t.size, 1)).ravel()

        # remove any residual out-of-transit trend, excluding +/- T14 (as legacy)
        # so ingress/egress stays out of the baseline when tc is off by < T14/2
        near_transit = np.abs(lc.time - tc) < duration
        use = ~near_transit[in_window]
        if np.any(use):
            _slope, _intercept = robust_polyfit(_t[use], _f_obs[use], 1)
            _f_obs /= _intercept + _slope * _t

        # STEP 1: generate tc_offset vs chisq vectors; the step must resolve the
        # chisq minimum (width ~ tc error), else too few points pass STEP 2
        gridstep = min(exptime / supersample / 5, tc_error_pred)
        tc_offset = np.arange(0, transit_window_size / 2, gridstep)
        tc_offset = tc + np.hstack([-tc_offset[:-1][::-1], tc_offset])
        chisq = np.zeros_like(tc_offset)

        for j, tc_o in enumerate(tc_offset):
            _f_mod = np.interp(_t_supersample - tc_o, time_template, flux_template)
            _f_mod = _f_mod.reshape(-1, len(exptime_ioff)).mean(axis=1)

            chisq[j] = np.sum(((_f_obs - _f_mod) / _f_err) ** 2)

        # STEP 2: isolate relevant portions of {tc_offset, chisq} vectors
        delta_chisq = 2.0

        loop = True
        while loop:
            # grab data near chisq minimum
            tc_fit = tc_offset[chisq < chisq.min() + delta_chisq]
            x2_fit = chisq[chisq < chisq.min() + delta_chisq]

            # eliminate points far from the local minimum
            faraway = np.abs(tc_fit - np.median(tc_fit)) / gridstep > 1 + 0.5 * len(
                tc_fit
            )

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
        if len(tc_fit) <= 3:
            return np.nan, np.nan, "too few chisq points"

        # polynomial fitting
        quad_coeffs = robust_polyfit(tc_fit, x2_fit, 2)
        quad_model = np.polyval(quad_coeffs, tc_fit)
        qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])

        # transit time
        _tti = np.mean([qtc_min, np.mean(tc_fit)])

        # check that the fit is well-conditioned
        convex_local_min = quad_coeffs[0] > 0
        within_bounds = (_tti > tc_fit.min()) and (_tti < tc_fit.max())

        if not (convex_local_min and within_bounds):
            return np.nan, np.nan, "ill-conditioned"

        # scaled error; sqrt is real only for a convex fit
        qtc_err = np.sqrt(1 / quad_coeffs[0])
        return _tti, qtc_err * (1 + np.std(x2_fit - quad_model)), None

    def _cross_correlation_times(
        self,
        tm,
        planet_no,
        overlap,
        rel_window_size=5.0,
        abs_window_size_buffer=1 / 24,
    ):
        """
        Fit transit times using the Mazeh+13 / Holczer+16 cross-correlation method

        Overlapping transits are skipped; the data window around each transit is
        rel_window_size * duration + abs_window_size_buffer (days) wide
        """
        # shorthand
        p = tm.planets[planet_no]

        assert len(p.transit_times.ttime_mod) == len(overlap[planet_no]), (
            f"Mismatched sizes for ttime ({len(p.transit_times.ttime_mod)}) "
            f"and overlap ({len(overlap[planet_no])})"
        )

        transit_obsmode = tm.transit_obsmode[planet_no]
        ntransit = len(p.transit_times.ttime_mod)
        transit_window_size = rel_window_size * p.duration + abs_window_size_buffer

        # one template per observing mode
        templates = {
            obsmode: self._construct_template(
                tm, planet_no, obsmode, transit_window_size
            )
            for obsmode in tm.unique_obsmodes
        }
        tc_error_pred = self._predicted_tc_error(tm, planet_no)

        ttime = np.full(ntransit, np.nan)
        ttime_err = np.full(ntransit, np.nan)
        skipped = {
            "overlapping": int(np.sum(overlap[planet_no])),
            "no in-transit data": 0,
            "too few chisq points": 0,
            "ill-conditioned": 0,
        }

        for i, tc in enumerate(p.transit_times.ttime_mod):
            if overlap[planet_no][i]:
                continue

            obsmode = transit_obsmode[i]
            ttime[i], ttime_err[i], skip_reason = self._fit_one_transit(
                tm,
                obsmode,
                tc,
                p.duration,
                tc_error_pred[i],
                transit_window_size,
                *templates[obsmode],
            )
            if skip_reason is not None:
                skipped[skip_reason] += 1

        nfit = int(np.sum(~np.isnan(ttime)))
        logger.log(
            logging.WARNING if nfit < 0.5 * ntransit else logging.INFO,
            "planet %d: cross-correlation fit %d of %d transits (skipped: %s)",
            planet_no,
            nfit,
            ntransit,
            ", ".join(f"{v} {k}" for k, v in skipped.items()),
        )

        return ttime, ttime_err

    @staticmethod
    def _window_flux(theta, tm, obsmode, t_obs):
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
        f_mod = self._window_flux(theta, tm, obsmode, t_obs)
        return (f_obs - f_mod) / f_err

    def _window_ln_likelihood(self, theta, tm, obsmode, t_obs, f_obs, f_err):
        res = self._window_residuals(theta, tm, obsmode, t_obs, f_obs, f_err)
        lnlike = -0.5 * np.sum(res**2)

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike

    def _optimize_times(self, tm, planet_no, overlap, relative_window_size=3.0):
        """
        Fit each transit time independently using least squares optimization

        Each varied transit time has a Normal(start, T14/3) prior (as legacy) and
        is bounded to the fit window. Returns NaN where the fit failed, did not
        improve the likelihood, or the transit is low quality
        """
        lc = tm.litecurve
        overlap = overlap[planet_no]
        obsmode = tm.transit_obsmode[planet_no]

        ttime_old = tm.planets[planet_no].transit_times.ttime_mod
        quality = tm.planets[planet_no].transit_times.quality
        ttime_new = np.full_like(ttime_old, np.nan)
        prior_sd = np.array([p.duration / 3 for p in tm.planets])
        nfail = 0

        for i, tc in enumerate(ttime_old):
            if not quality[i]:
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

            theta_start = theta.copy()
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
                (theta[planet_no] - theta_start[planet_no]) * 1440,
                logp_initial,
                logp_final,
                result.nfev,
            )
            if result.success and logp_final >= logp_initial:
                ttime_new[i] = theta[planet_no]
            else:
                nfail += 1

        if nfail > 0:
            logger.warning(
                "planet %d: optimizer failed for %d of %d transits "
                "(unmeasured unless cross-correlation succeeded)",
                planet_no,
                nfail,
                np.sum(quality),
            )

        return ttime_new

    def perform(self):
        """
        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve
        """
        star = self.star
        planets = self.planets
        litecurve = self.litecurve
        npl = len(planets)

        # fit transit times using Mazeh+13 / Holczer+16 cross-correlation method
        tm = TransitModel(star, planets, litecurve)
        overlap = self.identify_overlapping_transits(rel_size=1.0, abs_size=1 / 24)
        ttime_cc = [None] * npl
        error_cc = [None] * npl

        for n in range(npl):
            if np.any(overlap[n]):
                logger.info(
                    "planet %d: %d of %d transits overlap another planet "
                    "(fit jointly by the optimizer, not cross-correlation)",
                    n,
                    np.sum(overlap[n]),
                    len(overlap[n]),
                )

        for n, p in enumerate(planets):
            ttime_cc[n], error_cc[n] = self._cross_correlation_times(tm, n, overlap)

            assert len(ttime_cc[n]) == len(p.transit_times.ttime_mod)

            # low quality transits (FlagBadTransits) are not measured
            ttime_cc[n][~p.transit_times.quality] = np.nan
            error_cc[n][~p.transit_times.quality] = np.nan

            # discard fits with outsized errors; they fall back to the optimizer
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

        # interim transit times for the optimizer: cross-correlation measurements
        # where available, otherwise the previous model
        for n, p in enumerate(planets):
            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            tt = p.transit_times
            p.set_transit_times(
                TransitTimes(
                    tt.mission,
                    tt.target,
                    tt.index,
                    ttime_obs=np.where(q, ttime_cc[n], np.nan),
                    error_obs=np.where(q, error_cc[n], np.nan),
                    ttime_mod=np.where(q, ttime_cc[n], tt.ttime_mod),
                    quality=tt.quality,
                )
            )

        # fit transit times using least squares optimization
        tm = TransitModel(star, planets, litecurve)
        overlap = self.identify_overlapping_transits(rel_size=1.0, abs_size=1 / 24)
        ttime_opt = [None] * npl

        for n in range(npl):
            ttime_opt[n] = self._optimize_times(tm, n, overlap)

        # measured transit times, prioritizing cross-correlation over optimization;
        # no model yet (ttime_mod = ttime_obs) until RegularizeOMC, except where
        # neither method succeeded or the transit is low quality: those stay
        # unmeasured at their previous model time
        for n, p in enumerate(planets):
            good = p.transit_times.quality
            error_obs = error_cc[n]
            tc_error_pred = self._predicted_tc_error(tm, n)

            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            measured = good & (q | np.isfinite(ttime_opt[n]))
            ttime_new = np.where(q, ttime_cc[n], ttime_opt[n])
            # predicted error approximates the Cramer-Rao bound: a smaller
            # cross-correlation error is not credible (NaN stays NaN)
            error_new = np.maximum(error_obs, tc_error_pred)
            error_fill = np.nanpercentile(error_new, 90)
            error_new[~q] = error_fill
            ttime_new[~measured] = np.nan
            error_new[~measured] = np.nan
            if not np.isfinite(error_fill):
                raise RuntimeError(
                    f"planet {n}: no valid cross-correlation errors; "
                    "all transit errors are NaN"
                )

            tt = p.transit_times
            p.set_transit_times(
                TransitTimes(
                    tt.mission,
                    tt.target,
                    tt.index,
                    ttime_obs=ttime_new,
                    error_obs=error_new,
                    ttime_mod=np.where(measured, ttime_new, tt.ttime_mod),
                    quality=tt.quality,
                )
            )

            omc = p.transit_times.ttime_obs - p.transit_times.linear_ephemeris()
            logger.info(
                "planet %d: %d cross-correlation + %d optimizer times "
                "(optimizer error = %.2f min), %d unmeasured, %d low quality; "
                "P = %.6f d, epoch = %.5f BKJD, OMC rms = %.2f min, "
                "median error = %.2f min",
                n,
                np.sum(q),
                np.sum(~q & measured),
                error_fill * 1440,
                np.sum(good & ~measured),
                np.sum(~good),
                planets[n].period,
                planets[n].epoch,
                np.nanstd(omc) * 1440,
                np.nanmedian(error_new) * 1440,
            )

        return self.planets, self.litecurve
