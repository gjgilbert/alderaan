"""Fit individual transit times by cross-correlation and optimization"""

import logging

import numpy as np
from scipy.optimize import least_squares

from alderaan.algorithms.transit_fitting.base import BaseTransitFittingAlg
from alderaan.data_models.transit_times import TransitTimes
from alderaan.models.transit_model import TransitModel
from alderaan.utils.astro import bin_data, predict_tc_error
from alderaan.utils.stats import robust_polyfit

logger = logging.getLogger(__name__)


class QuickFitTransitTimes(BaseTransitFittingAlg):
    """
    Measure individual transit times for each planet

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    config : ConfigParser
        pipeline configuration, including RUN and PATHS values set by main
    """

    def __init__(self, star, planets, litecurve, config):
        super().__init__(star, planets, litecurve, config)

    @staticmethod
    def _construct_template(tm, planet_no, obsmode, transit_window_size):
        """
        planet_no (int) : planet number index

        IMPORTANT!!! Planet extensions must have desired [P, Rp/Rs, b, T14]
        """
        # shorthand
        p = tm.planets[planet_no]

        exptime = tm.exptime[obsmode]
        supersample = tm.supersample[obsmode]

        time_template = np.arange(
            0, transit_window_size / 2, exptime / supersample / 2, dtype=float
        )
        time_template = np.hstack([-time_template[:-1][::-1], time_template])

        flux_template = tm.transit_flux(
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
        """
        # shorthand
        lc = tm.litecurve
        p = tm.planets[planet_no]

        ttime = np.nan * np.ones(len(p.transit_times.ttime_mod))
        ttime_err = np.nan * np.ones(len(p.transit_times.ttime_mod))

        assert len(p.transit_times.ttime_mod) == len(overlap[planet_no]), (
            f"Mismatched sizes for ttime ({len(p.transit_times.ttime_mod)}) "
            f"and overlap ({len(overlap[planet_no])})"
        )

        transit_obsmode = tm.transit_obsmode[planet_no]
        ntransit = len(p.transit_times.ttime_mod)
        skipped = {
            "overlapping": int(np.sum(overlap[planet_no])),
            "no in-transit data": 0,
            "too few chisq points": 0,
            "ill-conditioned": 0,
        }

        for obsmode in tm.unique_obsmodes:
            exptime = tm.exptime[obsmode]
            supersample = tm.supersample[obsmode]
            exptime_ioff = tm.exptime_offsets[obsmode]

            transit_window_size = rel_window_size * p.duration + abs_window_size_buffer
            time_template, flux_template = self._construct_template(
                tm, planet_no, obsmode, transit_window_size
            )

            # primary loop for fitting transit times
            for i, tc in enumerate(p.transit_times.ttime_mod):
                if (not overlap[planet_no][i]) and (transit_obsmode[i] == obsmode):
                    # STEP 0: pull data near a single transit
                    in_transit = np.abs(lc.time - tc) < p.duration / 2
                    in_window = np.abs(lc.time - tc) < transit_window_size / 2

                    if np.sum(in_transit) == 0:
                        skipped["no in-transit data"] += 1
                    else:
                        _t = lc.time[in_window]
                        _f_obs = lc.flux[in_window]
                        _f_err = lc.error[in_window]

                        _t_supersample = (exptime_ioff + _t.reshape(_t.size, 1)).ravel()

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
                        if len(tc_fit) <= 3:
                            skipped["too few chisq points"] += 1
                        else:
                            # polynomial fitting
                            quad_coeffs = robust_polyfit(tc_fit, x2_fit, 2)
                            quad_model = np.polyval(quad_coeffs, tc_fit)
                            qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])

                            # transit time
                            _tti = np.nanmean([qtc_min, np.mean(tc_fit)])

                            # check that the fit is well-conditioned
                            convex_local_min = quad_coeffs[0] > 0
                            within_bounds = (_tti > tc_fit.min()) and (
                                _tti < tc_fit.max()
                            )

                            if convex_local_min and within_bounds:
                                # scaled error; sqrt is real only for a convex fit
                                qtc_err = np.sqrt(1 / quad_coeffs[0])
                                ttime[i] = _tti
                                ttime_err[i] = qtc_err * (
                                    1 + np.std(x2_fit - quad_model)
                                )
                            else:
                                skipped["ill-conditioned"] += 1

        nfit = int(np.sum(~np.isnan(ttime)))
        logger.info(
            "planet %d: cross-correlation fit %d of %d transits (skipped: %s)",
            planet_no,
            nfit,
            ntransit,
            ", ".join(f"{v} {k}" for k, v in skipped.items()),
        )
        if nfit < 0.5 * ntransit:
            logger.warning(
                "planet %d: only %d of %d transits fit by cross-correlation",
                planet_no,
                nfit,
                ntransit,
            )

        return ttime, ttime_err

    @staticmethod
    def _window_flux(theta, tm, obsmode, t_obs):
        f_mod = np.ones_like(t_obs)

        for n, p in enumerate(tm.planets):
            qld_flux = tm.transit_flux(
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
        """
        lc = tm.litecurve
        overlap = overlap[planet_no]
        obsmode = tm.transit_obsmode[planet_no]

        ttime_old = tm.planets[planet_no].transit_times.ttime_mod
        ttime_new = np.zeros_like(ttime_old)
        nfail = 0

        for i, tc in enumerate(ttime_old):
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
                return self._window_residuals(_x, tm, obsmode, t_obs, f_obs, f_err)

            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
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
            ttime_new[i] = theta[planet_no]
            logp_final = self._window_ln_likelihood(
                theta, tm, obsmode[i], t_obs, f_obs, f_err
            )

            logger.debug(
                "planet %d transit %d: tc %.5f -> %.5f d (%+.2f min), "
                "logp %.6g -> %.6g, nfev = %d",
                planet_no,
                i,
                tc,
                ttime_new[i],
                (ttime_new[i] - tc) * 1440,
                logp_initial,
                logp_final,
                result.nfev,
            )
            if not result.success or logp_final < logp_initial:
                nfail += 1

        if nfail > 0:
            logger.warning(
                "planet %d: optimizer failed or decreased logp for %d of %d transits",
                planet_no,
                nfail,
                len(ttime_old),
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
        overlap = self.identify_overlapping_transits(rtol=1.0, atol=1 / 24)
        ttime_cc = [None] * npl
        error_cc = [None] * npl

        for n in range(npl):
            if np.any(overlap[n]):
                logger.info(
                    "planet %d: %d of %d transits overlap another planet "
                    "(excluded from cross-correlation, fit jointly in optimization)",
                    n,
                    np.sum(overlap[n]),
                    len(overlap[n]),
                )

        for n, p in enumerate(planets):
            ttime_cc[n], error_cc[n] = self._cross_correlation_times(tm, n, overlap)

            assert len(ttime_cc[n]) == len(p.transit_times.ttime_mod)

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
                )
            )

        # fit transit times using least squares optimization
        tm = TransitModel(star, planets, litecurve)
        overlap = self.identify_overlapping_transits(rtol=1.0, atol=1 / 24)
        ttime_opt = [None] * npl

        for n in range(npl):
            ttime_opt[n] = self._optimize_times(tm, n, overlap)

        # measured transit times, prioritizing cross-correlation over optimization;
        # no model yet (ttime_mod = ttime_obs) until RegularizeOMC
        for n, p in enumerate(planets):
            error_obs = error_cc[n]
            error_exp = np.zeros_like(error_obs)

            for obsmode in tm.unique_obsmodes:
                exptime = tm.exptime[obsmode]
                sigma_f = np.nanmedian(litecurve.error[litecurve.obsmode == obsmode])
                error_exp[tm.transit_obsmode[n] == obsmode] = predict_tc_error(
                    p.ror, p.impact, p.duration, exptime, sigma_f
                )

            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            ttime_new = np.where(q, ttime_cc[n], ttime_opt[n])
            error_new = np.sqrt(error_obs**2 + error_exp**2)
            error_fill = np.nanpercentile(error_new, 90)
            error_new[~q] = error_fill
            if not np.isfinite(error_fill):
                logger.error(
                    "planet %d: no valid cross-correlation errors; "
                    "all transit errors are NaN",
                    n,
                )

            tt = p.transit_times
            p.set_transit_times(
                TransitTimes(
                    tt.mission,
                    tt.target,
                    tt.index,
                    ttime_obs=ttime_new,
                    error_obs=error_new,
                )
            )

            omc = p.transit_times.ttime_obs - p.transit_times.linear_ephemeris()
            logger.info(
                "planet %d: %d cross-correlation + %d optimized transit times "
                "(optimized assigned 90th-percentile error %.2f min); "
                "P = %.6f d, epoch = %.5f BKJD, OMC rms = %.2f min, "
                "median error = %.2f min",
                n,
                np.sum(q),
                np.sum(~q),
                error_fill * 1440,
                planets[n].period,
                planets[n].epoch,
                np.std(omc) * 1440,
                np.median(error_new) * 1440,
            )

        return self.planets, self.litecurve
