"""Least-squares fit of a single transit shape per planet"""

import logging

import numpy as np
from scipy.optimize import least_squares

from alderaan.algorithms.base import BaseAlg
from alderaan.transit_model import (
    TRANSIT_PARAMETER_LABELS,
    TransitModel,
)
from alderaan.utils.astro import approx_transit_depth

logger = logging.getLogger(__name__)


class QuickFitTransitShape(BaseAlg):
    """
    Fit radius ratio, impact parameter, and duration; limb darkening is fixed
    at the star's values

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    def _theta_bounds(self):
        """
        theta : array-like
            npl * [C0, C1, ror, b, T14] + [q1, q2]
        """
        bounds = []
        for p in self.planets:
            bounds.append([-np.inf, np.inf])
            bounds.append([-np.inf, np.inf])
            bounds.append([1e-5, 0.99])
            bounds.append([0.0, 1.0])
            bounds.append([0.01 * p.duration, 3 * p.duration])

        bounds.append([0.0, 1.0])
        bounds.append([0.0, 1.0])

        return np.array(bounds)

    def _optimize(self, tm, niter=3):
        theta = tm.theta_initial()
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, tm):
            _x = x0.copy()
            _x[~fix] = x
            return tm.residuals(_x)

        i_fix = np.arange(len(theta), dtype=int)
        var_names = np.array(TRANSIT_PARAMETER_LABELS)

        # fix ephemeris, vary [ror, b, T14]
        fix_C0_C1 = np.zeros(len(i_fix), dtype=bool)
        fix_C0_C1[i_fix % 5 == 0] = True
        fix_C0_C1[i_fix % 5 == 1] = True

        # fix [ror, b, T14], vary ephemeris
        fix_r_b_T14 = np.zeros(len(i_fix), dtype=bool)
        fix_r_b_T14[i_fix % 5 == 2] = True
        fix_r_b_T14[i_fix % 5 == 3] = True
        fix_r_b_T14[i_fix % 5 == 4] = True

        # limb darkening fixed
        fix_C0_C1[-2:] = True
        fix_r_b_T14[-2:] = True

        # do the optimization
        logp_start = tm.ln_likelihood(theta)
        for fix in [fix_r_b_T14, fix_C0_C1] * niter:
            logp_initial = tm.ln_likelihood(theta)

            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
                bounds=bounds[~fix, :].T,
                args=[theta, fix, tm],
            )

            theta[~fix] = result.x
            logp_final = tm.ln_likelihood(theta)

            logger.debug(
                "shape fit, optimized [%s]: logp %.6g -> %.6g (%s)",
                ", ".join(var_names[~fix[:5]]),
                logp_initial,
                logp_final,
                result.message,
            )
            if not result.success:
                logger.warning(
                    "shape fit, optimizing [%s] did not converge (%s)",
                    ", ".join(var_names[~fix[:5]]),
                    result.message,
                )
            if not np.isfinite(logp_final) or logp_final <= -1e299:
                logger.warning(
                    "shape fit: non-finite likelihood after optimizing [%s]",
                    ", ".join(var_names[~fix[:5]]),
                )

        logger.info(
            "shape fit: logp %.6g -> %.6g over %d alternating iterations",
            logp_start,
            logp_final,
            niter,
        )

        return theta

    def _update_planets(self, theta):
        for n, p in enumerate(self.planets):
            ror, impact = theta[2 + n * 5], theta[3 + n * 5]
            p.set_transit_shape(
                depth=approx_transit_depth(ror, impact),
                duration=theta[4 + n * 5],
                impact=impact,
                ror=ror,
            )
            logger.info(
                "planet %d: ror = %.4f, b = %.3f, T14 = %.3f hr, depth = %.0f ppm",
                n,
                p.ror,
                p.impact,
                p.duration * 24,
                p.depth * 1e6,
            )

    def perform(self):
        """
        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve
        """
        tm = TransitModel(self.star, self.planets, self.litecurve)
        # full width = max(3 x T14, 2 hours)
        tm.trim(
            self.make_transit_mask(rel_size=3.0, abs_size=2 / 24, quality_only=True)
        )

        for obsmode in tm.unique_obsmodes:
            logger.info("supersample factor (%s): %d", obsmode, tm.supersample[obsmode])

        theta = self._optimize(tm)
        self._update_planets(theta)

        return self.planets, self.litecurve
