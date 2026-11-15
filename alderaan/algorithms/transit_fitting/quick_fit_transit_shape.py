"""Least-squares fit of a single transit shape per planet"""

import logging

import numpy as np
from scipy.optimize import least_squares

from alderaan.algorithms.transit_fitting.base import BaseTransitFittingAlg
from alderaan.models.transit_model import TransitModel
from alderaan.utils.astro import estimate_transit_depth, ld_q_to_u

logger = logging.getLogger(__name__)


class QuickFitTransitShape(BaseTransitFittingAlg):
    """
    Fit radius ratio, impact parameter, duration, and limb darkening

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

    def _theta_bounds(self):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
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

    def _optimize(self, tm, fix_limbdark=True, niter=3):
        theta = self._theta_initial(self.star, self.planets)
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, tm):
            _x = x0.copy()
            _x[~fix] = x
            return self._model_residuals(_x, tm)

        i_fix = np.arange(len(theta), dtype=int)
        var_names = np.array("C0 C1 r b T14".split())

        # fix ephemeris, vary [r, b, T14]
        fix_C0_C1 = np.zeros(len(i_fix), dtype=bool)
        fix_C0_C1[i_fix % 5 == 0] = True
        fix_C0_C1[i_fix % 5 == 1] = True

        # fix [r, b, T14], vary ephemeris
        fix_r_b_T14 = np.zeros(len(i_fix), dtype=bool)
        fix_r_b_T14[i_fix % 5 == 2] = True
        fix_r_b_T14[i_fix % 5 == 3] = True
        fix_r_b_T14[i_fix % 5 == 4] = True

        # limb darkening
        fix_C0_C1[-2:] = fix_limbdark
        fix_r_b_T14[-2:] = fix_limbdark

        # do the optimization
        logp_start = self._ln_likelihood(theta, tm)
        for fix in [fix_r_b_T14, fix_C0_C1] * niter:
            logp_initial = self._ln_likelihood(theta, tm)

            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
                bounds=bounds[~fix, :].T,
                args=[theta, fix, tm],
            )

            theta[~fix] = result.x
            logp_final = self._ln_likelihood(theta, tm)

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
                ror, impact, theta[4 + n * 5], estimate_transit_depth(ror, impact)
            )
            logger.info(
                "planet %d: ror = %.4f, b = %.3f, T14 = %.3f hr, depth = %.0f ppm",
                n,
                p.ror,
                p.impact,
                p.duration * 24,
                p.depth * 1e6,
            )

    def _update_star(self, theta):
        # sampling basis (q1, q2) --> physical (u1, u2), see Kipping 2013
        q1, q2 = theta[-2], theta[-1]
        self.star.u1, self.star.u2 = ld_q_to_u(q1, q2)
        logger.info(
            "limb darkening: q1 = %.3f, q2 = %.3f (u1 = %.3f, u2 = %.3f)",
            q1,
            q2,
            self.star.u1,
            self.star.u2,
        )

    def perform(self):
        """
        Returns
        -------
        star : Star
        planets : list of Planet
        litecurve : LiteCurve
        """
        tm = TransitModel(self.star, self.planets, self.litecurve)
        tm.trim(
            self.make_transit_mask(rel_size=3.0, abs_size=1 / 24, mask_type="condensed")
        )

        for obsmode in tm.unique_obsmodes:
            logger.info("supersample factor (%s): %d", obsmode, tm.supersample[obsmode])

        theta = self._optimize(tm)
        self._update_planets(theta)
        self._update_star(theta)

        return self.star, self.planets, self.litecurve
