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
    Fit ror, b, and T14 of every planet, with limb darkening fixed at the star's

    The ephemeris perturbation (C0, C1) is fit alternately with the shape and then
    discarded. The fit starts from b = 0.38, 0.71, 0.87 (all planets at once),
    with ror scaled to preserve each planet's limb-darkened depth, and keeps the
    maximum likelihood result.
    """

    def _theta_start(self, tm, b):
        """
        Starting theta with every planet at impact parameter b

        ror is scaled so the limb-darkened depth ror**2 * I(mu) / <I> matches
        the depth of the planet's current (ror, b), with mu = sqrt(1 - b**2)
        and I(1) = 1
        """
        u1, u2 = self.star.u1, self.star.u2
        mu = np.sqrt(1 - b**2)
        ld_scale = (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / (1 - u1 / 3 - u2 / 6)

        theta = tm.theta_initial()
        for n, p in enumerate(self.planets):
            depth = approx_transit_depth(p.ror, p.impact)
            theta[5 * n + 2] = np.sqrt(depth / ld_scale)
            theta[5 * n + 3] = b

        return theta

    def _theta_bounds(self):
        """Bounds on theta = npl * [C0, C1, ror, b, T14] + [q1, q2]"""
        bounds = []
        for p in self.planets:
            bounds.append([-np.inf, np.inf])
            bounds.append([-np.inf, np.inf])
            bounds.append([1e-5, 0.99])
            bounds.append([0.0, 1.0])
            bounds.append([0.01 * p.duration, min(3 * p.duration, 0.49 * p.period)])

        bounds.append([0.0, 1.0])
        bounds.append([0.0, 1.0])

        return np.array(bounds)

    def _optimize(self, tm, theta, niter=3):
        """Least squares on [C0, C1], then [ror, b, T14], niter times; LD fixed"""
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, tm):
            """Residuals of x0 with its free entries (~fix) set to x"""
            _x = x0.copy()
            _x[~fix] = x
            return tm.residuals(_x)

        i_fix = np.arange(len(theta), dtype=int)
        var_names = np.array(TRANSIT_PARAMETER_LABELS)

        fix_C0_C1 = np.zeros(len(i_fix), dtype=bool)
        fix_C0_C1[i_fix % 5 == 0] = True
        fix_C0_C1[i_fix % 5 == 1] = True

        fix_r_b_T14 = np.zeros(len(i_fix), dtype=bool)
        fix_r_b_T14[i_fix % 5 == 2] = True
        fix_r_b_T14[i_fix % 5 == 3] = True
        fix_r_b_T14[i_fix % 5 == 4] = True

        # limb darkening fixed
        fix_C0_C1[-2:] = True
        fix_r_b_T14[-2:] = True

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
        """Set each planet's ror, b, and T14 from theta"""
        for n, p in enumerate(self.planets):
            p.set_transit_shape(*theta[2 + n * 5 : 5 + n * 5])
            logger.info(
                "planet %d: ror = %.4f, b = %.3f, T14 = %.3f hr",
                n,
                p.ror,
                p.impact,
                p.duration * 24,
            )

    def perform(self):
        """
        Fit the shape to good-photometry transits; set each planet's ror, b, T14

        Uses cadences within max(3 x T14, 2 hours) of transits with phot_coverage
        and phot_noise. Raises ValueError if no start reaches a finite likelihood.
        """
        tm = TransitModel(self.star, self.planets, self.litecurve)
        tm.trim(self.make_transit_mask(rel_size=3.0, abs_size=2 / 24, phot_only=True))

        best_logp, best_theta = -np.inf, None
        # b = sin(22.5 deg), sin(45 deg), sin(60 deg)
        for b in (0.38, 0.71, 0.87):
            logger.info("shape fit from b = %.2f", b)
            theta = self._optimize(tm, self._theta_start(tm, b))
            logp = tm.ln_likelihood(theta)
            # ln_likelihood returns -1e300 if not finite
            if logp > max(best_logp, -1e299):
                best_logp, best_theta = logp, theta

        if best_theta is None:
            raise ValueError("shape fit: no start reached a finite likelihood")
        self._update_planets(best_theta)

        return self.planets, self.litecurve
