__all__ = ["QuickFitTransitShape"]

import numpy as np
from scipy.optimize import least_squares

from alderaan.algorithms.base import BaseAlg
from alderaan.data_models.transit import TransitModel
from alderaan.utils.astro import estimate_transit_depth
from alderaan.utils.rbd import ln_likelihood, model_residuals, theta_initial


class QuickFitTransitShape(BaseAlg):
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
        theta = theta_initial(self.star, self.planets)
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, tm):
            _x = x0.copy()
            _x[~fix] = x
            return model_residuals(_x, tm)

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
        for fix in [fix_r_b_T14, fix_C0_C1] * niter:
            print(f"optimizing logp for variables: [{', '.join(var_names[~fix[:5]])}]")

            logp_initial = ln_likelihood(theta, tm).copy()

            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
                bounds=bounds[~fix, :].T,
                args=[theta, fix, tm],
            )

            theta[~fix] = result.x.copy()
            logp_final = ln_likelihood(theta, tm).copy()

            print(f"logp: {logp_initial} -> {logp_final}")
            print(f"message: {result['message']}")

        return theta

    def _update_planets(self, theta):
        for n, p in enumerate(self.planets):
            self.planets[n].ror = theta[2 + n * 5]
            self.planets[n].impact = theta[3 + n * 5]
            self.planets[n].duration = theta[4 + n * 5]
            self.planets[n].depth = estimate_transit_depth(p.ror, p.impact)

    def _update_star(self, theta):
        # sampling basis (q1, q2) --> physical (u1, u2), see Kipping 2013
        q1, q2 = theta[-2], theta[-1]
        self.star.u1 = 2 * np.sqrt(q1) * q2
        self.star.u2 = np.sqrt(q1) * (1 - 2 * q2)

    def perform(self):
        print("\n\nALGORITHM: QUICK FIT TRANSIT SHAPE\n")

        tm = TransitModel(self.star, self.planets, self.litecurve)
        tm.trim(
            self.make_transit_mask(rel_size=3.0, abs_size=1 / 24, mask_type="condensed")
        )

        print("Supersample factor")
        for obsmode in tm.unique_obsmodes:
            print(f"  {obsmode} : {tm.supersample[obsmode]}")

        print("\nFitting initial transit model")
        theta = self._optimize(tm)
        self._update_planets(theta)
        self._update_star(theta)

        return self.star, self.planets, self.litecurve
