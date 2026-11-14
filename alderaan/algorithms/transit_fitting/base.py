"""Base class for algorithms that fit the transit model"""

import numpy as np

from alderaan.algorithms.base import BaseAlg


class BaseTransitFittingAlg(BaseAlg):
    """
    Shared RBD transit model parameterization and likelihood

    theta = num_planets * [C0, C1, rp, b, T14] + [q1, q2]

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    config : ConfigParser
        pipeline configuration, including RUN and PATHS values set by main
    """

    @staticmethod
    def _theta_initial(star, planets):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        """
        theta = []
        for n, p in enumerate(planets):
            if not p.depth > 0:
                raise ValueError(
                    f"planet {n}: depth = {p.depth} must be positive to initialize ror"
                )
            theta.append(0.0)
            theta.append(0.0)
            theta.append(np.sqrt(p.depth))
            theta.append(0.5)
            theta.append(p.duration)

        # physical (u1, u2) --> sampling basis (q1, q2), see Kipping 2013
        u1, u2 = star.u1, star.u2
        theta.append((u1 + u2) ** 2)
        theta.append(u1 / (2 * (u1 + u2)))

        return np.array(theta)

    @staticmethod
    def _model_residuals(theta, tm):
        flux_mod = tm.model_flux(theta)
        flux_obs = tm.litecurve.flux
        flux_err = tm.litecurve.error

        return (flux_obs - flux_mod) / flux_err

    @staticmethod
    def _ln_likelihood(theta, tm):
        # calculate likelihood
        lnlike = -0.5 * np.sum(BaseTransitFittingAlg._model_residuals(theta, tm) ** 2)

        # enforce prior on physical limb darkening (see Kipping 2013)
        q1, q2 = theta[-2], theta[-1]
        u1 = 2 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2 * q2)

        sig_ld_sq = 0.01
        lnlike -= 1.0 / (2 * sig_ld_sq) * (u1 - tm.star.u1) ** 2
        lnlike -= 1.0 / (2 * sig_ld_sq) * (u2 - tm.star.u2) ** 2

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike
