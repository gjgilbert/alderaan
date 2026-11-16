"""Sample the transit model posterior with dynesty"""

import logging
import time

import dynesty
import numpy as np
from dynesty.utils import get_print_fn_args, quantile

from alderaan.algorithms.transit_fitting.base import BaseTransitFittingAlg
from alderaan.data_models.posteriors import Posteriors
from alderaan.models.transit_model import TransitModel
from alderaan.utils.stats import loguniform_ppf, norm_ppf, uniform_ppf

logger = logging.getLogger(__name__)


class SampleTransitModel(BaseTransitFittingAlg):
    """
    Sample transit shape posteriors with dynesty

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    @staticmethod
    def _prior_transform(uniform_hypercube, tm):
        """
        Prior transform over physical ALDERAAN basis {C0, C1, ror, b, T14}...{q1,q2}

        Distributions are hard-coded to be:
            Normal on ephemeris perturbations {C0,C1}
            Log-uniform on radius ratio ror = Rp/Rs and transit duration T14
            Uniform on impact parameter b
            Uniform on quadratic limb darkening coefficients {q1,q2}

        For motivation behind this prior-parameter choice see:
            Carter+ 2008 (2008ApJ...689..499C)
            Kipping 2013 (2013MNRAS.435.2152K)
            Gilbert, MacDougall, & Petigura 2022 (2022AJ....164...92G)
            MacDougal, Gilbert, & Petigura 2023 (2023AJ....166...61M)

        Assumes that transit durations are known with reasonable confidence
        * sampled T14 will be restricted between (0.01*fixed_duration, 3*fixed_duration)

        Parameters
        ----------
        uniform_hypercube : array-like, length 5 * npl + 2
            list of parameters npl * [C0, C1, ror, b, T14] + [q1, q2]
        tm : TransitModel
            transit model; planet durations set the T14 prior limits

        Returns
        -------
        transformed_hypercube : array_like, length 5 * npl + 2
            transformed samples
        """
        u = np.asarray(uniform_hypercube)  # U(0,1) priors
        x = np.zeros_like(u)  # physical priors

        # 5 * npl + 2 parameters: [C0, C1, ror, b, T14]...[q1, q2]
        for n in range(tm.npl):
            x[5 * n + 0] = norm_ppf(u[0 + n * 5], 0.0, 0.1)
            x[5 * n + 1] = norm_ppf(u[1 + n * 5], 0.0, 0.1)
            x[5 * n + 2] = loguniform_ppf(u[2 + n * 5], 1e-5, 0.99)
            x[5 * n + 3] = uniform_ppf(u[3 + n * 5], 0.0, 1 + x[5 * n + 2])
            x[5 * n + 4] = loguniform_ppf(
                u[4 + n * 5], 0.01 * tm.planets[n].duration, 3 * tm.planets[n].duration
            )

        # limb darkening coefficients (see Kipping 2013)
        x[-2] = uniform_ppf(u[-2], 0, 1)
        x[-1] = uniform_ppf(u[-1], 0, 1)

        return x

    @staticmethod
    def _throttled_print_fn(interval=60):
        last = [0]
        start = time.time()

        def callback(results, niter, ncall, *args, **kwargs):
            now = time.time()
            if now - last[0] >= interval:
                fn_args = get_print_fn_args(results, niter, ncall, *args, **kwargs)
                logger.info(
                    "iter: %d | %s | runtime: %.1f min",
                    niter,
                    " | ".join(fn_args.long_str),
                    (now - start) / 60,
                )
                last[0] = now

        return callback

    def _sample(self, tm):
        ndim = 5 * tm.npl + 2
        logger.info(
            "dynesty DynamicNestedSampler: ndim = %d, bound = multi, sample = rwalk, "
            "default nlive/dlogz",
            ndim,
        )
        sampler = dynesty.DynamicNestedSampler(
            self._ln_likelihood,
            self._prior_transform,
            ndim,
            bound="multi",
            sample="rwalk",
            logl_args=(tm,),
            ptform_args=(tm,),
        )
        sampler.run_nested(
            print_progress=True,
            print_func=self._throttled_print_fn(interval=60),
        )

        results = sampler.results
        weights = results.importance_weights()
        logger.info(
            "dynesty finished: niter = %d, ncall = %d, eff = %.1f%%, "
            "ln Z = %.2f +/- %.2f, effective samples = %.0f",
            results.niter,
            np.sum(results.ncall),
            results.eff,
            results.logz[-1],
            results.logzerr[-1],
            1 / np.sum(weights**2),
        )

        return results

    @staticmethod
    def _log_posterior_summary(results, tm):
        weights = results.importance_weights()

        def _q(i):
            return np.asarray(
                quantile(results.samples[:, i], [0.16, 0.5, 0.84], weights=weights)
            )

        for n, p in enumerate(tm.planets):
            ror, b, T14 = (_q(5 * n + k) for k in (2, 3, 4))
            logger.info(
                "planet %d posterior (16/50/84): ror = %.4f/%.4f/%.4f, "
                "b = %.3f/%.3f/%.3f, T14 = %.3f/%.3f/%.3f hr",
                n,
                *ror,
                *b,
                *(T14 * 24),
            )
            if T14[1] < 1.05 * 0.01 * p.duration or T14[1] > 0.95 * 3 * p.duration:
                logger.warning(
                    "planet %d: posterior median T14 = %.3f hr is near the prior edge "
                    "(%.3f to %.3f hr)",
                    n,
                    T14[1] * 24,
                    0.01 * p.duration * 24,
                    3 * p.duration * 24,
                )

        q1, q2 = _q(-2), _q(-1)
        logger.info(
            "limb darkening posterior (16/50/84): q1 = %.3f/%.3f/%.3f, "
            "q2 = %.3f/%.3f/%.3f",
            *q1,
            *q2,
        )

    def perform(self):
        """
        Returns
        -------
        posteriors : Posteriors
        """
        tm = TransitModel(self.star, self.planets, self.litecurve)
        tm.trim(self.make_transit_mask(rel_size=3.0, abs_size=2 / 24))

        for obsmode in tm.unique_obsmodes:
            logger.info("supersample factor (%s): %d", obsmode, tm.supersample[obsmode])

        results = self._sample(tm)
        self._log_posterior_summary(results, tm)

        return Posteriors.from_dynesty(
            self.litecurve.mission, self.litecurve.target, results, tm
        )
