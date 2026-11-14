"""Sample the transit model posterior with dynesty"""

import io
import os
import sys
import time

import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import print_fn

from alderaan.algorithms.transit_fitting.base import BaseTransitFittingAlg
from alderaan.data_models.transit import TransitModel
from alderaan.utils.io import dynesty_results_to_fits
from alderaan.utils.plotting import (
    dynesty_cornerplot,
    dynesty_runplot,
    dynesty_traceplot,
)
from alderaan.utils.stats import loguniform_ppf, norm_ppf, uniform_ppf


class SampleTransitModel(BaseTransitFittingAlg):
    """
    Sample transit shape posteriors and write results and quicklook plots

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
    def _prior_transform(uniform_hypercube, tm):
        """
        Prior transform over physical ALDERAAN basis {C0, C1, r, b, T14}...{q1,q2}

        Distributions are hard-coded to be:
            Normal on ephemeris perturbations {C0,C1}
            Log-uniform on radius ratio r and transit duration T14
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
            uniform_hypercube : array-like, length 5 * N + 2
                list of parameters N * [C0, C1, r, b, T14] + [q1, q2]
            tm : TransitModel
                transit model; planet durations set the T14 prior limits

        Returns
        -------
            transformed_hypercube : array_like, length 5 * N + 2
                transformed samples
        """
        u = np.array(uniform_hypercube)  # U(0,1) priors
        x = np.zeros_like(u)  # physical priors

        # 5 * npl + 2 parameters: [C0, C1, r, b, T14]...[q1, q2]
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
    def _throttled_print_fn(interval=10):
        last = [0]
        start = time.time()

        def callback(results, niter, ncall, *args, **kwargs):
            now = time.time()
            if now - last[0] >= interval:
                runtime = (now - start) / 60

                buf = io.StringIO()
                stdout = sys.stdout
                sys.stdout = buf
                print_fn(results, niter, ncall, *args, **kwargs)
                sys.stdout = stdout

                line = buf.getvalue().rstrip()
                print(f"{line} | runtime: {runtime:.1f} min")
                last[0] = now

        return callback

    def _sample(self, tm, checkpoint_file=None, checkpoint_every=60, progress_every=10):
        ndim = 5 * tm.npl + 2
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
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
            print_progress=True,
            print_func=self._throttled_print_fn(progress_every),
        )

        return sampler.results

    def perform(self):
        """
        Returns
        -------
        results : dynesty.results.Results
        """
        print("\n\nALGORITHM: SAMPLE TRANSIT MODEL\n")

        mission = self.config["RUN"]["mission"]
        target = self.config["RUN"]["target"]
        run_id = self.config["RUN"]["run_id"]
        results_dir = self.config["PATHS"]["results_dir"]
        quicklook_dir = self.config["PATHS"]["quicklook_dir"]

        koi_id = self.star.koi_id

        tm = TransitModel(self.star, self.planets, self.litecurve)
        tm.trim(
            self.make_transit_mask(rel_size=3.0, abs_size=2 / 24, mask_type="condensed")
        )

        print("Supersample factor")
        for obsmode in tm.unique_obsmodes:
            print(f"  {obsmode} : {tm.supersample[obsmode]}")

        print("\nFitting for Rp/Rs, b, and T14")
        results = self._sample(tm, progress_every=10)

        # save sampling results
        hdu_list = dynesty_results_to_fits(
            results, mission, target, run_id, results_dir
        )
        hdu_list.writeto(
            os.path.join(results_dir, f"{target}-results.fits"),
            overwrite=True,
        )

        # make quicklook plots
        filepath = os.path.join(quicklook_dir, f"{koi_id}_dynesty_runplot.png")
        dynesty_runplot(results, koi_id, filepath=filepath)

        for n in range(tm.npl):
            filepath = os.path.join(
                quicklook_dir, f"{koi_id}_dynesty_traceplot_{n:02d}.png"
            )
            dynesty_traceplot(results, koi_id, n, filepath=filepath)

            filepath = os.path.join(
                quicklook_dir, f"{koi_id}_dynesty_cornerplot_{n:02d}.png"
            )
            dynesty_cornerplot(results, koi_id, n, filepath=filepath)

        plt.close("all")

        return results
