__all__ = ["RegularizeOMC"]

import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std

from alderaan.modules.omc import OMC
from alderaan.utils.io import save_omc_ephemeris
from alderaan.utils.plotting import plot_omc
from alderaan.utils.stats import set_sigma_cutoff


class RegularizeOMC:
    def __init__(self, shire, target, results_dir, quicklook_dir, check_periodic=True):
        self.shire = deepcopy(shire)
        self.target = target
        self.results_dir = results_dir
        self.quicklook_dir = quicklook_dir
        self.check_periodic = check_periodic

    def _identify_periodic_signals(self, omc_list):
        planets = self.shire.planets
        npl = len(planets)

        print("searching for periodic signals")
        if npl == 1:
            critical_fap = 0.1
        elif npl > 1:
            critical_fap = 0.99

        freq = [None] * npl
        fap = [None] * npl

        for n in range(npl):
            freq[n], fap[n] = omc_list[n].identify_significant_frequencies(critical_fap)

        # single planet systems : use direct search output (FAP < 0.1)
        # multiplanet systems : compare marginal frequencies to find close matches
        if npl > 1:
            close = np.zeros((npl, npl), dtype=bool)

            for i in range(npl):
                df_min = 1 / (
                    planets[i].transit_times.max() - planets[i].transit_times.min()
                )

                for j in range(i + 1, npl):
                    if freq[i] is not None and freq[j] is not None:
                        df_ij = np.abs(freq[i] - freq[j])

                        if df_ij < df_min:
                            close[i, j] = True
                            close[j, i] = True

            for i in range(npl):
                if fap[i] is not None:
                    if not np.any(close[i]) and fap[i] > 0.1:
                        freq[i] = None
                        fap[i] = None

        for n in range(npl):
            if freq[n] is not None:
                print(
                    f"  Planet {n} : periodic signal found at "
                    f"P = {1 / freq[n]:.1f} days"
                )
            else:
                print(f"  Planet {n} : no significant periodic component found")

        return freq

    def perform(self):
        print("\n\nMODULE: REGULARIZE OMC\n")

        planets = self.shire.planets
        npl = len(planets)

        omc_list = [OMC(p) for p in planets]
        ttv_buffer = [None] * npl

        if self.check_periodic:
            freq = self._identify_periodic_signals(omc_list)

        # fit a regularized model
        for n, p in enumerate(planets):
            omc = omc_list[n]
            npts = np.sum(omc.quality)
            old_period = p.period

            print(f"\nperforming omc model selection for Planet {n} ({npts} transits)")

            traces = {}
            dofs = {}

            # Matern-3/2 model | don't use GP on very noisy data
            if (npts >= 8) & (np.median(omc.yerr) <= 0.5 * mad_std(omc.yobs)):
                print("\nmatern-3/2 model")
                with warnings.catch_warnings(record=True) as catch:
                    warnings.simplefilter("always", category=RuntimeWarning)
                    traces["matern32"] = omc.sample(omc.matern32_model())
                    dofs["matern32"] = np.nanmedian(traces["matern32"]["dof"])

            # Polynomial model | require 2^N transits, at most do 3rd order
            max_polyorder = np.max([1, np.min([3, int(np.log2(npts)) - 1])])
            for polyorder in range(1, max_polyorder + 1):
                print(f"\npolynomial (degree {polyorder}) model")
                with warnings.catch_warnings(record=True) as catch:
                    warnings.simplefilter("always", category=RuntimeWarning)
                    traces[f"poly_{polyorder}"] = omc.sample(omc.poly_model(polyorder))
                    dofs[f"poly_{polyorder}"] = 1 + polyorder

            # Sinusoidal model
            if self.check_periodic and npts >= 8 and freq[n] is not None:
                print("\nsinusoid model")
                with warnings.catch_warnings(record=True) as catch:
                    warnings.simplefilter("always", category=RuntimeWarning)
                    traces["sinusoid"] = omc.sample(
                        omc.sin_model(ttv_period=1 / freq[n])
                    )
                    dofs["sinusoid"] = 3

            if len(catch) > 0:
                print(f"{len(catch)} RuntimeWarnings caught during sampling")

            # perform model selection routine and update objects
            omc.ymod, best_omc_model_name = omc.select_best_model(traces, dofs)
            print(
                f"\nPlanet {n} : adopting {best_omc_model_name} as omc model "
                "and flagging outliers"
            )

            omc.out_prob, omc.out_class = omc.calculate_outlier_probability(omc.ymod)

            # save transit times
            filename = os.path.join(
                self.results_dir, f"{self.target}_{n:02d}_quick.ttvs"
            )
            save_omc_ephemeris(filename, omc)

            planets[n] = p.update_ephemeris(
                omc.index, omc.ymod, omc.yerr, omc.quality, full=False
            )
            print(f"Planet {n} : {old_period:.6f} --> {p.period:.6f}")

            ttv_scatter = mad_std(omc.yobs - omc.ymod)
            ttv_buffer[n] = set_sigma_cutoff(len(omc.yobs)) * ttv_scatter

        filepath = os.path.join(self.quicklook_dir, f"{self.target}_omc.png")
        plot_omc(omc_list, self.target, filepath)
        plt.close("all")

        return self.shire, ttv_buffer
