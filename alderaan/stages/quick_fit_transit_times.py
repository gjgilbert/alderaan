__all__ = ["QuickFitTransitTimes"]

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from alderaan.modules.transit import CrossCorrelationTTVFitter, OptimizationTTVFitter
from alderaan.utils.astro import predict_tc_error


class QuickFitTransitTimes:
    def __init__(self, shire, target, quicklook_dir):
        self.shire = deepcopy(shire)
        self.target = target
        self.quicklook_dir = quicklook_dir

    def perform(self):
        print("\n\nSTAGE: QUICK FIT TRANSIT TIMES\n")

        star = self.shire.star
        planets = self.shire.planets
        litecurve = self.shire.litecurve
        npl = len(planets)

        # fit transit times using Mazeh+13 / Holczer+16 cross-correlation method
        print("\nApplying cross-correlation method")
        ttvmodel = CrossCorrelationTTVFitter(litecurve, planets, star)
        ttime_cc = [None] * npl
        error_cc = [None] * npl

        for n, p in enumerate(planets):
            print(f"\nPlanet {n} : fitting {np.sum(p.transit_quality)} transit times")

            ttime_cc[n], error_cc[n] = ttvmodel.mazeh13_holczer16_method(
                n, target=self.target, quicklook_dir=self.quicklook_dir
            )

            assert len(ttime_cc[n]) == len(p.transit_times)
            assert len(error_cc[n]) == len(p.transit_errors)

            nfit = np.sum(~np.isnan(ttime_cc[n]))
            ntot = np.sum(p.transit_quality)

            print(
                f"Planet {n} : {nfit} of {ntot} transit times "
                f"({nfit / ntot * 100:.1f}%) fit successfully"
            )

        # update ephemerides
        for n, p in enumerate(planets):
            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            p.transit_times[q] = ttime_cc[n][q]
            p.transit_errors[q] = error_cc[n][q]
            planets[n] = p.refit_period_and_epoch()

        # fit transit times using least squares optimization
        print("\nOptimizing independent transit times")
        ttvmodel = OptimizationTTVFitter(litecurve, planets, star)
        ttime_opt = [None] * npl

        for n in range(npl):
            print(f"\nPlanet {n}")
            ttime_opt[n] = ttvmodel.optimize(n)

        # update ephemerides, prioritizing cross-correlation transit times
        for n, p in enumerate(planets):
            error_obs = np.copy(error_cc[n])
            error_exp = np.zeros_like(error_obs)

            for obsmode in ttvmodel.unique_obsmodes:
                exptime = ttvmodel.exptime[obsmode]
                sigma_f = np.nanmedian(litecurve.error[litecurve.obsmode == obsmode])
                error_exp[ttvmodel.transit_obsmode[n] == obsmode] = predict_tc_error(
                    p.ror, p.impact, p.duration, exptime, sigma_f
                )

            q = ~np.isnan(ttime_cc[n]) & ~np.isnan(error_cc[n])
            ttime_new = np.copy(ttime_cc[n])
            ttime_new[~q] = ttime_opt[n][~q]
            error_new = np.sqrt(error_obs**2 + error_exp**2)
            error_new[~q] = np.nanpercentile(error_new, 90)

            p.transit_times = ttime_new
            p.transit_errors = error_new
            planets[n] = p.refit_period_and_epoch()

        plt.close("all")

        return self.shire
