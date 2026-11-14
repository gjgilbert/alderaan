__all__ = ["CleanupOMC"]

import os
from copy import deepcopy

import matplotlib.pyplot as plt

from alderaan.modules.omc import OMC
from alderaan.modules.quicklook import plot_omc


class CleanupOMC:
    def __init__(self, shire, target, quicklook_dir, full=True):
        self.shire = deepcopy(shire)
        self.target = target
        self.quicklook_dir = quicklook_dir
        self.full = full

    def perform(self):
        print("\n\nSTAGE: CLEANUP OMC\n")

        planets = self.shire.planets

        # flag outliers and update each ephemeris
        omc_list = [None] * len(planets)

        for n, p in enumerate(planets):
            omc = OMC(p)
            omc.ymod = omc.quick_fit_trend()

            old_period = p.period
            planets[n] = p.update_ephemeris(
                omc.index, omc.ymod, omc.yerr, omc.quality, full=self.full
            )
            print(f"Planet {n} : {old_period:.6f} --> {p.period:.6f}")

            omc_list[n] = omc

        filepath = os.path.join(self.quicklook_dir, f"{self.target}_omc_initial.png")
        plot_omc(omc_list, self.target, filepath)
        plt.close("all")

        return self.shire
