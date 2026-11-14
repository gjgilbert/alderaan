__all__ = ["SampleTransitModel"]

import os

import matplotlib.pyplot as plt

from alderaan.modules.base import BaseAlg
from alderaan.modules.transit import RBDTransitModel
from alderaan.utils.io import dynesty_results_to_fits
from alderaan.utils.plotting import (
    dynesty_cornerplot,
    dynesty_runplot,
    dynesty_traceplot,
)


class SampleTransitModel(BaseAlg):
    def __init__(self, star, planets, litecurve, config):
        super().__init__(star, planets, litecurve, config)

    def perform(self):
        print("\n\nMODULE: SAMPLE TRANSIT MODEL\n")

        mission = self.config["RUN"]["mission"]
        target = self.config["RUN"]["target"]
        run_id = self.config["RUN"]["run_id"]
        results_dir = self.config["PATHS"]["results_dir"]
        quicklook_dir = self.config["PATHS"]["quicklook_dir"]

        koi_id = self.star.koi_id

        print("Fitting for Rp/Rs, b, and T14")
        transitmodel = RBDTransitModel(self.star, self.planets, self.litecurve)
        results = transitmodel.sample(progress_every=10)

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

        for n in range(len(transitmodel.planets)):
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
