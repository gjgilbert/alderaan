__all__ = ["SampleTransitModel"]

import os
from copy import deepcopy

import matplotlib.pyplot as plt

from alderaan.modules.quicklook import (
    dynesty_cornerplot,
    dynesty_runplot,
    dynesty_traceplot,
)
from alderaan.modules.transit import RBDTransitModel
from alderaan.utils.io import dynesty_results_to_fits


class SampleTransitModel:
    def __init__(self, shire, mission, target, run_id, results_dir, quicklook_dir):
        self.shire = deepcopy(shire)
        self.mission = mission
        self.target = target
        self.run_id = run_id
        self.results_dir = results_dir
        self.quicklook_dir = quicklook_dir

    def perform(self):
        print("\n\nSTAGE: SAMPLE TRANSIT MODEL\n")

        koi_id = self.shire.star.koi_id

        print("Fitting for Rp/Rs, b, and T14")
        transitmodel = RBDTransitModel(
            self.shire.litecurve, self.shire.planets, self.shire.star
        )
        results = transitmodel.sample(progress_every=10)

        # save sampling results
        hdu_list = dynesty_results_to_fits(
            results, self.mission, self.target, self.run_id, self.results_dir
        )
        hdu_list.writeto(
            os.path.join(self.results_dir, f"{self.target}-results.fits"),
            overwrite=True,
        )

        # make quicklook plots
        filepath = os.path.join(self.quicklook_dir, f"{koi_id}_dynesty_runplot.png")
        dynesty_runplot(results, koi_id, filepath=filepath)

        for n in range(len(transitmodel.planets)):
            filepath = os.path.join(
                self.quicklook_dir, f"{koi_id}_dynesty_traceplot_{n:02d}.png"
            )
            dynesty_traceplot(results, koi_id, n, filepath=filepath)

            filepath = os.path.join(
                self.quicklook_dir, f"{koi_id}_dynesty_cornerplot_{n:02d}.png"
            )
            dynesty_cornerplot(results, koi_id, n, filepath=filepath)

        plt.close("all")

        return results
