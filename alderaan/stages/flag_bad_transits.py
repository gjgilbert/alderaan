__all__ = ["FlagBadTransits"]

import warnings
from copy import deepcopy

import numpy as np

from alderaan.modules.quality_control import QualityControl


class FlagBadTransits:
    def __init__(self, shire):
        self.shire = deepcopy(shire)

    def perform(self):
        print("\n\nSTAGE: FLAG BAD TRANSITS\n")

        planets = self.shire.planets
        qc = QualityControl(self.shire.litecurve, planets)

        # check for transits with poor photometric coverage
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", category=RuntimeWarning)
            coverage = qc.check_coverage()

        # check for transits with unusually high noise
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", category=RuntimeWarning)
            good_rms = qc.check_rms(rel_size=3.0, abs_size=2 / 24, sigma_cut=5.0)

        for n, p in enumerate(planets):
            print(f"\n  Planet {n}:")

            assert len(coverage[n]) == len(p.transit_times)
            assert len(good_rms[n]) == len(p.transit_times)

            nbad = np.sum(~coverage[n])
            ntot = len(coverage[n])
            print(
                f"    {np.sum(nbad)} of {ntot} transits ({int(100 * nbad / ntot)}%) "
                "rejected for insufficient photometric coverage"
            )

            nbad = np.sum(~good_rms[n] & coverage[n])
            ntot = len(good_rms[n])
            print(
                f"    {np.sum(nbad)} of {ntot} transits ({int(100 * nbad / ntot)}%) "
                "rejected for high photometric noise"
            )

            p.transit_quality = coverage[n] & good_rms[n]
            planets[n] = p.remove_poor_quality_transits()

        return self.shire
