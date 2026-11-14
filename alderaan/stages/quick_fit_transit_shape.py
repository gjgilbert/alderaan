__all__ = ["QuickFitTransitShape"]

from copy import deepcopy

from alderaan.modules.transit import RBDTransitModel


class QuickFitTransitShape:
    def __init__(self, shire):
        self.shire = deepcopy(shire)

    def perform(self):
        print("\n\nSTAGE: QUICK FIT TRANSIT SHAPE\n")

        transitmodel = RBDTransitModel(
            self.shire.litecurve, self.shire.planets, self.shire.star
        )
        transitmodel = transitmodel.trim_flux_arrays(rel_size=3.0, abs_size=1 / 24)

        print("Supersample factor")
        for obsmode in transitmodel.unique_obsmodes:
            print(f"  {obsmode} : {transitmodel.supersample[obsmode]}")

        print("\nFitting initial transit model")
        theta = transitmodel.optimize()
        self.shire.planets = transitmodel.update_planet_parameters(theta)
        self.shire.star = transitmodel.update_star_parameters(theta)

        return self.shire
