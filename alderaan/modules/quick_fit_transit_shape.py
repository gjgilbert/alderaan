__all__ = ["QuickFitTransitShape"]

from alderaan.modules.base import BaseAlg
from alderaan.modules.transit import RBDTransitModel


class QuickFitTransitShape(BaseAlg):
    def __init__(self, star, planets, litecurve, config):
        super().__init__(star, planets, litecurve, config)

    def perform(self):
        print("\n\nMODULE: QUICK FIT TRANSIT SHAPE\n")

        transitmodel = RBDTransitModel(self.star, self.planets, self.litecurve)
        transitmodel = transitmodel.trim_flux_arrays(rel_size=3.0, abs_size=1 / 24)

        print("Supersample factor")
        for obsmode in transitmodel.unique_obsmodes:
            print(f"  {obsmode} : {transitmodel.supersample[obsmode]}")

        print("\nFitting initial transit model")
        theta = transitmodel.optimize()
        self.planets = transitmodel.update_planet_parameters(theta)
        self.star = transitmodel.update_star_parameters(theta)

        return self.star, self.planets, self.litecurve
