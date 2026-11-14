__all__ = ["ResetLitecurves"]

import numpy as np
from astropy.stats import mad_std, sigma_clip

from alderaan.algorithms.base import BaseAlg
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.transit import TransitModel
from alderaan.utils.rbd import theta_initial
from alderaan.utils.stats import set_sigma_cutoff


class ResetLitecurves(BaseAlg):
    def __init__(self, star, planets, litecurve, config):
        super().__init__(star, planets, litecurve, config)

    def perform(self, litecurve_raw):
        print("\n\nALGORITHM: RESET LITECURVES\n")

        star = self.star
        planets = self.planets
        litecurves = self.litecurve.split_visits()

        # clip outliers based on transit model
        for lc in litecurves:
            tm = TransitModel(star, planets, lc)

            f_mod = tm.model_flux(theta_initial(star, planets))
            sigma = set_sigma_cutoff(len(lc.flux), sigma_min=5.0)

            bad = sigma_clip(
                lc.flux - f_mod, sigma_upper=sigma, sigma_lower=sigma, stdfunc=mad_std
            ).mask

            npts_initial = len(lc.time)
            for k in lc.__dict__.keys():
                if type(lc.__dict__[k]) is np.ndarray:
                    lc.__setattr__(k, lc.__dict__[k][~bad])

            print(
                f"  Quarter {lc.visit[0]} : "
                f"{npts_initial - len(lc.time)} outliers rejected"
            )

        # reset litecurves to raw data w/ outliers removed
        litecurve = LiteCurve().from_list(litecurves)
        keep = np.isin(litecurve_raw.cadno, litecurve.cadno)

        for k, v in litecurve_raw.__dict__.items():
            if type(v) is np.ndarray:
                litecurve.__setattr__(k, v[keep])

        litecurve.quality = np.ones(len(litecurve.time), dtype=bool)

        return self.star, self.planets, litecurve
