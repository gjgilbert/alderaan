__all__ = ["ResetLitecurves"]

from copy import deepcopy

import numpy as np
from astropy.stats import mad_std, sigma_clip

from alderaan.data_models.litecurve import LiteCurve
from alderaan.modules.transit import RBDTransitModel
from alderaan.utils.stats import set_sigma_cutoff


class ResetLitecurves:
    def __init__(self, shire, litecurve_raw):
        self.shire = deepcopy(shire)
        self.litecurve_raw = deepcopy(litecurve_raw)

    def perform(self):
        print("\n\nMODULE: RESET LITECURVES\n")

        star = self.shire.star
        planets = self.shire.planets
        litecurves = self.shire.litecurve.split_visits()

        # clip outliers based on transit model
        for lc in litecurves:
            tm = RBDTransitModel(lc, planets, star)

            theta = tm._theta_initial()
            f_mod = tm.model_flux(theta, tm)
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
        keep = np.isin(self.litecurve_raw.cadno, litecurve.cadno)

        for k, v in self.litecurve_raw.__dict__.items():
            if type(v) is np.ndarray:
                litecurve.__setattr__(k, v[keep])

        litecurve.quality = np.ones(len(litecurve.time), dtype=bool)

        self.shire.litecurve = litecurve

        return self.shire
