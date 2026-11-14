__all__ = ["LoadKeplerData"]

import os

import matplotlib.pyplot as plt
import numpy as np

from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.planet import Planet
from alderaan.data_models.star import Star
from alderaan.utils.io import parse_holczer16_catalog, parse_koi_catalog
from alderaan.utils.plotting import plot_litecurve
from alderaan.utils.stats import robust_polyfit


class LoadKeplerData:
    def __init__(self, target, catalog_filepath, catalog_dir, data_dir, quicklook_dir):
        self.target = target
        self.catalog_filepath = catalog_filepath
        self.catalog_dir = catalog_dir
        self.data_dir = data_dir
        self.quicklook_dir = quicklook_dir

    def perform(self):
        print("\n\nMODULE: LOAD KEPLER DATA\n")

        # load KOI catalog
        catalog = parse_koi_catalog(self.catalog_filepath, self.target)

        assert np.all(np.diff(catalog.period) > 0), (
            "Planets should be ordered by ascending period"
        )

        npl = int(catalog.npl[0])
        star = Star(
            catalog.koi_id[0],
            int(catalog.kic_id[0]),
            catalog.limbdark_1[0],
            catalog.limbdark_2[0],
        )

        # load lightcurves
        litecurve = LiteCurve().from_kplr_pdcsap(
            self.data_dir, star.kic_id, "long cadence", visits=[1, 2, 3]
        )

        t_min = litecurve.time.min()
        t_max = litecurve.time.max()
        if t_min < 0:
            raise ValueError(
                "Lightcurve has negative timestamps...this will cause problems"
            )

        litecurves = litecurve.split_visits()
        for lc in litecurves:
            assert len(np.unique(lc.visit)) == 1, "expected one quarter per litecurve"
            assert len(np.unique(lc.obsmode)) == 1, "expected one obsmode per litecurve"

        print(f"{len(litecurves)} litecurves loaded for {self.target}")

        # initialize planets
        planets = [None] * npl
        for n in range(npl):
            planets[n] = Planet(
                catalog.period[n],
                catalog.epoch[n],
                catalog.depth[n],
                catalog.duration[n],
                catalog.impact[n],
                t_min,
                t_max,
            )

        print(f"\n{npl} planets loaded for {self.target}")
        print([np.round(p.period, 6) for p in planets])

        # load Holczer+2016 catalog
        filepath = os.path.join(self.catalog_dir, "holczer_2016_kepler_ttvs.txt")
        holczer_ephemerides = parse_holczer16_catalog(filepath, star.koi_id, npl)

        print(f"\n{len(holczer_ephemerides)} ephemerides found in Holczer+2016")

        # match Holczer ephemerides to Planets
        count = 0
        for n, p in enumerate(planets):
            for index, ttime, error in holczer_ephemerides:
                period = robust_polyfit(index, ttime, 1)[0]
                match = np.isclose(period, p.period, rtol=0.01, atol=p.duration)

                if match:
                    print(f"  Planet {n} : {p.period:.6f} --> {period:.6f}")
                    planets[n] = p.set_transit_times(index, ttime, error)
                    count += 1

        print(
            f"{count} matching ephemerides found ({len(holczer_ephemerides)} expected)"
        )

        # quicklook litecurve
        filepath = os.path.join(self.quicklook_dir, f"{self.target}_litecurve_raw.png")
        plot_litecurve(litecurve, self.target, planets, filepath)
        plt.close("all")

        # enforce (t_min, t_max) consistency on litecurve and planets
        for n, p in enumerate(planets):
            p.clip_range(t_min, t_max)
            p.set_reference_period_and_epoch()
            planets[n] = p.refit_period_and_epoch()

            print(f"  Planet {n} : {len(p.transit_times)} transits")

        return star, planets, litecurve
