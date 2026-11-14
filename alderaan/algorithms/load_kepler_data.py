"""Load Kepler photometry, KOI catalog values, and Holczer+2016 transit times"""

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
    """
    Load the star, planets, and long cadence litecurve for one Kepler target

    Parameters
    ----------
    config : ConfigParser
        pipeline configuration, including RUN and PATHS values set by main
    """

    def __init__(self, config):
        self.config = config

    def perform(self):
        """
        Returns
        -------
        star : Star
        planets : list of Planet
            ephemerides initialized from the KOI catalog and Holczer+2016
        litecurve : LiteCurve
            raw litecurve, all visits concatenated
        """
        print("\n\nALGORITHM: LOAD KEPLER DATA\n")

        target = self.config["RUN"]["target"]
        catalog_filepath = self.config["PATHS"]["catalog_filepath"]
        catalog_dir = self.config["PATHS"]["catalog_dir"]
        data_dir = self.config["PATHS"]["data_dir"]
        quicklook_dir = self.config["PATHS"]["quicklook_dir"]

        # load KOI catalog
        catalog = parse_koi_catalog(catalog_filepath, target)

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
            data_dir, star.kic_id, "long cadence", visits=[1, 2, 3]
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

        print(f"{len(litecurves)} litecurves loaded for {target}")

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

        print(f"\n{npl} planets loaded for {target}")
        print([np.round(p.period, 6) for p in planets])

        # load Holczer+2016 catalog
        filepath = os.path.join(catalog_dir, "holczer_2016_kepler_ttvs.txt")
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
        filepath = os.path.join(quicklook_dir, f"{target}_litecurve_raw.png")
        plot_litecurve(litecurve, target, planets, filepath)
        plt.close("all")

        # enforce (t_min, t_max) consistency on litecurve and planets
        for n, p in enumerate(planets):
            p.clip_range(t_min, t_max)
            p.set_reference_period_and_epoch()
            planets[n] = p.refit_period_and_epoch()

            print(f"  Planet {n} : {len(p.transit_times)} transits")

        return star, planets, litecurve
