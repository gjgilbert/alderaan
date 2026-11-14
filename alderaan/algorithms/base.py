"""Base class for algorithms that operate on a star, planets, and litecurve"""

from copy import deepcopy

import numpy as np

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.planet import Planet
from alderaan.data_models.star import Star
from alderaan.utils.astro import transit_obsmode


class BaseAlg:
    """
    Type-check and deepcopy inputs; provide shared transit bookkeeping

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    config : ConfigParser
        pipeline configuration, including RUN and PATHS values set by main
    """

    def __init__(self, star, planets, litecurve, config):
        # check inputs
        if not isinstance(litecurve, LiteCurve):
            raise TypeError("litecurve must be a LiteCurve")
        if not isinstance(planets, list):
            raise TypeError("planets must be a list of Planets")
        for p in planets:
            if not isinstance(p, Planet):
                raise TypeError("planets must be a list of Planets")
        if not isinstance(star, Star):
            raise TypeError("star must be a Star")

        # set attributes
        self.litecurve = deepcopy(litecurve)
        self.npl = len(planets)
        self.planets = deepcopy(planets)
        self.star = deepcopy(star)
        self.config = config

        self._init_obsmode_tracking()

    def _init_obsmode_tracking(self):
        self.exptime = {
            "long cadence": KEPLER_LCIT,
            "short cadence": KEPLER_SCIT,
        }

        self.transit_obsmode, self.unique_obsmodes = transit_obsmode(
            self.planets, self.litecurve
        )

    def make_transit_mask(
        self, rel_size=None, abs_size=None, mask_type="standard", time=None
    ):
        """
        Arguments
            rel_size : full width of masked region in units of transit duration
            abs_size : full width of masked region in units of time
            mask_type : type of transit mask to return (default = 'standard')
            time : cadence times to mask (default = self.litecurve.time)

        Supported mask_type
         * standard : bool, shape (n_planet, n_cadence),
                      True near transits of each planet
         * condensed : bool, shape (n_cadence), True near transit of any planet
        """
        if (rel_size is None) & (abs_size is None):
            raise ValueError("either rel_size or abs_size must be provided")

        if rel_size is None:
            rel_size = 0.0
        if abs_size is None:
            abs_size = 0.0

        if time is None:
            time = self.litecurve.time

        mask = np.zeros((self.npl, len(time)), dtype=int)

        for n, p in enumerate(self.planets):
            half_width = 0.5 * np.max([rel_size * p.duration, abs_size])
            for tc in p.transit_times:
                mask[n] += np.abs(time - tc) < half_width

        if mask_type == "standard":
            return mask.astype(bool)
        elif mask_type == "condensed":
            return np.sum(mask, axis=0).astype(bool)
        else:
            raise ValueError(f"mask_type {mask_type} not supported")

    def identify_overlapping_transits(self, rtol=None, atol=None):
        """
        Identify where transits overlap based on separation of transit midpoints

        Arguments
            rtol (float) : relative tolerance, in units of transit durations
            atol (float) : absolute tolerance, in units of days
        """
        if (rtol is None) and (atol is None):
            raise ValueError("at least one of rtol or atol must be provided")

        overlap = [None] * self.npl

        for i in range(self.npl):
            overlap[i] = np.zeros(len(self.planets[i].transit_times), dtype=bool)
            ttime_i = self.planets[i].transit_times

            for j in range(self.npl):
                if j == i:
                    continue
                for tc_j in self.planets[j].transit_times:
                    if rtol is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < rtol * 0.5 * (
                            self.planets[i].duration + self.planets[j].duration
                        )
                    if atol is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < atol

        return overlap
