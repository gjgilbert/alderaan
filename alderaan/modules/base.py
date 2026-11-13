__all__ = ["BaseAlg"]

from copy import deepcopy
import numpy as np
from alderaan.constants import kepler_lcit, kepler_scit
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.planet import Planet


class BaseAlg:
    """
    Base Algorithm class for modules which work on a litecurve + planets
    """

    def __init__(self, litecurve, planets):
        # check inputs
        if not isinstance(litecurve, LiteCurve):
            raise TypeError("litecurve must be a LiteCurve")
        if not isinstance(planets, list):
            raise TypeError("planets must be a list of Planets")
        for p in planets:
            if not isinstance(p, Planet):
                raise TypeError("planets must be a list of Planets")

        # set attributes
        self.litecurve = deepcopy(litecurve)
        self.npl = len(planets)
        self.planets = deepcopy(planets)

        self._init_obsmode_tracking()

    def _init_obsmode_tracking(self):
        lc = self.litecurve

        self.exptime = {
            "long cadence": kepler_lcit,
            "short cadence": kepler_scit,
        }

        # observing mode at the cadence nearest each transit
        self.transit_obsmode = [None] * self.npl
        for n, p in enumerate(self.planets):
            nearest = [np.argmin(np.abs(lc.time - tc)) for tc in p.transit_times]
            self.transit_obsmode[n] = np.array([lc.obsmode[i] for i in nearest])

        self.unique_obsmodes = np.unique(np.hstack(self.transit_obsmode))

        # ingress/egress timescale estimate following Winn 2010
        period = np.array([p.period for p in self.planets])
        depth = np.array([p.depth for p in self.planets])
        tau12 = (13 / 24) * (period / 365.25) ** (1 / 3) * np.sqrt(depth)

        # sigma so binning error is < 0.1% of photometric uncertainty
        sigma = np.nanmean(lc.error / lc.flux) * 0.04

        # supersample factor (forced odd) following Kipping 2010, and exposure time integration offsets
        self.supersample = {}
        self._exptime_offsets = {}
        for om in self.unique_obsmodes:
            exptime = self.exptime[om]
            supersample = np.array(
                np.ceil(np.sqrt((depth / tau12) * (exptime / 8 / sigma))), dtype=int
            )
            self.supersample[om] = np.max(supersample + (supersample % 2 + 1))
            self._exptime_offsets[om] = np.linspace(
                -exptime / 2, exptime / 2, self.supersample[om]
            )

        self._set_cadence_masks()

    def _set_cadence_masks(self):
        self._cadence_mask = {
            om: self.litecurve.obsmode == om for om in self.unique_obsmodes
        }

    def make_transit_mask(self, rel_size=None, abs_size=None, mask_type="standard"):
        """
        Arguments
            rel_size : full width of masked region in units of transit duration
            abs_size : full width of masked region in units of time
            mask_type : type of transit mask to return (default = 'standard')

        Supported mask_type
         * standard : bool, shape (n_planet, n_cadence), True near transits of each planet
         * condensed : bool, shape (n_cadence), True near transit of any planet
        """
        if (rel_size is None) & (abs_size is None):
            raise ValueError("either rel_size or abs_size must be provided")

        if rel_size is None:
            rel_size = 0.0
        if abs_size is None:
            abs_size = 0.0

        mask = np.zeros((self.npl, len(self.litecurve.time)), dtype=int)

        for n, p in enumerate(self.planets):
            half_width = 0.5 * np.max([rel_size * p.duration, abs_size])
            for tc in p.transit_times:
                mask[n] += np.abs(self.litecurve.time - tc) < half_width

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

            for j in range(i + 1, self.npl):
                for tc_j in self.planets[j].transit_times:
                    if rtol is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < rtol * 0.5 * (
                            self.planets[i].duration + self.planets[j].duration
                        )
                    if atol is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < atol

        return overlap
