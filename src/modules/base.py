__all__ = ['BaseAlg']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.constants import kepler_lcit, kepler_scit
from src.schema.litecurve import LiteCurve
from src.schema.planet import Planet


class BaseAlg():
    """
    Base Algorithm class for modules which work on a litecurve + planets
     * Detrend
     * Transit Model
     * TTV Model
     * Quality Control
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
        self.litecurve = litecurve
        self.npl = len(planets)
        self.planets = planets

        self.periods = np.zeros(self.npl, dtype=float)
        self.depths = np.zeros(self.npl, dtype=float)
        self.durs = np.zeros(self.npl, dtype=float)
        
        for n, p in enumerate(planets):
            self.periods[n] = p.period
            self.depths[n] = p.depth
            self.durs[n] = p.duration

        # define lookup
        self._define_exptime_lookup()


    def _define_exptime_lookup(self):
        self.__exptime_lookup = {
            'long cadence': kepler_lcit,
            'short cadence': kepler_scit
        }


    def _obsmode_to_exptime(self, obsmode):       
        return self.__exptime_lookup[obsmode]
            
    
    def make_transit_mask(self, rel_size=None, abs_size=None, mask_type='standard'):
        """
        Arguments
            rel_size : full width of masked region in units of transit duration
            abs_size : full width of masked region in units of time
            mask_type : type of transit mask to return (default = 'standard')
        
        Several output mask_type are supported
         * standard : bool, shape (n_planet, n_cadence), True near transits
         * condensed : bool, shape (n_cadence), True near transit
         * count : int, shape (n_cadence), value is number of planets near transit
         * overlap : bool, shanpe (n_cadnece), True if multiple planets are near transit
        """
        if (rel_size is None) & (abs_size is None):
            raise ValueError("either rel_size or abs_size must be provided")
        
        if rel_size is None:
            rel_size = 0.
        if abs_size is None:
            abs_size = 0.

        mask = np.zeros((self.npl,len(self.litecurve.time)), dtype=int)

        for n, p in enumerate(self.planets):
            half_width = 0.5*np.max([rel_size*p.duration, abs_size])
            for tc in p.ephemeris.ttime:
                mask[n] += np.abs(self.litecurve.time - tc) < half_width

        if mask_type == 'standard':
            mask = mask.astype(bool)
        elif mask_type == 'condensed':
            mask = np.sum(mask, axis=0).astype(bool)
        elif mask_type == 'count':
            mask = np.sum(mask, axis=0)
        elif mask_type == 'overlap':
            mask = np.sum(mask, axis=0) > 0
        else:
            raise ValueError(f"mask_type f{mask_type} not supported")

        return mask.astype(bool)


    def get_transit_obsmode(self):
        """
        Determine the observing mode at each transit time
        Returns a length num_planets list, each entry is a list of obsmode str
        """
        lc = self.litecurve
        obsmode = [None]*self.npl

        for n, p in enumerate(self.planets):
            obsmode[n] = []
            for tc in p.ephemeris.ttime:
                obsmode[n].append(lc.obsmode[np.argmin(np.abs(lc.time-tc))])
            obsmode[n] = np.array(obsmode[n])

        return obsmode
    

    def identify_overlapping_transits(self, rtol=None, atol=None):
        """
        Identify where transits overlap based on separation of transit midpoints

        Arguments
            rtol (float) : relative tolerance, in units of transit durations
            atol (float) : absolute tolerance, in units of hours
        """
        if (rtol is None) and (atol is None):
            raise ValueError("at least one of rtol or atol must be provided")

        overlap = [None]*self.npl

        for i in range(self.npl):
            overlap[i] = np.zeros(len(self.planets[i].ephemeris.ttime), dtype=bool)
            ttime_i = self.planets[i].ephemeris.ttime
            
            for j in range(i+1,self.npl):
                for tc_j in self.planets[j].ephemeris.ttime:
                    if rtol is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < rtol * 0.5 * (self.durs[i] + self.durs[j])
                    if atol is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < atol

        return overlap