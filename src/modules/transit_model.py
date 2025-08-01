__all__ = ['TransitModel']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.constants import kepler_lcit, kepler_scit
from src.modules.base import BaseAlg


class TransitModel(BaseAlg):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)

        self.transit_obsmode = self.get_transit_obsmode()
        self.unique_obsmodes = np.unique(np.hstack(self.transit_obsmode))
        
        self._define_supersample_lookup()
        self._define_exptime_integration_offset_lookup()


    def extract_ephemerides(self, center_index=True):
        ttime = [None]*self.npl
        index = [None]*self.npl

        for n, p in enumerate(self.planets):
            q = p.ephemeris.quality

            ttime[n] = p.ephemeris.ttime[q]
            index[n] = p.ephemeris.index[q]

            if center_index:
                index[n] -= index[n][-1] // 2

        return index, ttime
    

    def _compute_supersample_factor(self, obsmode):    
        # ingress/egress timescale estimate following Winn 2010
        tau12 = (13 / 24) * (np.array(self.periods) / 365.25) ** (1 / 3) * np.sqrt(np.array(self.depths)) 

        # sigma so binning error is < 0.1% of photometric uncertainty
        sigma = np.nanmean(self.litecurve.error / self.litecurve.flux) * 0.04

        # supersample factor following Kipping 2010
        exptime = self._obsmode_to_exptime(obsmode)
        supersample = np.array(np.ceil(np.sqrt((self.depths / tau12) * (exptime / 8 / sigma))), dtype=int)
        supersample = np.max([supersample + (supersample % 2 + 1)])

        return supersample
    

    def _define_supersample_lookup(self):
        self.__supersample_lookup = {}
        for om in np.unique(np.hstack(self.transit_obsmode)):
            self.__supersample_lookup[om] = self._compute_supersample_factor(om)
        

    def _obsmode_to_supersample(self, obsmode):
        return self.__supersample_lookup[obsmode]


    def _compute_exptime_integration_offset(self, obsmode):
        exptime = self._obsmode_to_exptime(obsmode)
        supersample = self._obsmode_to_supersample(obsmode)
        return np.linspace(-exptime/2, exptime/2, supersample)
    

    def _define_exptime_integration_offset_lookup(self):
        self.__exptime_integration_offset_lookup = {}
        for om in np.unique(np.hstack(self.transit_obsmode)):
            self.__exptime_integration_offset_lookup[om] = (
                self._compute_exptime_integration_offset(om)
            )
            
    
    def _obsmode_to_exptime_integration_offset(self, obsmode):
        return self.__exptime_integration_offset_lookup[obsmode]