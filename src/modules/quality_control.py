__all__ = ['QualityControl']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.stats import mad_std
import numpy as np
from src.constants import kepler_lcit, kepler_scit
from src.modules.base import BaseAlg
import warnings


class QualityControl(BaseAlg):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)

        self.transit_obsmode = self.get_transit_obsmode()


    def check_coverage(self):
        """
        Check that each transit has at least 50% coverage
          * number of cadences in transit (tc +/- 0.5*T14)
          * number of cadences near transit (tc +/- 1.5*T14)
        """
        lc = self.litecurve

        quality = [None]*self.npl

        for n, p in enumerate(self.planets):
            obsmode = np.array(self.transit_obsmode[n])

            exptime = np.zeros(len(p.ephemeris.ttime), dtype=float)
            exptime[obsmode == 'long cadence'] = kepler_lcit
            exptime[obsmode == 'short cadence'] = kepler_scit

            assert not any(exptime == 0), "all exposure times expected to be non-zero"

            count_expected = np.maximum(1, np.array(p.duration / exptime, dtype=int))

            quality[n] = np.zeros(len(p.ephemeris.ttime), dtype='bool')

            for i, tc in enumerate(p.ephemeris.ttime):
                in_transit = np.abs(lc.time - tc) / p.duration < 0.5
                near_transit = np.abs(lc.time - tc) / p.duration < 1.5

                enough_pts_in = np.sum(in_transit) > 0.5 * count_expected[i]
                enough_pts_near = np.sum(near_transit) > 1.5 * count_expected[i]

                quality[n][i] = enough_pts_in & enough_pts_near

            f_quality = np.sum(quality[n]) / len(quality[n])

            if f_quality < 0.8:
                warnings.warn(f"WARNING: {int((1-f_quality)*100)}% of transits for Planet {n} have been flagged as low quality")

        return quality

            
    def check_rms(self, rel_size=None, abs_size=None, sigma_cut=5.0):
        if (rel_size is None) & (abs_size is None):
            raise ValueError("either rel_size or abs_size must be provided")
        
        if rel_size is None:
            rel_size = 0.
        if abs_size is None:
            abs_size = 0.

        lc = self.litecurve

        overlap = self.identify_overlapping_transits(rtol=rel_size, atol=abs_size)
        quality = [None]*self.npl

        for n, p in enumerate(self.planets):
            rms = np.zeros(len(p.ephemeris.ttime))
            half_width = 0.5*np.max([rel_size*p.duration, abs_size])

            for i, tc in enumerate(p.ephemeris.ttime):
                flux = lc.flux[np.abs(lc.time - tc) < half_width]
                rms[i] = mad_std(flux, ignore_nan=True)

            obsmode = np.array(self.transit_obsmode[n])
            quality[n] = np.ones(len(p.ephemeris.ttime), dtype='bool')

            for om in np.unique(obsmode):
                use = (obsmode == om) & ~overlap[n]
                rms_mu = np.nanmedian(rms[use])
                rms_sd = mad_std(rms[use], ignore_nan=True)

                quality[n][use] &= np.abs(rms[use] - rms_mu)/rms_sd < sigma_cut

        return quality