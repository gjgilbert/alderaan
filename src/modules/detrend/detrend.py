__all__ = ['Detrend']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.stats import sigma_clip, mad_std
from astropy.timeseries import LombScargle
import numpy as np
from scipy.signal import medfilt as median_filter
from scipy.signal import savgol_filter
from src.schema.litecurve import LiteCurve
from src.schema.planet import Planet

class SimpleDetrender:
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

        self.periods = []
        self.durs = []
        for p in planets:
            self.periods.append(p.period)
            self.durs.append(p.duration)

    
    def make_transit_mask(self, rel_size=None, abs_size=None):
        """
        Arguments
            rel_size : full width of masked region in units of transit duration
            abs_size : full width of masked region in units of time
            ignore_bad (bool) : True to exclude bad quality transits from mask
        
        mask shape is (num_planets, num_cadences)
        boolean (1 = near transit, 0 = far from transit)
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

        return mask.astype(bool)


    def clip_outliers(self, 
                      kernel_size, 
                      sigma_upper, 
                      sigma_lower, 
                      mask=None,
                      trend=None
                      ):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a median filtered trend
        Applies an additional iteration wrapper to allow for masked cadences

        Arguments:
            kernel_size : int
                size of the smoothing filter kernel
            sigma_upper : float
                upper sigmga clipping threshold
            sigma_lower : float
                lower sigma clipping threshold
            mask : array-like, bool (optional)
                do not reject cadences within masked regions; useful for protecting transits
            trend : array-like, float (optional)
                precomputed model for the trend, e.g. a Keplerian transit lightcurve
        """
        lc = self.litecurve

        if mask is None:
            mask = np.zeros(len(lc.time), dtype=bool)
        
        if trend is not None:
            raise ValueError("Not yet configured for trend kwarg")

        loop = True
        count = 0
        while loop:
            # first pass: try median filter
            if loop:
                smoothed = median_filter(lc.flux, kernel_size=kernel_size)

                bad = sigma_clip(lc.flux - smoothed,
                                 sigma_upper=sigma_upper,
                                 sigma_lower=sigma_lower,
                                 stdfunc=mad_std
                                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter
            # if over 1% of points were flagged with median filter
            if np.sum(bad)/len(bad) > 0.01:
                smoothed = savgol_filter(lc.flux,
                                         window_length=kernel_size,
                                         polyorder=2
                                        )
                bad = sigma_clip(lc.flux - smoothed,
                                 sigma_upper=sigma_upper,
                                 sigma_lower=sigma_lower,
                                 stdfunc=mad_std
                                ).mask
                bad = bad * ~mask

            # third pass: skip outlier rejection
            if np.sum(bad) / len(bad) > 0.01:
                bad = np.zeros_like(mask)

            # set attributes
            for k in lc.__dict__.keys():
                if type(lc.__dict__[k]) is np.ndarray:
                    lc.__setattr__(k, lc.__dict__[k][~bad])

            # update mask and iterate
            mask = mask[~bad]

            if np.sum(bad) == 0:
                loop = False
            else:
                count += 1

            if count >= 3:
                loop = False

        return lc
    

    def estimate_oscillation_period(self, min_period):
        """
        Docstring
        """
        lc = self.litecurve
        lombscargle = LombScargle(lc.time, lc.flux)

        min_freq = 1 / (lc.time.max() - lc.time.min())
        max_freq = 1 / min_period

        xf, yf = lombscargle.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

        return peak_per
    


class GaussianProcessDetrender(SimpleDetrender):
    def __init__(self):
        pass

class AutocorrelationDetrender(SimpleDetrender):
    def __init__(self):
        pass