__all__ = ['Ephemeris']

import numpy as np
from scipy.interpolate import CubicSpline


class Ephemeris:
    def __init__(self, 
                 index, 
                 ttime, 
                 error=None,
                 quality=None,
                 t_min=None,
                 t_max=None
                ):
        """
        Docstring
        """
        # populate attributes
        self.index = index
        self.ttime = ttime
        self.error = error
        self.quality = quality

        # calculate period and epoch from linear emphemeris fit
        self.period, self.epoch = self.fit_linear_ephemeris()

        # set (t_min, t_max)
        if t_min is not None:
            self.t_min = t_min
        else:
            self.t_min = self.ttime.min() - self.period / 2
        
        if t_max is not None:
            self.t_max = t_max
        else:
            self.t_max = self.ttime.max() + self.period / 2

        # put epoch in range (t_min, t_min + period)
        self.epoch = self._adjust_epoch(self.t_min)

    
    def _adjust_epoch(self, t_min):
        """
        Put epoch in range (t_min, t_min + period)
        """
        if self.epoch < t_min:
            adj = 1 + (t_min - self.epoch) // self.period
            self.epoch += adj * self.period
        if self.epoch > (t_min + self.period):
            adj = (self.epoch - t_min) // self.period
            self.epoch -= adj * self.period

        return self.epoch
    
    
    def fit_linear_ephemeris(self):
        """
        Fit a linear ephmeris using unweighted least squares
        """
        A = np.ones((len(self.index), 2))
        A[:, 0] = self.index

        return np.linalg.lstsq(A, self.ttime, rcond=None)[0]
    

    def eval_linear_ephemeris(self, index=None):
        """
        Calculate linear ephemeris from period and epoch 
        """
        if index is None:
            index = self.index

        return self.epoch + self.period*index
        

    def full_ephemeris(self, return_index=True):
        """
        Interpolate ephemeris over missing and poor quality transit times
        Assumes indexing starts at zero
        """
        if self.quality is not None:
            q = self.quality
        else:
            q = np.ones(len(self.time), dtype=bool)

        spline = CubicSpline(self.index[q],
                              self.ttime[q], 
                              extrapolate=True,
                              bc_type='natural'
                             )

        full_index = np.arange(0, self.index.max()+1, dtype=int)
        full_ttime = spline(full_index)
        full_ttime[self.index] = self.ttime

        if return_index:
            return full_index, full_ttime
        return full_ttime