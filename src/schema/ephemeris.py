__all__ = ['Ephemeris']

import numpy as np
from scipy.interpolate import CubicSpline


class Ephemeris:
    def __init__(self, 
                 period=None,
                 epoch=None,
                 index=None, 
                 ttime=None, 
                 error=None,
                 quality=None,
                 t_min=None,
                 t_max=None
                ):
        """
        Docstring
        """
        # check inputs
        init_linear_ephem = (
            (period is not None) & 
            (epoch is not None) &
            (t_min is not None) &
            (t_max is not None)
        )

        init_ttv_ephem = (
            (index is not None) &
            (ttime is not None)
        )

        if init_linear_ephem & init_ttv_ephem:
            raise ValueError("must supply exactly on of (ttime, index) or (period, epoch)")
        
        # Case 1 : linear ephemeris
        if init_linear_ephem:
            self.period = period
            self.epoch = epoch
            self.t_min = t_min
            self.t_max = t_max

            self._adjust_epoch(self.t_min)

            self.ttime = np.arange(self.epoch, t_max, self.period)
            self.index = np.array(np.round((self.ttime - self.epoch) / self.period), dtype=int)
            self.error = None
            self.quality = None

        # Case 2 : TTV ephemeris
        elif init_ttv_ephem:
            self.index = index
            self.ttime = ttime
            self.error = error
            self.quality = quality

            self.period, self.epoch = self.fit_linear_ephemeris()

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

        # clip vectors to range (t_min, t_max)
        use = (self.ttime >= self.t_min) & (self.ttime <= self.t_max)
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][use])


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
    

    def update_period_and_epoch(self):
        """
        Recompute linear ephemeris to ensure consistency
        """
        self.period, self.epoch = self.fit_linear_ephemeris()

        return self
        

    def full_ephemeris(self, return_index=True):
        """
        Interpolate ephemeris over missing and poor quality transit times
        Assumes indexing starts at zero
        """
        if self.quality is not None:
            q = self.quality
        else:
            q = np.ones(len(self.ttime), dtype=bool)

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