__all__ = ['Ephemeris']

from copy import deepcopy
import numpy as np
from scipy.interpolate import CubicSpline
import warnings


class Ephemeris:
    """Ephemeris
    """
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

        if init_linear_ephem + init_ttv_ephem != 1:
            raise ValueError("must supply exactly one of (ttime, index) or (period, epoch)")
        
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

        # clip vectors to range (t_min, t_max)
        use = (self.ttime >= self.t_min) & (self.ttime <= self.t_max)
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][use])

        # put epoch in range (t_min, t_min + period)
        self._adjust_epoch(self.t_min)

        # set static period, epoch, and ephemeris
        self._set_static_references()


    def _adjust_epoch(self, t_min):
        """
        Put epoch in range (t_min, t_min + period)

        Args:
          t_min (float) : minimum time

        Returns:
          Ephemeris : self
        """
        if self.epoch < t_min:
            adj = 1 + (t_min - self.epoch) // self.period
            self.epoch += adj * self.period
        if self.epoch > (t_min + self.period):
            adj = (self.epoch - t_min) // self.period
            self.epoch -= adj * self.period
    
    
    def _set_static_references(self):
        assert not hasattr(self, '_static_period'), "attribute '_static_period' already exists"
        self._static_period = np.copy(self.period)

        assert not hasattr(self, '_static_epoch'), "attribute '_static_epoch' already exists"
        self._static_epoch = np.copy(self.epoch)


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

        Returns:
          ndarray : transit times according to linear ephemeris
        """
        if index is None:
            index = self.index

        return self.epoch + self.period*index
    

    def update_period_and_epoch(self):
        """
        Recompute linear ephemeris to ensure consistency

        Returns:
          Ephemeris : self
        """
        self.period, self.epoch = self.fit_linear_ephemeris()

        return self
        

    def interpolate(self, method, full=False):
        """
        Interpolate poor quality transit times and optionally interpolate missing transit times

        Args:
          method (str) : 'linear' or 'spline'
          full (bool) : True to interpolate missing transit times (default=False)

        Returns:
          Ephemeris : self
        """        
        if self.quality is not None:
            q = self.quality
        else:
            q = np.ones(len(self.ttime), dtype=bool)

        transit_exists = deepcopy(self.index)
        index_full = np.arange(self.index.min(), self.index.max()+1, dtype=int)

        if method == 'linear':
            ttime_full = self.epoch + self.period*index_full
        
        elif method == 'spline':
            spline = CubicSpline(self.index[q],
                                self.ttime[q], 
                                extrapolate=True,
                                bc_type='natural'
                                )

            ttime_full = spline(index_full)

        ttime_full[transit_exists[q]] = self.ttime[q]

        error_full = np.zeros(len(ttime_full))*np.nan
        if self.error is not None:
            error_full[transit_exists[q]] = self.error[q]

        quality_full = np.ones(len(ttime_full), dtype=bool)
        if self.quality is not None:
            quality_full[transit_exists] = self.quality
        
        if full:
            keep = index_full
        else:
            keep = transit_exists
        
        self.index = index_full[keep]
        self.ttime = ttime_full[keep]
        self.error = error_full[keep]
        self.quality = quality_full[keep]

        self = self.update_period_and_epoch()

        return self
    
    
    def update_from_omc(self, omc):
        assert np.all(np.isclose(self._static_period, omc._static_period, rtol=1e-12)), "static periods do not match"
        assert np.all(np.isclose(self._static_epoch, omc._static_epoch, rtol=1e-12)), "static epochs do not match"

        if len(self.ttime) != len(omc.xtime):
            warnings.warn(f"updated ephemeris has {len(omc.xtime)} transit times (was {len(self.ttimes)})")
        
        self.index = omc.index.copy()
        self.ttime = self._static_epoch + self._static_period*self.index + omc.ymod
        self.error = omc.yerr.copy()
        self.quality = omc.quality.copy()

        self.period, self.epoch = self.fit_linear_ephemeris()

        return self