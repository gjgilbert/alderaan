__all__ = ['Ephemeris']

from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
import warnings


class Ephemeris:
    """
    Ephemeris
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

            if t_min is None:
                t_min = self.ttime.min() - self.period / 2
            if t_max is None:
                t_max = self.ttime.max() + self.period / 2

        # clip vectors to range (t_min, t_max)
        self.clip_range(t_min, t_max)

        # put epoch in range (t_min, t_min + period)
        self.adjust_epoch(t_min)


    def clip_range(self, t_min, t_max, adjust_epoch=True):
        """
        Clip attribute arrays to range (t_min, t_max)

        Args:
            t_min (float) : minimum time
            t_max (float) : maximum time

        Returns:
            Ephemeris : self
        """
        self.t_min = t_min
        self.t_max = t_max

        use = (self.ttime >= self.t_min) & (self.ttime <= self.t_max)
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][use])

        if hasattr(self, 'index'):
            self.index -= self.index[0]

        if adjust_epoch:
            self.adjust_epoch(self.t_min)


    def adjust_epoch(self, t_min):
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

        if hasattr(self, 'index'):
            self.index -= self.index[0]
    
    
    def set_static_references(self):
        assert not hasattr(self, '_static_period'), "attribute '_static_period' already exists"
        self._static_period = deepcopy(self.period)

        assert not hasattr(self, '_static_epoch'), "attribute '_static_epoch' already exists"
        self._static_epoch = deepcopy(self.epoch)


    def fit_linear_ephemeris(self, ignore_bad=True):
        """
        Fit a linear ephmeris using unweighted least squares
        """
        if ignore_bad and self.quality is not None:
            q = self.quality
        else:
            q = np.ones(len(self.index), dtype=bool)

        A = np.ones((len(self.index[q]), 2))
        A[:, 0] = self.index[q]

        return np.linalg.lstsq(A, self.ttime[q], rcond=None)[0]
    

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
        

    def interpolate(self, full=False, reset_quality=True):
        """
        Interpolate poor quality transit times and optionally interpolate missing transit times

        Args:
          full (bool) : True to interpolate missing transit times (default=False)

        Returns:
          Ephemeris : self
        """
        assert self.index[0] == 0, f"index[0] = {self.index[0]}, expected zero-indexing"

        if self.quality is not None:
            q = self.quality
        else:
            q = np.ones(len(self.ttime), dtype=bool)

        interpolator = interp1d(self.index[q], self.ttime[q], kind='linear', fill_value='extrapolate')
        
        index_full = np.arange(0, self.index.max()+1, dtype=int)
        ttime_full = interpolator(index_full)

        if self.error is not None:
            error_full = np.nanmedian(self.error)*np.ones_like(index_full)
            error_full[np.isin(index_full, self.index[q])] = self.error[q]
        
        if reset_quality:
            quality_full = np.ones(len(index_full), dtype=bool)
        else:
            quality_full = np.zeros(len(index_full), dtype=bool)
            quality_full[np.isin(index_full, self.index)] = q
        
        if full:
            keep = index_full
        else:
            keep = np.isin(index_full, self.index)
        
        self.index = index_full[keep]
        self.ttime = ttime_full[keep]
        self.error = error_full[keep]
        self.quality = quality_full[keep]

        self = self.update_period_and_epoch()

        return self
    
    
    def update_from_omc(self, omc):
        assert np.allclose(self._static_period, omc._static_period, rtol=1e-12), "static periods do not match"
        assert np.allclose(self._static_epoch, omc._static_epoch, rtol=1e-12), "static epochs do not match"

        if len(self.ttime) != len(omc.xtime):
            warnings.warn(f"updated ephemeris has {len(omc.xtime)} transit times (was {len(self.ttimes)})")
        
        self.index = omc.index.copy()
        self.ttime = self._static_epoch + self._static_period*self.index + omc.ymod
        self.error = omc.yerr.copy()
        self.quality = omc.quality.copy()

        self.period, self.epoch = self.fit_linear_ephemeris()

        return self
    

    def remove_poor_quality_transits(self):
        q = np.copy(self.quality)

        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][q])

        return self