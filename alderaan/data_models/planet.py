__all__ = ['Planet']

from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
import warnings


class Planet:
    """Planet
    """
    def __init__(self,
                 catalog,
                 koi_id,
                 planet_no,
                 t_min,
                 t_max
                ):
        # read transit parameters from pandas dataframe
        self = self._from_dataframe(catalog, koi_id, planet_no)

        # initialize transit times from linear ephemeris
        self.transit_times = np.arange(self.epoch, t_max, self.period)
        self.transit_index = np.array(np.round((self.transit_times - self.epoch) / self.period), dtype=int)
        self.transit_errors = None
        self.transit_quality = None

        self.reference_period = None
        self.reference_epoch = None

        self.clip_range(t_min, t_max)
        self.refit_period_and_epoch()


    def _from_dataframe(self, catalog, koi_id, planet_no):
        df = catalog.loc[catalog.koi_id == koi_id].sort_values(by='period').reset_index(drop=True)

        self.koi_id = koi_id
        self.kic_id = str(df.at[planet_no, 'kic_id'])
        self.period = float(df.at[planet_no, 'period'])
        self.epoch = float(df.at[planet_no, 'epoch'])
        self.depth = float(df.at[planet_no, 'depth']) * 1e-6       # ppm
        self.duration = float(df.at[planet_no, 'duration']) / 24.  # hrs --> days
        self.impact = float(df.at[planet_no, 'impact'])

        return self


    def set_transit_times(self, index, ttime, error):
        """
        Replace transit times with measured values (e.g. from a catalog)

        Args:
            index (ndarray) : transit index
            ttime (ndarray) : transit times
            error (ndarray) : transit time uncertainties

        Returns:
            Planet : self
        """
        old_period = self.period

        self.transit_index = np.copy(index)
        self.transit_times = np.copy(ttime)
        self.transit_errors = np.copy(error)
        self.transit_quality = None

        self.period, self.epoch = self.fit_linear_ephemeris(self.transit_index, self.transit_times)
        self._check_period_change(old_period)

        t_min = self.transit_times.min() - self.period / 2
        t_max = self.transit_times.max() + self.period / 2
        self.clip_range(t_min, t_max)
        self.refit_period_and_epoch()

        return self


    @staticmethod
    def fit_linear_ephemeris(index, ttime, quality=None):
        """
        Fit a linear ephemeris using unweighted least squares

        Args:
            index (ndarray) : transit index
            ttime (ndarray) : transit times
            quality (ndarray) : boolean mask; True for transits to include in fit (default=None, use all)

        Returns:
            ndarray : (period, epoch)
        """
        if quality is None:
            quality = np.ones(len(index), dtype=bool)

        A = np.ones((len(index[quality]), 2))
        A[:, 0] = index[quality]

        return np.linalg.lstsq(A, ttime[quality], rcond=None)[0]


    def _check_period_change(self, old_period):
        if not np.isclose(old_period, self.period, rtol=0.1):
            raise ValueError(f"New period ({self.period:.6f}) differs from old period ({old_period:.6f}) by more than 10%")


    def clip_range(self, t_min, t_max):
        """
        Clip transit arrays to range (t_min, t_max), re-zero transit index, and put epoch in range (t_min, t_min + period)

        Args:
            t_min (float) : minimum time
            t_max (float) : maximum time

        Returns:
            Planet : self
        """
        use = (self.transit_times >= t_min) & (self.transit_times <= t_max)
        self._keep_transits(use)

        self.transit_index -= self.transit_index[0]
        self.adjust_epoch(t_min)

        return self


    def adjust_epoch(self, t_min):
        """
        Put epoch in range (t_min, t_min + period)

        Args:
            t_min (float) : minimum time

        Returns:
            Planet : self
        """
        if self.epoch < t_min:
            adj = 1 + (t_min - self.epoch) // self.period
            self.epoch += adj * self.period
        if self.epoch > (t_min + self.period):
            adj = (self.epoch - t_min) // self.period
            self.epoch -= adj * self.period

        return self


    def _keep_transits(self, keep):
        self.transit_index = self.transit_index[keep]
        self.transit_times = self.transit_times[keep]
        if self.transit_errors is not None:
            self.transit_errors = self.transit_errors[keep]
        if self.transit_quality is not None:
            self.transit_quality = self.transit_quality[keep]


    def set_reference_period_and_epoch(self):
        """
        Freeze current period and epoch as the fixed reference for O-C and time warping
        """
        assert self.reference_period is None, "reference_period already set"
        self.reference_period = deepcopy(self.period)

        assert self.reference_epoch is None, "reference_epoch already set"
        self.reference_epoch = deepcopy(self.epoch)


    def eval_linear_ephemeris(self, index=None):
        """
        Calculate linear ephemeris from period and epoch

        Returns:
            ndarray : transit times according to linear ephemeris
        """
        if index is None:
            index = self.transit_index

        return self.epoch + self.period*index


    def refit_period_and_epoch(self):
        """
        Fit period and epoch to good quality transit times

        Returns:
            Planet : self
        """
        self.period, self.epoch = self.fit_linear_ephemeris(self.transit_index, self.transit_times, self.transit_quality)

        return self


    def interpolate(self, full=False, reset_quality=True):
        """
        Interpolate poor quality transit times and optionally interpolate missing transit times

        Args:
            full (bool) : True to interpolate missing transit times (default=False)

        Returns:
            Planet : self
        """
        assert self.transit_index[0] == 0, f"transit_index[0] = {self.transit_index[0]}, expected zero-indexing"

        if self.transit_quality is not None:
            q = self.transit_quality
        else:
            q = np.ones(len(self.transit_times), dtype=bool)

        interpolator = interp1d(self.transit_index[q], self.transit_times[q], kind='linear', fill_value='extrapolate')

        index_full = np.arange(0, self.transit_index.max()+1, dtype=int)
        ttime_full = interpolator(index_full)

        if self.transit_errors is not None:
            error_full = np.nanmedian(self.transit_errors)*np.ones_like(index_full)
            error_full[np.isin(index_full, self.transit_index[q])] = self.transit_errors[q]

        if reset_quality:
            quality_full = np.ones(len(index_full), dtype=bool)
        else:
            quality_full = np.zeros(len(index_full), dtype=bool)
            quality_full[np.isin(index_full, self.transit_index)] = q

        if full:
            keep = index_full
        else:
            keep = np.isin(index_full, self.transit_index)

        self.transit_index = index_full[keep]
        self.transit_times = ttime_full[keep]
        self.transit_errors = error_full[keep]
        self.transit_quality = quality_full[keep]

        return self.refit_period_and_epoch()


    def update_from_omc(self, omc):
        """
        Replace transit times with O-C model added to the reference ephemeris

        Args:
            omc (OMC)

        Returns:
            Planet : self
        """
        assert np.allclose(self.reference_period, omc._static_period, rtol=1e-12), "reference periods do not match"
        assert np.allclose(self.reference_epoch, omc._static_epoch, rtol=1e-12), "reference epochs do not match"

        if len(self.transit_times) != len(omc.xtime):
            warnings.warn(f"updated ephemeris has {len(omc.xtime)} transit times (was {len(self.transit_times)})")

        old_period = self.period

        self.transit_index = omc.index.copy()
        self.transit_times = self.reference_epoch + self.reference_period*self.transit_index + omc.ymod
        self.transit_errors = omc.yerr.copy()
        self.transit_quality = omc.quality.copy()

        self.refit_period_and_epoch()
        self._check_period_change(old_period)

        return self


    def remove_poor_quality_transits(self):
        """
        Remove transits flagged as poor quality

        Returns:
            Planet : self
        """
        self._keep_transits(np.copy(self.transit_quality))

        return self
