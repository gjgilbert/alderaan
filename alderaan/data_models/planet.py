__all__ = ["Planet"]

import numpy as np
from scipy.interpolate import interp1d

from alderaan.utils.stats import robust_polyfit


class Planet:
    """Planet"""

    def __init__(self, period, epoch, depth, duration, impact, t_min, t_max):
        self.period = period
        self.epoch = epoch
        self.depth = depth
        self.duration = duration
        self.impact = impact
        self.ror = None

        # initialize transit times from linear ephemeris
        self.transit_times = np.arange(self.epoch, t_max, self.period)
        self.transit_index = np.array(
            np.round((self.transit_times - self.epoch) / self.period), dtype=int
        )
        self.transit_errors = None
        self.transit_quality = None

        self.static_period = None
        self.static_epoch = None

        self.clip_range(t_min, t_max)
        self.refit_period_and_epoch()

    def set_transit_times(self, index, ttime, error):
        """
        Replace transit times with measured values (e.g. from a catalog)

        Args:
            index (ndarray) : transit index
            ttime (ndarray) : transit times
            error (ndarray) : transit time uncertainties

        """
        old_period = self.period

        self.transit_index = index
        self.transit_times = ttime
        self.transit_errors = error
        self.transit_quality = None

        self.refit_period_and_epoch()
        self._check_period_change(old_period)

        t_min = self.transit_times.min() - self.period / 2
        t_max = self.transit_times.max() + self.period / 2
        self.clip_range(t_min, t_max)
        self.refit_period_and_epoch()

    def _check_period_change(self, old_period):
        if not np.isclose(old_period, self.period, rtol=0.1):
            raise ValueError(
                f"New period ({self.period:.6f}) differs from old period "
                f"({old_period:.6f}) by more than 10%"
            )

    def clip_range(self, t_min, t_max):
        """
        Clip transit arrays to range (t_min, t_max), re-zero transit index, and put
        epoch in range (t_min, t_min + period)

        Args:
            t_min (float) : minimum time
            t_max (float) : maximum time
        """
        use = (self.transit_times >= t_min) & (self.transit_times <= t_max)
        self.keep_transits(use)

        self.transit_index -= self.transit_index[0]

        if self.epoch < t_min:
            adj = 1 + (t_min - self.epoch) // self.period
            self.epoch += adj * self.period
        if self.epoch > (t_min + self.period):
            adj = (self.epoch - t_min) // self.period
            self.epoch -= adj * self.period

    def keep_transits(self, keep):
        """
        Keep only transits where keep is True; period and epoch are not refit

        Args:
            keep (ndarray) : boolean mask over transits
        """
        self.transit_index = self.transit_index[keep]
        self.transit_times = self.transit_times[keep]
        if self.transit_errors is not None:
            self.transit_errors = self.transit_errors[keep]
        if self.transit_quality is not None:
            self.transit_quality = self.transit_quality[keep]

    def set_static_period_and_epoch(self):
        """
        Freeze current period and epoch as the fixed reference for O-C and time warping
        """
        assert self.static_period is None, "static_period already set"
        self.static_period = self.period

        assert self.static_epoch is None, "static_epoch already set"
        self.static_epoch = self.epoch

    def set_transit_shape(self, ror, impact, duration, depth):
        """
        Set transit shape parameters (e.g. from QuickFitTransitShape)

        Args:
            ror (float) : planet-to-star radius ratio
            impact (float) : impact parameter
            duration (float) : transit duration (days)
            depth (float) : transit depth (fraction)
        """
        self.ror = ror
        self.impact = impact
        self.duration = duration
        self.depth = depth

    def update_transit_times(self, ttime, error=None):
        """
        Replace transit times (same transit index) and refit period and epoch

        Args:
            ttime (ndarray) : transit times, one per transit
            error (ndarray) : transit time uncertainties; None to leave unchanged
        """
        if len(ttime) != len(self.transit_index):
            raise ValueError("ttime must have one entry per transit")

        self.transit_times = ttime
        if error is not None:
            self.transit_errors = error

        self.refit_period_and_epoch()

    def linear_ephemeris(self):
        """
        Calculate linear ephemeris from period and epoch

        Returns:
            ndarray : transit times according to linear ephemeris
        """
        return self.epoch + self.period * self.transit_index

    def refit_period_and_epoch(self):
        """
        Fit period and epoch to good quality transit times
        """
        if self.transit_quality is not None:
            q = self.transit_quality
        else:
            q = np.ones(len(self.transit_index), dtype=bool)

        self.period, self.epoch = robust_polyfit(
            self.transit_index[q], self.transit_times[q], 1
        )

    def update_ephemeris(self, index, omc_model, error, quality, full=False):
        """
        Replace transit times with O-C model added to the current linear ephemeris,
        interpolate over poor quality transits, and reset all quality flags to True

        Args:
            index (ndarray) : transit index
            omc_model (ndarray) : O-C model relative to the current linear ephemeris
            error (ndarray) : transit time uncertainties; None if not yet measured
            quality (ndarray) : boolean mask; True for good quality transits
            full (bool) : True to also interpolate missing transit times (default=False)
        """
        old_period = self.period

        self.transit_times = self.epoch + self.period * index + omc_model
        self.transit_index = index
        self.transit_errors = error
        self.transit_quality = quality

        self.refit_period_and_epoch()
        self._check_period_change(old_period)

        # interpolate
        assert self.transit_index[0] == 0, (
            f"transit_index[0] = {self.transit_index[0]}, expected zero-indexing"
        )

        q = self.transit_quality
        interpolator = interp1d(
            self.transit_index[q],
            self.transit_times[q],
            kind="linear",
            fill_value="extrapolate",
        )

        index_full = np.arange(0, self.transit_index.max() + 1, dtype=int)
        ttime_full = interpolator(index_full)

        if self.transit_errors is not None:
            error_full = np.nanmedian(self.transit_errors) * np.ones_like(index_full)
            error_full[np.isin(index_full, self.transit_index[q])] = (
                self.transit_errors[q]
            )

        quality_full = np.ones(len(index_full), dtype=bool)

        if full:
            keep = index_full
        else:
            keep = np.isin(index_full, self.transit_index)

        self.transit_index = index_full[keep]
        self.transit_times = ttime_full[keep]
        if self.transit_errors is not None:
            self.transit_errors = error_full[keep]
        self.transit_quality = quality_full[keep]

        self.refit_period_and_epoch()
