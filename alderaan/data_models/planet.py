__all__ = ["Planet"]

import numpy as np


class Planet:
    """
    Physical planet parameters plus one attached TransitTimes

    Args:
        period (float) : initial orbital period (days), e.g. from a catalog
        epoch (float) : initial transit epoch
        depth (float) : transit depth (fraction)
        duration (float) : transit duration (days)
        impact (float) : impact parameter

    period and epoch are read-only; once transit times are attached they are the
    linear ephemeris fit to those transit times
    """

    def __init__(self, period, epoch, depth, duration, impact):
        self._period = period
        self._epoch = epoch
        self.depth = depth
        self.duration = duration
        self.impact = impact
        self.ror = None

        self.transit_times = None

        self.static_period = None
        self.static_epoch = None

    @property
    def period(self):
        if self.transit_times is None:
            return self._period
        return self.transit_times.period

    @property
    def epoch(self):
        if self.transit_times is None:
            return self._epoch
        return self.transit_times.epoch

    def set_transit_times(self, transit_times):
        """
        Attach transit times; the only way to change a planet's ephemeris

        Args:
            transit_times (TransitTimes)

        Raises:
            ValueError : if the new period differs from the current one by >10%
        """
        if not np.isclose(self.period, transit_times.period, rtol=0.1):
            raise ValueError(
                f"New period ({transit_times.period:.6f}) differs from old period "
                f"({self.period:.6f}) by more than 10%"
            )
        self.transit_times = transit_times

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
