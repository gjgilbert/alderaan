"""Planet: physical parameters plus attached TransitTimes"""

__all__ = ["Planet"]

import numpy as np


class Planet:
    """
    Physical planet parameters plus one attached TransitTimes

    Parameters
    ----------
    mission, target : str
    period : float
        initial orbital period (days), e.g. from a catalog
    epoch : float
        initial transit epoch
    ror : float
        planet-to-star radius ratio Rp/Rs
    impact : float
        impact parameter
    duration : float
        transit duration T14 (days)

    Notes
    -----
    period and epoch are read-only; once transit times are attached they are the
    linear ephemeris fit to those transit times
    """

    def __init__(self, mission, target, period, epoch, ror, impact, duration):
        self.mission = mission
        self.target = target
        self._period = period
        self._epoch = epoch
        self.ror = ror
        self.impact = impact
        self.duration = duration

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

        Parameters
        ----------
        transit_times : TransitTimes

        Raises
        ------
        ValueError
            if the new period differs from the current one by >10%
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
        if self.static_period is not None:
            raise ValueError("static period/epoch already set")
        self.static_period = self.period
        self.static_epoch = self.epoch

    def set_transit_shape(self, ror, impact, duration):
        """
        Set transit shape parameters (e.g. from QuickFitTransitShape)

        Parameters
        ----------
        ror : float
            planet-to-star radius ratio Rp/Rs
        impact : float
            impact parameter
        duration : float
            transit duration T14 (days)
        """
        self.ror = ror
        self.impact = impact
        self.duration = duration
