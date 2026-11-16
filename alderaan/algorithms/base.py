"""Base class for algorithms that operate on a star, planets, and litecurve"""

import functools
import logging
import time
from copy import deepcopy

import numpy as np

from alderaan.data_models import LiteCurve, Planet, Star


def _log_perform(perform, name, module):
    logger = logging.getLogger(module)

    @functools.wraps(perform)
    def wrapper(self, *args, **kwargs):
        params = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        logger.info("start %s(%s)", name, ", ".join(params))
        t0 = time.perf_counter()
        out = perform(self, *args, **kwargs)
        logger.info("done %s in %.1f s", name, time.perf_counter() - t0)
        return out

    return wrapper


class BaseAlg:
    """
    Type-check and deepcopy inputs; provide shared transit bookkeeping

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve

    Notes
    -----
    Each subclass's perform() is wrapped to log its start (with keyword
    arguments) and runtime to the subclass module's logger; failures propagate
    to pipeline.py, which logs them.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "perform" in cls.__dict__:
            cls.perform = _log_perform(cls.perform, cls.__name__, cls.__module__)

    def __init__(self, star, planets, litecurve):
        # check inputs
        if not isinstance(litecurve, LiteCurve):
            raise TypeError("litecurve must be a LiteCurve")
        if not isinstance(planets, list):
            raise TypeError("planets must be a list of Planets")
        for p in planets:
            if not isinstance(p, Planet):
                raise TypeError("planets must be a list of Planets")
        if not isinstance(star, Star):
            raise TypeError("star must be a Star")

        # set attributes
        self.litecurve = deepcopy(litecurve)
        self.npl = len(planets)
        self.planets = deepcopy(planets)
        self.star = deepcopy(star)

    def make_transit_mask(self, rel_size=0.0, abs_size=0.0, buffer=0.0, time=None):
        """
        Mask cadences near a transit of any planet

        Each masked window has full width max(rel_size * T14, abs_size) + buffer

        Parameters
        ----------
        rel_size : float
            full width of masked region in units of transit duration (T14)
        abs_size : float
            floor on rel_size * T14 in days, so short transits are not under-masked
        buffer : float
            extra full width in days added after the floor (e.g. cadence + TTVs)
        time : ndarray
            cadence times to mask (default = self.litecurve.time)

        Returns
        -------
        mask : ndarray, bool
            True for cadences near a transit of any planet
        """
        if rel_size == 0 and abs_size == 0:
            raise ValueError("either rel_size or abs_size must be nonzero")

        if time is None:
            time = self.litecurve.time

        mask = np.zeros((self.npl, len(time)), dtype=int)

        for n, p in enumerate(self.planets):
            half_width = 0.5 * (np.max([rel_size * p.duration, abs_size]) + buffer)
            for tc in p.transit_times.ttime_mod:
                mask[n] += np.abs(time - tc) < half_width

        return np.sum(mask, axis=0).astype(bool)

    def identify_overlapping_transits(self, rel_size=None, abs_size=None):
        """
        Identify where transits overlap based on separation of transit midpoints

        Parameters
        ----------
        rel_size : float
            tolerance in units of the mean transit duration of the two planets
        abs_size : float
            tolerance in days

        Returns
        -------
        overlap : list of ndarray, bool
            one array per planet; True for transits that overlap another planet
        """
        if (rel_size is None) and (abs_size is None):
            raise ValueError("at least one of rel_size or abs_size must be provided")

        overlap = [None] * self.npl

        for i in range(self.npl):
            overlap[i] = np.zeros(
                len(self.planets[i].transit_times.ttime_mod), dtype=bool
            )
            ttime_i = self.planets[i].transit_times.ttime_mod

            for j in range(self.npl):
                if j == i:
                    continue
                for tc_j in self.planets[j].transit_times.ttime_mod:
                    if rel_size is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < rel_size * 0.5 * (
                            self.planets[i].duration + self.planets[j].duration
                        )
                    if abs_size is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < abs_size

        return overlap
